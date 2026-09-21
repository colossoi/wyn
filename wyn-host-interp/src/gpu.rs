use wgpu::{BindGroupLayout, ErrorFilter, MapMode, PollType, PrimitiveTopology, TextureFormat};
mod bindings;
mod commands;
mod draw;

use crate::{Backend, Error, Options, Parameter, Program, Result, Value};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::mpsc;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, ComputePipeline, Device, Queue, RenderPipeline, Sampler,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, Texture, TextureView,
};

#[derive(Clone)]
enum Resource {
    Buffer(Buffer, u64),
    HostBuffer(Vec<u8>),
    Texture(Texture),
    View {
        view: TextureView,
        texture: u64,
        width: u32,
        height: u32,
    },
    Sampler(Sampler),
}

pub struct WgpuBackend {
    pub device: Device,
    pub queue: Queue,
    modules: BTreeMap<String, ShaderModule>,
    computes: BTreeMap<String, (ComputePipeline, Vec<BindGroupLayout>)>,
    renders: BTreeMap<String, (RenderPipeline, Vec<BindGroupLayout>)>,
    resources: BTreeMap<u64, Resource>,
    imported: BTreeSet<u64>,
    borrowed: BTreeSet<u64>,
    checkpoint: u64,
    next: u64,
    pub dispatch_overrides: BTreeMap<String, [u32; 3]>,
    pub vertex_count: Option<u32>,
    pub topology: Option<PrimitiveTopology>,
    pub index_buffer: Option<Value>,
}

fn gpu_error(error: impl std::fmt::Display) -> Error {
    Error::Gpu(error.to_string())
}

impl WgpuBackend {
    pub fn new(device: Device, queue: Queue, program: &Program, base: &Path) -> Result<Self> {
        let mut sources = BTreeMap::new();
        for (name, module) in &program.modules {
            sources.insert(name.clone(), fs::read(base.join(module.options.text(":path")?))?);
        }
        Self::with_sources(device, queue, program, &sources)
    }

    pub fn with_sources(
        device: Device,
        queue: Queue,
        program: &Program,
        sources: &BTreeMap<String, Vec<u8>>,
    ) -> Result<Self> {
        let mut modules = BTreeMap::new();
        for (name, declaration) in &program.modules {
            let Some(bytes) = sources.get(name) else {
                return Err(gpu_error(format!("missing shader module {name}")));
            };
            let source = match declaration.options.text(":format")? {
                ":wgsl" => {
                    ShaderSource::Wgsl(Cow::Borrowed(std::str::from_utf8(bytes).map_err(gpu_error)?))
                }
                ":spirv" => {
                    if bytes.len() % 4 != 0 {
                        return Err(gpu_error("SPIR-V length is not a multiple of four"));
                    }
                    let words = bytes
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect::<Vec<_>>();
                    ShaderSource::SpirV(Cow::Owned(words))
                }
                other => return Err(gpu_error(format!("unsupported shader format {other}"))),
            };
            device.push_error_scope(ErrorFilter::Validation);
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some(name),
                source,
            });
            if let Some(error) = pollster::block_on(device.pop_error_scope()) {
                return Err(gpu_error(error));
            }
            modules.insert(name.clone(), module);
        }
        Ok(Self {
            device,
            queue,
            modules,
            computes: BTreeMap::new(),
            renders: BTreeMap::new(),
            resources: BTreeMap::new(),
            imported: BTreeSet::new(),
            borrowed: BTreeSet::new(),
            checkpoint: 0,
            next: 0,
            dispatch_overrides: BTreeMap::new(),
            vertex_count: None,
            topology: None,
            index_buffer: None,
        })
    }

    fn insert(&mut self, resource: Resource, imported: bool) -> Value {
        self.next += 1;
        self.resources.insert(self.next, resource);
        if imported {
            self.imported.insert(self.next);
        }
        Value::Resource(self.next)
    }

    pub fn import_buffer(&mut self, buffer: Buffer, bytes: u64) -> Result<Value> {
        if bytes > buffer.size() {
            return Err(gpu_error("logical buffer size exceeds allocation"));
        }
        Ok(self.insert(Resource::Buffer(buffer, bytes), true))
    }
    pub fn import_texture(&mut self, texture: Texture) -> Value {
        self.insert(Resource::Texture(texture), true)
    }
    pub fn import_sampler(&mut self, sampler: Sampler) -> Value {
        self.insert(Resource::Sampler(sampler), true)
    }
    pub fn import_host_buffer(&mut self, bytes: Vec<u8>) -> Value {
        self.insert(Resource::HostBuffer(bytes), true)
    }

    fn resource(&self, value: &Value) -> Result<&Resource> {
        let id = value.handle()?;
        let Some(resource) = self.resources.get(&id) else {
            return Err(gpu_error(format!("invalid or freed resource {id}")));
        };
        if let Resource::View { texture, .. } = resource {
            if !self.resources.contains_key(texture) {
                return Err(gpu_error("view references a freed texture"));
            }
        }
        Ok(resource)
    }
    pub fn buffer(&self, value: &Value) -> Result<&Buffer> {
        let Resource::Buffer(buffer, _) = self.resource(value)? else {
            return Err(gpu_error("expected a device buffer"));
        };
        Ok(buffer)
    }
    pub fn texture(&self, value: &Value) -> Result<&Texture> {
        match self.resource(value)? {
            Resource::Texture(texture) => Ok(texture),
            Resource::View { texture, .. } => self.texture(&Value::Resource(*texture)),
            _ => Err(gpu_error("expected a texture")),
        }
    }
    pub fn buffer_size(&self, value: &Value) -> Result<u64> {
        match self.resource(value)? {
            Resource::Buffer(_, size) => Ok(*size),
            Resource::HostBuffer(bytes) => Ok(bytes.len() as u64),
            _ => Err(gpu_error("expected a buffer")),
        }
    }
    pub fn write_buffer(&mut self, value: &Value, offset: u64, bytes: &[u8]) -> Result<()> {
        let end =
            offset.checked_add(bytes.len() as u64).ok_or_else(|| gpu_error("buffer range overflow"))?;
        if end > self.buffer_size(value)? {
            return Err(gpu_error("write exceeds logical buffer size"));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        if matches!(self.resource(value)?, Resource::Buffer(..))
            && (offset % 4 != 0 || bytes.len() % 4 != 0)
        {
            let start = offset / 4 * 4;
            let Some(aligned_end) = end.checked_add(3).map(|end| end / 4 * 4) else {
                return Err(gpu_error("write range overflow"));
            };
            if aligned_end > self.buffer(value)?.size() {
                return Err(gpu_error("unaligned write exceeds physical buffer capacity"));
            }
            let preserved_end = aligned_end.min(self.buffer_size(value)?);
            let mut data = self.read_buffer(value, start, preserved_end - start)?;
            data.resize(usize::try_from(aligned_end - start).map_err(gpu_error)?, 0);
            data[(offset - start) as usize..(end - start) as usize].copy_from_slice(bytes);
            self.queue.write_buffer(self.buffer(value)?, start, &data);
            return Ok(());
        }
        match self.resources.get_mut(&value.handle()?) {
            Some(Resource::Buffer(buffer, _)) => {
                self.queue.write_buffer(buffer, offset, bytes);
            }
            Some(Resource::HostBuffer(storage)) => {
                storage[offset as usize..end as usize].copy_from_slice(bytes)
            }
            _ => return Err(gpu_error("expected a buffer")),
        }
        Ok(())
    }
    pub fn read_buffer(&self, value: &Value, offset: u64, bytes: u64) -> Result<Vec<u8>> {
        let end = offset.checked_add(bytes).ok_or_else(|| gpu_error("buffer range overflow"))?;
        if end > self.buffer_size(value)? {
            return Err(gpu_error("read exceeds logical buffer size"));
        }
        if let Resource::HostBuffer(storage) = self.resource(value)? {
            return Ok(storage[offset as usize..end as usize].to_vec());
        }
        if bytes == 0 {
            return Ok(Vec::new());
        }
        let start = offset / 4 * 4;
        let copy_end = end.checked_add(3).ok_or_else(|| gpu_error("read range overflow"))? / 4 * 4;
        let buffer = self.buffer(value)?;
        if !buffer.usage().contains(BufferUsages::COPY_SRC) || copy_end > buffer.size() {
            return Err(gpu_error("buffer does not support the aligned readback range"));
        }
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("WHL readback"),
            size: copy_end - start,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, start, &staging, 0, copy_end - start);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (send, receive) = mpsc::sync_channel(1);
        slice.map_async(MapMode::Read, move |result| {
            if send.send(result).is_err() { /* The caller has already returned a device polling error. */ }
        });
        self.device.poll(PollType::wait_indefinitely()).map_err(gpu_error)?;
        receive.recv().map_err(gpu_error)?.map_err(gpu_error)?;
        let mapped = slice.get_mapped_range();
        let result = mapped[(offset - start) as usize..(end - start) as usize].to_vec();
        drop(mapped);
        staging.unmap();
        Ok(result)
    }

    pub fn allocate_buffer(&mut self, bytes: u64) -> Result<Value> {
        let physical =
            bytes.max(4).checked_add(3).ok_or_else(|| gpu_error("allocation size overflow"))? / 4 * 4;
        if physical > self.device.limits().max_buffer_size {
            return Err(gpu_error("buffer exceeds device limit"));
        }
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("WHL buffer"),
            size: physical,
            usage: BufferUsages::STORAGE
                | BufferUsages::UNIFORM
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST
                | BufferUsages::VERTEX
                | BufferUsages::INDEX
                | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        Ok(self.insert(Resource::Buffer(buffer, bytes), false))
    }

    fn roots(&self, values: &[Value]) -> BTreeSet<u64> {
        let mut ids = BTreeSet::new();
        fn add(value: &Value, ids: &mut BTreeSet<u64>) {
            match value {
                Value::Resource(id) => {
                    ids.insert(*id);
                }
                Value::List(values) => {
                    for value in values {
                        add(value, ids);
                    }
                }
                _ => {}
            }
        }
        for value in values {
            add(value, &mut ids);
        }
        for id in ids.clone() {
            if let Some(Resource::View { texture, .. }) = self.resources.get(&id) {
                ids.insert(*texture);
            }
        }
        ids
    }

    pub fn retain_resources(&mut self, values: &[Value]) {
        let ids = self.roots(values);
        self.resources.retain(|id, _| ids.contains(id));
        self.imported.retain(|id| ids.contains(id));
    }
}

impl Backend for WgpuBackend {
    fn validate_parameter(&self, parameter: &Parameter, value: &Value) -> Result<()> {
        match parameter.kind.as_str() {
            ":buffer" => {
                self.buffer(value)?;
            }
            ":host-buffer" => {
                if !matches!(self.resource(value)?, Resource::HostBuffer(_)) {
                    return Err(gpu_error("expected a host byte span"));
                }
            }
            ":texture" | ":texture-view" => {
                let texture = self.texture(value)?;
                let format = parameter.options.text(":format")?;
                if format != ":caller" && texture.format() != texture_format(format)? {
                    return Err(gpu_error(format!(
                        "{} has the wrong texture format",
                        parameter.name
                    )));
                }
                let samples = parameter.options.get(":samples")?;
                let valid = if samples == &Value::Symbol(":multisampled".into()) {
                    texture.sample_count() > 1
                } else {
                    texture.sample_count() == samples.u32()?
                };
                if !valid {
                    return Err(gpu_error("texture sample count mismatch"));
                }
            }
            ":sampler" => {
                if !matches!(self.resource(value)?, Resource::Sampler(_)) {
                    return Err(gpu_error("expected a sampler"));
                }
            }
            _ => {}
        }
        if let Some(minimum) = parameter.minimum_bytes()? {
            if self.buffer_size(value)? < minimum {
                return Err(gpu_error(format!(
                    "{} requires at least {minimum} bytes",
                    parameter.name
                )));
            }
        }
        Ok(())
    }
    fn begin(&mut self, arguments: &[Value]) -> Result<()> {
        self.checkpoint = self.next;
        self.borrowed = self.roots(arguments);
        for id in &self.borrowed {
            self.resource(&Value::Resource(*id))?;
        }
        Ok(())
    }
    fn finish(&mut self, result: &Result<Value>) -> Result<()> {
        let retained = match result {
            Ok(value) => self.roots(std::slice::from_ref(value)),
            Err(_) => BTreeSet::new(),
        };
        self.resources.retain(|id, _| *id <= self.checkpoint || retained.contains(id));
        self.borrowed.clear();
        Ok(())
    }
    fn call(&mut self, program: &Program, operation: &str, arguments: &[Value]) -> Result<Value> {
        self.device.push_error_scope(ErrorFilter::Validation);
        let result = self.execute(program, operation, arguments);
        let validation = pollster::block_on(self.device.pop_error_scope());
        if let Some(error) = validation {
            return Err(gpu_error(format!("{operation}: {error}")));
        }
        result
    }
}

fn options<'a>(args: &'a [Value], count: usize, allowed: &[&str]) -> Result<Options> {
    if args.len() < count {
        return Err(gpu_error("not enough GPU arguments"));
    }
    Options::parse(&args[count..], allowed)
}

fn texture_format(name: &str) -> Result<TextureFormat> {
    Ok(match name {
        ":rgba8unorm" => TextureFormat::Rgba8Unorm,
        ":rgba16float" => TextureFormat::Rgba16Float,
        ":rgba32float" => TextureFormat::Rgba32Float,
        ":r32float" => TextureFormat::R32Float,
        ":depth32float" => TextureFormat::Depth32Float,
        _ => return Err(gpu_error(format!("unsupported texture format {name}"))),
    })
}
