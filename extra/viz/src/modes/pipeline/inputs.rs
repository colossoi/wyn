use super::{Frame, Runner};
use crate::config::FeedbackInitial;
use crate::gpu::{initialize, BufferInitSpec};
use crate::json::load_f32_json;
use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::Path;
use wgpu::{
    Color, CompareFunction, Extent3d, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, SamplerDescriptor, StoreOp,
    TexelCopyBufferLayout, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};
use wyn_host_interp::{Number, Parameter, Value};

pub(super) enum Update {
    Buffer {
        argument: usize,
        offset: u64,
        size: usize,
        name: String,
    },
    Keyboard {
        argument: usize,
    },
}

pub(super) enum Destination {
    Resource(usize),
    Field {
        argument: usize,
        offset: u64,
        size: u64,
    },
}

fn read_data(path: &Path) -> Result<Vec<u8>> {
    if path.extension().is_some_and(|ext| ext == "json") {
        Ok(load_f32_json(path)?.iter().flat_map(|v| v.to_le_bytes()).collect())
    } else {
        fs::read(path).with_context(|| format!("reading {}", path.display()))
    }
}

fn builtin(name: &str, frame: &Frame) -> Option<Vec<u8>> {
    let floats = match name {
        "resolution" | "iResolution" => vec![frame.width as f32, frame.height as f32, 1.0],
        "time" | "iTime" => vec![frame.time],
        "delta_time" | "iTimeDelta" => vec![frame.delta],
        "mouse" | "iMouse" => frame.mouse.to_vec(),
        "frame" | "iFrame" => return Some(frame.index.to_le_bytes().to_vec()),
        "keyboard" => return Some(frame.keyboard.to_vec()),
        _ => return None,
    };
    Some(floats.iter().flat_map(|v| v.to_le_bytes()).collect())
}

fn initial_frame(width: u32, height: u32) -> Frame {
    Frame {
        width,
        height,
        time: 0.0,
        delta: 0.0,
        index: 0,
        mouse: [0.0; 4],
        keyboard: [0; 768],
    }
}

fn put(bytes: &mut [u8], offset: u64, size: u64, data: &[u8]) -> Result<()> {
    let start = usize::try_from(offset)?;
    let size = usize::try_from(size)?;
    let Some(end) = start.checked_add(size) else {
        return Err(anyhow!("field range overflow"));
    };
    let Some(destination) = bytes.get_mut(start..end) else {
        return Err(anyhow!("field exceeds input buffer"));
    };
    if data.len() > size {
        return Err(anyhow!("value has {} bytes, field has {size}", data.len()));
    }
    destination.fill(0);
    destination[..data.len()].copy_from_slice(data);
    Ok(())
}

impl Runner {
    pub(super) fn destination(&self, name: &str) -> Result<Destination> {
        let mut matches = Vec::new();
        for (argument, parameter) in self.entry.parameters.iter().enumerate() {
            if parameter.source_name() == name {
                matches.push(Destination::Resource(argument));
            } else {
                for (field, offset, size) in parameter.fields()? {
                    if field == name {
                        matches.push(Destination::Field {
                            argument,
                            offset,
                            size,
                        });
                    }
                }
            }
        }
        if matches.len() != 1 {
            return Err(anyhow!("input {name} must select exactly one source argument"));
        }
        let Some(destination) = matches.pop() else {
            return Err(anyhow!("missing input {name}"));
        };
        Ok(destination)
    }

    pub(super) fn prepare(&mut self) -> Result<()> {
        for feedback in &self.spec.feedback {
            if feedback.entry != self.entry.source_name {
                return Err(anyhow!("feedback entry {} is not selected", feedback.entry));
            }
            self.destination(&feedback.input)?;
            if feedback.result >= self.entry.results.len() {
                return Err(anyhow!("unknown feedback result {}", feedback.result));
            }
        }
        for (index, parameter) in self.entry.parameters.clone().iter().enumerate() {
            let value = match parameter.kind.as_str() {
                ":buffer" | ":host-buffer" => self.input_buffer(index, parameter)?,
                ":texture" => self.input_texture(index, parameter)?,
                ":sampler" => {
                    let comparison = parameter.options.optional(":kind").map(Value::text).transpose()?
                        == Some(":comparison");
                    let sampler = self.backend.device.create_sampler(&SamplerDescriptor {
                        compare: comparison.then_some(CompareFunction::LessEqual),
                        ..Default::default()
                    });
                    self.backend.import_sampler(sampler)
                }
                ":i32" | ":u32" | ":f32" => {
                    let Some(constant) =
                        self.spec.constants.iter().find(|c| c.name == parameter.source_name())
                    else {
                        return Err(anyhow!("missing scalar input {}", parameter.source_name()));
                    };
                    let bytes: [u8; 4] = constant.data.as_slice().try_into()?;
                    Value::Number(match parameter.kind.as_str() {
                        ":i32" => Number::I32(i32::from_le_bytes(bytes)),
                        ":u32" => Number::U32(u32::from_le_bytes(bytes)),
                        _ => Number::f32(f32::from_le_bytes(bytes))?,
                    })
                }
                kind => return Err(anyhow!("unsupported host input {kind}")),
            };
            self.arguments.push(value);
        }
        self.present = self.entry.results.iter().filter(|p| p.kind == ":texture").find_map(|result| {
            result.options.optional(":alias").and_then(|alias| {
                self.entry.parameters.iter().position(|p| alias.text().is_ok_and(|alias| alias == p.name))
            })
        });
        if let Some(path) = &self.spec.opts.index_buffer {
            let bytes = read_data(path)?;
            if bytes.len() % 4 != 0 {
                return Err(anyhow!("index buffer must contain u32 values"));
            }
            self.backend.index_buffer = Some(self.upload(&bytes)?);
        }
        Ok(())
    }

    fn upload(&mut self, bytes: &[u8]) -> Result<Value> {
        let value = self.backend.allocate_buffer(bytes.len() as u64)?;
        let mut padded = bytes.to_vec();
        padded.resize(bytes.len().div_ceil(4) * 4, 0);
        if !padded.is_empty() {
            self.backend.queue.write_buffer(self.backend.buffer(&value)?, 0, &padded);
        }
        Ok(value)
    }

    fn input_buffer(&mut self, index: usize, parameter: &Parameter) -> Result<Value> {
        let name = parameter.source_name();
        let file = self.spec.inputs.get(name).cloned().or_else(|| {
            self.spec
                .opts
                .storage_dir
                .as_ref()
                .map(|dir| dir.join(format!("{name}.bin")))
                .filter(|path| path.is_file())
        });
        let feedback =
            self.spec.feedback.iter().find(|f| f.entry == self.entry.source_name && f.input == name);
        let file = file.or_else(|| match feedback.map(|f| &f.initial) {
            Some(FeedbackInitial::File { path }) => Some(path.clone()),
            _ => None,
        });
        let loaded = file.as_deref().map(read_data).transpose()?;
        let fields = parameter.fields()?;
        let frame = initial_frame(self.width, self.height);
        let automatic = builtin(name, &frame);
        let framebuffer = self
            .spec
            .opts
            .framebuffers
            .get(name)
            .map(|format| u64::from(self.width) * u64::from(self.height) * format.bytes_per_texel());
        let size = framebuffer
            .or_else(|| self.spec.opts.storage_bytes.get(name).copied())
            .or_else(|| loaded.as_ref().map(|bytes| bytes.len() as u64))
            .or(parameter.minimum_bytes()?)
            .or_else(|| automatic.as_ref().map(|bytes| bytes.len() as u64));
        let Some(size) = size else {
            return Err(anyhow!(
                "input {name} needs --input, --storage-bytes, or --framebuffer"
            ));
        };
        if size > self.backend.device.limits().max_buffer_size {
            return Err(anyhow!("input {name} exceeds the device buffer limit"));
        }
        let minimum = parameter.minimum_bytes()?.unwrap_or(0);
        if size < minimum {
            return Err(anyhow!(
                "input {name} has {size} bytes, requires at least {minimum}"
            ));
        }
        let mut bytes = vec![0; usize::try_from(size)?];
        let initializer = self.spec.opts.buffer_inits.get(name).copied().or_else(|| {
            feedback.map(|f| {
                if f.initial == FeedbackInitial::Rng {
                    BufferInitSpec::Rng
                } else {
                    BufferInitSpec::Zero
                }
            })
        });
        if let Some(initializer) = initializer {
            initialize(&mut bytes, initializer);
        }
        if let Some(loaded) = &loaded {
            put(&mut bytes, 0, size, loaded)?;
        }
        let supplied = self.spec.opts.uniform_values.iter().any(|value| value.name == name)
            || loaded.is_some()
            || initializer.is_some()
            || framebuffer.is_some()
            || self.spec.opts.storage_bytes.contains_key(name);
        let fields = if fields.is_empty() { vec![(name.to_owned(), 0, size)] } else { fields };
        for (field, offset, field_size) in fields {
            let field = if field.is_empty() { name.to_owned() } else { field };
            let feedback =
                self.spec.feedback.iter().find(|f| f.entry == self.entry.source_name && f.input == field);
            let field_path = self
                .spec
                .inputs
                .get(&field)
                .cloned()
                .or_else(|| {
                    self.spec
                        .opts
                        .storage_dir
                        .as_ref()
                        .map(|dir| dir.join(format!("{field}.bin")))
                        .filter(|path| path.is_file())
                })
                .or_else(|| match feedback.map(|f| &f.initial) {
                    Some(FeedbackInitial::File { path }) => Some(path.clone()),
                    _ => None,
                });
            let field_data = field_path.as_deref().map(read_data).transpose()?;
            let initializer = self.spec.opts.buffer_inits.get(&field).copied().or_else(|| {
                feedback.map(|f| {
                    if f.initial == FeedbackInitial::Rng {
                        BufferInitSpec::Rng
                    } else {
                        BufferInitSpec::Zero
                    }
                })
            });
            let explicit = self
                .spec
                .opts
                .uniform_values
                .iter()
                .find(|value| value.name == name && value.member.as_deref().unwrap_or(name) == field)
                .map(|value| value.data.clone())
                .or_else(|| {
                    self.spec
                        .constants
                        .iter()
                        .find(|value| value.name == field)
                        .map(|value| value.data.clone())
                });
            if let Some(data) = explicit {
                put(&mut bytes, offset, field_size, &data)?;
            } else if let Some(data) = field_data {
                put(&mut bytes, offset, field_size, &data)?;
            } else if let Some(initializer) = initializer {
                let mut data = vec![0; usize::try_from(field_size)?];
                initialize(&mut data, initializer);
                put(&mut bytes, offset, field_size, &data)?;
            } else if let Some(data) = builtin(&field, &frame) {
                put(&mut bytes, offset, field_size, &data)?;
                self.updates.push(Update::Buffer {
                    argument: index,
                    offset,
                    size: usize::try_from(field_size)?,
                    name: field,
                });
            } else if !supplied {
                return Err(anyhow!("missing input {field}; use --input or --push-constant"));
            }
        }
        if self.spec.verbose {
            eprintln!("input {name}: {size} bytes");
        }
        if parameter.kind == ":host-buffer" {
            Ok(self.backend.import_host_buffer(bytes))
        } else {
            self.upload(&bytes)
        }
    }

    fn target_texture(&mut self, parameter: &Parameter) -> Result<Value> {
        let format = match parameter.options.text(":format")? {
            ":caller" | ":rgba8unorm" => TextureFormat::Rgba8Unorm,
            ":depth32float" => TextureFormat::Depth32Float,
            ":rgba16float" => TextureFormat::Rgba16Float,
            ":rgba32float" => TextureFormat::Rgba32Float,
            ":r32float" => TextureFormat::R32Float,
            other => return Err(anyhow!("unsupported input texture format {other}")),
        };
        let texture = self.backend.device.create_texture(&TextureDescriptor {
            label: Some(parameter.source_name()),
            size: Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: parameter.options.u32(":samples")?,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST
                | if format == TextureFormat::Depth32Float {
                    TextureUsages::empty()
                } else {
                    TextureUsages::STORAGE_BINDING
                },
            view_formats: &[],
        });
        Ok(self.backend.import_texture(texture))
    }

    fn input_texture(&mut self, index: usize, parameter: &Parameter) -> Result<Value> {
        let name = parameter.source_name();
        if let Some(path) = self.spec.opts.images.get(name) {
            let image = image::open(path)?.to_rgba8();
            let size = Extent3d {
                width: image.width(),
                height: image.height(),
                depth_or_array_layers: 1,
            };
            let texture = self.backend.device.create_texture(&TextureDescriptor {
                label: Some(name),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.backend.queue.write_texture(
                texture.as_image_copy(),
                &image,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size.width * 4),
                    rows_per_image: Some(size.height),
                },
                size,
            );
            return Ok(self.backend.import_texture(texture));
        }
        if name == "keyboard" {
            let texture = self.backend.device.create_texture(&TextureDescriptor {
                label: Some(name),
                size: Extent3d {
                    width: 256,
                    height: 3,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.updates.push(Update::Keyboard { argument: index });
            return Ok(self.backend.import_texture(texture));
        }
        if parameter.access.as_deref() == Some(":read") {
            return Err(anyhow!("texture {name} needs --image {name}:FILE"));
        }
        self.targets.push(index);
        self.target_texture(parameter)
    }

    pub(super) fn resize_targets(&mut self) -> Result<()> {
        for index in self.targets.clone() {
            self.arguments[index] = self.target_texture(&self.entry.parameters[index].clone())?;
        }
        for index in 0..self.arguments.len() {
            let parameter = self.entry.parameters[index].clone();
            if self.spec.opts.framebuffers.contains_key(parameter.source_name()) {
                self.arguments[index] = self.input_buffer(index, &parameter)?;
            }
        }
        Ok(())
    }

    pub(super) fn update(&mut self, frame: &Frame) -> Result<()> {
        for update in &self.updates {
            match update {
                Update::Buffer {
                    argument,
                    offset,
                    size,
                    name,
                } => {
                    let Some(data) = builtin(name, frame) else {
                        return Err(anyhow!("unknown automatic input {name}"));
                    };
                    let mut padded = vec![0; *size];
                    put(&mut padded, 0, *size as u64, &data)?;
                    self.backend.write_buffer(&self.arguments[*argument], *offset, &padded)?;
                }
                Update::Keyboard { argument } => {
                    let texture = self.backend.texture(&self.arguments[*argument])?;
                    let bytes = frame.keyboard.iter().flat_map(|&v| [v, v, v, 255]).collect::<Vec<_>>();
                    self.backend.queue.write_texture(
                        texture.as_image_copy(),
                        &bytes,
                        TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(1024),
                            rows_per_image: Some(3),
                        },
                        texture.size(),
                    );
                }
            }
        }
        let mut encoder = self.backend.device.create_command_encoder(&Default::default());
        for &index in &self.targets {
            let texture = self.backend.texture(&self.arguments[index])?;
            let view = texture.create_view(&Default::default());
            if texture.format() == TextureFormat::Depth32Float {
                encoder.begin_render_pass(&RenderPassDescriptor {
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &view,
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(1.0),
                            store: StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
            } else if Some(index) == self.present {
                encoder.begin_render_pass(&RenderPassDescriptor {
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color::BLACK),
                            store: StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
            }
        }
        self.backend.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}
