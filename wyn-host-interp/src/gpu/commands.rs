use super::bindings::{comparison, dimension};
use super::{gpu_error, options, texture_format, Resource, WgpuBackend};
use crate::{Number, NumberType, Program, Result, Value};
use wgpu::{AddressMode, FilterMode, SamplerDescriptor};
use wgpu::{
    ComputePassDescriptor, ComputePipelineDescriptor, Extent3d, PipelineLayoutDescriptor, ShaderStages,
    TextureDescriptor, TextureDimension, TextureUsages, TextureViewDescriptor,
};

fn arity(args: &[Value], count: usize) -> Result<()> {
    if args.len() != count {
        return Err(gpu_error(format!(
            "expected {count} GPU arguments, got {}",
            args.len()
        )));
    }
    Ok(())
}

fn three(value: &Value) -> Result<[u32; 3]> {
    let [x, y, z] = value.list()? else {
        return Err(gpu_error("expected three dimensions"));
    };
    Ok([x.u32()?, y.u32()?, z.u32()?])
}

fn scalar_size(ty: &str) -> Result<u64> {
    Ok(match ty {
        "i8" | "u8" => 1,
        "i16" | "u16" => 2,
        "i32" | "u32" | "f32" | "bool" => 4,
        "i64" | "u64" | "f64" => 8,
        _ => return Err(gpu_error(format!("unsupported scalar transfer type {ty}"))),
    })
}

impl WgpuBackend {
    pub(super) fn execute(&mut self, program: &Program, name: &str, args: &[Value]) -> Result<Value> {
        match name {
            "gpu-alloc" => {
                arity(args, 1)?;
                self.allocate_buffer(args[0].u64()?)
            }
            "gpu-free" => {
                arity(args, 1)?;
                let id = args[0].handle()?;
                self.resource(&args[0])?;
                if self.imported.contains(&id) || self.borrowed.contains(&id) {
                    return Err(gpu_error("cannot free a borrowed resource"));
                }
                if matches!(self.resource(&args[0])?, Resource::View { .. }) {
                    return Err(gpu_error("views cannot be freed independently"));
                }
                self.resources.remove(&id);
                Ok(Value::Nil)
            }
            "gpu-buffer-size" => {
                arity(args, 1)?;
                Ok(Value::Number(Number::U64(self.buffer_size(&args[0])?)))
            }
            "gpu-read-scalar" => {
                arity(args, 3)?;
                let ty = args[2].text()?;
                let size = scalar_size(ty)?;
                let offset = args[1].u64()?;
                if offset % size != 0 {
                    return Err(gpu_error("misaligned scalar read"));
                }
                let bytes = self.read_buffer(&args[0], offset, size)?;
                let value = match ty {
                    "i8" => Number::I32(i32::from(bytes[0] as i8)),
                    "u8" => Number::U32(bytes[0].into()),
                    "i16" => Number::I32(i16::from_le_bytes([bytes[0], bytes[1]]).into()),
                    "u16" => Number::U32(u16::from_le_bytes([bytes[0], bytes[1]]).into()),
                    "i32" => Number::I32(i32::from_le_bytes(
                        bytes.as_slice().try_into().map_err(gpu_error)?,
                    )),
                    "u32" => Number::U32(u32::from_le_bytes(
                        bytes.as_slice().try_into().map_err(gpu_error)?,
                    )),
                    "i64" => Number::I64(i64::from_le_bytes(
                        bytes.as_slice().try_into().map_err(gpu_error)?,
                    )),
                    "u64" => Number::U64(u64::from_le_bytes(
                        bytes.as_slice().try_into().map_err(gpu_error)?,
                    )),
                    "f32" => Number::F32(f32::from_le_bytes(
                        bytes.as_slice().try_into().map_err(gpu_error)?,
                    )),
                    "f64" => Number::f64(f64::from_le_bytes(
                        bytes.as_slice().try_into().map_err(gpu_error)?,
                    ))?,
                    "bool" => {
                        return Ok(Value::boolean(
                            u32::from_le_bytes(bytes.as_slice().try_into().map_err(gpu_error)?) != 0,
                        ))
                    }
                    _ => return Err(gpu_error("invalid scalar type")),
                };
                Ok(Value::Number(value))
            }
            "gpu-write-scalar" => {
                arity(args, 4)?;
                let ty = args[2].text()?;
                let size = scalar_size(ty)?;
                let offset = args[1].u64()?;
                if offset % size != 0 {
                    return Err(gpu_error("misaligned scalar write"));
                }
                let bytes = if ty == "bool" {
                    match args[3] {
                        Value::True => 1u32.to_le_bytes().to_vec(),
                        Value::Nil => 0u32.to_le_bytes().to_vec(),
                        _ => return Err(gpu_error("boolean write requires t or nil")),
                    }
                } else {
                    let number = args[3].number()?;
                    match ty {
                        "i8" => i8::try_from(number.integer()?).map_err(gpu_error)?.to_le_bytes().to_vec(),
                        "u8" => u8::try_from(number.integer()?).map_err(gpu_error)?.to_le_bytes().to_vec(),
                        "i16" => {
                            i16::try_from(number.integer()?).map_err(gpu_error)?.to_le_bytes().to_vec()
                        }
                        "u16" => {
                            u16::try_from(number.integer()?).map_err(gpu_error)?.to_le_bytes().to_vec()
                        }
                        "i32" => {
                            i32::try_from(number.integer()?).map_err(gpu_error)?.to_le_bytes().to_vec()
                        }
                        "u32" => {
                            u32::try_from(number.integer()?).map_err(gpu_error)?.to_le_bytes().to_vec()
                        }
                        "i64" => {
                            i64::try_from(number.integer()?).map_err(gpu_error)?.to_le_bytes().to_vec()
                        }
                        "u64" => {
                            u64::try_from(number.integer()?).map_err(gpu_error)?.to_le_bytes().to_vec()
                        }
                        "f32" => match number {
                            Number::F32(value) => value.to_le_bytes().to_vec(),
                            number => {
                                (number.convert(NumberType::F32)?.real() as f32).to_le_bytes().to_vec()
                            }
                        },
                        "f64" => number.convert(NumberType::F64)?.real().to_le_bytes().to_vec(),
                        _ => return Err(gpu_error("invalid scalar type")),
                    }
                };
                self.write_buffer(&args[0], offset, &bytes)?;
                Ok(Value::Nil)
            }
            "gpu-copy" => {
                arity(args, 5)?;
                let destination = &args[0];
                let dst = args[1].u64()?;
                let source = &args[2];
                let src = args[3].u64()?;
                let size = args[4].u64()?;
                let src_end = src.checked_add(size).ok_or_else(|| gpu_error("copy range overflow"))?;
                let dst_end = dst.checked_add(size).ok_or_else(|| gpu_error("copy range overflow"))?;
                if src_end > self.buffer_size(source)? || dst_end > self.buffer_size(destination)? {
                    return Err(gpu_error("copy exceeds logical capacity"));
                }
                if size == 0 {
                    return Ok(Value::Nil);
                }
                if source == destination && src < dst_end && dst < src_end {
                    return Err(gpu_error("overlapping buffer copy"));
                }
                if matches!(self.resource(source)?, Resource::HostBuffer(_))
                    || matches!(self.resource(destination)?, Resource::HostBuffer(_))
                    || source == destination
                    || src % 4 != 0
                    || dst % 4 != 0
                    || size % 4 != 0
                {
                    let bytes = self.read_buffer(source, src, size)?;
                    self.write_buffer(destination, dst, &bytes)?;
                } else {
                    if src % 4 != 0 || dst % 4 != 0 || size % 4 != 0 {
                        return Err(gpu_error("device copies require four-byte alignment"));
                    }
                    let mut encoder = self.device.create_command_encoder(&Default::default());
                    encoder.copy_buffer_to_buffer(
                        self.buffer(source)?,
                        src,
                        self.buffer(destination)?,
                        dst,
                        size,
                    );
                    self.queue.submit(Some(encoder.finish()));
                }
                Ok(Value::Nil)
            }
            "gpu-dispatch" => self.dispatch(program, args),
            "gpu-draw" => self.draw(program, args),
            "gpu-alloc-texture" => {
                let o = options(
                    args,
                    0,
                    &[
                        ":dimension",
                        ":size",
                        ":format",
                        ":mip-levels",
                        ":samples",
                        ":usage",
                    ],
                )?;
                let [width, height, depth] = three(o.get(":size")?)?;
                if width == 0 || height == 0 || depth == 0 {
                    return Err(gpu_error("texture dimensions must be positive"));
                }
                let dimension = match o.text(":dimension")? {
                    ":d1" => TextureDimension::D1,
                    ":d2" => TextureDimension::D2,
                    ":d3" => TextureDimension::D3,
                    _ => return Err(gpu_error("invalid texture allocation dimension")),
                };
                let mut usage = TextureUsages::empty();
                for name in o.get(":usage")?.list()? {
                    usage |= match name.text()? {
                        ":storage" => TextureUsages::STORAGE_BINDING,
                        ":sampled" => TextureUsages::TEXTURE_BINDING,
                        ":render-target" => TextureUsages::RENDER_ATTACHMENT,
                        ":copy" => TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
                        other => return Err(gpu_error(format!("invalid texture usage {other}"))),
                    };
                }
                let texture = self.device.create_texture(&TextureDescriptor {
                    label: Some("WHL texture"),
                    size: Extent3d {
                        width,
                        height,
                        depth_or_array_layers: depth,
                    },
                    mip_level_count: o.u32(":mip-levels")?,
                    sample_count: o.u32(":samples")?,
                    dimension,
                    format: texture_format(o.text(":format")?)?,
                    usage,
                    view_formats: &[],
                });
                Ok(self.insert(Resource::Texture(texture), false))
            }
            "gpu-texture-view" => {
                let o = options(
                    args,
                    1,
                    &[
                        ":usage",
                        ":dimension",
                        ":mip",
                        ":mip-count",
                        ":layer",
                        ":layer-count",
                    ],
                )?;
                let texture = self.texture(&args[0])?;
                let mip = o.u32(":mip")?;
                if mip >= texture.mip_level_count() {
                    return Err(gpu_error("texture mip is out of range"));
                }
                let usage = match o.text(":usage")? {
                    ":sampled" => TextureUsages::TEXTURE_BINDING,
                    ":storage" => TextureUsages::STORAGE_BINDING,
                    ":render-target" => TextureUsages::RENDER_ATTACHMENT,
                    _ => return Err(gpu_error("invalid texture view usage")),
                };
                let width = (texture.width() >> mip).max(1);
                let height = (texture.height() >> mip).max(1);
                let view = texture.create_view(&TextureViewDescriptor {
                    dimension: Some(dimension(o.text(":dimension")?)?),
                    base_mip_level: mip,
                    mip_level_count: Some(o.u32(":mip-count")?),
                    base_array_layer: o.u32(":layer")?,
                    array_layer_count: Some(o.u32(":layer-count")?),
                    usage: Some(usage),
                    ..Default::default()
                });
                Ok(self.insert(
                    Resource::View {
                        view,
                        texture: args[0].handle()?,
                        width,
                        height,
                    },
                    false,
                ))
            }
            "gpu-texture-size" | "gpu-texture-dimension" => {
                arity(args, if name == "gpu-texture-size" { 2 } else { 3 })?;
                let texture = self.texture(&args[0])?;
                let mip = args[1].u32()?;
                if mip >= texture.mip_level_count() {
                    return Err(gpu_error("texture mip out of range"));
                }
                let size = [
                    (texture.width() >> mip).max(1),
                    (texture.height() >> mip).max(1),
                    if texture.dimension() == TextureDimension::D3 {
                        (texture.depth_or_array_layers() >> mip).max(1)
                    } else {
                        texture.depth_or_array_layers()
                    },
                ];
                if name == "gpu-texture-size" {
                    Ok(Value::List(
                        size.into_iter().map(|v| Value::Number(Number::U32(v))).collect(),
                    ))
                } else {
                    let axis = match args[2].text()? {
                        "width" => 0,
                        "height" => 1,
                        "depth-or-layers" => 2,
                        _ => return Err(gpu_error("unknown texture axis")),
                    };
                    Ok(Value::Number(Number::U32(size[axis])))
                }
            }
            "gpu-texture-mip-levels" => {
                arity(args, 1)?;
                Ok(Value::Number(Number::U32(
                    self.texture(&args[0])?.mip_level_count(),
                )))
            }
            "gpu-alloc-sampler" => {
                let o = options(
                    args,
                    0,
                    &[
                        ":kind",
                        ":address",
                        ":min-filter",
                        ":mag-filter",
                        ":mip-filter",
                        ":lod",
                        ":compare",
                    ],
                )?;
                let [u, v, w] = o.get(":address")?.list()? else {
                    return Err(gpu_error("sampler address needs three modes"));
                };
                let address = |value: &Value| -> Result<AddressMode> {
                    Ok(match value.text()? {
                        ":clamp-to-edge" => AddressMode::ClampToEdge,
                        ":repeat" => AddressMode::Repeat,
                        ":mirror-repeat" => AddressMode::MirrorRepeat,
                        other => return Err(gpu_error(format!("invalid address mode {other}"))),
                    })
                };
                let filter = |key| -> Result<FilterMode> {
                    Ok(match o.text(key)? {
                        ":nearest" => FilterMode::Nearest,
                        ":linear" => FilterMode::Linear,
                        other => return Err(gpu_error(format!("invalid filter {other}"))),
                    })
                };
                let compare = if matches!(o.get(":compare")?, Value::Nil) {
                    None
                } else {
                    Some(comparison(o.text(":compare")?)?)
                };
                let min_filter = filter(":min-filter")?;
                let mag_filter = filter(":mag-filter")?;
                let mip_filter = filter(":mip-filter")?;
                match o.text(":kind")? {
                    ":non-filtering"
                        if [min_filter, mag_filter, mip_filter]
                            .iter()
                            .any(|f| *f != FilterMode::Nearest) =>
                    {
                        return Err(gpu_error("non-filtering samplers require nearest filters"))
                    }
                    ":comparison" if compare.is_none() => {
                        return Err(gpu_error("comparison sampler requires a comparison"))
                    }
                    ":filtering" | ":non-filtering" if compare.is_some() => {
                        return Err(gpu_error("only comparison samplers accept a comparison"))
                    }
                    ":filtering" | ":non-filtering" | ":comparison" => {}
                    _ => return Err(gpu_error("unknown sampler kind")),
                }
                let [min, max] = o.get(":lod")?.list()? else {
                    return Err(gpu_error("LOD needs minimum and maximum"));
                };
                let min = min.number()?.convert(NumberType::F32)?.real() as f32;
                let max = max.number()?.convert(NumberType::F32)?.real() as f32;
                if min > max {
                    return Err(gpu_error("invalid LOD range"));
                }
                let sampler = self.device.create_sampler(&SamplerDescriptor {
                    address_mode_u: address(u)?,
                    address_mode_v: address(v)?,
                    address_mode_w: address(w)?,
                    mag_filter,
                    min_filter,
                    mipmap_filter: mip_filter,
                    lod_min_clamp: min,
                    lod_max_clamp: max,
                    compare,
                    ..Default::default()
                });
                Ok(self.insert(Resource::Sampler(sampler), false))
            }
            "gpu-device-property" => {
                arity(args, 1)?;
                let limits = self.device.limits();
                let value = match args[0].text()? {
                    "max-workgroup-size" => Number::U32(limits.max_compute_invocations_per_workgroup),
                    "max-workgroup-size-x" => Number::U32(limits.max_compute_workgroup_size_x),
                    "max-workgroup-size-y" => Number::U32(limits.max_compute_workgroup_size_y),
                    "max-workgroup-size-z" => Number::U32(limits.max_compute_workgroup_size_z),
                    "max-workgroups-x" | "max-workgroups-y" | "max-workgroups-z" => {
                        Number::U32(limits.max_compute_workgroups_per_dimension)
                    }
                    "shared-memory-per-workgroup" => {
                        Number::U64(limits.max_compute_workgroup_storage_size.into())
                    }
                    "max-buffer-bytes" => Number::U64(limits.max_buffer_size),
                    "max-texture-dimension-1d" => Number::U32(limits.max_texture_dimension_1d),
                    "max-texture-dimension-2d" => Number::U32(limits.max_texture_dimension_2d),
                    "max-texture-dimension-3d" => Number::U32(limits.max_texture_dimension_3d),
                    "max-texture-array-layers" => Number::U32(limits.max_texture_array_layers),
                    other => return Err(gpu_error(format!("unknown device property {other}"))),
                };
                Ok(Value::Number(value))
            }
            _ => Err(gpu_error(format!("unsupported GPU operation {name}"))),
        }
    }

    fn dispatch(&mut self, program: &Program, args: &[Value]) -> Result<Value> {
        let o = options(args, 1, &[":groups", ":args"])?;
        let name = args[0].text()?;
        let Some(kernel) = program.kernels.get(name) else {
            return Err(gpu_error(format!("unknown kernel {name}")));
        };
        let mut groups = three(o.get(":groups")?)?;
        let entry = kernel.options.text(":entry")?;
        if let Some(threads) = self.dispatch_overrides.get(entry) {
            let workgroup = three(kernel.options.get(":workgroup-size")?)?;
            groups = [
                threads[0].div_ceil(workgroup[0]),
                threads[1].div_ceil(workgroup[1]),
                threads[2].div_ceil(workgroup[2]),
            ];
        }
        if groups.iter().any(|n| *n > self.device.limits().max_compute_workgroups_per_dimension) {
            return Err(gpu_error("dispatch exceeds device limits"));
        }
        if !self.computes.contains_key(name) {
            let (layouts, pushes) = self.layouts(kernel, ShaderStages::COMPUTE)?;
            let layout_refs = layouts.iter().collect::<Vec<_>>();
            let layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &layout_refs,
                push_constant_ranges: &pushes,
            });
            let Some(module) = self.modules.get(kernel.options.text(":module")?) else {
                return Err(gpu_error("missing shader module"));
            };
            let pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            self.computes.insert(name.into(), (pipeline, layouts));
        }
        let (pipeline, layouts) = &self.computes[name];
        let (bindings, pushes) = self.bind_groups(kernel, o.get(":args")?.list()?, layouts)?;
        if groups.contains(&0) {
            return Ok(Value::Nil);
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(entry),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            for (set, group) in bindings.iter().enumerate() {
                pass.set_bind_group(set as u32, group, &[]);
            }
            for (offset, bytes) in pushes {
                pass.set_push_constants(offset, &bytes);
            }
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(Value::Nil)
    }
}
