use super::{gpu_error, texture_format, Resource, WgpuBackend};
use crate::{Declaration, Parameter, Result, Value};
use std::collections::{BTreeMap, BTreeSet};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, PushConstantRange, ShaderStages,
    TextureSampleType, TextureViewDimension,
};
use wgpu::{CompareFunction, Features, SamplerBindingType, StorageTextureAccess};

pub(super) fn dimension(name: &str) -> Result<TextureViewDimension> {
    Ok(match name {
        ":d1" => TextureViewDimension::D1,
        ":d2" => TextureViewDimension::D2,
        ":d2-array" => TextureViewDimension::D2Array,
        ":cube" => TextureViewDimension::Cube,
        ":cube-array" => TextureViewDimension::CubeArray,
        ":d3" => TextureViewDimension::D3,
        _ => return Err(gpu_error(format!("unknown view dimension {name}"))),
    })
}

pub(super) fn comparison(name: &str) -> Result<CompareFunction> {
    Ok(match name {
        ":never" => CompareFunction::Never,
        ":less" => CompareFunction::Less,
        ":less-equal" => CompareFunction::LessEqual,
        ":equal" => CompareFunction::Equal,
        ":greater-equal" => CompareFunction::GreaterEqual,
        ":greater" => CompareFunction::Greater,
        ":always" => CompareFunction::Always,
        _ => return Err(gpu_error(format!("unsupported comparison {name}"))),
    })
}

fn parameter<'a>(declaration: &'a Declaration, name: &Value) -> Result<(usize, &'a Parameter)> {
    let name = name.text()?;
    let Some((index, parameter)) = declaration.parameters.iter().enumerate().find(|(_, p)| p.name == name)
    else {
        return Err(gpu_error(format!("ABI references unknown parameter {name}")));
    };
    Ok((index, parameter))
}

fn binding_type(kind: &str, parameter: &Parameter) -> Result<BindingType> {
    let options = &parameter.options;
    Ok(match kind {
        ":storage" => BindingType::Buffer {
            ty: BufferBindingType::Storage {
                read_only: parameter.access.as_deref() == Some(":read"),
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        ":uniform" => BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        ":sampled-texture" => BindingType::Texture {
            sample_type: match options
                .optional(":sample-type")
                .map(Value::text)
                .transpose()?
                .unwrap_or(":float")
            {
                ":filterable-float" => TextureSampleType::Float { filterable: true },
                ":float" => TextureSampleType::Float { filterable: false },
                ":sint" => TextureSampleType::Sint,
                ":uint" => TextureSampleType::Uint,
                ":depth" => TextureSampleType::Depth,
                other => return Err(gpu_error(format!("unknown sample type {other}"))),
            },
            view_dimension: dimension(options.text(":dimension")?)?,
            multisampled: options.get(":samples")? == &Value::Symbol(":multisampled".into())
                || options.get(":samples")?.u32().is_ok_and(|n| n > 1),
        },
        ":storage-texture" => BindingType::StorageTexture {
            access: match parameter.access.as_deref() {
                Some(":read") => StorageTextureAccess::ReadOnly,
                Some(":write") => StorageTextureAccess::WriteOnly,
                Some(":read-write") => StorageTextureAccess::ReadWrite,
                _ => return Err(gpu_error("storage texture has no access")),
            },
            format: texture_format(options.text(":format")?)?,
            view_dimension: dimension(options.text(":dimension")?)?,
        },
        ":sampler" => BindingType::Sampler(match options.text(":kind")? {
            ":filtering" => SamplerBindingType::Filtering,
            ":non-filtering" => SamplerBindingType::NonFiltering,
            ":comparison" => SamplerBindingType::Comparison,
            other => return Err(gpu_error(format!("unknown sampler kind {other}"))),
        }),
        _ => return Err(gpu_error(format!("unsupported ABI binding {kind}"))),
    })
}

impl WgpuBackend {
    pub(super) fn layouts(
        &self,
        declaration: &Declaration,
        stages: ShaderStages,
    ) -> Result<(Vec<BindGroupLayout>, Vec<PushConstantRange>)> {
        let mut groups: BTreeMap<u32, Vec<BindGroupLayoutEntry>> = BTreeMap::new();
        let mut push_end = 0;
        let mut slots = BTreeSet::new();
        for abi in declaration.options.get(":abi")?.list()? {
            let [name, kind, a, b] = abi.list()? else {
                return Err(gpu_error("unsupported ABI entry shape"));
            };
            let (_, parameter) = parameter(declaration, name)?;
            let kind = kind.text()?;
            if kind == ":push-constant" {
                push_end = push_end.max(
                    a.u32()?
                        .checked_add(b.u32()?)
                        .ok_or_else(|| gpu_error("push constant range overflow"))?,
                );
            } else {
                let set = a.u32()?;
                let binding = b.u32()?;
                if set >= self.device.limits().max_bind_groups || !slots.insert((set, binding)) {
                    return Err(gpu_error("invalid or duplicate binding slot"));
                }
                groups.entry(set).or_default().push(BindGroupLayoutEntry {
                    binding,
                    visibility: stages,
                    ty: binding_type(kind, parameter)?,
                    count: None,
                });
            }
        }
        let count = groups.keys().next_back().map_or(0, |set| set + 1);
        let layouts = (0..count)
            .map(|set| {
                self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some(&declaration.name),
                    entries: groups.get(&set).map(Vec::as_slice).unwrap_or(&[]),
                })
            })
            .collect();
        let push = if push_end == 0 {
            Vec::new()
        } else {
            if push_end > self.device.limits().max_push_constant_size
                || !self.device.features().contains(Features::PUSH_CONSTANTS)
            {
                return Err(gpu_error("device does not support the required push constants"));
            }
            vec![PushConstantRange {
                stages,
                range: 0..push_end,
            }]
        };
        Ok((layouts, push))
    }

    pub(super) fn bind_groups(
        &self,
        declaration: &Declaration,
        arguments: &[Value],
        layouts: &[BindGroupLayout],
    ) -> Result<(Vec<BindGroup>, Vec<(u32, Vec<u8>)>)> {
        if arguments.len() != declaration.parameters.len() {
            return Err(gpu_error(format!(
                "wrong argument count for {}",
                declaration.name
            )));
        }
        for (parameter, value) in declaration.parameters.iter().zip(arguments) {
            if let Some(minimum) = parameter.minimum_bytes()? {
                if self.buffer_size(value)? < minimum {
                    return Err(gpu_error(format!(
                        "{} requires at least {minimum} bytes",
                        parameter.name
                    )));
                }
            }
        }
        let mut groups: BTreeMap<u32, Vec<BindGroupEntry<'_>>> = BTreeMap::new();
        let mut pushes = Vec::new();
        for abi in declaration.options.get(":abi")?.list()? {
            let [name, kind, a, b] = abi.list()? else {
                return Err(gpu_error("unsupported ABI entry shape"));
            };
            let (index, _) = parameter(declaration, name)?;
            let value = &arguments[index];
            if kind.text()? == ":push-constant" {
                pushes.push((a.u32()?, self.read_buffer(value, 0, b.u64()?)?));
                continue;
            }
            let resource = match self.resource(value)? {
                Resource::Buffer(buffer, _) => buffer.as_entire_binding(),
                Resource::View { view, .. } => BindingResource::TextureView(view),
                Resource::Sampler(sampler) => BindingResource::Sampler(sampler),
                Resource::Texture(_) => return Err(gpu_error("shader texture arguments require a view")),
                Resource::HostBuffer(_) => return Err(gpu_error("host spans require push-constant ABI")),
            };
            groups.entry(a.u32()?).or_default().push(BindGroupEntry {
                binding: b.u32()?,
                resource,
            });
        }
        let groups = layouts
            .iter()
            .enumerate()
            .map(|(set, layout)| {
                self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some(&declaration.name),
                    layout,
                    entries: groups.get(&(set as u32)).map(Vec::as_slice).unwrap_or(&[]),
                })
            })
            .collect();
        Ok((groups, pushes))
    }
}
