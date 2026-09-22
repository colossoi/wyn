//! Explicit pipeline layouts retain published bindings used only by host scalars.
use crate::rust_wgpu::texture_format;
use crate::{
    Access, Binding, HostError, Program, SamplerBindingType, ShaderFormat, TextureSampleType,
    TextureViewDimension,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::collections::BTreeMap;

impl Program {
    pub(crate) fn rust_layout(
        &self,
        pipeline: usize,
        stage: Option<usize>,
        format: ShaderFormat,
    ) -> Result<TokenStream, HostError> {
        let mut groups = BTreeMap::<u32, Vec<TokenStream>>::new();
        let stages = if stage.is_some() {
            quote!(ShaderStages::COMPUTE)
        } else {
            quote!(ShaderStages::VERTEX | ShaderStages::FRAGMENT)
        };
        let mut push_end = 0;
        for index in self.parameter_indices(pipeline, stage) {
            let binding = self.shader_binding(pipeline, stage, index, format)?;
            if let Binding::PushConstant { offset, size, .. } = binding.as_ref() {
                let Some(end) = offset.checked_add(*size) else {
                    return Err(HostError::Invalid("push constant range overflow".into()));
                };
                if offset % 4 != 0 || size % 4 != 0 {
                    return Err(HostError::Invalid("unaligned push constant range".into()));
                }
                push_end = push_end.max(end);
                continue;
            }
            let Some((set, slot)) = binding.slot() else {
                return Err(HostError::Invalid(
                    "compute layout contains an unlowered push constant".into(),
                ));
            };
            let ty = binding_type(&binding)?;
            groups.entry(set).or_default().push(quote!(BindGroupLayoutEntry {
                binding: #slot, visibility: #stages, ty: #ty, count: None,
            }));
        }
        let count = groups.keys().next_back().map_or(0, |set| set + 1);
        let mut code = vec![];
        let pushes = if push_end == 0 {
            quote!()
        } else {
            code.push(quote! {
                if #push_end > device.limits().max_push_constant_size || !device.features().contains(Features::PUSH_CONSTANTS) {
                    return Err(HostError::Invalid("device does not support the required push constants".into()));
                }
            });
            quote!(PushConstantRange { stages: #stages, range: 0..#push_end })
        };
        let mut layouts = vec![];
        for set in 0..count {
            let entries = groups.get(&set).map(Vec::as_slice).unwrap_or(&[]);
            let name = format_ident!("layout_{}", set);
            code.push(
                quote!(let #name = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None, entries: &[#(#entries),*],
            });),
            );
            layouts.push(name);
        }
        Ok(
            quote!(#(#code)* let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[#(&#layouts),*], push_constant_ranges: &[#pushes],
        });),
        )
    }
}

fn binding_type(binding: &Binding) -> Result<TokenStream, HostError> {
    Ok(match binding {
        Binding::StorageBuffer { access, .. } => {
            let read_only = *access == Access::ReadOnly;
            quote!(BindingType::Buffer {ty: BufferBindingType::Storage {read_only: #read_only}, has_dynamic_offset: false, min_binding_size: None})
        }
        Binding::Uniform { .. } => quote!(BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None
        }),
        Binding::Texture {
            sample_type,
            view_dimension,
            multisampled,
            ..
        } => {
            let sample = match sample_type {
                TextureSampleType::Float { filterable } => {
                    quote!(TextureSampleType::Float {filterable: #filterable})
                }
                TextureSampleType::Sint => quote!(TextureSampleType::Sint),
                TextureSampleType::Uint => quote!(TextureSampleType::Uint),
                TextureSampleType::Depth => quote!(TextureSampleType::Depth),
            };
            let dimension = match view_dimension {
                TextureViewDimension::D1 => quote!(D1),
                TextureViewDimension::D2 => quote!(D2),
                TextureViewDimension::D2Array => quote!(D2Array),
                TextureViewDimension::Cube => quote!(Cube),
                TextureViewDimension::CubeArray => quote!(CubeArray),
                TextureViewDimension::D3 => quote!(D3),
            };
            quote!(BindingType::Texture {sample_type: #sample, view_dimension: TextureViewDimension::#dimension, multisampled: #multisampled})
        }
        Binding::Sampler { binding_type, .. } => {
            let kind = match binding_type {
                SamplerBindingType::Filtering => quote!(Filtering),
                SamplerBindingType::NonFiltering => quote!(NonFiltering),
                SamplerBindingType::Comparison => quote!(Comparison),
            };
            quote!(BindingType::Sampler(SamplerBindingType::#kind))
        }
        Binding::StorageTexture { format, access, .. } => {
            let format = texture_format(*format);
            let access = match access {
                Access::ReadOnly => quote!(ReadOnly),
                Access::WriteOnly => quote!(WriteOnly),
                Access::ReadWrite => quote!(ReadWrite),
            };
            quote!(BindingType::StorageTexture {access: StorageTextureAccess::#access, format: #format, view_dimension: TextureViewDimension::D2})
        }
        Binding::PushConstant { .. } => {
            return Err(HostError::Invalid(
                "compute layout contains an unlowered push constant".into(),
            ))
        }
    })
}
