//! Persistent objects shared by calls into one generated module.

use crate::ShaderFormat;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use std::collections::BTreeMap;

#[derive(Default)]
pub(crate) struct RustContext {
    pub compute: BTreeMap<(usize, usize), TokenStream>,
    pub graphics: BTreeMap<usize, (usize, TokenStream)>,
    pub scratch: bool,
    pub device: bool,
}

pub(crate) fn compute_name(pipeline: usize, stage: usize) -> Ident {
    format_ident!("compute_{}_{}", pipeline, stage)
}

pub(crate) fn render_name(pipeline: usize) -> Ident {
    format_ident!("render_{}", pipeline)
}

impl RustContext {
    pub fn emit(&self, module_path: &str, format: ShaderFormat) -> TokenStream {
        let mut fields = Vec::new();
        let mut initializers = Vec::new();
        let mut setup = Vec::new();
        let mut methods = Vec::new();
        let mut clear = Vec::new();
        let device = if self.device { quote!(device) } else { quote!(_device) };
        if self.device {
            fields.push(quote!(device: Device));
            initializers.push(quote!(device: device.clone()));
        }
        if !self.compute.is_empty() || !self.graphics.is_empty() {
            let source = match format {
                ShaderFormat::Wgsl => quote!(ShaderSource::Wgsl(Cow::Borrowed(include_str!(#module_path)))),
                ShaderFormat::Spirv => quote!(ShaderSource::SpirV(Cow::Owned(support::spirv_words(
                    include_bytes!(#module_path)
                )?))),
            };
            setup.push(quote! {
                let shader = device.create_shader_module(ShaderModuleDescriptor {
                    label: Some(#module_path), source: #source,
                });
            });
        }
        for (&(pipeline, stage), create) in &self.compute {
            let name = compute_name(pipeline, stage);
            fields.push(quote!(#name: ComputePipeline));
            setup.push(quote!(let #name = { #create pipeline };));
            initializers.push(quote!(#name));
        }
        if !self.graphics.is_empty() {
            fields.push(quote!(shader: ShaderModule));
            initializers.push(quote!(shader));
        }
        for (&pipeline, (color_count, create)) in &self.graphics {
            let name = render_name(pipeline);
            fields.push(quote!(#name: HashMap<([TextureFormat; #color_count], Option<TextureFormat>, u32), RenderPipeline>));
            initializers.push(quote!(#name: HashMap::new()));
            clear.push(quote!(self.#name.clear();));
            methods.push(quote! {
                fn #name(&mut self, formats: [TextureFormat; #color_count], depth_format: Option<TextureFormat>, samples: u32) -> Result<RenderPipeline, HostError> {
                    let key = (formats, depth_format, samples);
                    if let Some(pipeline) = self.#name.get(&key) {
                        return Ok(pipeline.clone());
                    }
                    let device = &self.device;
                    let shader = &self.shader;
                    #create
                    self.#name.insert(key, pipeline.clone());
                    Ok(pipeline)
                }
            });
        }
        if self.scratch {
            fields.push(quote!(scratch: BTreeMap<usize, Buffer>));
            initializers.push(quote!(scratch: BTreeMap::new()));
            clear.push(quote!(self.scratch.clear();));
        }
        quote! {
            /// Persistent GPU state for this compiled module. Reuse across entry calls.
            pub struct HostContext { #(#fields,)* }
            impl HostContext {
                /// Load the shader and create compute pipelines once. Graphics variants
                /// are cached on first use by attachment formats and sample count.
                pub fn new(#device: &Device) -> Result<Self, HostError> {
                    #(#setup)*
                    Ok(Self { #(#initializers,)* })
                }
                /// Release cached scratch buffers and render-pipeline variants.
                pub fn clear_caches(&mut self) { #(#clear)* }
                #(#methods)*
            }
        }
    }
}
