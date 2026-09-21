use crate::rust_results;
use proc_macro2::{Group, Ident, TokenStream, TokenTree};
use quote::{format_ident, quote, ToTokens};
use std::collections::{BTreeMap, BTreeSet};
use syn::visit::{self, Visit};
use syn::{parse2, parse_file, parse_str, ExprPath, File, Item, PatIdent, UseTree, Visibility};

use crate::{
    Allocation, Binding, BlendMode, CullMode, DepthTest, DrawCall, DrawCount, Entry, Expr, FillMode,
    FrameResourceKind, FrontFace, HostError, IndexFormat, IntegerOp, Operation, Pipeline,
    PrimitiveTopology, Program, ResourceId, Scissor, ShaderFormat, ShaderStage, StorageImageFormat,
    TextureViewDimension, VertexFormat, Viewport,
};

pub(crate) fn resource(id: ResourceId) -> Ident {
    format_ident!("resource_{}", id.0)
}
fn scalar(name: &str) -> Ident {
    format_ident!("{}", name.replace('-', "_"))
}
fn texture_format(format: StorageImageFormat) -> TokenStream {
    let variant = match format {
        StorageImageFormat::Rgba8Unorm => quote!(Rgba8Unorm),
        StorageImageFormat::Rgba16Float => quote!(Rgba16Float),
        StorageImageFormat::Rgba32Float => quote!(Rgba32Float),
        StorageImageFormat::R32Float => quote!(R32Float),
    };
    quote!(TextureFormat::#variant)
}

impl Program {
    /// Generate a Rust module using WGPU 27 and fixed-width size arithmetic.
    pub fn to_rust_wgpu(&self, module_path: &str, format: ShaderFormat) -> Result<String, HostError> {
        if format != ShaderFormat::Wgsl {
            return Err(HostError::Invalid("rust-wgpu requires WGSL shader output".into()));
        }
        let support = parse_file(include_str!("rust_support.rs"))?;
        let arithmetic = parse_file(include_str!("arithmetic.rs"))?;
        let mut result_types = parse_file(include_str!("results.rs"))?;
        result_types.items.retain(|item| !matches!(item, Item::Impl(_)));
        let result_types = result_types.items;
        let output_types = parse_file(include_str!("rust_output.rs"))?;
        let functions = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| self.rust_entry(i, e, module_path))
            .collect::<Result<Vec<_>, _>>()?;
        let function_tokens = quote!(#(#functions)*);
        let mut used = BTreeSet::new();
        collect_identifiers(function_tokens.clone(), &mut used);
        let functions = functions
            .into_iter()
            .zip(&self.entries)
            .map(|(tokens, entry)| self.rust_input_names(entry, tokens))
            .collect::<Result<Vec<_>, _>>()?;
        let function_tokens = quote!(#(#functions)*);
        let resource_names = self
            .interface
            .frame_graph
            .resources
            .iter()
            .enumerate()
            .map(|(id, r)| {
                let name = &r.name;
                quote!((#id,#name))
            })
            .collect::<Vec<_>>();
        let mut buffer_fields = vec![];
        for (id, r) in self.interface.frame_graph.resources.iter().enumerate() {
            for b in &r.bindings {
                let members = match &self.bindings(b.pipeline_index)[b.binding_index] {
                    Binding::Uniform { members, .. } | Binding::StorageBuffer { members, .. } => members,
                    _ => continue,
                };
                for member in members {
                    let name = &member.name;
                    let offset = member.offset;
                    let size = member.size;
                    buffer_fields.push(quote!((#id,#name,#offset,#size)));
                }
            }
        }
        let mut syntax: File = parse2(quote! {
            //! Generated Wyn host code. Dependency: wgpu 27.
            mod support {#support pub mod arithmetic {#arithmetic}}
            pub mod output {#output_types #(#result_types)*}
            use output::{OutputDescriptor,OutputValue,OutputResource,BufferRange,ResultKind,ResultLayout,ResultScalar,ResultField};
            pub use support::HostError;
            use support::{ceiling, dimension, floor, size};
            use std::borrow::Cow;
            use wgpu::{
                BindGroupDescriptor, BindGroupEntry, BindingResource, BlendComponent, BlendFactor,
                BlendOperation, BlendState, Buffer, BufferDescriptor, BufferUsages, ColorTargetState,
                ColorWrites, CompareFunction, ComputePassDescriptor, ComputePipelineDescriptor,
                DepthStencilState, Device, Extent3d, Face, FragmentState, FrontFace, IndexFormat,
                LoadOp, MultisampleState, Operations, PolygonMode, PrimitiveState, PrimitiveTopology,
                Queue, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
                RenderPassDescriptor, RenderPipelineDescriptor, Sampler, ShaderModuleDescriptor,
                ShaderSource, StoreOp, Texture, TextureDescriptor, TextureDimension, TextureFormat,
                TextureUsages, TextureViewDescriptor, TextureViewDimension, VertexAttribute,
                VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
            };
            pub const RESOURCE_NAMES:&[(usize,&str)]=&[#(#resource_names),*];
            pub const BUFFER_FIELDS:&[(usize,&str,u32,u32)]=&[#(#buffer_fields),*];
            #function_tokens
        })?;
        for item in &mut syntax.items {
            if let Item::Mod(module) = item {
                if module.ident == "support" {
                    if let Some((_, items)) = &mut module.content {
                        retain_helpers(items, &used);
                    }
                }
            }
        }
        syntax.items.retain_mut(|item| match item {
            Item::Use(import) => retain_import(&mut import.tree, &used),
            _ => true,
        });
        Ok(prettyplease::unparse(&syntax))
    }

    fn rust_input_names(&self, entry: &Entry, tokens: TokenStream) -> Result<TokenStream, HostError> {
        let mut names = InputNames::default();
        names.visit_file(&parse2(tokens.clone())?);
        names.used.extend(["RESOURCE_NAMES".into(), "BUFFER_FIELDS".into()]);
        names.used.extend((0..self.entries.len()).map(|i| format!("ENTRY_{i}")));
        let pipelines: BTreeSet<_> = entry
            .operations
            .iter()
            .map(|operation| match operation {
                Operation::Dispatch { pipeline, .. } | Operation::Draw { pipeline } => *pipeline,
            })
            .collect();
        let mut replacements = BTreeMap::new();
        for &id in &entry.inputs {
            let frame_resource = &self.interface.frame_graph.resources[id.0];
            let source = frame_resource
                .bindings
                .iter()
                .find(|binding| pipelines.contains(&binding.pipeline_index))
                .map(|binding| binding.name.as_str())
                .unwrap_or(&frame_resource.name);
            let base = rust_results::name(&source.replace('-', "_"));
            let mut candidate = base.clone();
            let mut suffix = 2;
            let name = loop {
                if !names.used.contains(&candidate) {
                    if let Ok(ident) = parse_str::<Ident>(&candidate)
                        .or_else(|_| parse_str::<Ident>(&format!("r#{candidate}")))
                    {
                        names.used.insert(candidate);
                        break ident;
                    }
                }
                candidate = format!("{base}_{suffix}");
                suffix += 1;
            };
            replacements.insert(resource(id).to_string(), name);
        }
        Ok(rename_identifiers(tokens, &replacements))
    }

    fn rust_expr(&self, expr: &Expr, entry: &Entry) -> TokenStream {
        match expr {
            Expr::I32 { op, left, right } | Expr::U32 { op, left, right } => {
                let left = self.rust_expr(left, entry);
                let right = self.rust_expr(right, entry);
                let convert =
                    if matches!(expr, Expr::I32 { .. }) { quote!(i32_value) } else { quote!(u32_value) };
                let method = match op {
                    IntegerOp::Add => quote!(wrapping_add),
                    IntegerOp::Subtract => quote!(wrapping_sub),
                    IntegerOp::Multiply => quote!(wrapping_mul),
                };
                quote!(i64::from(support::arithmetic::#convert(#left)?.#method(support::arithmetic::#convert(#right)?)))
            }
            Expr::Integer(n) => quote!(#n),
            Expr::Input(name) => {
                let id = scalar(name);
                quote!(i64::from(#id))
            }
            Expr::BufferSize(r) => {
                let id = resource(*r);
                if entry.inputs.contains(r) {
                    quote!(support::arithmetic::signed_size(#id.size())?)
                } else {
                    let bytes = format_ident!("resource_{}_bytes", r.0);
                    quote!(support::arithmetic::signed_size(#bytes)?)
                }
            }
            Expr::ReadScalar {
                resource: r,
                offset,
                signed,
            } => {
                let id = resource(*r);
                let ty = if *signed { quote!(i32) } else { quote!(u32) };
                if matches!(self.resource_binding(*r), Some(Binding::PushConstant { .. })) {
                    quote!(i64::from(#ty::from_le_bytes(support::scalar_bytes(#id,#offset)?)))
                } else {
                    let read = if *signed { quote!(read_i32) } else { quote!(read_u32) };
                    quote!(i64::from(support::#read(device,queue,&#id,#offset)?))
                }
            }
            Expr::TextureDimension { resource: r, axis } => {
                let id = resource(*r);
                match axis {
                    0 => quote!(i64::from(#id.width())),
                    1 => quote!(i64::from(#id.height())),
                    _ => quote!(i64::from(#id.depth_or_array_layers())),
                }
            }
            Expr::Add(a, b) => {
                let a = self.rust_expr(a, entry);
                let b = self.rust_expr(b, entry);
                quote!(support::arithmetic::add(#a,#b)?)
            }
            Expr::Subtract(a, b) => {
                let a = self.rust_expr(a, entry);
                let b = self.rust_expr(b, entry);
                quote!(support::arithmetic::subtract(#a,#b)?)
            }
            Expr::Multiply(a, b) => {
                let a = self.rust_expr(a, entry);
                let b = self.rust_expr(b, entry);
                quote!(support::arithmetic::multiply(#a,#b)?)
            }
            Expr::Floor(a, b) => {
                let a = self.rust_expr(a, entry);
                let b = self.rust_expr(b, entry);
                quote!(floor(#a,#b)?)
            }
            Expr::Ceiling(a, b) => {
                let a = self.rust_expr(a, entry);
                let b = self.rust_expr(b, entry);
                quote!(ceiling(#a,#b)?)
            }
            Expr::Mod(a, b) => {
                let a = self.rust_expr(a, entry);
                let b = self.rust_expr(b, entry);
                quote!(support::arithmetic::modulo(#a,#b)?)
            }
            Expr::Min(a, b) => {
                let a = self.rust_expr(a, entry);
                let b = self.rust_expr(b, entry);
                quote!((#a).min(#b))
            }
            Expr::Max(a, b) => {
                let a = self.rust_expr(a, entry);
                let b = self.rust_expr(b, entry);
                quote!((#a).max(#b))
            }
        }
    }

    fn rust_entry(&self, index: usize, entry: &Entry, module_path: &str) -> Result<TokenStream, HostError> {
        let name = format_ident!("host_{}", rust_results::name(&entry.name));
        let source_name = &entry.name;
        let mut params = vec![];
        for &r in &entry.inputs {
            let id = resource(r);
            let ty = match self.interface.frame_graph.resources[r.0].kind {
                FrameResourceKind::StorageBuffer | FrameResourceKind::Uniform => quote!(&Buffer),
                FrameResourceKind::PushConstant => quote!(&[u8]),
                FrameResourceKind::Texture | FrameResourceKind::StorageTexture => quote!(&Texture),
                FrameResourceKind::Sampler => quote!(&Sampler),
            };
            params.push(quote!(#id:#ty));
        }
        for name in &entry.scalar_inputs {
            let name = scalar(name);
            params.push(quote!(#name:u32));
        }
        let mut code = vec![];
        for a in &entry.allocations {
            match a {
                Allocation::Buffer { resource: r, bytes } => {
                    let id = resource(*r);
                    let bytes = self.rust_expr(bytes, entry);
                    let length = format_ident!("resource_{}_bytes", r.0);
                    let label = &self.interface.frame_graph.resources[r.0].name;
                    code.push(quote!{
                        let #length=size(#bytes)?;
                        if #length>device.limits().max_buffer_size {return Err(HostError::Invalid(format!("buffer {} exceeds device limit",#label)));}
                        let #id=device.create_buffer(&BufferDescriptor{
                            label:Some(#label),size:#length.max(4),mapped_at_creation:false,
                            usage:BufferUsages::STORAGE|BufferUsages::COPY_SRC|BufferUsages::COPY_DST|BufferUsages::VERTEX|BufferUsages::INDEX|BufferUsages::INDIRECT,
                        });
                    });
                }
                Allocation::Texture {
                    resource: r,
                    width,
                    height,
                } => {
                    let id = resource(*r);
                    let width = self.rust_expr(width, entry);
                    let height = self.rust_expr(height, entry);
                    let Some(Binding::StorageTexture { format, .. }) = self.texture_binding(*r) else {
                        return Err(HostError::Invalid("texture allocation without format".into()));
                    };
                    let format = texture_format(*format);
                    let label = &self.interface.frame_graph.resources[r.0].name;
                    code.push(quote!{let #id=device.create_texture(&TextureDescriptor{
                        label:Some(#label),size:Extent3d{width:dimension(#width)?,height:dimension(#height)?,depth_or_array_layers:1},
                        mip_level_count:1,sample_count:1,dimension:TextureDimension::D2,format:#format,
                        usage:TextureUsages::STORAGE_BINDING|TextureUsages::TEXTURE_BINDING|TextureUsages::RENDER_ATTACHMENT|TextureUsages::COPY_SRC|TextureUsages::COPY_DST,
                        view_formats:&[],
                    });});
                }
            }
        }
        for (ordinal, op) in entry.operations.iter().enumerate() {
            code.push(match op {
                Operation::Dispatch {
                    pipeline,
                    stage,
                    groups,
                } => self.rust_dispatch(*pipeline, *stage, groups, entry)?,
                Operation::Draw { pipeline } => self.rust_draw(*pipeline, ordinal, entry)?,
            });
        }
        let results = self.rust_results(entry)?;
        let entry_id = format_ident!("ENTRY_{}", index);
        Ok(quote! {
            pub const #entry_id:&str=#source_name;
            pub fn #name(device:&Device,queue:&Queue,#(#params),*)->Result<OutputDescriptor,HostError>{
                let shader=device.create_shader_module(ShaderModuleDescriptor{label:Some(#source_name),source:ShaderSource::Wgsl(Cow::Borrowed(include_str!(#module_path)))});
                #(#code)*
                Ok(#results)
            }
        })
    }

    fn rust_bindings(
        &self,
        p: usize,
        s: Option<usize>,
    ) -> Result<(TokenStream, Vec<TokenStream>), HostError> {
        let mut groups = BTreeMap::<u32, Vec<TokenStream>>::new();
        let mut views = vec![];
        for b in self.parameter_indices(p, s) {
            let binding = &self.bindings(p)[b];
            let id = resource(self.binding_resource(p, b)?);
            let Some((set, slot)) = binding.slot() else {
                return Err(HostError::Invalid(
                    "rust-wgpu shader has an unlowered push constant".into(),
                ));
            };
            let value = match binding {
                Binding::StorageBuffer { .. } | Binding::Uniform { .. } => quote!(#id.as_entire_binding()),
                Binding::Texture { view_dimension, .. } => {
                    let name = format_ident!("view_{}", b);
                    let d = match view_dimension {
                        TextureViewDimension::D1 => quote!(D1),
                        TextureViewDimension::D2 => quote!(D2),
                        TextureViewDimension::D2Array => quote!(D2Array),
                        TextureViewDimension::Cube => quote!(Cube),
                        TextureViewDimension::CubeArray => quote!(CubeArray),
                        TextureViewDimension::D3 => quote!(D3),
                    };
                    views.push(quote!(let #name=#id.create_view(&TextureViewDescriptor{dimension:Some(TextureViewDimension::#d),..Default::default()});));
                    quote!(BindingResource::TextureView(&#name))
                }
                Binding::StorageTexture { .. } => {
                    let name = format_ident!("view_{}", b);
                    views.push(quote!(let #name=#id.create_view(&Default::default());));
                    quote!(BindingResource::TextureView(&#name))
                }
                Binding::Sampler { .. } => quote!(BindingResource::Sampler(&#id)),
                Binding::PushConstant { .. } => {
                    return Err(HostError::Invalid("rust-wgpu cannot bind push constants".into()))
                }
            };
            groups.entry(set).or_default().push(quote!(BindGroupEntry{binding:#slot,resource:#value}));
        }
        let mut binds = vec![];
        let mut sets = vec![];
        for (set, entries) in groups {
            let name = format_ident!("group_{}", set);
            binds.push(quote!(let #name=device.create_bind_group(&BindGroupDescriptor{label:None,layout:&pipeline.get_bind_group_layout(#set),entries:&[#(#entries),*]});));
            sets.push(quote!(pass.set_bind_group(#set,&#name,&[]);));
        }
        Ok((quote!(#(#views)* #(#binds)*), sets))
    }

    fn rust_dispatch(
        &self,
        p: usize,
        s: usize,
        groups: &[Expr; 3],
        entry: &Entry,
    ) -> Result<TokenStream, HostError> {
        let Pipeline::Compute(c) = &self.interface.pipelines[p] else {
            return Err(HostError::Invalid("dispatch without compute declaration".into()));
        };
        let name = &c.stages[s].entry_point;
        let (bindings, sets) = self.rust_bindings(p, Some(s))?;
        let dims = groups
            .iter()
            .map(|e| {
                let e = self.rust_expr(e, entry);
                quote!(dimension(#e)?)
            })
            .collect::<Vec<_>>();
        Ok(quote! {{
            let groups=[#(#dims),*];
            if groups.iter().any(|&group_count|group_count>device.limits().max_compute_workgroups_per_dimension){return Err(HostError::Invalid("dispatch exceeds device limits".into()));}
            let pipeline=device.create_compute_pipeline(&ComputePipelineDescriptor{
                label:Some(#name),layout:None,module:&shader,entry_point:Some(#name),compilation_options:Default::default(),cache:None,
            });
            #bindings
            let mut encoder=device.create_command_encoder(&Default::default());
            {
                let mut pass=encoder.begin_compute_pass(&ComputePassDescriptor{label:Some(#name),timestamp_writes:None});
                pass.set_pipeline(&pipeline);#(#sets)*
                pass.dispatch_workgroups(groups[0],groups[1],groups[2]);
            }
            queue.submit(Some(encoder.finish()));
        }})
    }
}

/// Emit only support items reachable from this program's host functions.
fn retain_helpers(items: &mut Vec<Item>, roots: &BTreeSet<String>) {
    fn dependencies(items: &[Item], used: &mut BTreeSet<String>) {
        for item in items {
            let name = match item {
                Item::Fn(item) => Some(&item.sig.ident),
                Item::Struct(item) => Some(&item.ident),
                Item::Enum(item) => Some(&item.ident),
                Item::Mod(item) => {
                    if let Some((_, items)) = &item.content {
                        dependencies(items, used);
                    }
                    None
                }
                _ => None,
            };
            if name.is_some_and(|name| used.contains(&name.to_string())) {
                collect_identifiers(item.to_token_stream(), used);
            }
        }
    }
    fn trim(items: &mut Vec<Item>, used: &BTreeSet<String>) {
        items.retain_mut(|item| match item {
            Item::Fn(item) => used.contains(&item.sig.ident.to_string()),
            Item::Struct(item) => used.contains(&item.ident.to_string()),
            Item::Enum(item) => used.contains(&item.ident.to_string()),
            Item::Mod(item) => {
                if let Some((_, items)) = &mut item.content {
                    trim(items, used);
                    items.iter().any(|item| !matches!(item, Item::Use(_)))
                } else {
                    true
                }
            }
            _ => true,
        });
        let mut references = BTreeSet::new();
        for item in items.iter().filter(|item| !matches!(item, Item::Use(_))) {
            collect_identifiers(item.to_token_stream(), &mut references);
        }
        // Public reexports can be referenced by the enclosing generated module.
        items.retain_mut(|item| match item {
            Item::Use(import) => {
                if matches!(import.vis, Visibility::Public(_)) {
                    retain_import(&mut import.tree, used)
                } else {
                    retain_import(&mut import.tree, &references)
                }
            }
            _ => true,
        });
    }
    let mut used = roots.clone();
    loop {
        let count = used.len();
        dependencies(items, &mut used);
        if used.len() == count {
            break;
        }
    }
    trim(items, &used);
}

impl Program {
    fn rust_draw(&self, p: usize, _ordinal: usize, entry: &Entry) -> Result<TokenStream, HostError> {
        let Pipeline::Graphics(g) = &self.interface.pipelines[p] else {
            return Err(HostError::Invalid("draw without graphics declaration".into()));
        };
        let Some(vertex) = g.stages.iter().find(|s| s.stage == ShaderStage::Vertex) else {
            return Err(HostError::Invalid(
                "graphics pipeline without vertex shader".into(),
            ));
        };
        let vertex_name = &vertex.entry_point;
        let fragment = g.stages.iter().find(|s| s.stage == ShaderStage::Fragment);
        let (bindings, sets) = self.rust_bindings(p, None)?;
        let mut views = vec![];
        let mut targets = vec![];
        let mut colors = vec![];
        let color_count = g.fragment_outputs.iter().map(|o| o.location as usize + 1).max().unwrap_or(0);
        targets.resize(color_count, quote!(None));
        colors.resize(color_count, quote!(None));
        let fragment_state = g.invocation.fragment_state;
        let blend = match fragment_state.blend {
            BlendMode::Replace => quote!(BlendState::REPLACE),
            BlendMode::SourceOver => quote!(BlendState::ALPHA_BLENDING),
            BlendMode::Add => quote!(BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add
                }
            }),
        };
        let write_mask = if fragment_state.color_write {
            quote!(ColorWrites::ALL)
        } else {
            quote!(ColorWrites::empty())
        };
        let mut target_size = None;
        for output in &g.fragment_outputs {
            let id = resource(self.target_resource(&output.name)?);
            let view = format_ident!("target_{}", output.location);
            target_size = Some(quote!((#id.width(),#id.height())));
            views.push(quote!(let #view=#id.create_view(&Default::default());));
            targets[output.location as usize] = quote!(Some(ColorTargetState{format:#id.format(),blend:Some(#blend),write_mask:#write_mask}));
            colors[output.location as usize] = quote!(Some(RenderPassColorAttachment{view:&#view,resolve_target:None,depth_slice:None,ops:Operations{load:LoadOp::Load,store:StoreOp::Store}}));
        }
        let (depth_stencil, depth_attachment) = if fragment_state.depth_test != DepthTest::Disabled {
            let id = resource(self.depth_target(p)?);
            if target_size.is_none() {
                target_size = Some(quote!((#id.width(),#id.height())));
            }
            let depth_compare = match fragment_state.depth_test {
                DepthTest::Never => quote!(Never),
                DepthTest::Less => quote!(Less),
                DepthTest::LessEqual => quote!(LessEqual),
                DepthTest::Equal => quote!(Equal),
                DepthTest::GreaterEqual => quote!(GreaterEqual),
                DepthTest::Greater => quote!(Greater),
                DepthTest::Always => quote!(Always),
                DepthTest::Disabled => return Err(HostError::Invalid("disabled depth state".into())),
            };
            let write = fragment_state.depth_write;
            views.push(quote!(let depth_view=#id.create_view(&Default::default());));
            (
                quote!(Some(DepthStencilState{format:TextureFormat::Depth32Float,depth_write_enabled:#write,depth_compare:CompareFunction::#depth_compare,stencil:Default::default(),bias:Default::default()})),
                quote!(Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store
                    }),
                    stencil_ops: None
                })),
            )
        } else {
            (quote!(None), quote!(None))
        };
        let Some(target_size) = target_size else {
            return Err(HostError::Invalid("draw has no attachments".into()));
        };
        let mut attributes = vec![];
        let mut vertex_buffers = vec![];
        let mut vertices = vec![];
        for (index, a) in g.vertex_inputs.iter().enumerate() {
            let name = format_ident!("attributes_{}", index);
            let location = a.slot;
            let stride = u64::from(a.format.byte_size());
            let variant = match a.format {
                VertexFormat::Float32 => quote!(Float32),
                VertexFormat::Float32x2 => quote!(Float32x2),
                VertexFormat::Float32x3 => quote!(Float32x3),
                VertexFormat::Float32x4 => quote!(Float32x4),
                VertexFormat::Sint32 => quote!(Sint32),
                VertexFormat::Sint32x2 => quote!(Sint32x2),
                VertexFormat::Sint32x3 => quote!(Sint32x3),
                VertexFormat::Sint32x4 => quote!(Sint32x4),
                VertexFormat::Uint32 => quote!(Uint32),
                VertexFormat::Uint32x2 => quote!(Uint32x2),
                VertexFormat::Uint32x3 => quote!(Uint32x3),
                VertexFormat::Uint32x4 => quote!(Uint32x4),
            };
            attributes.push(quote!(let #name=[VertexAttribute{format:VertexFormat::#variant,offset:0,shader_location:#location}];));
            vertex_buffers.push(quote!(VertexBufferLayout{array_stride:#stride,step_mode:VertexStepMode::Vertex,attributes:&#name}));
            let Some(id) = self.interface.frame_graph.resources.iter().position(|r| r.name == a.name)
            else {
                return Err(HostError::Invalid(format!("missing vertex buffer {}", a.name)));
            };
            let id = resource(ResourceId(id));
            let index = index as u32;
            vertices.push(quote!(pass.set_vertex_buffer(#index,#id.slice(..));));
        }
        let fragment = match fragment {
            Some(fragment) => {
                let name = &fragment.entry_point;
                quote!(Some(FragmentState{module:&shader,entry_point:Some(#name),compilation_options:Default::default(),targets:&[#(#targets),*]}))
            }
            None => quote!(None),
        };
        let topology = match g.invocation.topology {
            PrimitiveTopology::TriangleList => quote!(TriangleList),
            PrimitiveTopology::TriangleStrip => quote!(TriangleStrip),
            PrimitiveTopology::LineList => quote!(LineList),
            PrimitiveTopology::LineStrip => quote!(LineStrip),
            PrimitiveTopology::PointList => quote!(PointList),
        };
        let r = g.invocation.raster_state;
        let front = match r.front_face {
            FrontFace::Clockwise => quote!(Cw),
            FrontFace::CounterClockwise => quote!(Ccw),
        };
        let cull = match r.cull {
            CullMode::None => quote!(None),
            CullMode::Front => quote!(Some(Face::Front)),
            CullMode::Back => quote!(Some(Face::Back)),
        };
        let fill = match r.fill {
            FillMode::Fill => quote!(Fill),
            FillMode::Line => quote!(Line),
            FillMode::Point => quote!(Point),
        };
        let viewport = match r.viewport {
            Viewport::Target => quote!(),
            Viewport::Custom {
                origin,
                extent,
                depth,
            } => {
                let [x, y] = origin;
                let [w, h] = extent;
                let [lo, hi] = depth;
                quote!(pass.set_viewport(#x,#y,#w,#h,#lo,#hi);)
            }
        };
        let scissor = match r.scissor {
            Scissor::Target => quote!(),
            Scissor::Custom { origin, extent } => {
                let [x, y] = origin;
                let [w, h] = extent;
                quote! {
                    let (width,height)=#target_size;
                    let left=i64::from(#x).clamp(0,i64::from(width));
                    let top=i64::from(#y).clamp(0,i64::from(height));
                    let right=(i64::from(#x)+i64::from(#w)).clamp(left,i64::from(width));
                    let bottom=(i64::from(#y)+i64::from(#h)).clamp(top,i64::from(height));
                    pass.set_scissor_rect(left as u32,top as u32,(right-left) as u32,(bottom-top) as u32);
                }
            }
        };
        let draw = self.rust_draw_call(p, &g.invocation.draw, entry)?;
        Ok(quote! {{
            #(#views)* #(#attributes)*
            let pipeline=device.create_render_pipeline(&RenderPipelineDescriptor{
                label:Some(#vertex_name),layout:None,
                vertex:VertexState{module:&shader,entry_point:Some(#vertex_name),compilation_options:Default::default(),buffers:&[#(#vertex_buffers),*]},
                fragment:#fragment,
                primitive:PrimitiveState{topology:PrimitiveTopology::#topology,strip_index_format:None,front_face:FrontFace::#front,cull_mode:#cull,polygon_mode:PolygonMode::#fill,unclipped_depth:false,conservative:false},
                depth_stencil:#depth_stencil,multisample:MultisampleState::default(),multiview:None,cache:None,
            });
            #bindings
            let mut encoder=device.create_command_encoder(&Default::default());
            {
                let mut pass=encoder.begin_render_pass(&RenderPassDescriptor{
                    label:Some(#vertex_name),color_attachments:&[#(#colors),*],depth_stencil_attachment:#depth_attachment,
                    timestamp_writes:None,occlusion_query_set:None,
                });
                pass.set_pipeline(&pipeline);#(#sets)* #(#vertices)* #viewport #scissor #draw
            }
            queue.submit(Some(encoder.finish()));
        }})
    }

    fn rust_draw_call(&self, p: usize, draw: &DrawCall, entry: &Entry) -> Result<TokenStream, HostError> {
        let count = |c: &DrawCount, r: ResourceId| {
            let expr = match c {
                DrawCount::Fixed(n) => Expr::Integer((*n).into()),
                DrawCount::BufferLength => Expr::Input(format!("count-resource-{}", r.0)),
            };
            let expr = self.rust_expr(&expr, entry);
            quote!(dimension(#expr)?)
        };
        let index = |f: &IndexFormat| match f {
            IndexFormat::Uint16 => quote!(IndexFormat::Uint16),
            IndexFormat::Uint32 => quote!(IndexFormat::Uint32),
        };
        Ok(match draw {
            DrawCall::Direct {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => quote! {
                pass.draw(#first_vertex..support::draw_end(#first_vertex,#vertex_count)?,#first_instance..support::draw_end(#first_instance,#instance_count)?);
            },
            DrawCall::Indexed {
                indices,
                index_format,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            } => {
                let r = self.draw_resource(p, indices)?;
                let n = count(index_count, r);
                let id = resource(r);
                let format = index(index_format);
                quote! {pass.set_index_buffer(#id.slice(..),#format);pass.draw_indexed(#first_index..support::draw_end(#first_index,#n)?,#vertex_offset,#first_instance..support::draw_end(#first_instance,#instance_count)?);}
            }
            DrawCall::Indirect {
                commands,
                offset,
                draw_count,
            } => {
                let r = self.draw_resource(p, commands)?;
                let n = count(draw_count, r);
                let id = resource(r);
                quote!(pass.multi_draw_indirect(&#id,#offset,#n);)
            }
            DrawCall::IndexedIndirect {
                indices,
                index_format,
                commands,
                offset,
                draw_count,
            } => {
                let r = self.draw_resource(p, commands)?;
                let n = count(draw_count, r);
                let id = resource(r);
                let indices = resource(self.draw_resource(p, indices)?);
                let format = index(index_format);
                quote!(pass.set_index_buffer(#indices.slice(..),#format);pass.multi_draw_indexed_indirect(&#id,#offset,#n);)
            }
        })
    }
}

#[derive(Default)]
struct InputNames {
    used: BTreeSet<String>,
}

impl<'ast> Visit<'ast> for InputNames {
    fn visit_pat_ident(&mut self, pattern: &'ast PatIdent) {
        self.used.insert(pattern.ident.to_string());
        visit::visit_pat_ident(self, pattern);
    }

    fn visit_expr_path(&mut self, expression: &'ast ExprPath) {
        if let Some(segment) = expression.path.segments.first() {
            self.used.insert(segment.ident.to_string());
        }
        visit::visit_expr_path(self, expression);
    }
}

fn rename_identifiers(tokens: TokenStream, replacements: &BTreeMap<String, Ident>) -> TokenStream {
    tokens
        .into_iter()
        .map(|token| match token {
            TokenTree::Ident(id) => {
                TokenTree::Ident(replacements.get(&id.to_string()).cloned().unwrap_or(id))
            }
            TokenTree::Group(group) => {
                let mut renamed = Group::new(
                    group.delimiter(),
                    rename_identifiers(group.stream(), replacements),
                );
                renamed.set_span(group.span());
                TokenTree::Group(renamed)
            }
            TokenTree::Punct(_) | TokenTree::Literal(_) => token,
        })
        .collect()
}

fn collect_identifiers(tokens: TokenStream, used: &mut BTreeSet<String>) {
    for token in tokens {
        match token {
            TokenTree::Ident(id) => {
                used.insert(id.to_string());
            }
            TokenTree::Group(group) => collect_identifiers(group.stream(), used),
            TokenTree::Punct(_) | TokenTree::Literal(_) => {}
        }
    }
}

fn retain_import(tree: &mut UseTree, used: &BTreeSet<String>) -> bool {
    match tree {
        UseTree::Name(name) => used.contains(&name.ident.to_string()),
        UseTree::Rename(name) => used.contains(&name.rename.to_string()),
        UseTree::Path(path) => retain_import(&mut path.tree, used),
        UseTree::Group(group) => {
            group.items = std::mem::take(&mut group.items)
                .into_iter()
                .filter_map(|mut item| retain_import(&mut item, used).then_some(item))
                .collect();
            !group.items.is_empty()
        }
        UseTree::Glob(_) => true,
    }
}
