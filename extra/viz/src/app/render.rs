//! Render-pipeline construction for the interactive `vf` /
//! `testpattern` modes: surface format selection, surface
//! configuration, and the SPIR-V vs WGSL render-pipeline branches.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use wgpu::{
    Adapter, BindGroup, BindGroupLayout, Buffer, ColorTargetState, FragmentState, PipelineLayoutDescriptor,
    PresentMode, PrimitiveState, RenderPipeline, Surface, SurfaceConfiguration, TextureFormat,
    TextureUsages, VertexState,
};

use crate::app::uniforms::build_test_pattern_uniforms;
use crate::spirv::load_spirv_module;

/// Pick a non-sRGB 8-bit surface format. Shadertoy-style shaders
/// apply their own gamma (`pow(col, 0.45)`) and expect the
/// framebuffer to treat their output as final pixels — sRGB
/// formats would re-encode and double-gamma the result. Falls back
/// to the first reported format when no non-sRGB option is offered.
pub fn pick_surface_format(caps: &wgpu::SurfaceCapabilities) -> Result<TextureFormat> {
    caps.formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb())
        .or_else(|| caps.formats.first().copied())
        .ok_or_else(|| anyhow!("surface reports no supported formats"))
}

/// Build the surface configuration: format from `pick_surface_format`,
/// alpha mode from the first reported option, requested `present_mode`,
/// 2-frame latency.
pub fn configure_surface(
    surface: &Surface<'static>,
    device: &wgpu::Device,
    adapter: &Adapter,
    width: u32,
    height: u32,
    present_mode: PresentMode,
) -> Result<SurfaceConfiguration> {
    let caps = surface.get_capabilities(adapter);
    let format = pick_surface_format(&caps)?;
    let alpha_mode =
        caps.alpha_modes.first().copied().ok_or_else(|| anyhow!("surface reports no alpha modes"))?;

    let config = SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format,
        width: width.max(1),
        height: height.max(1),
        present_mode,
        alpha_mode,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(device, &config);
    Ok(config)
}

/// Render pipeline for a SPIR-V module with separate vertex + fragment
/// entry points. When `shadertoy_uniforms` is `Some`, the bind-group
/// layout for the shadertoy uniforms is plumbed at the requested set
/// index, with empty-layout placeholders padding any lower sets so
/// wgpu's contiguous-set rule is satisfied.
pub fn build_spirv_render_pipeline(
    device: &wgpu::Device,
    format: TextureFormat,
    path: &Path,
    vertex_entry: &str,
    fragment_entry: &str,
    shadertoy_uniforms: Option<(&BindGroupLayout, Option<&BindGroupLayout>, u32)>,
    topology: wgpu::PrimitiveTopology,
    // One per `#[location(n)]` vertex-shader input attribute, in
    // declaration order: `(format, shader_location)`. Each becomes its
    // own `VertexBufferLayout` (one buffer per attribute, offset 0,
    // stride = format byte size). Empty for shaders without vertex
    // attributes.
    vertex_attribs: &[(wyn_pipeline_descriptor::VertexFormat, u32)],
) -> Result<RenderPipeline> {
    let module =
        load_spirv_module(device, path).with_context(|| format!("load SPIR-V module {:?}", path))?;

    // The uniform layout (if any) lives at `set`; wgpu requires set
    // numbers to be contiguous, so prefix the empty layout for any
    // lower sets.
    let mut bind_group_layouts: Vec<&BindGroupLayout> = vec![];
    if let Some((uniforms_bgl, empty_bgl, set)) = shadertoy_uniforms {
        if let Some(empty) = empty_bgl {
            for _ in 0..set {
                bind_group_layouts.push(empty);
            }
        }
        bind_group_layouts.push(uniforms_bgl);
    }

    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("layout"),
        bind_group_layouts: &bind_group_layouts,
        push_constant_ranges: &[],
    });

    // One `wgpu::VertexAttribute` + matching `VertexBufferLayout` per
    // declared `#[location(n)]` attribute. Owned locally so both arrays
    // live until `create_render_pipeline` returns.
    let wgpu_attribs: Vec<wgpu::VertexAttribute> = vertex_attribs
        .iter()
        .map(|(fmt, location)| wgpu::VertexAttribute {
            format: super::vertex_buffers::wgpu_vertex_format(*fmt),
            offset: 0,
            shader_location: *location,
        })
        .collect();
    let vertex_buffer_layouts: Vec<wgpu::VertexBufferLayout> = wgpu_attribs
        .iter()
        .zip(vertex_attribs.iter())
        .map(|(attr, (fmt, _))| wgpu::VertexBufferLayout {
            array_stride: fmt.byte_size() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: std::slice::from_ref(attr),
        })
        .collect();

    Ok(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pipeline"),
        layout: Some(&layout),
        vertex: VertexState {
            module: &module,
            entry_point: Some(vertex_entry),
            buffers: &vertex_buffer_layouts,
            compilation_options: Default::default(),
        },
        fragment: Some(FragmentState {
            module: &module,
            entry_point: Some(fragment_entry),
            targets: &[Some(ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: PrimitiveState {
            topology,
            ..PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    }))
}

/// Render pipeline for the embedded WGSL test pattern. Returns the
/// pipeline alongside the resolution buffer + bind group it owns —
/// the caller stashes the buffer for per-frame writes and the bind
/// group for the render pass.
pub fn build_wgsl_render_pipeline(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: TextureFormat,
    width: u32,
    height: u32,
    source: &str,
) -> (RenderPipeline, Buffer, BindGroup) {
    eprintln!("[viz] Loading built-in test pattern shader (WGSL)");

    let (res_buffer, bind_group, bind_group_layout) =
        build_test_pattern_uniforms(device, queue, width, height);

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_pattern_shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("test_pattern_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("test_pattern_pipeline"),
        layout: Some(&layout),
        vertex: VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            targets: &[Some(ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (pipeline, res_buffer, bind_group)
}
