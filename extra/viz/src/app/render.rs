//! Render-pipeline construction for the interactive `vf` /
//! `testpattern` modes: surface format selection, surface
//! configuration, and the SPIR-V vs WGSL render-pipeline branches.


use anyhow::{Result, anyhow};
use wgpu::{
    Adapter, BindGroup, Buffer, ColorTargetState, CompareFunction, DepthStencilState,
    FragmentState, PipelineLayoutDescriptor, PresentMode, PrimitiveState, RenderPipeline, Surface,
    SurfaceConfiguration, TextureFormat, TextureUsages, VertexState,
};

use crate::app::uniforms::build_test_pattern_uniforms;

/// Depth attachment format for every render pipeline. `Depth32Float` is
/// universally supported on Vulkan/D3D/Metal and gives ample precision
/// for the masthead's `near=2, far=17000` projection range.
pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

pub(super) fn default_depth_state() -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Less,
        stencil: Default::default(),
        bias: Default::default(),
    }
}

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
        depth_stencil: Some(default_depth_state()),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (pipeline, res_buffer, bind_group)
}
