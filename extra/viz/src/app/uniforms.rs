//! Shadertoy-style uniform support for the interactive `vf` mode.
//!
//! The wyn compiler emits a JSON sidecar (`<spv_path>.json`) alongside
//! the SPIR-V module declaring the (set, binding) location of each
//! uniform the shader references — `iResolution`, `iTime`, `iMouse`,
//! `iMouse`. This module reads that sidecar, allocates the
//! uniform buffers, and builds a single bind group sized to the
//! actually-declared uniforms.
//!
//! `--shadertoy` is the only path that uses `build_shadertoy`; the
//! WGSL test pattern has its own simpler `build_test_pattern_uniforms`
//! that hardcodes a single resolution uniform at (0, 0).

use anyhow::{Result, anyhow};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, BufferDescriptor,
    BufferUsages, ShaderStages,
};

// ---------------------------------------------------------------------------
// Uniform repr-C structs. wgpu's std140 layout: vec3<f32> pads to 16 bytes,
// so `_pad` fields keep the rust struct in lockstep with the WGSL declarations.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResolutionUniform {
    pub resolution: [f32; 3],
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TimeUniform {
    pub time: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MouseUniform {
    pub mouse: [f32; 4],
}

/// `iFrame` — current frame number. wgpu uniform buffers require a
/// minimum 16-byte size on many adapters, so we pad to a vec4.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameUniform {
    pub frame: u32,
    pub _pad: [u32; 3],
}

// ---------------------------------------------------------------------------
// Sidecar parsing
// ---------------------------------------------------------------------------
pub fn build_test_pattern_uniforms(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
) -> (Buffer, BindGroup, BindGroupLayout) {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("test_pattern_resolution"),
        size: std::mem::size_of::<ResolutionUniform>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let initial = ResolutionUniform {
        resolution: [width as f32, height as f32, 1.0],
        _pad: 0.0,
    };
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&[initial]));

    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("test_pattern_bind_group_layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("test_pattern_bind_group"),
        layout: &layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: None,
            }),
        }],
    });

    (buffer, bind_group, layout)
}

/// Walk a graphics pipeline's declared uniforms and allocate a buffer
/// for each recognized name. Unknown uniform names error out.
/// Returns the buffers (each `Option<wgpu::Buffer>` is `Some` iff the
/// shader declared that uniform) and the per-uniform `BindGroupLayoutEntry`
/// list the caller appends to the render bind-group layout.
///
/// Recognized names:
/// - `iResolution`: vec3<f32> + pad, written per frame from
///   `config.width` / `config.height`.
/// - `iTime`: f32, written per frame from elapsed seconds.
/// - `grid_width` / `grid_height`: i32, set once from the CLI `--grid`.
///
/// The `display_binding` arg is reported only so the error message can
/// flag accidental name collisions; the storage binding itself is
/// added by the caller (it's not a uniform).
#[allow(clippy::type_complexity)]
pub struct PipelineUniforms {
    pub resolution: Option<wgpu::Buffer>,
    pub time: Option<wgpu::Buffer>,
    pub mouse: Option<wgpu::Buffer>,
    pub frame: Option<wgpu::Buffer>,
    pub by_set_binding: std::collections::HashMap<(u32, u32), wgpu::Buffer>,
}

/// Allocate Shadertoy-style uniform buffers for every recognized name
/// any pipeline in the descriptor declares as a `Binding::Uniform`.
/// Each `(set, binding)` is allocated once; compute pipelines that
/// declare the same `(set, binding)` as the graphics pipeline reuse
/// the same buffer.
///
/// Recognized names: `iResolution` (vec3 + pad), `iTime` (f32),
/// `iMouse` (vec4), `iFrame` (u32 + pad). Unknown uniform names error
/// — silently dropping them would leave their bind slot unbound at
/// draw time. Initial buffer contents are zero; the per-frame render
/// path writes updated values.
pub fn build_pipeline_uniforms(
    device: &wgpu::Device,
    all_uniform_bindings: &[wyn_pipeline_descriptor::Binding],
) -> Result<PipelineUniforms> {
    use std::collections::HashMap;
    use wyn_pipeline_descriptor::Binding;

    let mut resolution: Option<wgpu::Buffer> = None;
    let mut time: Option<wgpu::Buffer> = None;
    let mut mouse: Option<wgpu::Buffer> = None;
    let mut frame: Option<wgpu::Buffer> = None;
    let mut by_set_binding: HashMap<(u32, u32), wgpu::Buffer> = HashMap::new();

    for b in all_uniform_bindings {
        let Binding::Uniform {
            set,
            binding,
            name,
            size,
            ..
        } = b
        else {
            continue;
        };
        // Same (set, binding) may be declared by multiple pipelines —
        // only allocate the buffer once.
        if by_set_binding.contains_key(&(*set, *binding)) {
            continue;
        }
        let (size_bytes, label) = match name.as_str() {
            "iResolution" => (
                std::mem::size_of::<ResolutionUniform>() as u64,
                "pipeline.uniform.iResolution".to_string(),
            ),
            "iTime" => (
                // Pad to 16 bytes — wgpu's UNIFORM minimum binding size
                // is 16 on many adapters.
                16u64,
                "pipeline.uniform.iTime".to_string(),
            ),
            "iMouse" => (
                std::mem::size_of::<MouseUniform>() as u64,
                "pipeline.uniform.iMouse".to_string(),
            ),
            "iFrame" => (
                std::mem::size_of::<FrameUniform>() as u64,
                "pipeline.uniform.iFrame".to_string(),
            ),
            // Any other uniform with a published block size gets a
            // zero-initialized buffer of that size; `--uniform` writes
            // member values into it at startup. Descriptors that
            // predate size publication carry 0 and keep erroring.
            other if *size > 0 => ((*size).max(16) as u64, format!("pipeline.uniform.{other}")),
            other => {
                return Err(anyhow!(
                    "viz pipeline-interactive: graphics pipeline declares unknown uniform `{}` \
                     with no published size. Known names: iResolution, iTime, iMouse, iFrame.",
                    other
                ));
            }
        };

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&label),
            size: size_bytes,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Cloning a wgpu::Buffer is a refcount bump, so the lookup map
        // and the typed Option both hold valid handles to the same GPU
        // resource.
        by_set_binding.insert((*set, *binding), buffer.clone());
        match name.as_str() {
            "iResolution" => resolution = Some(buffer),
            "iTime" => time = Some(buffer),
            "iMouse" => mouse = Some(buffer),
            "iFrame" => frame = Some(buffer),
            // User block uniforms live only in by_set_binding; their
            // content is zeroed at creation and written by --uniform.
            _ => {}
        }
    }

    Ok(PipelineUniforms {
        resolution,
        time,
        mouse,
        frame,
        by_set_binding,
    })
}
