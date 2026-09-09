//! Shadertoy-style uniform support for the interactive `vf` mode.
//!
//! The wyn compiler emits a JSON sidecar (`<spv_path>.json`) alongside
//! the SPIR-V module declaring the (set, binding) location of each
//! uniform the shader references — `iResolution`, `iTime`, `iMouse`,
//! `iFrame`, or their unprefixed playground aliases. This module reads
//! that sidecar, allocates the uniform buffers, and builds a single bind
//! group sized to the actually-declared uniforms.
//!
//! `--shadertoy` is the only path that uses `build_shadertoy`; the
//! WGSL test pattern has its own simpler `build_test_pattern_uniforms`
//! that hardcodes a single resolution uniform at (0, 0).

use anyhow::{anyhow, Result};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, BufferDescriptor,
    BufferUsages, ShaderStages,
};

#[cfg(test)]
#[path = "uniforms_tests.rs"]
mod uniforms_tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DynamicUniform {
    Resolution,
    Time,
    Mouse,
    Frame,
}

fn dynamic_uniform(
    name: &str,
    members: &[wyn_pipeline_descriptor::UniformMember],
) -> Option<DynamicUniform> {
    if members.iter().any(|member| member.name != name) {
        return None;
    }
    match name {
        "iResolution" | "resolution" => Some(DynamicUniform::Resolution),
        "iTime" | "time" => Some(DynamicUniform::Time),
        "iMouse" | "mouse" => Some(DynamicUniform::Mouse),
        "iFrame" | "frame" => Some(DynamicUniform::Frame),
        _ => None,
    }
}

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
    dynamic: std::collections::HashMap<(u32, u32), DynamicUniform>,
    pub by_set_binding: std::collections::HashMap<(u32, u32), wgpu::Buffer>,
}

impl PipelineUniforms {
    /// Upload dynamic values and retain identical bytes for host expressions.
    pub fn update(
        &self,
        queue: &wgpu::Queue,
        snapshot: &mut crate::gpu::ParameterBlockBytes,
        resolution: [f32; 3],
        time: f32,
        mouse: [f32; 4],
        frame: u32,
    ) {
        for (key, kind) in &self.dynamic {
            let words = match kind {
                DynamicUniform::Resolution => [
                    resolution[0].to_bits(),
                    resolution[1].to_bits(),
                    resolution[2].to_bits(),
                    0,
                ],
                DynamicUniform::Time => [time.to_bits(), 0, 0, 0],
                DynamicUniform::Mouse => mouse.map(f32::to_bits),
                DynamicUniform::Frame => [frame, 0, 0, 0],
            };
            let bytes = bytemuck::cast_slice(&words);
            queue.write_buffer(&self.by_set_binding[key], 0, bytes);
            snapshot.insert(*key, bytes.to_vec());
        }
    }
}

/// Allocate Shadertoy-style uniform buffers for every recognized name
/// any pipeline in the descriptor declares as a `Binding::Uniform`.
/// Each `(set, binding)` is allocated once; compute pipelines that
/// declare the same `(set, binding)` as the graphics pipeline reuse
/// the same buffer.
///
/// Recognized dynamic names: `iResolution`, `iTime`, `iMouse`, and
/// `iFrame`, plus the unprefixed `resolution`, `time`, `mouse`, and
/// `frame` names used by the playground wrapper. Unknown uniform names
/// error when their descriptor has no published size — silently dropping
/// them would leave their bind slot unbound at draw time. Initial buffer
/// contents are zero; the per-frame render path writes dynamic values.
pub fn build_pipeline_uniforms(
    device: &wgpu::Device,
    all_uniform_bindings: &[wyn_pipeline_descriptor::Binding],
) -> Result<PipelineUniforms> {
    use std::collections::HashMap;
    use wyn_pipeline_descriptor::Binding;

    let mut dynamic_slots = HashMap::new();
    let mut by_set_binding: HashMap<(u32, u32), wgpu::Buffer> = HashMap::new();

    for b in all_uniform_bindings {
        let Binding::Uniform {
            set,
            binding,
            name,
            size,
            members,
        } = b
        else {
            continue;
        };
        // Same (set, binding) may be declared by multiple pipelines —
        // only allocate the buffer once.
        if by_set_binding.contains_key(&(*set, *binding)) {
            continue;
        }
        let dynamic = dynamic_uniform(name, members);
        let (size_bytes, label) = match dynamic {
            Some(DynamicUniform::Resolution) => (
                std::mem::size_of::<ResolutionUniform>() as u64,
                format!("pipeline.uniform.{name}"),
            ),
            Some(DynamicUniform::Time) => (
                // Pad to 16 bytes — wgpu's UNIFORM minimum binding size
                // is 16 on many adapters.
                16u64,
                format!("pipeline.uniform.{name}"),
            ),
            Some(DynamicUniform::Mouse) => (
                std::mem::size_of::<MouseUniform>() as u64,
                format!("pipeline.uniform.{name}"),
            ),
            Some(DynamicUniform::Frame) => (
                std::mem::size_of::<FrameUniform>() as u64,
                format!("pipeline.uniform.{name}"),
            ),
            // Any non-dynamic uniform with a published block size gets a
            // zero-initialized buffer of that size; `--uniform` writes
            // member values into it at startup. Descriptors that
            // predate size publication carry 0 and keep erroring.
            None if *size > 0 => ((*size).max(16) as u64, format!("pipeline.uniform.{name}")),
            None => {
                return Err(anyhow!(
                    "viz pipeline-interactive: graphics pipeline declares unknown uniform `{}` \
                     with no published size. Known dynamic names: iResolution/resolution, \
                     iTime/time, iMouse/mouse, iFrame/frame.",
                    name
                ));
            }
        };

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&label),
            size: size_bytes,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        by_set_binding.insert((*set, *binding), buffer);
        if let Some(kind) = dynamic {
            dynamic_slots.insert((*set, *binding), kind);
        }
    }

    Ok(PipelineUniforms {
        dynamic: dynamic_slots,
        by_set_binding,
    })
}
