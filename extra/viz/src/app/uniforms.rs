//! Shadertoy-style uniform support for the interactive `vf` mode.
//!
//! The wyn compiler emits a JSON sidecar (`<spv_path>.json`) alongside
//! the SPIR-V module declaring the (set, binding) location of each
//! uniform the shader references — `iResolution`, `iTime`, `iMouse`,
//! `difficulty`. This module reads that sidecar, allocates the
//! uniform buffers, and builds a single bind group sized to the
//! actually-declared uniforms.
//!
//! `--shadertoy` is the only path that uses `build_shadertoy`; the
//! WGSL test pattern has its own simpler `build_test_pattern_uniforms`
//! that hardcodes a single resolution uniform at (0, 0).

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, BufferDescriptor,
    BufferUsages, ShaderStages,
};

use crate::json::{Binding, Pipeline, PipelineDescriptor};

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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DifficultyUniform {
    pub difficulty: i32,
    pub _pad: [i32; 3],
}

// ---------------------------------------------------------------------------
// Sidecar parsing
// ---------------------------------------------------------------------------

/// One uniform binding declared in the wyn pipeline JSON sidecar.
#[derive(Debug, Clone)]
struct UniformDecl {
    set: u32,
    binding: u32,
    name: String,
}

/// One storage-buffer binding declared in the sidecar.
#[derive(Debug, Clone)]
struct StorageDecl {
    set: u32,
    binding: u32,
    name: String,
}

/// Read `<spv_path>.json` and extract the uniforms declared by the
/// first graphics pipeline. Empty vec when no sidecar is present.
fn load_sidecar_uniforms(spv_path: &Path) -> Vec<UniformDecl> {
    let json_path = spv_path.with_extension("json");
    let Ok(content) = fs::read_to_string(&json_path) else {
        return Vec::new();
    };
    let Ok(desc) = serde_json::from_str::<PipelineDescriptor>(&content) else {
        return Vec::new();
    };
    for p in &desc.pipelines {
        if let Pipeline::Graphics(g) = p {
            return g
                .bindings
                .iter()
                .filter_map(|b| {
                    if let Binding::Uniform { set, binding, name } = b {
                        Some(UniformDecl {
                            set: *set,
                            binding: *binding,
                            name: name.clone(),
                        })
                    } else {
                        None
                    }
                })
                .collect();
        }
    }
    Vec::new()
}

/// Read `<spv_path>.json` and extract the storage-buffer bindings
/// declared by the first graphics pipeline. Empty vec when no sidecar
/// is present or the pipeline declares no storage buffers.
fn load_sidecar_storage(spv_path: &Path) -> Vec<StorageDecl> {
    let json_path = spv_path.with_extension("json");
    let Ok(content) = fs::read_to_string(&json_path) else {
        return Vec::new();
    };
    let Ok(desc) = serde_json::from_str::<PipelineDescriptor>(&content) else {
        return Vec::new();
    };
    for p in &desc.pipelines {
        if let Pipeline::Graphics(g) = p {
            return g
                .bindings
                .iter()
                .filter_map(|b| {
                    if let Binding::StorageBuffer { set, binding, name, .. } = b {
                        Some(StorageDecl {
                            set: *set,
                            binding: *binding,
                            name: name.clone(),
                        })
                    } else {
                        None
                    }
                })
                .collect();
        }
    }
    Vec::new()
}

// ---------------------------------------------------------------------------
// Shadertoy uniforms (vf --shadertoy)
// ---------------------------------------------------------------------------

/// Buffers + bind group for the shadertoy-style uniforms a SPIR-V
/// shader declared in its JSON sidecar. Each `*_buffer` field is
/// `Some` only when the corresponding uniform appears in the sidecar.
pub struct ShadertoyBindings {
    pub resolution_buffer: Option<Buffer>,
    pub time_buffer: Option<Buffer>,
    pub mouse_buffer: Option<Buffer>,
    pub difficulty_buffer: Option<Buffer>,
    /// Host-uploaded storage buffers, kept alive for the lifetime of
    /// the bind group. Populated only when `--storage-dir` was supplied.
    pub storage_buffers: Vec<Buffer>,
    pub bind_group: BindGroup,
    pub bind_group_layout: BindGroupLayout,
    /// Descriptor-set index where the shader expects this bind group
    /// (read from the sidecar). All declared shadertoy uniforms must
    /// share one set.
    pub bind_group_set: u32,
}

/// Build shadertoy uniform buffers + bind group for a SPIR-V shader.
/// Errors if no sidecar is found, or if the declared uniforms span
/// more than one descriptor set.
///
/// When `storage_dir` is `Some(dir)`, each `storage_buffer` binding the
/// sidecar declares is filled from `<dir>/<binding_name>.bin` — a flat
/// little-endian byte file. Name-matched, so file order and size are
/// never guessed; a missing file is a hard error. The storage buffers
/// join the shadertoy uniforms in the same bind group (they must share
/// a descriptor set).
pub fn build_shadertoy(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    spv_path: &Path,
    difficulty: i32,
    storage_dir: Option<&Path>,
) -> Result<ShadertoyBindings> {
    /// Where the shader expects one of the four shadertoy uniforms.
    /// `actual_*` come from the sidecar; `present` is set if the
    /// sidecar mentioned this uniform's well-known name.
    struct Slot {
        actual_set: u32,
        actual_binding: u32,
        present: bool,
    }
    let mut resolution = Slot {
        actual_set: 0,
        actual_binding: 0,
        present: false,
    };
    let mut time = Slot {
        actual_set: 0,
        actual_binding: 1,
        present: false,
    };
    let mut difficulty_slot = Slot {
        actual_set: 0,
        actual_binding: 2,
        present: false,
    };
    let mut mouse = Slot {
        actual_set: 0,
        actual_binding: 5,
        present: false,
    };

    let sidecar = load_sidecar_uniforms(spv_path);
    if sidecar.is_empty() {
        return Err(anyhow!(
            "viz vf --shadertoy: no sidecar pipeline descriptor found next to {:?}. \
             Recompile the shader with `wyn compile` to emit the `.json` alongside the `.spv`.",
            spv_path
        ));
    }

    // Match well-known names; ignore everything else.
    for u in &sidecar {
        let slot = match u.name.as_str() {
            "iResolution" => Some(&mut resolution),
            "iTime" => Some(&mut time),
            "difficulty" | "iDifficulty" => Some(&mut difficulty_slot),
            "iMouse" => Some(&mut mouse),
            _ => None,
        };
        if let Some(s) = slot {
            s.actual_set = u.set;
            s.actual_binding = u.binding;
            s.present = true;
        }
    }

    // All declared shadertoy uniforms must share a single set; per-set
    // bind groups aren't supported here.
    let present_sets: Vec<u32> = [&resolution, &time, &difficulty_slot, &mouse]
        .iter()
        .filter(|s| s.present)
        .map(|s| s.actual_set)
        .collect();
    let bind_group_set = present_sets.first().copied().unwrap_or(0);
    if present_sets.iter().any(|s| *s != bind_group_set) {
        return Err(anyhow!(
            "viz vf: shadertoy uniforms are split across multiple descriptor sets; \
             only a single set is currently supported"
        ));
    }

    let make_buffer = |label: &'static str, size: u64| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };

    let resolution_buffer = resolution.present.then(|| {
        let buf = make_buffer(
            "resolution_buffer",
            std::mem::size_of::<ResolutionUniform>() as u64,
        );
        let initial = ResolutionUniform {
            resolution: [800.0, 600.0, 1.0],
            _pad: 0.0,
        };
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[initial]));
        buf
    });
    let time_buffer = time.present.then(|| {
        let buf = make_buffer("time_buffer", std::mem::size_of::<TimeUniform>() as u64);
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[TimeUniform { time: 0.0 }]));
        buf
    });
    let difficulty_buffer = difficulty_slot.present.then(|| {
        let buf = make_buffer(
            "difficulty_buffer",
            std::mem::size_of::<DifficultyUniform>() as u64,
        );
        let initial = DifficultyUniform {
            difficulty,
            _pad: [0, 0, 0],
        };
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[initial]));
        buf
    });
    let mouse_buffer = mouse.present.then(|| {
        let buf = make_buffer("mouse_buffer", std::mem::size_of::<MouseUniform>() as u64);
        queue.write_buffer(
            &buf,
            0,
            bytemuck::cast_slice(&[MouseUniform {
                mouse: [0.0, 0.0, 0.0, 0.0],
            }]),
        );
        buf
    });

    let buf_ty = BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let buf_layout = |binding: u32| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX_FRAGMENT,
        ty: buf_ty,
        count: None,
    };
    let slots: Vec<(u32, &Buffer)> = [
        resolution_buffer.as_ref().map(|b| (resolution.actual_binding, b)),
        time_buffer.as_ref().map(|b| (time.actual_binding, b)),
        difficulty_buffer.as_ref().map(|b| (difficulty_slot.actual_binding, b)),
        mouse_buffer.as_ref().map(|b| (mouse.actual_binding, b)),
    ]
    .into_iter()
    .flatten()
    .collect();

    // Storage buffers, if a `--storage-dir <dir>` was supplied. Each
    // `storage_buffer` binding the sidecar declares is filled from
    // `<dir>/<binding_name>.bin` — name-matched, one file per buffer.
    let storage_decls = if storage_dir.is_some() {
        load_sidecar_storage(spv_path)
    } else {
        Vec::new()
    };
    let storage_buffers: Vec<Buffer> = if let Some(dir) = storage_dir {
        if storage_decls.is_empty() {
            return Err(anyhow!(
                "viz vf --storage-dir: shader's sidecar declares no storage_buffer bindings; \
                 nothing to upload"
            ));
        }
        if let Some(stray) = storage_decls.iter().find(|s| s.set != bind_group_set) {
            return Err(anyhow!(
                "viz vf --storage-dir: storage binding '{}' is in descriptor set {} but the \
                 uniforms are in set {}; only a single set is supported",
                stray.name,
                stray.set,
                bind_group_set,
            ));
        }
        storage_decls
            .iter()
            .map(|decl| {
                let path = dir.join(format!("{}.bin", decl.name));
                let data = fs::read(&path).with_context(|| {
                    format!(
                        "viz vf --storage-dir: storage binding '{}' expects {:?}",
                        decl.name, path,
                    )
                })?;
                let buf = device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("storage_{}", decl.name)),
                    size: data.len() as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buf, 0, &data);
                Ok(buf)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let storage_layout: Vec<BindGroupLayoutEntry> = storage_decls
        .iter()
        .map(|d| BindGroupLayoutEntry {
            binding: d.binding,
            visibility: ShaderStages::VERTEX_FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();

    let mut layout_entries: Vec<BindGroupLayoutEntry> =
        slots.iter().map(|(binding, _)| buf_layout(*binding)).collect();
    layout_entries.extend(storage_layout);

    let mut group_entries: Vec<BindGroupEntry> = slots
        .iter()
        .map(|(binding, buffer)| BindGroupEntry {
            binding: *binding,
            resource: BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: None,
            }),
        })
        .collect();
    group_entries.extend(storage_decls.iter().zip(&storage_buffers).map(|(d, buf)| {
        BindGroupEntry {
            binding: d.binding,
            resource: BindingResource::Buffer(wgpu::BufferBinding {
                buffer: buf,
                offset: 0,
                size: None,
            }),
        }
    }));

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("uniform_bind_group_layout"),
        entries: &layout_entries,
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("uniform_bind_group"),
        layout: &bind_group_layout,
        entries: &group_entries,
    });

    Ok(ShadertoyBindings {
        resolution_buffer,
        time_buffer,
        mouse_buffer,
        difficulty_buffer,
        storage_buffers,
        bind_group,
        bind_group_layout,
        bind_group_set,
    })
}

// ---------------------------------------------------------------------------
// Test pattern uniforms (testpattern)
// ---------------------------------------------------------------------------

/// Single resolution uniform at (set 0, binding 0) for the WGSL test
/// pattern. Returns the buffer alongside the bind group + its layout
/// so the caller can keep the buffer for runtime writes and feed the
/// layout into the pipeline-layout descriptor.
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
