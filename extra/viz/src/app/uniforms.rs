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

/// Read `<spv_path>.json` and extract the uniforms declared across
/// every graphics pipeline. The compiler emits one `Graphics`
/// pipeline per entry point, so a typical vertex+fragment shader has
/// two — only the fragment's typically lists the uniforms.
fn load_sidecar_uniforms(spv_path: &Path) -> Result<Vec<UniformDecl>> {
    let json_path = spv_path.with_extension("json");
    let content = fs::read_to_string(&json_path)
        .map_err(|e| anyhow!("sidecar read failed for {:?}: {}", json_path, e))?;
    let desc: PipelineDescriptor = serde_json::from_str(&content)
        .map_err(|e| anyhow!("sidecar parse failed for {:?}: {}", json_path, e))?;
    if !desc.pipelines.iter().any(|p| matches!(p, Pipeline::Graphics(_))) {
        return Err(anyhow!("sidecar {:?} contains no graphics pipeline", json_path));
    }
    Ok(desc
        .pipelines
        .iter()
        .filter_map(|p| if let Pipeline::Graphics(g) = p { Some(&g.bindings) } else { None })
        .flat_map(|bs| bs.iter())
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
        .collect())
}

/// Read `<spv_path>.json` and extract storage-buffer bindings across
/// every graphics pipeline. See `load_sidecar_uniforms` for the
/// error-branch contract.
fn load_sidecar_storage(spv_path: &Path) -> Result<Vec<StorageDecl>> {
    let json_path = spv_path.with_extension("json");
    let content = fs::read_to_string(&json_path)
        .map_err(|e| anyhow!("sidecar read failed for {:?}: {}", json_path, e))?;
    let desc: PipelineDescriptor = serde_json::from_str(&content)
        .map_err(|e| anyhow!("sidecar parse failed for {:?}: {}", json_path, e))?;
    if !desc.pipelines.iter().any(|p| matches!(p, Pipeline::Graphics(_))) {
        return Err(anyhow!("sidecar {:?} contains no graphics pipeline", json_path));
    }
    Ok(desc
        .pipelines
        .iter()
        .filter_map(|p| if let Pipeline::Graphics(g) = p { Some(&g.bindings) } else { None })
        .flat_map(|bs| bs.iter())
        .filter_map(|b| {
            if let Binding::StorageBuffer {
                set, binding, name, ..
            } = b
            {
                Some(StorageDecl {
                    set: *set,
                    binding: *binding,
                    name: name.clone(),
                })
            } else {
                None
            }
        })
        .collect())
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

    let sidecar = load_sidecar_uniforms(spv_path).map_err(|e| anyhow!("viz vf --shadertoy: {}", e))?;
    if sidecar.is_empty() {
        return Err(anyhow!(
            "viz vf --shadertoy: sidecar next to {:?} declares no Uniform bindings on its first graphics pipeline. \
             The shader's entry-param `#[uniform(...)]` attributes aren't being exported into the pipeline descriptor.",
            spv_path
        ));
    }

    // Match well-known names; anything else is a hard error — silently
    // ignoring an unrecognized uniform leaves its binding slot unbound
    // at draw time and the driver crashes with "device lost".
    let mut unknown: Vec<String> = Vec::new();
    for u in &sidecar {
        let slot = match u.name.as_str() {
            "iResolution" => Some(&mut resolution),
            "iTime" => Some(&mut time),
            "difficulty" | "iDifficulty" => Some(&mut difficulty_slot),
            "iMouse" => Some(&mut mouse),
            _ => {
                unknown.push(u.name.clone());
                None
            }
        };
        if let Some(s) = slot {
            s.actual_set = u.set;
            s.actual_binding = u.binding;
            s.present = true;
        }
    }
    if !unknown.is_empty() {
        return Err(anyhow!(
            "viz vf --shadertoy: shader declares uniforms `{}` that aren't in the \
             shadertoy well-known set (iResolution / iTime / iMouse / difficulty). \
             --shadertoy can't supply them; a non-shadertoy host runner is needed.",
            unknown.join("`, `"),
        ));
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
        load_sidecar_storage(spv_path).map_err(|e| anyhow!("viz vf --storage-dir: {}", e))?
    } else {
        Vec::new()
    };
    // It's fine for a shader to declare no storage buffers at all —
    // a vertex-attribute shader puts its data in vertex buffers, not
    // storage. `build_vertex_buffers` covers that side. An empty
    // `storage_decls` here just produces an empty `storage_buffers`.
    let storage_buffers: Vec<Buffer> = if let Some(dir) = storage_dir {
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
    group_entries.extend(
        storage_decls.iter().zip(&storage_buffers).map(|(d, buf)| BindGroupEntry {
            binding: d.binding,
            resource: BindingResource::Buffer(wgpu::BufferBinding {
                buffer: buf,
                offset: 0,
                size: None,
            }),
        }),
    );

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
pub fn build_simulate_uniforms(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    graphics: &wyn_pipeline_descriptor::GraphicsPipeline,
    grid: (u32, u32),
) -> Result<(
    Option<wgpu::Buffer>,
    Option<wgpu::Buffer>,
    Option<wgpu::Buffer>,
    Option<wgpu::Buffer>,
    Vec<wgpu::BindGroupLayoutEntry>,
)> {
    use wyn_pipeline_descriptor::Binding;

    let mut resolution = None;
    let mut time = None;
    let mut grid_w = None;
    let mut grid_h = None;
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();

    for b in &graphics.bindings {
        let Binding::Uniform { binding, name, .. } = b else {
            continue;
        };
        let (size_bytes, init_bytes): (u64, Vec<u8>) = match name.as_str() {
            "iResolution" => {
                let u = ResolutionUniform {
                    resolution: [0.0, 0.0, 1.0],
                    _pad: 0.0,
                };
                (
                    std::mem::size_of::<ResolutionUniform>() as u64,
                    bytemuck::bytes_of(&u).to_vec(),
                )
            }
            "iTime" => {
                let u = TimeUniform { time: 0.0 };
                (
                    std::mem::size_of::<TimeUniform>() as u64,
                    bytemuck::bytes_of(&u).to_vec(),
                )
            }
            "grid_width" => {
                let v: i32 = grid.0 as i32;
                (4, v.to_le_bytes().to_vec())
            }
            "grid_height" => {
                let v: i32 = grid.1 as i32;
                (4, v.to_le_bytes().to_vec())
            }
            other => {
                return Err(anyhow!(
                    "simulate: graphics pipeline declares unknown uniform {:?}; \
                     known: iResolution, iTime, grid_width, grid_height",
                    other
                ));
            }
        };
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("simulate.uniform.{}", name)),
            size: size_bytes,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &init_bytes);

        entries.push(wgpu::BindGroupLayoutEntry {
            binding: *binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        match name.as_str() {
            "iResolution" => resolution = Some(buffer),
            "iTime" => time = Some(buffer),
            "grid_width" => grid_w = Some(buffer),
            "grid_height" => grid_h = Some(buffer),
            _ => unreachable!(),
        }
    }
    Ok((resolution, grid_w, grid_h, time, entries))
}
