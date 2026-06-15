//! Headless GPU helpers shared across compute / run / validate / info
//! modes: device acquisition, GPU timestamp profiling, generic
//! buffer & bind-group construction over `Binding`,
//! readback, push-constant assembly, and the `--shadertoy`-style
//! sidecar uniform declarations.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use wgpu::{
    Adapter, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor,
    InstanceFlags, Limits, MemoryHints, PowerPreference, Queue, RequestAdapterOptions, ShaderStages,
    Surface, Trace,
};

use crate::json::{
    Access, Binding, BufferUsage, DispatchLen, DispatchSize, Pipeline, PipelineDescriptor,
    StorageImageFormat, load_f32_json,
};
use crate::specs::PushConstantSpec;
use wyn_pipeline_descriptor::StorageTextureSize;

/// Bundle of the wgpu objects produced by a single
/// `Instance::request_adapter` + `Adapter::request_device` cycle.
/// `surface` is `Some` only when the caller asked for one (the
/// interactive `vf` / `testpattern` modes); headless callers see `None`.
pub struct GpuContext {
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface: Option<Surface<'static>>,
}

impl GpuContext {
    /// Discard the adapter / instance / surface and return just the
    /// device + queue. Convenience for headless callers that don't
    /// need to inspect the adapter after construction.
    pub fn into_device_queue(self) -> (Device, Queue) {
        (self.device, self.queue)
    }

    /// Build a `GpuContext` from a `DeviceRequest`. The features
    /// declared in `desired_features` are intersected with the
    /// adapter's actual support before being requested, so callers
    /// can ask for everything they'd like to use without erroring on
    /// adapters that lack one or another extension.
    pub async fn request(req: DeviceRequest<'_>) -> Result<Self> {
        let instance = Instance::new(&InstanceDescriptor {
            flags: req.instance_flags,
            ..Default::default()
        });

        let surface = match req.surface_target {
            Some(make) => Some(make(&instance)?),
            None => None,
        };

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
            })
            .await
            .context("request_adapter failed")?;

        let supported_features = adapter.features() & req.desired_features;

        let mut limits = Limits::default();
        if let Some(overlay) = req.limits_overlay {
            overlay(&mut limits, &adapter);
        }

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: None,
                required_features: supported_features,
                required_limits: limits,
                memory_hints: MemoryHints::Performance,
                trace: Trace::Off,
            })
            .await
            .context("failed to create logical device")?;

        Ok(Self {
            adapter,
            device,
            queue,
            surface,
        })
    }
}

/// Knobs for `GpuContext::request`. Callers fill in only the fields
/// they care about — `..Default::default()` covers the rest.
pub struct DeviceRequest<'a> {
    /// Wgpu instance flags (e.g. `VALIDATION`, `DEBUG`). Defaults to empty.
    pub instance_flags: InstanceFlags,
    /// Features the caller would like to use. The actual
    /// `required_features` passed to `request_device` is this set
    /// intersected with the adapter's support — features the
    /// adapter lacks are silently dropped.
    pub desired_features: Features,
    /// Optional callback to mutate a default `Limits` in place
    /// (e.g. raise `max_push_constant_size` to the adapter's
    /// reported maximum). Receives the `Adapter` so callers can
    /// query its limits.
    pub limits_overlay: Option<Box<dyn FnOnce(&mut Limits, &Adapter) + 'a>>,
    /// Optional surface-creation callback. `Some` produces a
    /// `Surface<'static>` and routes it through the adapter request
    /// as the compatibility hint; `None` means a headless setup.
    pub surface_target: Option<Box<dyn FnOnce(&Instance) -> Result<Surface<'static>> + 'a>>,
}

impl Default for DeviceRequest<'_> {
    fn default() -> Self {
        Self {
            instance_flags: InstanceFlags::empty(),
            desired_features: Features::empty(),
            limits_overlay: None,
            surface_target: None,
        }
    }
}

/// Description of a single compute-pass dispatch — pipeline,
/// bind-group set, push-constant bytes, dispatch dims, optional
/// timestamp writes. `record` encodes the
/// `begin_compute_pass → set_pipeline/groups/constants/dispatch → end`
/// sequence into the caller's encoder. Encoder lifecycle, queue
/// submit, polling, and any readback all stay at the call site,
/// which keeps multi-stage / chunked / per-stage-encoder flows
/// (pipeline mode) free to drive multiple passes per encoder
/// or split encoders mid-loop.
pub struct ComputeExecutor<'a> {
    pub label: &'a str,
    pub pipeline: &'a wgpu::ComputePipeline,
    /// Bind groups in set order — `bind_groups[i]` binds at set `i`.
    pub bind_groups: &'a [&'a BindGroup],
    /// Empty slice means "don't call `set_push_constants`".
    pub push_constant_bytes: &'a [u8],
    pub dispatch: (u32, u32, u32),
    pub timestamps: Option<wgpu::ComputePassTimestampWrites<'a>>,
}

impl ComputeExecutor<'_> {
    pub fn record(self, encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.label),
            timestamp_writes: self.timestamps,
        });
        cpass.set_pipeline(self.pipeline);
        for (i, bg) in self.bind_groups.iter().enumerate() {
            cpass.set_bind_group(i as u32, *bg, &[]);
        }
        if !self.push_constant_bytes.is_empty() {
            cpass.set_push_constants(0, self.push_constant_bytes);
        }
        cpass.dispatch_workgroups(self.dispatch.0, self.dispatch.1, self.dispatch.2);
    }
}

/// Headless device + queue with the feature set every compute / pipeline /
/// validate / info path expects: SPIR-V passthrough, push
/// constants, and timestamp queries (each conditional on adapter
/// support), plus the adapter's max push-constant size as the limit.
pub async fn create_headless_device(verbose: bool) -> Result<(Device, Queue)> {
    let ctx = GpuContext::request(DeviceRequest {
        desired_features: Features::SPIRV_SHADER_PASSTHROUGH
            | Features::PUSH_CONSTANTS
            | Features::TIMESTAMP_QUERY,
        limits_overlay: Some(Box::new(|limits, adapter| {
            limits.max_push_constant_size = adapter.limits().max_push_constant_size;
        })),
        ..Default::default()
    })
    .await?;

    if verbose {
        let info = ctx.adapter.get_info();
        println!("GPU: {} ({:?})", info.name, info.backend);
        println!("Driver: {} {}", info.driver, info.driver_info);
        println!(
            "SPIRV_SHADER_PASSTHROUGH supported: {}",
            ctx.device.features().contains(Features::SPIRV_SHADER_PASSTHROUGH)
        );
    }

    Ok(ctx.into_device_queue())
}

pub fn create_binding_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bindings: &[Binding],
    inputs: &HashMap<String, PathBuf>,
    dispatch: Option<&DispatchSize>,
    pc_bytes: &[u8],
    verbose: bool,
) -> Result<HashMap<u32, (wgpu::Buffer, u64)>> {
    let mut buffers = HashMap::new();
    // (set, binding) -> byte size, so a length-policy buffer can be sized
    // relative to an already-allocated source buffer.
    let mut byte_sizes: HashMap<(u32, u32), u64> = HashMap::new();

    let make_buffer = |device: &wgpu::Device, queue: &wgpu::Queue, name: &str, data: &[u8]| {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(name),
            size: data.len() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, data);
        buffer
    };

    // Pass 1: host inputs (file-backed) and buffers without an explicit length
    // policy (outputs / un-sized intermediates). Record byte sizes so pass 2
    // can resolve `length: LikeInput` references.
    for b in bindings {
        if let Binding::StorageBuffer {
            set,
            binding,
            name,
            usage,
            length,
            ..
        } = b
        {
            if length.is_some() {
                continue;
            }
            let data_bytes = if let Some(path) = inputs.get(name.as_str()) {
                let data = load_f32_json(path)?;
                if verbose {
                    println!(
                        "Loaded {} elements for '{}' from {}",
                        data.len(),
                        name,
                        path.display()
                    );
                }
                data.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()
            } else if *usage == BufferUsage::Input {
                return Err(anyhow!(
                    "No input file provided for '{}'. Use --input {}:<file.json>",
                    name,
                    name
                ));
            } else {
                // Un-sized intermediate/output: fall back to a default size.
                // Reduce/redomap phase 1 runs a fixed saturating grid
                // (PHASE1_SATURATING_GROUPS=1024 workgroups × 64), so its
                // `partials` buffer needs one element per worker; size the
                // default to cover that grid so phase 1 never writes out of
                // bounds. (Length-bearing buffers are sized in pass 2.)
                let count = 1024 * 64usize;
                if verbose {
                    println!("Allocated {} default zero elements for '{}'", count, name);
                }
                vec![0u8; count * 4]
            };
            let byte_size = data_bytes.len() as u64;
            let buffer = make_buffer(device, queue, name, &data_bytes);
            buffers.insert(*binding, (buffer, byte_size));
            byte_sizes.insert((*set, *binding), byte_size);
        }
    }

    // Pass 2: compiler-managed buffers with an explicit length policy (gather
    // intermediates), sized from their source buffer allocated in pass 1.
    for b in bindings {
        if let Binding::StorageBuffer {
            binding,
            name,
            length: Some(len),
            ..
        } = b
        {
            if buffers.contains_key(binding) {
                continue;
            }
            // A `SameAsDispatch` output holds one element per dispatched thread,
            // so its size needs the resolved dispatch — one workgroup's worth of
            // padding over the exact length, which is harmless slack.
            let byte_size = if let Some(elem_bytes) = len.dispatch_elem_bytes() {
                let dispatch = dispatch.ok_or_else(|| {
                    anyhow!(
                        "cannot size dispatch-length buffer '{}': no dispatch in scope",
                        name
                    )
                })?;
                let (groups, _, _) = resolve_dispatch_size(dispatch, &buffers, pc_bytes);
                let wg = match dispatch {
                    DispatchSize::DerivedFrom { workgroup_size, .. } => *workgroup_size,
                    DispatchSize::Fixed { .. } => 1,
                };
                ((groups * wg) as u64 * elem_bytes as u64).max(4)
            } else {
                len.resolve_bytes(|s, bnd| byte_sizes.get(&(s, bnd)).copied())
                    .ok_or_else(|| {
                        anyhow!(
                            "cannot size buffer '{}': its source buffer was not allocated",
                            name
                        )
                    })?
                    .max(4)
            };
            if verbose {
                println!(
                    "Allocated {} bytes for compiler-managed buffer '{}'",
                    byte_size, name
                );
            }
            let buffer = make_buffer(device, queue, name, &vec![0u8; byte_size as usize]);
            buffers.insert(*binding, (buffer, byte_size));
        }
    }

    Ok(buffers)
}

/// Build a bind group layout + bind group from binding descriptors.
pub fn build_bind_group(
    device: &wgpu::Device,
    bindings: &[Binding],
    buffers: &HashMap<u32, (wgpu::Buffer, u64)>,
) -> Result<(wgpu::BindGroupLayout, BindGroup)> {
    let mut layout_entries = Vec::new();
    let mut group_entries = Vec::new();

    for b in bindings {
        if let Binding::StorageBuffer { binding, access, .. } = b {
            let read_only = *access == Access::ReadOnly;
            layout_entries.push(BindGroupLayoutEntry {
                binding: *binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });

            let (buf, _) =
                buffers.get(binding).ok_or_else(|| anyhow!("No buffer for binding {}", binding))?;
            group_entries.push(BindGroupEntry {
                binding: *binding,
                resource: BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: 0,
                    size: None,
                }),
            });
        }
    }

    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("pipeline_bind_group_layout"),
        entries: &layout_entries,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("pipeline_bind_group"),
        layout: &layout,
        entries: &group_entries,
    });

    Ok((layout, bind_group))
}

/// Compute dispatch dimensions from a DispatchSize spec.
///
/// For 1D-shaped sources (`InputBinding`, `Fixed`, `PushConstant`)
/// returns `(workgroup_count_x, 1, 1)`. For `StorageImage` returns a
/// 2D shape `(width/wg_x, height/wg_y, 1)` sized from the
/// storage-texture pool. Compute entries whose primary output is a
/// storage image rely on this 2D path.
pub fn resolve_dispatch_size(
    dispatch: &DispatchSize,
    buffers: &HashMap<u32, (wgpu::Buffer, u64)>,
    pc_bytes: &[u8],
) -> (u32, u32, u32) {
    // For the 1D-style sources, the pipeline workgroup_size triple
    // doesn't matter — only the x dim (carried inside `DerivedFrom`)
    // is consulted. The `StorageImage` source ignores this wrapper
    // and is routed through the explicit path instead.
    resolve_dispatch_size_with_textures(dispatch, (1, 1, 1), buffers, pc_bytes, &HashMap::new())
}

/// Like `resolve_dispatch_size` but knows how to size a `StorageImage`
/// dispatch from the storage-texture pool. The headless `pipeline`
/// path passes an empty pool; the interactive path passes its
/// `storage_textures` map.
///
/// `pipeline_workgroup_size` is the compute pipeline's full
/// `(x, y, z)` workgroup size from the descriptor — needed to compute
/// per-axis workgroup counts for the 2D `StorageImage` dispatch.
/// `DispatchSize::DerivedFrom` only carries the x dim and treats every
/// other source as 1D; this path uses the full triple instead.
pub fn resolve_dispatch_size_with_textures(
    dispatch: &DispatchSize,
    pipeline_workgroup_size: (u32, u32, u32),
    buffers: &HashMap<u32, (wgpu::Buffer, u64)>,
    pc_bytes: &[u8],
    storage_textures: &HashMap<(u32, u32), StorageTextureResource>,
) -> (u32, u32, u32) {
    match dispatch {
        DispatchSize::Fixed { x, y, z } => (*x, *y, *z),
        DispatchSize::DerivedFrom { len, workgroup_size } => {
            let wg_x = (*workgroup_size).max(1);
            match len {
                DispatchLen::StorageImage { set, binding } => {
                    // 2D dispatch over the texture's resolution.
                    // Workgroup counts per axis divide by the full
                    // pipeline workgroup_size (x and y), not by the
                    // 1D `workgroup_size` in `DerivedFrom`.
                    if let Some(res) = storage_textures.get(&(*set, *binding)) {
                        let size = res.textures[0].size();
                        let wg_y = pipeline_workgroup_size.1.max(1);
                        let wg_z = pipeline_workgroup_size.2.max(1);
                        (
                            size.width.div_ceil(pipeline_workgroup_size.0.max(1)),
                            size.height.div_ceil(wg_y),
                            1u32.div_ceil(wg_z),
                        )
                    } else {
                        // Storage texture wasn't allocated (headless
                        // path with no graphics pipeline, or descriptor
                        // mismatch). Conservative: 1 workgroup.
                        (1, 1, 1)
                    }
                }
                _ => {
                    let elements = resolve_dispatch_len(len, buffers, pc_bytes);
                    (elements.div_ceil(wg_x), 1, 1)
                }
            }
        }
    }
}

/// Resolve a `DerivedFrom` dispatch's iteration count from its `DispatchLen`
/// source: a buffer's element count, a compile-time constant, or a scalar read
/// from the push-constant block.
///
/// `StorageImage` is handled by `resolve_dispatch_size_with_textures`
/// directly (it's 2D, not 1D) and never reaches this helper.
fn resolve_dispatch_len(
    len: &DispatchLen,
    buffers: &HashMap<u32, (wgpu::Buffer, u64)>,
    pc_bytes: &[u8],
) -> u32 {
    match len {
        DispatchLen::InputBinding {
            binding, elem_bytes, ..
        } => buffers.get(binding).map(|(_, size)| (*size / *elem_bytes as u64) as u32).unwrap_or(0),
        DispatchLen::Fixed { count } => *count,
        DispatchLen::PushConstant { offset } => {
            let o = *offset as usize;
            pc_bytes.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).unwrap_or(0)
        }
        DispatchLen::StorageImage { .. } => 1,
    }
}

/// Read back a GPU buffer to CPU as f32 data.
pub fn readback_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    byte_size: u64,
) -> Result<Vec<f32>> {
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("readback_staging"),
        size: byte_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_size);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv().unwrap()?;

    let data = slice.get_mapped_range();
    let u32_data: &[u32] = bytemuck::cast_slice(&data);
    let f32_data: Vec<f32> = u32_data.iter().map(|&x| f32::from_bits(x)).collect();
    drop(data);
    staging.unmap();

    Ok(f32_data)
}

/// Build push constant bytes from descriptor PushConstant bindings + CLI --push-constant args.
/// The descriptor provides the layout (offsets and sizes); the CLI provides values by name.
pub fn build_push_constant_bytes(
    bindings: &[Binding],
    push_constants: &[PushConstantSpec],
    verbose: bool,
) -> Result<Vec<u8>> {
    // Collect PushConstant bindings from the descriptor
    let pc_bindings: Vec<_> = bindings
        .iter()
        .filter_map(|b| {
            if let Binding::PushConstant { offset, size, name } = b {
                Some((name.as_str(), *offset, *size))
            } else {
                None
            }
        })
        .collect();

    if pc_bindings.is_empty() && push_constants.is_empty() {
        return Ok(vec![]);
    }

    // No descriptor bindings: pack sequentially using the offsets the
    // caller assigned via `PushConstantSpec::lay_out_sequential`. The
    // helper guarantees specs are in offset-ascending order, so the
    // last one tells us the total range.
    if pc_bindings.is_empty() && !push_constants.is_empty() {
        let total = push_constants.last().map(|s| s.offset + s.byte_size()).unwrap_or(0) as usize;
        let mut bytes = vec![0u8; total];
        for spec in push_constants {
            let start = spec.offset as usize;
            let end = start + spec.data.len();
            bytes[start..end].copy_from_slice(&spec.data);
        }
        if verbose {
            println!("Push constants ({} bytes, sequential layout):", total);
            for spec in push_constants {
                println!(
                    "  {} @ offset {}: {} bytes",
                    spec.name,
                    spec.offset,
                    spec.byte_size()
                );
            }
        }
        return Ok(bytes);
    }

    // Index CLI push constants by name for descriptor-driven layout
    let cli_specs: HashMap<&str, &PushConstantSpec> =
        push_constants.iter().map(|spec| (spec.name.as_str(), spec)).collect();

    // Compute total size from descriptor
    let total_size = pc_bindings.iter().map(|(_, offset, size)| offset + size).max().unwrap_or(0) as usize;

    let mut bytes = vec![0u8; total_size];

    for (name, offset, size) in &pc_bindings {
        if let Some(spec) = cli_specs.get(*name) {
            if spec.data.len() != *size as usize {
                return Err(anyhow!(
                    "Push constant '{}' expects {} bytes but got {} bytes from CLI",
                    name,
                    size,
                    spec.data.len()
                ));
            }
            let start = *offset as usize;
            let end = start + spec.data.len();
            bytes[start..end].copy_from_slice(&spec.data);
            if verbose {
                println!("Push constant '{}' @ offset {}: {} bytes", name, offset, size);
            }
        }
        // If not provided via CLI, leave as zeros
    }

    Ok(bytes)
}

// =============================================================================
// Storage textures (Phase 1b — Path B)
// =============================================================================
//
// A `Binding::StorageTexture` declares a 2D wgpu::Texture that a compute
// shader writes via `image_store` and a later stage samples via
// `texture_sample` on a sibling `Binding::Texture` at the same
// `(set, binding)` slot. The host allocates ONE `wgpu::Texture` per
// unique `(set, binding)` slot and binds it in each consuming pipeline
// through the appropriate view (storage view for `StorageTexture`
// bindings, sampled view for `Texture` bindings).
//
// Sizing: v1 uses a fixed 1024×1024 dimension. A per-binding sizing
// policy in the descriptor (analogous to `BufferLen::LikeInput`) is
// future work — comes in Phase 3 when the multi-stage simulate mode
// needs descriptor-driven dispatch sizing for the chained passes.

/// Fallback storage-texture extent for `SameAsWindow` allocations
/// before the surface size is known (e.g. headless smoke runs). The
/// runnable simulate path overrides this with the actual swapchain
/// dimensions and re-allocates on resize.
pub const STORAGE_TEXTURE_FALLBACK_SIZE: u32 = 1024;

/// Map a descriptor `StorageImageFormat` to the matching
/// `wgpu::TextureFormat`. Kept in lock-step with the SPIR-V
/// `ImageFormat` map on the compiler side (see
/// `wyn-core/src/spirv/mod.rs::storage_image_format_to_spirv`).
pub fn storage_image_format_to_wgpu(f: StorageImageFormat) -> wgpu::TextureFormat {
    match f {
        StorageImageFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        StorageImageFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        StorageImageFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        StorageImageFormat::R32Float => wgpu::TextureFormat::R32Float,
    }
}

/// Map descriptor `Access` to wgpu's `StorageTextureAccess`. v1 ships
/// `read_only` and `write_only`; `read_write` requires an adapter
/// feature flag and is exposed but the caller must verify support.
fn access_to_storage_texture_access(a: Access) -> wgpu::StorageTextureAccess {
    match a {
        Access::ReadOnly => wgpu::StorageTextureAccess::ReadOnly,
        Access::WriteOnly => wgpu::StorageTextureAccess::WriteOnly,
        Access::ReadWrite => wgpu::StorageTextureAccess::ReadWrite,
    }
}

/// Allocated storage texture + its two views. The storage view is
/// bound for `Binding::StorageTexture` consumers; the sampled view is
/// bound for `Binding::Texture` consumers reading the same underlying
/// resource (cross-stage sharing).
pub struct StorageTextureResource {
    /// Physical texture slots. Always length 1 for ordinary storage
    /// textures; length 2 for ping-pong pairs (one "current"-writable
    /// + one "previous"-readable, swapped each frame by the host).
    pub textures: Vec<wgpu::Texture>,
    /// One storage view per slot (binding-as-storage_image consumer).
    pub storage_views: Vec<wgpu::TextureView>,
    /// One sampled view per slot (binding-as-texture2d consumer).
    pub sampled_views: Vec<wgpu::TextureView>,
}

/// A resolved feedback (ping-pong) pair. The read binding at
/// `(read_set, read_binding)` is bound to the slot OPPOSITE the current
/// parity; the write binding at `(write_set, write_binding)` is bound
/// to the slot AT the current parity. The host increments parity each
/// frame so what was "current" becomes "previous" for next frame.
#[derive(Debug, Clone, Copy)]
pub struct FeedbackPair {
    pub read_set: u32,
    pub read_binding: u32,
    pub write_set: u32,
    pub write_binding: u32,
}

/// Walk every pipeline's bindings, allocate one `wgpu::Texture` per
/// unique `(set, binding)` slot declared as `Binding::StorageTexture`
/// in at least one pipeline. The map is keyed by `(set, binding)` so
/// consumers (compute pipelines that write, fragment pipelines that
/// sample) can look the resource up by descriptor coordinates.
///
/// The texture is created with `STORAGE_BINDING | TEXTURE_BINDING |
/// COPY_SRC | COPY_DST`; the storage view targets the storage binding
/// path, the sampled view targets the texture binding path. wgpu
/// allows the same `wgpu::Texture` to satisfy both binding types as
/// long as the texture was created with both usages, which it is.
pub fn create_storage_textures(
    device: &wgpu::Device,
    descriptor: &PipelineDescriptor,
    surface_size: Option<(u32, u32)>,
    feedback_pairs: &[FeedbackPair],
) -> HashMap<(u32, u32), StorageTextureResource> {
    // A (set, binding) that appears as the WRITE side of any feedback
    // pair needs two physical textures; everything else gets one.
    let pingpong_write_keys: std::collections::HashSet<(u32, u32)> =
        feedback_pairs.iter().map(|p| (p.write_set, p.write_binding)).collect();

    let mut out: HashMap<(u32, u32), StorageTextureResource> = HashMap::new();
    for pipeline in &descriptor.pipelines {
        let bindings: &[Binding] = match pipeline {
            Pipeline::Compute(cp) => &cp.bindings,
            Pipeline::MultiCompute(mc) => &mc.bindings,
            Pipeline::Graphics(gp) => &gp.bindings,
        };
        for b in bindings {
            let Binding::StorageTexture {
                set,
                binding,
                name,
                format,
                size: size_policy,
                ..
            } = b
            else {
                continue;
            };
            let key = (*set, *binding);
            if out.contains_key(&key) {
                continue;
            }
            let wgpu_format = storage_image_format_to_wgpu(*format);
            let (width, height) = match size_policy {
                StorageTextureSize::Fixed { width, height } => (*width, *height),
                StorageTextureSize::SameAsWindow => {
                    surface_size.unwrap_or((STORAGE_TEXTURE_FALLBACK_SIZE, STORAGE_TEXTURE_FALLBACK_SIZE))
                }
            };
            let size = wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };
            let slot_count = if pingpong_write_keys.contains(&key) { 2 } else { 1 };

            let mut textures = Vec::with_capacity(slot_count);
            let mut storage_views = Vec::with_capacity(slot_count);
            let mut sampled_views = Vec::with_capacity(slot_count);
            for slot in 0..slot_count {
                let label_suffix = if slot_count == 1 { String::new() } else { format!(".slot{slot}") };
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("storage_texture_{name}{label_suffix}")),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu_format,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let storage_view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("storage_view_{name}{label_suffix}")),
                    format: Some(wgpu_format),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    ..Default::default()
                });
                let sampled_view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("sampled_view_{name}{label_suffix}")),
                    format: Some(wgpu_format),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    ..Default::default()
                });
                textures.push(texture);
                storage_views.push(storage_view);
                sampled_views.push(sampled_view);
            }
            out.insert(
                key,
                StorageTextureResource {
                    textures,
                    storage_views,
                    sampled_views,
                },
            );
        }
    }
    out
}

/// Host-uploaded texture: a `wgpu::Texture` that the viz host writes
/// each frame from CPU state (today: Shadertoy's keyboard convention,
/// a 256×3 R8Unorm texture). The pool is keyed by `(set, binding)`
/// the same way storage textures are.
pub struct HostTextureResource {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub extent: (u32, u32),
    /// Identifier for the per-frame writer. Drives the
    /// `update_host_textures` dispatch on `State`.
    pub kind: HostTextureKind,
}

/// What CPU state feeds a `HostTextureResource` each frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostTextureKind {
    /// Shadertoy's 256×3 keyboard texture. Row 0 = currently-down,
    /// row 1 = pressed-this-frame, row 2 = toggled. R8Unorm.
    Keyboard,
}

/// Walk every pipeline's bindings; allocate one `wgpu::Texture` for
/// each `Binding::Texture` whose name matches a recognized
/// host-uploaded pattern AND has no sibling `StorageTexture` at the
/// same `(set, binding)`. Texture bindings that look like
/// compute-written content (a sibling `StorageTexture` exists) flow
/// through `create_storage_textures`'s `sampled_view` instead; this
/// pool is only for textures whose contents come from the CPU.
///
/// Today the only recognized host pattern is "keyboard" (case-
/// insensitive) → 256×3 R8Unorm. Future host-uploaded textures
/// (image files, video, audio) would add their own match arms.
pub fn create_host_textures(
    device: &wgpu::Device,
    descriptor: &PipelineDescriptor,
    storage_textures: &HashMap<(u32, u32), StorageTextureResource>,
) -> HashMap<(u32, u32), HostTextureResource> {
    let mut out: HashMap<(u32, u32), HostTextureResource> = HashMap::new();
    for pipeline in &descriptor.pipelines {
        let bindings: &[Binding] = match pipeline {
            Pipeline::Compute(cp) => &cp.bindings,
            Pipeline::MultiCompute(mc) => &mc.bindings,
            Pipeline::Graphics(gp) => &gp.bindings,
        };
        for b in bindings {
            let Binding::Texture {
                set, binding, name, ..
            } = b
            else {
                continue;
            };
            let key = (*set, *binding);
            // If a storage texture covers this slot, the texture binding
            // reads its sampled view — not a host-uploaded path.
            if storage_textures.contains_key(&key) {
                continue;
            }
            if out.contains_key(&key) {
                continue;
            }
            // Name-based classification. v1 recognizes "keyboard" only.
            let lower = name.to_ascii_lowercase();
            let kind = if lower == "keyboard" || lower == "ikeyboard" {
                HostTextureKind::Keyboard
            } else {
                // Unrecognized — leave it out; the bind-group builder
                // will surface a clear "no resource" error when the
                // slot is bound.
                continue;
            };
            let (width, height, format) = match kind {
                HostTextureKind::Keyboard => (256u32, 3u32, wgpu::TextureFormat::R8Unorm),
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("host_texture_{}", name)),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("host_view_{}", name)),
                format: Some(format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });
            out.insert(
                key,
                HostTextureResource {
                    texture,
                    view,
                    extent: (width, height),
                    kind,
                },
            );
        }
    }
    out
}

/// Host-uploaded storage buffer — content lives in CPU state and gets
/// written to the GPU each frame. Parallel to `HostTextureResource`
/// but for `Binding::StorageBuffer` slots. Today the only kind is
/// the keyboard state buffer (768 u32 entries: 3 rows × 256 keycodes,
/// `keyboard[row * 256 + key]` non-zero when set).
pub struct HostBufferResource {
    pub buffer: wgpu::Buffer,
    pub kind: HostBufferKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostBufferKind {
    /// Same encoding as `HostTextureKind::Keyboard` but laid out flat:
    /// 768 u32 entries, `row * 256 + keycode`, row 0 = currently-down,
    /// row 1 = pressed-this-frame, row 2 = toggled.
    Keyboard,
    /// Loaded once at startup from `<storage_dir>/<binding_name>.bin`.
    /// The host doesn't rewrite the contents per frame — the shader
    /// reads whatever the file contained.
    FileLoaded,
}

/// Walk every pipeline's bindings; allocate a `wgpu::Buffer` for each
/// `Binding::StorageBuffer` whose name matches a recognized host-
/// uploaded pattern. Mirrors `create_host_textures` but for storage
/// buffers — used for resources that aren't really 2D image data
/// (e.g. the keyboard state, which is a keycode → state table).
pub fn create_host_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    descriptor: &PipelineDescriptor,
    storage_dir: Option<&std::path::Path>,
    zero_buffers: &HashMap<String, u64>,
) -> Result<HashMap<(u32, u32), HostBufferResource>> {
    let mut out: HashMap<(u32, u32), HostBufferResource> = HashMap::new();
    for pipeline in &descriptor.pipelines {
        let bindings: &[Binding] = match pipeline {
            Pipeline::Compute(cp) => &cp.bindings,
            Pipeline::MultiCompute(mc) => &mc.bindings,
            Pipeline::Graphics(gp) => &gp.bindings,
        };
        for b in bindings {
            let Binding::StorageBuffer {
                set, binding, name, ..
            } = b
            else {
                continue;
            };
            let key = (*set, *binding);
            if out.contains_key(&key) {
                continue;
            }
            let lower = name.to_ascii_lowercase();
            if lower == "keyboard" || lower == "ikeyboard" {
                let byte_size = 768u64 * 4; // 768 u32 entries
                let buffer = device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("host_buffer_{name}")),
                    size: byte_size,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                out.insert(
                    key,
                    HostBufferResource {
                        buffer,
                        kind: HostBufferKind::Keyboard,
                    },
                );
                continue;
            }

            // `--zero-buffer NAME:BYTES`: a host-provided buffer the shader
            // writes (e.g. a scatter framebuffer) with no input data. wgpu
            // zero-initializes new buffers, so no upload is needed.
            if let Some(&bytes) = zero_buffers.get(name.as_str()) {
                let buffer = device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("host_buffer_{name}")),
                    size: bytes,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                out.insert(
                    key,
                    HostBufferResource {
                        buffer,
                        kind: HostBufferKind::FileLoaded,
                    },
                );
                continue;
            }

            // No host pattern recognized; try `<storage_dir>/<name>.bin`
            // if the caller supplied a storage dir.
            let Some(dir) = storage_dir else {
                continue;
            };
            let path = dir.join(format!("{name}.bin"));
            let Ok(data) = std::fs::read(&path) else {
                continue;
            };
            let byte_size = data.len() as u64;
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("host_buffer_{name}")),
                size: byte_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, &data);
            out.insert(
                key,
                HostBufferResource {
                    buffer,
                    kind: HostBufferKind::FileLoaded,
                },
            );
        }
    }
    Ok(out)
}

/// A ping-pong storage-buffer pair backing a `--feedback` spec where
/// both sides are `Binding::StorageBuffer`. Keyed in the host pool by
/// the WRITE side's `(set, binding)`; the two `wgpu::Buffer`s alternate
/// roles each frame (parity slot = current write target, opposite
/// parity = previous-frame read source). Mirrors `StorageTextureResource`
/// but with no view objects — wgpu binds buffers directly.
pub struct FeedbackBufferResource {
    /// Always length 2. `parity % 2` is the current write slot; the
    /// opposite is the previous-frame read slot.
    pub buffers: Vec<wgpu::Buffer>,
}

/// Allocate a 2-slot `wgpu::Buffer` pair for every storage-buffer
/// feedback spec. The byte size comes from the WRITE side's
/// `Binding::StorageBuffer.length` policy in the descriptor.
///
/// Phase 1 supports the two length policies that don't need the
/// host-buffer or input-buffer byte-size pool (which isn't materialised
/// yet at this point in startup): `BufferLen::Fixed` and
/// `BufferLen::SameAsDispatch` (resolved via the owning compute
/// pipeline's `dispatch_size`). `BufferLen::LikeInput` errors with a
/// clean message — wire the necessary input-byte-size lookup if a real
/// workload needs it.
pub fn create_feedback_buffers(
    device: &wgpu::Device,
    descriptor: &PipelineDescriptor,
    buffer_feedback_pairs: &[FeedbackPair],
) -> Result<HashMap<(u32, u32), FeedbackBufferResource>> {
    let mut out: HashMap<(u32, u32), FeedbackBufferResource> = HashMap::new();
    for pair in buffer_feedback_pairs {
        let write_key = (pair.write_set, pair.write_binding);
        if out.contains_key(&write_key) {
            continue;
        }

        // Locate the compute pipeline owning the write binding so we
        // can read its `dispatch_size` (needed for `SameAsDispatch`).
        let mut write_binding_desc: Option<&Binding> = None;
        let mut owning_compute: Option<&wyn_pipeline_descriptor::ComputePipeline> = None;
        for pipeline in &descriptor.pipelines {
            let Pipeline::Compute(cp) = pipeline else {
                continue;
            };
            for b in &cp.bindings {
                if let Binding::StorageBuffer { set, binding, .. } = b {
                    if (*set, *binding) == write_key {
                        write_binding_desc = Some(b);
                        owning_compute = Some(cp);
                    }
                }
            }
        }
        let (write_b, cp) = match (write_binding_desc, owning_compute) {
            (Some(b), Some(cp)) => (b, cp),
            _ => {
                return Err(anyhow!(
                    "--feedback buffer pair write side ({}, {}) names no \
                     storage_buffer binding on any compute pipeline",
                    write_key.0,
                    write_key.1,
                ));
            }
        };
        let (name, length) = match write_b {
            Binding::StorageBuffer { name, length, .. } => (name.clone(), length.clone()),
            _ => unreachable!("filtered to StorageBuffer above"),
        };
        let length = length.ok_or_else(|| {
            anyhow!(
                "--feedback buffer pair write side '{}' ({}, {}) has no `length` policy in \
                 the descriptor; both ping-pong slots need a concrete byte size",
                name,
                write_key.0,
                write_key.1,
            )
        })?;

        let byte_size: u64 = match &length {
            wyn_pipeline_descriptor::BufferLen::Fixed { bytes } => *bytes,
            wyn_pipeline_descriptor::BufferLen::SameAsDispatch { elem_bytes } => {
                let (gx, gy, gz) = resolve_dispatch_size(&cp.dispatch_size, &HashMap::new(), &[]);
                let wg = match cp.dispatch_size {
                    DispatchSize::DerivedFrom { workgroup_size, .. } => workgroup_size,
                    DispatchSize::Fixed { .. } => 1,
                };
                let threads = (gx as u64) * (gy as u64) * (gz as u64) * (wg as u64);
                (threads * *elem_bytes as u64).max(4)
            }
            wyn_pipeline_descriptor::BufferLen::LikeInput { .. } => {
                return Err(anyhow!(
                    "--feedback buffer pair write side '{}' ({}, {}) uses `LikeInput` \
                     sizing; not supported yet — declare a `Fixed` or `SameAsDispatch` \
                     length, or wire input-byte-size lookup into create_feedback_buffers",
                    name,
                    write_key.0,
                    write_key.1,
                ));
            }
        };

        let mut buffers = Vec::with_capacity(2);
        for slot in 0..2 {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("feedback_buffer_{name}.slot{slot}")),
                size: byte_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            buffers.push(buffer);
        }
        out.insert(write_key, FeedbackBufferResource { buffers });
    }
    Ok(out)
}

/// Allocate one `wgpu::Sampler` per unique `(set, binding)` slot
/// declared as `Binding::Sampler` in any pipeline. v1 emits the
/// linear-filter / clamp-to-edge default which matches Shadertoy's
/// per-channel sampler in the absence of explicit per-channel
/// overrides; a per-binding sampler-spec extension is future work.
pub fn create_samplers(
    device: &wgpu::Device,
    descriptor: &PipelineDescriptor,
) -> HashMap<(u32, u32), wgpu::Sampler> {
    let mut out: HashMap<(u32, u32), wgpu::Sampler> = HashMap::new();
    for pipeline in &descriptor.pipelines {
        let bindings: &[Binding] = match pipeline {
            Pipeline::Compute(cp) => &cp.bindings,
            Pipeline::MultiCompute(mc) => &mc.bindings,
            Pipeline::Graphics(gp) => &gp.bindings,
        };
        for b in bindings {
            let Binding::Sampler {
                set, binding, name, ..
            } = b
            else {
                continue;
            };
            let key = (*set, *binding);
            if out.contains_key(&key) {
                continue;
            }
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(&format!("sampler_{}", name)),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            out.insert(key, sampler);
        }
    }
    out
}

/// Build a single set's bind-group layout + bind group for a pipeline,
/// covering the `Binding::StorageTexture` / `Binding::Texture` /
/// `Binding::Sampler` entries that share a wgpu::Texture or sampler
/// resource pool. Compute pipelines pass
/// `visibility = ShaderStages::COMPUTE`, fragment pipelines pass
/// `ShaderStages::FRAGMENT`, etc.
///
/// Resources are looked up from the pools by `(set, binding)`. A
/// `Binding::Texture` whose `(set, binding)` slot also has a
/// `StorageTextureResource` uses that resource's `sampled_view` —
/// this is the cross-stage compute-write / fragment-sample handoff.
/// A `Texture` slot with no storage-texture entry would need a
/// host-uploaded texture (e.g. for image inputs); that path isn't
/// wired yet and falls through with an error.
pub fn build_resource_bind_group_for_set(
    device: &wgpu::Device,
    bindings: &[Binding],
    set: u32,
    visibility: ShaderStages,
    storage_textures: &HashMap<(u32, u32), StorageTextureResource>,
    host_textures: &HashMap<(u32, u32), HostTextureResource>,
    host_buffers: &HashMap<(u32, u32), HostBufferResource>,
    feedback_buffers: &HashMap<(u32, u32), FeedbackBufferResource>,
    samplers: &HashMap<(u32, u32), wgpu::Sampler>,
    uniforms: &HashMap<(u32, u32), wgpu::Buffer>,
    feedback_reads: &HashMap<(u32, u32), (u32, u32)>,
    buffer_feedback_reads: &HashMap<(u32, u32), (u32, u32)>,
    parity: usize,
) -> Result<(wgpu::BindGroupLayout, BindGroup)> {
    let mut layout_entries: Vec<BindGroupLayoutEntry> = Vec::new();
    let mut group_entries: Vec<BindGroupEntry> = Vec::new();

    for b in bindings {
        match b {
            Binding::StorageTexture {
                set: bset,
                binding,
                format,
                access,
                ..
            } if *bset == set => {
                let key = (*bset, *binding);
                let res = storage_textures
                    .get(&key)
                    .ok_or_else(|| anyhow!("no storage texture allocated for ({}, {})", bset, binding))?;
                // Ping-pong write side: storage_view at the current
                // parity. For non-ping-pong textures, slot count is 1
                // so the index folds back to 0 either way.
                let slot = parity % res.storage_views.len();
                layout_entries.push(BindGroupLayoutEntry {
                    binding: *binding,
                    visibility,
                    ty: BindingType::StorageTexture {
                        access: access_to_storage_texture_access(access.clone()),
                        format: storage_image_format_to_wgpu(*format),
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                });
                group_entries.push(BindGroupEntry {
                    binding: *binding,
                    resource: BindingResource::TextureView(&res.storage_views[slot]),
                });
            }
            Binding::Texture {
                set: bset,
                binding,
                view_dimension,
                multisampled,
                sample_type,
                ..
            } if *bset == set => {
                let key = (*bset, *binding);
                // Resolve the sampled view: storage_texture pool first
                // (cross-stage compute-write / fragment-sample handoff),
                // then host-uploaded pool (keyboard, etc.).
                //
                // If this binding is the READ side of a feedback pair,
                // its view comes from the WRITE side's storage texture
                // at the OPPOSITE parity ("previous frame's value");
                // otherwise we use the current-parity sampled view.
                let view = if let Some(&(ws, wb)) = feedback_reads.get(&key) {
                    let res = storage_textures.get(&(ws, wb)).ok_or_else(|| {
                        anyhow!(
                            "Feedback read at ({}, {}) targets ({}, {}) but no \
                             storage texture is allocated for the write side",
                            bset,
                            binding,
                            ws,
                            wb
                        )
                    })?;
                    if res.sampled_views.len() < 2 {
                        return Err(anyhow!(
                            "Feedback read at ({}, {}) targets non-ping-pong \
                             storage texture at ({}, {}); the write side must \
                             be allocated with 2 slots",
                            bset,
                            binding,
                            ws,
                            wb
                        ));
                    }
                    &res.sampled_views[(parity + 1) % 2]
                } else {
                    storage_textures
                        .get(&key)
                        .map(|r| &r.sampled_views[parity % r.sampled_views.len()])
                        .or_else(|| host_textures.get(&key).map(|r| &r.view))
                        .ok_or_else(|| {
                            anyhow!(
                                "Texture binding ({}, {}) has no resource — neither \
                                 a compute-written storage_texture nor a \
                                 host-uploaded host_texture (e.g. `keyboard`)",
                                bset,
                                binding
                            )
                        })?
                };
                layout_entries.push(BindGroupLayoutEntry {
                    binding: *binding,
                    visibility,
                    ty: BindingType::Texture {
                        sample_type: match sample_type {
                            wyn_pipeline_descriptor::TextureSampleType::Float { filterable } => {
                                wgpu::TextureSampleType::Float {
                                    filterable: *filterable,
                                }
                            }
                            wyn_pipeline_descriptor::TextureSampleType::Sint => {
                                wgpu::TextureSampleType::Sint
                            }
                            wyn_pipeline_descriptor::TextureSampleType::Uint => {
                                wgpu::TextureSampleType::Uint
                            }
                            wyn_pipeline_descriptor::TextureSampleType::Depth => {
                                wgpu::TextureSampleType::Depth
                            }
                        },
                        view_dimension: match view_dimension {
                            wyn_pipeline_descriptor::TextureViewDimension::D1 => {
                                wgpu::TextureViewDimension::D1
                            }
                            wyn_pipeline_descriptor::TextureViewDimension::D2 => {
                                wgpu::TextureViewDimension::D2
                            }
                            wyn_pipeline_descriptor::TextureViewDimension::D2Array => {
                                wgpu::TextureViewDimension::D2Array
                            }
                            wyn_pipeline_descriptor::TextureViewDimension::Cube => {
                                wgpu::TextureViewDimension::Cube
                            }
                            wyn_pipeline_descriptor::TextureViewDimension::CubeArray => {
                                wgpu::TextureViewDimension::CubeArray
                            }
                            wyn_pipeline_descriptor::TextureViewDimension::D3 => {
                                wgpu::TextureViewDimension::D3
                            }
                        },
                        multisampled: *multisampled,
                    },
                    count: None,
                });
                group_entries.push(BindGroupEntry {
                    binding: *binding,
                    resource: BindingResource::TextureView(view),
                });
            }
            Binding::Sampler {
                set: bset,
                binding,
                binding_type,
                ..
            } if *bset == set => {
                let key = (*bset, *binding);
                let sampler = samplers
                    .get(&key)
                    .ok_or_else(|| anyhow!("no sampler allocated for ({}, {})", bset, binding))?;
                layout_entries.push(BindGroupLayoutEntry {
                    binding: *binding,
                    visibility,
                    ty: BindingType::Sampler(match binding_type {
                        wyn_pipeline_descriptor::SamplerBindingType::Filtering => {
                            wgpu::SamplerBindingType::Filtering
                        }
                        wyn_pipeline_descriptor::SamplerBindingType::NonFiltering => {
                            wgpu::SamplerBindingType::NonFiltering
                        }
                        wyn_pipeline_descriptor::SamplerBindingType::Comparison => {
                            wgpu::SamplerBindingType::Comparison
                        }
                    }),
                    count: None,
                });
                group_entries.push(BindGroupEntry {
                    binding: *binding,
                    resource: BindingResource::Sampler(sampler),
                });
            }
            Binding::Uniform {
                set: bset, binding, ..
            } if *bset == set => {
                let key = (*bset, *binding);
                let buffer = uniforms.get(&key).ok_or_else(|| {
                    anyhow!(
                        "no uniform buffer allocated for ({}, {}); caller must \
                         pre-allocate via build_pipeline_uniforms",
                        bset,
                        binding
                    )
                })?;
                layout_entries.push(BindGroupLayoutEntry {
                    binding: *binding,
                    visibility,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
                group_entries.push(BindGroupEntry {
                    binding: *binding,
                    resource: BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: None,
                    }),
                });
            }
            Binding::StorageBuffer {
                set: bset,
                binding,
                access,
                ..
            } if *bset == set => {
                let key = (*bset, *binding);
                let read_only = matches!(access, Access::ReadOnly);
                // Resolve the backing `wgpu::Buffer`:
                //   1. READ side of a buffer-feedback pair → write-side
                //      pool, opposite-parity slot ("previous frame").
                //   2. WRITE side of a buffer-feedback pair (or any other
                //      binding pointing at the WRITE-side `(set, binding)`,
                //      including a fragment shader's read-only view of
                //      the same storage) → current-parity slot.
                //   3. Otherwise → host-uploaded pool (keyboard, file).
                let buffer_ref: &wgpu::Buffer = if let Some(&(ws, wb)) = buffer_feedback_reads.get(&key) {
                    let res = feedback_buffers.get(&(ws, wb)).ok_or_else(|| {
                        anyhow!(
                            "feedback buffer pair at write ({ws}, {wb}) was not allocated; \
                             read side ({bset}, {binding}) can't bind"
                        )
                    })?;
                    &res.buffers[(parity + 1) % 2]
                } else if let Some(res) = feedback_buffers.get(&key) {
                    &res.buffers[parity % 2]
                } else {
                    let res = host_buffers.get(&key).ok_or_else(|| {
                        anyhow!(
                            "storage buffer at ({}, {}) is not in the host-uploaded \
                             pool (e.g. `keyboard`) nor any feedback pair; declare it \
                             host-uploaded or wire a `--feedback` spec for it",
                            bset,
                            binding
                        )
                    })?;
                    &res.buffer
                };
                layout_entries.push(BindGroupLayoutEntry {
                    binding: *binding,
                    visibility,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
                group_entries.push(BindGroupEntry {
                    binding: *binding,
                    resource: BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buffer_ref,
                        offset: 0,
                        size: None,
                    }),
                });
            }
            _ => {}
        }
    }

    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(&format!("resource_bgl_set{}", set)),
        entries: &layout_entries,
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some(&format!("resource_bg_set{}", set)),
        layout: &layout,
        entries: &group_entries,
    });
    Ok((layout, bind_group))
}
