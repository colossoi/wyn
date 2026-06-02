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
use wyn_pipeline_descriptor::StorageTextureSize;
use crate::specs::{PushConstantSpec, StorageBufferSpec, UniformSpec};
use crate::spirv::SpirvAccess;

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
                        let size = res.texture.size();
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

/// Allocate the storage + staging buffer pair for each `StorageBufferSpec`,
/// upload its initial data (zeros, or the JSON file the spec points at),
/// and return tuples for the bind-group + readback steps.
pub fn prepare_storage_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    specs: Vec<StorageBufferSpec>,
    verbose: bool,
) -> Result<Vec<(StorageBufferSpec, wgpu::Buffer, wgpu::Buffer)>> {
    let mut out = Vec::with_capacity(specs.len());
    for spec in specs {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("storage_buffer_set{}_b{}", spec.set, spec.binding)),
            size: spec.byte_size(),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_data = spec.load_initial_data()?;
        if verbose && spec.input_file.is_some() {
            println!(
                "Loaded {} bytes for set {} binding {} from {:?}",
                init_data.len(),
                spec.set,
                spec.binding,
                spec.input_file
            );
        }
        queue.write_buffer(&buffer, 0, &init_data);

        let staging = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("staging_buffer_set{}_b{}", spec.set, spec.binding)),
            size: spec.byte_size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        out.push((spec, buffer, staging));
    }
    Ok(out)
}

/// Build one BindGroupLayout + BindGroup per descriptor set spanning
/// `storage_buffers` and `uniform_buffers`, indexed by set number.
/// wgpu requires set indices in the pipeline layout to be contiguous
/// from 0, so any unused intermediate sets get an empty layout/group.
///
/// Storage-buffer access mode is read from `access_by_binding`
/// (extracted from the SPIR-V's `NonWritable` / `NonReadable`
/// decorations); the pipeline layout must match exactly.
pub fn build_per_set_bind_groups(
    device: &wgpu::Device,
    storage_buffers: &[(StorageBufferSpec, wgpu::Buffer, wgpu::Buffer)],
    uniform_buffers: &[(UniformSpec, wgpu::Buffer)],
    access_by_binding: &HashMap<(u32, u32), SpirvAccess>,
) -> (Vec<wgpu::BindGroupLayout>, Vec<BindGroup>) {
    let max_set = storage_buffers
        .iter()
        .map(|(s, _, _)| s.set)
        .chain(uniform_buffers.iter().map(|(u, _)| u.set))
        .max()
        .unwrap_or(0);

    let mut layouts: Vec<wgpu::BindGroupLayout> = Vec::with_capacity((max_set + 1) as usize);
    let mut bind_groups: Vec<BindGroup> = Vec::with_capacity((max_set + 1) as usize);
    for set in 0..=max_set {
        let mut layout_entries: Vec<BindGroupLayoutEntry> = Vec::new();
        let mut group_entries: Vec<BindGroupEntry> = Vec::new();

        for (spec, buf, _) in storage_buffers {
            if spec.set != set {
                continue;
            }
            // Default to read-write if the shader has no opinion. Read-only
            // when `NonWritable` is set; `WriteOnly` falls back to
            // read_only=false because wgpu's `Storage { read_only }` has
            // no separate write-only form.
            let read_only = matches!(
                access_by_binding.get(&(spec.set, spec.binding)),
                Some(SpirvAccess::ReadOnly)
            );
            layout_entries.push(BindGroupLayoutEntry {
                binding: spec.binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            group_entries.push(BindGroupEntry {
                binding: spec.binding,
                resource: BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: 0,
                    size: None,
                }),
            });
        }
        for (spec, buf) in uniform_buffers {
            if spec.set != set {
                continue;
            }
            layout_entries.push(BindGroupLayoutEntry {
                binding: spec.binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            group_entries.push(BindGroupEntry {
                binding: spec.binding,
                resource: BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: 0,
                    size: None,
                }),
            });
        }
        layout_entries.sort_by_key(|e| e.binding);
        group_entries.sort_by_key(|e| e.binding);

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(&format!("compute_bgl_set{}", set)),
            entries: &layout_entries,
        });
        let group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(&format!("compute_bg_set{}", set)),
            layout: &layout,
            entries: &group_entries,
        });
        layouts.push(layout);
        bind_groups.push(group);
    }
    (layouts, bind_groups)
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
    pub texture: wgpu::Texture,
    pub storage_view: wgpu::TextureView,
    pub sampled_view: wgpu::TextureView,
    pub format: wgpu::TextureFormat,
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
) -> HashMap<(u32, u32), StorageTextureResource> {
    let mut out: HashMap<(u32, u32), StorageTextureResource> = HashMap::new();
    for pipeline in &descriptor.pipelines {
        let bindings: &[Binding] = match pipeline {
            Pipeline::Compute(cp) => &cp.bindings,
            Pipeline::MultiCompute(mc) => &mc.bindings,
            Pipeline::Graphics(gp) => &gp.bindings,
        };
        for b in bindings {
            let Binding::StorageTexture {
                set, binding, name, format, size: size_policy, ..
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
                StorageTextureSize::SameAsWindow => surface_size
                    .unwrap_or((STORAGE_TEXTURE_FALLBACK_SIZE, STORAGE_TEXTURE_FALLBACK_SIZE)),
            };
            let size = wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("storage_texture_{}", name)),
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
                label: Some(&format!("storage_view_{}", name)),
                format: Some(wgpu_format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });
            let sampled_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("sampled_view_{}", name)),
                format: Some(wgpu_format),
                dimension: Some(wgpu::TextureViewDimension::D2),
                ..Default::default()
            });
            out.insert(
                key,
                StorageTextureResource {
                    texture,
                    storage_view,
                    sampled_view,
                    format: wgpu_format,
                },
            );
        }
    }
    out
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
            let Binding::Sampler { set, binding, name, .. } = b else {
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
    samplers: &HashMap<(u32, u32), wgpu::Sampler>,
    uniforms: &HashMap<(u32, u32), wgpu::Buffer>,
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
                let res = storage_textures.get(&key).ok_or_else(|| {
                    anyhow!("no storage texture allocated for ({}, {})", bset, binding)
                })?;
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
                    resource: BindingResource::TextureView(&res.storage_view),
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
                let view = storage_textures
                    .get(&key)
                    .map(|r| &r.sampled_view)
                    .ok_or_else(|| {
                        anyhow!(
                            "Texture binding ({}, {}) has no storage_texture entry to share; \
                             host-uploaded textures not yet wired",
                            bset,
                            binding
                        )
                    })?;
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
                let sampler = samplers.get(&key).ok_or_else(|| {
                    anyhow!("no sampler allocated for ({}, {})", bset, binding)
                })?;
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
                set: bset,
                binding,
                ..
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
