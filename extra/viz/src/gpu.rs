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

use crate::json::{Access, Binding, BufferUsage, DispatchLen, DispatchSize, load_f32_json};
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
            set, binding, name, usage, length, ..
        } = b
        {
            if length.is_some() {
                continue;
            }
            let data_bytes = if let Some(path) = inputs.get(name.as_str()) {
                let data = load_f32_json(path)?;
                if verbose {
                    println!("Loaded {} elements for '{}' from {}", data.len(), name, path.display());
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
            binding, name, length: Some(len), ..
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
                    anyhow!("cannot size dispatch-length buffer '{}': no dispatch in scope", name)
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
                        anyhow!("cannot size buffer '{}': its source buffer was not allocated", name)
                    })?
                    .max(4)
            };
            if verbose {
                println!("Allocated {} bytes for compiler-managed buffer '{}'", byte_size, name);
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
pub fn resolve_dispatch_size(
    dispatch: &DispatchSize,
    buffers: &HashMap<u32, (wgpu::Buffer, u64)>,
    pc_bytes: &[u8],
) -> (u32, u32, u32) {
    match dispatch {
        DispatchSize::Fixed { x, y, z } => (*x, *y, *z),
        DispatchSize::DerivedFrom { len, workgroup_size } => {
            let elements = resolve_dispatch_len(len, buffers, pc_bytes);
            let wg = *workgroup_size;
            ((elements + wg - 1) / wg, 1, 1)
        }
    }
}

/// Resolve a `DerivedFrom` dispatch's iteration count from its `DispatchLen`
/// source: a buffer's element count, a compile-time constant, or a scalar read
/// from the push-constant block.
fn resolve_dispatch_len(
    len: &DispatchLen,
    buffers: &HashMap<u32, (wgpu::Buffer, u64)>,
    pc_bytes: &[u8],
) -> u32 {
    match len {
        DispatchLen::InputBinding { binding, elem_bytes, .. } => buffers
            .get(binding)
            .map(|(_, size)| (*size / *elem_bytes as u64) as u32)
            .unwrap_or(0),
        DispatchLen::Fixed { count } => *count,
        DispatchLen::PushConstant { offset } => {
            let o = *offset as usize;
            pc_bytes
                .get(o..o + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .unwrap_or(0)
        }
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
