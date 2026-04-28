//! Headless GPU helpers shared across compute / run / miner / validate /
//! info modes: device acquisition, GPU timestamp profiling, generic
//! buffer & bind-group construction over `pipeline_desc::Binding`,
//! readback, push-constant assembly, and the `--shadertoy`-style
//! sidecar uniform declarations.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, DeviceDescriptor, Instance, InstanceDescriptor, PowerPreference,
    RequestAdapterOptions, ShaderStages, Trace,
};

use crate::json::{load_f32_json, pipeline_desc};
use crate::specs::PushConstantSpec;

pub async fn create_headless_device(verbose: bool) -> Result<(wgpu::Device, wgpu::Queue)> {
    let instance = Instance::new(&InstanceDescriptor::default());

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .context("request_adapter failed")?;

    let adapter_features = adapter.features();
    let spirv_passthrough_supported = adapter_features.contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH);

    if verbose {
        let info = adapter.get_info();
        println!("GPU: {} ({:?})", info.name, info.backend);
        println!("Driver: {} {}", info.driver, info.driver_info);
        println!(
            "SPIRV_SHADER_PASSTHROUGH supported: {}",
            spirv_passthrough_supported
        );
    }

    let mut required_features = wgpu::Features::empty();
    if spirv_passthrough_supported {
        required_features |= wgpu::Features::SPIRV_SHADER_PASSTHROUGH;
    }
    if adapter_features.contains(wgpu::Features::PUSH_CONSTANTS) {
        required_features |= wgpu::Features::PUSH_CONSTANTS;
    }
    if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required_features |= wgpu::Features::TIMESTAMP_QUERY;
    }

    let mut limits = wgpu::Limits::default();
    limits.max_push_constant_size = adapter.limits().max_push_constant_size;

    adapter
        .request_device(&DeviceDescriptor {
            label: None,
            required_features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Trace::Off,
        })
        .await
        .context("failed to create logical device")
}

/// GPU timestamp profiler for compute passes.
/// Wraps a wgpu QuerySet + resolve/read buffers. Each "slot" records
/// begin/end timestamps for one compute pass (2 queries per slot).
pub struct GpuTimestamps {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    ns_per_tick: f64,
    num_slots: u32,        // total slots allocated
    queries_per_slot: u32, // 2 queries per slot (begin + end)
}

impl GpuTimestamps {
    /// Create a new profiler with `num_slots` slots. Returns None if the device
    /// doesn't support TIMESTAMP_QUERY.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, num_slots: u32) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let queries_per_slot = 2;
        let total_queries = num_slots * queries_per_slot;
        let byte_size = total_queries as u64 * 8;
        Some(Self {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("gpu_timestamps"),
                count: total_queries,
                ty: wgpu::QueryType::Timestamp,
            }),
            resolve_buf: device.create_buffer(&BufferDescriptor {
                label: Some("timestamp_resolve"),
                size: byte_size,
                usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            read_buf: device.create_buffer(&BufferDescriptor {
                label: Some("timestamp_read"),
                size: byte_size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            ns_per_tick: queue.get_timestamp_period() as f64,
            num_slots,
            queries_per_slot,
        })
    }

    /// Returns the `ComputePassTimestampWrites` for the given slot index.
    pub fn writes_for(&self, slot: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        let base = slot * self.queries_per_slot;
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(base),
            end_of_pass_write_index: Some(base + 1),
        }
    }

    /// Resolve all queries and read back the timestamps. Returns a Vec of
    /// (begin_ns, end_ns) per slot, stopping at the first unused slot.
    pub async fn read_back(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<(f64, f64)>> {
        let total = self.num_slots * self.queries_per_slot;
        let byte_size = total as u64 * 8;
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("timestamp_resolve"),
        });
        encoder.resolve_query_set(&self.query_set, 0..total, &self.resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&self.resolve_buf, 0, &self.read_buf, 0, byte_size);
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::Wait);

        let slice = self.read_buf.slice(..byte_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap()?;

        let data = slice.get_mapped_range();
        let raw: &[u64] = bytemuck::cast_slice(&data);

        let mut results = Vec::new();
        for i in 0..self.num_slots as usize {
            let begin = raw[i * 2];
            let end = raw[i * 2 + 1];
            if begin == 0 && end == 0 {
                break;
            }
            results.push((begin as f64 * self.ns_per_tick, end as f64 * self.ns_per_tick));
        }
        drop(data);
        self.read_buf.unmap();
        Ok(results)
    }
}


pub fn create_binding_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bindings: &[pipeline_desc::Binding],
    inputs: &HashMap<String, PathBuf>,
    verbose: bool,
) -> Result<HashMap<u32, (wgpu::Buffer, u64)>> {
    let mut buffers = HashMap::new();

    for b in bindings {
        if let pipeline_desc::Binding::StorageBuffer {
            binding, name, usage, ..
        } = b
        {
            let (data_bytes, element_count) = if let Some(path) = inputs.get(name.as_str()) {
                let data = load_f32_json(path)?;
                let count = data.len();
                let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                if verbose {
                    println!("Loaded {} elements for '{}' from {}", count, name, path.display());
                }
                (bytes, count)
            } else if *usage == pipeline_desc::BufferUsage::Input {
                return Err(anyhow!(
                    "No input file provided for '{}'. Use --input {}:<file.json>",
                    name,
                    name
                ));
            } else {
                // Intermediate/output buffers: allocate a reasonable size.
                // For intermediate buffers in multi-compute, the runtime needs to
                // allocate based on the pipeline's needs. Use 1024 elements as default.
                let count = 1024usize;
                if verbose {
                    println!("Allocated {} zero elements for '{}'", count, name);
                }
                (vec![0u8; count * 4], count)
            };

            let byte_size = data_bytes.len() as u64;
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(name),
                size: byte_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, &data_bytes);

            buffers.insert(*binding, (buffer, byte_size));
            let _ = element_count; // used in verbose output above
        }
    }

    Ok(buffers)
}

/// Build a bind group layout + bind group from binding descriptors.
pub fn build_bind_group(
    device: &wgpu::Device,
    bindings: &[pipeline_desc::Binding],
    buffers: &HashMap<u32, (wgpu::Buffer, u64)>,
) -> Result<(wgpu::BindGroupLayout, BindGroup)> {
    let mut layout_entries = Vec::new();
    let mut group_entries = Vec::new();

    for b in bindings {
        if let pipeline_desc::Binding::StorageBuffer { binding, access, .. } = b {
            let read_only = *access == pipeline_desc::Access::ReadOnly;
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
    dispatch: &pipeline_desc::DispatchSize,
    buffers: &HashMap<u32, (wgpu::Buffer, u64)>,
    bindings: &[pipeline_desc::Binding],
    reads: Option<&[usize]>,
) -> (u32, u32, u32) {
    match dispatch {
        pipeline_desc::DispatchSize::Fixed { x, y, z } => (*x, *y, *z),
        pipeline_desc::DispatchSize::DerivedFromInputLength { workgroup_size } => {
            // Find the first input binding to derive length from
            let input_binding = reads
                .and_then(|r| r.first())
                .and_then(|&idx| bindings.get(idx))
                .or_else(|| bindings.iter().find(|b| b.is_input()));

            let elements = input_binding
                .map(|b| {
                    let binding_num = b.wgpu_binding();
                    buffers.get(&binding_num).map(|(_, size)| (*size / 4) as u32).unwrap_or(0)
                })
                .unwrap_or(0);

            let wg = *workgroup_size;
            let groups = (elements + wg - 1) / wg;
            (groups, 1, 1)
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
    bindings: &[pipeline_desc::Binding],
    push_constants: &[String],
    verbose: bool,
) -> Result<Vec<u8>> {
    // Collect PushConstant bindings from the descriptor
    let pc_bindings: Vec<_> = bindings
        .iter()
        .filter_map(|b| {
            if let pipeline_desc::Binding::PushConstant { offset, size, name } = b {
                Some((name.as_str(), *offset, *size))
            } else {
                None
            }
        })
        .collect();

    if pc_bindings.is_empty() && push_constants.is_empty() {
        return Ok(vec![]);
    }

    // If we have CLI push constants but no descriptor bindings, use sequential layout
    if pc_bindings.is_empty() && !push_constants.is_empty() {
        let mut pc_specs: Vec<PushConstantSpec> =
            push_constants.iter().map(|s| PushConstantSpec::parse(s)).collect::<Result<Vec<_>>>()?;

        let mut offset = 0u32;
        for spec in &mut pc_specs {
            spec.offset = offset;
            offset += spec.byte_size();
        }
        let total = offset as usize;
        let mut bytes = vec![0u8; total];
        for spec in &pc_specs {
            let start = spec.offset as usize;
            let end = start + spec.data.len();
            bytes[start..end].copy_from_slice(&spec.data);
        }
        if verbose {
            println!("Push constants ({} bytes, sequential layout):", total);
            for spec in &pc_specs {
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

    // Parse CLI push constants into a map by name
    let cli_specs: HashMap<String, PushConstantSpec> = push_constants
        .iter()
        .map(|s| {
            let spec = PushConstantSpec::parse(s)?;
            Ok((spec.name.clone(), spec))
        })
        .collect::<Result<HashMap<_, _>>>()?;

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


