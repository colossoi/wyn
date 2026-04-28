//! `compute` subcommand — run a single SPIR-V compute entry point
//! headlessly with CLI-described storage buffers, uniforms, and push
//! constants, then dump each storage buffer's contents to stdout.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    PipelineLayoutDescriptor, ShaderStages,
};

use crate::gpu::create_headless_device;
use crate::specs::{PushConstantSpec, StorageBufferSpec, StorageElementType, UniformSpec};
use crate::spirv::{SpirvAccess, detect_storage_access, load_spirv_module};

fn print_storage_buffer(spec: &StorageBufferSpec, data: &[u8]) {
    println!(
        "\n=== Storage Buffer (set {}, binding {}, {} elements) ===",
        spec.set, spec.binding, spec.size_elements
    );

    let u32_data: &[u32] = bytemuck::cast_slice(data);
    let elements_to_show = (spec.size_elements as usize).min(64); // Show at most 64 elements

    match spec.element_type {
        StorageElementType::I32 => {
            let i32_data: Vec<i32> = u32_data.iter().map(|&x| x as i32).collect();
            for (i, chunk) in i32_data[..elements_to_show].chunks(8).enumerate() {
                print!("  [{:3}]: ", i * 8);
                for val in chunk {
                    print!("{:8} ", val);
                }
                println!();
            }
        }
        StorageElementType::U32 => {
            for (i, chunk) in u32_data[..elements_to_show].chunks(8).enumerate() {
                print!("  [{:3}]: ", i * 8);
                for val in chunk {
                    print!("{:08x} ", val);
                }
                println!();
            }
        }
        StorageElementType::F32 => {
            let f32_data: Vec<f32> = u32_data.iter().map(|&x| f32::from_bits(x)).collect();
            for (i, chunk) in f32_data[..elements_to_show].chunks(8).enumerate() {
                print!("  [{:3}]: ", i * 8);
                for val in chunk {
                    print!("{:8.3} ", val);
                }
                println!();
            }
        }
    }

    if spec.size_elements as usize > elements_to_show {
        println!(
            "  ... ({} more elements)",
            spec.size_elements as usize - elements_to_show
        );
    }
    println!();
}

pub async fn run_compute_shader(
    path: PathBuf,
    entry: String,
    workgroups: (u32, u32, u32),
    storage_specs: Vec<StorageBufferSpec>,
    uniform_specs: Vec<UniformSpec>,
    push_constants: &[String],
    verbose: bool,
) -> Result<()> {
    let (device, queue) = create_headless_device(verbose).await?;

    // ---- Storage buffers --------------------------------------------------
    // Per spec we keep (spec, gpu_buffer, staging_buffer); staging is used
    // for the readback after dispatch.
    let mut storage_buffers: Vec<(StorageBufferSpec, wgpu::Buffer, wgpu::Buffer)> = Vec::new();
    for spec in &storage_specs {
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

        storage_buffers.push((spec.clone(), buffer, staging));
    }

    // ---- Uniform buffers --------------------------------------------------
    let mut uniform_buffers: Vec<(UniformSpec, wgpu::Buffer)> = Vec::new();
    for spec in &uniform_specs {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("uniform_buffer_set{}_b{}", spec.set, spec.binding)),
            size: spec.data.len() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &spec.data);
        uniform_buffers.push((spec.clone(), buffer));
    }

    // ---- SPIR-V access discovery ------------------------------------------
    // Pipeline-layout access must match the shader's `NonWritable` /
    // `NonReadable` decorations exactly. Read them from the SPIR-V so the
    // user doesn't have to spell out per-binding access in the CLI.
    let spirv_bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let spirv_words: Vec<u32> =
        spirv_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let access_by_binding = detect_storage_access(&spirv_words)?;

    // ---- Per-set bind group construction ----------------------------------
    // Group bindings by descriptor set, then build one BindGroupLayout +
    // BindGroup per set. wgpu requires the pipeline layout to cover every
    // set index from 0 to max contiguously, so any unused intermediate
    // sets get an empty layout / group.
    let max_set = storage_buffers
        .iter()
        .map(|(s, _, _)| s.set)
        .chain(uniform_buffers.iter().map(|(u, _)| u.set))
        .max()
        .unwrap_or(0);

    let mut layouts: Vec<wgpu::BindGroupLayout> = Vec::with_capacity((max_set + 1) as usize);
    let mut bind_groups: Vec<wgpu::BindGroup> = Vec::with_capacity((max_set + 1) as usize);
    for set in 0..=max_set {
        let mut layout_entries: Vec<BindGroupLayoutEntry> = Vec::new();
        let mut group_entries: Vec<BindGroupEntry> = Vec::new();

        for (spec, buf, _) in &storage_buffers {
            if spec.set != set {
                continue;
            }
            // Default to read-write if the shader has no opinion (e.g.
            // a buffer the SPIR-V doesn't decorate). Read-only when
            // `NonWritable` is present, write-only on `NonReadable`.
            // wgpu's `BufferBindingType::Storage { read_only }` covers
            // read-only and read-write; there's no separate write-only
            // form, so `WriteOnly` falls back to `read_only: false`.
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
        for (spec, buf) in &uniform_buffers {
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

    let module = load_spirv_module(&device, &path)?;

    // ---- Push constants ---------------------------------------------------
    let mut pc_specs: Vec<PushConstantSpec> =
        push_constants.iter().map(|s| PushConstantSpec::parse(s)).collect::<Result<Vec<_>>>()?;

    let mut offset = 0u32;
    for spec in &mut pc_specs {
        spec.offset = offset;
        offset += spec.byte_size();
    }
    let total_pc_size = offset;

    let mut pc_bytes = vec![0u8; total_pc_size as usize];
    for spec in &pc_specs {
        let start = spec.offset as usize;
        let end = start + spec.data.len();
        pc_bytes[start..end].copy_from_slice(&spec.data);
    }

    if verbose && !pc_specs.is_empty() {
        println!("Push constants ({} bytes):", total_pc_size);
        for spec in &pc_specs {
            println!(
                "  {} @ offset {}: {} bytes",
                spec.name,
                spec.offset,
                spec.byte_size()
            );
        }
    }

    let pc_ranges: Vec<wgpu::PushConstantRange> = if total_pc_size > 0 {
        vec![wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..total_pc_size,
        }]
    } else {
        vec![]
    };

    let layout_refs: Vec<&wgpu::BindGroupLayout> = layouts.iter().collect();
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("compute_layout"),
        bind_group_layouts: &layout_refs,
        push_constant_ranges: &pc_ranges,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(&entry),
        compilation_options: Default::default(),
        cache: None,
    });

    if verbose {
        println!(
            "Dispatching {} x {} x {} workgroups...",
            workgroups.0, workgroups.1, workgroups.2
        );
        println!(
            "Storage buffers: {:?}",
            storage_specs.iter().map(|s| (s.set, s.binding)).collect::<Vec<_>>()
        );
        if !uniform_specs.is_empty() {
            println!(
                "Uniforms: {:?}",
                uniform_specs.iter().map(|s| (s.set, s.binding)).collect::<Vec<_>>()
            );
        }
    }

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("compute_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        for (set_idx, group) in bind_groups.iter().enumerate() {
            cpass.set_bind_group(set_idx as u32, group, &[]);
        }
        if !pc_bytes.is_empty() {
            cpass.set_push_constants(0, &pc_bytes);
        }
        cpass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    // Copy storage buffers to staging for readback
    for (spec, buffer, staging) in &storage_buffers {
        encoder.copy_buffer_to_buffer(buffer, 0, staging, 0, spec.byte_size());
    }

    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    // Read back storage buffers
    for (spec, _, staging) in &storage_buffers {
        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait);

        if rx.recv().unwrap().is_ok() {
            let data = buffer_slice.get_mapped_range();
            print_storage_buffer(spec, &data);
            drop(data);
            staging.unmap();
        }
    }

    Ok(())
}

// --- App state ---------------------------------------------------------------
