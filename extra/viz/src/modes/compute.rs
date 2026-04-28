//! `compute` subcommand — run a single SPIR-V compute entry point
//! headlessly with CLI-described storage buffers, uniforms, and push
//! constants, then dump each storage buffer's contents to stdout.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use wgpu::{BufferDescriptor, BufferUsages, CommandEncoderDescriptor, PipelineLayoutDescriptor};

use crate::gpu::{
    build_per_set_bind_groups, build_push_constant_bytes, create_headless_device, prepare_storage_buffers,
};
use crate::specs::{PushConstantSpec, StorageBufferSpec, StorageElementType, UniformSpec};
use crate::spirv::{detect_storage_access, load_spirv_module};

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
    push_constants: Vec<PushConstantSpec>,
    verbose: bool,
) -> Result<()> {
    let (device, queue) = create_headless_device(verbose).await?;

    // ---- Storage buffers --------------------------------------------------
    let storage_buffers = prepare_storage_buffers(&device, &queue, storage_specs, verbose)?;

    // ---- Uniform buffers --------------------------------------------------
    let uniform_buffers: Vec<(UniformSpec, wgpu::Buffer)> = uniform_specs
        .into_iter()
        .map(|spec| {
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("uniform_buffer_set{}_b{}", spec.set, spec.binding)),
                size: spec.data.len() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, &spec.data);
            (spec, buffer)
        })
        .collect();

    // ---- SPIR-V access discovery ------------------------------------------
    // Pipeline-layout access must match the shader's `NonWritable` /
    // `NonReadable` decorations exactly. Read them from the SPIR-V so the
    // user doesn't have to spell out per-binding access in the CLI.
    let spirv_bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let spirv_words: Vec<u32> =
        spirv_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let access_by_binding = detect_storage_access(&spirv_words)?;

    // ---- Per-set bind group construction ----------------------------------
    let (layouts, bind_groups) =
        build_per_set_bind_groups(&device, &storage_buffers, &uniform_buffers, &access_by_binding);

    let module = load_spirv_module(&device, &path)?;

    // ---- Push constants ---------------------------------------------------
    // No descriptor in compute mode — assign sequential offsets at the
    // call site, then `build_push_constant_bytes` packs the bytes.
    let mut pc_specs = push_constants;
    let total_pc_size = PushConstantSpec::lay_out_sequential(&mut pc_specs);
    let pc_bytes = build_push_constant_bytes(&[], &pc_specs, verbose)?;

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
            storage_buffers.iter().map(|(s, _, _)| (s.set, s.binding)).collect::<Vec<_>>()
        );
        if !uniform_buffers.is_empty() {
            println!(
                "Uniforms: {:?}",
                uniform_buffers.iter().map(|(s, _)| (s.set, s.binding)).collect::<Vec<_>>()
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
