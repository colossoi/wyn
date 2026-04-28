//! `pipeline` subcommand — execute a pipeline described by a JSON
//! pipeline-descriptor sidecar (the same format `wyn-core` emits),
//! reading inputs from `--input name:file.json` and writing outputs
//! to `--output name:file.json` (or stdout if no output path).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use wgpu::{CommandEncoderDescriptor, PipelineLayoutDescriptor};

use crate::gpu::{
    build_bind_group, build_push_constant_bytes, create_binding_buffers, create_headless_device,
    readback_buffer, resolve_dispatch_size,
};
use crate::json::{Binding, BufferUsage, ComputePipeline, MultiComputePipeline, Pipeline, PipelineDescriptor, write_f32_json};
use crate::spirv::load_spirv_module;

pub async fn run_pipeline(
    spv_path: PathBuf,
    pipeline_path: PathBuf,
    inputs: HashMap<String, PathBuf>,
    outputs: HashMap<String, PathBuf>,
    push_constants: &[String],
    verbose: bool,
) -> Result<()> {
    let desc_json = fs::read_to_string(&pipeline_path)
        .with_context(|| format!("Failed to read pipeline descriptor: {}", pipeline_path.display()))?;
    let desc: PipelineDescriptor =
        serde_json::from_str(&desc_json).with_context(|| "Failed to parse pipeline descriptor JSON")?;

    if desc.pipelines.is_empty() {
        return Err(anyhow!("Pipeline descriptor has no pipelines"));
    }

    let (device, queue) = create_headless_device(verbose).await?;
    let module = load_spirv_module(&device, &spv_path)?;

    for (pi, pipeline) in desc.pipelines.iter().enumerate() {
        match pipeline {
            Pipeline::Compute(cp) => {
                run_single_compute(
                    &device,
                    &queue,
                    &module,
                    cp,
                    &inputs,
                    &outputs,
                    push_constants,
                    verbose,
                )
                .with_context(|| format!("Pipeline {} (compute) failed", pi))?;
            }
            Pipeline::MultiCompute(mp) => {
                run_multi_compute(
                    &device,
                    &queue,
                    &module,
                    mp,
                    &inputs,
                    &outputs,
                    push_constants,
                    verbose,
                )
                .with_context(|| format!("Pipeline {} (multi_compute) failed", pi))?;
            }
            Pipeline::Graphics(_) => {
                eprintln!(
                    "Pipeline {} is a graphics pipeline (not yet supported by `run`)",
                    pi
                );
            }
        }
    }

    Ok(())
}

/// Create wgpu buffers for a set of bindings. Returns a map from binding number

fn run_single_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    module: &wgpu::ShaderModule,
    cp: &ComputePipeline,
    inputs: &HashMap<String, PathBuf>,
    outputs: &HashMap<String, PathBuf>,
    push_constants: &[String],
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Running compute pipeline: {}", cp.entry_point);
    }

    let buffers = create_binding_buffers(device, queue, &cp.bindings, inputs, verbose)?;
    let (layout, bind_group) = build_bind_group(device, &cp.bindings, &buffers)?;

    // Build push constant data from CLI args matched against descriptor bindings
    let pc_bytes = build_push_constant_bytes(&cp.bindings, push_constants, verbose)?;
    let total_pc_size = pc_bytes.len() as u32;

    let pc_ranges: Vec<wgpu::PushConstantRange> = if total_pc_size > 0 {
        vec![wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..total_pc_size,
        }]
    } else {
        vec![]
    };

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("compute_layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &pc_ranges,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(&cp.entry_point),
        compilation_options: Default::default(),
        cache: None,
    });

    let dispatch = resolve_dispatch_size(&cp.dispatch_size, &buffers, &cp.bindings, None);
    if verbose {
        println!("Dispatch: {} x {} x {}", dispatch.0, dispatch.1, dispatch.2);
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
        cpass.set_bind_group(0, &bind_group, &[]);
        if !pc_bytes.is_empty() {
            cpass.set_push_constants(0, &pc_bytes);
        }
        cpass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
    }
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    // Read back and output results
    output_results(device, queue, &cp.bindings, &buffers, outputs)?;

    Ok(())
}

/// Run a multi-dispatch compute pipeline (e.g. reduce with phase1 + phase2).
fn run_multi_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    module: &wgpu::ShaderModule,
    mp: &MultiComputePipeline,
    inputs: &HashMap<String, PathBuf>,
    outputs: &HashMap<String, PathBuf>,
    push_constants: &[String],
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Running multi-compute pipeline ({} stages)", mp.stages.len());
        for (i, stage) in mp.stages.iter().enumerate() {
            println!(
                "  Stage {}: {} (reads {:?}, writes {:?})",
                i, stage.entry_point, stage.reads, stage.writes
            );
        }
    }

    let buffers = create_binding_buffers(device, queue, &mp.bindings, inputs, verbose)?;
    let (layout, bind_group) = build_bind_group(device, &mp.bindings, &buffers)?;

    // Build push constant data from CLI args matched against descriptor bindings
    let pc_bytes = build_push_constant_bytes(&mp.bindings, push_constants, verbose)?;
    let total_pc_size = pc_bytes.len() as u32;

    let pc_ranges: Vec<wgpu::PushConstantRange> = if total_pc_size > 0 {
        vec![wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..total_pc_size,
        }]
    } else {
        vec![]
    };

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("multi_compute_layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &pc_ranges,
    });

    // Execute stages in order
    for (si, stage) in mp.stages.iter().enumerate() {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("stage_{}", stage.entry_point)),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(&stage.entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        let dispatch =
            resolve_dispatch_size(&stage.dispatch_size, &buffers, &mp.bindings, Some(&stage.reads));
        if verbose {
            println!(
                "Stage {} ({}): dispatch {} x {} x {}",
                si, stage.entry_point, dispatch.0, dispatch.1, dispatch.2
            );
        }

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!("stage_{}_encoder", si)),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("stage_{}", si)),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            if !pc_bytes.is_empty() {
                cpass.set_push_constants(0, &pc_bytes);
            }
            cpass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
        }
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::Wait);
    }

    // Read back and output results
    output_results(device, queue, &mp.bindings, &buffers, outputs)?;

    Ok(())
}

/// Print f32 data to stdout — fallback for outputs without an
/// explicit `--output name:file.json` redirection.
fn print_f32_data(name: &str, data: &[f32]) {
    println!("\n=== {} ({} elements) ===", name, data.len());
    let show = data.len().min(64);
    for (i, chunk) in data[..show].chunks(8).enumerate() {
        print!("  [{:3}]: ", i * 8);
        for val in chunk {
            print!("{:8.3} ", val);
        }
        println!();
    }
    if data.len() > show {
        println!("  ... ({} more elements)", data.len() - show);
    }
    println!();
}

/// Read back output/intermediate buffers and write/print results.
fn output_results(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bindings: &[Binding],
    buffers: &HashMap<u32, (wgpu::Buffer, u64)>,
    outputs: &HashMap<String, PathBuf>,
) -> Result<()> {
    for b in bindings {
        if let Binding::StorageBuffer {
            binding, name, usage, ..
        } = b
        {
            // Only read back output and intermediate buffers (skip inputs unless
            // explicitly requested via --output)
            let should_output =
                *usage != BufferUsage::Input || outputs.contains_key(name.as_str());

            if !should_output {
                continue;
            }

            if let Some((buf, size)) = buffers.get(binding) {
                let data = readback_buffer(device, queue, buf, *size)?;

                if let Some(path) = outputs.get(name.as_str()) {
                    write_f32_json(path, &data)?;
                    println!("Wrote {} elements to {}", data.len(), path.display());
                } else {
                    print_f32_data(name, &data);
                }
            }
        }
    }
    Ok(())
}
