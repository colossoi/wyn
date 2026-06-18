//! `pipeline` subcommand — execute a pipeline described by a JSON
//! pipeline-descriptor sidecar (the same format `wyn-core` emits),
//! reading inputs from `--input name:file.json` and writing outputs
//! to `--output name:file.json` (or stdout if no output path).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use wgpu::{CommandEncoderDescriptor, PipelineLayoutDescriptor};
use winit::event_loop::EventLoop;

use crate::app::{App, InteractivePipelineSpec};
use crate::gpu::{
    ComputeExecutor, build_bind_group, build_push_constant_bytes, create_binding_buffers,
    create_headless_device, readback_buffer, resolve_dispatch_size,
};
use crate::json::{
    Binding, BufferUsage, ComputePipeline, MultiComputePipeline, Pipeline, PipelineDescriptor,
    write_f32_json,
};
use crate::specs::PushConstantSpec;
use crate::spirv::load_spirv_module;
use wyn_pipeline_descriptor::ShaderStage;

/// Knobs that only mean anything on the interactive path. Bundled so
/// `run_pipeline`'s signature doesn't grow N positional args every
/// time the interactive mode learns another flag.
pub struct InteractiveOpts {
    pub storage_dir: Option<PathBuf>,
    /// Host storage buffers to allocate and seed once, by binding name →
    /// init spec (from `--buffer-init NAME:BYTES:SPEC`).
    pub buffer_inits: HashMap<String, crate::gpu::BufferInit>,
    pub index_buffer: Option<PathBuf>,
    pub present_mode: wgpu::PresentMode,
    pub validate: bool,
    pub size: Option<(u32, u32)>,
    pub max_frames: Option<u32>,
    pub vertex_count: u32,
    pub topology: wgpu::PrimitiveTopology,
}

pub async fn run_pipeline(
    spv_path: PathBuf,
    pipeline_path: PathBuf,
    inputs: HashMap<String, PathBuf>,
    outputs: HashMap<String, PathBuf>,
    push_constants: &[PushConstantSpec],
    dispatch_overrides: &HashMap<String, (u32, u32, u32)>,
    feedback_specs: &[(String, String, String)],
    interactive_opts: InteractiveOpts,
    verbose: bool,
) -> Result<()> {
    let desc_json = fs::read_to_string(&pipeline_path)
        .with_context(|| format!("Failed to read pipeline descriptor: {}", pipeline_path.display()))?;
    let desc: PipelineDescriptor =
        serde_json::from_str(&desc_json).with_context(|| "Failed to parse pipeline descriptor JSON")?;

    if desc.pipelines.is_empty() {
        return Err(anyhow!("Pipeline descriptor has no pipelines"));
    }

    // Auto-detect interactive vs headless. A descriptor with a
    // `Graphics` pipeline asks for a window: switch to the interactive
    // path. Otherwise stick with the original headless compute runner.
    let has_graphics = desc.pipelines.iter().any(|p| matches!(p, Pipeline::Graphics(_)));
    if has_graphics {
        if !inputs.is_empty() {
            eprintln!(
                "[viz pipeline] --input is ignored in interactive mode \
                 (descriptor has a graphics pipeline; inputs come from --feedback / \
                 --buffer-init / --storage-dir)"
            );
        }
        if !outputs.is_empty() && interactive_opts.max_frames.is_none() {
            eprintln!(
                "[viz pipeline] --output in interactive mode dumps buffers at the \
                 --max-frames exit; without --max-frames the run never reaches the \
                 dump. Pass --max-frames N to snapshot after N frames."
            );
        }
        return run_pipeline_interactive(
            spv_path,
            desc,
            dispatch_overrides.clone(),
            feedback_specs.to_vec(),
            outputs,
            interactive_opts,
            verbose,
        );
    }
    if !feedback_specs.is_empty() {
        eprintln!(
            "[viz pipeline] --feedback is ignored in headless mode \
             (no frames, no previous-state notion)"
        );
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
                // Unreachable now (caught above), kept for completeness
                // if a future descriptor shape allows mixed headless +
                // graphics dispatches.
            }
        }
    }

    Ok(())
}

/// Interactive path. Opens a window, runs every compute pipeline in
/// the descriptor each frame, then renders the one graphics pipeline.
/// Activated automatically when the descriptor contains a graphics
/// pipeline (the headless `--input` / `--output` flags are ignored).
fn run_pipeline_interactive(
    spv_path: PathBuf,
    desc: PipelineDescriptor,
    dispatch_overrides: HashMap<String, (u32, u32, u32)>,
    feedback_specs: Vec<(String, String, String)>,
    outputs: HashMap<String, PathBuf>,
    opts: InteractiveOpts,
    verbose: bool,
) -> Result<()> {
    // Resolve the vertex and fragment entry-point names from the
    // descriptor. Wyn emits one Graphics pipeline per entry point, so
    // a vertex-only and a fragment-only pipeline coexist; we collect
    // their stage names and re-merge here.
    let stages: Vec<&wyn_pipeline_descriptor::GraphicsStage> = desc
        .pipelines
        .iter()
        .filter_map(|p| if let Pipeline::Graphics(g) = p { Some(g) } else { None })
        .flat_map(|g| g.stages.iter())
        .collect();
    let vertex_entry = stages
        .iter()
        .find_map(|s| matches!(s.stage, ShaderStage::Vertex).then(|| s.entry_point.clone()))
        .ok_or_else(|| anyhow!("descriptor lacks a vertex stage"))?;
    let fragment_entry = stages
        .iter()
        .find_map(|s| matches!(s.stage, ShaderStage::Fragment).then(|| s.entry_point.clone()))
        .ok_or_else(|| anyhow!("descriptor lacks a fragment stage"))?;

    let spec = InteractivePipelineSpec {
        shader_path: spv_path,
        descriptor: desc,
        vertex_entry,
        fragment_entry,
        dispatch_overrides,
        feedback_specs,
        max_frames: opts.max_frames,
        verbose,
        validate: opts.validate,
        present_mode: opts.present_mode,
        size: opts.size,
        vertex_count: opts.vertex_count,
        topology: opts.topology,
        storage_dir: opts.storage_dir,
        buffer_inits: opts.buffer_inits,
        index_buffer: opts.index_buffer,
        outputs,
    };

    let event_loop = EventLoop::new().context("failed to create event loop")?;
    let mut app = App::new_pipeline(spec);
    event_loop.run_app(&mut app).map_err(|e| anyhow!(e)).context("winit event loop errored")?;
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
    push_constants: &[PushConstantSpec],
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Running compute pipeline: {}", cp.entry_point);
    }

    // Build push constant data from CLI args matched against descriptor bindings
    let pc_bytes = build_push_constant_bytes(&cp.bindings, push_constants, verbose)?;
    let total_pc_size = pc_bytes.len() as u32;

    let buffers = create_binding_buffers(
        device,
        queue,
        &cp.bindings,
        inputs,
        Some(&cp.dispatch_size),
        &pc_bytes,
        verbose,
    )?;
    let (layout, bind_group) = build_bind_group(device, &cp.bindings, &buffers)?;

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

    let dispatch = resolve_dispatch_size(&cp.dispatch_size, &buffers, &pc_bytes);
    if verbose {
        println!("Dispatch: {} x {} x {}", dispatch.0, dispatch.1, dispatch.2);
    }

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("compute_encoder"),
    });
    ComputeExecutor {
        label: "compute_pass",
        pipeline: &pipeline,
        bind_groups: &[&bind_group],
        push_constant_bytes: &pc_bytes,
        dispatch,
        timestamps: None,
    }
    .record(&mut encoder);
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
    push_constants: &[PushConstantSpec],
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

    // Build push constant data from CLI args matched against descriptor bindings
    let pc_bytes = build_push_constant_bytes(&mp.bindings, push_constants, verbose)?;
    let total_pc_size = pc_bytes.len() as u32;

    let buffers = create_binding_buffers(device, queue, &mp.bindings, inputs, None, &pc_bytes, verbose)?;
    let (layout, bind_group) = build_bind_group(device, &mp.bindings, &buffers)?;

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

        let dispatch = resolve_dispatch_size(&stage.dispatch_size, &buffers, &pc_bytes);
        if verbose {
            println!(
                "Stage {} ({}): dispatch {} x {} x {}",
                si, stage.entry_point, dispatch.0, dispatch.1, dispatch.2
            );
        }

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!("stage_{}_encoder", si)),
        });
        let stage_label = format!("stage_{}", si);
        ComputeExecutor {
            label: &stage_label,
            pipeline: &pipeline,
            bind_groups: &[&bind_group],
            push_constant_bytes: &pc_bytes,
            dispatch,
            timestamps: None,
        }
        .record(&mut encoder);
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
            let should_output = *usage != BufferUsage::Input || outputs.contains_key(name.as_str());

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
