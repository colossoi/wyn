//! Tephra: Minimal ash-based compute shader runner for Wyn
//!
//! Usage:
//!   tephra <shader.spv> [--entry <name>] [--size <n>] [--input <values>]
//!   tephra --pipeline <config.json>

mod pipeline;
mod vk_helpers;

use anyhow::{Context, Result, bail};
use clap::Parser;
use std::path::PathBuf;
use vk_helpers::{ComputeContext, StorageBuffer};

#[derive(Parser, Debug)]
#[command(name = "tephra")]
#[command(about = "Run Wyn compute shaders via Vulkan")]
struct Args {
    /// Path to SPIR-V compute shader (for simple mode)
    #[arg(required_unless_present = "pipeline")]
    shader: Option<PathBuf>,

    /// Entry point name
    #[arg(short, long, default_value = "main")]
    entry: String,

    /// Number of elements in the buffer
    #[arg(short = 'n', long, default_value = "64")]
    size: usize,

    /// Workgroup size X (must match shader's LocalSize)
    #[arg(short = 'w', long, default_value = "64")]
    workgroup: u32,

    /// Input values (comma-separated floats, or 'iota' for 0,1,2,...)
    #[arg(short, long, default_value = "iota")]
    input: String,

    /// Path to pipeline configuration JSON (for multi-buffer mode)
    #[arg(short, long)]
    pipeline: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Pipeline mode: multi-buffer JSON configuration
    if let Some(pipeline_path) = args.pipeline {
        return run_pipeline(&pipeline_path);
    }

    // Simple mode: single buffer
    let shader = args.shader.expect("shader required without --pipeline");

    // Parse input data
    let input_data: Vec<f32> = if args.input == "iota" {
        (0..args.size).map(|i| i as f32).collect()
    } else {
        args.input
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to parse input values")?
    };

    if input_data.len() != args.size {
        bail!(
            "Input size mismatch: got {} values, expected {}",
            input_data.len(),
            args.size
        );
    }

    // Load SPIR-V
    let spirv_bytes =
        std::fs::read(&shader).with_context(|| format!("Failed to read shader: {:?}", shader))?;

    if spirv_bytes.len() % 4 != 0 {
        bail!("SPIR-V file size must be a multiple of 4 bytes");
    }

    let spirv_words: Vec<u32> = spirv_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Run compute shader
    let output_data = run_compute(&spirv_words, &args.entry, &input_data, args.workgroup)?;

    // Print results
    println!("Input:  {:?}", &input_data[..input_data.len().min(16)]);
    if input_data.len() > 16 {
        println!("        ... ({} total)", input_data.len());
    }
    println!("Output: {:?}", &output_data[..output_data.len().min(16)]);
    if output_data.len() > 16 {
        println!("        ... ({} total)", output_data.len());
    }

    Ok(())
}

fn run_pipeline(config_path: &PathBuf) -> Result<()> {
    use pipeline::{PipelineConfig, ShaderInterface, load_buffer_data, load_spirv};

    // Load and resolve config
    let mut config = PipelineConfig::load(config_path)?;
    if let Some(parent) = config_path.parent() {
        config.resolve_paths(parent);
    }

    eprintln!("Pipeline: {:?}", config_path);
    eprintln!("  SPIR-V: {:?}", config.spirv);
    eprintln!("  Interface: {:?}", config.interface);
    eprintln!("  Dispatch: {:?}", config.dispatch);

    // Load interface and SPIR-V
    let interface = ShaderInterface::load(&config.interface)?;
    let spirv = load_spirv(&config.spirv)?;

    // Find entry point
    let entry_point = interface
        .find_entry_point(&config.entry)
        .with_context(|| format!("Entry point '{}' not found", config.entry))?;

    if entry_point.execution_model != "compute" {
        bail!("Entry point '{}' is not a compute shader", config.entry);
    }

    eprintln!(
        "  Entry: {} (workgroup {:?})",
        entry_point.name, entry_point.workgroup_size
    );

    // Initialize Vulkan
    let ctx = ComputeContext::new()?;
    eprintln!("  Device: {}", ctx.device_name());

    // Get buffers in binding order
    let buffers_info = interface.buffers_by_binding();
    let binding_count = buffers_info.len() as u32;

    // Calculate push constant size (4 bytes per u32 constant)
    let push_constant_size = (config.push_constants.len() * 4) as u32;

    // Create pipeline with correct binding count
    let pipeline =
        ctx.create_compute_pipeline_multi(&spirv, &config.entry, binding_count, push_constant_size)?;

    // Create and populate buffers
    let mut storage_buffers = Vec::new();
    for buf_info in &buffers_info {
        let data = if let Some(data_path) = config.buffer_data.get(&buf_info.name) {
            load_buffer_data(data_path)?
        } else {
            // Default: zero-filled buffer based on buffer size estimate
            // For now, use a reasonable default size
            eprintln!("  Warning: No data for buffer '{}', using zeros", buf_info.name);
            vec![0.0f32; 1024]
        };

        eprintln!(
            "  Buffer '{}' (binding {}): {} elements",
            buf_info.name,
            buf_info.binding,
            data.len()
        );

        let mut buffer = StorageBuffer::new(&ctx, data.len())?;
        buffer.upload(&data)?;
        storage_buffers.push(buffer);
    }

    // Build push constants bytes
    let push_bytes: Vec<u8> = config.push_constants.values().flat_map(|v| v.to_le_bytes()).collect();

    // Dispatch
    let buffer_refs: Vec<&StorageBuffer> = storage_buffers.iter().collect();
    pipeline.dispatch_multi(&buffer_refs, config.dispatch, &push_bytes)?;

    // Download and print results from all writable buffers
    for (i, buf_info) in buffers_info.iter().enumerate() {
        if buf_info.access != "readonly" {
            let output = storage_buffers[i].download()?;
            println!("Buffer '{}' output:", buf_info.name);
            println!("  {:?}", &output[..output.len().min(16)]);
            if output.len() > 16 {
                println!("  ... ({} total)", output.len());
            }
        }
    }

    Ok(())
}

fn run_compute(spirv: &[u32], entry_name: &str, input: &[f32], workgroup_size: u32) -> Result<Vec<f32>> {
    // Initialize Vulkan context
    let ctx = ComputeContext::new()?;
    eprintln!("Using device: {}", ctx.device_name());

    // Create storage buffer with input data
    let mut buffer = StorageBuffer::new(&ctx, input.len())?;
    buffer.upload(input)?;

    // Create and run compute pipeline
    let pipeline = ctx.create_compute_pipeline(spirv, entry_name)?;

    let num_workgroups = (input.len() as u32 + workgroup_size - 1) / workgroup_size;
    pipeline.dispatch(&buffer, num_workgroups)?;

    // Read back results
    buffer.download()
}
