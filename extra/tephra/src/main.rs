//! Tephra: Minimal ash-based compute shader runner for Wyn
//!
//! Usage: tephra <shader.spv> [--entry <name>] [--size <n>] [--input <values>]

mod vk_helpers;

use anyhow::{bail, Context, Result};
use clap::Parser;
use std::path::PathBuf;
use vk_helpers::{ComputeContext, StorageBuffer};

#[derive(Parser, Debug)]
#[command(name = "tephra")]
#[command(about = "Run Wyn compute shaders via Vulkan")]
struct Args {
    /// Path to SPIR-V compute shader
    shader: PathBuf,

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
}

fn main() -> Result<()> {
    let args = Args::parse();

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
    let spirv_bytes = std::fs::read(&args.shader)
        .with_context(|| format!("Failed to read shader: {:?}", args.shader))?;

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
