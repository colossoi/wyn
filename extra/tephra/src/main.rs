//! Tephra: Minimal ash-based compute shader runner for Wyn
//!
//! Usage:
//!   tephra run <shader.spv> [--entry <name>] [--size <n>] [--input <values>]
//!   tephra pipeline <config.json>
//!   tephra mine <miner.spv> --header-hex <152 hex chars> [--nonces <n>]

mod miner;
mod pipeline;
mod vk_helpers;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use vk_helpers::{ComputeContext, StorageBuffer};

#[derive(Parser, Debug)]
#[command(name = "tephra")]
#[command(about = "Run Wyn compute shaders via Vulkan")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run a single-buffer compute shader and print its output.
    Run {
        /// Path to SPIR-V compute shader.
        shader: PathBuf,

        /// Entry point name.
        #[arg(short, long, default_value = "main")]
        entry: String,

        /// Number of elements in the buffer.
        #[arg(short = 'n', long, default_value = "64")]
        size: usize,

        /// Workgroup size X (must match shader's LocalSize).
        #[arg(short = 'w', long, default_value = "64")]
        workgroup: u32,

        /// Input values (comma-separated floats, or 'iota' for 0,1,2,...).
        #[arg(short, long, default_value = "iota")]
        input: String,
    },

    /// Run a multi-buffer compute pipeline from a configuration JSON.
    Pipeline {
        /// Path to pipeline configuration JSON.
        config: PathBuf,
    },

    /// Drive the Bitcoin miner's two-stage reduce pipeline over a nonce range.
    Mine {
        /// Path to the linked miner SPIR-V module.
        shader: PathBuf,

        /// Raw block header hex (152 hex chars = 76 bytes, everything except
        /// the nonce), converted to 19 big-endian u32 words.
        #[arg(long, value_parser = parse_header_hex)]
        header_hex: [u32; 19],

        /// Number of nonces to try.
        #[arg(short = 'n', long, default_value = "1024")]
        nonces: u32,

        /// Starting nonce offset.
        #[arg(long, default_value = "0")]
        nonce_offset: u32,

        /// Workgroups per dispatch (each 64 threads). The phase1 grid is fixed
        /// (saturating), so this is a constant grid width, default 1024.
        #[arg(long)]
        workgroups: Option<u32>,

        /// Max nonces per GPU dispatch; the range is chunked to dodge watchdog timeouts.
        #[arg(short = 'c', long, default_value = "262144")]
        chunk_size: u32,

        /// Verbose output.
        #[arg(short, long, action = clap::ArgAction::Count)]
        verbose: u8,
    },
}

/// Parse a 76-byte block header from 152 hex chars into 19 big-endian u32 words.
fn parse_header_hex(s: &str) -> std::result::Result<[u32; 19], String> {
    let s = s.trim();
    if s.len() != 152 {
        return Err(format!(
            "expected 152 hex chars (76 bytes), got {} chars",
            s.len()
        ));
    }
    let mut words = [0u32; 19];
    for (i, word) in words.iter_mut().enumerate() {
        let hex = &s[i * 8..(i + 1) * 8];
        let bytes: [u8; 4] = [
            u8::from_str_radix(&hex[0..2], 16).map_err(|e| format!("bad hex at byte {}: {}", i * 4, e))?,
            u8::from_str_radix(&hex[2..4], 16)
                .map_err(|e| format!("bad hex at byte {}: {}", i * 4 + 1, e))?,
            u8::from_str_radix(&hex[4..6], 16)
                .map_err(|e| format!("bad hex at byte {}: {}", i * 4 + 2, e))?,
            u8::from_str_radix(&hex[6..8], 16)
                .map_err(|e| format!("bad hex at byte {}: {}", i * 4 + 3, e))?,
        ];
        *word = u32::from_be_bytes(bytes);
    }
    Ok(words)
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Command::Pipeline { config } => run_pipeline(&config),
        Command::Mine {
            shader,
            header_hex,
            nonces,
            nonce_offset,
            workgroups,
            chunk_size,
            verbose,
        } => miner::run(
            &shader,
            header_hex,
            nonces,
            nonce_offset,
            workgroups,
            chunk_size,
            verbose,
        ),
        Command::Run {
            shader,
            entry,
            size,
            workgroup,
            input,
        } => run_simple(&shader, &entry, size, workgroup, &input),
    }
}

/// Single-buffer compute: load `shader`, run `entry` over `size` elements of
/// `input` (comma-separated floats or `iota`), print the output.
fn run_simple(shader: &PathBuf, entry: &str, size: usize, workgroup: u32, input: &str) -> Result<()> {
    let input_data: Vec<f32> = if input == "iota" {
        (0..size).map(|i| i as f32).collect()
    } else {
        input
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to parse input values")?
    };

    if input_data.len() != size {
        bail!(
            "Input size mismatch: got {} values, expected {}",
            input_data.len(),
            size
        );
    }

    let spirv_bytes =
        std::fs::read(shader).with_context(|| format!("Failed to read shader: {:?}", shader))?;

    if spirv_bytes.len() % 4 != 0 {
        bail!("SPIR-V file size must be a multiple of 4 bytes");
    }

    let spirv_words: Vec<u32> = spirv_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let output_data = run_compute(&spirv_words, entry, &input_data, workgroup)?;

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
