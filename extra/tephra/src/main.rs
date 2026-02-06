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

    /// Bitcoin miner mode
    #[arg(long)]
    mine: bool,

    /// Mining difficulty (leading zero bits in hash)
    #[arg(long, default_value = "10")]
    difficulty: u32,

    /// Mining batch size (nonces per dispatch)
    #[arg(long, default_value = "65536")]
    batch_size: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Pipeline mode: multi-buffer JSON configuration
    if let Some(pipeline_path) = args.pipeline {
        return run_pipeline(&pipeline_path);
    }

    // Simple mode: single buffer
    let shader = args.shader.expect("shader required without --pipeline");

    // Mine mode: bitcoin miner
    if args.mine {
        return run_mine(
            &shader,
            &args.entry,
            args.difficulty,
            args.batch_size,
            args.workgroup,
        );
    }

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

fn load_spirv_file(path: &std::path::Path) -> Result<Vec<u32>> {
    let spirv_bytes = std::fs::read(path).with_context(|| format!("Failed to read shader: {:?}", path))?;
    if spirv_bytes.len() % 4 != 0 {
        bail!("SPIR-V file size must be a multiple of 4 bytes");
    }
    Ok(spirv_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Count leading zero bits of a 256-bit hash in Bitcoin display order.
/// Bitcoin displays hashes in reversed byte order: word[7] first, word[0] last,
/// with bytes within each word also reversed.
fn leading_zero_bits(hash: &[u32; 8]) -> u32 {
    let mut zeros = 0u32;
    // Bitcoin display order: word 7 down to word 0
    for i in (0..8).rev() {
        let word = hash[i].swap_bytes(); // reverse bytes within word
        if word == 0 {
            zeros += 32;
        } else {
            zeros += word.leading_zeros();
            break;
        }
    }
    zeros
}

fn format_hash(hash: &[u32; 8]) -> String {
    // Bitcoin display order: reversed words, reversed bytes
    let mut s = String::with_capacity(64);
    for i in (0..8).rev() {
        let bytes = hash[i].swap_bytes().to_be_bytes();
        for b in bytes {
            s.push_str(&format!("{:02x}", b));
        }
    }
    s
}

fn run_mine(
    shader_path: &std::path::Path,
    entry_name: &str,
    difficulty: u32,
    batch_size: u32,
    workgroup_size: u32,
) -> Result<()> {
    use std::time::Instant;

    // Bitcoin genesis block header (80 bytes = 20 u32 words, big-endian)
    // Version(1) + PrevHash(8) + MerkleRoot(8) + Time(1) + Bits(1) + Nonce(1)
    // We pass the first 19 words (76 bytes) as header_base; the shader appends the nonce.
    //
    // Known answer: nonce = 0x1DAC2B7C (= 2083236893)
    let header_base: [u32; 19] = [
        0x01000000, // version
        0x00000000, 0x00000000, 0x00000000, 0x00000000, // prev hash (all zeros for genesis)
        0x00000000, 0x00000000, 0x00000000, 0x00000000, //
        0x3BA3EDFD, 0x7A7B12B2, 0x7AC72C3E, 0x67768F61, // merkle root
        0x7FC81BC3, 0x888A5132, 0x3A9FB8AA, 0x4B1E5E4A, //
        0x29AB5F49, // time (2009-01-03 18:15:05 UTC)
        0xFFFF001D, // bits (difficulty target)
    ];

    let spirv = load_spirv_file(shader_path)?;

    eprintln!("tephra mine v0.1");

    let ctx = ComputeContext::new()?;
    eprintln!("Device: {}", ctx.device_name());

    // Push constants: 19 u32 header_base (76 bytes) + 1 i32 n (4 bytes) + 1 i32 nonce_offset (4 bytes) = 84 bytes
    let push_constant_size = 84u32;
    let max_pc = ctx.max_push_constants_size();
    if push_constant_size > max_pc {
        bail!(
            "Shader requires {} bytes of push constants but device supports at most {}",
            push_constant_size,
            max_pc
        );
    }

    eprintln!("Header: Bitcoin genesis block (2009-01-03)");
    eprintln!("Target: {} leading zero bits", difficulty);
    eprintln!("Batch:  {} nonces/round", batch_size);
    eprintln!();

    // Output buffer: batch_size * 8 u32 values (one 8-word hash per nonce)
    let output_len = batch_size as usize * 8;
    let mut output_buf = StorageBuffer::new(&ctx, output_len)?;
    output_buf.upload_u32(&vec![0u32; output_len])?;

    let binding_count = 1u32;

    let pipeline =
        ctx.create_compute_pipeline_multi(&spirv, entry_name, binding_count, push_constant_size)?;

    let num_workgroups = (batch_size + workgroup_size - 1) / workgroup_size;

    let mut best_zeros = 0u32;
    let mut total_hashes = 0u64;
    let start_time = Instant::now();

    eprintln!("Mining...");

    // Mining loop: each round tries batch_size nonces
    for round in 0u32.. {
        // Build push constant data: header_base (76 bytes) + n (4 bytes) + nonce_offset (4 bytes)
        let nonce_offset = round * batch_size;
        let mut pc_bytes: Vec<u8> = Vec::with_capacity(84);
        for &word in &header_base {
            pc_bytes.extend_from_slice(&word.to_le_bytes());
        }
        pc_bytes.extend_from_slice(&(batch_size as i32).to_le_bytes());
        pc_bytes.extend_from_slice(&(nonce_offset as i32).to_le_bytes());

        // Zero the output buffer
        output_buf.upload_u32(&vec![0u32; output_len])?;

        pipeline.dispatch_multi(&[&output_buf], [num_workgroups, 1, 1], &pc_bytes)?;

        let output = output_buf.download_u32()?;
        total_hashes += batch_size as u64;

        let elapsed = start_time.elapsed().as_secs_f64();
        let hashrate = total_hashes as f64 / elapsed;

        // Scan hashes
        let mut round_best_zeros = 0u32;
        let mut round_best_nonce = 0u32;
        let mut round_best_hash = [0u32; 8];

        for i in 0..batch_size as usize {
            let mut hash = [0u32; 8];
            hash.copy_from_slice(&output[i * 8..(i + 1) * 8]);

            let zeros = leading_zero_bits(&hash);
            if zeros > round_best_zeros {
                round_best_zeros = zeros;
                round_best_nonce = i as u32;
                round_best_hash = hash;
            }
        }

        eprint!(
            "\r  Round {} | {} hashes | {:.1} kH/s | best: {} bits",
            round + 1,
            total_hashes,
            hashrate / 1000.0,
            best_zeros.max(round_best_zeros),
        );

        if round_best_zeros > best_zeros {
            best_zeros = round_best_zeros;
            eprintln!();
            eprintln!(
                "  New best: {} bits | nonce {} | hash: {}",
                best_zeros,
                nonce_offset + round_best_nonce,
                format_hash(&round_best_hash),
            );
        }

        if best_zeros >= difficulty {
            eprintln!();
            eprintln!("  BLOCK FOUND!");
            eprintln!("  Nonce:  {}", nonce_offset + round_best_nonce);
            eprintln!("  Hash:   {}", format_hash(&round_best_hash));
            eprintln!("  Zeros:  {} bits", best_zeros);
            eprintln!("  Time:   {:.2}s ({:.1} kH/s)", elapsed, hashrate / 1000.0,);
            return Ok(());
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
