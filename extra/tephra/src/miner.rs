//! `--mine` mode: drive the Wyn miner's two-stage reduce pipeline.
//!
//! The compiled miner (`miner.wyn`) reduces over the nonce space, so it
//! lowers to a `multi_compute` pipeline: phase 1 hashes a chunk of nonces
//! per thread and writes each thread's best `(nonce, hash)` to a
//! `partials` buffer; phase 2 reduces the partials to a single best hit in
//! a `result` buffer. We read the pipeline descriptor (`<shader>.json`) to
//! find the two entry points and the partials/result bindings, decode the
//! difficulty target from the header's `bits` word, and dispatch the two
//! stages per nonce chunk (chunking dodges GPU watchdog timeouts).
//!
//! `partials` is descriptor-tagged `usage: "intermediate"` — pure
//! inter-stage GPU scratch the host never touches — so it's allocated
//! device-local. Only the tiny `result` buffer is host-visible.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

use crate::pipeline::load_spirv;
use crate::vk_helpers::{ComputeContext, MultiStagePipeline, StorageBuffer};

/// Push-constant block layout the shader expects: header_base(76) +
/// target(32) + n(4) + nonce_offset(4) = 116 bytes.
const PUSH_CONSTANT_BYTES: usize = 116;

/// Phase-1 workgroup width: each thread hashes one nonce.
const WORKGROUP_SIZE: u32 = 64;

/// Bytes per result entry: u32 nonce + 8 × u32 hash = 36 bytes.
const RESULT_BYTES: usize = 36;

/// Sentinel nonce the shader writes when no nonce in the chunk produced a
/// hash below the target.
const NO_HIT: u32 = 0xFFFF_FFFF;

// ---------------------------------------------------------------------------
// Pipeline descriptor (minimal subset of wyn's `<shader>.json`)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Descriptor {
    pipelines: Vec<PipelineEntry>,
}

#[derive(Deserialize)]
struct PipelineEntry {
    kind: String,
    #[serde(default)]
    bindings: Vec<BindingEntry>,
    #[serde(default)]
    stages: Vec<StageEntry>,
}

/// Storage-buffer and push-constant bindings flatten into one shape; the
/// `type` tag distinguishes them and the irrelevant fields stay defaulted.
#[derive(Deserialize)]
struct BindingEntry {
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    binding: u32,
    #[serde(default)]
    usage: String,
}

#[derive(Deserialize)]
struct StageEntry {
    entry_point: String,
}

/// Everything the dispatch loop needs out of the descriptor.
struct MinerPlan {
    phase1: String,
    phase2: String,
    partials_binding: u32,
    result_binding: u32,
    storage_count: u32,
}

fn parse_plan(descriptor_path: &Path) -> Result<MinerPlan> {
    let text = std::fs::read_to_string(descriptor_path).with_context(|| {
        format!(
            "Failed to read pipeline descriptor: {}",
            descriptor_path.display()
        )
    })?;
    let desc: Descriptor = serde_json::from_str(&text).context("Failed to parse pipeline descriptor")?;

    let mc = desc
        .pipelines
        .iter()
        .find(|p| p.kind == "multi_compute")
        .ok_or_else(|| anyhow!("descriptor has no multi_compute pipeline (is this a miner shader?)"))?;
    if mc.stages.len() < 2 {
        return Err(anyhow!("expected 2 compute stages, found {}", mc.stages.len()));
    }

    let storage: Vec<&BindingEntry> = mc.bindings.iter().filter(|b| b.ty == "storage_buffer").collect();
    let partials_binding = storage
        .iter()
        .find(|b| b.usage == "intermediate")
        .ok_or_else(|| anyhow!("descriptor has no intermediate (partials) storage binding"))?
        .binding;
    let result_binding = storage
        .iter()
        .find(|b| b.usage == "output")
        .ok_or_else(|| anyhow!("descriptor has no output (result) storage binding"))?
        .binding;

    Ok(MinerPlan {
        phase1: mc.stages[0].entry_point.clone(),
        phase2: mc.stages[1].entry_point.clone(),
        partials_binding,
        result_binding,
        storage_count: storage.len() as u32,
    })
}

// ---------------------------------------------------------------------------
// Bitcoin target / push-constant helpers
// ---------------------------------------------------------------------------

/// Decode Bitcoin compact target (nBits) into a 256-bit target (8 × u32,
/// big-endian): top byte is the exponent, low 3 bytes the coefficient, so
/// `target = coefficient · 2^(8·(exponent-3))`.
fn decode_compact_target(bits: u32) -> [u32; 8] {
    let exponent = (bits >> 24) as usize;
    let coefficient = bits & 0x007f_ffff;
    let mut target_bytes = [0u8; 32];
    if (3..=32).contains(&exponent) {
        let start = 32 - exponent;
        target_bytes[start] = ((coefficient >> 16) & 0xff) as u8;
        if start + 1 < 32 {
            target_bytes[start + 1] = ((coefficient >> 8) & 0xff) as u8;
        }
        if start + 2 < 32 {
            target_bytes[start + 2] = (coefficient & 0xff) as u8;
        }
    }
    let mut words = [0u32; 8];
    for (i, chunk) in target_bytes.chunks_exact(4).enumerate() {
        words[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    words
}

fn format_u256_hex(words: &[u32; 8]) -> String {
    words.iter().map(|w| format!("{:08x}", w)).collect()
}

/// Assemble the 116-byte push-constant block for one chunk.
fn push_constants_for(header_base: &[u32; 19], target: &[u32; 8], n: u32, nonce_offset: u32) -> Vec<u8> {
    let mut pc = Vec::with_capacity(PUSH_CONSTANT_BYTES);
    for w in header_base {
        pc.extend_from_slice(&w.to_le_bytes());
    }
    for w in target {
        pc.extend_from_slice(&w.to_le_bytes());
    }
    pc.extend_from_slice(&n.to_le_bytes());
    pc.extend_from_slice(&nonce_offset.to_le_bytes());
    pc
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn run(
    shader_path: &Path,
    header_base: [u32; 19],
    nonces: u32,
    nonce_offset: u32,
    workgroups_override: Option<u32>,
    chunk_size: u32,
    verbose: u8,
) -> Result<()> {
    let descriptor_path = shader_path.with_extension("json");
    let plan = parse_plan(&descriptor_path)?;
    let spirv = load_spirv(shader_path)?;

    let chunk = chunk_size.max(WORKGROUP_SIZE);
    let num_chunks = nonces.div_ceil(chunk);
    let target = decode_compact_target(header_base[18].swap_bytes());

    if verbose >= 1 {
        println!("Miner configuration:");
        println!("  Header: {:08x?}", header_base);
        println!("  Nonces: {} (offset {})", nonces, nonce_offset);
        println!("  Target: {}", format_u256_hex(&target));
        println!("  Chunk size: {} ({} chunks)", chunk, num_chunks);
        println!("  Stages: {} -> {}", plan.phase1, plan.phase2);
    }

    let ctx = ComputeContext::new()?;
    if verbose >= 1 {
        println!("Device: {}", ctx.device_name());
    }

    let max_pc = ctx.max_push_constants_size();
    if PUSH_CONSTANT_BYTES as u32 > max_pc {
        return Err(anyhow!(
            "shader needs {} bytes of push constants but device supports at most {}",
            PUSH_CONSTANT_BYTES,
            max_pc
        ));
    }

    let pipeline = MultiStagePipeline::new(
        &ctx,
        &spirv,
        &[&plan.phase1, &plan.phase2],
        plan.storage_count,
        PUSH_CONSTANT_BYTES as u32,
    )?;

    // partials: device-local scratch sized for a full chunk's worst case;
    // the host never reads or writes it (descriptor `usage: intermediate`).
    // result: host-visible, read back after each chunk.
    let partials = StorageBuffer::new_device_local(&ctx, chunk as usize * RESULT_BYTES)?;
    let result = StorageBuffer::new_host_bytes(&ctx, RESULT_BYTES)?;

    // Bind buffers in descriptor-binding order: buffers[binding] = buffer.
    let mut by_binding: Vec<(u32, &StorageBuffer)> =
        vec![(plan.partials_binding, &partials), (plan.result_binding, &result)];
    by_binding.sort_by_key(|(b, _)| *b);
    let buffers: Vec<&StorageBuffer> = by_binding.iter().map(|(_, buf)| *buf).collect();

    let start = Instant::now();
    let mut hit: Option<(u32, [u32; 8])> = None;

    for chunk_idx in 0..num_chunks {
        let chunk_offset = nonce_offset + chunk_idx * chunk;
        let chunk_n = chunk.min(nonces - chunk_idx * chunk);
        let num_workgroups = workgroups_override.unwrap_or_else(|| chunk_n.div_ceil(WORKGROUP_SIZE));

        let pc = push_constants_for(&header_base, &target, chunk_n, chunk_offset);
        pipeline.dispatch(&buffers, &[[num_workgroups, 1, 1], [1, 1, 1]], &pc)?;

        let words = result.read_u32(RESULT_BYTES / 4)?;
        let nonce = words[0];
        let mut hash = [0u32; 8];
        hash.copy_from_slice(&words[1..9]);

        if verbose >= 1 {
            print!(
                "  chunk {}/{}: {} wg × {} threads ({} nonces) -> nonce {:>10} ",
                chunk_idx + 1,
                num_chunks,
                num_workgroups,
                WORKGROUP_SIZE,
                chunk_n,
                nonce
            );
            for w in &hash {
                print!("{:08x}", w);
            }
            println!();
        }

        if nonce != NO_HIT {
            hit = Some((nonce, hash));
            break;
        }
    }

    let elapsed = start.elapsed();
    let computed = num_chunks as u64 * chunk as u64;
    let rate = computed as f64 / elapsed.as_secs_f64();
    println!(
        "Mined {} nonces in {:.2?} ({:.0} H/s wall clock)",
        nonces, elapsed, rate
    );

    match hit {
        Some((nonce, hash)) => {
            print!("Hit found:\n  nonce {:>10} -> ", nonce);
            for w in &hash {
                print!("{:08x}", w);
            }
            println!();
        }
        None => println!("No hits found"),
    }

    Ok(())
}
