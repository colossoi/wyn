//! `miner` subcommand — Bitcoin double-SHA256 miner harness driving
//! the linked miner SPIR-V module's two-stage reduce pipeline (phase 1
//! per-thread hash + check, phase 2 reduce). Decodes the compact
//! target, chunks the nonce range to dodge GPU watchdog timeouts, and
//! reports per-chunk + total throughput plus optional GPU timestamps.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use wgpu::{BufferDescriptor, BufferUsages, CommandEncoderDescriptor, PipelineLayoutDescriptor};

use crate::gpu::{ComputeExecutor, GpuTimestamps, build_bind_group, create_headless_device};
use crate::json::{Binding, BufferUsage, MultiComputePipeline, Pipeline, PipelineDescriptor};
use crate::spirv::{detect_entry_points, load_spirv_module};

/// Sentinel value the miner shader writes when no nonce in the chunk
/// produced a hash below the target.
const NO_HIT: u32 = 0xFFFF_FFFF;

/// Push-constant block layout: header_base(76) + target(32) + n(4) +
/// nonce_offset(4) = 116 bytes. The shader expects this exact layout.
const PUSH_CONSTANT_BYTES: u32 = 116;

/// Workgroup width: each phase-1 thread hashes one nonce.
const WORKGROUP_SIZE: u32 = 64;

/// Bytes per per-thread (phase-1) result entry: u32 nonce + 8 × u32
/// hash words = 36 bytes.
const RESULT_BYTES: u64 = 36;

// ---------------------------------------------------------------------------
// Stage data — split out of the original ~530-line `run_miner` so each
// piece is independently readable.
// ---------------------------------------------------------------------------

/// CLI-derived configuration: header words, decoded compact target,
/// chunking math, and the parsed pipeline descriptor. Independent of
/// any GPU state, so it's the first thing built.
struct MinerConfig {
    header_base: [u32; 19],
    target: [u32; 8],
    nonces: u32,
    nonce_offset: u32,
    workgroups_override: Option<u32>,
    /// `chunk_size` clamped to `>= WORKGROUP_SIZE`.
    chunk: u32,
    num_chunks: u32,
    descriptor: PipelineDescriptor,
    validate: bool,
    /// 0 = quiet, 1 = `-v` per-chunk progress, 2 = `-vv` also dumps partials each chunk.
    verbose: u8,
}

impl MinerConfig {
    fn new(
        header_hex: [u32; 19],
        nonces: u32,
        nonce_offset: u32,
        workgroups_override: Option<u32>,
        chunk_size: u32,
        validate: bool,
        verbose: u8,
        descriptor_path: &Path,
    ) -> Result<Self> {
        let chunk = chunk_size.max(WORKGROUP_SIZE);
        let num_chunks = (nonces + chunk - 1) / chunk;
        let bits = header_hex[18].swap_bytes();
        let target = decode_compact_target(bits);

        let descriptor_json = fs::read_to_string(descriptor_path).with_context(|| {
            format!(
                "Failed to read pipeline descriptor: {}",
                descriptor_path.display()
            )
        })?;
        let descriptor: PipelineDescriptor = serde_json::from_str(&descriptor_json)
            .with_context(|| "Failed to parse pipeline descriptor JSON")?;

        if !matches!(descriptor.pipelines.first(), Some(Pipeline::MultiCompute(_))) {
            return Err(anyhow!("Expected multi_compute pipeline in descriptor"));
        }

        Ok(Self {
            header_base: header_hex,
            target,
            nonces,
            nonce_offset,
            workgroups_override,
            chunk,
            num_chunks,
            descriptor,
            validate,
            verbose,
        })
    }

    fn mp(&self) -> &MultiComputePipeline {
        match &self.descriptor.pipelines[0] {
            Pipeline::MultiCompute(mp) => mp,
            // Constructor verifies the first pipeline is MultiCompute.
            _ => unreachable!(),
        }
    }

    fn print_intro(&self) {
        if self.verbose == 0 {
            return;
        }
        println!("Miner configuration:");
        println!("  Header: {:08x?}", self.header_base);
        println!("  Nonces: {} (offset {})", self.nonces, self.nonce_offset);
        println!("  Target: {}", format_u256_hex(&self.target));
        println!("  Chunk size: {} ({} chunks)", self.chunk, self.num_chunks);
    }

    /// Per-chunk push constants: header(76) + target(32) + n(4) + offset(4) = 116.
    fn push_constants_for(&self, chunk_idx: u32) -> Vec<u8> {
        let chunk_offset = self.nonce_offset + chunk_idx * self.chunk;
        let chunk_n = self.chunk.min(self.nonces - chunk_idx * self.chunk);
        let mut pc = Vec::with_capacity(PUSH_CONSTANT_BYTES as usize);
        for w in &self.header_base {
            pc.extend_from_slice(&w.to_le_bytes());
        }
        for w in &self.target {
            pc.extend_from_slice(&w.to_le_bytes());
        }
        pc.extend_from_slice(&chunk_n.to_le_bytes());
        pc.extend_from_slice(&chunk_offset.to_le_bytes());
        pc
    }

    fn workgroups_for(&self, chunk_idx: u32) -> u32 {
        self.workgroups_override.unwrap_or_else(|| {
            let chunk_n = self.chunk.min(self.nonces - chunk_idx * self.chunk);
            (chunk_n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
        })
    }
}

/// GPU buffers used by the chunk loop. The two staging buffers are
/// shared across all chunks (allocated once, mapped/unmapped per
/// iteration); the storage buffers are sized for the largest chunk's
/// worst case (`chunk × RESULT_BYTES` for partials).
struct MinerBuffers {
    /// All storage bindings keyed by binding number.
    storage: HashMap<u32, (wgpu::Buffer, u64)>,
    result_staging: wgpu::Buffer,
    partials_staging: wgpu::Buffer,
    result_binding: u32,
    partials_binding: u32,
}

impl MinerBuffers {
    fn new(device: &wgpu::Device, mp: &MultiComputePipeline, chunk: u32, verbose: bool) -> Result<Self> {
        let partials_size = chunk as u64 * RESULT_BYTES;

        let mut storage: HashMap<u32, (wgpu::Buffer, u64)> = HashMap::new();
        for binding in &mp.bindings {
            if let Binding::StorageBuffer {
                binding: b, usage, ..
            } = binding
            {
                let size = match usage {
                    BufferUsage::Intermediate => partials_size,
                    BufferUsage::Output => RESULT_BYTES,
                    _ => continue,
                };
                let buffer = device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("miner_binding_{}", b)),
                    size,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                storage.insert(*b, (buffer, size));
            }
        }

        if verbose {
            for (b, (_, size)) in &storage {
                println!("  Buffer binding {}: {} bytes", b, size);
            }
        }

        let result_binding = mp
            .bindings
            .iter()
            .find_map(|b| match b {
                Binding::StorageBuffer {
                    binding,
                    usage: BufferUsage::Output,
                    ..
                } => Some(*binding),
                _ => None,
            })
            .ok_or_else(|| anyhow!("No output binding in pipeline descriptor"))?;
        let partials_binding = mp
            .bindings
            .iter()
            .find_map(|b| match b {
                Binding::StorageBuffer {
                    binding,
                    usage: BufferUsage::Intermediate,
                    ..
                } => Some(*binding),
                _ => None,
            })
            .ok_or_else(|| anyhow!("No intermediate binding in pipeline descriptor"))?;

        let result_staging = device.create_buffer(&BufferDescriptor {
            label: Some("miner_staging"),
            size: RESULT_BYTES,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let partials_staging = device.create_buffer(&BufferDescriptor {
            label: Some("partials_staging"),
            size: partials_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            storage,
            result_staging,
            partials_staging,
            result_binding,
            partials_binding,
        })
    }

    fn result_buffer(&self) -> &wgpu::Buffer {
        &self.storage[&self.result_binding].0
    }

    fn partials_buffer(&self) -> &(wgpu::Buffer, u64) {
        &self.storage[&self.partials_binding]
    }
}

/// The bind group + two compute pipelines the chunk loop dispatches.
/// `pipeline_layout` is held only to keep the layout alive for the
/// pipeline's lifetime — wgpu pipelines store an internal reference
/// but the explicit ownership keeps us safely out of any "layout
/// dropped" footgun.
struct MinerPipeline {
    bind_group: wgpu::BindGroup,
    _pipeline_layout: wgpu::PipelineLayout,
    phase1: wgpu::ComputePipeline,
    phase2: wgpu::ComputePipeline,
}

impl MinerPipeline {
    fn new(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        mp: &MultiComputePipeline,
        buffers: &MinerBuffers,
        verbose: bool,
    ) -> Result<Self> {
        let (bind_group_layout, bind_group) = build_bind_group(device, &mp.bindings, &buffers.storage)?;
        if verbose {
            println!("  Bind group created");
        }

        let pc_range = wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..PUSH_CONSTANT_BYTES,
        };
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("miner_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[pc_range],
        });

        if verbose {
            println!("  Creating phase 1 pipeline ({})...", mp.stages[0].entry_point);
        }
        let phase1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("miner_phase1"),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(&mp.stages[0].entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
        if verbose {
            println!("  Creating phase 2 pipeline ({})...", mp.stages[1].entry_point);
        }
        let phase2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("miner_phase2"),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(&mp.stages[1].entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
        if verbose {
            println!("  Pipelines created");
        }

        Ok(Self {
            bind_group,
            _pipeline_layout: pipeline_layout,
            phase1,
            phase2,
        })
    }

    /// Cross-check: every entry point the descriptor names must exist
    /// in the SPIR-V module. Run after the module has been parsed but
    /// before pipeline construction so we fail fast on a stale `.json`.
    fn validate_against_spirv(spv_words: &[u32], mp: &MultiComputePipeline, verbose: bool) -> Result<()> {
        let entry_points = detect_entry_points(spv_words)?;
        let descriptor_entries: Vec<&str> = mp.stages.iter().map(|s| s.entry_point.as_str()).collect();
        let descriptor_bindings: Vec<u32> = mp
            .bindings
            .iter()
            .filter_map(|b| match b {
                Binding::StorageBuffer { binding, .. } => Some(*binding),
                _ => None,
            })
            .collect();

        if verbose {
            println!(
                "  SPIR-V entry points: {:?}",
                entry_points.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
            );
            println!("  Descriptor stages: {:?}", descriptor_entries);
            println!("  Descriptor storage bindings: {:?}", descriptor_bindings);
        }

        for stage_name in &descriptor_entries {
            if !entry_points.iter().any(|(n, _)| n == stage_name) {
                return Err(anyhow!(
                    "Descriptor references entry point '{}' not found in SPIR-V",
                    stage_name
                ));
            }
        }

        let descriptor_pc_size: u32 = mp
            .bindings
            .iter()
            .filter_map(|b| match b {
                Binding::PushConstant { offset, size, .. } => Some(offset + size),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        if descriptor_pc_size > 0 && verbose {
            println!(
                "  Push constant size from descriptor: {} bytes",
                descriptor_pc_size
            );
        }

        Ok(())
    }
}

/// Phase 1's per-thread + phase 2's single-thread output: nonce +
/// 8-word hash. `nonce == NO_HIT` means the chunk had no nonce whose
/// hash was below the target.
struct ChunkOutcome {
    nonce: u32,
    hash: [u32; 8],
}

impl ChunkOutcome {
    fn is_hit(&self) -> bool {
        self.nonce != NO_HIT
    }
}

// ---------------------------------------------------------------------------
// Per-chunk + readback machinery
// ---------------------------------------------------------------------------

/// Run one chunk: assemble push constants, dispatch phase 1
/// (optionally with a debug-partials readback in verbose mode),
/// dispatch phase 2, and read back the 36-byte result.
fn run_chunk(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cfg: &MinerConfig,
    pipeline: &MinerPipeline,
    buffers: &MinerBuffers,
    timestamps: Option<(&GpuTimestamps, &GpuTimestamps)>,
    chunk_idx: u32,
) -> Result<ChunkOutcome> {
    let chunk_n = cfg.chunk.min(cfg.nonces - chunk_idx * cfg.chunk);
    let num_workgroups = cfg.workgroups_for(chunk_idx);

    if cfg.verbose >= 1 {
        println!(
            "  Chunk {}/{}: {} workgroups × {} threads = {} threads ({} nonces)",
            chunk_idx + 1,
            cfg.num_chunks,
            num_workgroups,
            WORKGROUP_SIZE,
            num_workgroups * WORKGROUP_SIZE,
            chunk_n
        );
    }

    let pc_bytes = cfg.push_constants_for(chunk_idx);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("miner_encoder"),
    });

    // Phase 1: per-thread hash + threshold check, partial → partials buffer.
    ComputeExecutor {
        label: "miner_phase1",
        pipeline: &pipeline.phase1,
        bind_groups: &[&pipeline.bind_group],
        push_constant_bytes: &pc_bytes,
        dispatch: (num_workgroups, 1, 1),
        timestamps: timestamps.and_then(|(p1, _)| p1.writes_for(chunk_idx)),
    }
    .record(&mut encoder);

    // Optional debug (`-vv` only): dump partials between phase 1 and
    // phase 2. Splits the encoder so the readback's queue.submit doesn't
    // carry phase 2 along with it. This adds a multi-hundred-KB
    // map_async + CPU round-trip per chunk, so it stays off at `-v`.
    if cfg.verbose >= 2 {
        dump_partials(device, queue, buffers, encoder, num_workgroups)?;
        encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("miner_encoder_phase2"),
        });
    }

    // Phase 2: single-thread reduce of the partials → final result.
    ComputeExecutor {
        label: "miner_phase2",
        pipeline: &pipeline.phase2,
        bind_groups: &[&pipeline.bind_group],
        push_constant_bytes: &pc_bytes,
        dispatch: (1, 1, 1),
        timestamps: timestamps.and_then(|(_, p2)| p2.writes_for(chunk_idx)),
    }
    .record(&mut encoder);

    encoder.copy_buffer_to_buffer(
        buffers.result_buffer(),
        0,
        &buffers.result_staging,
        0,
        RESULT_BYTES,
    );
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    read_result(device, &buffers.result_staging)
}

/// Map the result staging buffer and decode the (nonce, hash) tuple.
fn read_result(device: &wgpu::Device, staging: &wgpu::Buffer) -> Result<ChunkOutcome> {
    let slice = staging.slice(..RESULT_BYTES);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv().unwrap()?;

    let data = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&data);
    let nonce = words[0];
    let mut hash = [0u32; 8];
    hash.copy_from_slice(&words[1..9]);
    drop(data);
    staging.unmap();

    Ok(ChunkOutcome { nonce, hash })
}

/// Verbose-mode dump of the partials buffer right after phase 1. The
/// caller hands over an already-encoded `encoder` (with phase 1
/// recorded); this submits it after appending the copy, then maps
/// the staging buffer and prints the first few entries.
fn dump_partials(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffers: &MinerBuffers,
    mut encoder: wgpu::CommandEncoder,
    num_workgroups: u32,
) -> Result<()> {
    let (partials_buf, partials_buf_size) = buffers.partials_buffer();
    let read_size = (num_workgroups as u64 * WORKGROUP_SIZE as u64 * RESULT_BYTES).min(*partials_buf_size);
    encoder.copy_buffer_to_buffer(partials_buf, 0, &buffers.partials_staging, 0, read_size);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    let slice = buffers.partials_staging.slice(..read_size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv().unwrap()?;
    let data = slice.get_mapped_range();
    let words: &[u32] = bytemuck::cast_slice(&data);
    // Each entry: 1 nonce + 8 hash words = 9 u32s.
    let num_entries = words.len() / 9;
    let entries_to_show = num_entries.min(8);
    println!("  Partials (first {} of {}):", entries_to_show, num_entries);
    for i in 0..entries_to_show {
        let base = i * 9;
        let nonce = words[base];
        print!("    [{}] nonce {:>10} -> ", i, nonce);
        for j in 1..9 {
            print!("{:08x}", words[base + j]);
        }
        println!();
    }
    drop(data);
    buffers.partials_staging.unmap();
    Ok(())
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// Resolve and print GPU timestamps (per-chunk + total) when both
/// phase queries are present. No-op if either is `None`.
async fn print_timing(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cfg: &MinerConfig,
    p1: Option<&GpuTimestamps>,
    p2: Option<&GpuTimestamps>,
) -> Result<()> {
    let (Some(t1), Some(t2)) = (p1, p2) else {
        return Ok(());
    };
    let p1_times = t1.read_back(device, queue).await?;
    let p2_times = t2.read_back(device, queue).await?;

    let mut total_phase1_ns = 0.0f64;
    let mut total_phase2_ns = 0.0f64;

    println!("\nGPU timing (per chunk):");
    for (i, ((p1_begin, p1_end), (p2_begin, p2_end))) in p1_times.iter().zip(p2_times.iter()).enumerate() {
        let p1_ns = p1_end - p1_begin;
        let p2_ns = p2_end - p2_begin;
        let gap_ns = p2_begin - p1_end;
        total_phase1_ns += p1_ns;
        total_phase2_ns += p2_ns;
        if cfg.verbose >= 1 || p1_times.len() <= 4 {
            let chunk_n = cfg.chunk.min(cfg.nonces - i as u32 * cfg.chunk);
            let nwg =
                cfg.workgroups_override.unwrap_or_else(|| (chunk_n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
            println!(
                "  chunk {:>3}: phase1 {:.3}ms ({} wg × {} = {} threads)  phase2 {:.3}ms  gap {:.3}ms",
                i + 1,
                p1_ns / 1_000_000.0,
                nwg,
                WORKGROUP_SIZE,
                nwg * WORKGROUP_SIZE,
                p2_ns / 1_000_000.0,
                gap_ns / 1_000_000.0
            );
        }
    }
    if !p1_times.is_empty() {
        let total_gpu_ns = total_phase1_ns + total_phase2_ns;
        let total_nonces = p1_times.len() as u64 * cfg.chunk as u64;
        let gpu_hash_rate = total_nonces as f64 / (total_phase1_ns / 1_000_000_000.0);
        println!(
            "  total:    phase1 {:.3}ms  phase2 {:.3}ms  gpu total {:.3}ms",
            total_phase1_ns / 1_000_000.0,
            total_phase2_ns / 1_000_000.0,
            total_gpu_ns / 1_000_000.0
        );
        println!(
            "  GPU hash rate: {:.0} H/s (phase1 only, excludes dispatch overhead)",
            gpu_hash_rate
        );
    }
    Ok(())
}

/// Optional pre-flight: feed the SPIR-V to wgpu's naga validator
/// inside an error scope, so a failure prints a diagnostic without
/// taking the device down.
async fn naga_validate_spirv(device: &wgpu::Device, path: &Path) -> Result<()> {
    let spv_bytes = fs::read(path)?;
    let spv_words: Vec<u32> =
        spv_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("naga_validate"),
        source: wgpu::ShaderSource::SpirV(std::borrow::Cow::Borrowed(&spv_words)),
    });
    match device.pop_error_scope().await {
        Some(e) => eprintln!("Naga validation error: {}", e),
        None => println!("Naga validation passed"),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

pub async fn run_miner(
    path: PathBuf,
    header_hex: [u32; 19],
    nonces: u32,
    nonce_offset: u32,
    workgroups_override: Option<u32>,
    chunk_size: u32,
    validate: bool,
    verbose: u8,
) -> Result<()> {
    let chatty = verbose >= 1;
    let descriptor_path = path.with_extension("json");
    let cfg = MinerConfig::new(
        header_hex,
        nonces,
        nonce_offset,
        workgroups_override,
        chunk_size,
        validate,
        verbose,
        &descriptor_path,
    )?;
    cfg.print_intro();

    let (device, queue) = create_headless_device(chatty).await?;

    if cfg.validate {
        naga_validate_spirv(&device, &path).await?;
    }

    let module = load_spirv_module(&device, &path)?;

    {
        let spv_bytes = fs::read(&path)?;
        let spv_words: Vec<u32> =
            spv_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        MinerPipeline::validate_against_spirv(&spv_words, cfg.mp(), chatty)?;
    }

    let buffers = MinerBuffers::new(&device, cfg.mp(), cfg.chunk, chatty)?;
    let pipeline = MinerPipeline::new(&device, &module, cfg.mp(), &buffers, chatty)?;

    // Two-slot timestamp profiler per chunk: phase1 + phase2.
    let gpu_phase1 = GpuTimestamps::new(&device, &queue, cfg.num_chunks);
    let gpu_phase2 = GpuTimestamps::new(&device, &queue, cfg.num_chunks);
    if gpu_phase1.is_none() && chatty {
        eprintln!("  Warning: TIMESTAMP_QUERY not supported, no GPU timing available");
    }

    let start_time = Instant::now();
    let mut hit: Option<ChunkOutcome> = None;

    for chunk_idx in 0..cfg.num_chunks {
        let chunk_start = Instant::now();
        let outcome = run_chunk(
            &device,
            &queue,
            &cfg,
            &pipeline,
            &buffers,
            gpu_phase1.as_ref().zip(gpu_phase2.as_ref()),
            chunk_idx,
        )?;
        let chunk_elapsed = chunk_start.elapsed();

        if chatty {
            print!(
                "  chunk {:>4}: {:>7.1}ms  nonce {:>10} -> ",
                chunk_idx + 1,
                chunk_elapsed.as_secs_f64() * 1000.0,
                outcome.nonce
            );
            for word in &outcome.hash {
                print!("{:08x}", word);
            }
            println!();
        }

        if outcome.is_hit() {
            hit = Some(outcome);
            break;
        }

        if chatty && cfg.num_chunks > 1 {
            let elapsed = start_time.elapsed();
            let computed = (chunk_idx + 1) as u64 * cfg.chunk as u64;
            let rate = computed as f64 / elapsed.as_secs_f64();
            eprint!(
                "\r  chunk {}/{} ({:.0}%) {:.0} H/s",
                chunk_idx + 1,
                cfg.num_chunks,
                (chunk_idx + 1) as f64 / cfg.num_chunks as f64 * 100.0,
                rate
            );
        }
    }

    if chatty && cfg.num_chunks > 1 {
        eprintln!();
    }

    let elapsed = start_time.elapsed();
    let total_computed = cfg.num_chunks.min((nonces + cfg.chunk - 1) / cfg.chunk) as u64 * cfg.chunk as u64;
    let hash_rate = total_computed as f64 / elapsed.as_secs_f64();

    print_timing(&device, &queue, &cfg, gpu_phase1.as_ref(), gpu_phase2.as_ref()).await?;

    println!(
        "Mined {} nonces in {:.2?} ({:.0} H/s wall clock)",
        nonces, elapsed, hash_rate
    );

    if let Some(outcome) = &hit {
        println!("Hit found:");
        print!("  nonce {:>10} -> ", outcome.nonce);
        for word in &outcome.hash {
            print!("{:08x}", word);
        }
        println!();
    } else {
        println!("No hits found");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Bitcoin compact-target decoding
// ---------------------------------------------------------------------------

/// Decode Bitcoin compact target format (nBits) into a 256-bit target (8 x u32, big-endian).
/// Format: top byte = exponent, bottom 3 bytes = coefficient.
/// target = coefficient * 2^(8*(exponent-3))
fn decode_compact_target(bits: u32) -> [u32; 8] {
    let exponent = (bits >> 24) as usize;
    let coefficient = bits & 0x007fffff;
    // The coefficient occupies 3 bytes starting at byte position (exponent - 3) from the MSB end.
    // In a 32-byte (256-bit) big-endian number, byte 0 is the most significant.
    let mut target_bytes = [0u8; 32];
    if exponent >= 3 && exponent <= 32 {
        let start = 32 - exponent; // byte index of the most significant coefficient byte
        target_bytes[start] = ((coefficient >> 16) & 0xff) as u8;
        if start + 1 < 32 {
            target_bytes[start + 1] = ((coefficient >> 8) & 0xff) as u8;
        }
        if start + 2 < 32 {
            target_bytes[start + 2] = (coefficient & 0xff) as u8;
        }
    }
    // Convert to 8 x BE u32 words
    let mut words = [0u32; 8];
    for (i, chunk) in target_bytes.chunks_exact(4).enumerate() {
        words[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    words
}

/// Format a 256-bit value (8 x u32, big-endian) as hex.
fn format_u256_hex(words: &[u32; 8]) -> String {
    words.iter().map(|w| format!("{:08x}", w)).collect()
}
