//! `miner` subcommand — Bitcoin double-SHA256 miner harness driving
//! the linked miner SPIR-V module's two-stage reduce pipeline (phase 1
//! per-thread hash + check, phase 2 reduce). Decodes the compact
//! target, chunks the nonce range to dodge GPU watchdog timeouts, and
//! reports per-chunk + total throughput plus optional GPU timestamps.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use wgpu::{BufferDescriptor, BufferUsages, CommandEncoderDescriptor, PipelineLayoutDescriptor};

use crate::gpu::{GpuTimestamps, build_bind_group, create_headless_device};
use crate::json::{Binding, BufferUsage, Pipeline, PipelineDescriptor};
use crate::spirv::{detect_entry_points, load_spirv_module};

pub async fn run_miner(
    path: PathBuf,
    header_hex: [u32; 19],
    nonces: u32,
    nonce_offset: u32,
    workgroups_override: Option<u32>,
    chunk_size: u32,
    validate: bool,
    verbose: bool,
) -> Result<()> {
    let header_base = header_hex;
    let workgroup_size = 64u32;
    let chunk = chunk_size.max(workgroup_size);
    let num_chunks = (nonces + chunk - 1) / chunk;

    let bits_be = header_base[18];
    let bits = bits_be.swap_bytes();
    let target = decode_compact_target(bits);

    if verbose {
        println!("Miner configuration:");
        println!("  Header: {:08x?}", header_base);
        println!("  Nonces: {} (offset {})", nonces, nonce_offset);
        println!("  Target: {}", format_u256_hex(&target));
        println!("  Chunk size: {} ({} chunks)", chunk, num_chunks);
    }

    // Load pipeline descriptor (JSON sidecar next to the .spv)
    let descriptor_path = path.with_extension("json");
    let descriptor_json = fs::read_to_string(&descriptor_path).with_context(|| {
        format!(
            "Failed to read pipeline descriptor: {}",
            descriptor_path.display()
        )
    })?;
    let descriptor: PipelineDescriptor = serde_json::from_str(&descriptor_json)
        .with_context(|| "Failed to parse pipeline descriptor JSON")?;

    let mp = match descriptor.pipelines.first() {
        Some(Pipeline::MultiCompute(mp)) => mp,
        _ => return Err(anyhow!("Expected multi_compute pipeline in descriptor")),
    };

    let (device, queue) = create_headless_device(verbose).await?;

    if validate {
        let spv_bytes = fs::read(&path)?;
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
    }

    let module = load_spirv_module(&device, &path)?;

    // Validate: cross-check SPIR-V bindings against pipeline descriptor
    {
        let spv_bytes = fs::read(&path)?;
        let spv_words: Vec<u32> =
            spv_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let entry_points = detect_entry_points(&spv_words)?;
        let descriptor_entries: Vec<&str> = mp.stages.iter().map(|s| s.entry_point.as_str()).collect();
        let descriptor_bindings: Vec<u32> =
            mp.bindings
                .iter()
                .filter_map(|b| {
                    if let Binding::StorageBuffer { binding, .. } = b { Some(*binding) } else { None }
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

        // Check push constant size matches
        let descriptor_pc_size: u32 = mp
            .bindings
            .iter()
            .filter_map(|b| {
                if let Binding::PushConstant { offset, size, .. } = b { Some(offset + size) } else { None }
            })
            .max()
            .unwrap_or(0);
        if descriptor_pc_size > 0 && verbose {
            println!(
                "  Push constant size from descriptor: {} bytes",
                descriptor_pc_size
            );
        }
    }

    // Create buffers from pipeline descriptor bindings.
    // Result buffer: 1 element of (u32, [8]u32) = 36 bytes
    // Partials buffer: one entry per thread in the largest chunk dispatch.
    // Max threads per chunk = chunk (one thread per nonce in the chunk).
    let result_size = 36u64;
    let max_threads_per_chunk = chunk as u64;
    let partials_size = max_threads_per_chunk * result_size;

    let mut buffers: HashMap<u32, (wgpu::Buffer, u64)> = HashMap::new();
    for binding in &mp.bindings {
        if let Binding::StorageBuffer {
            binding: b, usage, ..
        } = binding
        {
            let size = match usage {
                BufferUsage::Intermediate => partials_size,
                BufferUsage::Output => result_size,
                _ => continue,
            };
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("miner_binding_{}", b)),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            buffers.insert(*b, (buffer, size));
        }
    }

    if verbose {
        for (b, (_, size)) in &buffers {
            println!("  Buffer binding {}: {} bytes", b, size);
        }
    }

    let (bind_group_layout, bind_group) = build_bind_group(&device, &mp.bindings, &buffers)?;
    if verbose {
        println!("  Bind group created");
    }

    // Build pipeline layout with push constants
    let pc_range = wgpu::PushConstantRange {
        stages: wgpu::ShaderStages::COMPUTE,
        range: 0..116, // header_base(76) + target(32) + n(4) + nonce_offset(4)
    };
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("miner_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[pc_range],
    });

    // Create compute pipelines for both stages
    if verbose {
        println!("  Creating phase 1 pipeline ({})...", mp.stages[0].entry_point);
    }
    let phase1_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("miner_phase1"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(&mp.stages[0].entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    if verbose {
        println!("  Creating phase 2 pipeline ({})...", mp.stages[1].entry_point);
    }
    let phase2_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("miner_phase2"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(&mp.stages[1].entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    if verbose {
        println!("  Pipelines created");
    }

    // Staging buffer for reading back the 36-byte result
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("miner_staging"),
        size: result_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Find buffer bindings
    let result_binding = mp
        .bindings
        .iter()
        .find_map(|b| {
            if let Binding::StorageBuffer {
                binding,
                usage: BufferUsage::Output,
                ..
            } = b
            {
                Some(*binding)
            } else {
                None
            }
        })
        .ok_or_else(|| anyhow!("No output binding in pipeline descriptor"))?;
    let partials_binding = mp
        .bindings
        .iter()
        .find_map(|b| {
            if let Binding::StorageBuffer {
                binding,
                usage: BufferUsage::Intermediate,
                ..
            } = b
            {
                Some(*binding)
            } else {
                None
            }
        })
        .ok_or_else(|| anyhow!("No intermediate binding in pipeline descriptor"))?;

    // Staging buffer for partials readback (debug)
    let partials_staging = device.create_buffer(&BufferDescriptor {
        label: Some("partials_staging"),
        size: partials_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // GPU timestamp profiling (2 slots per chunk: phase1, phase2)
    let gpu_phase1 = GpuTimestamps::new(&device, &queue, num_chunks);
    let gpu_phase2 = GpuTimestamps::new(&device, &queue, num_chunks);
    if gpu_phase1.is_none() && verbose {
        eprintln!("  Warning: TIMESTAMP_QUERY not supported, no GPU timing available");
    }

    let start_time = std::time::Instant::now();
    let mut hit: Option<(u32, Vec<u32>)> = None;

    for chunk_idx in 0..num_chunks {
        let chunk_offset = nonce_offset + chunk_idx * chunk;
        let chunk_n = chunk.min(nonces - chunk_idx * chunk);
        let num_workgroups =
            workgroups_override.unwrap_or_else(|| (chunk_n + workgroup_size - 1) / workgroup_size);

        if verbose {
            println!(
                "  Chunk {}/{}: {} workgroups × {} threads = {} threads ({} nonces)",
                chunk_idx + 1,
                num_chunks,
                num_workgroups,
                workgroup_size,
                num_workgroups * workgroup_size,
                chunk_n
            );
        }

        // Push constants: header_base(76) + target(32) + n(4) + nonce_offset(4) = 116 bytes
        let mut pc_bytes = Vec::with_capacity(116);
        for word in &header_base {
            pc_bytes.extend_from_slice(&word.to_le_bytes());
        }
        for word in &target {
            pc_bytes.extend_from_slice(&word.to_le_bytes());
        }
        pc_bytes.extend_from_slice(&chunk_n.to_le_bytes());
        pc_bytes.extend_from_slice(&chunk_offset.to_le_bytes());

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("miner_encoder"),
        });

        // Phase 1: each thread hashes its chunk, writes partial to partials buffer
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("miner_phase1"),
                timestamp_writes: gpu_phase1.as_ref().map(|t| t.writes_for(chunk_idx)),
            });
            cpass.set_pipeline(&phase1_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_push_constants(0, &pc_bytes);
            cpass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Debug: dump partials after phase 1
        if verbose {
            let (partials_buf, partials_buf_size) = &buffers[&partials_binding];
            let read_size =
                (num_workgroups as u64 * workgroup_size as u64 * result_size).min(*partials_buf_size);
            encoder.copy_buffer_to_buffer(partials_buf, 0, &partials_staging, 0, read_size);
            queue.submit(Some(encoder.finish()));
            let _ = device.poll(wgpu::PollType::Wait);

            let slice = partials_staging.slice(..read_size);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
            let _ = device.poll(wgpu::PollType::Wait);
            rx.recv().unwrap()?;
            let data = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&data);
            let num_entries = words.len() / 9; // each entry: 1 nonce + 8 hash words
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
            partials_staging.unmap();

            // Need a new encoder since we submitted the old one
            encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("miner_encoder_phase2"),
            });
        }

        // Phase 2: single thread combines partials → result
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("miner_phase2"),
                timestamp_writes: gpu_phase2.as_ref().map(|t| t.writes_for(chunk_idx)),
            });
            cpass.set_pipeline(&phase2_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_push_constants(0, &pc_bytes);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // Copy result to staging
        let (result_buf, _) = &buffers[&result_binding];
        encoder.copy_buffer_to_buffer(result_buf, 0, &staging, 0, result_size);
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::Wait);

        // Read back result
        let buffer_slice = staging.slice(..result_size);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap()?;

        let data = buffer_slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);
        let nonce = words[0];
        let hash = &words[1..9];

        if verbose {
            print!("  chunk {:>4}: nonce {:>10} -> ", chunk_idx + 1, nonce);
            for word in hash {
                print!("{:08x}", word);
            }
            println!();
        }

        // Check sentinel: 0xFFFFFFFF means no hit
        if nonce != 0xFFFFFFFF {
            hit = Some((nonce, hash.to_vec()));
        }

        drop(data);
        staging.unmap();

        if hit.is_some() {
            break;
        }

        if verbose && num_chunks > 1 {
            let elapsed = start_time.elapsed();
            let computed = (chunk_idx + 1) as u64 * chunk as u64;
            let rate = computed as f64 / elapsed.as_secs_f64();
            eprint!(
                "\r  chunk {}/{} ({:.0}%) {:.0} H/s",
                chunk_idx + 1,
                num_chunks,
                (chunk_idx + 1) as f64 / num_chunks as f64 * 100.0,
                rate
            );
        }
    }

    if verbose && num_chunks > 1 {
        eprintln!();
    }

    let elapsed = start_time.elapsed();
    let total_computed = num_chunks.min((nonces + chunk - 1) / chunk) as u64 * chunk as u64;
    let hash_rate = total_computed as f64 / elapsed.as_secs_f64();

    // Resolve and print GPU timestamps
    if let (Some(t1), Some(t2)) = (&gpu_phase1, &gpu_phase2) {
        let p1_times = t1.read_back(&device, &queue).await?;
        let p2_times = t2.read_back(&device, &queue).await?;

        let mut total_phase1_ns = 0.0f64;
        let mut total_phase2_ns = 0.0f64;

        println!("\nGPU timing (per chunk):");
        for (i, ((p1_begin, p1_end), (p2_begin, p2_end))) in
            p1_times.iter().zip(p2_times.iter()).enumerate()
        {
            let p1_ns = p1_end - p1_begin;
            let p2_ns = p2_end - p2_begin;
            let gap_ns = p2_begin - p1_end;
            total_phase1_ns += p1_ns;
            total_phase2_ns += p2_ns;
            if verbose || p1_times.len() <= 4 {
                let chunk_n = chunk.min(nonces - i as u32 * chunk);
                let nwg =
                    workgroups_override.unwrap_or_else(|| (chunk_n + workgroup_size - 1) / workgroup_size);
                println!(
                    "  chunk {:>3}: phase1 {:.3}ms ({} wg × {} = {} threads)  phase2 {:.3}ms  gap {:.3}ms",
                    i + 1,
                    p1_ns / 1_000_000.0,
                    nwg,
                    workgroup_size,
                    nwg * workgroup_size,
                    p2_ns / 1_000_000.0,
                    gap_ns / 1_000_000.0
                );
            }
        }
        if !p1_times.is_empty() {
            let total_gpu_ns = total_phase1_ns + total_phase2_ns;
            let total_nonces = p1_times.len() as u64 * chunk as u64;
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
    }

    println!(
        "Mined {} nonces in {:.2?} ({:.0} H/s wall clock)",
        nonces, elapsed, hash_rate
    );

    if let Some((nonce, hash)) = &hit {
        println!("Hit found:");
        print!("  nonce {:>10} -> ", nonce);
        for word in hash {
            print!("{:08x}", word);
        }
        println!();
    } else {
        println!("No hits found");
    }

    Ok(())
}

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
