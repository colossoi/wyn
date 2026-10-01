// Parallel stable two-bit radix sort corresponding to src/radix_sort.wyn.
// See FUTHARK-LICENSE for the upstream algorithm's license.
// Build: RUSTFLAGS="-C target-cpu=native" cargo build --release --manifest-path pkg/sort/cpu-bench/Cargo.toml
// Run: RAYON_NUM_THREADS=8 pkg/sort/cpu-bench/target/release/radix-rayon 25000000 5
// Timings include input copying and scratch allocation; exclude pool setup,
// input generation, reference sorting, and verification. No unsafe code.
use rayon::prelude::*;
use std::{env, error::Error, hint::black_box, mem, time::Instant};

fn radix_sort(input: &[i32]) -> Vec<i32> {
    let mut current = input.to_vec();
    if current.is_empty() {
        return current;
    }
    let mut scratch = vec![0; current.len()];
    let chunk_size = current.len().div_ceil(rayon::current_num_threads() * 4);
    for digit in (0..32).step_by(2) {
        let bin = |value: i32| (((value as u32 ^ 0x8000_0000) >> digit) & 3) as usize;
        let counts: Vec<[usize; 4]> = current
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut counts = [0; 4];
                for &value in chunk {
                    counts[bin(value)] += 1;
                }
                counts
            })
            .collect();
        let mut totals = [0; 4];
        for counts in &counts {
            for bucket in 0..4 {
                totals[bucket] += counts[bucket];
            }
        }
        // Partition the output first by bucket, then by ordered input chunk.
        // Prefix sums over chunk counts give disjoint mutable slices, so each
        // worker scatters independently without atomics or raw pointers.
        let mut tail = scratch.as_mut_slice();
        let mut buckets: [&mut [i32]; 4] = std::array::from_fn(|bucket| {
            let (head, rest) = mem::take(&mut tail).split_at_mut(totals[bucket]);
            tail = rest;
            head
        });
        let destinations: Vec<[&mut [i32]; 4]> = counts
            .iter()
            .map(|counts| {
                std::array::from_fn(|bucket| {
                    let (head, rest) = mem::take(&mut buckets[bucket]).split_at_mut(counts[bucket]);
                    buckets[bucket] = rest;
                    head
                })
            })
            .collect();
        current.par_chunks(chunk_size).zip(destinations.into_par_iter()).for_each(|(chunk, output)| {
            let mut positions = [0; 4];
            for &value in chunk {
                let bucket = bin(value);
                output[bucket][positions[bucket]] = value;
                positions[bucket] += 1;
            }
        });
        mem::swap(&mut current, &mut scratch);
    }
    current
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let count: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(25_000_000);
    let runs: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(5);
    if runs == 0 || args.next().is_some() {
        return Err("usage: radix-rayon [element-count] [positive-run-count]".into());
    }
    // Initialize Rayon before timing, just as GPU pipeline setup is excluded.
    println!("rayon_threads={}", rayon::current_num_threads());
    // Match the 100 MB Rust/WGPU benchmark's xorshift32 input exactly.
    let mut seed = 0x12345678u32;
    let input: Vec<i32> = (0..count)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as i32
        })
        .collect();
    let mut expected = input.clone();
    expected.sort_unstable();
    let mut timings = Vec::with_capacity(runs);
    for run in 0..=runs {
        let start = Instant::now();
        let output = radix_sort(black_box(&input));
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if output != expected {
            return Err(format!("incorrect sort on run {run}").into());
        }
        println!(
            "run={run} sort_ms={elapsed:.3} verified_elements={count}{}",
            if run == 0 { " warmup" } else { "" }
        );
        if run != 0 {
            timings.push(elapsed);
        }
    }
    timings.sort_by(f64::total_cmp);
    let middle = runs / 2;
    let median = (timings[(runs - 1) / 2] + timings[middle]) / 2.0;
    println!(
        "bytes={} median_ms={median:.3} min_ms={:.3} max_ms={:.3}",
        count * mem::size_of::<i32>(),
        timings[0],
        timings[runs - 1]
    );
    Ok(())
}
