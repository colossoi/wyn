// Stable two- or four-bit radix sort corresponding to src/radix_sort.wyn.
// See FUTHARK-LICENSE for the upstream algorithm's license.
// Build: rustc --edition=2021 -O -C target-cpu=native pkg/sort/radix_cpu.rs -o /tmp/radix-cpu
// Run: /tmp/radix-cpu 4194304 5 4
// Single-threaded; timings include the input copy and scratch allocation,
// but exclude input generation, the reference sort, and output verification.
use std::{env, error::Error, hint::black_box, mem, time::Instant};

fn radix_sort<const BINS: usize>(input: &[i32]) -> Vec<i32> {
    let mut current = input.to_vec();
    if current.is_empty() {
        return current;
    }
    let mut scratch = vec![0; current.len()];
    for digit in (0..32).step_by(BINS.trailing_zeros() as usize) {
        // Flipping the sign bit maps signed order to unsigned bit-pattern order.
        let bin = |value: i32| (((value as u32 ^ 0x8000_0000) >> digit) & (BINS as u32 - 1)) as usize;
        let mut totals = [0usize; BINS];
        for &value in &current {
            totals[bin(value)] += 1;
        }
        let mut base = 0;
        let bases: [usize; BINS] = std::array::from_fn(|bucket| {
            let start = base;
            base += totals[bucket];
            start
        });
        let mut prefixes = [0usize; BINS];
        // Consume each inclusive prefix immediately in the stable scatter.
        for &value in &current {
            let bucket = bin(value);
            prefixes[bucket] += 1;
            scratch[bases[bucket] + prefixes[bucket] - 1] = value;
        }
        mem::swap(&mut current, &mut scratch);
    }
    current
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let count: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(25_000_000);
    let runs: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(5);
    let bits: u32 = args.next().map(|v| v.parse()).transpose()?.unwrap_or(4);
    if runs == 0 || !matches!(bits, 2 | 4) || args.next().is_some() {
        return Err("usage: radix-cpu [element-count] [positive-run-count] [digit-bits: 2|4]".into());
    }
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
        let output = match bits {
            2 => radix_sort::<4>(black_box(&input)),
            _ => radix_sort::<16>(black_box(&input)),
        };
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if output != expected {
            return Err(format!("incorrect sort on run {run}").into());
        }
        println!(
            "run={run} sort_ms={elapsed:.6} verified_elements={count}{}",
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
        "digit_bits={bits} bytes={} median_ms={median:.3} min_ms={:.3} max_ms={:.3}",
        count * mem::size_of::<i32>(),
        timings[0],
        timings[runs - 1]
    );
    Ok(())
}
