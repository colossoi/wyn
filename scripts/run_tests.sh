#!/usr/bin/env bash
#
# Compile + validate the rng/dist library testfiles, and (with --run) execute
# them on the GPU via tephra to eyeball the output. Hand-coded commands, one
# block per driver — when you add a driver to lib/testfiles/, add its commands
# here. The unit-test suite is separate: run it with `cargo test`.
#
# Usage:  scripts/run_tests.sh [--run]
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN=false
[ "${1:-}" = "--run" ] && RUN=true

echo "Building wyn..."
cargo build -p wyn
WYN=./target/debug/wyn

echo "== compile + validate =="

$WYN compile lib/testfiles/dist_demo.wyn -o /tmp/dist_demo.spv
spirv-val /tmp/dist_demo.spv
$WYN compile lib/testfiles/dist_demo.wyn -t wgsl -o /tmp/dist_demo.wgsl

$WYN compile lib/testfiles/stats_normal.wyn -o /tmp/stats_normal.spv
spirv-val /tmp/stats_normal.spv
$WYN compile lib/testfiles/stats_normal.wyn -t wgsl -o /tmp/stats_normal.wgsl

$WYN compile lib/testfiles/stats_uniform.wyn -o /tmp/stats_uniform.spv
spirv-val /tmp/stats_uniform.spv
$WYN compile lib/testfiles/stats_uniform.wyn -t wgsl -o /tmp/stats_uniform.wgsl

$WYN compile lib/testfiles/stats_exponential.wyn -o /tmp/stats_exponential.spv
spirv-val /tmp/stats_exponential.spv
$WYN compile lib/testfiles/stats_exponential.wyn -t wgsl -o /tmp/stats_exponential.wgsl

$WYN compile lib/testfiles/stats_uniform_int.wyn -o /tmp/stats_uniform_int.spv
spirv-val /tmp/stats_uniform_int.spv
$WYN compile lib/testfiles/stats_uniform_int.wyn -t wgsl -o /tmp/stats_uniform_int.wgsl

$WYN compile lib/testfiles/noise_smoke.wyn -o /tmp/noise_smoke.spv
spirv-val /tmp/noise_smoke.spv
$WYN compile lib/testfiles/noise_smoke.wyn -t wgsl -o /tmp/noise_smoke.wgsl

echo "compile + validate: OK"

if ! $RUN; then
    echo "(pass --run to execute on the GPU via tephra)"
    exit 0
fi

echo "== tephra run (needs a Vulkan device) =="
(cd extra/tephra && cargo build)
TEPHRA=./extra/tephra/target/debug/tephra

$TEPHRA run /tmp/dist_demo.spv --entry dist_normal_fill -n 256 -w 64

echo
echo "Stats slots: [count, mean, variance, stddev, min, max] over N=65536 draws."

echo
echo "--- stats_normal (standard normal) ---"
$TEPHRA run /tmp/stats_normal.spv --entry stats_normal -n 6 -w 64
echo "Expect: mean 0, variance 1, stddev 1, extremes ~ ± 4 to 5."

echo
echo "--- stats_uniform (uniform real [0,1)) ---"
$TEPHRA run /tmp/stats_uniform.spv --entry stats_uniform -n 6 -w 64
echo "Expect: mean 0.5, variance 1/12 ≈ 0.0833, stddev ≈ 0.289, min ≈ 0, max ≈ 1."

echo
echo "--- stats_exponential (exponential rate 1) ---"
$TEPHRA run /tmp/stats_exponential.spv --entry stats_exponential -n 6 -w 64
echo "Expect: mean 1, variance 1, stddev 1, min ≈ 0, max ≈ 10 to 12."

echo
echo "--- stats_uniform_int (uniform int [0,10), lifted to f32) ---"
$TEPHRA run /tmp/stats_uniform_int.spv --entry stats_uniform_int -n 6 -w 64
echo "Expect: mean 4.5, variance 99/12 ≈ 8.25, stddev ≈ 2.872, min 0, max 9."

echo
echo "--- noise_smoke (one query at p=(3.5, 7.25), seed 0x9e3779b9) ---"
$TEPHRA run /tmp/noise_smoke.spv --entry noise_smoke -n 5 -w 64
echo "Slots: [value2, perlin2, simplex2, worley2, fbm_perlin(6oct, lac=2, gain=0.5)]."
echo "Expect: first three in [-1, 1]; worley2 a small positive distance (~0..1.5);"
echo "        fbm_perlin a damped sum of octaves (no fixed range, but bounded)."
