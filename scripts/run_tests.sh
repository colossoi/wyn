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

echo "compile + validate: OK"

if ! $RUN; then
    echo "(pass --run to execute on the GPU via tephra)"
    exit 0
fi

echo "== tephra run (needs a Vulkan device) =="
(cd extra/tephra && cargo build)
TEPHRA=./extra/tephra/target/debug/tephra

$TEPHRA run /tmp/dist_demo.spv --entry dist_normal_fill -n 256 -w 64
$TEPHRA run /tmp/stats_normal.spv --entry stats_normal -n 1 -w 1
