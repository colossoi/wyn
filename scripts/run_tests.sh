#!/usr/bin/env bash
#
# Compile + validate the rng library demos, and (with --run) execute them on
# the GPU via tephra. Hand-coded commands, one block per demo — when you add a
# driver to lib/testfiles/, add its commands here.
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

$WYN compile lib/testfiles/fasthash_demo.wyn -o /tmp/fasthash_demo.spv
spirv-val /tmp/fasthash_demo.spv
$WYN compile lib/testfiles/fasthash_demo.wyn -t wgsl -o /tmp/fasthash_demo.wgsl

$WYN compile lib/testfiles/threefry_demo.wyn -o /tmp/threefry_demo.spv
spirv-val /tmp/threefry_demo.spv
$WYN compile lib/testfiles/threefry_demo.wyn -t wgsl -o /tmp/threefry_demo.wgsl

echo "compile + validate: OK"

if ! $RUN; then
    echo "(pass --run to execute on the GPU via tephra)"
    exit 0
fi

echo "== tephra run (needs a Vulkan device) =="
(cd extra/tephra && cargo build)
TEPHRA=./extra/tephra/target/debug/tephra

$TEPHRA run /tmp/fasthash_demo.spv --entry fasthash_fill -n 256 -w 64 --input iota
$TEPHRA run /tmp/threefry_demo.spv --entry threefry_fill -n 256 -w 64 --input iota
