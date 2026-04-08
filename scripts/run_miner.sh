#!/usr/bin/env bash
set -euo pipefail

# Run the miner compute shader with 2^N nonces.
# Usage: ./scripts/run_miner.sh [N]
# Default N=10 (1024 nonces)

N="${1:-10}"
NONCES=$((1 << N))
BUFFER_SIZE=$((NONCES * 8))  # 8 u32s per hash

cd "$(dirname "$0")/.."

# Build the linked miner if needed
if [ ! -f testfiles/miner.spv ]; then
    echo "Building miner..."
    bash scripts/build_miner.sh
fi

echo "Mining 2^$N = $NONCES nonces..."

cd extra/viz
cargo run --release -q -- compute ../../testfiles/miner.spv \
    --push-constant "header_base:u32x19=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" \
    --push-constant "n:i32=$NONCES" \
    --push-constant "nonce_offset:i32=0" \
    --storage "0:${BUFFER_SIZE}:u32" \
    2>/dev/null

echo "Done: $NONCES hashes computed"
