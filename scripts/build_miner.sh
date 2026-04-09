#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Compile miner.wyn (produces .spv with Import linkage for sha256_compress)
cargo run --release --bin wyn -- compile testfiles/miner.wyn -o /tmp/miner.spv

# Assemble the linked SPIR-V module
spirv-as lib/sha256_compress.spvasm -o /tmp/sha256_compress.spv --target-env spv1.5

# Link the two modules
spirv-link /tmp/miner.spv /tmp/sha256_compress.spv -o testfiles/miner.spv

# Copy pipeline descriptor alongside the linked .spv
cp /tmp/miner.json testfiles/miner.json

# Validate
spirv-val testfiles/miner.spv
echo "OK: testfiles/miner.spv validated"
