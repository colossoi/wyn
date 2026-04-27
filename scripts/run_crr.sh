#!/usr/bin/env bash
set -euo pipefail

# Run testfiles/multi/crr.wyn end-to-end: compile, dispatch via
# `viz compute`, print the per-option prices.
#
# The shader is a Cox-Ross-Rubinstein 32-step American-option pricer.
# Inputs come from testfiles/multi/crr_fixtures/*.json (8 sample
# options); uniforms `now` and `rfr` are passed on the command line.
#
# Usage: ./scripts/run_crr.sh
#
# Override defaults via env vars: NOW=0.0 RFR=0.05.

cd "$(dirname "$0")/.."

NOW="${NOW:-0.0}"
RFR="${RFR:-0.05}"
SPV="/tmp/crr.spv"
FIX="testfiles/multi/crr_fixtures"

# Build wyn + viz if missing.
if [ ! -x ./target/release/wyn ]; then
    echo "Building wyn (release)..."
    cargo build --release -q
fi
if [ ! -x ./extra/viz/target/release/viz ]; then
    echo "Building viz (release)..."
    (cd extra/viz && cargo build --release -q)
fi

echo "Compiling testfiles/multi/crr.wyn -> $SPV"
./target/release/wyn compile testfiles/multi/crr.wyn -o "$SPV"

echo "Dispatching with now=$NOW, rfr=$RFR"
./extra/viz/target/release/viz compute "$SPV" \
    --entry price_options \
    --workgroups-x 1 \
    --storage "0:0:8:f32:$FIX/spot.json" \
    --storage "0:1:8:f32:$FIX/strike.json" \
    --storage "0:2:8:i32:$FIX/opt_type.json" \
    --storage "0:3:8:f32:$FIX/tte.json" \
    --storage "0:4:8:f32:$FIX/sigma.json" \
    --storage 0:5:8:f32 \
    --uniform "1:0:f32=$NOW" \
    --uniform "1:1:f32=$RFR"
