#!/usr/bin/env bash

# Compile and validate all .wyn test files
# Runs twice: once without partial-eval, once with partial-eval

set -e

echo "Building wyn (release)..."
cargo build --release -p wyn

for f in testfiles/*.wyn; do
    base=$(basename "$f" .wyn)

    # First pass: without --partial-eval
    spv_path="/tmp/${base}.spv"
    printf "Compiling %s... " "$f"

    if ! compile_err=$(./target/release/wyn compile "$f" -o "$spv_path" 2>&1); then
        echo "COMPILE FAILED"
        echo "$compile_err"
        continue
    fi

    printf "validating... "

    if ! val_err=$(spirv-val "$spv_path" 2>&1); then
        echo "VALIDATION FAILED"
        echo "$val_err"
        continue
    fi

    echo "OK"

    # Second pass: with --partial-eval
    spv_path_pe="/tmp/${base}_pe.spv"
    printf "Compiling %s (partial-eval)... " "$f"

    if ! compile_err=$(./target/release/wyn compile "$f" --partial-eval -o "$spv_path_pe" 2>&1); then
        echo "COMPILE FAILED"
        echo "$compile_err"
        continue
    fi

    printf "validating... "

    if ! val_err=$(spirv-val "$spv_path_pe" 2>&1); then
        echo "VALIDATION FAILED"
        echo "$val_err"
        continue
    fi

    echo "OK"
done
