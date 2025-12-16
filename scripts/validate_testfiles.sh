#!/usr/bin/env bash

# Compile and validate all .wyn test files

set -e

echo "Building wyn (release)..."
cargo build --release

for f in testfiles/*.wyn; do
    base=$(basename "$f" .wyn)
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
done
