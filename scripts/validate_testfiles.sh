#!/usr/bin/env bash

# Compile and validate all .wyn test files with partial-eval

set -e

KEEP=false
OUT_DIR="/tmp"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep|-k)
            KEEP=true
            shift
            ;;
        --out-dir|-o)
            OUT_DIR="$2"
            KEEP=true
            shift 2
            ;;
        *)
            echo "Usage: $0 [--keep|-k] [--out-dir|-o DIR]"
            echo "  --keep, -k       Keep generated .spv files (in /tmp by default)"
            echo "  --out-dir, -o    Output directory for .spv files (implies --keep)"
            exit 1
            ;;
    esac
done

echo "Building wyn (release)..."
cargo build --release -p wyn

for f in testfiles/*.wyn; do
    base=$(basename "$f" .wyn)
    spv_path="${OUT_DIR}/${base}.spv"
    printf "Compiling %s... " "$f"

    if ! compile_err=$(./target/release/wyn compile "$f" --partial-eval -o "$spv_path" 2>&1); then
        echo "COMPILE FAILED"
        echo "$compile_err"
        continue
    fi

    printf "validating... "

    if ! val_err=$(spirv-val "$spv_path" 2>&1); then
        echo "VALIDATION FAILED"
        echo "$val_err"
        if [ "$KEEP" = false ]; then rm -f "$spv_path"; fi
        continue
    fi

    if [ "$KEEP" = true ]; then
        echo "OK → $spv_path"
    else
        echo "OK"
        rm -f "$spv_path"
    fi
done
