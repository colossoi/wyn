#!/usr/bin/env bash

# Compile and validate all .wyn test files with partial-eval

set -e

KEEP=false
OUT_DIR="/tmp"
MODE="spirv"

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
        --glsl)
            MODE="glsl"
            shift
            ;;
        *)
            echo "Usage: $0 [--keep|-k] [--out-dir|-o DIR] [--glsl]"
            echo "  --keep, -k       Keep generated files (in /tmp by default)"
            echo "  --out-dir, -o    Output directory (implies --keep)"
            echo "  --glsl           Compile to GLSL (shadertoy) instead of SPIR-V"
            exit 1
            ;;
    esac
done

echo "Building wyn (release)..."
cargo build --release -p wyn

FAIL=0
PASS=0
SKIP=0

for f in testfiles/*.wyn; do
    base=$(basename "$f" .wyn)

    if [ "$MODE" = "glsl" ]; then
        # GLSL mode: compile to shadertoy. Skip compute-only shaders
        # and anything with module-scope `#[storage]` — Shadertoy
        # GLSL is fragment-only with fixed iResolution/iTime/iMouse
        # uniforms; custom storage bindings have no equivalent.
        if grep -q '#\[compute\]' "$f" && ! grep -q '#\[fragment\]' "$f"; then
            printf "Skipping %s (compute-only)\n" "$f"
            SKIP=$((SKIP + 1))
            continue
        fi
        if grep -q '#\[storage' "$f"; then
            printf "Skipping %s (uses module-scope #[storage])\n" "$f"
            SKIP=$((SKIP + 1))
            continue
        fi

        out_path="${OUT_DIR}/${base}.glsl"
        printf "Compiling %s → GLSL... " "$f"

        if ! compile_err=$(./target/release/wyn compile "$f" -t shadertoy -o "$out_path" 2>&1); then
            echo "FAILED"
            echo "$compile_err"
            FAIL=$((FAIL + 1))
            continue
        fi

        if [ "$KEEP" = true ]; then
            echo "OK → $out_path"
        else
            echo "OK"
            rm -f "$out_path"
        fi
        PASS=$((PASS + 1))
    else
        # SPIR-V mode (default)
        spv_path="${OUT_DIR}/${base}.spv"
        printf "Compiling %s... " "$f"

        if ! compile_err=$(./target/release/wyn compile "$f" -o "$spv_path" 2>&1); then
            echo "COMPILE FAILED"
            echo "$compile_err"
            FAIL=$((FAIL + 1))
            continue
        fi

        printf "validating... "

        if ! val_err=$(spirv-val "$spv_path" 2>&1); then
            echo "VALIDATION FAILED"
            echo "$val_err"
            if [ "$KEEP" = false ]; then rm -f "$spv_path"; fi
            FAIL=$((FAIL + 1))
            continue
        fi

        if [ "$KEEP" = true ]; then
            echo "OK → $spv_path"
        else
            echo "OK"
            rm -f "$spv_path"
        fi
        PASS=$((PASS + 1))
    fi
done

echo ""
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped"
