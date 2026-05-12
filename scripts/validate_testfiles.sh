#!/usr/bin/env bash

# Compile and validate all .wyn test files with partial-eval

set -e

KEEP=false
OUT_DIR="/tmp"
MODE="spirv"
PROFILE="debug"

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
        --wgsl)
            MODE="wgsl"
            shift
            ;;
        --release)
            PROFILE="release"
            shift
            ;;
        *)
            echo "Usage: $0 [--keep|-k] [--out-dir|-o DIR] [--glsl|--wgsl] [--release]"
            echo "  --keep, -k       Keep generated files (in /tmp by default)"
            echo "  --out-dir, -o    Output directory (implies --keep)"
            echo "  --glsl           Compile to GLSL (shadertoy) instead of SPIR-V"
            echo "  --wgsl           Compile to WGSL and validate with 'viz validate'"
            echo "  --release        Build wyn (and viz, for --wgsl) with --release"
            echo "                   (default: debug — builds faster, runs slower)"
            exit 1
            ;;
    esac
done

if [ "$PROFILE" = "release" ]; then
    CARGO_FLAG="--release"
    WYN_BIN="./target/release/wyn"
    VIZ_BIN="./extra/viz/target/release/viz"
else
    CARGO_FLAG=""
    WYN_BIN="./target/debug/wyn"
    VIZ_BIN="./extra/viz/target/debug/viz"
fi

echo "Building wyn ($PROFILE)..."
cargo build $CARGO_FLAG -p wyn

if [ "$MODE" = "wgsl" ]; then
    echo "Building viz ($PROFILE) for WGSL validation..."
    (cd extra/viz && cargo build $CARGO_FLAG --quiet)
fi

FAIL=0
PASS=0
SKIP=0

# Files directly in testfiles/ (compiler-feature tests) and
# testfiles/playground/ (visual shader demos / shadertoy ports).
# Walk both, not deeper — anything nested under playground/ is the
# canonical playground gallery, and any future subdirs should be
# added explicitly.
for f in testfiles/*.wyn testfiles/playground/*.wyn; do
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
        # Auto-lifted compute pre-passes (e.g. ripples.wyn) synthesize
        # storage bindings during compilation. Shadertoy GLSL can't
        # express those either. `--single-stage` would disable the
        # lift, but the default sweep doesn't pass it.
        if [ "$base" = "ripples" ]; then
            printf "Skipping %s (auto-lifted compute pre-pass, no Shadertoy equivalent)\n" "$f"
            SKIP=$((SKIP + 1))
            continue
        fi

        out_path="${OUT_DIR}/${base}.glsl"
        printf "Compiling %s → GLSL... " "$f"

        if ! compile_err=$("$WYN_BIN" compile "$f" -t shadertoy -o "$out_path" 2>&1); then
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
    elif [ "$MODE" = "wgsl" ]; then
        # WGSL mode: compile + validate via viz (naga in-process).
        # Skip testfiles that depend on `impl_source`-linked SPIR-V
        # helpers (e.g. sha256_compress) — WGSL has no equivalent
        # linkage path, and reimplementing those helpers inline is a
        # separate workstream.
        if [ "$base" = "miner" ] || [ "$base" = "sha256_test" ]; then
            printf "Skipping %s (depends on linked SPIR-V helpers)\n" "$f"
            SKIP=$((SKIP + 1))
            continue
        fi

        out_path="${OUT_DIR}/${base}.wgsl"
        printf "Compiling %s → WGSL... " "$f"

        if ! compile_err=$("$WYN_BIN" compile "$f" -t wgsl -o "$out_path" 2>&1); then
            echo "COMPILE FAILED"
            echo "$compile_err"
            FAIL=$((FAIL + 1))
            continue
        fi

        printf "validating... "

        if ! val_err=$("$VIZ_BIN" validate "$out_path" 2>&1); then
            echo "VALIDATION FAILED"
            echo "$val_err"
            if [ "$KEEP" = false ]; then rm -f "$out_path"; fi
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

        if ! compile_err=$("$WYN_BIN" compile "$f" -o "$spv_path" 2>&1); then
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
