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
        --wgsl)
            MODE="wgsl"
            shift
            ;;
        --release)
            PROFILE="release"
            shift
            ;;
        *)
            echo "Usage: $0 [--keep|-k] [--out-dir|-o DIR] [--wgsl] [--release]"
            echo "  --keep, -k       Keep generated files (in /tmp by default)"
            echo "  --out-dir, -o    Output directory (implies --keep)"
            echo "  --wgsl           Compile to WGSL and validate with 'viz validate'"
            echo "  --release        Build wyn (and viz, for --wgsl) with --release"
            echo "                   (default: debug — builds faster, runs slower)"
            exit 1
            ;;
    esac
done

if [ "$PROFILE" = "release" ]; then
    CARGO_FLAG="--release"
else
    CARGO_FLAG=""
fi

# Ask Cargo so CARGO_TARGET_DIR and build.target-dir work for both crates.
# Python 3 is used only to read Cargo's JSON metadata.
cargo_target_dir() {
    cargo metadata --no-deps --format-version 1 --manifest-path "$1" |
        python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])'
}

WYN_BIN="$(cargo_target_dir Cargo.toml)/$PROFILE/wyn"

echo "Building wyn ($PROFILE)..."
cargo build $CARGO_FLAG -p wyn

if [ "$MODE" = "wgsl" ]; then
    VIZ_BIN="$(cd extra/viz && cargo_target_dir Cargo.toml)/$PROFILE/viz"
    echo "Building viz ($PROFILE) for WGSL validation..."
    (cd extra/viz && cargo build $CARGO_FLAG --quiet)
fi

FAIL=0
PASS=0
SKIP=0
PLAYGROUND_HEADER="scripts/playground_image_header.wyn"
PREPARED_SOURCE=""

cleanup_prepared_source() {
    if [ -n "$PREPARED_SOURCE" ]; then
        rm -f "$PREPARED_SOURCE"
    fi
}
trap cleanup_prepared_source EXIT

# Files directly in testfiles/ (compiler-feature tests) and
# testfiles/playground/ (visual shader demos / shadertoy ports).
# Walk both, not deeper — anything nested under playground/ is the
# canonical playground gallery, and any future subdirs should be
# added explicitly.
for f in testfiles/*.wyn testfiles/playground/*.wyn; do
    base=$(basename "$f" .wyn)
    compile_source="$f"

    # Bare playground image shaders intentionally omit their fullscreen
    # pipeline. Recreate the source the web playground compiles by prepending
    # its shared image header. Explicit compute/feedback pipelines retain an
    # entry declaration and compile unchanged.
    if [[ "$f" == testfiles/playground/* ]] && ! grep -q '^entry ' "$f"; then
        PREPARED_SOURCE=$(mktemp "testfiles/playground/.validate-${base}-XXXXXXXX.wyn")
        { cat "$PLAYGROUND_HEADER"; printf '\n'; cat "$f"; } > "$PREPARED_SOURCE"
        compile_source="$PREPARED_SOURCE"
    else
        PREPARED_SOURCE=""
    fi

    if [ "$MODE" = "wgsl" ]; then
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

        if ! compile_err=$("$WYN_BIN" build "$compile_source" --graphics -t wgsl -o "$out_path" 2>&1); then
            cleanup_prepared_source
            PREPARED_SOURCE=""
            echo "COMPILE FAILED"
            echo "$compile_err"
            FAIL=$((FAIL + 1))
            continue
        fi
        cleanup_prepared_source
        PREPARED_SOURCE=""

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

        if ! compile_err=$("$WYN_BIN" build "$compile_source" --graphics -o "$spv_path" 2>&1); then
            cleanup_prepared_source
            PREPARED_SOURCE=""
            echo "COMPILE FAILED"
            echo "$compile_err"
            FAIL=$((FAIL + 1))
            continue
        fi
        cleanup_prepared_source
        PREPARED_SOURCE=""

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
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
