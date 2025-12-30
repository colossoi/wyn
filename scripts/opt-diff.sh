#!/bin/bash
# opt-diff.sh - Compare SPIR-V before/after individual optimization passes
#
# Usage: ./scripts/opt-diff.sh <file.wyn>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <file.wyn>"
    exit 1
fi

WYN_FILE="$1"
BASE=$(basename "$WYN_FILE" .wyn)
TEMP_DIR=$(mktemp -d)

trap "rm -rf $TEMP_DIR" EXIT

# Compile the wyn file
echo "Compiling $WYN_FILE..."
cargo run --release -q --bin wyn -- compile "$WYN_FILE" --partial-eval -o "$TEMP_DIR/original.spv"
spirv-dis "$TEMP_DIR/original.spv" > "$TEMP_DIR/original.spvasm"

# Optimization flags to try (excluding loop unrolling)
OPT_FLAGS=(
    "--ccp"
    "--cfg-cleanup"
    "--combine-access-chains"
    "--convert-local-access-chains"
    "--eliminate-dead-branches"
    "--eliminate-dead-code-aggressive"
    "--eliminate-dead-const"
    "--eliminate-dead-functions"
    "--eliminate-dead-inserts"
    "--eliminate-dead-variables"
    "--eliminate-insert-extract"
    "--eliminate-local-multi-store"
    "--eliminate-local-single-block"
    "--eliminate-local-single-store"
    "--if-conversion"
    "--inline-entry-points-exhaustive"
    "--local-redundancy-elimination"
    "--redundancy-elimination"
    "--scalar-replacement=100"
    "--simplify-instructions"
    "--ssa-rewrite"
    "--strength-reduction"
    "--vector-dce"
    "--workaround-1209"
    "--wrap-opkill"
)

ORIG_LINES=$(wc -l < "$TEMP_DIR/original.spvasm")
echo "Original: $ORIG_LINES lines"
echo ""

for FLAG in "${OPT_FLAGS[@]}"; do
    # Run spirv-opt with this single flag
    if spirv-opt "$FLAG" "$TEMP_DIR/original.spv" -o "$TEMP_DIR/optimized.spv" 2>/dev/null; then
        spirv-dis "$TEMP_DIR/optimized.spv" > "$TEMP_DIR/optimized.spvasm" 2>/dev/null

        OPT_LINES=$(wc -l < "$TEMP_DIR/optimized.spvasm")
        DIFF_LINES=$(diff "$TEMP_DIR/original.spvasm" "$TEMP_DIR/optimized.spvasm" | wc -l)

        if [ "$DIFF_LINES" -gt 0 ]; then
            echo "========================================"
            echo "FLAG: $FLAG"
            echo "Lines: $ORIG_LINES -> $OPT_LINES (diff: $DIFF_LINES lines changed)"
            echo "========================================"
            diff --color=always -u "$TEMP_DIR/original.spvasm" "$TEMP_DIR/optimized.spvasm" | head -100
            echo ""
            read -p "Press Enter to continue, 'q' to quit, 's' to skip rest... " response
            if [ "$response" = "q" ]; then
                exit 0
            elif [ "$response" = "s" ]; then
                break
            fi
        else
            echo "$FLAG: no change"
        fi
    else
        echo "$FLAG: failed (unsupported or error)"
    fi
done

echo ""
echo "Done."
