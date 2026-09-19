#!/usr/bin/env bash
# Same build/run layout as scripts/play.sh; this program needs generated stages.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export RUST_MIN_STACK=${RUST_MIN_STACK:-67108864}
skip=false
compile_only=false
args=()
for arg in "$@"; do
    case "$arg" in
        --skip-build) skip=true ;;
        --compile-only) compile_only=true ;;
        *) args+=("$arg") ;;
    esac
done
if [[ $skip == false ]]; then
    cargo build --release --manifest-path "$root/Cargo.toml" -p wyn
    cargo build --release --manifest-path "$root/extra/viz/Cargo.toml"
fi
out="$root/tmp/fluid-simulation"
mkdir -p -- "$out"
"$root/target/release/wyn" build "$root/fluid-simulation" --graphics -o "$out/fluid.spv"
launch=(python3 "$root/fluid-simulation/launch.py" "$out/fluid.spv")
if [[ $compile_only == true ]]; then
    "${launch[@]}" --prepare-only
else
    "${launch[@]}" "${args[@]}"
fi
