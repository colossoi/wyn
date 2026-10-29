#!/usr/bin/env bash
# Same build/run layout as scripts/play.sh; this program needs generated stages.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export RUST_MIN_STACK=${RUST_MIN_STACK:-67108864}
skip=false
compile_only=false
args=()
build_args=()
while (($#)); do
    case "$1" in
        --skip-build) skip=true ;;
        --compile-only) compile_only=true ;;
        --particles|--tree-capacity|--target|--gpu-cost-profile)
            build_args+=("$1" "$2"); shift ;;
        *) args+=("$1") ;;
    esac
    shift
done
if [[ $skip == false ]]; then
    cargo build --release --manifest-path "$root/Cargo.toml" -p wyn
    cargo build --release --manifest-path "$root/extra/viz/Cargo.toml"
fi
out="$root/tmp/fluid-simulation"
mkdir -p -- "$out"
shader=$(python3 "$root/fluid-simulation/build.py" --output "$out" "${build_args[@]}")
launch=(python3 "$root/fluid-simulation/launch.py" "$shader")
if [[ $compile_only == true ]]; then
    "${launch[@]}" --prepare-only
else
    "${launch[@]}" "${args[@]}"
fi
