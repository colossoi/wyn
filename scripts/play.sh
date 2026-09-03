#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/play.sh EXAMPLE [--skip-build] [--output-directory DIR] [--max-frames N]

Compile and run one local playground example. EXAMPLE may be a .wyn path or a
bare name from testfiles/playground. Bare main_image examples automatically
receive scripts/playground_image_header.wyn.
EOF
}

example=''
skip_build=false
output_directory=''
max_frames=0

while (($#)); do
    case "$1" in
        --skip-build)
            skip_build=true
            shift
            ;;
        --output-directory)
            (($# >= 2)) || { echo 'Missing value for --output-directory' >&2; usage >&2; exit 2; }
            output_directory=$2
            shift 2
            ;;
        --max-frames)
            (($# >= 2)) || { echo 'Missing value for --max-frames' >&2; usage >&2; exit 2; }
            max_frames=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n $example ]]; then
                echo "Unexpected argument: $1" >&2
                usage >&2
                exit 2
            fi
            example=$1
            shift
            ;;
    esac
done

[[ -n $example ]] || { usage >&2; exit 2; }
[[ $max_frames =~ ^[0-9]+$ ]] || { echo '--max-frames must be a non-negative integer' >&2; exit 2; }

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd -- "$script_directory/.." && pwd)
playground_directory="$workspace/testfiles/playground"
playground_header="$workspace/scripts/playground_image_header.wyn"

source_candidate=$example
if [[ ! -f $source_candidate && $example != */* ]]; then
    [[ $source_candidate == *.wyn ]] || source_candidate+=.wyn
    source_candidate="$playground_directory/$source_candidate"
fi
[[ -f $source_candidate ]] || { echo "Playground example not found: $example" >&2; exit 1; }
source_path=$(cd -- "$(dirname -- "$source_candidate")" && pwd)/$(basename -- "$source_candidate")
[[ $source_path == *.wyn ]] || { echo "Playground examples must use the .wyn extension: $example" >&2; exit 1; }

if [[ -z $output_directory ]]; then
    artifact_directory="$workspace/tmp/playground"
elif [[ $output_directory = /* ]]; then
    artifact_directory=$output_directory
else
    artifact_directory="$PWD/$output_directory"
fi

name=$(basename -- "$source_path" .wyn)
spv_path="$artifact_directory/$name.spv"
descriptor_path="$artifact_directory/$name.json"
source_directory=$(dirname -- "$source_path")
viz_config="$source_directory/$name.viz.json"
wyn_binary="$workspace/target/release/wyn"
viz_binary="$workspace/extra/viz/target/release/viz"
export RUST_MIN_STACK=${RUST_MIN_STACK:-67108864}

cd -- "$workspace"
if [[ $skip_build == false ]]; then
    echo 'Building Wyn and viz in release mode...'
    cargo build --release --package wyn --bin wyn
    cargo build --release --manifest-path extra/viz/Cargo.toml
else
    for binary in "$wyn_binary" "$viz_binary"; do
        [[ -f $binary ]] || { echo "Release binary not found: $binary (rerun without --skip-build)" >&2; exit 1; }
    done
fi

mkdir -p -- "$artifact_directory"
compile_source=$source_path
prepared_source=''
prepared_source_base=''
cleanup() {
    if [[ -n $prepared_source ]]; then
        rm -f -- "$prepared_source"
    fi
    if [[ -n $prepared_source_base ]]; then
        rm -f -- "$prepared_source_base"
    fi
}
trap cleanup EXIT

if ! grep -Eq '^entry ' "$source_path"; then
    prepared_source_base=$(mktemp "$source_directory/.run-$name.XXXXXX")
    prepared_source="$prepared_source_base.wyn"
    mv -- "$prepared_source_base" "$prepared_source"
    prepared_source_base=''
    {
        cat -- "$playground_header"
        printf '\n'
        cat -- "$source_path"
    } >"$prepared_source"
    compile_source=$prepared_source
fi

echo "Compiling $source_path"
"$wyn_binary" build "$compile_source" --graphics --direct -o "$spv_path"
cleanup
prepared_source=''

for artifact in "$spv_path" "$descriptor_path"; do
    [[ -f $artifact ]] || { echo "Compiler did not produce expected artifact: $artifact" >&2; exit 1; }
done

viz_arguments=(pipeline "$spv_path")
[[ -f $viz_config ]] && viz_arguments+=(--config "$viz_config")
((max_frames > 0)) && viz_arguments+=("--max-frames=$max_frames")

echo "SPIR-V:     $spv_path"
echo "Descriptor: $descriptor_path"
echo "Running $name..."
"$viz_binary" "${viz_arguments[@]}"
