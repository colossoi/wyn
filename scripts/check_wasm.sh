#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for crate in wyn-wasm; do
    cargo check --manifest-path "$repo_root/$crate/Cargo.toml" \
        --target wasm32-unknown-unknown --locked "$@"
done
