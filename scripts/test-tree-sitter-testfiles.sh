#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

cargo test \
  --manifest-path "$repo_dir/extra/tree-sitter-wyn/Cargo.toml" \
  parse_all_repository_testfiles
