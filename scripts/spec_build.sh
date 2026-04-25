#!/usr/bin/env bash
# Build the spec into the playground's static asset tree so it serves
# at /spec/ on the deployed playground domain.
#
# Steps:
#   1. Split SPECIFICATION.md into per-chapter files under docs/src/.
#   2. mdbook build → playground/dist/client/spec/ (a subpath of the
#      Cloudflare Worker's `assets.directory`).
#
# Run from any cwd; resolves paths relative to the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 scripts/split_spec.py
mdbook build docs --dest-dir ../playground/dist/client/spec
echo "built → playground/dist/client/spec/index.html"
