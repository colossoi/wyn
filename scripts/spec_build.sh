#!/usr/bin/env bash
# Build the spec into the playground's static asset tree so it serves
# at /spec/ in every mode:
#   * npm run dev      — Vite serves from playground/public/
#   * wrangler dev     — Cloudflare assets reads playground/dist/client/,
#                        which react-router build copies public/ into
#   * production       — same as wrangler dev
#
# Steps:
#   1. Split SPECIFICATION.md into per-chapter files under docs/src/.
#   2. mdbook build → playground/public/spec/.
#
# Run from any cwd; resolves paths relative to the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 scripts/split_spec.py
mdbook build docs --dest-dir ../playground/public/spec
echo "built → playground/public/spec/index.html"
