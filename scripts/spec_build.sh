#!/usr/bin/env bash
# Render SPECIFICATION.md into an mdBook served at /spec/.
#
# Steps:
#   1. Split SPECIFICATION.md into per-chapter files under docs/src/.
#   2. mdbook build → <output-dir>.
#
# The output directory is the playground's static asset tree (its
# public/spec/), passed as an argument since the playground lives in a
# separate repo and consumes this script through the wyn submodule.
#
#   spec_build.sh <output-dir>     # or set SPEC_DEST=<output-dir>
#
# A relative <output-dir> resolves against the invoking CWD, not the repo
# root. Run from any cwd.
set -euo pipefail

dest="${1:-${SPEC_DEST:-}}"
if [[ -z "$dest" ]]; then
  echo "usage: spec_build.sh <output-dir>   (or set SPEC_DEST)" >&2
  exit 2
fi
# Resolve a relative dest against the invoking CWD before we cd to the repo
# root, so the path means what the caller expects.
case "$dest" in
  /*) ;;
  *) dest="$(pwd)/$dest" ;;
esac

cd "$(dirname "$0")/.."

python3 scripts/split_spec.py
mdbook build docs --dest-dir "$dest"
echo "built → $dest/index.html"
