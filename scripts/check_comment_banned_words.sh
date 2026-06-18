#!/usr/bin/env bash

# Scan Rust source comments for banned historical-reference phrases.
# Comments must describe current behavior; history belongs in git/PR
# descriptions. The output is `file:line: ...comment text...` per hit,
# suitable for grep-driven editor jumping.
#
# Exit status:
#   0 — no hits
#   1 — one or more hits
#
# Override the search root (default: workspace crates):
#   $ ./scripts/check_comment_banned_words.sh wyn-core/src/egir

set -euo pipefail

BANNED='previously|used to|was previously|no longer|originally|in v1|in v2|before, this'
ROOTS=("$@")
if [ ${#ROOTS[@]} -eq 0 ]; then
    ROOTS=(wyn-core/src wyn/src wyn-analyzer/src wyn-wasm/src wyn-pipeline-descriptor/src)
fi

# Only match Rust line comments (`//`, `///`, `//!`). Block comments
# (`/* … */`) and string literals aren't matched — they're rare in
# this codebase. Case-insensitive on the banned words.
PATTERN="^\\s*//.*\\b(${BANNED})\\b"

# Aggregate output ourselves so the exit code reflects "any hits"
# rather than grep's per-file 0/1 dance.
hits=$(grep -rniE --include='*.rs' "$PATTERN" "${ROOTS[@]}" || true)

if [ -n "$hits" ]; then
    echo "$hits"
    exit 1
fi
exit 0
