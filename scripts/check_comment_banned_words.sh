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
    cat >&2 <<'GUIDANCE'

------------------------------------------------------------------------
How to remediate (read before editing):

These words flag comments that narrate the PREVIOUS STATE of the code
("X used to do Y", "previously Z", "no longer W"). The record of what
the code used to do is git history, not the source. So the fix is to
DELETE the historical clause, not to reword it into a synonym that
dodges the regex. A comment must describe only what the code does now
and why; if removing the history leaves nothing, delete the comment.

  BAD  (history):     // Previously this folded to v*v, which was wrong.
  BAD  (dodge):       // Earlier this folded to v*v, which was wrong.
  GOOD (current why): // Vec bases are skipped: there is no correct
                      // scalar fold for `vec ** k`.

One genuine false positive: "Used to <verb>" meaning "serves to <verb>"
(a purpose statement, not history). Reword to a plain verb — "Used to
build the index" -> "Builds the index" — so the heuristic stays simple.
------------------------------------------------------------------------
GUIDANCE
    exit 1
fi
exit 0
