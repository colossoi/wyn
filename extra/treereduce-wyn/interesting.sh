#!/usr/bin/env bash
# Interestingness check for the demo bug.
#
# treereduce-wyn feeds us the reduced candidate either via stdin or by
# substituting `@@` with a tempfile path. We run `wyn compile` on the
# candidate and exit 0 IFF the compiler still panics with the
# distinctive demo-bug message — that's how treereduce knows the
# reduction still triggers the same bug.
#
# Any other outcome (clean compile, different panic, parse error,
# whatever) is reported as uninteresting so treereduce tries a
# different reduction.

set -euo pipefail

WYN="${WYN:-./target/release/wyn}"
candidate="${1:--}"

if [[ "$candidate" == "-" ]]; then
  tmp=$(mktemp /tmp/tr_wyn_XXXXXX.wyn)
  trap 'rm -f "$tmp"' EXIT
  cat > "$tmp"
  candidate="$tmp"
fi

# Run the compiler. We don't care about stdout — only the stderr panic.
# The sentinel string identifies the *specific* bug we're hunting.
output=$("$WYN" compile --fill-holes "$candidate" -o /dev/null 2>&1 || true)

if grep -q "demo-bug: f32.sqrt lowering intentionally broken" <<< "$output"; then
  exit 0
else
  exit 1
fi
