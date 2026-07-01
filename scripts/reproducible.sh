#!/usr/bin/env bash
# reproducible.sh — determinism probe for the wyn compiler.
#
# Reads a Wyn program on stdin, compiles it N times in independent OS processes
# (so each run gets a fresh, independent HashMap RandomState seed), and reports
# whether every IR dump is byte-identical across all runs.
#
# Dumps checked: --output-tlc, --output-mir, and the pipeline descriptor .json
# (auto-emitted next to the .spv). The .spv binary itself is intentionally NOT
# compared — SPIR-V word-ids come from dedup caches and are out of scope; we
# only care that the IR dumps are reproducible.
#
# Exit: 0 = reproducible, 1 = non-reproducible (a dump differed), 2 = error
#       (compile failed, or the compiler binary is missing).
#
# Env knobs:
#   RUNS=<n>     number of compiles to compare (default 12)
#   WYN=<path>   compiler binary (default ./target/debug/wyn)
#   TARGET=spirv|wgsl  (default spirv)
#
# Usage:  ./reproducible.sh < program.wyn
#         echo 'entry e() i32 = 1' | ./reproducible.sh
set -uo pipefail

RUNS="${RUNS:-12}"
WYN="${WYN:-./target/debug/wyn}"
TARGET="${TARGET:-spirv}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
src="$work/input.wyn"
cat > "$src"

if [[ ! -f "$WYN" ]]; then
  echo "compiler not found at $WYN — building (cargo build)..." >&2
  cargo build -q || { echo "build failed" >&2; exit 2; }
fi

# The dump kinds we compare. `json` is the descriptor written beside the .spv.
kinds=(tlc mir json)

compile_run() {
  local outdir="$1"; mkdir -p "$outdir"
  "$WYN" compile "$src" \
    --output-tlc "$outdir/dump.tlc" \
    --output-mir "$outdir/dump.mir" \
    -t "$TARGET" \
    -o "$outdir/dump.spv" \
    > "$outdir/stdout.txt" 2> "$outdir/stderr.txt"
  echo $? > "$outdir/exit"
}

compile_run "$work/run0"
if [[ "$(cat "$work/run0/exit")" != "0" ]]; then
  echo "RESULT: ERROR — compile failed (exit $(cat "$work/run0/exit"))"
  sed 's/^/  | /' "$work/run0/stderr.txt" >&2
  exit 2
fi

differ=0
declare -A diffkind
for ((i = 1; i < RUNS; i++)); do
  compile_run "$work/run$i"
  if [[ "$(cat "$work/run$i/exit")" != "0" ]]; then
    echo "RESULT: ERROR — compile was non-deterministic in *success*: run $i failed but run0 succeeded"
    sed 's/^/  | /' "$work/run$i/stderr.txt" >&2
    exit 1
  fi
  for k in "${kinds[@]}"; do
    f0="$work/run0/dump.$k"; fi_="$work/run$i/dump.$k"
    e0=0; ei=0
    [[ -e "$f0" ]] || e0=1
    [[ -e "$fi_" ]] || ei=1
    [[ $e0 == 1 && $ei == 1 ]] && continue        # neither run produced it
    if [[ $e0 != "$ei" ]] || ! cmp -s "$f0" "$fi_"; then
      differ=1; diffkind[$k]=1
    fi
  done
done

if [[ "$differ" == "0" ]]; then
  echo "RESULT: REPRODUCIBLE across $RUNS runs (tlc, mir, annotated, descriptor all identical)"
  exit 0
fi

echo "RESULT: NON-REPRODUCIBLE across $RUNS runs — differing dump(s): ${!diffkind[*]}"
for k in "${kinds[@]}"; do
  if [[ "${diffkind[$k]:-0}" == "1" ]]; then
    echo "----- sample diff in .$k (run0 vs first differing run) -----"
    for ((i = 1; i < RUNS; i++)); do
      if ! cmp -s "$work/run0/dump.$k" "$work/run$i/dump.$k"; then
        diff "$work/run0/dump.$k" "$work/run$i/dump.$k" | head -40
        break
      fi
    done
  fi
done
exit 1
