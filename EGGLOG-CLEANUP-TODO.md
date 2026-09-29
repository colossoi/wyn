# Egglog cleanup

- [x] Delete dormant ABI planning rules and declarations; retain consumed binding facts.
- [x] Delete unused scalar composition/argument round trips and facts.
- [x] Delete obsolete fusion analyses and unused domain/dependency distinctions.
- [x] Restrict final internal-operand analysis to surviving fusion plans.
- [x] Share arithmetic speculation-safety classification between importers.
- [x] Keep losing extraction candidates out of the selected DAG.
- [x] Insert topology and initial fusion-round facts through the native API.
- [x] Run compiler tests/checks and compare tinyporto output and timing.

## Implementation notes

- Retained input-storage bindings and explicit output bindings used by SSA publication.
- Removed dormant ABI rules, their readout consumers, and write-only grid, slot, and scalar readout facts.
- Removed fusion purity/callee and dependency representations with no production inputs, including their always-false memory blocker.
- Preserved structural planning policies, context-local scalar optimization, and distinct safety properties.
- Candidate extraction uses temporary native TermDags; only the winning extraction enters the DAG retained for placement and SSA.

## Validation

- Core library: 1,288 passed, 14 ignored, no failures.
- Focused egglog suite: 88 passed.
- WASM: `cargo check --manifest-path wyn-wasm/Cargo.toml --target wasm32-unknown-unknown` passed.
- Release CLI build passed; no compiler warnings in the validation logs.
- Tinyporto SPIR-V and Rust wrapper are byte-for-byte identical to the committed baseline: 156 calls, 62 helpers, 17 compute and 20 graphics entries, 471,392 SPIR-V bytes.
- SPIR-V validation passed.
- Tinyporto compile: 20.55 s (baseline 20.81 s); egglog: 19.455 s (baseline 19.701 s). Single runs, not evidence of a significant speedup.
- Code diff: 507 net lines removed (506 production lines and one test line). Changes remain uncommitted.
- Comparison artifacts: `/tmp/wyn-tinyporto-comparison.9zwpt3/cleanup/`.
