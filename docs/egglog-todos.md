# Egglog follow-ups

- [ ] Continue auditing direct fact import. Keep traversal/interner caches only where they prevent repeated work or serve later compiler passes; avoid Rust collections that merely stage relations. Review shared dependency/effect summaries, region/symbol lookup needs, and scheduling inputs. Preserve linear or near-linear work when moving deduplication to egglog.
  - [x] Read expression and arithmetic facts directly into an `EGraph::update` batch. An egglog function table maintains canonical source roots until extraction. No fact text, command staging, or per-root name lookup remains in this path.
  - [ ] Apply direct insertion to fusion and scheduling inputs. Their commands are consumed during execution; no combined export program is retained.
- [ ] Determine whether fusion `plan::Step` and its `Vec`/`BTreeMap` readout are necessary. Can `FusionStep`, `StepUse`, and `StepObserved` be read directly into the final body construction or IR records? Preserve ordered application and avoid one full table scan per fusion step.

- [ ] Replace the fixed four algebra rounds with a measured exploration policy. Constant folding and arithmetic identities must continue to saturation independently of that budget.
