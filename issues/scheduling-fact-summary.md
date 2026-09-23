# Scheduling fact summaries

Scheduling imports immutable source summaries through its existing native egglog
update. Rust computes device availability, mutable backing propagation, and
duplication cost (saturating above the eight-unit budget). Egglog retains the
effect checks, rematerialization decisions, scalar grouping, phase placement,
allocation, and output attachment in the existing scheduling egraph.

The source worklist counts repeated expression uses, avoids charging for a
stored producer again, treats view lengths as metadata reads, and handles
recursive dependencies with a bounded fixed point. Scheduling receives host
expression facts and one availability/budget summary per live operation instead
of importing the evaluation graph and cost lists.

Other import simplifications:

- Scalar classification does not carry types; buffer element types remain
  allocation metadata.
- Output binding facts carry output IDs; names are reconstructed during readout.
- Type strides are imported only for referenced result and scratch types.
- Resource leaves are imported directly for tuple inputs.
- Remove unused host-handoff facts, duplicate defaults, and the source-position
  fallback. Production schedules provide positions for every live operation.

Content versus descriptor dependencies, capacity versus live length, value
identity versus storage backing, and operation versus body effect proofs remain
distinct because they affect scheduling correctness.

## Tinyporto measurement

Measured on the frozen 22-source-file snapshot at
`tmp/tinyporto-fusion-recheck/tinyporto-snapshot`, without touching the sibling
checkout. Both compilers used the same input and options: release build,
`--graphics --target wgsl --max-warnings 0 --verbose --output-mir`, without `-O`.
Three interleaved process runs per compiler, after builds and tests stopped.
Before is the saved Stage 3 compiler at `6819b085` with the same fine-grained
timers; after includes scheduling cleanup. Timers remain separate uncommitted
changes.

| Measurement | Before | After | Change |
| --- | ---: | ---: | ---: |
| Whole compile | 3.99 s | 3.65 s | -8.5% |
| Egglog total | 2726.47 ms | 2341.47 ms | -14.1% |
| Scheduling total | 953.64 ms | 540.98 ms | -43.3% |
| Scheduling rule load | 52.16 ms | 48.43 ms | -7.2% |
| Scheduling fact import | 186.53 ms | 133.59 ms | -28.4% |
| Scheduling rule execution | 677.97 ms | 322.46 ms | -52.4% |

Whole-compile samples were 3.89, 4.07, 3.99 seconds before and 3.72, 3.65,
3.61 seconds after. Each table cell is an independent median; nested spans are
inclusive. Arithmetic equality saturation remains the largest egglog pass at
1304.06 ms. Some unaffected pass timings increased in this sample, including
SSA lowering and WGSL lowering; these measurements do not attribute those
differences to a specific cause.

Reproduction script: `tmp/scheduling-summary/benchmark.ps1`. Full per-run values,
medians, logs, generated artifacts, and compiler copy are under
`tmp/scheduling-summary/`. These local artifacts are not committed.

## Validation

- `cargo check -p wyn-core -p wyn` and release build passed.
- Package tests: 1,440 passed, 18 ignored; no failures.
- Testfiles: 110 SPIR-V and 109 WGSL passed; one expected WGSL skip.
- Generated Rust host GPU tests passed on Vulkan / Radeon RX 580, exercising
  batching, filtering, filter post-maps, indirect commands, scalar epilogues,
  capture handoff, and readback boundaries in both shader formats.
- Frozen tinyporto validates in SPIR-V and WGSL and retains 22 entry points
  (14 compute, four vertex, four fragment).
- All 117 MIR function bodies match the baseline after normalizing function
  numbering/order and the swap of internal scratch bindings 36 and 37.
- The host program matches after normalizing those same scratch bindings and
  the shader filename; dispatch order and external bindings are unchanged.
- Formatting and whitespace checks passed.
