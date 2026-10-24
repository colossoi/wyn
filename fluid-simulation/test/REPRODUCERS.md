# Current WIP validation: 2026-10-21

The 4097-slot octree reproducer completes on NVIDIA GeForce GTX 1660 Ti
with correct split and merge results on both SPIR-V and WGSL. It is included
in `spatial_check.py`. A sandbox without GPU device access instead selects
llvmpipe, where pipeline creation stalls. Run GPU acceptance tests with
host GPU access; `viz pipeline --verbose` reports the adapter.

The checkout compiler passes the spatial, linked-neighbor, 76-query traversal,
neighbor-based DFSPH reference, and indexed-renderer pixel checks on both
backends. Four full simulator frames produce identical SPIR-V/WGSL positions
and velocities. A 120-frame WGSL run through block contact preserves finite
values and simulation bounds for all 4,320 particles. These runs use 320×240
render targets and do not establish 100k-particle performance.

The core suite passes 1,509 tests with 14 ignored; the remaining workspace
passes 148 tests, the viewer passes 10, and native `wyn-wasm` passes 3.
The WebAssembly target check and simulator SPIR-V validation also pass.
Both testfile gates pass: 124 SPIR-V cases and 123 WGSL cases, with the
linked-SPIR-V helper case skipped by the WGSL gate. These include the local
untracked `kogepan.wyn` playground shader. Workspace all-target builds,
formatting, and diff whitespace checks pass. The all-target build retains
two existing analyzer deprecation warnings for `DocumentSymbol::deprecated`.

# Batched traversal acceptance probes: 2026-10-20

`batched_traversal.wyn` expresses a breadth-first search using nested maps,
prefix scans, and scatters inside a loop. It uses a fixed 73-node octree and
returns ordered leaf IDs plus a count for each independent query.
`check_batched_traversal.py` compiles the Wyn and compares GPU results against
an independent Python BFS for 76 pruning patterns. Python is only the test
harness and reference; the GPU algorithm is entirely Wyn.

Run from `/home/jonah/wyn`:

```sh
python3 fluid-simulation/test/check_batched_traversal.py
```

The harness defaults to `target/release/wyn`; set `WYN` to compare another
compiler. Its first check is `repro_loop_scatter_snapshot.wyn`. The installed
compiler returned `[0,11,0,0]` instead of `[10,11,12,0]` for its first row,
on both GPU backends. Local scatter results were loaded before the writes.
The checkout fix loads the completed buffer, and this small numerical check
passes on both SPIR-V and WGSL on NVIDIA GTX 1660 Ti.

The planner selects map execution through `MapExecution`. Local recipes use
`LocalExecution(operation, calling_scope)`, so the same helper can execute in
an invocation callback or in a workgroup without a lowering-time fallback.
Ordinary maps share the `elements` launch path; segmented output streaming
remains a distinct recipe. Default widths live in `schedule.egg`.

`planning/execution.egg` separates eligibility from scope selection. Lane
dependence follows value uses and captures; direct calls inherit the caller's
scope and element callbacks execute at invocation scope. Recursive expansion
is excluded separately. `InlineCall` selects lexical emission for workgroup
calls, including their invocation-local callbacks, preserving physical buffer identity.

`planning/boundaries.egg` propagates storage demand by tuple field through
projections, captures, and loop backedges. It selects each boundary field's
representation and transfer strategy. Reused loop destinations use staging to
preserve simultaneous assignment; fresh branch destinations copy directly.
The materializer consumes that plan through one recursive conversion path,
using the common array-copy primitive and barriers around whole transfers.

`check_linked_neighbors.py` covers the actual Wyn linked-node implementation:
root-only and mixed-depth trees, leaf bounds, inactive siblings, and narrow
phase results compared with an all-pairs geometric oracle. The implementation
generates and sorts Warren–Salmon keys, precomputes bounds and first-child links,
and traverses with upstream's 128-entry parent queue and 32-child batches.
`Spatial.prepare` uses that path.

Before the scope consolidation, the 76-query traversal and linked-neighbor
checks passed on both backends. The simulator completed eight frames with
finite positions and velocities for 4,320 particles; keeping loop state in
shared storage brought the broad neighbor pass to about 10.8 ms. These results
predate the consolidation. After the scope and boundary-plan cleanup, the
scatter, array-swap, inferred-layout, 76-query traversal, and linked-neighbor
GPU checks pass on both SPIR-V and WGSL. An eight-frame headless run at
640×480 on GTX 1660 Ti completed with finite outputs. Averaging frames 2–7,
leaf traversal took 3.63 ms, particle-neighbor construction 9.61 ms, rendering
12.92 ms, and all GPU passes together 29.93 ms. These are short-run GPU
timestamps, not end-to-end desktop frame times. The core suite passed 1,496
tests with 14 ignored.
This is not a claim of 100k-particle readiness or complete upstream parity.

Commit validation on 2026-10-20: workspace all-target build and formatting
checks passed; workspace tests passed 1,645 with 16 ignored. The Bash
testfile gates passed 124 SPIR-V cases and 123 WGSL cases (one linked-SPIR-V
helper case skipped by the script). These counts include the local untracked
`kogepan.wyn` playground shader. The `viz` tests passed 10, native `wyn-wasm`
tests passed 3, and `scripts/check_wasm.sh` passed. The generated simulator
also passed `spirv-val`.

Useful checks (checkout compiler; NVIDIA access required for GPU tests):

```sh
RUST_MIN_STACK=16777216 cargo test -p wyn-core --lib
python3 fluid-simulation/test/check_batched_traversal.py
python3 fluid-simulation/test/check_linked_neighbors.py
```

`repro_branch_array_layout.wyn` isolates an inferred queue array crossing a
conditional loop-state boundary. Its expected output is `[6,10]`; the failure
was a missing selected control-boundary layout.

# Retest: installed compiler 2026-10-20 17:08 EDT

`repro_cooperative_leaf.wyn` builds and returns [496,1520,2544,3568] on
NVIDIA GTX 1660 Ti. SPIR-V still contains a serial sum per map invocation,
with no Workgroup storage, control barriers, or subgroup operations. This
is a scheduling observation rather than a numerical failure.

`nested_candidate_ranks.wyn` translates the discussed nested map/scan/reduce
example into Wyn. Compilation fails with `SSA lowering: local slot output
has no planned static capacity` (function `_w_lambda_3`, 2 arguments).
The source and expected result description are retained without a workaround.

The current full simulator builds in 3.12 seconds and completes eight headless
NVIDIA frames. All 4,320 positions and velocities are finite. Summed GPU pass
times average 46.449 ms over frames 2–7; this is not end-to-end desktop FPS
or a controlled comparison against earlier source/compiler revisions.
Raw timings: `../profiles/2026-10-20-retest.json`.

# Compiler reproducer: loop-carried replicate

Using the installed `/home/jonah/.cargo/bin/wyn` on 2026-09-18:

```sh
wyn check fluid-simulation/test/replicate_repro.wyn
wyn build fluid-simulation/test/replicate_repro.wyn -o /tmp/repro.spv
wyn build fluid-simulation/test/replicate_repro.wyn -t wgsl -o /tmp/repro.wgsl
```

Type checking passes. Both builds fail during egglog-to-SSA lowering with
`block BlockId(9) argument types differ`: the loop argument contains an array
storage view with `SizePlaceholder`, while the other edge supplies a composite
array with `Size(4)`.

`replicate_literal_control.wyn` changes only the initializer to
`[-1i32,-1i32,-1i32,-1i32]` and compiles. Expected output for `n=2` is
`[0,1,-1,-1]`; for `n=0`, all four elements remain `-1`.

This reproducer demonstrates a compiler rejection, not silent wrong output.
The larger neighbor traversal also needs separate runtime verification.

# Separate output-size miscompilation

```sh
wyn build fluid-simulation/test/array_return_repro.wyn -o /tmp/array-return.spv
```

This three-line program declares a `[4]i32` result and should return
`[-1,-1,-1,2]`. Compilation succeeds, but the emitted output binding has
`length: { kind: fixed, bytes: 4 }`, rather than 16 bytes. The larger
`neighbors.wyn` test similarly publishes 4 bytes for its `[98]i32` result.
This is distinct from the loop-carried `replicate` lowering error above.

## Retest after compiler update

The installed compiler dated 2026-09-18 19:39 now compiles both reproducers.
SPIR-V and WGSL readbacks match the expected results for `n=2` and the
array-return case. Output allocations are correctly 16 bytes. The neighbor
traversal's result allocation is now correctly 392 bytes (`98 * 4`).

# Boolean reduction storage readback

```sh
wyn build fluid-simulation/test/bool_reduce_repro.wyn -o /tmp/bool-reduce.spv
extra/viz/target/release/viz validate /tmp/bool-reduce.spv
```

Compilation succeeds, but validation rejects LogicalOr with a bool left operand
and a u32 right operand. The materialized boolean reduction result is loaded
as u32 without conversion back to bool. This also occurred in the combined
spatial pipeline's overflow aggregation. The boolean expression is retained in the port; no integer-flag workaround
is applied.

## No compiler workarounds

The spatial pipeline retains boolean overflow flags, replicate-initialized local
arrays, and a shared integration map with tuple outputs. Earlier temporary
source workarounds were removed at the user's request. The isolated replicate
reproducer is fixed, but `neighbors.wyn` still exposes a more complex lowering
failure. `step.wyn` / `step_neighbors.wyn` expose the fusion-planning failure.
The spatial pipeline is not yet ready to replace the running simulator.

## Retest with installed compiler dated 2026-09-18 20:27

- `bool_reduce_repro.wyn`: both targets compile; SPIR-V validates; GPU
  readbacks for all-zero and nonzero inputs pass in both targets.
- `neighbors.wyn`: both targets compile and SPIR-V validates. GPU
  output matches the expected 98-element result in both targets.
- `fusion_repro.wyn`: still fails in both targets with
  `fusion body composition disagrees with its legality facts`
  (`family 1, OperationId(1) -> OperationId(2)`).

No source workarounds were introduced for this retest.

## Retest with installed compiler dated 2026-09-18 20:36

`fusion_repro.wyn` now builds in both targets. Both the pairwise and tiled
neighbor-list DFSPH fixtures compile and match the scalar reference on the GPU
in SPIR-V and WGSL, including both wall collisions. The shared integration map
is unchanged; no workaround was applied. All three isolated blockers reported
above are resolved; the complete spatial viewer still needs integration checks.
