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
