# Array-copy reproducer (compile only)

Originally reproduced on compiler source revision `5273b8d0`.
**Fixed on updated master `a066284b`**, including `32a6abc0` ("Schedule old
element reads before owned array updates"). Verified in normal and `-O` builds:
one local array, no whole-array value assignments in the update loop, with both
old element reads before either write. Both outputs pass CPU-only Naga validation.
No shader execution, graphics initialization or browser use is needed.

From the repository root:

```sh
target/release/wyn build fold/test/array_update_copy.wyn -t wgsl --direct -o /tmp/wyn-array-update-copy.wgsl
```

The seven-line source reduces the cloth edge solve to two indexed updates.
The older compiler's generated update loop contained (SSA names may change):

```wgsl
var v34_1: array<vec4<f32>, 32> = v27_1;
v34_1[v31_1] = (v32_1 + 1.0f);
// Read b from the old state, then write the new state.
let v37_1: vec4<f32> = v27_1[v36_1];
v34_1[v36_1] = (v37_1 - 1.0f);
let _copy3: array<vec4<f32>, 32> = v34_1;
v27_1 = _copy3;
```

This demonstrates whole-array **value copies in emitted WGSL**, not measured
physical memory traffic. A downstream driver may optimize some copies; no
GPU performance claim has been validated.

The old state remains live for the second indexed read. If `a == b`, the
result must be `old[b] - 1`, so blindly mutating the original array before
that read would be incorrect. An optimization would need to load both old
values first, or otherwise prove that overwriting `a` cannot affect the
later read. The cloth's real edge endpoints are distinct, but this reduced
program does not encode that fact.

`array_update_in_place.wyn` is a control case with one indexed update per
iteration. It emits a local element assignment without whole-array copies
inside the update loop. This narrows the remaining issue to old/new state
liveness across multiple updates, rather than all array-valued loops.

## Follow-up: branch-dependent traversal

`edge_copy_repro.wyn` reduces the remaining Fold transfers to nested loops
whose traversal direction is selected by a scalar conditional. The original
paired-update case passed on `a066284b`, while this case still copied arrays.

```sh
target/release/wyn build fold/test/edge_copy_repro.wyn -O -t wgsl --direct -o /tmp/reverse-updates.wgsl
```

The local compiler fix propagates a bound producer's choice depth into its
scheduling rank. Previously, the binding had rank zero even if materializing
it executed a choice; sorting could move the first update before a later
old-element read. The binding remains a separate emission anchor. No aliasing
or storage-liveness checks are relaxed.

With the fix, this reproducer and the complete Fold shader emit no local
whole-array value assignments and pass CPU-only Naga validation. Both also
compile to SPIR-V and pass `spirv-val`. The native solver has subsequently been executed on a GPU with explicit scratch storage.
The compiler regression fixture is
`tests/compiler/cases/paired_updates_scalar_branch.wyn`.
Two CPU-only Naga checks also pass: both old elements are read before either
write, and the same ordering preserves semantics when the indices are equal.
The compiler fix is committed on local master at `a9643b12`. The regression
fixture metadata was corrected in `f5a35661` and passes for both targets.
The native solver now runs using explicit serial scratch storage; see the
project README for execution comparisons against upstream.
