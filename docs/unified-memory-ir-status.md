# Unified Memory IR — WIP status

Paused at the user's request on 2026-11-11. This checkpoint is an incomplete
implementation, not a validated release. It combines the previously staged
implementation and subsequent unstaged replacement work.

## Goal and architecture

Replace the separate source, computation, memory, planning, and reconstruction
paths with a common egglog statement/body representation. Fusion, motion,
distribution, scheduling, and allocation should transform that representation;
Rust imports TLC once and emits SSA at the end. Do not introduce semantic
egglog-to-Rust-to-egglog round trips or restore compatibility representations to
make old tests pass. Simplification and fewer representable illegal states remain
the criteria; production line count is supporting evidence, not proof.

The implementation follows Futhark's separation of logical array optimization
from explicit allocation, with shared lexical bodies, names, effects, and layouts.
Reference checkout: `/tmp/wyn-futhark-review`, revision
`964c8d1b79985831ee74e35ad882485fb0105d0f`.

## Implemented

- `from_tlc.rs` imports normalized scalar/product/array components directly into
  the common IR. The former source graph and LocalLayout-driven planning paths
  have been removed. Arrays of products use normalized components; ragged values
  carry offsets and payloads. Scratch arrays use ordinary uninitialized allocation.
- Shared schema-derived traversal, renaming, substitution, statement replacement,
  scope checking, effects, and alias analysis support inlining, motion, fusion,
  and coordinate distribution. These transformations run inside egglog.
- Scheduling and allocation elaborate the same bodies into launches, loops,
  allocations, views, copies, stores, barriers, and accumulator operations.
  Portable collective lowering uses ordinary statements. Final SSA and ABI
  emission read those bodies and preserve external buffer layouts.
- Recent fixes preserve launch-local intrinsics, implicit index bindings,
  coordinate permutation proofs, scratch-array shapes, and small-loop placement.
  Loop placement accounts for array storage and iteration work; invariant
  reductions can move outside loops before placement.

This does not establish complete correctness, performance parity, or a complete
type/ownership verifier. Compilation of the latest targeted core tests succeeds,
but significant test migration and end-to-end validation remain.

## Validation at pause

Recent targeted tests passed for coordinate permutations and overlapping-digit
rejection; loop placement and bounded backedges; nested captures; equal/unequal
logical-shape fusion; scratch dead arithmetic; authoritative dispatch readout;
shrinking-loop descriptors; independent rectangular outputs; collective dimension
metadata; and accumulator scope escapes.

The 1,219-test core run was interrupted for this pause. Its binary preceded the
latest fixes. It reported two failures before interruption: dispatch readout and
scratch-array import. Both subsequently passed targeted reruns after fixes.
There is no completed passing full core run. No claim is made that full workspace,
fixture, Wasm, shader, GPU, or numerical gates pass.

Selected local logs (not committed):

- `/tmp/wyn-coordinate-validation-141.log`: permutation/rejection test passed.
- `/tmp/wyn-core-validation-143.log`: interrupted broader run.
- `/tmp/wyn-structure-validation-146.log`: three structural regressions passed.
- `/tmp/wyn-scope-dimensions-150.log`: both tests passed.
- `/tmp/wyn-loop-validation-152.log`: both tests passed.

`git diff --check` passed at this checkpoint. No additional compilation or tests
were started after the pause request.

## Remaining work

1. Finish native atomic/subgroup integration. Concurrent combining currently
   falls back to compare-exchange; the shared addition matcher should generalize
   to supported native combiners. Preserve correct treatment of partial/trapping
   reducers and ordered effects. This next change was investigated, not written.
2. Migrate remaining tests that inspect deleted tables to actual logical bodies,
   scheduled launches, or emitted SSA. Preserve their semantic and performance
   assertions. Improved optimization needs stronger evidence, not weaker tests.
3. Resolve behavior exposed by those tests: dead accumulator/output elimination,
   compacted streams, multi-clause collective dispatch counts, and captured-total
   barriers are remaining areas to verify. Fully general abstract ragged domains
   are not demonstrated by the current lexical dense/offset-payload import path.
4. Complete the end-to-end correctness/round-trip audit and run full gates:
   formatting, workspace checks/tests, compiler fixtures, local-package checks,
   Wasm checks, and backend/numerical contracts. Separate baseline failures from
   introduced failures. Do not restore the old architecture as an adapter.

## Production line counts

Counts include newly added production files and exclude tests and documentation.
Measured by `/tmp/wyn-memory-production-count.py` before this documentation file:

| Comparison | Rust | egglog | Total |
| --- | ---: | ---: | ---: |
| This WIP versus parent `e1a75e47` | -809 | -683 | -1,492 |
| Whole branch versus original base `a066284b` | -1,553 | +77 | -1,476 |

These are net textual production counts, not a claim that all replacement
functionality is finished. The earlier tracked-only diff omitted new files and
must not be used to claim a roughly 10,000-line reduction.

## Resume context

Worktree: `/tmp/wyn-parallel-ir-spike`; branch: `codex/parallel-ir-spike`.
The shared checkout and submodules must not be edited. Prior durable research and
continuation notes remain in `/tmp/wyn-*-continuation-2026-11-10.txt` and
`/tmp/wyn-futhark-guidance-2026-11-10.txt`; older status entries there are historical.
The untracked `docs/memory_ir_examples.egg` is deliberately excluded from this
commit, following the earlier request not to stage examples.

Build environment: `CARGO_TARGET_DIR=/tmp/wyn-spike-target`,
`CARGO_INCREMENTAL=0`, `CARGO_PROFILE_TEST_DEBUG=0`,
`CARGO_PROFILE_DEV_DEBUG=0`. The temporary filesystem has little free space;
clean only task-owned build artifacts if needed.
