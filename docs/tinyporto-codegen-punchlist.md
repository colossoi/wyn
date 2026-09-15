# Tinyporto code generation punchlist

Sources: `../tinyporto/repro/hand_codegen/` and the related standalone camera
and capture-fold reproducers. Baseline inspection: September 15, 2026.

- [x] Verify capture-fold sharing. The original conditional-fold regression
  was already fixed by `78ea2a52`. A later serial helper still blocked sharing
  in the full application and `repro/capture_fold_per_lane/standalone.wyn`.
  That case now shares the singleton producer too.
- [x] Remove unused fixed-array state and copy loops from scalar/item consumers
  in `dead_arrays.wyn` and `dead_arrays_scalar.wyn`.
- [x] Share repeated pure clamp calls in `repeated_helpers.wyn`.
- [x] Verify invariant camera work using `world_to_clip_loop_invariant.wyn`.
  Existing loop hoisting works; identical pure helper calls now share results.
- [x] Reproduce and remove the redundant camera copy dispatch. The saved
  `copy_prepass.wyn` output alone did not reproduce it; the new camera fixture
  does. Its two compute prepasses become one.
- [x] Fix the missing SPIR-V entry-point count-buffer interface in
  `filter_scan.wyn`.
- [x] Scan contiguous filter tiles cooperatively and remove the separate
  add-offsets pass by consuming group offsets in scatter.
- [x] Add regression coverage and validate the compiler, shaders, and GPU results.

## Production code by category

Line counts are the final working-tree diff against HEAD, including comments
and blank lines. New files are included. Tests and documentation are separate.

| Category | Added | Removed | Net |
| --- | ---: | ---: | ---: |
| Dead-array elimination | 228 | 1 | **+227** |
| Cooperative filter scan and pass removal | 273 | 138 | **+135** |
| Repeated pure calculations | 49 | 2 | **+47** |
| Redundant camera dispatch | 11 | 1 | **+10** |
| Remaining capture-sharing case | 4 | 1 | **+3** |
| **Production total** | **565** | **143** | **+422** |

### Dead-array elimination

- `wyn-core/src/egir/block_interface.rs` adds product-column splitting:
  tuple/record fields become independent block parameters, with their declared
  types preserved on incoming edges.
- New `wyn-core/src/egir/flow_liveness.rs` traces used fields backward through
  loops and branches. An unused array recurrence no longer keeps itself alive.
  The pass rewrites references as well as aliases so later lowering can reuse
  the correct storage.
- The same module removes unused direct local allocations and stores after
  required call expansion. Derived addresses remain conservative.
- `physical_flow.rs`, `skel_opt.rs`, and `mod.rs` connect the passes to lowering.
  Arrays returned or read by a live result remain intact.

### Cooperative filter scan

- New `wyn-core/src/egir/soac_lowering/filter_scan.rs` builds an inclusive scan
  over contiguous 64-element tiles. Four workgroups use two shared-memory banks,
  uniform barriers, and a carry between tiles. Empty inputs and partial tiles
  are masked explicitly; output order remains stable.
- `soac_lowering/filter_lowering.rs` combines each element's prefix with its
  group's offset during scatter.
- `parallelize/filter.rs` schedules four phases instead of five, declares the
  scatter's count-buffer read, and reduces each group-summary buffer from
  1,024 bytes to 16 bytes. `parallelize/mod.rs` removes the unused phase import.

### Pure helper reuse

`wyn-core/src/egir/ir.rs` reuses identical pure calls within a block.
`egir/elaborate.rs` also reuses a result from a dominating scope: execution must
already have passed through the earlier call. Different arguments and calls
in separate branch arms remain separate. This reduces repeated camera helpers
and clamps without changing effectful calls.

### Camera dispatch and capture sharing

Both changes are in `wyn-core/src/egir/allocation/residency.rs`. Generated
singleton scalar stages skip another prelude extraction. Serial consumers
after the shared load in the same block can observe the capture handoff.

## Tests and supporting code

Supporting test code is **+392 net lines**: 427 added and 35 removed.

- `wyn-core/src/tinyporto_codegen_tests.rs`: eight compiler regressions covering
  dispatch counts, dead/live arrays, capture sharing, declared field types,
  pure-call scopes, count-buffer dependencies, and filter boundaries.
- `wyn-core/src/integration_tests.rs`: registers those tests and updates existing
  filter expectations for four phases and smaller scratch buffers.
- Four `testfiles/tinyporto_*.wyn` fixtures cover camera, capture handoff,
  dead-array state, and a filter with a runtime length.
- `scripts/test_tinyporto_filter_gpu.ps1`: CPU-reference checks for counts and
  stable ordering at ten lengths, including zero, tile boundaries, and 39,592
  elements, with four predicate patterns.
- `scripts/test_tinyporto_state_gpu.ps1`: CPU-reference checks for shared
  capture, scalar-only state, reads of prior array state, and returned arrays.

## Validation

All generated artifacts are under `tmp/tinyporto-codegen/`; the sibling
Tinyporto sources were read without modification.

- `cargo test --workspace`: 1,871 passed, 19 ignored, zero failures.
- Tracked shader gate: 92 SPIR-V cases passed; 91 WGSL cases passed, one skipped.
- GPU reference checks on a Radeon RX 580: 40 filter and 12 state cases per
  backend, **104 cases total** across SPIR-V and WGSL.
- Full Tinyporto graphics build passes `spirv-val --target-env vulkan1.3`.
  Its shared producer contains the two 32-event folds (UI and capture);
  point/item/head consumers contain no event folds. Item/head consumers have
  no local array allocations. The point consumer retains one live array;
  the producer retains the arrays needed to compute the point output.
- The full application has one camera vertex prepass. In the inspected resolve
  fragment, static rotation-helper calls decrease from seven to two and clamps
  from eighteen to six; the remaining rotation calls have different arguments.
- `git diff --check` passes.

These are correctness and generated-code checks, not performance benchmarks.
The state GPU script supplies explicit test output capacities because headless
`viz` currently resolves `SameAsDispatch` capacity from the first stage. The
test-local descriptor keeps the emitted shaders and dispatches intact.
