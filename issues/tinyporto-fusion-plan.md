# Tinyporto fusion: remaining work

Initially audited on 2026-09-22 against the Wyn working tree and sibling
`../tinyporto` package. Stage 1 was reviewed; Stage 2's review is below.
Counts labeled as targets remain estimates, not proofs of optimality or measured
GPU speedups. Pause after each conceptual stage for user review.

## Stage 1 review

Length placement and propagation use the existing scheduling egraph. Length
queries depend on view metadata and the stored live count, independently of
element reads. They retain slice bounds and mutable-reader ordering. Rust
constructs the selected metadata queries without capturing the whole array.
The general scalar rematerialization budget remains unchanged.

Tinyporto now emits **26 entries: 18 compute and 8 graphics**, in both WGSL and
SPIR-V. The three GI length-only dispatches and the visibility count-conversion
dispatch are gone. World updates still have 5 compute entries, visibility 5,
AO/coarse depth 3, and GI 5. SPIR-V validation passes.

Regression coverage includes empty arrays, input and mapped-array slice lengths, live filter
counts versus capacity, mutation after a length query, and a count-only view
whose unused element buffer is discarded. The last case bypasses fusion to
exercise scheduling, then validates WGSL and lowers SPIR-V.

Final validation:

- `cargo test -p wyn-core -p wyn --quiet`: 1,419 passed, 18 ignored, no failures
  (1,387 core unit tests, 27 CLI integration tests, and 5 documentation tests).
- `scripts/validate_testfiles.ps1 -Release`: 106 SPIR-V testfiles passed.
- The same script with `-Wgsl`: 105 passed; `miner` was skipped because it uses
  linked SPIR-V helpers.
- Tinyporto's final WGSL validates with `viz validate`; SPIR-V validates with
  `spirv-val --target-env vulkan1.3`. Formatting and whitespace checks pass.

These are semantic/compiler and shader validation checks, not GPU benchmarks.

Stage 1 was committed as `97d7da7b` and reviewed. Stage 3 and the optional
profiling work have not started.

Stage 1 artifacts: `tmp/tinyporto-fusion-recheck/stage1.{wgsl,spv,spvasm,mir}`;
the test log is `tmp/tinyporto-fusion-recheck/stage1-final-tests.log`.

## Stage 2 review

The existing fusion egraph selects a pure map after a filter and attaches it to
compaction. Named scalar helpers receive purity proofs from their region/call
edges in that graph. Rust imports those edges and constructs the selected body;
it does not choose fusion eligibility. The predicate stays in the prefix phase,
and the post-map executes only for survivors in both parallel and serial filters.

Tinyporto is being edited concurrently, so the integration check uses a copied,
hash-checked source snapshot in `tmp/tinyporto-fusion-recheck/tinyporto-snapshot/`.
Hashes of its original manifest and source files are in `source-hashes.json`;
only the snapshot's dependency paths were adjusted. The sibling checkout is
untouched. On this snapshot, enabling the post-map reduces **26 entries to 25**:
**17 compute and 8 graphics**. World updates have 5 compute entries, visibility
4, AO/coarse depth 3, and GI 5. Prop construction runs inside the selected branch
of compaction, eliminating the separate map and compacted-index buffer
(39,592 i32 elements, or 158,368 bytes of capacity).

Length metadata now resolves at its definition and travels through the existing
`References` propagation. The duplicate length-specific child, forwarding, and
projection rules are removed. The shared dependency representation distinguishes
buffer contents from an input descriptor, so querying a mutable array's extent
does not introduce an old-content read or a cyclic dispatch dependency.

Type-changing post-maps allocate fresh output storage. Empty record arrays
reserve one physical element to satisfy shader binding requirements while their
logical length remains zero; the sizing rule is also in egglog.

Coverage includes record and tuple outputs, captured values and predicates,
named helpers, chained maps, multiple length observers, repeated inputs,
survivor-only partial arithmetic, and nested serial filters. Raw-array observers,
slices, captured filter results, effect barriers, and effectful/read-opaque
callbacks retain separate operations.

Validation:

- `cargo test -p wyn-core -p wyn --quiet`: 1,430 passed, 18 ignored; the final
  fusion-only run also passed all 39 tests after tightening summary propagation.
- `scripts/validate_testfiles.ps1 -Release`: 107 SPIR-V files passed. With
  `-Wgsl`: 106 passed, with the existing `miner` linked-SPIR-V skip.
- `scripts/test_rust_host_gpu.ps1 -Compiler target/release/wyn.exe`: passed on
  the Radeon RX 580 Vulkan adapter. Both backends match CPU references across
  40 post-map cases each: sizes 0, 1, 63, 64, 65, 255, 256, 257, 4096, and 39592,
  each with dense, empty, alternating, and sparse survivors. These include
  changing scratch sizes, count publication, and record layout checks.
- Frozen tinyporto WGSL passes `viz validate`; SPIR-V passes
  `spirv-val --target-env vulkan1.3`. Both have 25 entries. Formatting and
  whitespace checks pass.

Artifacts and logs are under `tmp/tinyporto-fusion-recheck/`, including
`snapshot-before.wgsl`, `stage2-snapshot.{wgsl,spv,spvasm,mir}`,
`stage2-final-tests.log`, `stage2-final-fusion-tests.log`, and
`stage2-gpu-tests.log`. GPU execution above checks correctness, not performance.

Stage 2 was reviewed and approved for commit. Stage 3 is the next conceptual
stage; pause for review before committing it.

## Verified baseline

Rebuilt the release compiler with Cargo, compiled tinyporto to WGSL and SPIR-V,
and repeated WGSL generation with `-O`. All three builds emit **30 shader entry
points: 22 compute and 8 graphics**. `spirv-val --target-env vulkan1.3` passes.
The previous audit counted 32. Grouping state decoding, UI updates, and stroke
capture into one kernel accounts for the two-dispatch reduction in world updates.

| Work | Previous audit | Audited baseline | Core target |
| --- | ---: | ---: | ---: |
| World/UI/stroke updates (`compute_0`) | 7 | 5 | 3 |
| Visibility and prop generation (`compute_1`) | 6 | 6 | 3 |
| AO and coarse depth (`compute_2`) | 3 | 3 | 3 |
| GI (`compute_3`) | 8 | 8 | 5 |
| Four vertex/fragment pairs | 8 | 8 | 8 |
| Total | 32 | 30 | 22 |

Generated artifacts and two small compilation probes are under
`tmp/tinyporto-fusion-recheck/` (local, ignored artifacts). Reproduce the main
build with:

```powershell
cargo run --release -p wyn -- build ../tinyporto --graphics --target wgsl --max-warnings 0 --output tmp/tinyporto-fusion-recheck/main.wgsl --output-mir tmp/tinyporto-fusion-recheck/main.mir
target/release/wyn.exe build ../tinyporto --graphics --max-warnings 0 --output tmp/tinyporto-fusion-recheck/main.spv
spirv-val --target-env vulkan1.3 tmp/tinyporto-fusion-recheck/main.spv
```

Bookkeeping is partially fixed, but these cases remain in actual generated code:

- Three GI kernels (`compute_3_compute_2`, `_5`, `_7`) only publish array lengths;
  two independently publish the same GI-history input length.
- Visibility's `compute_1_compute_2` copies/converts the filter count into another
  scalar buffer. `compute_1_finish` then constructs the indirect draw command.
- World's `compute_0_compute_4` computes the new stroke head, and
  `compute_0_finish` packs UI and head arrays into their output buffers.
- AO's `compute_2_compute` computes a larger constants record. This is a cost
  tradeoff, not equivalent to the length-only kernels.

The `lengths.wyn` probe computes `n = length(xs)` and maps over `iota(n)` while
using `n` in the callback. It still emits two kernels. The `filtered.wyn` probe
returns both `map(f, filter(p, xs))` and the filtered length; it emits six kernels.

## Implementation order

### 1. Finish length placement and propagation (30 -> approximately 26)

Extend the current execution analysis, rather than adding a separate optimizer.
Represent metadata dependencies separately from reading array contents:

- Input/view lengths can come from host descriptors or consumer-local view
  metadata. Preserve slice bounds and logical lengths rather than substituting
  allocation capacity.
- A filtered length remains a GPU-produced value from the offsets phase. Read
  that value directly in its consumers; remove only the redundant conversion/
  publication kernel. Preserve the count-writer dependency and empty-input path.
- Do not broaden the general rematerialization budget to make these examples
  pass. Unknown effects and mutable element snapshots must retain their current
  safeguards.

Start in `wyn-core/src/egglog/execution.rs`, `execution.egg`, `residency.egg`, and
the builtin/length representation feeding them. The execution importer currently
treats builtin calls conservatively; confirm the length operation's exact path
with the small probe before changing classification. Reuse existing extent and
host-expression handling in planning and scheduling.

Acceptance: one kernel for the input-length probe; no GI length-only dispatches
or associated scalar scratch buffers; no extra visibility count-conversion
dispatch. Add cases for slices, shared lengths, zero lengths, and filtered count
versus capacity, including a later mutation of array contents.

### 2. Fuse a map after a filter into compaction (approximately 26 -> 25)

Add an explicit filter post-map with an identity default. Keep the existing
pre-map and predicate separate: moving `prop_from_index` before the predicate
would construct props for rejected candidates and defeat tinyporto's algorithm.

Initial eligibility should cover a pure whole-stream map with one filtered-array
consumer and any number of compatible length observers. Conservatively reject
extra raw-array observers, slices, effects, and consumers needing other global
results until those cases have explicit proofs. Preserve capture routing and
stable output order.

The compact kernel already knows each survivor's original value and compacted
destination. Run the post-map only on that selected branch and write its result
directly to the final output. This removes both the prop-generation dispatch and
the intermediate compacted-index array. Preserve flags, local offsets, group
totals, and offsets needed by the three-phase stable filter.

Main implementation areas: `data.rs` and filter visitors/importers;
`fusion/analysis`, `fusion/fusion.egg`, `fusion/plan.rs`,
`fusion/build/filter.rs`; `schedule/kernels/filter.rs`; and the matching serial
and nested filter paths. Update output element types, capacity byte sizes,
reuse legality, and ABI publication when the post-map changes element type.
Keep legality decisions in egglog and body construction in the Rust sidecar.

Acceptance: filtering plus a pure mapped output still has exactly the three
filter phases. Exercise type-changing record/tuple outputs, captures, multiple
length observers, chained maps, and rejection of unsafe fusion. Execute empty,
all-rejected, all-selected, alternating, and sparse cases around 64-element
boundaries and at tinyporto's 39,592 candidates.

### 3. Attach scalar output work to its producer (approximately 25 -> 22)

Plan scalar epilogues at the phase where their inputs become available:

- Visibility: write the indirect command `(36, count, 0, 0)` in the offsets
  phase that computes `count`. Remove the separate finish kernel. The draw must
  still wait for both the command writer and the compacted prop writer.
- World update: move `record_capture_progress` and the small UI/head output
  stores into the existing grouped scalar capture kernel. They do not consume
  either of the following array-map results. Remove the head-update and finish
  kernels, leaving one scalar kernel and two maps.

Implement dependency-proven producer attachment and output-writer selection,
not source-name special cases. Existing scalar grouping only joins consecutive
compatible operations; independent array maps currently separate the world
head update from its producer. Respect effects, aliasing, execution scope, and
old-value readers when moving work across those maps. Keep the event folds
executed once, never duplicated across map lanes.

Main areas: `execution.egg`, scheduling/planning, output backing and copying,
`dispatch.egg`, and ABI publication. Preserve the external result layout, output
lifetimes, and required host readbacks for genuinely host-returned scalar data.

Acceptance: visibility has three dispatches, world updates have three; zero
survivors still write a zero-instance command; UI/head contents match the
unfused program. Check old-state rendering, mutable aliases, intervening effects,
and the draw's dependencies on both filter output components.

### 4. Profile optional reductions after reaching the core target

Keep these as separate, measured decisions rather than prerequisites:

1. AO constants: compare the existing setup dispatch with evaluating constants
   inside the AO consumer. Potential total **22 -> 21**, at the cost of repeated
   arithmetic. Retain the prepass if it is faster.
2. World maps: consider a guarded elementwise kernel over the 1,024-point and
   128-item domains. This requires fusion across different iteration sizes;
   existing horizontal fusion requires matching domains. Potential saving: one
   dispatch. Ensure each body's loads and work execute only within its domain.
3. AO/coarse-depth maps: consider shared dispatch or workgroup partitioning for
   their full-resolution and 8x8-tile domains. Potential saving: one dispatch,
   but combining a heavy AO body and tile reduction can hurt register use and
   scheduling. Preserve both output extents and graphics dependencies.

If all three optional changes win, the arithmetic target is 19 total entries.
Do not promise 19 as the fastest configuration. Measure warmed-up GPU time,
whole-frame time, intermediate bytes, and compile time on identical inputs and
resolutions. The current generated Rust compute passes use
`timestamp_writes: None`; add optional timing instrumentation if per-pass GPU
measurements are needed. Tinyporto's existing median frame-time report is useful
for an end-to-end check but does not isolate fusion costs.

## Boundaries to preserve

Keep the four current graphics stage pairs. Keep the GI producer boundaries:
rays read arbitrary radiance pixels, reconstruction reads neighboring rays, and
recurrent filtering reads neighboring temporal results. These are not ordinary
same-index map chains. Also preserve AO availability before its GI consumers.

The small BVH build may be a candidate for constant evaluation or persistence
across frames, but that is separate work involving resource lifetime and geometry
invalidation. It is not counted in the fusion targets above.

## Validation and completion

For each change, add focused semantic and scheduling regressions in the existing
fusion, planning, scheduling, SSA, and host test suites as appropriate. Test
values and producer/consumer ordering as well as kernel counts. Compile both
backends, validate SPIR-V, and execute representative WGSL/SPIR-V cases against a
CPU reference or the baseline program.

Rebuild tinyporto and compare the per-group counts, scratch allocations, and
dispatch graph after each milestone. Use controlled frames covering no events,
stroke updates, empty/dense visibility, history reset, and GI modes. Preserve the
existing scene-depth/coarse-depth ordering regression.

The older `scripts/test_tinyporto_filter_gpu.ps1` and
`scripts/test_tinyporto_state_gpu.ps1` still consume JSON pipeline descriptors;
adapt their useful case matrices to the current published host-program runner
before relying on them as validation. Do not depend on the sibling tinyporto
checkout for ordinary unit tests: promote the small probes to self-contained
fixtures and keep full-application verification as an integration check.

Core completion means the length-only and avoidable finish dispatches are gone,
filter post-maps preserve stable semantics, the current tinyporto pipeline is
around 22 entries, and correctness checks pass. Further fusion ships only when
measurements support it. The baseline audit ran compilation and SPIR-V
validation; Stage 2 adds GPU correctness checks. Performance benchmarking
remains separate work.
