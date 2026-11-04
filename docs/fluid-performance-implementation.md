# Fluid performance implementation

The scope is the complete four-part implementation: expose where time goes,
preserve parallel work inside tree loops, select collective launches using
separate cost components, and reduce host overhead with a path to larger inputs.
Implementation precedes further tests and benchmarking, as requested.

## Compiler architecture

Fixed fusion remains before execution placement. Source import flattens loop
products into carried leaves without changing their computation. Egglog establishes
dispatch eligibility; Rust checks bounded iteration cost before placement installs
storage identities. Small loops stay invocation-local when estimated serial work
and state traffic cost less than dispatch and predicate readback. Unknown extents,
nested loops and unmodeled operations retain the dispatch path. Scalar callbacks
continue to execute locally. A loop has initialization, an optional
predicate sequence, an iteration sequence and completion. Counted and conditional
loops share this representation, including nested loops.

Each carried leaf has current and next storage. Transfers complete before any
bindings are exchanged, preserving simultaneous tuple assignment. The existing
single-array counted recipe retains direct producer-buffer swaps. Array choices
in transfers are read elementwise. SPIR-V and WGSL use the same SSA emitter;
WHL and generated Rust consume the same published loop/control model.

Execution selection installs the component storage fields; loops kept local do
not acquire dispatch storage identities. Logical lengths remain distinct from
allocation capacity, including runtime-sized and zipped carried arrays. Transfers
use those logical lengths. Graphics entry results also publish scalar metadata,
so the simulator's overflow result reaches numerical validation.

Carried arrays must have invariant shapes. Existential or bounded arrays whose
live lengths can change retain local lowering; supporting those requires carrying
shape descriptors alongside storage, rather than assuming the initial length.

Distributed conditional loops dispatch their predicate and read one boolean before each
iteration. That synchronization is explicit in both host APIs. It avoids placing
the tree's maps, scans and scatters inside a single shader invocation. It does
not require indirect dispatch, device-wide barriers or backend-specific features.

Collective numeric selection follows shape/layout analysis and recipe discovery,
before phase storage and dependencies are finalized. Rust reads facts and ranks
legal group counts; egglog derives geometry and matching temporary capacities.
The cost module separates workload descriptions, hardware rates, candidate
geometry and component estimates. It includes compute, global traffic, temporary
storage, barriers, launch latency and limited group residency. Default rates are
portable priors, not calibrated RX 580 measurements. Source operation counts are
coarse summaries; these estimates rank schedules rather than predict frame time.

`wyn build --gpu-cost-profile rates.toml` supplies measured hardware rates. The
fields are `compute_ops_per_ns`, `memory_bytes_per_ns`, `barrier_ns`, `launch_ns`,
`readback_ns` and `resident_workgroups`. Unspecified fields retain their defaults. Invalid or
unknown values are rejected. `WYN_SCHEDULE_REPORT=1` prints workload, selected
geometry and cost components. Unknown extents and explicit grids retain their
existing bounded recipes.

The model ranks collective group counts and guards against distributing cheap
bounded loops. The loop guard estimates one iteration, including nested map/scan/
reduction extents and scalar work, and uses carried storage size to account for
memory traffic. Its serial throughput estimate uses a conservative 32-lane share
of the configured aggregate rates; this is a portable prior, not the device's
subgroup width. It does not estimate register spills, cache behavior or coalescing.
Unknown work keeps its existing placement. Source work summaries retain their
uncapped counts; bounded inlining queries and collective ranking retain their
previous caps. Work summaries are gathered once per selection rather than scanning
all planner facts for each collective.

## Host execution and measurement

The WHL backend records dispatches, copies, scalar uploads and draws in a shared
command encoder. CPU reads and entry completion submit the pending commands.
Uploads inside a batch use staging copies so loop iterations see their own
values. Invocation-private buffers can be reused after their last tracked alias
is released; reuse clears their bytes in command order. Imported and escaped
buffers are excluded. The pool has a bounded memory budget.

`WYN_GPU_PROFILE=1` enables start/end messages around shader and pipeline
creation, CPU readback-wait timings, per-pass GPU timestamps, submission counts
and complete-frame wall timings. Profile readback occurs outside the frame wall
timer. Normal execution does not wait for profiling. These records distinguish
driver compilation stalls from GPU execution and host waits.

## Scaling and validation

`fluid-simulation/build.py` compiles unchanged shared sources with a generated
configuration module. Its shader metadata supplies the launcher's feedback
buffer sizes. Larger presets scale world dimensions to preserve particle
spacing; tree capacity scales with particle count and can be overridden.
Spatial overflow is exposed as a result as well as the existing rendering cue.

After implementation, run compiler/host regression checks, numerical GPU cases
for zero-trip, counted/conditional product state and nested loops, then the
spatial and fluid gates. Full-frame measurements cover 4,320, 25,000, 50,000
and 100,000 particles, with matching capacities, warmup exclusion and final
finite/bounds/overflow checks.
Compare both shader targets and run the full repository gates after any repairs.

The earlier sort-only measurements in `fluid-collective-scheduling.md` describe
the preceding launch-sizing change. They are not measurements of this expanded
implementation; new numbers belong to the final validated revision.

## Futhark comparison

This follows the separation visible in Futhark's kernel extraction and explicit
memory pipeline: preserve structured control, expose parallel array operations,
then lower selected kernels and storage. Wyn keeps its existing egglog legality
facts and uses Rust for numeric choices and shared SSA emission. Its cost ranking
is a small portable model, not Futhark's full tuning/runtime infrastructure.
There is no runtime multiversioning, structural fusion search, CUDA-only scan
protocol, or independent SPIR-V/WGSL scheduling policy.

Local reference points are `extra/futhark/src/Futhark/Pass/ExtractKernels.hs`
(`transformStm` retains `Loop merge form` while transforming its body) and
`extra/futhark/src/Futhark/CodeGen/ImpGen.hs` (`defCompileExp` initializes merge
parameters, selects `ForLoop`/`WhileLoop`, and publishes their final values).
Wyn's explicit predicate readback is a consequence of its portable host API;
Futhark's memory and scalar placement machinery is more developed.

## Measured result on this machine

The complete simulator now runs through 100,000 particles on the Radeon RX 580
using either shader target. All sixteen measured runs passed final particle-count,
finite-value, world-bounds and spatial-overflow checks.

These are complete headless frames at 640×480, including simulation, rendering,
host encoding and GPU completion. Each cell is the median of forty frames:
two batches with five warmup frames and twenty measured frames each. The second
batch reverses backend order. Builds and other validation runs were stopped
before measurement. Timestamp readback is outside the frame wall timer.

| Particles | WGSL frame | SPIR-V frame | Equivalent throughput |
|---:|---:|---:|---:|
| 4,320 | 20.27 ms | 19.75 ms | 49–51 fps |
| 25,000 | 48.19 ms | 48.76 ms | 20.5–20.8 fps |
| 50,000 | 82.23 ms | 80.16 ms | 12.2–12.5 fps |
| 100,000 | 196.44 ms | 184.24 ms | 5.1–5.4 fps |

The old compiler output at `57cf20b8`, run with the same instrumented viewer,
did not finish driver creation of the serial tree pipeline within 120 seconds.
The profile stops at `fluid__compute_7_compute_4`, before a frame completes.
The new output completes setup and runs the full simulation. That demonstrates
the tree-scheduling improvement, but supplies no steady-state baseline from
which to calculate a whole-frame speedup percentage. The earlier sort-only
before/after measurements remain separately documented.

Normal measured frames use four or five queue submissions, including the
conditional-loop/readback boundaries. At 100k, the WGSL GPU passes total about
183 ms per frame. Neighbor construction (`fluid__compute_7_compute_15`) accounts
for roughly 90 ms. Serial copy/range publication stages account for roughly
75 ms. The tree backedge also spends about 5.4 ms copying arrays unnecessarily
while evaluating scalar fields of a conditional product. Its actual array
transfer is parallel. These are visible remaining costs, not sorting costs.

The next compiler work should stay in shared representation and output planning:
retain slice/view backing where ownership allows; use a parallel copy recipe
when materialization is required; and project scalar branch results without
evaluating unrelated array fields, preserving effects. After those costs are
removed, use the neighbor profile to guide shared tiling/coalescing work. The
current result is a working scaling path, with substantial throughput work left
to reach runtime parity.

## Completed gates

The measurements above used the original capacities and short frame windows.
The 2026-10-30 interactive run exposed a later capacity failure; the correction
and longer validation are recorded below.

- Release workspace build/all targets and all 1,516 workspace tests passed;
  formatting and clippy passed. Fourteen existing tests remain ignored.
- All nine standalone crates built and passed their tests; the browser target
  passed its wasm32 check.
- Shader corpus: 123 SPIR-V and 122 WGSL passed; the existing WGSL miner case
  remains skipped. All eight measured full-program SPIR-V artifacts also passed
  `spirv-val`.
- Thirty GPU loop cases passed across SPIR-V/WGSL: zero trips, counted/conditional
  products, nested loops, runtime extents and empty arrays. Generated Rust host
  code for the same fixture passed `cargo check` for both targets.
- Forty scan/reduction GPU cases passed across both targets and optimization
  modes, including an associative noncommutative operator.
- Spatial, solver/reference, neighbor, batched traversal, renderer and tiled/
  subgroup numerical checks passed on both targets, including the 4,097-slot
  split/merge case. The WGPU runtime's seventeen tests passed, including upload
  order and scratch reuse after the last alias.

GPU numerical runs and timings explicitly disabled Vulkan validation layers.
The strict-layer attempt still reports the existing wgpu/Naga `ArrayStride`
decoration error on private/function/workgroup arrays (VUID 10684). Compiler
SPIR-V validation and WGSL/Naga validation passed; strict driver-layer validation
is not claimed to pass. Full gate logs are under `tmp/fluid-complete-gates`;
the final successful workspace rerun is `tmp/fluid-workspace-final2.log`.

## 2026-10-30: reference capacity correction

The 25,000-particle scene froze and rendered red after several hundred frames.
A 600-frame reproduction returned `overflow=true`, finite positions/velocities,
and 6,231 active leaves in 6,260 tree slots. Recomputing the rebalance from that
captured state found six required splits: 6,273 leaves plus a sentinel need
6,274 slots. This is a tree allocation failure.

The pinned reference revision `db25b1794c0d420a76436fef81d27a84d83f7742`
also passes `maintenance_mode=false` in its
[simulation solver](https://github.com/MarcVivas/fluid-simulation/blob/db25b1794c0d420a76436fef81d27a84d83f7742/src/backends/vulkan/particles/physics/solver.rs#L149).
Its merge implementation is exercised in tests but disabled in that simulation
path. Enabling merging would change the reference behavior.

The actual allocation divergence is in
[upstream Octree::max_leaves](https://github.com/MarcVivas/fluid-simulation/blob/db25b1794c0d420a76436fef81d27a84d83f7742/src/backends/vulkan/particles/physics/octree/octree.rs#L95):
it reserves `max(1024, 24 * particles / (leaf_threshold + 1))` leaves.
The port used approximately `8 * particles / leaf_threshold`. The builder now
uses the upstream formula, retains the default scene's 4,097-slot reserve,
and rounds up to whole sibling groups plus a sentinel. At the port's threshold
of 32, the corrected 25k capacity is 18,188. Upstream derives its threshold from
the device subgroup size; comparisons of the formula here use the same threshold.
The explicit `--tree-capacity` override remains available. This is the reference's
capacity heuristic, not a guarantee for arbitrary particle distributions.

The reference's neighbor traversal uses the same 128 queue entries and 256
candidate leaves. It bounds queue writes and clamps candidate output. The port
additionally reports overflow and freezes/red-colors the scene instead of
continuing with incomplete search results. Those checks remain enabled.

The corrected 25k specialization passed 1,200 measured frames after five warmup
frames on each backend, with no overflow and with particle-count, finite-value
and world-bounds checks passing. The three configuration regression checks also
passed. Logs, metadata and readbacks are in `tmp/fluid-capacity-reference`.
These GPU checks used the same validation-layer exception described above.

The larger reserve has a runtime cost because some compiler stages still process
allocated capacity. The first twenty measured frames were 64.87 ms (SPIR-V) and
67.55 ms (WGSL), versus roughly 48 ms in the earlier smaller-capacity run. Over
the full longer run, medians were 83.18 ms and 85.20 ms, respectively; those
long-run values also include the evolving particle distribution. This fix restores
the reference allocation policy and resolves the reproduced freeze, but is not
a performance improvement. Removing capacity-sized work remains compiler work.

Run the configuration regression checks:

```text
python fluid-simulation/test/check_build.py
```

## Amendment review checkpoint

The pending amendment adds the local-loop cost guard, gathers collective work
in single table passes, decodes loop carries into named fields, and includes the
capacity correction above. Review found that treating every nested loop as
unknown left the small-loop regression in place for a constant two-trip inner
loop. The estimator now recursively weights counted-loop headers and bodies by
their constant trip count, without counting those regions again as lexical
children. Zero-trip bodies contribute no work; unknown trip counts retain the
dispatch fallback. Planning checks cover zero, small, large and unknown counts.

Both the flat and nested four-element, 128-iteration cases now emit no host
dispatch loop on SPIR-V or WGSL. The nested SPIR-V probe improved from 127.59 ms
before this repair to 0.29 ms; the pre-commit compiler measured 0.37 ms in the
same rerun. These are medians of five frames after five warmups, using the same
runtime binary and identical output checks. This measures the scheduling
regression, not a full-fluid speedup.

The GPU runner now executes `conditional_runtime`, `counted_runtime` and
`nested_runtime`, checking
every carried array and scalar at zero, one and three iterations, with empty,
single-element, longer and early-exit inputs. All 116 loop cases passed across
both shader targets, including the fixed nested case at 128 iterations. The
host contract checks that `nested_runtime` still contains an outer distributed
while loop and an inner distributed counted loop, so a future scheduling change
cannot silently remove this GPU coverage.

Release workspace build, formatting, clippy and all 1,520 workspace tests passed
(14 ignored), including 40 planning tests and four host contracts. The 40
collective GPU cases, three capacity checks, 17 WGPU runtime tests and fluid
spatial, solver, neighbor, batched-traversal and renderer checks passed. Tests
used the documented 64 MiB Windows stack. All nine standalone crates built and
passed their tests; the wasm32 check passed. Shader validation passed 123 SPIR-V
and 122 WGSL cases, with the existing WGSL miner skip. The viewer build passed
on retry after a running GPU check released its executable's Windows file lock.
Logs are under `tmp/amend-fixed-gates`; loop execution and retiming logs are
`tmp/amend-fix-gpu-loops.log` and
`tmp/amend-fix-retiming.log`. GPU runs use the validation-layer exception above.

The multi-entry scan fixture also exposed a frame-resource identity bug: inputs
named `xs` in independent entries were merged despite having different lengths.
Input buffer identity now includes the published authored-entry owner. Generated
passes of one entry still share the input, explicitly named resources still
share across entries, and public parameter names remain unchanged. Regressions
cover independent inputs with both equal and different lengths, shared entry
owners, interfaces without stages, and ABI refresh. Both shader targets preserve
the three-entry compiler regression's independent inputs and generated host
contracts. All 40 collective GPU cases now pass with the debug compiler too.
At the requested commit checkpoint, the follow-up workspace build, clippy and
1,522 tests (14 ignored) passed, along with the GPU regressions and fluid
numerical checks. The remaining full gates are running under
`tmp/input-scope-gates`.

Materialization and neighbor optimization remain subsequent work.

## 2026-11-01: reuse finish dispatches for parallel output copies

Required array copies now use the existing finish dispatch, strided SSA copy
emitter and host launch expressions. There is no new operation identity, copy
kernel phase, runtime representation or backend-specific implementation.
Egglog proves eligibility from existing views/ranges and the region's read-only
summary, then selects a 64-thread finish workgroup when every array output in
that finish is eligible. Rust derives the grid from the allocated output buffer
sizes using the existing linear dispatch builder. No device length readback is
needed for the launch; each invocation checks the actual logical length.
Scalar outputs execute only in the first invocation. Unsupported or effectful
finishes retain their existing serial behavior.

Zero-offset views also reuse their existing backing when the normal output
ownership/binding rules permit it. This extends the existing offset proof to
constant sums rather than introducing another alias model. Two host-boundary
repairs were needed for the broader view coverage: shortened borrowed views no
longer promise exact input-handle identity, and stored unsigned counts convert
to Wyn's signed length type before host arithmetic.

Paired measurements on the RX580 compare compiler `5273b8d0` with this patch,
using the same viewer binary, scene, reference-derived tree capacity and 640x480
rendering. Each variant ran 25 frames, discarding the first five; a second batch
reversed case and variant order. The table combines the 40 measured frames per
variant. These are short-run, whole-frame measurements including GPU completion,
not comparisons against the older long-run numbers above.

| Particles | SPIR-V before → after | Speedup | WGSL before → after | Speedup |
| --- | --- | --- | --- | --- |
| 4,320 | 20.40 → 17.33 ms | 1.18x | 20.96 → 17.85 ms | 1.17x |
| 25,000 | 65.79 → 49.43 ms | 1.33x | 67.97 → 51.39 ms | 1.32x |
| 100,000 | 255.28 → 184.06 ms | 1.39x | 266.29 → 192.55 ms | 1.38x |

Final positions and velocities matched bit for bit in all twelve pairs; all
runs passed finite-value, world-bounds, particle-count and overflow checks.
The first 25k SPIR-V pair's finish kernels fell from 19.82 to 3.46 ms. Some
serial finish work remains under the conservative eligibility rule; neighbor
traversal and tree-loop work are unchanged by this patch.

The new regression executes 58 GPU cases across both shader targets: borrowed
prefixes, nonzero-offset copies, produced and compacted arrays, mixed scalar and
array results, nested arrays, zero lengths, branch-dependent lengths and
preserved pre-update values.
Compiler contracts validate both shader formats and assert parallel finish
launches; host interpreter checks cover view lifetimes and count arithmetic.
All 25 full gate commands passed: formatting, workspace build, 1,523 workspace
tests (14 ignored), clippy, all nine standalone crate builds/tests, wasm32, and
the shader corpus (123 SPIR-V; 122 WGSL with the existing miner skip). The 116
loop cases, 40 collective cases, three capacity checks, fluid numerical checks
and 17 WGPU runtime tests also passed. GPU execution uses the validation-layer
exception described above; compiler SPIR-V validation remains enabled.
Temporary benchmark harnesses and raw data are excluded from the changeset.

## 2026-11-02: retain shared arrays through cooperative traversal

Cooperative maps now demand shared storage for their array results, using the
existing storage-demand analysis. Demand propagates through helper returns and
arguments as it already did through captures and control-flow boundaries.
Shared functional updates use fresh buffers and the existing strided copy
emitter, preserving reads of the previous value. Scatter initialization and
array-result publication distribute their copies across the workgroup.

The existing boundary plan also skips a loop's staging copy when its backedge
is a branch result: the branch has already copied shared fields into fresh
destinations. Other loop backedges retain staging for parallel assignment.
Egglog selects storage and proves this transfer condition; common SSA lowering
emits the selected copies and barriers for both shader targets. The patch adds
78 net core-compiler production lines without a new scheduling representation.

The 25k traversal shader has zero private array buffers, down from five. The
regression checks shared results through helper calls, branch swaps, zero/one/
three iterations, 65-element arrays, and old-value reads after an update. GPU
checks compare the results with an independent CPU calculation; a compiler
contract prevents this fixture from silently reverting to private arrays.

Paired RX580 measurements compare `d47b724d` with this patch, using the same
viewer binary, scene, tree capacities and 640x480 rendering. Each variant ran
25 frames, discarding five; a second batch reversed case and variant order.
The medians below combine 40 measured frames per variant. Builds and other GPU
tests had finished before measurement.

| Particles | SPIR-V before → after | Speedup | WGSL before → after | Speedup |
| --- | --- | --- | --- | --- |
| 4,320 | 17.55 → 12.54 ms | 1.40x | 16.99 → 13.61 ms | 1.25x |
| 25,000 | 48.25 → 28.49 ms | 1.69x | 52.18 → 30.06 ms | 1.74x |
| 100,000 | 183.40 → 90.43 ms | 2.03x | 194.04 → 99.13 ms | 1.96x |

Final positions and velocities matched bit for bit in all twelve pairs, with
no overflow, non-finite values or world-bounds failures. In the first SPIR-V
pairs, traversal fell from 27.09 to 7.29 ms at 25k and from 131.43 to 37.46 ms
at 100k; tree-loop and finish timings stayed approximately constant. The
traversal shader's static barrier count increases from 45 to 72, but eliminating
private array copies and distributing initialization/publication more than pays
for that synchronization in these runs. These are short-run measurements on
this RX580, not a claim about every device or scene.

All 25 full gate commands passed: formatting, release workspace build, 1,524
workspace tests (14 ignored), clippy, all nine standalone crate builds/tests,
wasm32, and the shader corpus (123 SPIR-V; 122 WGSL with the existing miner
skip). All 11 GPU gate commands passed, covering 116 loop cases, 40 collective
cases, 58 output-copy cases, three capacity checks, the fluid numerical checks
and 17 WGPU runtime tests. GPU execution uses the validation-layer exception
described above; compiler shader validation remains enabled. Logs and temporary
measurement tools are under `tmp/boundary-copy-perf` and are not committed.

## 2026-11-03: consolidate output copies

`copy_array_strided` now accepts a destination buffer and index path. Map
publication uses this existing emitter alongside scatter initialization,
boundary transfers and finish outputs. Recursive component copies extend the
path and store directly into the destination; they no longer read an output
row to obtain a copy target. This removes the map-specific copy loop. The
complete patch is now +76 net core-compiler production lines, down from +78.

Final validation passed all 25 full gate commands: formatting, release workspace
build, 1,525 workspace tests (14 ignored), clippy, all nine standalone crate
builds/tests, wasm32, and both shader corpora (123 SPIR-V; 122 WGSL with the
existing miner skip). All 11 GPU gate commands passed, covering 116 loop cases,
40 collective cases, 62 output-copy cases, three capacity checks, the fluid
numerical checks and 17 WGPU runtime tests. The new fixture checks both branches
of nested tuple-component copies against CPU results on both shader targets.
GPU execution uses the existing viewer-layer exception; `spirv-val` remains
enabled. Final gate logs are under `tmp/shared-array-final-gates`.

An exploratory build combining the new fixture with the other output-copy
entries exposes an existing SPIR-V layout error: a Function variable's type
inherits explicit Block layout. The pre-consolidation compiler fails the same
validation. The new fixture validates and runs independently; that separate
multi-entry backend issue is not repaired by this consolidation.

The generated traversal WGSL is byte-identical before and after consolidation
at 4,320, 25k and 100k particles; the 100k SPIR-V host program is also identical.
Short whole-frame comparisons at 25k/100k on both targets ranged from 1.3% faster
to 4.3% slower. A longer 100k SPIR-V follow-up used four pairs of 100 frames,
discarding the first 20 each time: median frame time was 93.74 → 95.02 ms (1.4%
higher), while median traversal GPU time was 38.234 → 38.227 ms. Tree-loop and
finish GPU timings were also unchanged to within 0.001 ms. This supports
unchanged GPU work, with some variation in whole-frame timings. Final particle
outputs matched bit for bit in every pair, including the longer runs; all
finite-value, world-bounds and overflow checks passed. Artifacts are excluded
from the changeset under `tmp/unified-copy-perf`.

## 2026-11-03: admit loop fields and parallelize stored output copies

Dispatch-loop transfers now admit their initial and next fields as scalar
optimization roots. Previously the synthetic projections bypassed the existing
field-selection logic, so reading a boolean or counter could materialize the
entire containing product. Elementwise branch transfers also admit their
predicate independently. Both changes extend the existing admission rules;
scalar optimization and shared SSA emission remain unchanged.

Parallel-output eligibility now follows the copied value. Read-only views and
ranges qualify independently of other effects in their source region. A stored
result from a dispatched producer also qualifies: its effects have already
executed, and the finish kernel only reads the result. Device-local producers
retain serial evaluation. Existing dependencies order the producer, copy and
draw; existing finish lowering runs scalar outputs once.

The change adds 22 net core-compiler production lines, all in egglog. No new
execution representation, Rust lowering, backend exception or fluid-specific
rule is needed. The 100k shader loses all four private 72,732-element tree
copies, and renderer index publication changes from one invocation to the
existing strided parallel copy.

Regressions cover zero/one/multiple loop iterations, both array-choice outcomes,
scalar siblings of a 4,097-element array, and copies alongside an effectful producer
with observable sums. The runtime-sized renderer validates both shader formats;
GPU checks at 3 and 129 particles compare every index and pixel with a CPU
reference.

Paired RX580 measurements compare `456e8670` with this patch, using the same
viewer, scene, capacities and 640x480 rendering. Each variant ran 25 frames,
discarding five; a second batch reversed variant and case order. These are
whole-frame medians over 40 measured frames per variant. Builds and GPU gates
were stopped during measurement.

| Particles | SPIR-V before → after | Speedup | WGSL before → after | Speedup |
| --- | --- | --- | --- | --- |
| 25,000 | 27.72 → 20.04 ms | 1.38x | 30.49 → 20.10 ms | 1.52x |
| 100,000 | 93.42 → 56.63 ms | 1.65x | 103.75 → 60.62 ms | 1.71x |

At 100k, the first SPIR-V pair's tree-loop transfer fell from 15.67 to about
0.005 ms, and renderer index publication from 13.77 to about 0.004 ms. Neighbor
traversal stayed at 37.45–37.46 ms. The GPU timestamps attribute about 29.44 ms
of savings to the two targeted phases. Whole-frame timings include host and
driver waiting, which were not attributed separately. These short-run
measurements use this RX580 and scene.

Positions and velocities matched bit for bit in all eight pairs. Every run
passed particle-count, finite-value, world-bounds and overflow checks. Temporary
measurement scripts and raw data remain excluded from the changeset under
`tmp/projection-copy-perf`.

All 25 full gate commands passed: formatting, release workspace build, 1,527
workspace tests (14 ignored), clippy, all nine standalone crate builds/tests,
wasm32, and the shader corpus (123 SPIR-V; 122 WGSL with the existing miner
skip). All 11 GPU gate commands passed, covering 122 loop cases, 40 collective
cases, 78 output-copy cases, three capacity checks, the fluid numerical checks
and 17 WGPU runtime tests. GPU execution uses the existing viewer-layer
validation exception; compiler SPIR-V and Naga validation remain enabled.
Final gate records are under `tmp/projection-copy-verified-gates`. The gate
run was interrupted for uncontended measurement and resumed from its recorded
successful checks afterward.

## 2026-11-03: share storage liveness and reuse workgroup scratch

The existing SSA storage pass now handles cooperative workgroup allocations as
well as private array versions. Both use the same edge-aware liveness rules,
including loop backedges. Workgroup interference additionally keeps accesses
pending until a barrier: a last use by one lane does not free the allocation
while another lane may still be using it. Rust selects compatible physical
buffers from these egglog interference facts, using the largest member's
capacity and retaining each logical view's offset and length.

The storage contracts now describe slices and inherited views as aliases.
Common provenance queries follow those aliases, aggregate fields, block
parameters and derived places. Private allocation reuse uses the same place
provenance. Transitive call effects distinguish scalar helpers from workgroup
reads/writes; unknown calls and escaped views remain conservative.
Views nested in arrays or tuples also retain their memory provenance. Returning
a descriptor publishes its contents, so escape analysis and barrier scheduling
both account for that read.

Synchronization uses these same physical memory facts after allocation reuse.
The SSA pass removes a barrier only when the accesses on either side have no
shared-memory conflict before another retained barrier. Each removal updates
the subsequent queries, so a chain of redundant barriers retains the required
producer/consumer ordering. Rebuilding provenance after pooling also preserves
barriers required by newly aliased buffers. This runs in common SSA preparation
for SPIR-V and WGSL, within the existing storage stage.

This follows the placement of Futhark's `MemoryBlockMerging`: reuse allocations
after memory is explicit, then select a coloring of the interference graph.
Wyn additionally models accesses still pending across lanes and currently pools
only matching element types. Fixed fusion and source execution planning retain
their existing stages. No traversal-specific names, capacities or rules are
part of the implementation.

The traversal shader's shared allocations fall from 31 to 15, using 7,080
instead of 16,224 bytes. Its static workgroup-barrier count falls from 72 to 45.
Copies between simultaneously live array versions remain explicit; this change
reduces their storage pressure and redundant synchronization.

Paired RX580 measurements compare `b876b492` with this patch, using the same
viewer, scene, capacities and 640x480 rendering. Each run uses 25 frames and
discards five. Two batches reverse case and variant order; the table combines
40 measured frames per variant. No builds or other GPU work ran during timing.

| Particles | SPIR-V before → after | Speedup | WGSL before → after | Speedup |
| --- | --- | --- | --- | --- |
| 25,000 | 20.18 → 15.56 ms | 1.30x | 20.26 → 15.68 ms | 1.29x |
| 100,000 | 59.12 → 36.35 ms | 1.63x | 60.40 → 37.73 ms | 1.60x |

The first 100k SPIR-V pair's traversal falls from 37.44 to 14.60 ms; neighbor
construction's 5.40 ms partials phase stays unchanged. Positions and velocities
match bit for bit in all eight pairs. All runs pass particle-count, finite-value, bounds and
overflow checks. The independent traversal oracle passes 76 queries on each
backend, and the tiled-output/subgroup checks pass both targets. Temporary
measurement tools and raw data remain excluded under `tmp/workgroup-reuse-perf`.

Regressions cover synchronization before reuse, descriptor aliases, cross-block
dependencies, loop backedges, transitive calls and returned arrays containing
views. Rebuilt final WGSL shaders at 4,320, 25k and 100k retain the measured
allocation and barrier counts. Both shader formats validate at all three sizes,
and their generated host programs remain byte-identical to the measured builds.

All 25 full gate commands passed on the final source: formatting, release
workspace build, 1,533 workspace tests (14 ignored), clippy, all nine standalone
crate builds/tests, wasm32, and the shader corpus (123 SPIR-V; 122 WGSL with the
existing miner skip). All 11 GPU gate commands passed, covering nested dispatch
loops, collectives, output copies, capacity and fluid numerical checks, and 17
WGPU runtime tests. GPU execution uses the existing viewer-layer validation
exception; compiler SPIR-V and Naga validation remain enabled. Final gate records
are under `tmp/workgroup-reuse-final-gates`.

## 2026-11-04: defer memory transfers until storage selection

The common array-copy emitter now retains representation-preserving transfers
between views as SSA copies. Shared updates and control-boundary transfers keep
their source, destination, extent and worker distribution visible to storage
planning. The existing interference analysis can then coalesce a full transfer
when the source contents are dead and earlier cross-lane accesses have finished.
Live snapshots, partial copies and simultaneous assignments retain the storage
they need. Other buffers' pending accesses still cross a transfer unchanged.

Selected identity copies disappear before barrier scheduling; retained copies
expand afterward through one common SSA implementation. Matching element layouts
are checked in one helper, with no rank restriction. Nested arrays and tuple
fields use element places recursively, avoiding aggregate row snapshots. Both
backends consume the resulting loads, stores and structured loops.

View offset/length projections use ordinary SSA constant folding while retaining
the view's storage identity. Fixed-size parameters and their derived views also
use the same fixed extent, resolving a pre-existing planning conflict exposed by
the new nested-array regression. No fusion or backend-specific scheduling change
is required.

Final paired RX580 measurements compare `00cbc8b6` with this patch, including
640x480 rendering. Each run uses 25 frames and discards five; two batches reverse
case and variant order, giving 40 measured frames per variant. Builds and GPU
gates were stopped during timing. Positions and velocities match bit for bit
in all eight pairs; count, finite-value, bounds and overflow checks pass.

| Particles | SPIR-V before → after | Speedup | WGSL before → after | Speedup |
| --- | --- | --- | --- | --- |
| 25,000 | 14.97 → 12.51 ms | 1.20x | 14.56 → 12.34 ms | 1.18x |
| 100,000 | 34.79 → 25.76 ms | 1.35x | 35.74 → 26.11 ms | 1.37x |

The first 100k SPIR-V pair's traversal falls from 14.60 to 4.83 ms. Neighbor
construction remains at 5.56 → 5.51 ms. The final traversal WGSL has 29 static
barriers, down from 45. A separate unprofiled 100k SPIR-V offscreen run at
1920x1080 completes 600 frames in 15.98 seconds including startup (37.5 fps),
with no spatial-index overflow. This measures offscreen throughput rather than
window presentation pacing.

All 25 full gate commands and 11 GPU gate commands pass on the final source:
formatting, release workspace build, 1,540 workspace tests (14 ignored), clippy,
all nine standalone crate builds/tests, wasm32, shader validation (123 SPIR-V;
122 WGSL with the existing miner skip), and GPU loop, collective, output-copy,
capacity, fluid numerical and runtime checks. Output-copy coverage includes 82
cases across both targets, including both branches of a rank-three result.
GPU execution uses the existing viewer-layer validation exception; compiler
SPIR-V and Naga validation remain enabled. Final records are under
`tmp/array-transfer-final-gates`; measurements and temporary tools remain
excluded under `tmp/array-transfer-perf`.

Excluding test files and inline test items, this patch adds 447 net compiler
production lines. The branch adds 2,506 net compiler production lines relative
to `342d62f9`, including this patch: 2,306 in `wyn-core`, 156 in host IR/code
generation and 44 in the compiler CLI. Runtime, fluid code, documentation,
configuration and harnesses are excluded from these counts.
