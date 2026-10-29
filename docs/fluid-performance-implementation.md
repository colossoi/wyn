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
