# Detailed CLI compiler timings

## Latest: fusion adjacency reuse

Both workloads compile without `-O`. Against frontend/host revision `a18f4149`,
five paired fresh-process medians are **Tinyporto 3.700 -> 3.613 s** and
**fluid 2.169 -> 1.954 s (9.9% less time)**. Fluid's fusion phase falls
**741.954 -> 533.603 ms (28.1% less time)**. This pass is complete; the
250-300 ms fusion target is still not achieved.

Fluid is below two seconds at the median, but **one of five runs is still
2.084 s**. Tinyporto remains approximately 1.61 s above the goal. Its fusion
phase changes by only 4 ms; the broader 87-ms whole-process difference includes
variation in unrelated phases and should not be attributed entirely to this
patch.

| Work | Tinyporto before ms | Tinyporto after ms | Fluid before ms | Fluid after ms |
| --- | ---: | ---: | ---: | ---: |
| Fusion dependencies | 83.165 | 81.475 | 104.392 | 103.509 |
| Build adjacency snapshots | - | 1.408 | - | 6.870 |
| Choose fusion candidates | 46.918 | 45.524 | 472.815 | 267.381 |
| Commit contractions | 7.529 | 7.198 | 111.843 | 101.793 |
| **Whole fusion phase** | **225.540** | **221.641** | **741.954** | **533.603** |
| **Whole process** | **3,700.263** | **3,612.814** | **2,168.974** | **1,954.293** |

Nested parent spans overlap; independent phase medians do not add exactly.
The baseline and modified compiler each received one warmup per workload,
followed by five fresh processes each with binary order alternating by round.
No builds or tests ran during timing; `WYN_EGGLOG_PROFILE` was unset. Graphics,
SPIR-V, Tinyporto's Rust/WGPU host, and fluid's subgroups/WHL host are preserved.
Filesystem caches were warm. Tinyporto ranges are 3.588-3.864 s before and
3.596-3.681 s after; fluid ranges are 2.150-2.314 s before and 1.948-2.084 s after.
Desktop activity and hash/order variation remain uncontrolled.

Raw samples, executable hashes, source patch, harnesses, output comparisons,
and gate results are in [fusion evidence](profiles/2026-10-28-compiler-fusion.json).

### Granularity and implementation

Fusion remains at the operation/group level. The adjacency snapshot contains
only source groups and their neighbors, with no scalar syntax or unrelated
source details. Two immutable adjacency vectors are built once per candidate
round and passed explicitly into pure path predicates. Each vector has rows
sorted by source group; each row contains that source followed by its neighbors.
The predicates retain the same path/intervening-group tests. They can now use
ordinary incremental Egglog rules instead of repeatedly reading live graph
state in naive rules. Snapshots are cleared before every contraction, including
contractions that absorb length observers, and are owned by the compiler graph.
No graph-dependent candidate answer is retained across contractions.

`ElementFusion` compatibility is now demanded only for candidate pairs rather
than a cross product of current groups. In separate engine diagnostics, its
accumulated fact count drops **8,617 -> 56**. Both versions still perform
**32 contractions in 33 candidate rounds**, ending with **456 current groups**
from 488 source operations. The final direct edge relations each contain 383
edges. Snapshot roots are empty after selection, before later compiler passes.
These are table/cardinality diagnostics, not an isolated performance attribution.

### Validation and code size

All full gates passed after the final code change:

- Workspace formatting, all-target build, and **1,505 tests passed**, zero
  failures, 14 existing ignores; all 127 compiler fixture files passed.
- All-target builds/tests for seven standalone crates and both parser manifests,
  plus WASM formatting and locked browser-target check.
- **123 SPIR-V** validations; **122 WGSL** validations and one expected skip;
  the library package compile/validation runner passed.
- Regression cases cover direct paths, intervening barriers, disconnected/reverse
  paths, contraction refresh, cycles, and independence of cloned graph snapshots.

Both final paired shaders and generated hosts are **byte-identical** for both
workloads. Both before/after shaders also pass `spirv-val`. Optional GPU execution
scripts were not run. Gates used the existing writable workspace temporary
folder and `RUST_MIN_STACK=67108864`; parser checks followed shader sweeps and
the package runner followed workspace tests.

Production code is **+53 net lines: 88 added, 35 removed**, including Rust and
Egglog comments, excluding tests and documentation. No dependency was added.

### Remaining work

Further fusion optimization is possible, but this patch stops here. Fluid still
spends 267 ms choosing candidates, 104 ms deriving dependencies, and 102 ms
committing contractions. A reliable sub-two-second result needs margin for the
observed 2.084-s run. Inspect the remaining candidate joins/proofs before adding
cross-round retention; contracting intermediate groups can change paths between
untouched endpoints. Immutable plan summaries and result counts already persist.

Tinyporto's larger remaining opportunities are its basic scalar/evaluation
path and structural planning. Its current total is 3.613 s, so further fusion
work alone will not reach two seconds for that workload. The earlier phase
budgets below remain engineering targets rather than demonstrated speedups.

## Previous: frontend and host generation

Both workloads still compile without `-O`. Against `ac0cfef0`, five paired
fresh-process medians show **Tinyporto 3.720 -> 3.703 s (0.4% less time)** and
**fluid 2.207 -> 2.195 s (0.5% less time)**. The targeted Tinyporto phases save
44 ms, but the whole-process difference is only 17 ms and is smaller than the
sample spread. Do not attribute the unrelated phase variation to these changes.

| Work | Tinyporto before ms | Tinyporto after ms | Fluid before ms | Fluid after ms |
| --- | ---: | ---: | ---: | ---: |
| Helper expansion and cleanup | 167 | 155 | 38 | 34 |
| Rust host generation | 213 | 181 | 0 | 0 |
| AST semantic frontend | 303 | 302 | 68 | 68 |
| Whole process | 3,720 | 3,703 | 2,207 | 2,195 |

One warmup per binary/workload preceded five sequential fresh processes each;
binary order alternated by round. No builds/tests ran during timing and
`WYN_EGGLOG_PROFILE` was unset. Tinyporto retains its Rust/WGPU host, graphics,
and SPIR-V target; fluid retains graphics, subgroups, SPIR-V, and its WHL host.
Tinyporto ranges were 3.642-3.733 s before and 3.626-3.760 s after; fluid ranges
were 2.136-2.360 s before and 2.148-2.335 s after. Filesystem caches were warm.
Raw samples, binary hashes, patch, harnesses, diagnostics, and gate results are in
[frontend/host evidence](profiles/2026-10-28-compiler-frontend-host.json).

The two behavior-preserving changes are small. Helper candidate selection
borrows bodies while checking eligibility and clones only selected candidates.
Its current Tinyporto cost is 14 ms. Rust host generation gathers references and
input names from one parsed function AST rather than parsing each function twice.
Fixed support parsing is only 2.7 ms, so it was left alone. Current host work is
43 ms generating functions, 64 ms parsing/visiting references and input names,
54 ms assembling the final syntax, and 10 ms formatting.

The quick type-inference check found 174 ms checking Tinyporto user declarations,
40 ms checking module functions, and 9 ms resolving the final table. The full
inference parent is 229 ms. No obvious repeated-work issue was established and
inference behavior is unchanged.

Helper expansion includes four fixed-point rounds of call expansion, constant
folding, normalization, and dead-definition cleanup. A separate temporary probe
split rewriting into approximately 54 ms actual call expansion, 38 ms constant
folding, and 25 ms normalization. These diagnostic clocks were removed before
full gates; they are not paired performance measurements. Current whole-pass
median is 155 ms, versus the preliminary single-run 143-ms result.

Full gates passed: workspace formatting/all-target build and **1,504 tests**
(zero failures, 14 existing ignores), standalone builds/tests for all seven
non-workspace crates and both parser manifests, WASM formatting/browser-target
check, **123 SPIR-V** validations, **122 WGSL** validations with one expected
skip, and the library package compile/validation runner. Timing output is gated
by `-v`. Final paired shaders and generated hosts are byte-identical for both
workloads. GPU execution was not run. Test environment uses the same writable
workspace temporary directory and 64-MiB Rust stack as the previous gates.

Production Rust is **+78 net lines: 128 added, 50 removed**. Of the net increase,
18 lines implement the two optimizations and 60 add optional phase timing.
There is one local `wyn-base` dependency entry and two corresponding lockfile
edges; no external dependency or package version changed.

### Next: fluid fusion reuse

The final paired profile puts fusion at **745 ms** in fluid; the previous
820-ms figure below belongs to the preceding measurement session. Existing
separate engine diagnostics show **32 contractions and 33 candidate rounds**.
Each native path query currently scans its entire edge relation to rebuild
adjacency. These predicates operate on fusion groups/operations, not scalar AST
nodes: the saved diagnostic contains 488 source operations and 383 edges in
each final edge relation. The granularity is appropriate; repeated construction
and proof demand are the first targets.

1. Share immutable `GroupEdge`/`GroupBefore` adjacency within a candidate round;
   discard it after every contraction, including absorbed length observers.
   Scope it to the graph/compilation as well as the round.
2. Demand `ElementFusion` only for candidate pairs. Its two producers currently
   cross current groups, and fluid accumulated 8,617 compatibility facts. This
   is a fact-count finding, not a measured speedup.
3. Measure path calls, adjacency builds, traversal work, and per-round candidate
   counts before adding further retention. Immutable plan summaries and result
   counts already persist. Endpoint-local proofs may be reusable by immutable
   plan identities, but path/barrier answers need broader invalidation: merging
   intermediate groups can change reachability between untouched endpoints.

Preserve greedy choice order and compare selected plans and generated outputs.
The **250-300 ms fusion budget remains a target**, not an established benefit of
adjacency reuse. No fusion code is part of the frontend/host change.

## Previous structural/output measurements

The snapshot below compares `46e696b5` with `ac0cfef0`. It is retained for the
structural/output optimization's impact and remaining phase detail; its times
are from the preceding session, not the latest frontend/host measurements.

Structural/output snapshot: **both workloads without `-O`**, after buffering SPIR-V output
and reducing structural-planning work. Paired median compilation times are
**4.445 -> 3.893 s for Tinyporto (12.4% less time)** and
**2.697 -> 2.392 s for fluid (11.3% less time)**.

Both remain above the two-second goal: approximately **1.89 s** still needs to
be removed from Tinyporto and **0.39 s** from fluid. The earlier `-O` Tinyporto
measurements belong to a different configuration; historical results remain in
[evaluation evidence](profiles/2026-10-28-compiler-evaluation.json),
[storage report](compiler-profile-storage-2026-10-28.md), and Git history.

## Measurement method

Release CLI on the same Windows i7-9700 host. One warmup per binary/workload,
then **five fresh processes each**, with binary order alternating by round.
The first three pairs were extended after a broad Tinyporto timing spike; all
samples were retained. No builds or tests ran during timing. Filesystem caches
were warm; this is a full compilation in each fresh compiler process.

Tinyporto retains graphics and its Rust/WGPU host. Fluid retains graphics,
subgroups, and its WHL host. Both target SPIR-V; neither command includes `-O`.
The baseline was rebuilt from clean revision `46e696b5`. The source patch,
binary hashes, raw samples, and harnesses are recorded in
[structural/output evidence](profiles/2026-10-28-compiler-structural.json).

Tinyporto ranges: **4.286-5.536 s before**, **3.722-4.124 s after**. Fluid ranges:
**2.662-2.799 s before**, **2.304-2.479 s after**. Desktop activity and hash/order
variation remain uncontrolled. Small changes in unrelated phases should not be
attributed to this patch.

## Change impact

| Measurement | Tinyporto before | Tinyporto after | Fluid before | Fluid after |
| --- | ---: | ---: | ---: | ---: |
| Artifact/host output | 474.0 ms | 223.0 ms | 281.0 ms | 3.0 ms |
| Operation ordering, within placement | 228.5 ms | 2.0 ms | 35.7 ms | 0.7 ms |
| Execution placement | 428.9 ms | 214.4 ms | 192.8 ms | 185.0 ms |
| Structural scheduling | 647.7 ms | 478.7 ms | 352.8 ms | 269.0 ms |
| Interface preparation | 243.1 ms | 243.7 ms | 127.5 ms | 134.5 ms |
| **Whole process** | **4.445 s** | **3.893 s** | **2.697 s** | **2.392 s** |

Nested rows overlap. Timers are summed within a run before taking medians;
independent phase medians need not sum exactly to the whole-process median.

SPIR-V output now uses `BufWriter` with an explicit, fallible flush. The byte
order and emitted words are unchanged. Tinyporto's remaining output time is
mostly **219 ms generating Rust host code**; fluid's whole output phase is 3 ms.

Operation-level ordering was the right granularity. The avoidable work was
repeated prerequisite/satisfied-set propagation and selecting one ready
operation per Egglog round. The existing stable topological sorter now computes
the same data/effect order once per region, and only final order/adjacency facts
enter Egglog. Cycle diagnostics are preserved. The original ordering program
is retained only as a regression-test oracle.

Representation/layout rules reuse the existing indexed `PlanCallArgument`
facts instead of repeatedly decoding argument vectors. A flattened
`SourceProjectionFields` index retains the exact original projection parent and
index. It is deliberately distinct from `Projection`, which can also contain
alias-forwarded parents. The proof conditions and selected layouts are unchanged.

Separate engine diagnostics put Tinyporto's old `order-select` search at about
205 ms. That ruleset is gone. Its representation-rule search fell from about
107 to 29 ms, and readout search from 95 to 13 ms. These are diagnostic engine
statistics, not partitions of the wall-clock timers. Interface preparation has
not materially improved and remains a separate investigation target.

## Current phase breakdown

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Import TLC | 259 | 99 |
| Structural analysis | 80 | 27 |
| Fusion | 242 | 820 |
| Execution placement | 214 | 185 |
| Structural scheduling | 479 | 269 |
| Interface preparation | 244 | 134 |
| Basic scalar optimization/extraction | 541 | 160 |
| Host preparation | 11 | 4 |
| Evaluation planning | 281 | 52 |
| Emit SSA | 131 | 88 |
| **Whole Egglog-through-SSA group** | 2,583 | 1,895 |

The largest current Fluid phase is **fusion: 820 ms**, including **516 ms**
choosing candidates, **118 ms** computing dependencies, and **126 ms** committing
contractions. Fusion was not changed in this patch.

### Basic scalar optimization

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Project scalar graph | 80 | 45 |
| Import scalar expressions | 95 | 23 |
| Scalar rewrite schedule | 43 | 11 |
| Effect summaries | 49 | 15 |
| Compact extraction costs | 200 | 42 |
| Reconstruct compact DAG | 22 | 7 |
| Copy selected DAG | 11 | 3 |
| Whole basic scalar phase | 541 | 160 |

Only 43 ms of Tinyporto's 541-ms basic scalar phase executes the rewrite
schedule. Graph handling and extraction remain the larger opportunity without
`-O`.

### Evaluation planning

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Collect facts | 12 | 3 |
| Initialize schema | 2 | 2 |
| Import facts | 55 | 13 |
| Reachability rules | 81 | 6 |
| Availability facts | 2 | <1 |
| Placement rules | 24 | 3 |
| Placement classes/sites | 19 | 3 |
| Target selection | 14 | 3 |
| Read targets/common expressions | 3 | <1 |
| Schedule anchors | 66 | 16 |
| Whole evaluation phase | 281 | 52 |

The previous shared-rank and placement-class optimization remains in place.
Tinyporto still spends approximately 81 ms on reachability and 55 ms importing
these facts. Its anchor scheduling is 66 ms, including 37 ms of rank rules.

### SSA and backend

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Prepare SSA values | 2.4 | 1.9 |
| Place floating SSA | 3.5 | 3.0 |
| Prune unreachable SSA | 0.5 | 0.4 |
| Simplify SSA | 36.9 | 30.5 |
| Local storage | 39.1 | 156.5 |
| Texture sampling | 0.2 | 0.2 |
| Validate types | 4.4 | 4.1 |
| Emit SPIR-V | 44.0 | 44.7 |
| Construct host program | 0.5 | 0.3 |
| Whole backend | 135.0 | 246.0 |

Both programs spend about 44 ms emitting SPIR-V. Backend arithmetic/serialization
is a smaller target than the remaining planning and frontend costs.

## Updated roadmap toward two seconds

1. **Fusion reuse for fluid.** Its 820-ms fusion phase is the first target for
   the remaining 392-ms gap. Build adjacency once per contraction round,
   invalidate after contraction, and measure which candidate facts can be
   retained for unaffected groups. The earlier 250-300 ms phase budget remains
   a target to test, not an established benefit of adjacency caching alone.
2. **Tinyporto's basic scalar/evaluation path.** Together these cost 822 ms.
   Start with graph projection/import, the 200-ms compact extraction-cost
   calculation, and repeated reachability proofs. Preserve the basic-mode
   arithmetic and safety policy.
3. **Tinyporto's frontend and host generation.** Parsing/type checking/TLC cost
   901 ms, plus 219 ms generating the Rust host. Profile type checking,
   helper inlining, and parsing/formatting the fixed Rust support code before
   selecting the next changes.
4. **Remaining structural planning.** Placement, scheduling, and interfaces
   still cost approximately 936 ms in Tinyporto and
   591 ms in fluid. Split rule installation from execution and time
   physical ABI/publication separately. Inspect cardinalities and demanded
   roots before changing more rules; operation ordering is now only 2 ms.

Per full-build budget, with headroom below the two-second requirement:

| Work | Tinyporto target | Fluid target |
| --- | ---: | ---: |
| Frontend and TLC | 500 ms | 170 ms |
| Graph import and analysis | 150 ms | 80 ms |
| Fusion and structural planning | 550 ms | 1,000 ms |
| Basic scalar and evaluation | 250 ms | 130 ms |
| SSA and backend | 150 ms | 200 ms |
| Host/output and remaining overhead | 200 ms | 180 ms |
| **Total budget** | **1.80 s** | **1.76 s** |

These are engineering budgets, not demonstrated savings. Each subsequent
change needs paired no-`-O` measurements, output checks, full gates, and a net
production-line count. Keep the same targets and generated hosts in the
acceptance benchmark.

## Code size and validation

This patch is **+20 net production lines: 75 added, 55 removed**. Buffered output
is +1; structural planning is +19. Counts include comments and test-module
registration, and exclude tests/documentation. The old 32-line ordering program
is now a test-only oracle rather than production code.

All full gates passed:

- Workspace formatting, all-target build, and tests: **1,504 passed, zero failed,
  14 existing ignores**, including all 127 compiler fixture files.
- All-target builds and tests for `wyn-wasm`, viz, tephra, spirv-validator,
  wyn-doc, mdbook-wyn-highlight, treereduce-wyn, and both tree-sitter manifests.
- WASM formatting and locked browser-target check.
- **123 SPIR-V files passed**, **122 WGSL files passed**, one expected WGSL skip;
  the library package compile/validation runner passed.
- Ordering tests match the original implementation on generated DAGs and
  explicitly check ready priority, effect edges, region boundaries, and cycles.

The final paired shaders and generated hosts are **byte-identical** for both
workloads. Both versions' shaders pass `spirv-val`. Other diagnostic runs can
vary in fluid entry-interface ordering. GPU execution was not run.

Tests used `RUST_MIN_STACK=67108864` and a writable workspace temporary directory.
Parser tests followed shader sweeps; the package runner followed workspace
tests. Full logs, binaries, and outputs are under
`tmp/compiler-structural-opt-2026-10-28`. Diagnostic runs use
`WYN_EGGLOG_PROFILE=1` and are kept separate from timing samples.
