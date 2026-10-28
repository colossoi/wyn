# Detailed CLI compiler timings

Latest analysis: **both workloads without `-O`**, after buffering SPIR-V output
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
