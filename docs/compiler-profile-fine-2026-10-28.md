# Detailed CLI compiler timings

Updated after storage optimization **`60a400de`** and evaluation-planning
optimization **`edcd3590`**. All current tables below use the final paired run's
updated compiler. The [coarse report](compiler-profile-2026-10-28.md) and
[original fine-profile evidence](profiles/2026-10-28-compiler-fine.json) preserve
the earlier baseline.

Tinyporto evaluation planning is now **463 ms**, versus **1,224 ms** in the paired
baseline: **2.64x faster**, with total compilation **10.7% shorter**. Fluid
planning falls from **83 ms to 50 ms (1.66x)**. Fluid's previously optimized
local-storage phase remains about **151 ms**, rather than the original 1.46 s.

## Method

Release CLI, one warmup per binary/workload, then three sequential fresh
processes each, alternating before/after order by round. Tables show median
milliseconds unless otherwise labeled. Builds and tests finished before timing.
Tinyporto uses `-O`, graphics, and a Rust/WGPU host; fluid uses graphics and
subgroups without `-O`. Both target SPIR-V on the same Windows i7-9700 host.

Parent rows include their children; do not add nested tables together. Repeated
timers are summed within each run. Independent medians and untimed cleanup mean
rows need not sum exactly to parent medians. Small changes outside the optimized
phases can reflect desktop scheduling and normal run variation.

## Egglog through SSA generation

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Import TLC | 275 | 99 |
| Structural analysis | 85 | 26 |
| Fusion | 263 | 836 |
| Execution placement | 432 | 205 |
| Structural scheduling | 636 | 360 |
| Interface preparation | 244 | 130 |
| Scalar optimization including extraction | 2,254 | 156 |
| Host preparation | 16 | 4 |
| Evaluation planning | 463 | 50 |
| Emit SSA | 223 | 84 |
| Other work within the group | 100 | 50 |
| **Whole Egglog-through-SSA group** | **5,001** | **2,039** |

## Evaluation-planning improvement

The paired baseline already contains the storage optimization. Only the
subsequent evaluation-planning change differs between these two binaries.

| Measurement | Tinyporto before | Tinyporto after | Fluid before | Fluid after |
| --- | ---: | ---: | ---: | ---: |
| Evaluation planning | 1,224.5 ms | 463.3 ms | 83.4 ms | 50.3 ms |
| Anchor scheduling | 628.2 ms | 128.5 ms | 36.3 ms | 15.0 ms |
| Placement-site preparation | 139.5 ms | 28.5 ms | 5.8 ms | 3.0 ms |
| Target-selection rules | 163.3 ms | 13.9 ms | 11.0 ms | 3.0 ms |
| Whole compiler process | 7.458 s | 6.657 s | 2.906 s | 2.804 s |

Tinyporto process ranges were **7.422-7.504 s before** and **6.619-6.732 s after**.
Fluid ranges overlap: **2.736-2.910 s before** and **2.728-2.902 s after**. Its
3.5% lower whole-process median is an observation, not a precise attribution of
all that difference to this change; the evaluation phase itself saves 33 ms.

### Granularity

Expression-level facts are needed for dependency ordering and legal placement.
The excessive detail was replication of identical decisions across requests:

- The first Tinyporto anchor batch had **329,846 request/expression memberships
  for 26,984 distinct expressions**. Across all batches, 338,668 priority seeds
  were imported. Rank and condition are now read once per expression; the
  request-indexed priority/group tables and their copy rules are gone. Each
  request still retains its own roots, order, and permission to reorder.
- Tinyporto discovers **90,787 scope/expression queries**. After eligibility
  filtering, equivalent queries share **3,245 placement classes**, keyed by
  context, lexical scope, and required definitions. Candidate rows fall from
  **145,274 to 12,136**. Fluid candidate rows fall from **5,973 to 1,619** across
  847 classes. The same Egglog rules still choose execution frequency and
  lexical depth; each class's decision is expanded back to its members.

Effectful requests retain dependency postorder and separate choice groups.
Source leaves remain evaluation boundaries. The change does not approximate
arithmetic, remove effects/guard proofs, or change placement policy.

### Current evaluation costs

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Collect input facts | 16 | 3 |
| Initialize schema | 1 | 2 |
| Import facts | 86 | 13 |
| Reachability rules | 128 | 6 |
| Prepare availability facts | 3 | <1 |
| Placement rules | 41 | 3 |
| Prepare placement classes and sites | 29 | 3 |
| Frequency, lifetime, and target rules | 14 | 3 |
| Read targets/common expressions | 6 | <1 |
| Schedule anchors | 129 | 15 |
| **Whole evaluation-planning phase** | 463 | 50 |

Anchor scheduling now consists of shared rank rules and native request ordering:

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Compute shared expression ranks | 63 | 10 |
| Build anchor request orders | 35 | 2 |
| Read/group schedules | 26 | 1 |

The timer named `egglog anchors / import requests` now covers construction of
native request orders; those memberships are no longer imported into Egglog.
The old per-request `schedule-anchors` rules no longer execute. Remaining
reachability, import, and rank costs are still measurable; this change does not
establish that all remaining structural analysis is minimal.

## Scalar optimization and extraction

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Project scalar graph | 84 | 44 |
| Import scalar expressions | 121 | 21 |
| Execute scalar rewrite schedule | 493 | 11 |
| Compute effect summaries | 166 | 15 |
| Compact extraction costs | 481 | 41 |
| Reconstruct compact DAG | 28 | 8 |
| Inlined extraction costs | 707 | Not run |
| Reconstruct inlined DAG | 35 | Not run |
| Compare/select DAGs | 66 | Not run |

The compact and inlined cost computations are part of extraction, which is
included in scalar optimization. Fluid does not run the inlined search without
`-O`; its basic-path DAG copy takes 3.1 ms. Arithmetic rewrite execution is
only one part of the scalar parent.

## Fusion

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Dependencies | 95 | 117 |
| Choose candidates | 56 | 531 |
| Commit contractions | 8 | 131 |

Fluid still performs 33 selection rounds and 32 contractions; Tinyporto performs
two rounds and one contraction. These are sums of the repeated timer calls.

## SSA preparation and SPIR-V generation

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Prepare SSA values | 2.8 | 2.0 |
| Place floating SSA instructions | 3.9 | 2.6 |
| Prune unreachable SSA | 0.4 | 0.3 |
| Simplify SSA values | 85.2 | 29.1 |
| Local-storage analysis and rewriting | 34.3 | 151.0 |
| Publish texture sampling | 0.2 | 0.2 |
| Validate backend types | 5.1 | 4.0 |
| Emit SPIR-V | 48.5 | 43.5 |
| Construct host program | 0.5 | 0.3 |
| **Whole backend group** | 187.0 | 236.0 |

Local storage now uses storage-event groups rather than every instruction:
Tinyporto has 17 groups for 1,020 instructions, and fluid has 533 for 16,233.
All CFG blocks and required array/alias facts remain. Allocation-origin
reachability is restricted to eligible reuse candidates. See the
[storage optimization report](compiler-profile-storage-2026-10-28.md) for its
paired **9.65x phase improvement** and the associated **32.6% fluid compile-time
reduction**. Those historical improvements must not be added arithmetically to
the new paired percentages.

There are five analyzed storage bodies in Tinyporto and 19 in fluid:

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Collect inputs | 0.3 | 2.8 |
| Initialize graph/schema | 12.8 | 47.8 |
| Import facts | 0.3 | 7.9 |
| Execute storage rules | 5.4 | 58.9 |
| Read results | 0.0 | 0.3 |

Schema setup and untimed graph cleanup now account for a larger share of this
phase. The storage parent also includes allocation, verification, rewriting,
and ownership work.

## Production line counts

| Optimization | Added | Removed | Net production lines |
| --- | ---: | ---: | ---: |
| Storage granularity | 134 | 45 | +89 |
| Evaluation ranks and placement classes | 78 | 50 | +28 |

Counts include comments and diagnostic counters, and exclude tests,
documentation, and the earlier general timing instrumentation. Evaluation's
production changes are confined to `scalar/evaluation.rs`, `evaluation.egg`, and
`placement.rs`.

## Validation and reproduction

All full gates passed on the evaluation change:

- Workspace formatting, all-target build, and tests: **1,501 passed, zero failed,
  14 existing ignores**, including all 127 compiler fixture files.
- All-target builds and tests for `wyn-wasm`, viz, tephra, spirv-validator,
  wyn-doc, mdbook-wyn-highlight, treereduce-wyn, and both tree-sitter-wyn manifests.
- WASM formatting and locked `wasm32-unknown-unknown` check.
- **123 SPIR-V files passed**; **122 WGSL files passed**, with one expected
  linked-SPIR-V-helper skip; the library package compile/validation gate passed.
- Targeted tests verify effectful ordering, grouped choice boundaries, separate
  source anchors, and agreement between grouped and individual placement queries.

Both final paired benchmark shaders and both generated hosts are byte-identical.
All four shaders pass `spirv-val`. Other diagnostic runs can differ in fluid's
`OpEntryPoint` interface-ID ordering, as documented in the storage report.
GPU execution was not performed.

Windows tests used `RUST_MIN_STACK=67108864`; the workspace tests used a writable
workspace directory for `TMP`/`TEMP`. Parser tests ran after the shader sweeps,
and the package runner followed workspace tests to avoid file-lifecycle races.

The existing CLI `-v` flag emits timings. `WYN_EGGLOG_PROFILE=1` additionally
prints evaluation cardinalities and engine diagnostics; keep those invocations
separate from timed runs. Commands, raw logs, all samples, binary hashes,
production diffs, and gate results are in
[evaluation evidence](profiles/2026-10-28-compiler-evaluation.json). The local
harness and full logs are under `tmp/compiler-evaluation-opt-2026-10-28`.
