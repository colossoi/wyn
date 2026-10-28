# Detailed CLI compiler timings

Follow-up: the [local-storage optimization](compiler-profile-storage-2026-10-28.md)
reduces fluid's storage phase by 9.65x and whole compilation time by 32.6%, with
all full gates passing. The measurements below describe the earlier baseline.

The fluid simulator spends **1.46 seconds in SSA local-storage analysis**, of
which **1.34 seconds executes Egglog storage rules**. Actual SPIR-V emission
takes **42 ms**. Tinyporto spends **2.13 seconds in scalar optimization**,
including **1.09 seconds computing extraction costs**, followed by **1.20
seconds of evaluation planning**.

These measurements refine the two largest groups in the
[coarse profile](compiler-profile-2026-10-28.md). They support prioritizing
structural analysis costs: the main timer named `egglog` is not synonymous with
arithmetic optimization, and backend preparation runs another Egglog program.

## Egglog through SSA generation

Median milliseconds, three fresh processes after one warmup, normal project
options: Tinyporto with `-O` and a Rust/WGPU host; fluid without `-O`, with
`--subgroups` and a WHL host. Both target SPIR-V.

| Phase | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Import TLC into the compiler graph | 261 | 96 |
| Structural analysis | 77 | 25 |
| Fusion | 242 | 795 |
| Execution placement | 445 | 212 |
| Structural scheduling | 620 | 344 |
| Interface preparation | 242 | 120 |
| Scalar optimization including extraction | 2,130 | 146 |
| Host preparation | 17 | 4 |
| Evaluation planning | 1,196 | 78 |
| Emit SSA from graph facts | 228 | 79 |
| Other untimed work within the group | 99 | 50 |
| **Whole Egglog-through-SSA group** | **5,562** | **1,979** |

These are sequential groups. Parent timings include their children in the
tables below. Independent medians and untimed cleanup mean the displayed rows
need not sum exactly to the group median.

### Scalar optimization and extraction

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Project scalar graph | 78 | 41 |
| Import scalar expressions | 114 | 19 |
| Execute scalar rewrite schedule | 485 | 10 |
| Compute effect summaries | 163 | 14 |
| Compute compact extraction costs | 420 | 39 |
| Reconstruct compact DAG | 27 | 6 |
| Compute inlined extraction costs | 670 | Not run without `-O` |
| Reconstruct inlined DAG | 33 | Not run without `-O` |
| Compare and copy selected DAGs | 67 | 3 for copying only |

The two cost computations together take **1.09 seconds** in Tinyporto. They use
different cost models over the scalar graph. Reconstructing and selecting the
resulting DAGs is much smaller. Arithmetic rewrite execution itself is **485
ms**, so the 2.13-second scalar parent is not all rewrite search.

The extraction parent is 1,231 ms in Tinyporto and 49 ms in fluid. These parent
figures overlap the extraction-cost/reconstruction rows above and must not be
added to them.

### Evaluation planning

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Collect input facts | 16 | 3 |
| Initialize evaluation schema | 2 | 2 |
| Import facts into evaluation graph | 85 | 12 |
| Reachability rules | 123 | 5 |
| Prepare availability facts | 3 | <1 |
| Placement rules | 40 | 3 |
| Prepare placement-site facts | 140 | 5 |
| Frequency, lifetime, and target rules | 160 | 11 |
| Read targets and common expressions | 6 | <1 |
| Schedule evaluation anchors | 606 | 35 |

Tinyporto's 606-ms anchor scheduling breaks down into **59 ms** ranking rules,
**87 ms** building/importing requests, **254 ms** anchor rules, and **202 ms**
reading/grouping schedules. There are four request batches. Thus both rule
execution and the code feeding and querying the graph deserve attention.

### Fluid fusion

Fluid performs **33 candidate-selection rounds and 32 contractions**; Tinyporto
performs two rounds and one contraction.

| Fusion schedule work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Dependencies | 89 | 111 |
| Find and choose candidates | 52 | 503 |
| Commit contractions | 9 | 127 |

The separate engine-statistics run attributes approximately **435 ms** to
`fusion-candidates` search/merge/rebuild. Candidate selection is the largest
measured component of fluid's structural fusion.

## SSA preparation and SPIR-V generation

This flattens the nested backend timers into non-overlapping work:

| Work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Prepare SSA values | 2.7 | 1.9 |
| Place floating SSA instructions | 3.9 | 2.7 |
| Prune unreachable SSA | 0.4 | 0.3 |
| Simplify SSA values | 85.4 | 28.8 |
| **Local-storage analysis and rewriting** | **39.9** | **1,458.8** |
| Publish texture sampling | 0.2 | 0.2 |
| Validate backend types | 5.0 | 4.0 |
| **Emit SPIR-V** | **47.2** | **41.8** |
| Construct host program | 0.4 | 0.3 |
| **Whole backend group** | **190** | **1,542** |

Local storage accounts for approximately **95% of fluid's backend group**.
Tinyporto's backend is instead split mainly between SSA simplification,
SPIR-V emission, and local storage.

### Fluid local storage

There are **19 analyzed bodies** in fluid and **five** in Tinyporto. Totals
below sum repeated per-body timers within each run before taking medians.

| Analysis work | Tinyporto ms | Fluid ms |
| --- | ---: | ---: |
| Collect SSA inputs | <1 | <1 |
| Initialize graph and load schema | 12 | 47 |
| Import facts | 1 | 24 |
| **Execute storage rules** | **12** | **1,340** |
| Read results | <1 | <1 |

Fluid spends another approximately **15 ms** splitting parameters, removing
loads, allocating, checking, and rewriting. Parent analysis time also includes
graph destruction and other untimed overhead.

Four generated kernels account for roughly **1.30 seconds**, approximately
**89% of fluid's entire local-storage phase** (median of their per-run sums):

| Generated kernel | Local-storage time ms |
| --- | ---: |
| `fluid__compute_7_partials` | 275 |
| `fluid__compute_7_partials_2` | 276 |
| `fluid__compute_7_combine` | 363 |
| `fluid__compute_7_combine_2` | 369 |

A separate diagnostic invocation enabled Egglog's own rule statistics. Across
the 19 graphs, `storage-liveness` reports approximately **1.005 seconds** of
search/merge/rebuild. The highest aggregated rule-search costs are:

| Rule purpose | Search/apply ms | Reported matches |
| --- | ---: | ---: |
| Check whether an origin allocation is used later through an alias | 300 | 0 |
| Propagate past an allocation while a related array version is live | 208 | 1,812 |
| Propagate `AfterOrigin` along control flow | 165 | 272,854 |
| Propagate `Live` across ordinary instructions | 66 | 77,322 |

The **300-ms zero-match join** is a concrete first investigation target. A
negative result can still be necessary for correctness; zero matches do not
justify removing the rule. Inspect join ordering, the size of intermediate
relations, and opportunities to restrict the analysis to relevant origins and
uses. The supplied statistics identify cost, not a proven replacement algorithm.

Engine statistics are from one additional diagnostic run and rounded to whole
milliseconds. They exclude some scheduling/engine overhead and are not a
partition of the wall-clock timers. The main compiler graph's statistics are
printed at structural scheduling; local-storage graphs are reported separately.
Their sums must not be added to the measured phase times.

## Instrumentation and validation

The existing timing scope is now shared through `wyn-base::timing`, so CLI
`-v` covers backend preparation as well as Egglog. New spans cover scalar DAG
reconstruction/selection, evaluation planning, fusion rounds, backend phases,
and local-storage analysis per body. Compiler rules, pass order, and
optimization decisions are unchanged. Library calls remain quiet unless their
timing scope is explicitly enabled.

Use the commands in the coarse report with the updated release compiler to
reproduce the phase breakdown. For additional diagnostic statistics:

```powershell
$env:WYN_EGGLOG_PROFILE = '1'
& .\target\release\wyn.exe build fluid-simulation --graphics --subgroups -v -o tmp/compiler-profile/fluid.spv
Remove-Item Env:WYN_EGGLOG_PROFILE
```

Keep those diagnostic runs separate from measurements. Repeated timer names
are summed within a run; nested parent and child timers are not added together.
[Detailed evidence](profiles/2026-10-28-compiler-fine.json) preserves commands,
all measured runs, raw logs, the diagnostic statistics, and the measurement
harness. The local harness and outputs are in
`tmp/compiler-profile-fine-2026-10-28`.

The release build and all nine `wyn-base` unit tests passed. The timing test
checks nested scopes, unwind restoration, and thread isolation. CLI smoke
checks verified that `-v` controls the added diagnostics and leaves SPIR-V/WGSL
output identical. Both large SPIR-V outputs pass `spirv-val`. Tinyporto is byte
identical to the original coarse-run output. The final fluid diagnostic output
differs only in the ordering of interface IDs on 59 `OpEntryPoint` instructions;
all other 27,472 disassembly lines are identical.

The same CPU, revisions, and project options as the coarse run were used.
Final wall times were 7.131–7.238 seconds for Tinyporto and 3.960–4.085 seconds
for fluid. Timer logging, hashing/order variation, and desktop scheduling limit
small cross-run comparisons. No runtime optimization or GPU execution was
attempted, and the full regression suite was not run.

## Recommended order of investigation

1. Fluid's `storage-liveness` joins and control-flow propagation in the four
   expensive partial/combine kernels.
2. Fluid's repeated fusion-candidate selection.
3. Tinyporto's evaluation scheduling, especially request import and result
   queries alongside its rule execution.
4. Tinyporto's two extraction cost calculations within arithmetic optimization.

An order-of-magnitude reduction in structural analysis is a useful target,
not a demonstrated result of these measurements. The profiles identify where
to test that target without changing arithmetic optimization quality.
