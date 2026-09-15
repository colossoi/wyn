# EGIR planning profile

Profiled 2026-09-14 at commit `b9d85fa59cf29241879308d72104051f0b902e67` on Windows, using Cargo's optimized `profiling` profile.

## Indexed result lookup and monotone dependence updates

The first two parts of the [data structure design](C:/Users/gmiller_amilarcap/dev/wyn/docs/egir-invariant-structures.md) are implemented:

- `Facts::add_body` indexes results directly to fusion ports during its existing construction walk. The maps use borrowed result-tree keys and preserve direct-binding precedence and origin fallback order. The abstraction review removed the separate `ResultProvenance`, its IDs and occurrence records, and the added `GraphAnalysis` cache.
- `ResultTree` stores complete child trees and exposes borrowed fields and leaves. Ordinary `&ResultBinding` references are sufficient; the separate `ResultTreeRef` was removed. Fallible borrowed traversal creates projected ports without intermediate owned leaf records. `single_value` counts returned leaves without allocating, including the distinction between one return plus place destinations and repeated return leaves.
- Dependence propagation uses a private, pre-seeded `GrowingFacts` table whose only update is an in-place join at an existing key. Reads borrow facts, the fourteen provenance categories use `SourceSet(u16)`, and constant/parameter/effect seeds are initialized once. Missing parameter seeds remain conservative; missing required graph keys produce analysis errors. Solver sweeps and existing loop-dependency sets remain in place.

Seven alternating runs per variant compare this patch with the immediately preceding empty-dependency fast-path binary. These are new measurements of both variants, rather than comparisons against an older timing session.

| Input | Plan before (ms) | Plan after (ms) | Reduction |
| --- | ---: | ---: | ---: |
| mountains | 38.532 | 24.048 | 37.6% |
| shadow_test_f | 22.688 | 18.296 | 19.4% |
| ui_test_5 | 28.094 | 21.142 | 24.7% |
| particles | 3.027 | 2.869 | 5.2% |
| filter_constellation | 2.010 | 1.886 | 6.2% |
| scan_compute | 1.398 | 1.362 | 2.6% |
| hist_atomic | 1.003 | 0.958 | 4.5% |
| filter_then_map | 1.868 | 1.797 | 3.8% |
| tinyporto `wyn/main.wyn` | 181.415 | 99.827 | 45.0% |

Tinyporto benefits outside `egir_plan` because semantic optimization and stage lifting share these analyses:

| Tinyporto measurement | Before | After |
| --- | ---: | ---: |
| Driver-reported build time, median | 0.82 s | 0.64 s |
| Sum of timed compiler phases, median | 721.835 ms | 541.962 ms |
| `egir_plan` | 181.415 ms | 99.827 ms |
| Residency | 127.933 ms | 48.112 ms |
| Topology policy / stage lifting | 115.694 ms | 34.631 ms |
| Semantic optimization | 56.004 ms | 41.287 ms |
| Physicalization | 24.634 ms | 24.878 ms |

All 126 shader compilations succeeded. Tinyporto source hashes were unchanged throughout the comparison. The emitted SPIR-V modules have matching entry points and opcode counts across both variants and all seven runs for every input. Binary hashes vary within the baseline as well as the new binary, so byte equality is not an equivalence check here; matching opcode counts also does not prove runtime equivalence.

The new regression cases cover origin/alias ordering, direct-binding precedence, repeated return leaves versus place destinations, and monotone updates on seeded keys. They passed in the final workspace suite.

### Final validation gates

| Gate | Result |
| --- | --- |
| `cargo fmt --all` and formatting check | Passed |
| `cargo test --offline --workspace` | 1,859 passed, 0 failed, 19 ignored, including doc tests |
| `scripts/validate_testfiles.ps1 -TrackedOnly` | 92 SPIR-V compilations and validations passed |
| Same corpus script with `-Wgsl` | 91 passed; `miner` skipped by the script because it uses linked SPIR-V helpers |
| `scripts/check_wasm.ps1 --offline` | Both excluded Wasm crates passed their locked wasm32 checks |
| Tinyporto debug build and `spirv-val` | Passed; profiled optimized output also validated |
| Core compiler checks, including `--tests`, and optimized profiling build | Passed |
| Diff whitespace check | Passed |
| `cargo clippy --offline --workspace --all-targets -- -D warnings` | Failed on existing `large_enum_variant` in `wyn-pipeline-descriptor::Pipeline` and denied `expect` calls in `wyn-spirv/src/lib_tests.rs` |
| `cargo clippy --offline -p wyn-core --all-targets` | Failed on 67 existing denied `unwrap`/`expect`/`panic` uses in `egir/inlining_tests.rs` and `wgsl/ssa_lowering_tests.rs` |

The Clippy error sites are unchanged from the baseline. No lint settings were relaxed. Gate logs are saved as `tmp/egir-plan-profile/gate-*.log`. Tinyporto source hashes remain unchanged.

Artifacts are under `tmp/egir-plan-profile/`: `before-invariant-structures.exe`, `profile_egir_plan.exe`, `compare-invariants.ps1`, `consolidated-comparison.json`, and `consolidated-compare-{before,after}-{1..7}/`. `compare-invariant-output-shapes.cjs` compares emitted entry points and opcode counts.

The effect editor, stage-lift ownership changes, solver worklist, and scoped compact loop identities remain separate design follow-ups. No per-call effect-index or speculative graph-clone changes are included in these measurements.

## Earlier empty-dependency fast path

The first patch skips construction of a dependency index when its entire input has no SOACs and builds recipe-selection dependencies only after finding an eligible operation. Mixed inputs retain all bodies and ordinary effects. Scalar-only graphics still runs residency and can create scalar preludes.

Seven runs per variant, alternating which variant runs first, produced these median `egir_plan` times:

| Input | Before (ms) | After (ms) | Reduction |
| --- | ---: | ---: | ---: |
| mountains | 69.48 | 36.89 | 46.9% |
| shadow_test_f | 43.18 | 22.39 | 48.2% |
| ui_test_5 | 31.50 | 28.17 | 10.6% |
| tinyporto `wyn/main.wyn` | 201.17 | 181.04 | 10.0% |

Tinyporto was built with `--graphics` using its current working tree and the local package dependencies declared by its root `wyn.toml`. Its HEAD was `6be57ffd744c092ffa8cb82d436c970e2ffcb65b`; existing changes to `wyn/main.wyn` and `wyn/paint.wyn` were included. No tinyporto source was edited. Timed compiler phases totalled 732.72 ms before and 706.41 ms after. The largest remaining phases are:

| Tinyporto phase | Median after (ms) |
| --- | ---: |
| `egir_plan` | 181.04 |
| └ residency | 126.90 |
| └ physicalization | 25.09 |
| └ recipe finalization | 12.44 |
| └ resource allocation | 9.02 |
| └ schedule construction | 6.30 |
| topology policy / stage lifting | 115.25 |
| type checking | 96.58 |
| semantic optimization | 56.03 |
| EGIR to SSA | 49.27 |

Recipe finalization fell from 32.51 to 12.44 ms. Residency remained essentially unchanged (128.09 to 126.90 ms): tinyporto has a mixed input, so it still needs a dependency index.

The earlier fast-path core run reported 1,705 passed and 17 ignored. All 92 corpus outputs plus tinyporto passed SPIR-V validation, and four representative WGSL outputs passed validation. Formatting checks passed. Strict Clippy stopped on the existing `large_enum_variant` diagnostic in the unchanged `wyn-pipeline-descriptor::Pipeline` enum. The final gates above cover the complete reviewed patch.

Baseline and optimized binaries, comparison scripts, logs, and all phase medians are in `tmp/egir-plan-profile/`; see `comparison.json` and `comparison-summary.txt`.

### Hotspots identified after the fast path

Detailed tinyporto traces use the saved baseline profiling binaries. The residency, dependence, stage-lifting, and physical-call implementations are unchanged by the fast-path patch. Baseline recipe-selection costs have already been reduced and are not counted as remaining savings below.

1. **Repeated result-origin scans: measured 54.29 ms during residency.** `Facts::port` made 16,457 fallback searches across effect results in the fine trace. Its outer dependency construction visited 50 bodies over four residency iterations. Build an index for result bindings/single return values once per body, canonicalize the queried value once, and use the value's existing `result_origins()` to query the index. `ResultBinding` already implements `Eq` and `Hash`, so keys can borrow the existing result trees. Preserve direct-producer precedence and the first matching result's original order; do not clone result trees just to key the map.
2. **Allocation inside that scan.** `value_has_result_origin` calls `ResultBinding::single_value`, which calls `values()` and allocates a vector. Replace this with a traversal that records zero, one, or multiple return values without allocating. This is part of the 54.29 ms above, not an additional measured saving. This shared helper also appears outside semantic dependency construction.
3. **Repeated dependence cloning: approximately 26 ms in residency's propagation loops.** `accumulate` clones the stored fact, clones it again to merge, then compares two complete facts. Each fact owns two hash sets. An in-place merge can update uniformity and insert missing set members while reporting whether anything changed. `edge_dependence` also eagerly evaluates `control.clone()` as a `map_or` default even when the condition exists. The 26 ms includes propagation work; cloning alone has not been isolated.
4. **Rebuilding an existing lookup map for every call.** `reconcile_calls` builds `SideEffectIndex`, extracts call IDs, and discards their sites. `resolve_call` then rebuilds the entire index for each call to recover one site. Reuse sites from the first index, accounting for inserted preludes—for example, process sites from the end of each block so earlier indices stay valid. Physicalization was 25.09 ms overall; this subcost has not yet been isolated.
5. **Speculative graph clones during stage lifting.** `analyze_direct_entry_calls` clones every entry graph before discovering whether it has a mixed-stage call to inline. `prepare_lift` clones an entire function before checking whether its liftable frontier is empty. Both are repeated by the candidate-search loops. Analyze borrowed graphs first and clone only an accepted mutation candidate. Stage lifting's containing phase is 115.25 ms; clone-only cost has not yet been measured.

The first three findings motivated the result-lookup and dependence patch summarized above. Effect-order pair comparisons cost only about 1.17 ms across the baseline tinyporto plan and remain lower priority.

## Original profile conclusion

`egir_plan` is an orchestrator: resource allocation, residency resolution, staged-IR/recipe finalization, recipe scratch allocation, schedule construction, and physicalization. The driver times their combined cost. See [plan](../wyn-core/src/egir/parallelize/mod.rs#L218).

There is a concrete avoidable cost: building full semantic dependency graphs when there are no SOACs (array operations) to query. On mountains this costs approximately **31 ms of a 68 ms plan**. Dependence propagation is the next substantial cost. Scratch allocation, scheduling, publication, and final validation are not the concerning paths in this corpus.

The original investigation retained no production source changes. The measurements below describe that baseline; the implemented follow-up is summarized above.

## Measurements

All 92 tracked `.wyn` files directly under `testfiles/` and `testfiles/playground/` compiled successfully to SPIR-V. Playground files without explicit entries received the same image header used by the repository validation script. The untracked avatar experiment was excluded.

Across that one-pass sweep, `egir_plan` had a **1.58 ms median**, **15.38 ms 90th percentile**, and **68.47 ms maximum**. Its combined time was 570 ms, approximately 13% of the 4.46 s of timed compiler phases. These exclude process startup and other untimed driver work.

The following are medians from nine separate compilations per input, in milliseconds. Phase medians need not sum to the median total.

| Input | Allocate resources | Residency | Finalize recipes | Scratch | Build schedule | Physicalize | Total plan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mountains | 2.00 | 46.16 | 14.40 | 0.009 | 0.95 | 4.35 | **67.94** |
| shadow_test_f | 1.28 | 28.23 | 11.22 | 0.009 | 0.77 | 1.95 | **43.69** |
| ui_test_5 | 4.04 | 14.51 | 7.41 | 0.008 | 1.43 | 6.19 | **33.27** |
| particles | 0.56 | 0.35 | 0.35 | 0.008 | 0.35 | 1.17 | **2.91** |
| filter_constellation | 0.35 | 0.37 | 0.41 | 0.008 | 0.22 | 0.64 | **2.10** |
| scan_compute | 0.13 | 0.16 | 0.21 | 0.013 | 0.40 | 0.40 | **1.41** |
| hist_atomic | 0.15 | 0.22 | 0.16 | 0.006 | 0.16 | 0.24 | **1.02** |
| filter_then_map | 0.15 | 0.41 | 0.19 | 0.011 | 0.54 | 0.50 | **1.91** |

Five-run WGSL checks for mountains, particles, filter_constellation, and scan_compute all compiled successfully and showed the same pattern. Mountains' median plan was 66.16 ms. Five-run `--direct` checks for mountains, ui_test_5, and scan_compute also succeeded; mountains was 27.36 ms. Direct mode changes topology and scheduling policy, so this is context, not an equivalent-output optimization comparison.

## Findings, in priority order

### 1. Avoid constructing an empty SOAC dependency index

[residency_facts](../wyn-core/src/egir/allocation/residency.rs#L58) unconditionally constructs `SemanticGraph::for_bodies`. [analyze_parallel_scremas](../wyn-core/src/egir/parallelize/planning.rs#L42) unconditionally constructs another semantic graph before checking execution model and whether any segmented operation is eligible.

In the coarse instrumented mountains run:

- Two residency iterations built five graph bodies in total: **19.68 ms** in `SemanticGraph::for_bodies`.
- Recipe selection built three further graph bodies: **11.14 ms** in `SemanticGraph::new`.
- Shape counters confirmed **zero SOACs in every one of those eight bodies**. The resulting SOAC sites, consumers, captures, and residency-demand index are empty.

First optimization: an empty-result fast path when a dependency-index input contains no SOACs, plus lazy construction in recipe selection after cheap eligibility checks. Keep the stage-invariant scalar prelude logic running: mountains still legitimately creates a scalar prelude even without SOACs. Do not skip residency as a whole.

This identifies about 31 ms of unnecessary graph construction on this example, not a measured 31 ms end-to-end speedup. Check mixed-SOAC programs when implementing the shortcut.

### 2. Result-origin lookup repeatedly scans all effect results

In [Facts::port](../wyn-core/src/egir/semantic_graph/facts.rs#L221), a miss in the direct producer map falls back to scanning the complete `results` list with `value_has_result_origin`. Ports are cached by block scope and value, so the same value can incur work in several blocks.

A finer mountains trace counted **6,251 fallback searches, taking 13.66 ms** across the eight bodies. This is part of finding 1's dependency-graph cost, not additional savings. The same pattern appeared in shadow_test_f (10,482 searches) and ui_test_5 (3,168).

For inputs that actually require these graphs, consider indexing canonical values/result origins once per body. Preserve the current first-match and alias behavior. The absent-SOAC shortcut should come first because it eliminates the entire unnecessary construction on mountains.

### 3. Dependence propagation repeatedly clones and scans

[StageDependenceAnalysis::for_graph](../wyn-core/src/egir/stage_variance.rs#L172) repeatedly visits every block interface and node until convergence. [value_dependence](../wyn-core/src/egir/stage_variance.rs#L504) clones dependence sets, and [accumulate](../wyn-core/src/egir/stage_variance.rs#L520) clones the current fact twice before checking whether anything changed.

Mountains performed five entry analyses across its two residency iterations. The detailed repetitions recorded **26–29 full solver sweeps in total**, taking **13–16 ms**. These are totals across five analyses, not 29 residency rewrites. There was one successful scalar-prelude materialization followed by the final no-change iteration.

After the empty-index shortcut, investigate an in-place lattice merge and, if necessary, a worklist that revisits affected dependents. No nontermination or runaway residency loop appeared in the measured examples. Analysis invalidation after an actual graph rewrite is necessary; blindly caching across rewrites is inappropriate.

## Checks that did not justify changes

- The all-pairs effect-ordering loop in `Facts::add_body` is quadratic, including comparisons across different blocks. It consumed only **0.19 ms** across the eight mountains bodies. It is a scaling concern, not this profile's main bottleneck.
- Mountains' coarse trace spent **0.16 ms** publishing physical layout and **0.14 ms** in final physical validation. Removing these checks is not warranted by the measurements.
- Recipe scratch allocation was about **0.01 ms**. Kernel schedule construction was about **1 ms** on mountains.
- `GraphAnalysis` already lazily shares producer, interface, loop, and slice facts within an immutable graph snapshot. A blanket new cache would duplicate an existing mechanism.

## Method and artifacts

The baseline harness copies the CLI into an ignored scratch directory, invokes the same six public planning operations, and records sub-millisecond phase durations. Phase timers exclude their own reporting; the outer total includes six reporting calls and therefore has a small overhead, particularly visible for tiny inputs.

Coarse and fine function timers were temporarily added to the compiler, compiled, and then removed by restoring exact source bytes. Fine timings include instrumentation overhead and should explain the baseline timings rather than replace them. Windows sampling did not start successfully; these results use direct elapsed-time instrumentation, not statistical CPU samples. No GPU execution was measured.

Scratch artifacts are in `tmp/egir-plan-profile/`:

- `build.ps1`, `sweep.ps1`, `summarize.cjs`: baseline harness and measurements.
- `sweep/timings.json`, `repeats/timings.json`, `wgsl/timings.json`, `direct/timings.json`: recorded phase timings.
- `detail-mountains.log`: coarse nested profile.
- `detail/*.log`: fine profiles and graph-shape counts for five representative inputs, three compilations each.
- `instrument.cjs`, `timers.rs`, `build-instrumented.ps1`: temporary instrumentation and source restoration.

All 199 baseline compilations and 15 fine-profile compilations succeeded. These were compilation checks; the profiling run did not execute shaders or run the full unit-test suite.

Cleanup of the original investigation verified that all 18 temporarily instrumented source files matched their original bytes. The normal profiling binary was rebuilt and successfully compiled mountains with no profiling markers. Subsequent production changes are the fast paths described above.
