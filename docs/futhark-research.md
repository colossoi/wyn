Futhark inlining, fusion, and repeated radix passes
=================================================

Research date: 2026-09-19. This investigation changes documentation and adds reproducible experiments; it does not change Wyn's compiler policy.

Implementation follow-up: Wyn now uses candidate-local reachability checks over the current quotient graph, replacing the transitive-closure/“between” sets described below. Early SOAC-helper inlining has been restored and the post-fusion helper-expansion workaround removed. References to Wyn's old analysis below describe the failure investigated by this report.

The central finding
-------------------

Keeping a large helper out of line saves work when it remains a function call. It does **not** solve repeated fusion analysis after that helper is expanded inside a map. The objection to the earlier explanation was correct.

Futhark's ordinary pipeline still exposes nested array computation by inlining before fusion. It does not generally fuse a helper once, expand the fused result everywhere, and prohibit further fusion. I found no cache of structurally identical fusion plans in the examined inliner and fusion implementation. Its practical advantages here are a dependency-graph fusion algorithm, selective inlining, and representing repeated radix passes as a loop. Recent function lifting provides another option, with significant compiler machinery and optimization tradeoffs.

The most relevant lesson for Wyn is to fix the expensive dependency representation without sacrificing the visibility needed for nested parallelism. Sharing bodies is useful, but making every helper an optimization boundary is a semantic restriction on optimization opportunities, not a complete replacement for scalable analysis.

Evidence and versions
---------------------

Experiments use the official Linux release **Futhark 0.27.1**, compiler commit `f547a39abb1f3773f7cdb324ec8ea7cb8cf13ea0`. I inspected its exact source and also checked master at `7cd43302b8b406df021815aaee92e3f81337429c` (September 18, 2026). Source links below are pinned to the measured release. The inliner is identical between those snapshots. [Release and binary](https://github.com/diku-dk/futhark/releases/tag/v0.27.1).

Older papers describe different compiler generations. In particular, historical statements that all functions are inlined do not describe every call in the current compiler. The 2024 [inlining article](https://futhark-lang.org/blog/2024-10-28-inlining.html) explains the move toward selective inlining, including compile-time benefits for repeated top-level calls. That improvement does not promise shared optimization of bodies expanded inside maps.

What happens before fusion
--------------------------

The [standard pipeline](https://github.com/diku-dk/futhark/blob/f547a39abb1f3773f7cdb324ec8ea7cb8cf13ea0/src/Futhark/Passes.hs) runs simplification, conservative inlining, more simplification, aggressive inlining, CSE/simplification, then SOAC fusion. GPU flattening follows the standard pipeline.

The [inliner](https://github.com/diku-dk/futhark/blob/f547a39abb1f3773f7cdb324ec8ea7cb8cf13ea0/src/Futhark/Optimise/InliningDeadFun.hs) makes several distinct decisions:

* Conservative inlining selects functions called once and sufficiently small or explicitly inline functions.
* Aggressive inlining additionally selects array-related functions reachable from SOAC bodies. The condition includes array arguments, array results, array-valued locals, and indexing; it is broader than just “contains a map.”
* Selection propagates through the call graph. Selected callees are expanded leaf-first, with periodic simplification and CSE to limit intermediate growth.
* Selection is by function name, not a promise that only one particular call site is expanded. Explicit `noinline` attributes can prevent expansion.

Thus a substantial radix pass called 30 times at top level can remain one function. Put those calls inside a map, and normal aggressive inlining exposes 30 bodies to fusion. This is exactly what the fixtures below show.

“Bottom-up” needs qualification. Leaf-first call-graph inlining means replacing calls with callee bodies; it does not mean caching their fusion plans. Current fusion also descends into nested bodies and repeatedly tries transformations. Neither mechanism establishes that all identical call subtrees must receive the same context-independent fusion decision.

What the fusion algorithm stores
--------------------------------

The [fusion driver](https://github.com/diku-dk/futhark/blob/f547a39abb1f3773f7cdb324ec8ea7cb8cf13ea0/src/Futhark/Optimise/Fusion.hs) constructs a dependency graph for a body, tries graph transformations, and converts the result back into statements. It handles functions separately and recurses into nested bodies. Its fixed-point sequence includes vertical fusion, horizontal fusion, inner fusion, transformations involving accumulators, and removing unused outputs.

The [graph representation and feasibility checks](https://github.com/diku-dk/futhark/blob/f547a39abb1f3773f7cdb324ec8ea7cb8cf13ea0/src/Futhark/Optimise/Fusion/GraphRep.hs) distinguish dependencies and node kinds. Candidate legality uses graph reachability. Vertical fusion checks for interfering dependencies/alternate paths; horizontal fusion requires independence. Successful fusion contracts and updates the graph.

This is materially different from Wyn's current `GroupBefore` transitive closure followed by `GroupBetween(p,c)` membership for every intermediate `m`. On a chain, the number of ordered `(p,m,c)` combinations grows cubically even though the original graph has only a linear number of adjacent edges. Native insertion reduces loading overhead but does not remove this derived-relation explosion.

Futhark does not precompute our all-triples “between” relation. That does **not** make its optimizer linear-time: horizontal candidate enumeration considers pairs, graph queries cost work, and fusion iterates. Its representation nevertheless avoids this particular persistent combinatorial fact set. This is the most directly applicable implementation difference.

Measurements
------------

The six fixtures implement the same stable binary radix pass over unsigned integers: classify bits, scan, obtain the total, compute destinations, scatter. They sort the low 30 bits. They are deliberately small diagnostic programs, not ports of fluid-simulation or copies of Futhark's production sorting library.

Each compiler process ran with a **3 GiB address-space cap**, a 2 GiB GHC heap cap, one GHC capability, and a 120-second timeout. These are single observations, not statistical benchmarks. Times include compiler startup; fusion times come from verbose pass logs. GPU measurements stop at optimized GPU IR: they do not include downstream C compilation, driver shader compilation, or GPU execution.

| Fixture | Scan bodies before fusion | Fusion time | Time through GPU IR | Peak RSS through GPU IR |
|---|---:|---:|---:|---:|
| 30 calls, top level | 1 | 0.005 s | 0.35 s | 66.8 MiB |
| 30 calls, inside map | 30 | 0.445 s | 3.03 s | 73.2 MiB |
| Loop, top level | 1 | 0.006 s | 0.32 s | 64.6 MiB |
| Loop, inside map | 1 | 0.010 s | 0.43 s | 76.8 MiB |
| 30 calls, top level, forced inline | 30 | 0.309 s | 1.55 s | 67.5 MiB |
| 30 calls, inside map, forced noinline | 1 | 0.007 s | 0.40 s | 71.9 MiB |

All 18 compilation runs succeeded. The normally inlined nested version contains 121 printed SOAC nodes before fusion and 61 afterwards; it still has 30 scans. The normal top-level version retains 30 calls to one pass definition. Counts describe static IR occurrences, not runtime iterations, kernel launches, or input array elements.

These results demonstrate that Futhark can analyze this expanded example without Wyn's memory explosion. They do not establish how it would perform on the complete Wyn program, or prove arbitrary scaling bounds.

[Fixtures and runner](futhark-research/), [machine-readable measurements](futhark-research/results.json), and [verbose logs](futhark-research/logs/) are retained. Reproduce on Linux with the official 0.27.1 binary, Python 3, GNU time, `prlimit`, and `timeout`:

```sh
python3 docs/futhark-research/run_experiments.py /path/to/futhark /tmp/futhark-results
```

The runner saves pre-fusion, standard-pipeline, and GPU IR, plus timings and stderr. It explicitly runs the standard pipeline prefix to obtain the pre-fusion snapshot. It does not need a GPU. The unusual GHC limits avoid the binary's default thread count interfering with the address-space cap.

Why the loop matters
--------------------

Futhark's [sorting library](https://github.com/diku-dk/sorts/blob/master/lib/github.com/diku-dk/sorts/radix_sort.fut), inspected on the research date, uses a loop over radix passes, processing two bits per iteration. Its pass includes a scan and a separate reduction for bucket totals. The library reference is a moving branch, unlike the pinned compiler references above.

A loop describes one body executed repeatedly. Inlining its helper exposes that one body; it does not require unrolling every iteration. Our loop-inside-map experiment retains one scan body while still exposing nested parallel computation to the compiler. This directly answers the repeated-stage problem without inventing a fusion-plan cache.

That source rewrite is not immediately sufficient in Wyn: our earlier attempted loop reached an unsupported host-stage publication path. Supporting repeated dispatches in the shader/runtime interface is a separate prerequisite. It would be misleading to prescribe a loop as a working one-line fix today.

Also, 30 passes are not necessarily 30 stages that can legally collapse into one kernel. Each pass depends on global information and the previous permutation. In our fixture, extracting the last scan result creates a real dependency. The recent [scan/scatter fusion discussion](https://futhark-lang.org/blog/2026-03-24-scan-scatter-fusion.html) illustrates why intermediate uses and representation affect fusion. It does not imply that all consecutive radix scans can fuse together.

The new function-lifting option
-------------------------------

The `noinline` nested experiment is especially informative. Current Futhark can keep that parallel helper: its GPU IR contains one `radix_bit_..._uniform_lifted` definition and 30 calls. The compiler transforms the helper to operate on a batch of inputs, rather than simply leaving an opaque parallel call inside an ordinary scalar kernel.

The release's [flattenApply implementation](https://github.com/diku-dk/futhark/blob/f547a39abb1f3773f7cdb324ec8ea7cb8cf13ea0/src/Futhark/Pass/Flatten.hs) requests uniform or nonuniform lifted variants depending on size-argument variation. This is specialized nested-parallelism support, not fusion memoization. The implementation explicitly limits intrablock handling of parallel function calls. Retained boundaries also prevent ordinary cross-call fusion.

The 2026 full-flattening thesis documents the lifting design and limitations. Some details have already evolved: the measured release supports separate uniform/nonuniform variants, so repeating every limitation from the thesis as a current universal restriction would be inaccurate. The [full-flattening announcement](https://futhark-lang.org/blog/2026-07-31-full-flattening.html) gives the broader context.

This could be a long-term architecture for reusable parallel functions in Wyn. It is substantially more work than moving an inlining pass, and it does not preserve all optimization opportunities of expanded code.

Recommendation for Wyn
----------------------

First address the analysis pathology. Keep the native fact insertion, stage-producer sharing fixes, redundant specialization fixes, and detailed timing/debug dumps. Replace broad eager “between” materialization with candidate-specific legality checks over a compact dependency graph. A practical design to investigate is computing candidate reachability/alternate-path answers in Rust and feeding only needed legality results to the fusion machinery. That is a proposal, not a verified drop-in replacement: results must be updated after contractions, and effects, ownership, aliases, and alternate paths must remain sound.

Then restore the inlining visibility required for SOAC-containing helpers, especially within SOAC bodies, and measure the expanded case again under the watchdog. The recent policy of fusing every helper independently and expanding it only afterwards should not be treated as the general solution for nested fusion. Context-sensitive retention of large top-level helpers remains useful where the backend supports calls or their eventual expansion.

Separately, add the host-loop/dispatch support needed to express radix iteration as one loop body. This removes avoidable source-level duplication independently of how fusion legality is implemented.

Only introduce structural fusion-plan caching if measurements still justify it. A cache would need a definition of equivalent shapes, captures, ownership, observable outputs, and surrounding uses. Reusing an internal plan while allowing caller fusion may be possible, but “identical call syntax always fuses identically” deliberately excludes context-dependent opportunities. Futhark does not provide evidence that such a cache is necessary to solve our immediate failure.

Papers saved in docs/
--------------------

The originals are preserved, checked with `pdfinfo`, and text-extracted for research. A source/size/page-count/SHA-256 manifest is in [sources.json](futhark-research/sources.json).

| Local PDF | Relevance and reading guidance | Original |
|---|---|---|
| [2013 graph-reduction fusion](futhark-2013-t2-graph-reduction-fusion.pdf) | Historical intraprocedural fusion and aggressive inlining. Its bottom-up analysis/top-down rewrite terminology is not a cache of callee fusion results. | [FHPC paper](https://futhark-lang.org/publications/fhpc13.pdf) |
| [2016 redomap fusion](futhark-2016-redomap-fusion.pdf) | Representation of combined maps/reductions and horizontal/vertical opportunities. Useful for understanding what can be fused, rather than just how many helpers exist. | [ARRAY paper](https://futhark-lang.org/publications/array16.pdf) |
| [2017 design/implementation thesis](futhark-2017-design-implementation-thesis.pdf) | Historical pipeline, chapter 7 on fusion, and discussion of greedy decisions. Do not read its all-functions-inlined pipeline as current policy. | [Thesis](https://futhark-lang.org/publications/troels-henriksen-phd-thesis.pdf) |
| [2017 nested parallelism](futhark-2017-nested-parallelism.pdf) | Explains the need to expose nested parallel structure and the interaction of high-level optimization with GPU lowering. | [PLDI paper](https://futhark-lang.org/publications/pldi17.pdf) |
| [2019 incremental flattening](futhark-2019-incremental-flattening.pdf) | Different mappings of nested parallelism to hardware and multiversioning. This concerns execution strategy, not deduplicating identical fusion analyses. | [PPoPP paper](https://futhark-lang.org/publications/ppopp19.pdf) |
| [2019 flattening by expansion](futhark-2019-flattening-by-expansion.pdf) | Background on translating nested array computation and allocation. Helpful context for why retaining a parallel function requires more than a scalar call ABI. | [ARRAY paper](https://futhark-lang.org/publications/array19.pdf) |
| [2026 full-flattening thesis](futhark-2026-full-flattening-thesis.pdf) | Function lifting and limitations: sections 5.3, 6.7.4, 6.8, and future work. Compare with current uniform/nonuniform lifting code. | [Thesis](https://futhark-lang.org/student-projects/amir-msc-thesis.pdf) |
