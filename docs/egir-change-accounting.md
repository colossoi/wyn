# EGIR change accounting and abstraction review

## Scope and counting method

Baseline: `b9d85fa59cf29241879308d72104051f0b902e67`, before the profiling work. This accounts for both the earlier empty-dependency fast path and the result-lookup/dependence changes, including removal of `ResultTreeRef` and `ResultProvenance` during review.

Counts use Git's physical-line convention: additions minus deletions, including blank lines and comments in source files. They are not statement counts or code-only SLOC. Newly created Rust files are included explicitly; plain `git diff --numstat` omits them while they remain untracked.

| Compiler change category | Added | Removed | Net |
| --- | ---: | ---: | ---: |
| Production Rust | 386 | 226 | **+160** |
| Rust regression cases | 251 | 2 | **+249** |
| Total Rust | 637 | 228 | **+409** |

| Patch | Production net | Regression net | Total net |
| --- | ---: | ---: | ---: |
| Earlier empty-dependency fast path | +16 | +141 | +157 |
| Result lookup and dependence structures, including review | +144 | +108 | +252 |
| Combined | **+160** | **+249** | **+409** |

## File accounting

| File | Added | Removed | Net |
| --- | ---: | ---: | ---: |
| [ir.rs](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/ir.rs) | 107 | 56 | +51 |
| [parallelize/planning.rs](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/parallelize/planning.rs) | 6 | 2 | +4 |
| [semantic_graph.rs](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/semantic_graph.rs) | 15 | 3 | +12 |
| [semantic_graph/facts.rs](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/semantic_graph/facts.rs) | 49 | 35 | +14 |
| [stage_variance.rs](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/stage_variance.rs) | 209 | 130 | +79 |
| [semantic_graph_tests.rs](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/semantic_graph_tests.rs) | 152 | 2 | +150 |
| [stage_variance_tests.rs](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/stage_variance_tests.rs) | 17 | 0 | +17 |
| [types_tests.rs](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/types_tests.rs) | 23 | 0 | +23 |
| [integration_tests.rs](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/integration_tests.rs) | 59 | 0 | +59 |

All retained Rust changes are now in existing files. The new provenance module and its test module were removed; its regression scenarios moved to the existing semantic-graph and type tests.

## Abstraction decisions

### ResultTreeRef: removed

The first implementation added a borrowed wrapper around the private `ResultNode`. It worked, but it existed because only the outer root was a `ResultTree`; product children were raw `ResultNode`s. That representation prevented returning an ordinary `&ResultTree` for a child.

The final representation makes product children `Box<[ResultTree<...>]>`. A subtree is now a complete tree, and `fields()` returns `&[Self]`. Port lookup can key directly on `&ResultBinding`, using ordinary structural equality, hashing, and Rust borrowing.

This removes the separate view type, its `Copy`/`Clone` implementations, the `borrowed()` conversion, and the view's ownership/traversal API. Existing owned field accessors reuse the borrowed fields. A fallible borrowed leaf visitor constructs ports directly and propagates builder errors without allocating intermediate leaf records. The infallible visitor and owned collector reuse that traversal. `single_value()` counts return leaves directly without an intermediate vector.

This is the better abstraction here: owned and borrowed trees share one representation and API. The recursive edge still uses the existing boxed child slice; wrapping a child node as a tree adds no additional allocation or pointer indirection. The representation change is private to `ir.rs`; existing public constructors and owned accessors retain their behavior.

Cost: recursive mapping, binding, and result construction now wrap child nodes as complete trees. That touches several existing helpers, so the diff is broader than the net growth. `ir.rs` is +51 net lines, including the fallible visitor used by direct port construction. Both abstraction reviews together reduced production growth from +311 in the first structural draft to +160: 151 fewer lines.

### ResultProvenance: removed

The standalone abstraction was not justified. Its 144-line module repeated the graph walk, created field and return IDs, stored leaf paths, and required another lookup from those IDs into fusion ports. Its only consumer was `Facts`, which already creates and owns those ports.

The existing concepts have distinct jobs:

| Existing concept | Responsibility |
| --- | --- |
| `ResultTree` / `ResultBinding` | Authoritative type and return/place routing shape |
| Value `result_origins()` | Registered structural origins, preserved through rewrites |
| `SideEffectIndex` | Result/call IDs mapped to explicit effect locations |
| Semantic `Facts` | Scoped fusion ports, projection paths, and incidences |
| `GraphAnalysis` | Lazy shared structural analyses of a borrowed graph |

The final code builds `last_declared`, `first_origins`, and `first_single_returns` maps while `Facts::add_body` allocates ports. Each maps directly to a port and its defining block. Field matches retain traversal order so origin registration order cannot change selection. An immutable closure performs lookup during recursive port construction. These maps borrow result trees and die with body construction; there is no additional invalidation protocol or ID translation.

Every repeated field/leaf still gets its own port. Map updates explicitly preserve the existing last-exact-return and first-matching-field rules. Structural equality does not become a uniqueness assertion. `Facts` grows by only 14 net lines compared with the original scanning implementation.

Folding fusion port state into `SideEffectIndex` would put consumer-specific data and policy in a location index. Its `effect_result_field` query canonicalizes first and falls back in registered-origin order, while `Facts` uses raw direct IDs first and fallback in skeleton order. Existing callers also use its locations during controlled graph edits. Those are reasons to retain that existing API and improve the shared tree traversal, rather than make either index pretend to be universal provenance.

Removing this layer reduced production growth by a further 115 lines after the `ResultTreeRef` review. `GraphAnalysis` and `LoopInvariance` are back to their baseline code. Existing graph-scoped IDs remain unbranded; the patch does not claim to reject every possible cross-graph ID mix-up.

### Stage dependence: keep the small domain-specific structures

`stage_variance.rs` changes 339 physical lines but grows by only 79. Much of the diff replaces owned temporary facts and clone/compare/update sequences with borrowed reads and in-place joins.

- `SourceSet` represents the closed set of fourteen source categories in a `u16`. Its exhaustive enum-to-bit mapping and private bits fit this domain. A separate general-purpose bit-set abstraction would add machinery without serving another current use.
- `GrowingFacts<K>` has two actual key types: value IDs and block IDs. It constructs all entries before solving and exposes reads and monotone joins, with an error for an absent required graph key. Keeping these operations together is worthwhile; a free helper accepting an arbitrary mutable map would leave weaker updates available to pass callers.
- The solved result is published after convergence. Read-only dependence and call-argument queries borrow facts. Copies remain where capture seeds need independent ownership.
- Fixed seeds are initialized once because their definitions do not depend on evolving facts. The existing transfer semantics and solver sweeps are retained.

The map wrapper does not prove transfer-function correctness or solve cross-graph loop identity. Those remain separate responsibilities. A worklist would also need complete control-flow and place-dependency edges; adding one solely to reduce the visible loop would broaden this patch substantially.

### Supporting changes and regression cases

The 249 net regression lines consist of 141 from the fast path and 108 from the structures patch. They cover scalar preludes, mixed bodies, origin/alias ordering, direct precedence, return-leaf cardinality, and monotone updates. The result-lookup scenario now checks constructed ports and projection paths through `Facts`, rather than the removed index's private IDs. All these cases passed in the final workspace suite; the profiling report records every gate and the pre-existing Clippy failures.

## Local profiling tools and non-code artifacts

The ignored `tmp/egir-plan-profile` directory also contains **375 authored source lines** across 13 profiling/accounting scripts and the temporary Rust timer helper. They are separate from the compiler patch. Including these tools brings retained authored Rust/script growth to **784 lines**: 409 compiler/regression lines plus 375 tooling lines.

| Local tool | Lines |
| --- | ---: |
| `build-instrumented.ps1` | 12 |
| `build.ps1` | 32 |
| `compare-invariant-output-shapes.cjs` | 27 |
| `compare-invariants.ps1` | 20 |
| `compare-summary.cjs` | 23 |
| `compare.ps1` | 22 |
| `instrument.cjs` | 68 |
| `line-accounting.cjs` | 28 |
| `read-detail.cjs` | 16 |
| `summarize-invariants.cjs` | 30 |
| `summarize.cjs` | 16 |
| `sweep.ps1` | 40 |
| `timers.rs` | 41 |

The generated `profile_egir_plan.rs` is a copy of the CLI with the timed invocation generated by `build.ps1`; counting it again would count copied/generated source as newly authored code. Binaries, PDBs, shaders, logs, timing JSON, and generated temporary shader wrappers are also excluded from authored-code totals.

The profiling report, design note, and this accounting note are documentation, counted separately by the accounting script. Pre-existing untracked review notes and the avatar shader are excluded; they were not authored or changed by this work. Submodules and tinyporto sources were not edited.

Reproduce the accounting with [line-accounting.cjs](C:/Users/gmiller_amilarcap/dev/wyn/tmp/egir-plan-profile/line-accounting.cjs). Its JSON output includes every source file, separate tooling/documentation counts, and the baseline commit. Performance and validation limits are recorded in the [profiling report](C:/Users/gmiller_amilarcap/dev/wyn/docs/egir-plan-profiling.md).
