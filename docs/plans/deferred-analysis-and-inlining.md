# Deferred analysis and callee-first inlining

Status: first-use deferrals and callee-first helper inlining implemented.
Branch: `codex/defer-analysis-callee-first-wip`.
Starting commit: `040a9f7a26b16e8220e82c3fbb5d3eca2f240031`.

## Current progress

The first-use deferrals put backward liveness and reducer metadata at host
preparation, and host/kernel classification and loop state sizing at placement.
Host preparation requests liveness explicitly and reuses the valid structural
analysis from optimization. The unused `ResourceLiveAfter` table is removed.

The inliner computes recursion, ownership, and eligibility once per invocation.
Egglog propagates the parallel-work requirement through nonrecursive callees
and identifies the original call sites. Rust visits the selected dependencies
callee-first and splices into the edited regions. It clears body-dependent
analysis before the first splice, then works from the call-site snapshot.
Focused tests cover first-use metadata, storage effects, dependent shape
arguments, nested region cloning, shared callees, and recursive calls.

The simplifier still uses a whole-graph analysis refresh. Finer fact lifetimes
and invalidation in that pass are follow-up work. The measurements below are
from before callee-first inlining. Measurement logs remain under ignored `tmp/`.

## Finding and measurements

Full TinyPorto's initial inliner completed in 68.03 seconds, with 670 expansions:

| Work | Seconds |
| --- | ---: |
| Structural refreshes, 671 rounds | 53.59 |
| Remaining candidate-selection overhead | 2.34 |
| Splicing | 8.94 |
| Cloning | 2.53 |
| Substitution | 0.50 |

The structural timer includes invalidating derived facts, lexical ownership and
ancestry, reachable functions/operations, and transitive call reachability for
recursion detection. These components have not yet been timed separately.
The complete compile still reaches its 300-second deadline in simplification.

The unchanged 12-camera fixture spends 0.259 seconds in both inliner invocations
combined. Its SPIR-V and Rust output match the pre-change artifacts byte-for-byte.
Measurements: `tmp/tinyporto-inline-isolated/` and
`tmp/tinyporto-inline-isolated-5min/`.

## Facts checked in the code

1. Expansion clones, substitutes, and splices canonical graph regions. It adds
   no functions and changes no existing call's target to a different function.
   Only nonrecursive callees are expanded. Expanding those calls preserves the
   original recursive components, so recursion classification can be frozen
   for one invocation; reachability of individual operations cannot.
2. Parameter/result categories used by inline eligibility remain stable.
   Exact metadata is not all immutable: substitution rewrites accumulator shape
   operands and return metadata. Those must continue to follow value renaming.
3. The mandatory rule "contains parallel work" can become true after expanding
   a callee. Freezing today's `InlineRequired` flags would miss that change.
   An alternative needs the appropriate transitive requirement, processing
   order, or updates; indiscriminately propagating every mandatory flag would
   also change policy for scalar wrappers around logical-value helpers.
4. Cloning and substitution already work without global ancestry or liveness.
   New operations, nested region ownership, removed calls, and reachable call
   sites still require correct enumeration after each edit.
5. Ordinary shader calls and logical product types already have emission paths.
   This establishes a narrower-inlining option, not proof that every record or
   array helper can remain opaque to fusion, placement, or effect analysis.

## Refinement: fact lifetimes and invalidation scope

Finishing inlining should not trigger an unconditional refresh of every analysis
for the entire program. Each fact group needs an explicit lifetime and dependency
boundary. The following simplifier should request analyses for the function it is
actually processing, reusing facts that are already valid.

| Fact group | Proposed lifetime / smallest safe work boundary |
| --- | --- |
| Immutable operand and type-layout facts | Keep for the graph's lifetime. `IndexUse`, `ShapeAt`, `SliceAt`, and `ElementBytes` depend on immutable terms. Rewriting creates new terms; derive facts for those terms while retaining facts about existing terms. |
| Recursion and stable eligibility inputs | Compute once per inlining invocation. Determine the parallel-work requirement in callee-first order, because expanding a callee can expose parallel work in its caller. |
| Operation membership, order, and region ownership | Update the cloned or moved nodes and their direct ownership edges. Traverse canonical child-region edges while inlining; avoid constructing the whole program's ancestor closure for call discovery. Materialize derived ancestry only where a subsequent pass needs it. |
| Definitions, uses, dependencies, liveness, resources, and effects | Invalidate the affected groups in the edited function, including its nested regions. Recompute on demand. Function scope is a conservative starting boundary for backward liveness through loops and captures; unrelated functions retain their facts. |
| Work, extent, and physical-planning summaries | Separate immutable calculations from body-dependent results. Update affected region/entry summaries and propagate to enclosing summaries or dependent callers only when their exported result changes. Request physical planning when its consumer needs it. |

The current global `AnalysisRevision` appears in even immutable-term rules, forcing
them to revisit existing inputs after unrelated edits. Persistent term rules need
to lose that global revision dependency as well as being removed from blanket
clearing. Mutable analyses need explicit owner/dependency scopes instead.

There are real scoping gaps to resolve before implementing this safely:
`InvalidDataflow` currently holds only a diagnostic string, and the liveness
publisher rebuilds a global name-to-ordinal assignment. Selective invalidation
must give such facts adequate ownership and stable identities, or conservatively
retain the broader boundary for that group. Leaving stale rows in these tables
would be incorrect.

This should be staged as specific changes to the existing fact families and
editor contracts, rather than a general-purpose incremental-analysis framework.
For inlining itself, avoid requesting liveness, effects, or physical planning;
update canonical structure directly and carry function-level summaries through
the callee-first traversal. The subsequent simplifier requests only its needed
analyses for each changed function.

## Defer calculations until their first actual consumer

The fact-lifetime work should be preceded by splitting analyses that currently
run together despite having different consumers. Some global work can disappear
from mutation loops entirely instead of receiving an incremental cache.

| Calculation | Current consumer / proposed boundary |
| --- | --- |
| Full backward live-after analysis: `live::publish`, `flow-live`, `NameLiveAfter` | Its canonical-IR production reader is execution preparation (`execution.rs`, CPU values surviving dispatch). Compute after graph transformations finish, before that reader. DCE uses `NeededName`/`NeededOperation`, a different calculation. |
| Function lane dependence and initial host/kernel classification | `planning::analyze` runs `graph-placement` during simplification, including transitive `LaneDependentFunction`. Consumers are later host/placement selection. Separate this from rewrite metadata and first request it at placement/scheduling. |
| Final whole-graph analysis/validation refresh | `optimize_with_policy` ends with `analysis::run`; `prepare_host` immediately repeats it without an intervening edit. Reuse that valid result. More generally retain edit-local checks and reserve final global closure checks for phase boundaries. |
| Physical reuse, snapshots, execution features/atomics, captures and allocation | These are already principally late: `prepare_host` calls `planning::epilogues`, then physical execution preparation. Preserve that boundary and avoid pulling their complete prerequisites into early rewrite loops. |

One related deletion candidate is `ResourceLiveAfter`: the reference audit found
its schema, construction, and clearing, but no production reader. Verify its
diagnostic/public-table contract before removing it rather than merely deferring
its calculation. The separate shared `dataflow.rs` liveness API has other
consumers and must retain its behavior.

Deferring canonical live-after analysis requires separating its incidental
`ResourceDemand` seeding from the roots needed by early effect/alias proofs.
Existing reads/writes already demand their accessed resources explicitly, but
that dependency must be covered by focused tests before removing the early
liveness invocation.

Early transforms still need neededness, dependency/effect legality, relevant
bounds, and work estimates where they affect hoisting or fusion. In particular,
native-reducer recognition currently contributes to some early bounds proofs;
moving the entire metadata ruleset late would discard required information.
Split actual prerequisites from placement-only and final-execution facts.

The current 6--7s graph-analysis and 12--13s metadata spans on TinyPorto include
multiple calculations. The exact savings from each deferral are unmeasured.
Measure the split phases before attributing those totals to any single solver.

## Options considered

| Option | Change | Benefit | Main obligation / tradeoff |
| --- | --- | --- | --- |
| 1. Freeze function summaries; visit edited calls locally | Compute recursion and function-level requirements once per invocation. Track pending calls in canonical regions, including copies introduced by cloning. | Removes repeated global solves while retaining call-at-a-time edits. | Derive the parallel-work requirement correctly; cover nested regions, dead calls, and generated copies. Keep policy in egglog and Rust worklist handling mechanical. |
| 2. Expand callees before callers | Prepare each eligible helper body before copying it into its callers. Reuse the existing graph dependency-order utilities. | Avoids repeatedly expanding the same nested helper in many copies. | Respect recursive components and mandatory versus `InlineAll` policy; test shared helpers and changed expansion order. |
| 3. Require less early inlining | Preserve helpers whose argument/result representation and behavior are already supported by ordinary calls. Inline where array work, capabilities, or other pass requirements demand it. | Eliminates some work and tests whether the present mandatory boundary is too broad. | Establish which calls downstream analyses can safely keep opaque. With `-O`, scalar expansion may merely move later, so measure the whole pipeline. |
| 4. Maintain structural facts incrementally | Update ownership, reachability, and call facts affected by graph edits. | Preserves the current selection loop without whole-graph refreshes. | Deleted calls and moved/cloned regions need retraction or rebuilding of affected facts; saturation alone does not remove stale facts. Broader editor contract and more invalidation risk. |
| 5. Batch independent expansions | Select several calls from one structural snapshot. | Reduces refresh frequency with relatively little scheduling change. | Prove independence, handle argument/result substitutions between calls, and discover newly exposed calls. It still retains repeated global solves and is not a selected default. |

These can be combined, but their individual effects should be established first.
An earlier TLC boundary is also worth checking when evaluating option 3: a TLC
inliner already exists, but stage extraction and defunctionalization follow it.
Moving work earlier must account for helpers introduced later and preserve both
mandatory expansion and the separate `-O` behavior.

## Implementation checks and remaining decisions

1. Split the structural measurement into invalidation, ownership/reachability,
   and call-graph closure. Record operation/function counts as expansion grows.
2. Compare the applicable options against both initial and late inlining. State
   what must be recomputed, what can be frozen, the expected production-code
   delta, and any policy change. Review existing implementations/deletion diffs
   before adding a new traversal or cache.
3. Check the proposed fact lifetimes and owner scopes before changing
   invalidation outside inlining. Keep broader policy changes separately
   reviewable.

Targeted coverage must include recursion, helpers that expose parallel work
through other helpers, shared callees across entries, nested controls/reducers,
dependent arguments and shapes, and the mandatory/`InlineAll` distinction. Keep
phase units below three seconds. Reuse the existing compiler fixtures; time the
unchanged camera fixture with a release compiler and use a hard five-minute
limit for full TinyPorto. Explain artifact differences if policy/order changes.

## Verification and existing failure at this starting commit

Workspace build, enforced units (1,343 passed, 13 ignored), doctests, Clippy,
runner checks, standalone builds/tests, and browser WASM check pass. Testfile
sweeps pass: 124 SPIR-V; 123 WGSL with one documented skip.

The full integration gate fails after 151 passing cases at
`memory_ir_branch_backings`: `SSA lowering: output leaf has no storage stride`.
It reproduces with both original HEAD `6fc999f6` and the current compiler.
Later cases were not reached. The corrected uniform-row fixtures pass targeted
checks. Gate logs and timings are in `tmp/inline-acceptance-20261116/`.

## Code and reuse sources inspected

- `wyn-core/src/egglog/memory.rs`, `analysis.rs`, and `analysis/dataflow.egg`:
  current loop, invalidation, ownership/reachability, and recursion rules.
- `wyn-core/src/egglog/dataflow/edits.egg`, `edit.rs`, and `clone.egg`:
  eligibility, canonical edits, and recursive region cloning.
- `wyn-graph/src/traversal.rs::dag_postorder` and
  `wyn-graph/src/lib.rs::topo_sort_by_dependencies`: existing traversal tools;
  respect their acyclic-input requirements.
- `wyn-core/src/tlc/inline.rs`, `egglog/representation.rs`, and
  `egglog/to_ssa.rs`: existing earlier expansion and ordinary call/type support.
- Historical `a3c04a89:wyn-core/src/egglog/memory.rs`: the same original edit
  orchestration, with whole-program analysis per expansion; no independent
  call-worklist solution to restore from that version.
