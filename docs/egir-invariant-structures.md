# EGIR: data structures that own the invariants

Design notes from the `egir_plan` and tinyporto profiling investigation, 2026-09-14.

## Implementation status

The patch implements borrowed result-tree access, indexed port lookup inside the existing semantic-incidence builder, a compact source-category set, borrowed dependence queries, and in-place monotone updates through a private fact table. Fixed dependence seeds are initialized once. The solver still uses full sweeps and retains its existing loop-dependency representation.

Abstraction review removed both `ResultTreeRef` and `ResultProvenance`. Product children are complete `ResultTree`s, so ordinary references support borrowed fields and structural lookup keys. `Facts::add_body` builds its lookup maps while creating ports, then uses an immutable query closure. There is no new cache in `GraphAnalysis` and no intermediate occurrence-ID universe. See the [line accounting and review](C:/Users/gmiller_amilarcap/dev/wyn/docs/egir-change-accounting.md).

The effect editor, stage-lift ownership states, worklist, and scoped compact loop identities below remain proposals. Standalone producer-index APIs and solved dependence results have not received the broader snapshot-ownership migration.

Final validation passed 1,859 workspace tests, 92 SPIR-V cases, 91 WGSL cases, both Wasm checks, and tinyporto compilation/validation. Existing Clippy violations in unchanged code prevent a clean lint gate. See the [profiling report](C:/Users/gmiller_amilarcap/dev/wyn/docs/egir-plan-profiling.md) for measurements, complete gate results, and limits.

## Recommendation

Give each relation one owner, expose queries through that owner, and put mutation behind operations that preserve the relation. Reuse the existing result tree, producer index, incidence builder, and dependence solver before introducing another peer abstraction.

The existing `GraphAnalysis` borrows an immutable graph and lazily owns shared structural facts. Consumer-specific port mappings belong to `Facts`; dependence solutions remain specific to their parameter/capture seeds. A cache with one consumer and its selection policy does not automatically belong in the shared graph analysis.

| Invariant | Owner and representation | What callers cannot do |
| --- | --- | --- |
| Derived caches retain their source snapshot | Borrowed `GraphAnalysis`; port lookup closure scoped to one body construction | Mutate the source while those borrows remain live; standalone index APIs still have broader contracts |
| Structural equality need not imply a unique producer | `Facts` builds every output port; its maps explicitly select first fields and last exact leaves | Expose a supposedly unique structural producer through a new shared API |
| Dependence facts only grow and queries see a completed solution | Private fact table, `join_assign`, solver-owned scheduling, immutable solved result | Replace accumulated facts with weaker facts or publish a partial solution |
| Insertions cannot invalidate a caller-held effect position | An editor that exclusively borrows the graph and owns its cursor | Retain a numeric site and use it after structural edits |
| Specialization does not mutate the shared source | Borrowed source, private draft, prepared specialization installed as one operation | Publish a region separately from the capture binding that calls it |

These guarantees require private fields and restricted APIs. Renaming a public map or adding a lifetime parameter to a copyable index does not establish them.

## 1. Index result lookup inside its existing consumer

### Current facts

The original `Facts::port` consulted a direct `ValueId` producer map, then searched every top-level result field for a matching origin. The query canonicalizes aliases and accepts either a field with exactly one matching returned value or a structurally equal registered origin. A value can have several registered origins.

The current contracts have details that a replacement must preserve:

- Direct producer lookup precedes origin lookup. In `Facts`, it uses the original value ID before alias canonicalization.
- Origin fallback selects the earliest matching field in skeleton traversal order, not the first registered origin or arbitrary hash iteration order.
- The direct map currently overwrites repeated value bindings. Duplicate occurrences must be represented or classified explicitly before tightening this behavior.
- `SideEffectIndex::effect_result_field` has a different lookup policy: canonical direct lookup followed by registered-origin order. Sharing structural extraction does not justify silently making these two policies identical.
- Returned value IDs, place IDs, types, and product structure participate in result-tree equality.

Sources: [incidence construction and lookup](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/semantic_graph/facts.rs), [SideEffectIndex and origin matching](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/ir.rs).

### Implemented representation

`Facts::add_body` already walks every effect and allocates every field/leaf port. It constructs three maps during that same walk:

1. Exact returned value IDs to the last declared leaf's `(block, port)`.
2. Borrowed result subtrees to the first matching field's `(order, (block, port))`.
3. The sole returned value of a field, when present, to the first such field's ordered port.

Repeated fields still have distinct ports in `Facts`. The lookup maps encode selection, not uniqueness. `insert` preserves last-declaration precedence, `entry.or_insert` preserves first-field precedence, and explicit traversal order chooses among different matching origins without depending on hash iteration order.

After construction, a closure immutably borrows these maps and the source graph. It resolves exact returns first, then canonicalizes once and checks the short registered-origin list plus the sole-return map. Recursive incidence construction receives that query, so it cannot mutate the selection tables. The maps are dropped when body construction finishes.

This replaces the scan with construction proportional to the indexed bindings and queries proportional to the queried value's origins, including the cost of hashing those trees. It does not require precomputing answers for every value if only a subset is queried.

The rejected `ResultProvenance` added a second graph scan, field/return records, IDs, path copies, a lazy cache, and a translation back into the ports `Facts` had already allocated. No other consumer needed that relation. `SideEffectIndex` remains the location index: its different precedence and callers that edit effects make adding fusion-specific state there inappropriate. The useful shared primitive is borrowed traversal of `ResultTree`.

### Borrow result trees when inspecting them

`ResultTree` keeps its structure private and stores complete child trees. `fields()` returns borrowed trees, and leaf traversal can borrow destinations and paths. The existing `field`, `top_level_fields`, and `destination_leaves` methods remain available when a consumer needs owned copies across a mutation. A separate borrowed wrapper type is unnecessary.

`single_value()` inspects return leaves without allocating and stops at the second returned leaf. Its meaning is **exactly one return leaf**, not exactly one physical destination and not exactly one distinct ID. A product containing one returned value and a place still qualifies; two return leaves containing the same ID do not.

Avoid a separately mutable cached count. The current tree mutation APIs can turn value routes into place routes. Borrowed traversal fixes the allocation without introducing another consistency obligation.

Sources: [result-tree APIs](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/ir.rs), [recursive origin registration](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/graph_ops.rs).

### Boundary of this change

Keep these maps local to the immutable body construction. Graph-owned origins are rewritten during value/place substitution and graph copying. Permanently interning them inside mutable EGIR would require routing all those operations through the interner, including cross-graph remapping. That is a substantially larger ownership change and is unnecessary to eliminate the measured lookup hotspot.

## 2. Dependence propagation should own accumulation and convergence

### Representation

Keep uniformity, loop dependencies, and provenance independent. They form a product of finite, monotonically growing domains:

- Uniformity joins by taking the less uniform classification.
- Source categories join by set union. There are currently fourteen categories, so a private `SourceSet` backed by a small integer is appropriate. An exhaustive conversion from `DependenceSource` prevents new enum variants from silently having no bit.
- Loop dependencies join by set union. They need identities with an explicit analysis context.

Expose an in-place join returning whether the fact changed. Computing that flag is part of the join: update uniformity, OR source bits, and record successful loop insertions. Callers should not clone the old value and compare it with a separately assembled replacement.

### Fact table and solver

Create a private table initialized for every analyzed value and block control. Build transfer relationships from the unchanged graph. Reads borrow facts; updates can only join them. A generation-aware side table keyed by existing arena IDs is a possible storage choice; its correctness comes from complete construction and restricted updates, not from replacing hash lookup with indexing.

Distinguish three states:

- A known constant is lattice bottom.
- A missing parameter seed is explicitly converted to the existing conservative unknown dependence.
- A missing table entry for a required graph value is an invalid graph/reference, not a constant and not an ordinary unseeded parameter.

The first implementation can keep the existing full sweeps with in-place joins. A subsequent worklist should be owned by the solver: changing a fact automatically schedules affected transfers. Returning `changed` for an unrelated caller to remember to enqueue dependents would leave the critical invariant outside the data structure.

For a worklist, extraction must include control edges, branch conditions, block arguments, call operands, place dependencies, unions, and the existing loop facts. A pure value-use graph alone is insufficient. Finish returns an immutable solved result only after all scheduled transfers have run. Invalid transfer inputs produce an error before publishing the result.

### Context matters

The same graph is analyzed with different parameter seeds. Structural relations can be shared through `GraphAnalysis`; solved facts cannot be cached just by function or graph ID.

Captures also transfer dependence facts from an enclosing graph into a region analysis. A graph-local loop bit number cannot be copied across that boundary: bit zero could identify a different loop in each graph. Use scoped loop identities, or a shared loop universe that explicitly imports enclosing dependencies before assigning compact ordinals. Plain `BlockId`s from separately owned slot maps also do not establish cross-graph identity. This is a design requirement, not a claim that the current production callers exhibit a wrong result; they predominantly ask whether the loop set is empty.

Seeded solves should expose graph queries through their own snapshot owner. Copying a fact for use as a capture seed is a distinct operation from applying a solved value table to a different graph.

Sources: [StageDependence](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/stage_variance.rs:120), [solver](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/stage_variance.rs:263), [monotone fact table](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/stage_variance.rs:198), [seed-separation examples](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/stage_variance_tests.rs).

## 3. An ordered effect editor should own location changes

`SideEffectSite` is explicitly valid only for a snapshot, but it is a copyable pair with public fields. `SideEffectIndex` can also be passed a different or mutated graph. The contract currently lives in comments and caller discipline.

For physical call reconciliation, introduce a narrow editor that exclusively borrows the graph and owns traversal of its ordered effects. It obtains the current call, performs boundary adaptation, inserts its prelude immediately before it, and advances past that call and its inserted effects. The caller never receives a location that it must adjust itself. Every subsequent call is read from the live graph after earlier substitutions.

The editor must not expose an unrestricted `&mut EGraph` or mutable effect vector to clients while retaining cursor state. Its internal primitives may change node/call contents, while the editor alone performs structural effect edits. Existing call-boundary validation remains the authority for argument channels, returned slots, destination ownership, and types.

This removes the need to construct a side-effect index merely to visit calls, and the per-call reconstruction disappears with it. Dense per-block vectors remain a good ordered representation.

Two tempting shortcuts are unsafe without extra work:

- Preparing all call payloads from the original graph can retain old operands after an earlier call rewrites value references.
- Draining effects out of the graph during editing hides their operands from `replace_value_references`. Future effects must remain visible to global remapping, or the editor must explicitly include detached storage in that operation.

A descending-order batch of insertions is another valid primitive when edits are independent and already prepared. It should be implemented by the owner, not repeated in each pass.

If broader transformations need effect identity to survive moves and block splits, the next representation is an effect arena with stable `EffectId`s and private per-block order lists. Its APIs must also enforce exactly one block membership, removal of call anchors, and ordered traversal. Adding IDs while leaving vectors publicly mutable would only create more state to synchronize. The narrow cursor addresses this hotspot without requiring that larger migration.

Sources: [effect sites and index](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/ir.rs:2462), [reconcile_calls](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/physical_call_abi.rs:698), [boundary adaptation](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/physical_call_abi.rs:194), [checked call rebinding](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/ir.rs:3056).

## 4. Stage lifting needs explicit ownership states

### Direct entry calls

Entries are uniquely owned by the consumed program. Candidate selection can borrow the entry; applying the selected inline can move and edit that entry through the existing body rewrite machinery. Separate immutable callee access from ownership of the entry being changed. There is no semantic requirement to copy the whole entry merely to discover that no call qualifies.

Keep prepared edits private to the operation that owns the source. A general `Patch { ids }` with `apply(&mut arbitrary_graph)` does not establish that the IDs still describe the graph that was analyzed. An edit session holding the mutable owner, or a prepared object containing both the owned source and the edit, can consume itself to apply exactly once. A borrow lifetime alone does not uniquely brand two unrelated graphs.

### Shared region specialization

Regions are shared, so their original bodies must remain unchanged for other uses. Represent preparation as a borrowed source that can become an owned draft on first mutation, using an explicit enum or private copy-on-write wrapper.

1. Analyze the borrowed region with this use's capture seeds.
2. If there is no eligible inline and no liftable frontier, return without cloning.
3. Clone when the first speculative inline is needed, or when a known frontier requires building a private specialization.
4. Publish only a prepared specialization with a nonempty frontier, its owned function, and the complete replacement capture binding.

Inlining can expose no useful frontier. In that case a private draft may legitimately be discarded. Removing that clone would require an overlay or rollback machinery, whose complexity is not justified by the current measurements.

Installation should consume the prepared specialization and update the enclosing use and function catalog together. Derive added parameters and capture arguments from one ordered binding representation. Do not separately expose a new function and a body referring to it for callers to remember to install in agreement.

This makes the ownership distinction explicit: direct entries are moved for editing; shared regions are cloned for actual specialization work. A complete prepared value also replaces the temporary `Option` used today to shuttle the newly specialized function out of the entry rewrite closure.

Sources: [direct entry preparation](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/stage_lift.rs:143), [region preparation](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/stage_lift.rs:213), [consuming body rewrite](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/ir.rs:4777).

## Suggested implementation order

1. Borrowed result-tree access and indexed port lookup within `Facts`. This addresses the strongest measured hotspot: approximately 54 ms of tinyporto residency spent in origin fallback searches, including the nested allocations.
2. Private source bit set, borrowed dependence queries, and in-place monotone joins. Tinyporto spends approximately 26 ms in propagation loops; that is not a measurement of clone cost alone. Preserve sweeps initially, then consider solver-owned work scheduling and compact scoped loop sets.
3. Cursor-based call reconciliation and explicit stage-lift ownership states. Their individual savings have not been isolated. Retain the larger stable-effect-ID migration as a separate structural change if other mutation consumers need it.

The desired end state is that passes select and orchestrate transformations. Graph/query owners preserve provenance, mutation owners preserve locations, and the dependence solver preserves monotonicity and convergence.
