# `plan_logical_resources`: objectives and backward audit

Reviewed 2026-09-10 against `8d60314a69fe3bcad261a3e07921d2995aec87b4`.

This document records the pre-refactor source review against that revision. The coordinated refactor has since been implemented in the working tree. See the [current struct/enum audit and validation](C:/Users/gmiller_amilarcap/dev/wyn/docs/plan-logical-resources-type-audit.md) for the resulting design, changes made, and remaining concerns. The historical findings and source line references below describe the reviewed baseline.

The original review covers all production functions and local type declarations in `allocation/{mod,residency,cost,host_length}.rs`, plus the supporting projection, dependency, resource, and stage structures relevant to their contracts. It does not audit every function in those supporting modules. Existing review documents and submodules were left untouched.

The strongest opportunity is to represent a complete handoff once. Today, resource allocation, producer output configuration, consumer replacement, and stage-flow publication repeatedly reconstruct different parts of that same relationship. Most legality checks are necessary; much of the bookkeeping around them is not intrinsic to the job.

## Business objectives

These describe outcomes for people compiling and running Wyn programs. A scan, fixpoint, projection, or enum is a means, not an objective.

| ID | Objective | Observable success criterion |
| --- | --- | --- |
| O1 | Give the host a consistent resource contract. | A repeated binding denotes one compatible resource; forward size references work; unresolved external sizing is distinguishable from unfinished compiler work; generated storage has a usable logical size. |
| O2 | Make results available wherever execution needs them. | Shared arrays, gathers, reductions, and runtime arrays have usable storage across the relevant scheduling boundaries. A Filter consumer sees the survivor count, not the capacity. |
| O3 | Preserve the program's meaning when moving work. | Moving a producer preserves its inputs, control dependencies, legal execution context, effects, and every still-observed output. Nothing reads a result before its producer runs. |
| O4 | Deliver the promised outputs. | Host outputs and internal routes still refer to the correct value, writer, storage, and dynamic-length representation after rewrites. |
| O5 | Avoid expensive repeated work when the handoff is worthwhile. | Scalar precomputation saves enough repeated work to justify a launch and all consumer loads; cheap or unsafe work stays local. |
| O6 | Produce a coherent, target-independent stage/resource plan that respects the requested topology. | Generated stages obey topology policy; the stage graph is valid and executable bodies have owners; backend descriptor assignment remains a later concern. |

The current policy treats some multi-consumer array materializations as required structure. O2 captures that contract without claiming that consumer count alone proves an intrinsic semantic need in every possible compiler design. Likewise, first-candidate ordering is current policy, not a separate business objective.

## Corrections to the supplied pass description

- The actual entry point is allocation, residency resolution, then `finalize_staged_ir`. Allocation starts with `host_length::retain_output_lengths`, before binding reservation. [Entry points](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:95)
- `residency_facts` already supplies shared `GraphAnalysis` objects and one `SemanticGraph` for each unchanged program snapshot. Optional prelude selection uses the same facts as required selection. There is no second dependency rebuild between them. [Read phase and loop](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:215)
- Array residency demands are now a field built with `SemanticGraph`, rather than a separately invoked pass. [Dependency construction](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/semantic_graph.rs:218)
- A parallel prelude can be profitable for one parallel operation: sharing occurs across its invocations, not necessarily across several operations. [Existing example](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/integration_tests.rs:6941)
- Prelude selection is not entirely cost-driven: `PreludeMaterializationPolicy::Required` bypasses profitability for a structured prefix with storage loads. This classification needs to be reconciled with the outer loop's required/optional distinction. [Policy](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/cost.rs:104)

## Data structures worth using to remove code

### 1. An output record owned by a prepared handoff

Start with the existing `OperationMaterializationPlan` and `StagePreludePlan`. Replace parallel collections with a record per output:

```text
PreparedOutput
    source value / result field
    projected producer value
    value type
    storage requirement: scalar | fixed array | runtime array
    resolved storage, once allocated

PreparedPrelude
    projected recipe
    ordered, complete outputs
    consumer edit boundary
```

The constructor must obtain all output identities from the projection and validate the supported storage layout. Keep fields private so callers cannot separately replace the recipe or supply a different output list. After allocation, each record owns its storage and supplies both the producer and consumer operations. The first structured boundary result should be explicit, rather than an incidental `loaded_values[0]` convention.

This could remove:

- Repeated result-field lookup in `output_specs`, `configure_operation_materialization`, `configure_materialized_result`, and `rewrite_materialized_operation_source`.
- Alignment by `zip` between output specs, routed resources, allocated resources, views, replacements, and loaded values.
- The output-coverage reconstruction at the start of `materialize_stage_prelude`: coverage becomes the constructor's invariant.
- Recomputing prelude source/projected correspondence when the projection already owns it.

This is the best first structural change because it replaces existing state rather than introducing a new planning language. Scalar stores, array destination binding, and runtime-length handling still require different emission logic. A single enormous generic materializer would likely add indirection without eliminating that logic. [Fixed-output setup](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:1336), [prelude setup](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:1523)

### 2. One owner for staged entries and resident flows

The existing `StagedIrBuilder` is already the right graph structure. Its body type can be `AllocatedEntry`; it already exposes body access. Currently residency stores bodies in `entry_points`, stores entry IDs in the stage builder, and maintains `stage_ids` to connect them. Finalization then joins these representations and asserts that the join was bijective.

An EGIR-facing builder owning stage bodies could eliminate `add_generated_stage`'s coordinated writes, much of the three materializers' program disassembly/reassembly, and the final entry-ID-to-body join. Retain an index only if a real caller needs lookup by authored `EntryId`; derive it at one owner. Adapting index-based iteration and public draft APIs makes this a larger change than output records. [Draft state](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/program.rs:1563), [final join](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:1050), [existing builder](C:/Users/gmiller_amilarcap/dev/wyn/wyn-staged-ir/src/lib.rs:194)

The same owner should distinguish a produced value/flow from the storage holding it. `connect_resident_flow` currently looks up a flow by data resource alone and, on reuse, does not compare the supplied producer, type, or length resource. Finalization separately discovers additional readers and separately creates host publication flows. A completed output/flow record could connect these operations once, including the runtime length when applicable. Do not assume that a resource identity always identifies one value across multiple writes, or blindly merge flows for all uses of the same buffer. [Flow connection](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:181), [reader discovery and publication](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:985)

The generic stage builder proves graph properties, not EGIR resource existence, storage compatibility, or publication correctness. Those validations remain at the adapter boundary unless the new owner actually enforces them.

### 3. A canonical resource-access map and a complete array replacement

Use an ordered map from resource identity to access, with one merge operation: Read plus Write becomes ReadWrite. `materialize_runtime_array_result`, `realize_graph_dynamic_publication`, `configure_materialized_soac`, and `refresh_resource_reads_for_values` currently perform related find/insert/merge/sort sequences.

Use the output record to construct an array replacement containing the source value and a resident view with its element type and logical extent. Applying this replacement should update the operand, SOAC input type, domain, resource access, and entry routes together. `InputReplacement` is already a partial version of this idea; extend or simplify it instead of adding an unrelated substitution system. [Input repair](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:1860)

Two distinctions must survive: replacing an entry's role is not always the same as joining access roles, and refreshing derived reads is not simply adding more reads forever. Also, a Filter's view length is its stored survivor count. A resource-access map cannot replace that cardinality rule. [Role assignment](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/program.rs:963), [runtime view creation](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:1287)

### 4. A binding registry whose payload is resolved without reallocating identities

The host builder currently reserves IDs in one vector/map, stores optional lowered policies, then allocates the same resources again into `LogicalResourceArena` and verifies that the identities match. A consuming transformation of an ordered binding registry could keep IDs fixed while converting unresolved size references to logical ones. This would remove the second allocation and identity-comparison code.

Keep the two logical phases: collect every binding, then resolve references. No topological sort is needed merely to replace a `LikeInput` binding with a resource ID; this code does not evaluate the referenced resource's eventual numeric size. Keep conflicts and explicit externally provided sizing distinct. Replacing `None` with `RuntimeProvided` before resolution would erase a useful invariant. [Current draft and finalization](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:132)

This is a moderate opportunity. The existing `interface_resources` iterator already unifies input/output enumeration, and the arena already owns dense IDs. A second general-purpose symbol table would duplicate those benefits. The useful change is preserving one registry through resolution.

### 5. Smaller query results over existing analysis

Use `SemanticGraph::operation_site` to inspect known consumers directly. Current code gathers consumer IDs and then scans every entry effect to find matching ones. A located operation or consumer query can carry the site and scheduling classification once. `parallel_preludes` can similarly use an insertion-ordered map of root to distinct consumers instead of a vector plus an index map. Preserve candidate order; an unordered collection can change which optimization runs first.

Do not introduce another dependency graph or a general worklist framework. `GraphAnalysis`, `SemanticGraph`, and the projection's `LiveSlice` already own the important facts. A rewrite changes uses and effect positions, so snapshot invalidation remains necessary. Incremental maintenance has not been shown to reduce code here. [Consumer rescanning](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:952), [prelude grouping](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:802)

### 6. An explicit relocation decision

Represent safety, necessity, and profitability separately. For example, an admitted candidate could carry `Required(reason)` or `Optional(cost)`; the outer selector then applies topology policy consistently. This is primarily an ownership/clarity improvement, not a demonstrated LOC reduction.

The current `Required` prelude policy says input-dependent structured storage work must not be multiplied. Yet `AuthoredOnly` exits before evaluating it, and `analyze_prelude` must successfully compute costs before producing even a required result. Establish whether this is a semantic requirement or an optimization preference before changing behavior. If semantic, it belongs in required planning with the appropriate topology diagnostic; if a preference, its name and comment should say so. This review identifies the inconsistency; it does not establish an end-to-end miscompilation. [Outer decision](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:258), [analysis and decision](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/cost.rs:64)

## Backward audit

Read each row as: what outcome does this code support, and does its present representation earn its complexity? “Consolidate” retains the behavior while replacing its bookkeeping. “Keep” does not mean every line is irreducible.

### A. Final stage graph: the actual last step

| Code | Objectives | Assessment |
| --- | --- | --- |
| `finalize_staged_ir`: resource-reference and route checks | O1, O4, O6 | Keep the boundary validation. The resource arena prevents duplicate IDs; it does not prevent a size expression or entry field from referring to an absent resource. Nonempty route/writer checks likewise are not graph-cycle checks. Consolidate the repeated entry-field traversal, not the semantic checks. |
| `finalize_staged_ir`: add missing resident consumers | O2, O3, O6 | Necessary under current ownership: a later generated producer can read an earlier handoff. Move this relation to the stage/flow owner; deleting the loop alone would lose dependencies. |
| `finalize_staged_ir`: external inputs and published outputs | O1, O4, O6 | Necessary host boundary. Consolidate publication with completed output records; verify ownership and runtime-length behavior before merging any existing flows. |
| `finalize_staged_ir`: `finish`, body join, empty-entry assertion | O3, O6 | Keep graph validity. Replace the body join and associated assertions by stage-body ownership. These assertions are currently defending duplicated storage. |
| `ResourcesAllocatedTag`, `ResourcesAllocated`; `ResidencyDraftTag`, `ResidencyDraft` | O6 | Keep distinct phase contracts: a draft still discovers stages; the final program owns finished stages. The marker enums have no runtime payload and are not procedural overhead worth targeting. |
| `ResourcesAllocated::semantic_ir`, `logical_resources` | O1, O6 | Keep small inspection APIs. They let callers inspect the contract; they do not participate in candidate selection. |

Source: [finalization](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:913), [phase types](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:35).

### B. Last rewrite: `materialize_stage_prelude`

| Code | Objectives | Assessment |
| --- | --- | --- |
| `materialize_stage_prelude`: output coverage check | O3, O4 | The invariant is essential: publish every live-out before removing source effects. The current set reconstruction can disappear when one prepared plan constructs and owns both recipe and outputs. Both current planners call the same output factory and pass the result unchanged. |
| Producer entry, one cell per output, producer stores and consumer loads | O2, O5 | Keep actual emission. Consolidate output allocation and use in output records. Charging and publishing all live-outs is essential, even when they are not the originally selected root. |
| Value and route replacement, resource refresh, DCE, interface compaction | O3, O4, O6 | Keep semantics. A shared entry-rewrite operation can own graph and route replacement plus required repair. DCE roots must include routes outside the graph. Interface compaction preserves only supported, used producer inputs. |
| `replace_prelude_effects_with_load` | O3 | Preserve placement before the earliest consumer. Sorting removed positions and subtracting earlier deletions are positional bookkeeping; a single filtered/spliced block edit can replace them. Current use relies on all selected effects belonging to the entry block. Encode/check that boundary at plan construction. |
| `replace_entry_prelude_with_load` | O3 | Necessary distinct placement at entry start. Can share effect removal/insertion machinery with the preceding case. |
| `replace_structured_prefix_with_load` | O3, O4 | Keep the continuation branch, boundary argument, CFG cleanup, alias installation, and live-header cleanup. Merely removing effects is not equivalent to detaching a structured prefix. |
| `scalar_handoff_store`, `scalar_handoff_load` | O2 | Keep tiny adapters for the actual one-cell storage operations. At most share place construction; a new storage framework would cost more than these wrappers. |
| `projected_materialization_entry` | O3, O6 | Keep the common projected-body construction. Let a stage owner adopt the resulting entry directly. Input/interface copying and subsequent compaction are separate from output handoff wiring. |
| `StagePreludePlan`, `StagePreludeOutput` | O2–O5 | Useful preparation boundary, but recipe/output agreement is implicit. Replace with the complete prepared plan above; derive `size` from the admitted scalar layout and retain explicit output order. |
| `insertion_site: Option<SideEffectSite>` combined with `ValueRecipeSource` | O3 | Consolidate into one consumer edit: before a site, entry start, or structured continuation. The current product admits combinations that are unused or ignored. Do not erase the three behaviors. |

Source: [materializer and helpers](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:1523), [plan types](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:100).

### C. Last selector: `plan_direct_stage_prelude`

| Code | Objectives | Assessment |
| --- | --- | --- |
| `plan_direct_stage_prelude` | O3, O5 | Keep frontier selection and projection. Consolidate the shared project → enumerate outputs → admit/price → construct plan tail with parallel prelude preparation. Do not merge the two candidate-discovery rules just because their tails agree. |
| `direct_stage_value_is_liftable` | O3, O5 | Stage invariance, exclusion of compile-time constants, loop independence, scalar storage, and supported node kind all have a purpose. Excluding `Project` is a candidate-shaping heuristic, not a general proof that projections are unsafe. The second node lookup can reuse the first borrow. |
| `direct_stage_invocations`, `DIRECT_STAGE_INVOCATION_FALLBACK` | O5 | Keep an estimate of repeated work. Explicit dispatch, inferred image domain, workgroup rounding, and unknown domains are materially different inputs. A keyed dispatch-domain record could avoid the pipeline/stage-index search, but only if it replaces equivalent lookup elsewhere. The fallback 64 is a heuristic, not a correctness fact. |
| `stage_prelude_outputs` | O2, O3, O4 | Keep complete live-out enumeration and storable-layout admission. Make this the prepared plan constructor rather than a factory for an independently supplied vector. |

Source: [direct selector](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:647).

### D. Earlier scalar selector: `plan_parallel_prelude`

| Code | Objectives | Assessment |
| --- | --- | --- |
| `plan_parallel_prelude` | O3, O5 | Keep same-block admission, supported consumer checks, observer restrictions, legal projection, and profitability. The common output/cost/plan tail can be consolidated with direct preparation. |
| `parallel_preludes`, `ParallelPrelude` | O5 | Group captures into one candidate per boundary root. The vector plus root-to-index map and manual consumer deduplication can become an ordered map to a set/list with defined order. Do not demand at least two consumer operations: one operation may launch many invocations. |
| `parallel_prelude_boundary_root` | O3, O5 | Keep whole-boundary selection for a tuple projected inside a structured continuation. Selecting separate fields can duplicate or fail to detach the prefix. Reuse dependency facts if doing so removes this repeated pure walk. |
| `operation_sites` | O3 | Small identity-to-location adapter. Return located consumers once and reuse them for legality, earliest insertion, and invocation estimates. |
| `supports_parallel_prefix_consumer` | O3 | Current implementation supports segmented maps with identity post-processing and outputs. This is a capability restriction, not a universal business rule. Keep until broader recipe support is demonstrated. |
| `source_is_observed_only_by_consumers_or_outputs` | O3, O4 | Keep. Output writers can be retargeted; other serial observers or terminator uses can make this move unsafe. Stage invariance alone does not prove this condition. |
| `launched_consumer_invocations` | O5 | Keep launched rather than just logical invocation accounting. Reuse resolved consumer sites instead of resolving them again. Unknown domains currently price at a workgroup; direct stages use a different fallback. Preserve that policy during a mechanical consolidation. |
| `select_stage_prelude_candidate` | O5 | Tiny priority wrapper: parallel before direct. It expresses current policy. Inlining saves only a few lines and is not a structural improvement. |

Source: [parallel selector](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:574), [supporting queries](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:802).

### E. Cost and relocation support for both prelude selectors

| Code | Objectives | Assessment |
| --- | --- | --- |
| `PreludeMaterializationPolicy`, `PreludeAnalysis`, `should_materialize`, `prelude_materialization_policy` | O3, O5, O6 | Necessity and profitability both matter, but their ownership is mixed. Resolve the meaning of `Required` as discussed above. A decision enum is useful only if the selector consumes that distinction. |
| `analyze_prelude` | O3, O5 | Keep parameter-relocation admission, producer closure, callable analysis, structured cost, and complete output count. Returning `None` conservatively skips unsupported optional moves. Do not make failed pricing silently override a confirmed semantic requirement. |
| `materialization_is_profitable` | O5 | Keep the comparison of repeated producer cost against launch, one producer execution, and a load for every output at every invocation, including the 25% margin. Saturating arithmetic keeps the heuristic bounded. A separate cost object would not simplify this short formula. |
| `entry_parameter_is_scalar_relocatable` | O3 | Keep ABI/context legality separate from uniformity: a stage-invariant texture/sampler/image parameter is not automatically legal in a scalar compute prepass. Already shared by scalar-result and prelude analysis. |
| `effect_cost`, `operation_cost` | O3, O5 | Keep rejection of effects/intrinsics that cannot move and estimates for admitted operations. A plain numeric cost table would lose the distinction between unsupported relocation and expensive computation. A catalog capability record may help if it replaces the same classification elsewhere; otherwise the matches are appropriate. |
| `function_cost` | O3, O5 | Keep recursive/external-function rejection and memoized cost of admitted local callables. Sharing summaries across candidates can save work, but is not by itself a code-size simplification. |
| `graph_block_costs`, `local_value_cost` | O5 | Keep block roots and per-block deduplication. A whole-graph producer closure has different charging semantics: it may cross effect-result boundaries or combine values evaluated in different blocks. Reuse a graph walk only if those boundaries remain explicit. |
| `StructuredCost`, `new`, `path_cost`, `selection_cost`, `loop_cost`, `loop_body`, `linear_cost` | O5 | Keep the small structured fold: sequences add, alternatives take a maximum, loops account for trips or use an estimate. Building a separate cost-expression AST would add construction and evaluation code without a demonstrated second consumer. |
| `fixed_loop_trip_count`, `branch_argument`, `integer_literal` | O5 | Recognize one conservative fixed-loop form. The existing `LoopAnalysis` records membership/invariance, not trip counts, so it is not a replacement. Reusing block-interface incoming arguments could reduce CFG inspection; keep predecessor-edge semantics. These are estimates, not proof that a loop terminates. |
| Cost constants, including launch/load/margin and unknown-loop estimates | O5 | They control whether extra stages are worthwhile. Keep scheduling costs separate from expression-extraction costs. Turning a handful of constants into configuration would add code unless a real tuning interface needs it. |
| Local `Semantic`, `SemanticGraph` aliases | O6 | Readability only; harmless and cheap. |

Source: [cost module](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/cost.rs:38).

### F. Earlier rewrites: runtime arrays and fixed operation outputs

| Code | Objectives | Assessment |
| --- | --- | --- |
| `materialize_runtime_array_result` | O2, O3, O4, O6 | Keep paired capacity/length binding, producer configuration, consumer rewrite, and explicit flow. Consolidate stage construction and output/access records. The projected Filter/runtime checks defend a generic projected body; remove them only if a prepared runtime-Filter recipe guarantees those facts. |
| `RuntimeArrayHandoff` | O2, O3 | Useful bundle, but it repeats the data/length pair already represented by `filter::RuntimeStorage`. Compose that existing pair with the element/result type information, or fold it into the prepared output. Element type and result type have different roles and are not interchangeable. |
| `rewrite_runtime_array_source` | O2, O3, O4 | Keep loading the stored length, constructing a view bounded by that count, retargeting consumers/routes, and removing the original Filter. The capacity-sized view is not an equivalent replacement. Combine common route/cleanup plumbing with other entry rewrites. |
| `materialize_operation_result` | O2–O4, O6 | Keep reuse of an already routed output resource, generated storage for other outputs, producer creation, consumer repair, and flow connection. Replace parallel vectors and repeated enum-to-name/resource-kind/stage-kind matches with prepared output/origin records. |
| `configure_operation_materialization` | O2, O4 | Internal routes and output views must describe the same result. A bound output record can supply both, replacing the zip and another result lookup. |
| `configure_materialized_soac` | O2, O3, O4 | Keep segmented-Screma admission and replacement of source output-write metadata. Canonical access records can eliminate collection/sort code. The `(field, resource)` tuple in `array_outputs` carries a field that is discarded when consumed. |
| `configure_materialized_result` | O2, O3 | Keep the distinction between binding an array destination and emitting a scalar store. A prepared result leaf removes repeated fallible field lookup; the two emission paths remain. |
| `rewrite_materialized_operation_source` | O2, O3, O4 | Keep scalar loads versus array views, SOAC metadata repair, reference/route replacement, and producer removal. Store source/replacement pairs directly: the `resource` component of its replacement tuple is never used after insertion. |
| `output_specs`, `OutputSpec`, `OutputStorage` | O1, O2 | Output shape, value type, element layout, and scalar/array representation are real requirements. Resolve source/projected leaves once, rather than retaining only `field` and looking it up repeatedly. `OutputStorage` earns its two variants because producer and consumer emission differ. |
| `FixedMaterializationKind`, `is_scalar` | O2, O6 | The reason controls array-resource provenance, naming, generated stage kind, and whether stage space is attached. Keep the information, but avoid storing/mapping the same reason into another enum repeatedly. The helper itself is trivial. `SharedArray` and `Gather` share emission; their provenance is still used downstream. |
| `OperationMaterializationPlan` | O2, O3, O6 | Keep the fixed/runtime distinction: a variable result needs cardinality storage. Factor common entry/projection data without forcing fixed and runtime output representations to share invalid fields. |
| `ProjectedOperation` | O3 | A projected graph and source/projected locations are useful. Its runtime variant computes `projected_result` and then discards it. Specializing the prepared output removes that payload; any necessary projection-membership validation should remain in construction. |
| `add_generated_stage`, `connect_resident_flow` | O3, O6 | Their coordinated writes and error checks compensate for stage/entry/flow ownership spread across collections. Move these responsibilities to the owner proposed above. Keep cycle, producer, consumer, and storage consistency requirements. |
| `refresh_resource_reads_for_values` | O3, O6 | Keep rebuilding reads after new handoff loads. The fresh analysis is justified because the graph has changed. Canonical access-map operations can replace its manual update loop; retaining stale reads or dropping write information would not be equivalent. |
| `retarget_input_metadata`, `replace_space_references`, `InputReplacement` | O2, O3 | Necessary today because the value, SOAC input type, segmented extent, and resource metadata are stored separately. A complete resident-array replacement is the right owner. `view_ty` and `resource` can potentially be obtained from the admitted view, but eliminating stored fields must also eliminate repeated extraction. |
| `AllocatedSemantic`, `AllocatedGraph`, `AllocatedSideEffect` | O6 | Local aliases make the resource phase explicit. They are not duplicated runtime state. Keep. |

Source: [fixed materialization](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:997), [runtime materialization](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:1143), [metadata repair](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:1815).

### G. Required result selection and the normalization loop

| Code | Objectives | Assessment |
| --- | --- | --- |
| `plan_scalar_result_handoff` | O2, O3 | Keep used-reduction boundary selection. Reuse a located-operation enumeration with structural selection; preserve global structural-first priority. Its per-entry `OnceCell` already avoids recomputing invocation invariance for each reduction. |
| `scalar_result_requires_handoff` | O2, O3 | Reduction-only shape, no post results, one reduction, relevant boundary, and a used result describe supported candidates. The “one reduction” restriction is implementation scope, not a general semantic theorem. Do not remove it without extending output handling. |
| `scalar_result_is_used` | O2, O5 | Avoid allocating an unobserved result. Keep effect/terminator use checks; ensure caller-owned routes remain represented if their retention contract changes. |
| `invocation_invariant` | O3 | Keep checking the projected execution's parameter dependence and relocatable ABI. It presently examines parameters in the whole entry execution slice, which can conservatively reject a movable reduction because of unrelated entry work. Narrowing to the producer is a possible capability change, not a safe deletion-only refactor. |
| `has_segmented_screma_consumer`, `has_parallel_consumer`, `has_matching_consumer` | O2 | Keep their distinct scheduling predicates. Replace whole-entry rescanning with lookup of the known consumer sites. All current call chains pass `Some(&consumers)`, so the optional input and `None` branch add no business behavior. |
| `plan_operation_result` | O2 | Keep discovery of structural array and runtime-Filter requirements. It shares the effect/result/operation enumeration pattern with scalar selection. A located-operation iterator can remove the repetition without adding a general rule engine. |
| `array_result_residency` | O2 | Shared-use and runtime storage-demand tests express current residency policy. Consumer count and `array_residency_demands` answer different questions; one cannot replace the other. The `Option` around known consumers is unnecessary. |
| `filter_runtime_array_plan` | O1–O3 | Keep runtime-output admission, parallel-consumer requirement, projection, capacity derivation, and optional reuse of existing backing/length. Those two resource options represent actual unresolved/reusable state, unlike the consumer option above. |
| `operation_result_plan` | O2, O3 | Keep fresh ownership, supported writes, projection feasibility, cloneability of dependencies, and valid output layout. A dependency-closed slice does not prove that copying its effects is safe. The projector already owns producer closure; adding a second traversal would regress the design. |
| `resolve_residency_with_policy` | O2, O3, O5, O6 | Keep required-before-optional ordering, one rewrite followed by fresh facts, and authored-only rejection of selected required materializations. A unified candidate result can reduce dispatch argument unpacking, but the loop is small and already clear. Batch application would need invalidation/conflict handling and is not a demonstrated code reduction. |
| `resolve_residency` | O6 | Public compatibility wrapper that selects the default policy and translates errors to `String`. Keep unless its callers/API contract are deliberately changed. |
| `residency_facts` | O2, O3, O5 | Entry snapshot ownership is already useful. Callable bodies are also added to `SemanticGraph`, but current selection queries only entry operations/captures, and the incidence adapter has no call-argument-to-callee value edges. That callable-body indexing is a candidate for removal after checking callable fixtures; callable analysis used by dependence/cost must remain. |

Source: [normalization](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:215), [required selection](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:285), [invariance and consumers](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:935).

### H. Dependency and storage-demand structures used by that loop

| Structure or operation | Objectives | Assessment |
| --- | --- | --- |
| `GraphAnalysis` and its lazy producer/interface/loop/slice facts | O2, O3 | Keep. It already binds reusable facts to an immutable graph borrow. Another per-pass analysis framework would mostly duplicate it. Solved dependence with entry-specific seeds is a different result. |
| `SemanticGraph`: consumers, captures, sites, array residency demands | O2, O3, O5 | All four relations have callers in selection. A site/classification query can replace consumer rescans. Keep body-qualified `SourceValue`; graph-local value IDs alone cannot safely index several entries. |
| `Facts`, `OperationFact`, `ValueFact`, `Incidence`, `ScopeKey` | O2, O3 | The owned incidence representation is shared with fusion. Pure dependencies, projections, block boundaries, operation sites, and storage uses matter. Its fusion ordering/resource machinery is not all directly consumed by residency; splitting it is worthwhile only if it avoids work without duplicating adapters. |
| Runtime array/storage-use propagation in `SemanticGraph::from_facts` | O2 | Keep demand discovery through pure nodes, projections, and cross-block inputs. A syntactic search for `Index` alone would miss indirect requirements. |
| `LiveSlice`, `GraphProjection`, `ProjectedValueRecipe`, `ValueRecipeSource` | O3, O4 | These already replace substantial procedural closure/projection code. Keep dependency closure, source/projected correspondence, live-outs, and detachment boundary. Extend their output contract instead of rebuilding those relations in allocation. |

The current structured-prelude primary-output convention is supported by source inspection: the projector requests one boundary value, and `LiveSlice::outputs` yields requested values before additional live-outs. It is not an observed ordering bug. A prepared plan would make that dependency local and explicit. [Slice output order](C:/Users/gmiller_amilarcap/dev/wyn/wyn-slice/src/lib.rs:60), [structured boundary admission](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/graph_projector.rs:359)

### I. Earlier allocation step: dynamic publication

| Code | Objectives | Assessment |
| --- | --- | --- |
| `realize_dynamic_publication` | O1, O2, O4 | Keep applying realization to entries, functions, and constants. Only entry outputs need host capacity metadata. Returning per-entry publication records could avoid rediscovering affected resources later. |
| `realize_graph_dynamic_publication` | O2, O4 | Keep identifying published runtime Filters, selecting backing, binding length, updating access, and retyping results. Inferring backing from the first writable access is procedural reconstruction of an output relation; a publication record should name the actual destination explicitly. Preserve deferred/unpublished cases. |
| `filter_capacity_size` | O1, O2 | Keep the Filter-specific diagnostic around shared `LogicalSize::for_space`. It is a small adapter, not competing size policy. |
| `bind_filter_storage` | O1, O2 | Already the correct shared policy for direct publication and runtime handoffs. Keep data and length admission together. Reuse existing resources or create them, refine an unspecified host capacity, and reject incompatible layouts. |
| `require_filter_resource` | O1, O2 | Keep compatibility/refinement checks: arena interning alone does not compare repeated allocations' type/size. A checked storage-binding operation can own them, but they cannot disappear simply because IDs are interned. |
| `filter_capacity_buffer_len` | O1, O4 | Keep the host-ABI conversion while entry output metadata still needs it. Every successful arm returns `Some`, so its result can be `Result<BufferLen, String>` and the output assignment can add `Some` once. Do not remove rejection of a host capacity depending on a non-host resource. |
| `realize_filter_output_capacities` | O1, O4 | Keep synchronizing host output capacity until publication reads it exclusively from the resource contract. A resource-to-publication index can replace the affected-resource list search. Do not replace capacity with survivor length. |
| `realize_filter_result_types` | O2, O3 | Keep making bound results storage-backed. Combining this with storage binding can remove the second graph scan, but node mutation must occur after mutable effect borrows end. A short collected retype list may still be the cleanest implementation. |
| `filter::Output`, `RuntimeOutput`, `RuntimeBacking`, `RuntimeLength`, `RuntimeCapacity` | O1, O2 | Their distinctions contribute: local vs runtime representation, capacity provenance, deferred vs bound backing, and implicit vs stored cardinality are separate facts before realization. Do not globally collapse them to `Option<RuntimeStorage>` without proving which intermediate combinations are legal. |
| `filter::RuntimeStorage` | O2 | Good existing complete data/length pair. Reuse it in handoff records. Keep its required length, unlike fixed/scalar storage. |

Source: [dynamic publication and storage policy](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:515), [Filter states](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/soac/filter.rs:58).

### J. Earlier allocation step: remap bindings to logical identities

| Code | Objectives | Assessment |
| --- | --- | --- |
| `remap_program_resources` | O1, O6 | Keep the phase conversion for functions, constants, entries, and the initial authored stages. Stage-body ownership can remove separate stage-ID construction. Program reconstruction across different type families is an actual transformation, not automatically removable boilerplate. |
| `remap_entry_resources` | O1, O4, O6 | Keep all inputs, outputs, internal routes, block references, parameters, result types, and declarations aligned. This is a useful candidate for a shared fallible entry mapper if other phase conversions need it too. A mapper covering only the graph is insufficient. |
| `remap_function_resources`, `remap_constant_resources` | O1, O6 | Keep signatures/result types and body resources in agreement. Function ABI reconstruction is intentional. Reuse family mapping primitives where they remove repeated code. |
| `remap_graph_resources` | O1, O6 | Keep graph/phase mapping and embedded type rewrite. Its three callers discard the returned node map; remove that map from this wrapper's return value, while retaining the lower-level map needed during graph construction. |
| `remap_soac_resources` | O1, O6 | Keep variant-specific resource mapping. It already uses `soac::remap::Remap`. Wrapping and unwrapping `screma::Segmented` for common space/slots/access fields is a possible shared-record improvement; a new generic visitor solely for this match may add more than it removes. |
| `allocate_type_resources` | O1, O6 | Keep recursive `Buffer(binding)` → `Resource(id)` conversion and missing-binding diagnostics. Fallible type visitors could remove captured error variables here, in graph type rewrite, and in function/entry result rewrite. This is API consolidation, not a new data structure. |
| `entry_resource_declarations` | O1, O3 | Keep one declaration per resource and merging input/output roles. An ordered resource-use map can replace the vector-plus-position map. Do not drop role merging for a resource used both ways. |
| `ResourceAllocationContext`, its `resource_for_binding` | O1 | The wrapper contains only the finalized arena and a checked lookup; it adds no independent invariant. It can disappear if the arena or a shared lookup adapter provides the same diagnostic, also used during size resolution. Savings are modest. |

Source: [program conversion](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:262), [graph conversion](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:403), [entry conversion](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:821).

### K. First described steps: size resolution and reservation

| Code | Objectives | Assessment |
| --- | --- | --- |
| `lower_host_size_policies` | O1 | Keep resolving sizes after all bindings exist. A registry transformation can own this traversal; it cannot remove the forward-reference requirement. |
| `ResourceAllocationBuilder::logical_size` | O1 | Each size variant describes a real host contract. Preserve referenced-resource identity and source/destination element strides. Generic resource-parameterized size policies might consolidate this with reverse conversion, but the host/non-host admission distinction remains. |
| `set_host_size` | O1 | Keep merge behavior: unknown plus known becomes known; compatible repeats agree; conflicting known sizes fail. This is domain policy appropriate for the registry. The explicit `RuntimeProvided/RuntimeProvided` arm is subsumed by the following `Some(_)/RuntimeProvided` arm and can be removed. |
| `ResourceAllocationBuilder::finalize` | O1, O6 | Keep rejecting unprocessed drafts. Eliminate allocate-again-and-compare-ID bookkeeping by preserving the registry's IDs through payload resolution. |
| `reserve_host_resources`, `reserve_host` | O1 | Keep deduplication and element-type conflict detection. The current arena's `allocate_host` returns an existing ID without checking compatibility, so replacing the builder with calls to it alone would lose required behavior. |
| `ResourceAllocationBuilder::resource_for_binding` | O1 | Keep missing-reference errors. Consolidate its duplicate diagnostic with finalized lookup. |
| `DraftLogicalResource`, `ResourceAllocationBuilder` | O1 | Binding, element type, and unresolved size state all have meaning. Their temporary vector/map duplicates the eventual arena; preserve one ID registry instead. Until then, `Option<HostSizePolicy>` correctly distinguishes unvisited from deliberately unspecified. |
| `InterfaceResource`, `interface_resources` | O1 | Keep. They already remove duplicate enumeration of input/output bindings and retain distinct optional storage role/size information. Replacing this small borrowed view with an elaborate registry adapter would not automatically save code. |
| `plan_logical_resources`, `plan_logical_resources_with_policy`, `allocate_semantic_resources` | O1–O6 | Keep the short public orchestration and default-policy wrapper. The phase boundaries explain responsibility and error propagation; collapsing these functions is not where the code volume resides. |

Source: [registry](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:117), [interface traversal](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:318).

### L. Actual first step: retain host length inputs

| Code | Objectives | Assessment |
| --- | --- | --- |
| `retain_output_lengths` | O1, O2 | Keep recording scalar extent provenance before it is lost and propagating it into generated storage size policy. Reuse the `host_extents` map when collecting output dependencies: the same scalar extents are traversed again through `host_dependencies`. Use one classified extent result to replace repeated scalar/non-scalar scans. |
| Per-output `(known_inputs, known_elem_bytes)` merge | O1 | Unioning host inputs from several writers is meaningful. The stride-conflict check compares values read from the same immutable `entry.outputs[slot]` field during that read phase; it cannot observe differing strides in this loop. Keep type/stride validation where distinct declarations are actually compared. |
| Rewrite `SegExtent::Value` to `HostProvided` | O1, O2 | Keep the extent's graph value and host input provenance together. Rebuilding the entire `SegSpace` is an implementation choice; a safe dimension-mapping method could remove cloning/constructor plumbing. Preserve nonempty-space validity. |
| `scalar` | O1 | Keep the supported host scalar ABI classification (`i32`, `u32`, `f32`). It is deliberately narrower than all possible numeric types. |
| `host_dependencies` | O1 | Keep dependency closure, canonicalization, host-input provenance, and deterministic deduplication. Reuse results for the same extent rather than recomputing the closure. |
| `uniform_location` | O1 | Keep parameter-to-interface mapping, uniform binding, std140/member offsets, vector element offsets, names, and checked arithmetic. Generic dependency reachability cannot replace ABI layout. A memoized provenance result could remove repeated path traversal; the depth guard is not itself evidence that provenance is unnecessary. |

Source: [host length retention](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/host_length.rs:14).

### M. Resource and stage definitions that make the contract possible

| Definition | Objectives | Assessment |
| --- | --- | --- |
| `ResourceId`, `SemanticResourceRef` | O1, O6 | Keep logical identity separate from backend binding identity. These are useful type distinctions. |
| `LogicalSize`, `HostSizePolicy` | O1, O2 | Keep explicit logical sizing and the host-only externally supplied state. A generated resource must not silently become unsized. The arena/origin representation already enforces part of this contract. |
| `LogicalResourceArena`, `LogicalResource`, `ResourceOrigin`, `HostResource`, `CompilerResource`, `CompilerResourceKind` | O1, O6 | Keep identity, ownership, and compiler-resource provenance. The arena guarantees dense unique identity, not all reference/type/size validity. Do not remove compatibility checks merely because allocation interns keys. Scope future registry changes around the existing arena. |
| `SemanticResourceDecl` | O3, O6 | Read/write use information feeds ordering and interfaces. An ordered use map could replace vectors, but the relation itself remains necessary. |
| `ResidencyProgramData`, `ResourceProgramData`, `AllocatedProgramData` | O1, O6 | Phase-specific state is useful. Remove duplicated entry/stage ownership and externally synchronized maps, not the separation between a draft and finalized stage graph. |
| `GeneratedStageKind`, `StageOrigin` | O3, O6 | Authored/generated distinction affects later planning; kind also supplies diagnostics/provenance. Array kinds with absent space and scalar kind with present space are combinations the current producers do not emit. Variant-specific payloads could prevent these states and eliminate mapping branches; avoid merely replacing one enum with two. |
| `ResidentStorage` | O2, O6 | Scalar/fixed storage versus capacity-plus-length storage is meaningful. An enum can make that distinction explicit, or a checked output record can supply it. The current optional length is not automatically redundant. |
| `StagedProgram`, `StagedProgramBuilder`, `StagedIrBuilder` | O3, O6 | Keep the graph and its checked edge operations. It is already a data structure replacing scheduling bookkeeping. Let it own more of the relationship rather than introducing another topology graph. |

Source: [resources](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/program.rs:214), [stage state](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/program.rs:1515).

## Concrete cleanup candidates before a larger redesign

These findings follow directly from current callers and data flow. No production changes or measured LOC savings are claimed.

| Candidate | Evidence and deletion boundary |
| --- | --- |
| Remove optional consumer arguments | Both selection paths construct a consumer set and pass `Some`; helpers are private. Remove option wrapping, `map_or`, and the `None` case together. Keep empty-set behavior. [Call sites](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:299) |
| Drop unused fields from temporary tuples | `array_outputs` discards its field index; source replacements discard their resource ID. Use resources and source/replacement pairs respectively. [Array output tuple](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:1398), [replacement tuple](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:1494) |
| Return a required `BufferLen` | `filter_capacity_buffer_len` has no successful `None` case. Wrap once when assigning optional interface metadata. [Conversion](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:659) |
| Stop exposing the unused graph node map from the allocation wrapper | All three callers discard it. Keep the lower graph mapper's internal node correspondence and keep the returned block map used by entry routes. [Wrapper](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:486) |
| Reuse stored host extent dependencies | The first dimension loop fills `host_extents`; the later input collection re-runs the same analysis. Collect from the map for admitted scalar extents. [Retention loop](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/host_length.rs:14) |
| Remove the impossible per-slot stride conflict and redundant size match alternative | They do not distinguish any current reachable input. Retain genuine cross-declaration size/type conflicts. [Per-slot comparison](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/host_length.rs:71), [size merge](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:214) |
| Reuse the existing node borrow in direct liftability | The node kind and type are read from the same immutable graph entry. [Predicate](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency.rs:757) |

The prelude coverage guard, runtime `projected_result`, and final stage-body assertions are a different category: their representation can be eliminated, but their invariant must first acquire a clear owner. They are not unconditional dead-code deletions.

## Order of work and validation

1. Apply the small proven cleanups as one bounded change. Retain all existing behavior and diagnostics that correspond to reachable invalid inputs.
2. Implement complete prelude outputs and the explicit consumer edit boundary, starting at the last rewrite. Then extend the same output ownership to fixed/runtime operation materialization where it removes real duplication.
3. Centralize resource-access updates and complete array replacement. This should delete the scattered repair steps it supersedes, not leave both representations alive.
4. Move stage-body/flow ownership only after the smaller output change is stable. This has the highest potential to remove coordinated state across functions, and the widest API impact.
5. Revisit host registry finalization if its replacement produces a net deletion. Do not add incremental fixpoint machinery or a separate cost AST without a demonstrated benefit.

For every structural change, count production lines added and removed including constructors, adapters, validation, and call-site changes. Fewer lines in `residency.rs` do not count as a reduction if equivalent or greater code moved elsewhere. Savings described here are qualitative until implemented.

Relevant existing checks to preserve when implementing:

- Forward binding references, compatible repeated declarations, conflicting types/sizes, unresolved host sizing, and missing final size sources in [program tests](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/program_tests.rs:104).
- Filter capacity/length compatibility in [allocation tests](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/mod.rs:1132).
- Structural-before-scalar ordering and idempotent residency in [residency tests](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/residency_tests.rs:26).
- Relocation legality versus invariance, profitability with multiple outputs, structured-storage policy, and fixed-loop estimates in [cost tests](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/egir/allocation/cost_tests.rs:29).
- One/multiple parallel consumers, composite live-outs and host outputs, structured prefixes, cheap recomputation, resource-manifest flows, and precedence before an expanded Filter in [scalar integration cases](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/integration_tests.rs:6595).
- Runtime Filter handoff, actual count versus capacity, and paired output length in [Filter handoff](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/integration_tests.rs:3025), [cardinality](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/integration_tests.rs:7692), and [publication](C:/Users/gmiller_amilarcap/dev/wyn/wyn-core/src/integration_tests.rs:11354).

Add focused regression coverage only for newly established behavior or an invariant not covered by those cases. In particular, resolve the `Required` prelude/topology question with an end-to-end authored-only case before changing its classification. Source inspection alone does not prove that later lowering mishandles it.

Coverage was checked against the production function/type declarations in the four allocation files, and every local source link was checked for an existing file and line. This is declaration-level coverage with grouped body assessments, not an assertion that every individual statement received a separate finding.

This review changed only this Markdown report. Builds and runtime tests were not run; the evidence is current source, call sites, data-flow inspection, and existing test definitions.
