# Wyn

A minimal compiler for a Futhark-like programming language that generates SPIR-V and WGSL code for GPU shaders.

## Features

- Functional syntax for shader programming
- Hindley-Milner type inference with polymorphic types
- Higher-order functions (map, reduce, zip, etc.)
- Pattern matching
- SPIR-V and WGSL code generation for Vulkan/WebGPU shaders
- Vertex, fragment, and compute shader support
- Vector and matrix types optimized for GPU operations
- Array operations with size tracking
- Loop constructs

## Project Structure

The project is organized as a Rust workspace:

- **`wyn-core/`** - Compiler library (lexer, parser, type checker, TLC, EGIR mid-end, SSA, SPIR-V/WGSL backends). Includes an in-crate generic SSA framework at `ssa::framework` (blocks, values, instructions, terminators) used only for codegen.
- **`wyn-staged-ir/`** - Invariant-preserving typed DAG of executable stages and resident flows
- **`wyn/`** - Command-line executable
- **`wyn-analyzer/`** - Language server (in development)
- **`extra/viz/`** - Visualization tool for rendering SPIR-V shaders
- **`prelude/`** - Standard library functions written in Wyn

For package layout, manifests, dependencies, imports, and build commands, see
[Writing Wyn Packages](PACKAGES.md).

## Compiler Architecture

The compiler uses a multi-stage pipeline with typestate-driven phases. Each
stage consumes `self` and returns the next stage, enforcing valid ordering at
compile time.

The principal IR phases are:

1. **AST** — parsed and typed source structure.
2. **TLC** — normalized typed functional calculus with explicit ownership and
   first-order calls.
3. **Semantic EGIR** — per-body acyclic e-graphs with explicit semantic SOACs,
   effects, output routes, and logical resources.
4. **Staged IR** — a typed DAG whose nodes own semantic EGIR bodies and whose
   resident flows make every cross-stage value explicit.
5. **Physical EGIR** — a validated physical kernel DAG whose nodes reference
   target-planned `EGraph<Physical>` bodies by stable entry identity.
6. **SSA** — demand-elaborated backend codegen IR.

Staged IR is a real representation boundary even though its stage bodies are
still semantic EGIR. It replaces the flat collection of authored entries plus
separately reconstructed materialization requirements with explicit stage and
flow topology.

Physical EGIR is graph-based at both levels. `PhysicalKernelGraph` retains
stable kernel identities, dependency edges, dispatch domains, resource
accesses, output projection, and one-to-one body ownership. Each referenced
body is itself an e-graph sea plus a CFG skeleton. The bodies remain in the
program's existing entry arena so generic EGIR body traversals stay reusable;
the arena is storage, not an alternative flat-program authority.

When describing transitions, the tables distinguish three related units:

- A **checkpoint** is a named compiler state whose invariant has been
  established. EGIR inspector checkpoints are snapshots of these states or of
  explicitly exposed intermediate states.
- A **checkpoint transition** consumes one checkpoint and produces the next.
  Its public function may be an orchestrator rather than one indivisible pass.
- A **sub-pass** is an independently meaningful analysis, rewrite, validation,
  or other traversal. One transition can run several sub-passes, including
  multiple sub-passes over the same Rust typestate.

The tables list the actual functions that act as sub-passes inside each
checkpoint transition. In the **Sub-pass sequence** column, commas mean
sequential execution and `|` means mutually exclusive alternatives. A repeated
function name means the transition function is itself the sub-pass. Local
per-node helpers are omitted. These functions are the unit used for inspector
rundowns even when the inspector does not yet expose an intermediate snapshot.

A full compilation follows one of the explicitly selected typestate paths. The
CLI and Wasm API use the profile-aware entry points; the canonical test helpers
take the `AllowGenerated` path through their convenience entry points. Each
invokes the required sub-passes for that path in the order below;
`resolve_imports` is simply a no-op when the source contains no imports.
Variability comes from four branch points:

- type holes are either rejected or filled;
- the pipeline-topology policy either lifts stage-uniform values into generated
  stages or preserves authored stage boundaries;
- `egir::plan` builds a serial or parallel schedule from the selected profile;
- SSA is prepared and lowered through either the SPIR-V or WGSL backend.

Rows whose function boundaries are easy to conflate with sub-pass boundaries
need these clarifications:

- `types::run::type_check` builds value-name resolution, performs inference and
  checking, and materializes the typed AST;
- `tlc::normalize_soacs` performs the SoA transform, Map+Zip flattening, and Zip
  elimination during one traversal, so those are responsibilities of one
  sub-pass rather than three sub-passes;
- `tlc::monomorphize` first specializes intrinsics, then monomorphizes reachable
  user definitions;
- `tlc::defunctionalize` composes closure conversion, higher-order
  specialization, and closure-call lowering;
- `to_egraph` orchestrates the smaller construction sub-passes named in its
  table description, while semantic optimization is exposed as two typed
  transitions and `egir::reify_soacs` is one reification sub-pass;
- `egir::apply_pipeline_topology_policy` selects either
  `lift_stage_uniform_values` or `preserve_authored_stage_boundaries`;
- `egir::plan_logical_resources_with_policy` orchestrates semantic resource
  allocation and policy-aware residency, then crosses from semantic EGIR into
  staged IR; `egir::plan_logical_resources` is its allow-generated convenience
  wrapper;
- `egir::plan` consumes staged IR and crosses into physical EGIR by selecting,
  scheduling, and publishing target-aware recipes;
- `lower_egir_to_ssa` is the convenience wrapper over the eight physical EGIR
  checkpoint transitions listed in the Physical EGIR table;
- `ssa::prepare_spirv` eliminates dead SSA values before abstract-type and
  buffer-layout validation; `ssa::prepare_wgsl` first promotes addressable
  constants, then eliminates dead values and validates abstract types.

### Mid-End: Acyclic E-Graph

The mid-end is an **acyclic e-graph**: a sea of hash-consed pure nodes
plus a skeleton CFG of effectful instructions. Purity is
**blacklisted**, not whitelisted — only `Alloca` / `Load` / `Store`
stay in the skeleton; everything else (calls, intrinsics, storage
views, projections, indexing) is hash-consed in the pure sea. Named
user calls are classified from the complete TLC call graph and enter EGIR in
the pure sea only when their definitions are proven effect-free.

Most optimizations fall out of the data structure without dedicated
passes:

- **GVN** — pure-node interning deduplicates by structural equality.
- **Constant folding** — the interner consults a folder before
  inserting, so folds compose.
- **DCE** — elaboration is demand-driven from skeleton roots;
  unreached pure nodes are never emitted.
- **CSE along the domtree** — emitted pure nodes are scoped to
  dominator regions; siblings never share.
- **LICM** — pure nodes float to the outermost loop where all their
  operands are available.
- **Partial inlining** — a bounded policy fixpoint uses the generic EGIR
  inliner for profitable explicit-loop calls with both invariant and varying
  arguments, after which ordinary LICM hoists invariant-only subgraphs.
- **Branch folding + redundant-phi elimination** — the skeleton CFG
  is rewritten before elaboration.

### Frontend (AST)

| Checkpoint transition | Sub-pass sequence | Description |
|-----------------------|-------------------|-------------|
| `parser::parse_file` | `lexer::tokenize`, `Parser::parse` | Tokenize and parse source into the AST |
| `resolve_imports::resolve_imports` | `resolve_imports` | Recursively expand filesystem imports; this is a no-op when the program has no imports |
| `elaborate_modules::elaborate_modules` | `elaborate_modules` | Elaborate inline modules into the module manager and remove module declarations from the source AST |
| `name_resolution::resolve_names` | `resolve_names` | Resolve qualified module member syntax |
| `resolve_resources::resolve_resources` | `resolve_resources` | Replace source resource declarations and `#[view]` attributes with concrete entry binding metadata |
| `ast_const_fold::fold_constants` | `fold_constants` | Fold compile-time integer expressions needed by static-size inference |
| `resolve_placeholders::resolve_type_placeholders` | `resolve_type_placeholders` | Replace type and size placeholders with stable inference variables and build module-spec schemes |
| `resolve_opens::resolve_opens` | `resolve_opens` | Consume `open` declarations and qualify affected identifiers |
| `types::run::type_check` | `build_name_resolution`, `TypeChecker::load_builtins`, `TypeChecker::check_program`, `materialize`, `stage_context::validate` | Build value-name resolution, perform Hindley-Milner inference/checking (including one-directional `*T -> T` weakening at coercion sites), materialize the typed AST, and validate its context |
| `ast_type_holes::reject_type_holes` / `fill_type_holes` | `reject_type_holes` \| `fill_type_holes` | Reject typed holes for checking, or replace them with typed defaults for compilation, before TLC lowering |

### TLC (Typed Lambda Calculus)

| Checkpoint transition | Sub-pass sequence | Description |
|-----------------------|-------------------|-------------|
| `tlc::lower_from_ast` | `Transformer::transform_program`, `check_unextracted`, `stage_extract::extract` | Convert the fully typed AST to minimal typed lambda calculus, validate still-unextracted linear pipeline handles, then extract pipeline stages |
| `tlc::pin_entry_buffers` | `pin_entry_buffers` | Substitute each storage entry parameter's concrete `Buffer(set, binding)` into its type, so view provenance flows by unification |
| `tlc::validate_ownership` | `validate_ownership` | Reject source-level use-after-move before simplification or inlining can erase the call boundary carrying the `*T` contract |
| `tlc::partial_eval` | `partial_eval` | Apply constant folding and algebraic simplifications |
| `tlc::normalize_soacs` | `soa::transform_program` | Transform SoA types, flatten Map+Zip, and eliminate standalone Zip operations during one traversal |
| `tlc::monomorphize` | `specialize_intrinsics`, `Monomorphizer::monomorphize` | Specialize polymorphic intrinsics, then emit reachable user-function monomorphs, including separate monomorphs for distinct view buffers |
| `tlc::rep_specialize` | `rep_specialize` | Clone callees whose abstract array parameters receive producer-known concrete variants, before forced SOAC-helper inlining |
| `tlc::inline_small` | `inline_small` | Inline small user functions and constants |
| `tlc::force_inline_soac_helpers` | `force_inline_soac_helpers` | Force-inline functions whose bodies recursively contain a SOAC or `length`, exposing producer/consumer edges to EGIR |
| `tlc::renormalize_inlined_soa` | `soa::transform_program` | Re-run that combined normalization traversal after inlining exposes new structure |
| `tlc::canonicalize_conditional_producers` | `canonicalize_conditional_producers` | Turn eligible array-valued conditionals into pointwise producers without choosing fusion, routes, or storage |
| `tlc::normalize_soacs_to_anf` | `normalize_soacs_to_anf` | Lift nested SOAC expressions onto flat let chains so EGIR conversion sees every semantic producer/consumer edge |
| `tlc::float_runtime_index_nested_producers` | `float_runtime_index_nested_producers` | Float nested producers used by runtime indexing into let bindings so residency can recognize ordinary gathers |
| `tlc::defunctionalize` | `convert_closures`, `specialize_higher_order_functions`, `lower_closure_calls` | Lift lambdas and make captures explicit, eliminate higher-order parameters by specialization, then thread captures into direct calls; verifiers guard their boundaries |
| `tlc::fold_generated_lambdas` | `fold_generated_lambdas` | Inline compiler-generated lambda definitions back at call sites and remove dead definitions |
| `tlc::apply_ownership` | `apply_ownership` | Promote safe functional updates and attach `UniqueInput` candidates; EGIR later decides physical reuse from post-fusion liveness |
| `tlc::filter_reachable` | `filter_reachable` | Eliminate unreachable definitions |
| `tlc::infer_input_slice_bounds` | `infer_input_slice_bounds` | Infer minimum input-buffer lengths before semantic EGIR conversion |

#### Sub-pass-ordering dependency assertions

The table above is one valid topological sort of the constraints below
(`optimize_tlc_for_test` in `wyn-core/src/lib.rs` and the CLI pipeline in
`wyn/src/main.rs` must stay in sync with it). `A ≺ B` means A runs before B.
Each notes how it's enforced; when you move a sub-pass, check it here.

- **`validate_ownership` ≺ `partial_eval`** — source-level consumption must be
  checked while the call boundary carrying the `*T` contract still exists.
  *Enforced by:* `partial_eval` is defined only on `TlcOwnershipValidated`.
- **`monomorphize` ≺ `defunctionalize`** — monomorphization specializes the
  still-higher-order program; defunctionalization then removes function-typed
  parameters and gives EGIR concrete callable references and captures.
  *Enforced by:* the TLC typestate chain.
- **`defunctionalize` ≺ `fold_generated_lambdas`** — fold inlines the
  `_w_lambda_*` defs that defunc generates. *Enforced by:* the TLC typestate
  chain.
- **`monomorphize` ≺ `force_inline_soac_helpers`** — force-inline's free-type-var
  guard skips any helper still carrying an unresolved element-type `Variable`, so
  helpers must be concrete first. *Enforced by:* the TLC typestate chain.
- **`rep_specialize` ≺ `force_inline_soac_helpers`** — makes `filter`-result
  helpers representation-concrete (`Abstract` → `Bounded`/`View`) so the guard
  admits them. *Enforced by:* the TLC typestate chain.
- **`force_inline_soac_helpers` ≺ `normalize_soacs_to_anf`** — every SOAC helper is inlined so
  EGIR receives explicit intra-def producer/consumer edges rather than needing
  cross-call summaries. *Enforced by:* the TLC typestate chain.
- **`normalize_soacs_to_anf` ≺ `float_runtime_index_nested_producers` ≺ `defunctionalize`
  ≺ `to_egraph`** — nested producers become explicit let-bound demand edges
  before callables are lowered to the reference/capture ABI consumed by EGIR.
  *Enforced by:* the TLC typestate chain.
- **`apply_ownership` ≺ `to_egraph`** — TLC records source-level uniqueness but
  does not decide physical reuse. EGIR resolves `UniqueInput` only after output
  realization and semantic optimization. *Enforced by:* the TLC/EGIR typestate
  chain and a physical-expansion assertion rejecting unresolved candidates.

### Semantic EGIR (Acyclic E-Graph IR)

Each row below is one sub-pass. The separate **Checkpoint orchestrator** column
names the public transition function; an orchestrator is not an additional
sub-pass merely because it calls the listed rows. Consecutive rows for the same
orchestrator are in execution order unless the **Role / condition** column marks
them as alternatives or part of a fixpoint.

`allocate_semantic_resources` and `resolve_residency_with_policy` are nested
orchestrators within `plan_logical_resources_with_policy`. The policy-free
`plan_logical_resources` and `resolve_residency` wrappers select
`AllowGenerated`. Allocation runs the first four allocation rows. Residency
first normalizes required operation-result handoffs to a fixpoint, then, when
generated topology is allowed, considers the cost-driven prelude rows. A
prelude rewrite restarts required-residency normalization because it changed
the graph. Under `AuthoredOnly`, a required new handoff is an error and the
optional prelude phase is skipped. The nested orchestrators do not receive
sub-pass rows of their own. In the table, `plan_logical_resources` labels the
shared implementation reached through either public resource-planning entry
point.

The global logical-resource arena is authoritative for element types and
logical sizes. Entry-local `SemanticResourceDecl` values contain only a
resource identity and its role in that entry. Dynamic Filter storage follows
one allocation policy: host output slots and cross-scheduling-boundary
residency determine the required backing and length resources, then the Filter
publication state is rewritten to name them.

| Checkpoint orchestrator | Sub-pass | Role / condition |
|-------------------------|----------|------------------|
| **`to_egraph`** | `convert_program` | Discover and hoist pure arity-zero constants; convert the remaining functions and entries to raw per-body e-graphs; normalize callable interfaces; finalize output routes and ABI size policies without choosing scheduling or physical resources |
| **`reify_soacs`** | `reify_soacs` | Link output routes to semantic producers and reify reachable raw SOACs with authoritative spaces, bodies, captures, publication/resource effects, placement, and dependencies |
| **`reify_soacs`** | `verify_canonical_resource_accesses` | Debug builds only: verify stored Screma and Filter resource summaries |
| **`optimize_semantic_operations`** | `analyze_dead_seg_ops` | Find the next dead segmented operation before attempting fusion |
| **`optimize_semantic_operations`** | `apply_dead_seg_ops` | Run when dead-operation analysis returns a patch, then restart the fixpoint |
| **`optimize_semantic_operations`** | `semantic_graph::dependencies` | When no dead-operation patch exists, rebuild the dependency graph used by fusion legality checks |
| **`optimize_semantic_operations`** | `rewrite_once` | Run when there is no dead-operation patch; apply at most one legal fusion rewrite and restart if it changed the graph |
| **`apply_pipeline_topology_policy`** | `lift_stage_uniform_values` | `AllowGenerated` only: lift values that are uniform at their execution stage after the fused graph is final |
| **`apply_pipeline_topology_policy`** | `preserve_authored_stage_boundaries` | Alternative for `AuthoredOnly`: cross the optimization checkpoint without creating stages |
| **`apply_pipeline_topology_policy`** | `semantic_graph::verify` | Debug builds only: validate the final semantic dependency graph after either alternative |
| **`plan_logical_resources`** | `reserve_host_resources` | Reserve every authored interface binding before any cross-resource size is resolved |
| **`plan_logical_resources`** | `lower_host_size_policies` | Resolve logical host-resource sizes after all referenced bindings exist |
| **`plan_logical_resources`** | `remap_program_resources` | Replace descriptor bindings in types, graphs, interfaces, and routes with target-independent logical resource identities |
| **`plan_logical_resources`** | `realize_dynamic_publication` | Realize direct host publication of dynamic results, binding Filter output backing and allocating its length resource through the shared Filter storage policy |
| **`plan_logical_resources`** | `semantic_graph::dependencies` | At the start of every residency fixpoint iteration, rebuild operation dependencies used by candidate selection and legality checks |
| **`plan_logical_resources`** | `semantic_graph::array_residency_demands` | At the start of every residency fixpoint iteration, rediscover runtime-composite array results whose current uses require storage |
| **`plan_logical_resources`** | `plan_operation_result` | First-priority residency analysis: select at most one structural shared-array, gather, or cross-boundary runtime-array result and project its producer recipe |
| **`plan_logical_resources`** | `plan_scalar_result_handoff` | When no structural result requires residency, select at most one used invocation-invariant reduction result that must cross a segmented or non-compute scheduling boundary |
| **`plan_logical_resources`** | `materialize_operation_result` | When either planner selects fixed outputs, create the producer entry and logical handoffs, rewrite consumers, then restart required-residency normalization |
| **`plan_logical_resources`** | `materialize_runtime_array_result` | Alternative for a selected runtime-array result: apply the shared Filter capacity-and-length storage policy, then restart required-residency normalization |
| **`plan_logical_resources`** | `semantic_graph::dependencies` | After required residency reaches a fixpoint, rebuild dependencies for cost-driven prelude selection when generated topology is allowed |
| **`plan_logical_resources`** | `plan_parallel_prelude` | When no structural or scalar-result candidate exists, select at most one cost-eligible scalar prelude shared by parallel consumers |
| **`plan_logical_resources`** | `materialize_stage_prelude` | When `plan_parallel_prelude` succeeds, create its scalar handoff entry, rewrite the consumer prefix, then restart the fixpoint |
| **`plan_logical_resources`** | `plan_direct_stage_prelude` | When neither earlier planner succeeds, select at most one cost-eligible stage-invariant scalar frontier for a direct shader stage |
| **`plan_logical_resources`** | `materialize_stage_prelude` | When `plan_direct_stage_prelude` succeeds, create its scalar handoff entry, rewrite the stage prefix, then restart the fixpoint; otherwise residency is complete |

The semantic EGIR order is load-bearing:

- **`from_tlc` before `reify_soacs`** - conversion constructs every declared output route; reification then links those routes against the completed graph before constructing semantic SOAC state.
- **`reify_soacs` before `optimize_semantic_operations`** - fusion legality depends on explicit domains, canonical resource summaries, semantic operation IDs, effects, and dependency edges.
- **`optimize_semantic_operations` before `apply_pipeline_topology_policy` before `plan_logical_resources_with_policy`** - the selected topology transition consumes the final fused graph, while residency and uniqueness resolution use its final liveness and demands.

Every dependency above is enforced by the top-level typestate chain. Internal
sub-passes within a transition are ordered by that transition's body rather
than by additional public typestates.

### Staged IR

`plan_logical_resources_with_policy` begins with semantic-EGIR allocation and
residency, then `finalize_staged_ir` performs the representation change. (The
policy-free wrapper selects `AllowGenerated`.) A finalized
`StagedIr` owns every executable EGIR body exactly once, distinguishes
resource-backed host inputs from resident compiler flows, gives each resident
flow one producer and at least one stage consumer or published output, and
guarantees that the stage graph is acyclic. Its topology is private; subsequent
planning may mutate stage bodies but cannot directly create an invalid graph.

The current `finalize_staged_ir` function groups three responsibilities,
implemented by multiple traversals: it completes consumer edges for stages
introduced after a flow, records the host-facing input/output boundary, and
checks resource references and output routes while finalizing the builder and
moving executable bodies into their stages. Inconsistencies return from this
construction boundary; there is no later resource-verification pass.

| Checkpoint orchestrator | Sub-pass | Role / condition |
|-------------------------|----------|------------------|
| **`plan_logical_resources`** | `finalize_staged_ir` | Check resource references and output routes, complete resident-flow incidence, publish only host-origin external inputs and outputs, validate destinations and acyclicity, and replace the semantic program's flat entry collection with staged body ownership |
| **`plan`** | `bind_mapped_output_destinations` | Bind mapped stage outputs to the resource destinations selected during logical planning |
| **`plan`** | `planning::analyze` | Analyze target-aware physical recipes for every staged body |
| **`plan`** | `allocate_scratch` | Parallel schedule only: allocate work buffers required by selected recipes |
| **`plan`** | `serial_plan` | Serial schedule only: select serial recipes without parallel scratch allocation |
| **`plan`** | `build_parallel_schedule` | Parallel schedule only: build dispatches and generated callables, using resident flows as scheduling dependencies |
| **`plan`** | `build_serial_schedule` | Serial schedule only: build the serial schedule and generated callables, using resident flows as scheduling dependencies |
| **`plan`** | `install_generated_callables` | Add scheduler-generated callables and their identities to the program |
| **`plan`** | `KernelPlan::validate` | Debug builds only: validate kernel identities, placements, and dependency acyclicity before physical publication |
| **`plan`** | `KernelPlan::check_explicit_dispatch_coverage` | Validate that every explicit dispatch extent is covered before publishing the physical layout |
| **`plan`** | `publish_physical_layout` | Allocate physical bindings and derive descriptor stages, dispatches, binding uses, graphics I/O, and the frame graph from the kernel plan |
| **`plan`** | `PhysicalKernelGraph::from` | Freeze stable kernel identities and dependency/resource topology without copying physical bodies |
| **`plan`** | `physicalize_program` | Convert scheduled semantic bodies and resources to `EGraph<Physical>` bodies, establish one-to-one kernel/body ownership, and construct entry parameters directly on their value, resource-view, or read-only-place ABI channels |
| **`plan`** | `physicalize_function_boundary` | Consume each transitional physical function, flatten callable products, lower aggregate/view inputs to read-only places, append aggregate-result destinations, and rewrite every return before producing a stable callable ABI |
| **`plan`** | `reconcile_program_calls` | After every internal boundary is stable, reconcile calls in functions, entries, and constants through the canonical call-binding API; extern declarations retain their explicit ABI |
| **`plan`** | `verify_physical::check` | Validate physical types and irreducible cross-arena links: parameter bindings, returns, calls, and SOAC operand metadata |

The staged order is load-bearing:

- **Filter storage construction before `finalize_staged_ir`** — direct
  publication and residency materialization derive and check capacity sizes
  when they bind backing resources, so every resident flow is complete before
  the graph is sealed.
- **`finalize_staged_ir` before `plan`** — target planning consumes stored stage
  and flow topology; it no longer reconstructs compiler-resource flow edges
  from entry declarations.
- **`planning::analyze` before schedule construction before
  `KernelPlan::finalize`** — scheduling consumes selected recipes and finalized
  publication consumes the validated schedule.

### Physical EGIR

`KernelPlan::finalize` exits staged IR by freezing a `PhysicalKernelGraph` and
physicalizing its bodies. Every later checkpoint is the same
`PhysicalProgram<Tag>` family: the typestate changes body invariants while the
kernel DAG, identities, and body associations remain intact. Functions and
constants remain ordinary auxiliary body arenas rather than kernel nodes.

| Checkpoint orchestrator | Sub-pass | Role / condition |
|-------------------------|----------|------------------|
| **`expand_soacs`** | `expand_one` to fixpoint | Expand each selected physical SOAC recipe into explicit loop or kernel operations |
| **`expand_soacs`** | `verify_physical::check_graph_flow` | Validate each expanded body's physical control and data flow |
| **`eliminate_internal_place_calls`** | `eliminate_body` to fixpoint, `verify_ssa_lowerable_calls` | Inline every internal call whose arguments or results use place channels, then prove that all remaining calls can use the SSA value channel |
| **`partially_inline_calls`** | `inline_body` to bounded fixpoint | Inline profitable mixed-variance calls inside explicit loops so invariant subgraphs can hoist |
| **`materialize_dynamic_extracts`** | `materialize::run_one_body` | Normalize place-backed stores and consumers, then materialize dynamic aggregate extraction where the SSA boundary requires explicit control and data flow |
| **`rewrite`** | `rewrite` | Add cost-arbitrated equivalent e-graph alternatives, such as multiply chains for constant powers |
| **`optimize_skeleton`** | `fold_constant_branches`, `remove_unreachable_blocks`, `eliminate_redundant_params` to fixpoint | Fold branches, prune newly unreachable CFG blocks, and eliminate redundant block parameters in the effect skeleton |
| **`erase_resources`** | `erase_resources` | Replace compile-time resource handles with their physical storage representation |
| **`elaborate`** | `elaborate` | Demand-elaborate physical e-graphs into backend-bound SSA, naturally applying DCE, scoped CSE, and LICM |

The physical order is also enforced by typestate:

- **`plan` before `expand_soacs` before `eliminate_internal_place_calls` before
  `partially_inline_calls` before `materialize_dynamic_extracts` before
  `rewrite` before `optimize_skeleton` before `erase_resources` before
  `elaborate`** — every transition consumes the checkpoint produced by the
  preceding transition, and expansion accepts only a validated kernel plan.

### SSA (codegen only)

| Checkpoint transition | Sub-pass sequence | Description |
|-----------------------|-------------------|-------------|
| Physical EGIR `ResourcesErased` -> SSA `Elaborated` | `elaborate` | Demand-elaborate the validated physical program to SSA while retaining its published schedule and descriptor |
| `Elaborated` -> `Reachable` | `filter_reachable` | Remove final SSA functions and constants not reachable from an entry point |
| `Reachable` -> `SpirvReady` | `eliminate_dead_values`, `verify_no_abstract_types`, `verify_buffer_layouts` | Remove dead SSA values, validate abstract types and buffer layouts, and record SPIR-V readiness in the typestate |
| `Reachable` -> `WgslReady` | `promote_addressable_constants`, `eliminate_dead_values`, `verify_no_abstract_types` | Move constants whose addresses are taken into functions, remove dead SSA values, validate abstract types, and record WGSL readiness in the typestate |
| Backend-ready -> emitted module | `lower_ssa_program` \| `wgsl::lower` | Emit SPIR-V words or WGSL source from the selected backend-ready SSA checkpoint |

SSA is intentionally minimal: optimization and canonicalization live in EGIR;
SSA performs only final definition reachability and target validation. A
generic CFG-with-block-params representation is provided in
`ssa::framework`; the concrete instantiation lives in `ssa::types`.

Key properties:
- CFG with basic blocks and block parameters (not phi nodes).
- No effect tokens at the SSA layer — instruction order is fixed by
  elaboration.
- Function parameters and block parameters are distinct kinds of
  values.

### SOAC Parallelization Boundary

Parallel semantics begin in semantic EGIR. TLC performs source-level
normalization and uniqueness reasoning but emits no per-entry strategy record.
Semantic EGIR reifies every reachable SOAC; staged IR retains those semantic
EGIR bodies while making cross-stage residency explicit. `plan` selects recipes
and constructs scheduled physical EGIR entries, and `expand_soacs` replaces the
selected SOACs with explicit physical operations before SSA elaboration. The
published physical program carries its kernels, dependency/resource schedule,
and descriptor together. Its initial portable scheduler implements:


- **Map** — lane-indexed scalar kernel: one thread per element, guarded
  by a bounds check. The serial-loop builder is still used for
  non-entry maps (intermediate `map` inside a function body) which
  legitimately want sequential execution per thread.
- **Reduce** — two-phase: phase 1 chunks the input per thread and
  writes a partial; phase 2 is a synthesized combine entry over the
  partials.
- **Redomap** — same two-phase shape as Reduce, with the redomap's
  combiner in phase 2.
- **Scan** — three-phase Blelloch-style: phase 1 chunks the scan
  per-thread + writes per-block sums; phase 2 sequentially scans the
  per-block sums into per-block offsets; phase 3 applies the
  per-element fold. When the input is marked consumable (`*[]T`),
  phases 1 and 3 write back in place and the pipeline descriptor
  skips the auto-output slot.

### SOAC Implementation Status

TLC has six `SoacOp` variants; the table also lists Redomap, which is a fused
semantic EGIR form rather than a TLC constructor. "Serial" = correct sequential
lowering through `soac_expand`. "Consuming-input DPS" = TLC records a uniqueness
candidate and EGIR residency verifies post-fusion death before choosing in-place
reuse instead of a fresh output buffer. "Parallel" = EGIR-side parallelization
fires on a compute-entry SOAC matching the strategy's shape.

| SOAC               | Surface syntax                          | Serial | Consuming-input DPS | Parallel  |
|--------------------|-----------------------------------------|--------|---------------------|-----------|
| `Map`              | `map f xs`                              | ✓      | ✓                   | ✓ (lane-indexed) |
| `Reduce`           | `reduce op ne xs`                       | ✓      | n/a (scalar result) | ✓ (chunked + combine) |
| `Redomap`          | `reduce op ne (map f xs)` (fused)       | ✓      | n/a (scalar result) | ✓ (chunked + combine) |
| `Scan`             | `scan op ne xs`                         | ✓      | ✓                   | ✓ (3-phase Blelloch-style) |
| `Filter`           | `filter pred xs`                        | ✓ (static **and** runtime-sized) | ✓      | ✓ for escaping entry outputs (flags + scan + scatter); non-escaping consumers fuse semantically |
| `Scatter`          | `scatter(dest, indices, values)`        | ✓ (sequential per-lane indexed store; envelope `(xs..) -> (index, value)` lets the fusion engine fuse map producers into the scatter) | ✓ (writes in place into the bound storage view) | ✗ |
| `ReduceByIndex`    | histogram-style indexed reduction       | ✗ EGIR `convert_soac` rejects with `Unsupported` | n/a | ✗ (atomics not yet implemented) |

Notes:
- TLC ownership emits only `UniqueInput`. After fusion,
  `egir::plan_logical_resources` promotes it to `InputBuffer`
  only when the final semantic graph proves the compatible input dead;
  otherwise it becomes `Fresh` (or output routing has already selected an
  `OutputView`). `soac_expand` turns an accepted in-place destination into
  `array_with_inplace` operations.
- Phase 3 of parallel scan applies `op(off, elem)`, not `op(elem, off)`:
  `egir::parallelize` synthesizes a swap-args wrapper EgirFunc
  `\(a, b) -> op(b, a)` alongside the phase entries, and phase 3's Map
  routes through the wrapper. Correct for non-commutative associative
  combiners (string concat, matmul).

#### `Filter` — runtime-sized inputs and parallel lowering

`filter` is shape-changing: it returns the existential `?k. [k]T`, opened
to a runtime length `k ≤ n` at the consumer. Two lowerings, by input size:

- **Static input** (`[N]T`, capacity known): a function-local **Bounded**
  `{buffer:[N]T, len:i32}` struct — `soac_expand::build_filter_loop`.
- **Runtime input** (a storage view / entry param, length only known at
  dispatch): the serial loop compacts kept elements into a reserved scratch
  **storage** buffer (capacity `n`, host-sized `LikeInput` of the input) and
  yields a runtime-length **view** over it (`StorageView(scratch)[0, count]`).
  The surviving count is the view's `len` *operand* — a value, not a
  type-level size — so `length` and `reduce` consume it like any view. The
  scratch binding is represented in EGIR's logical resource manifest and
  published by terminal scheduling alongside gather/shared-array resources.
  A runtime `filter` reached in a *standalone* function (one
  inlining didn't fold into a compute entry) errors — only an entry owns a
  descriptor set to host the scratch buffer; `from_tlc::convert_function`
  guards this.
- **`filter` as a compute output**: conversion gives the route a host output
  binding and `LikeInput` capacity policy. Reification records that publication
  uniformly as an output slot plus resource write; logical resource planning
  then binds the filter to that output and allocates the paired **`u32` length
  cell** (`Fixed{4}`) read back by the host.

**Parallelization status.** Semantic EGIR folds non-escaping filters into
masked reduction steps; `length` becomes one shared count reduction, so no
compacted intermediate is required. SOAC helpers are force-inlined before EGIR
conversion, making the same rewrite available when source code factors the
filter and consumer into different functions.

An escaping `SegFilter` uses the parallel map → flag scan → scatter algorithm
(`if flags[i] { out[offsets[i]-1] = xs[i] }`) and publishes the final count
through the paired length cell. Ordered general-purpose `SegHist`/scatter
lowering remains serial; the filter scheduler owns its guarded compaction
scatter as part of the specialized filter phase model.

#### Remaining-work ordering

The unimplemented cells above have a few hard dependencies between
them, plus some softer reuse opportunities. Anything not on this list
is independent.

- **`Scatter` serial is in place** (`(xs..) -> (index, value)` envelope
  with map-producer fusion, lowered to a sequential indexed-store loop
  against the bound storage view). The OOB-guarded indexed-store builder
  it produces is the natural starting point for `ReduceByIndex` serial,
  which adds a read-combine-write step on top.
- **Surface parsing.** `reduce_by_index` already has a producer at
  `tlc::mod::transform_soac_reduce_by_index`; `scatter` is parsed as an
  ordinary function call (no dedicated `SoacOp` surface form — the
  envelope lambda is what marks it as a SOAC).
- **Parallel `ReduceByIndex` → atomic intrinsics.** The catalog has
  no `atomicAdd`/`atomicMin`/etc. today; adding them is a
  prerequisite for parallel histograms. Serial ReduceByIndex doesn't
  need them.
- **Parallel `Scatter` → no hard prerequisite**, but the duplicate-
  index semantics ("last write wins" sequentially) become racy in
  parallel. Either accept the race (matches Futhark's documented
  behavior) or gate on atomic-store availability.

### View Buffer Provenance

A view array (`[]T`) is a window into a storage buffer: a runtime
`{offset, len}` pair plus a **static** descriptor `(set, binding)` — Vulkan
can't pick a descriptor by a runtime value, so the binding *must* be a
compile-time constant at every consumer. Wyn makes the binding a property
of the type: the `Array` type's trailing **buffer/region** slot (always the
last type argument) holds `Buffer(set, binding)`. Thus a rank-one view has
four type arguments, `[elem, ArrayVariantView, dim_0, buffer]`; the runtime
`{offset, len}` value is separate from this static type-level buffer slot.

- **Born at entry params.** `pin_entry_buffers` (the first TLC sub-pass)
  computes each storage entry-param's binding (auto-allocated `set 0,
  0..N`, or an explicit `#[storage(set, binding)]`) and substitutes the
  param's buffer *variable* → `Buffer(set, binding)` throughout the entry.
- **Flows by unification.** A view is buffer-polymorphic everywhere else
  (`∀b. View[…, b]`), so a slice, a `let`, a function argument, or a SOAC
  capture inherits its buffer the same way it inherits its element type —
  no side-channel, no manual threading. `if c then xs else ys` over two
  different buffers fails to unify, which is the correct "can't pick a
  descriptor at runtime" error.
- **Specialized by monomorphize.** Because the buffer is an ordinary type
  parameter, `monomorphize` specializes a view function per buffer exactly
  as it does per element type — `f(xs)` and `f(ys)` over two buffers become
  two monomorphs. (This is what let the dedicated `buffer_specialize` pass
  be deleted.)
- **Read by the backends from the type.** A view is a runtime value (SPIR-V:
  a `{offset,len}` struct; WGSL: a `vec2<u32>`). `ViewIndex` recovers the
  backing buffer via `array_view_buffer(value_type)` →
  `get_or_assign_buffer_id` (SPIR-V) / `storage_name` (WGSL). No
  `ValueId → binding` side-map. The lone exception is workgroup-shared
  views, whose `_wg_<id>` isn't a descriptor and rides a small side map.

### Defunctionalization

The `defunctionalize()` typestate transition composes three sequential
sub-passes.
No analysis sidecar crosses their boundaries: closure environments live directly
on closure terms and SOAC bodies.

1. **`closure_convert::convert_closures`** - lifts every standalone lambda to a
   top-level definition, turns free variables into trailing parameters, removes
   let-bound callable aliases, and pulls in transitive captures.

2. **`hof_specialize::specialize_higher_order_functions`** - clones each
   user-defined higher-order function for every concrete callable that reaches
   it, eliminating function-typed parameters.

3. **`lower_calls::lower_closure_calls`** - performs the final global walk that
   threads captures into direct call sites. It is idempotent with the
   pre-threading performed while specialized bodies are cloned.

Lambdas embedded in SOAC operators are lifted the same way, but the
SOAC keeps a `(lambda, captures)` payload so the loop body and its
closed-over values stay together for later expansion.

Three verifier-checked invariants guard the phase boundaries: no
standalone lambdas survive outside SOAC envelopes, no reachable def
has a function-typed parameter, and every call is fully arity-matched
to a non-callable target.

### Type Schema

Throughout TLC and EGIR, types are `polytype::Type<TypeName>`. The
underlying carrier is `Type::Constructed(TypeName, Vec<Type>)` — the
`Vec<Type>` (the "args") means different things per variant. This
schema is the canonical mapping. Helpers in `wyn-core/src/types/mod.rs`
(`array_elem`, `array_size`, `array_variant`, `strip_unique`,
`extract_function_signature`) centralize the position queries so compiler
sub-passes don't pattern-match on args indices directly.

| Variant | args[0] | args[1] | args[2…] | Notes |
|---|---|---|---|---|
| `Bool`, `Float(n)`, `UInt(n)`, `Int(n)` | — | — | — | Nullary scalars |
| `Unit` | — | — | — | The `()` value |
| `SideEffect` | — | — | — | "No return value, side effects only"; renders as `!()`. Used by effect-only computations and imperative builtin signatures |
| `Arrow` | param | return | — | Curry by chaining (`a → b → c` = `Arrow(a, Arrow(b, c))`) |
| `Tuple(n)` | t₁ | t₂ | … | n elements; arity in the variant tag |
| `Vec` | elem | `Size(n)` | — | n-component vector |
| `Mat` | elem | `Size(cols)` | `Size(rows)` | Column-major |
| `Array` | elem | variant | dim_0 … dim_{rank-1}, then buffer/region | Exact layout: `[elem, variant, dim_0…dim_{rank-1}, buffer]`, so `args.len() == rank + 3`. All arrays are rank-one today and therefore have **four** arguments: `args[0]` elem, `args[1]` variant, `args[2]` dimension, and `args[3]` buffer. Do not identify arrays or views with an `args.len() == 3` check; use `array_elem`, `array_variant`, `array_size`, and `array_view_buffer`. Each dim is `Size(n)` \| `SizeVar(name)` \| `SizePlaceholder` \| `Variable`; variant is `ArrayVariantView` \| `Composite` \| `Virtual` \| `Bounded`. The trailing **buffer/region** is `Buffer(set, binding)` (a storage view), `NoBuffer` (a non-view array), or a variable (buffer-polymorphic) — making a view's buffer a static type property (see View Buffer Provenance) |
| `Pointer` | pointee | addrspace | — | addrspace is one of `PointerFunction` / `PointerInput` / `PointerOutput` / `PointerStorage` |
| `Unique` | inner | — | — | `*T` uniqueness marker (consumed by ownership) |
| `Record(fields)` | t₁ | t₂ | … | Field names in the variant payload (declared order); per-field types in args |
| `Sum(variants)` | — | — | — | Both names and per-variant payload types are in the `Sum` payload itself; args is empty |
| `Existential(vars)` | t | — | — | Bound size-var names in the variant payload; inner type in args[0] |
| `Named(s)`, `Size(n)`, `SizeVar(s)`, `UserVar(s)` | — | — | — | Nullary; data carried in the variant payload |
| `SizePlaceholder`, `AddressPlaceholder`, `ArrayVariant{View,Composite,Virtual,Bounded}`, `Pointer{Function,Input,Output,Storage}` | — | — | — | Nullary marker types used only as args of other variants |
| `Texture2D`, `Sampler`, `StorageTexture` | — | — | — | Nullary opaque GPU handles. Format/access for storage textures live on `EntryInput.storage_image_binding` (per-param), not on the language-level type |

## Example Program

```wyn
-- Render a full-screen triangle through one explicit graphics operation.

def vertex_main(vertex: vertex_invocation) vertex<vec2f32> =
  let verts = [@[-1.0, -1.0, 0.0, 1.0],
               @[ 3.0, -1.0, 0.0, 1.0],
               @[-1.0,  3.0, 0.0, 1.0]] in
  vertex_output(verts[i32(vertex.vertex_index)], @[0.0, 0.0])

entry image(screen: render_target<vec4f32>) render_target<vec4f32> =
  let raster = rasterize_triangles(direct_draw(3u32, 1u32), vertex_main) in
  shade(screen, raster,
    |fragment| @[0.529, 0.808, 0.922, 1.0])  -- Sky blue
```

## Usage

```bash
# Compile to SPIR-V
cargo run --bin wyn -- build input.wyn -o output.spv

# Compile to WGSL
cargo run --bin wyn -- build input.wyn -o output.wgsl -t wgsl

# Compile a graphics program directly, without compiler-created prepasses
cargo run --bin wyn -- build input.wyn -o output.spv --graphics --direct
cargo run --bin wyn -- build input.wyn -o output.wgsl -t wgsl --graphics --direct

# Opt in to backend-local u64 emulation for WGSL
cargo run --bin wyn -- build input.wyn -o output.wgsl -t wgsl --wgsl-emulate-u64

# Type check without generating code (`--graphics` is required for graphics vocabulary)
cargo run --bin wyn -- check input.wyn --graphics

# Output intermediate representations
cargo run --bin wyn -- build input.wyn --output-init-ssa out.ssa   # Initial SSA
cargo run --bin wyn -- build input.wyn --output-annotated out.ann  # Annotated source

# Run a compiler-published SPIR-V or WGSL pipeline
cd extra/viz && cargo run -- pipeline ../../shader.wgsl
```

Graphics vocabulary is opt-in. Without `--graphics`, names such as
`direct_draw`, `rasterize_triangles`, `shade`, `vertex_invocation`, and
`render_target` are ordinary, unreserved identifiers: user code may define
them, and otherwise receives the normal undefined-name diagnostic.
`--direct` is a backend-neutral output policy. It preserves authored graphics
stages and rejects programs that would require compiler-created prepass entry
points or intermediate storage.

## Building and Testing

```bash
cargo build --release
cargo test
```

Use `cargo test --workspace` for the full Rust suite. All SPIR-V testfiles in `testfiles/` compile and validate (`bash scripts/validate_testfiles.sh`); the WGSL subset also validates (`bash scripts/validate_testfiles.sh --wgsl` — a handful skip because they depend on linked SPIR-V helpers).

## Language Overview

### Types

- **Primitives**: `i32`, `u32`, `f32`, `bool`
- **Arrays**: `[N]T` for fixed size, `[]T` for inferred size
- **Vectors**: `vec2f32`, `vec3f32`, `vec4f32` (SPIR-V types)
- **Matrices**: `mat2f32`, `mat3f32`, `mat4f32`
- **Tuples**: `(T1, T2, ...)`
- **Functions**: `T1 -> T2`

### Key Syntax

```
-- Top-level definitions
def add(x: i32, y: i32) i32 = x + y
def first(xs: []i32) i32 = xs[0]

-- Compute entry point
entry sum_array(data: []f32) f32 =
  reduce(|a: f32, b: f32| a + b, 0.0, data)

-- Graphics uses an orchestration entry plus ordinary callbacks.
def vs_main(vertex: vertex_invocation) vertex<vec3f32> = ...
def fs_main(fragment: fragment_invocation<vec3f32>) vec4f32 = ...
entry frame(screen: render_target<vec4f32>) render_target<vec4f32> =
  let raster = rasterize_triangles(direct_draw(3u32, 1u32), vs_main) in
  shade(screen, raster, fs_main)

-- Lambdas
|x: i32| x + 1
|x: i32, y: i32| x + y

-- Loops (the supported looping primitive)
loop (acc, i) = (0, 0) while i < n do (acc + arr[i], i + 1)

-- Higher-order functions
map(|x: i32| x * 2, arr)
reduce(|a: i32, b: i32| a + b, 0, arr)
```

### Type Inference

```
def identity(x) = x
-- Inferred: ∀a. a -> a

def zip_arrays(xs, ys) = zip(xs, ys)
-- Inferred: ∀n t1 t2. [n]t1 -> [n]t2 -> [n](t1, t2)
```

## Current Limitations

- Module system covers the common path (`open`, qualified access, multi-file imports — see `testfiles/open_module_demo.wyn`); some advanced features remain unimplemented.
- `match` expressions work for literals, wildcards, and sum-type constructors (`testfiles/match_*.wyn`, `testfiles/sum_demo.wyn`); guards and nested patterns are pending.

- WGSL has no concrete 64-bit integer type. The WGSL backend rejects `u64` by
  default and can optionally emulate the BLAKE2b-oriented subset with
  `--wgsl-emulate-u64`: wrapping addition/subtraction, bitwise operations,
  shifts, comparisons, and `u32`/`i32` conversions. Multiplication, division,
  remainder, floating-point conversions, signed `i64`, and 64-bit atomics are
  not emulated.

## Design Choices

These are deliberate and not on a fix-it list:

- **No recursion.** Use `loop` or higher-order functions. Aligns with Futhark and GPU-targeted execution.

## Key Dependencies

- **nom** — Parser combinators
- **polytype** — Hindley-Milner type system
- **rspirv** — SPIR-V builder

For complete language details, see [SPECIFICATION.md](SPECIFICATION.md).
