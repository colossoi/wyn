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
- **`wyn/`** - Command-line executable
- **`wyn-analyzer/`** - Language server (in development)
- **`viz/`** - Visualization tool for rendering SPIR-V shaders
- **`prelude/`** - Standard library functions written in Wyn

## Compiler Architecture

The compiler uses a multi-stage pipeline with typestate-driven phases. Each stage consumes `self` and returns the next stage, enforcing valid ordering at compile time.

### Mid-End: Acyclic E-Graph

The mid-end is an **acyclic e-graph**: a sea of hash-consed pure nodes
plus a skeleton CFG of effectful instructions. Purity is
**blacklisted**, not whitelisted — only `Alloca` / `Load` / `Store`
stay in the skeleton; everything else (calls, intrinsics, storage
views, projections, indexing) is hash-consed in the pure sea.

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
- **Branch folding + redundant-phi elimination** — the skeleton CFG
  is rewritten before elaboration.

### Frontend (AST)

| Stage | Module | Description |
|-------|--------|-------------|
| **Parsed** | `parser` | Tokenization and parsing into AST |
| **Desugared** | `desugar` | Range/slice expressions desugared; SOAC names rewritten to intrinsics |
| **Resolved** | `name_resolution` | Name resolution and module imports |
| **AstConstFoldedEarly** | `ast_const_fold` | Compile-time integer constant folding |
| **TypeChecked** | `types::checker` | Hindley-Milner type inference and checking, plus one-directional `*T → T` weakening at coercion sites (return position, let ascription) |

### TLC (Typed Lambda Calculus)
| Stage | Module | Description |
|-------|--------|-------------|
| **TlcTransformed** | `tlc` (`TypeChecked::to_tlc`) | AST converted to minimal typed lambda calculus |
| **TlcBuffersPinned** | `tlc::pin_entry_buffers` | Each storage entry-param's concrete `Buffer(set, binding)` is substituted into its type, so a view's buffer is a statically-known type property that flows by unification. A distinct typestate, so the rest of the pipeline can't run without it (see View Buffer Provenance below) |
| **TlcPartialEvaled** | `tlc::partial_eval` | Constant folding and algebraic simplifications |
| **TlcSoaNormalized** | `tlc::soa` | SoA transform (`[n](A,B)` → `([n]A, [n]B)`) + Map+Zip flattening + standalone Zip elimination |
| **TlcMonomorphized** | `tlc::specialize`, `tlc::monomorphize` | Polymorphic intrinsics specialized; user functions monomorphized — including over a view's **buffer**, so a function called on two buffers yields two monomorphs (this subsumes the former `buffer_specialize` pass) |
| **TlcRepSpecialized** | `tlc::rep_specialize` | Phase 2 of array-variant-abstract: at call edges, clone any user-defined callee whose `Abstract`-typed param receives a producer-known concrete variant (Bounded / View from filter), and rewrite the call to invoke the clone. Runs before force-inline so SOAC helpers are representation-concrete when it inlines them |
| **TlcSmallInlined** | `tlc::inline` | Inline small user functions and constants |
| **TlcSoacHelpersInlined** | `tlc::inline::run_force_soac_helpers` | Force-inline every user function whose body (recursively) contains a SOAC (or `length`), regardless of control flow, so no SOAC is reachable behind a call and fusion is purely intraprocedural. Checked by `fusion::verify_soac_helpers_inlined` |
| **TlcProducerCanonicalized** | `tlc::soa`, `tlc::if_over_producer` | Re-run SoA normalization (inlining may have exposed new tuple/zip/map structure), then `if-over-producer` lifting, so fusion sees clean top-of-let-chain SOAC producers |
| **TlcFused** | `tlc::fusion`, `tlc::if_over_producer` | Intraprocedural SOAC fusion (horizontal map, map+reduce/scan, filter+length — "merge compatible nodes, union outputs"), then `if-over-producer` lifting and reachable-DCE. No cross-function summary path: every producer/consumer edge is within one def (force-inline guarantees it). Runs on **inline** SOAC operators (pre-defunctionalize) so `compose_*` can fold operator bodies directly |
| **TlcEntryProducersExposed** | `tlc::materialize_entry_soacs` | Inlines producer-helper calls into the entry's top-level let-chain so the next two passes can see the SOAC producer + its indexed uses in the same scope. Refuses to descend into per-element lambdas — exposing a per-element scan as an entry producer would wreck cost semantics |
| **TlcStaticIndexFused** | `tlc::static_index_fusion` | `map(f, src)[k]` (constant `k`) collapses to `f(src[k])`. A producer demanded only at a known slot becomes a scalar element computation rather than a runtime-sized buffer materialization |
| **TlcRuntimeIndexProducersFloated** | `tlc::runtime_index_producers` | `map(\i. (map(f, xs))[i], is)` floats the inner producer out into a let-binding so it looks like an ordinary gather (`let p = map(f, xs) in map(\i. p[i], is)`) for the residency pass to rewrite |
| **TlcGathersLifted** | `tlc::lift_gathers` | Plans and executes gather residency (the `plan_execute_gather_residency` transition): materializes randomly-indexed computed arrays into storage buffers by splitting the producer into its own pre-pass compute entry, then rewrites the consumer's indexed reads to load from that buffer |
| **TlcScalarPrepassesHoisted** | `tlc::parallelize` | Hoists a compute entry's scalar pre-pass computations (e.g. a reduction whose result a later phase consumes) into their own pre-pass compute entry, so the value is resident at dispatch. Skipped under `--single-stage` (the `disable` flag). Runs in the residency cluster, before defunctionalize, while producers are still recognizable as `Soac(Map/Scan)` |
| **TlcDefunctionalized** | `tlc::closure_convert` → `tlc::hof_specialize` → `tlc::closure_calls_lower` | Three sequential passes: lambdas lifted to top-level defs, higher-order functions specialized away, captures threaded into call sites — including the SOAC operators fusion composed earlier. Verifier-checked invariants guard each phase boundary (see Defunctionalization below) |
| **TlcGeneratedLambdasFolded** | `tlc::inline` | Fold compiler-generated lambda defs back at call sites + DCE |
| **TlcOwnershipApplied** | `tlc::ownership` | Backward ownership-liveness analysis. Reports use-after-move; rewrites array-update operations into in-place forms when the source is mutable and dead after the call. Runs before output normalization so its liveness walk never sees `OutputSlotStore` |
| **TlcOutputsNormalized** | `tlc::normalize_outputs` | Rewrites each compute entry's tail into a chain of explicit per-slot output writes. Single-output and multi-output entries share one structural shape; the entry's `def.ty` is kept in sync with its rewritten body |
| **TlcParallelized** | `tlc::parallelize` | Remaining source-level equal-domain map fusion plus empty host pipeline shells. No strategy or recognition facts are emitted; EGIR derives placement and scheduling from semantic operations |
| **TlcReachable** | `tlc::inline` | Dead definition elimination |

#### Pass-ordering dependency assertions

The table above is one valid topological sort of the constraints below
(`optimize_for_test` in `wyn-core/src/lib.rs` and the CLI pipeline in
`wyn/src/main.rs` must stay in sync with it). `A ≺ B` means A runs before B.
Each notes how it's enforced; when you move a pass, check it here.

- **`monomorphize` ≺ `defunctionalize`** — fusion must run post-mono
  (intraprocedural) *and* pre-defunc (so it composes **inline** SOAC operators,
  not function references), so mono runs first. Mono handles the still-higher-
  order program; defunc removes function-typed params and lifts the fused
  operators to refs afterward. *Enforced by:* convention.
- **`defunctionalize` ≺ `fold_generated_lambdas`** — fold inlines the
  `_w_lambda_*` defs that defunc generates. *Enforced by:* convention.
- **`monomorphize` ≺ `force_inline_soac_helpers`** — force-inline's free-type-var
  guard skips any helper still carrying an unresolved element-type `Variable`, so
  helpers must be concrete first. *Enforced by:* indirectly via the validator
  below.
- **`rep_specialize` ≺ `force_inline_soac_helpers`** — makes `filter`-result
  helpers representation-concrete (`Abstract` → `Bounded`/`View`) so the guard
  admits them. *Enforced by:* convention.
- **`force_inline_soac_helpers` ≺ `fuse_maps`** — every SOAC helper inlined so
  fusion sees only intra-def producer/consumer edges (no summary path).
  *Enforced by:* `fusion::verify_soac_helpers_inlined` — `debug_assert!` at the
  end of `run_force_soac_helpers` and at the top of `fusion::run`.
- **`apply_ownership` ≺ `normalize_outputs`** — ownership's liveness analysis has
  no case for `OutputSlotStore`, which `normalize_outputs` introduces. *Enforced
  by:* `unreachable!` in `ownership.rs`'s `analyze`.
- **residency cluster (`expose_entry_producer_helpers` … `hoist_scalar_prepasses`)
  ≺ `defunctionalize`** — the gather/static-index/scalar-prepass passes match on
  `Soac(Map/Scan)` producers, which only survive while operators are still
  function values; defunctionalize lifts them to refs and must run after. *Enforced
  by:* convention (see the comment in `compile_file`).

### EGIR (Acyclic E-Graph IR)
| Stage | Module | Description |
|-------|--------|-------------|
| **EgirRaw** | `egir::from_tlc` | TLC → EGraph; intrinsic calls become pure nodes (with explicit arms for effectful ones). Per-slot output writes are bridged back into a tail tuple so the next stage can retarget per slot |
| **EgirOutputsRealized** | `egir::realize_outputs` | Per-slot output realization: each declared output's writes are materialized as side effects against the bound storage view (compute) or `OutputSlot` place (graphics); the body's `Return` carries no value. The post-pass verifier checks no runtime-sized Composite array is reachable from any entry output |
| **EgirSegmented** | `egir::parallelize::reify` | Every reachable Screma becomes a semantic SegMap/SegRed/SegScan with authoritative SegSpace, typed bodies, explicit captures, output routing, effects, placement, and dependencies. No phases are selected here |
| **EgirOptimized** | `egir::semantic_opt`, `egir::fusion` | Conflict-aware same-space sibling fusion, single-consumer producer/consumer region composition, and dead-SegOp elimination; SegOps remain semantic |
| **EgirAllocated** | `egir::program`, `egir::multi_consumer` | Owns the authoritative host/compiler resource manifest. Scalar handoffs, reduce/scan/filter scratch, and shared multi-consumer array materializations have `ResourceId`s; physical publication still waits for terminal lowering |
| **SsaConverted** | `egir::target_lowering` | `lower_to_ssa(LoweringProfile)` transactionally chooses algorithms, scratch, bindings, domains, KernelPlan, and the final descriptor, then expands SegOps to SSA. `egir::resource_erasure` then drops compile-time-only resource handles (buffer-monomorphized storage-image params/operands) so no opaque image is threaded as an SSA value |

### SSA (codegen only)
| Stage | Source | Description |
|-------|--------|-------------|
| **SsaConverted** | `EgirAllocated::lower_to_ssa` | Target-aware terminal EGIR lowering produces SSA, schedule, and descriptor together |
| **Lowered** | `spirv` / `wgsl` | SSA to SPIR-V or WGSL |

SSA is intentionally minimal: all mid-end machinery (effect tokens,
canonicalization, verification, generic transform passes) lives in
EGIR. A generic CFG-with-block-params representation is provided in
`ssa::framework`; the concrete instantiation lives in `ssa::types`.

Key properties:
- CFG with basic blocks and block parameters (not phi nodes).
- No effect tokens at the SSA layer — instruction order is fixed by
  elaboration.
- Function parameters and block parameters are distinct kinds of
  values.

### SOAC Parallelization Boundary

Parallel semantics live in EGIR. TLC performs source-level normalization and
fusion but emits no per-entry strategy record. EGIR reifies every reachable
SOAC, retains it through semantic optimization and logical allocation, and
destroys it only at the target-aware lower-to-SSA boundary. That terminal
operation produces SSA kernels, the dependency/resource schedule, and the
descriptor as one result. Its initial portable scheduler implements:


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

The seven SOAC variants (`SoacOp` in `tlc/mod.rs`) at varying stages of
the pipeline. "Serial" = correct sequential lowering through
`soac_expand`. "Consuming-input DPS" = the ownership pass marks a
unique-and-dead input and the SOAC rewrites to write back in place
instead of allocating a fresh output buffer. "Parallel" = EGIR-side
parallelization fires on a compute-entry tail SOAC matching the
strategy's shape.

| SOAC               | Surface syntax                          | Serial | Consuming-input DPS | Parallel  |
|--------------------|-----------------------------------------|--------|---------------------|-----------|
| `Map`              | `map f xs`                              | ✓      | ✓                   | ✓ (lane-indexed) |
| `Reduce`           | `reduce op ne xs`                       | ✓      | n/a (scalar result) | ✓ (chunked + combine) |
| `Redomap`          | `reduce op ne (map f xs)` (fused)       | ✓      | n/a (scalar result) | ✓ (chunked + combine) |
| `Scan`             | `scan op ne xs`                         | ✓      | ✓                   | ✓ (3-phase Blelloch-style) |
| `Filter`           | `filter pred xs`                        | ✓ (static **and** runtime-sized) | ✓      | partial — `reduce(filter)` fuses to a parallel redomap; standalone `filter` is serial (see below) |
| `Scatter`          | `scatter(dest, indices, values)`        | ✓ (sequential per-lane indexed store; envelope `(xs..) -> (index, value)` lets the fusion engine fuse map producers into the scatter) | ✓ (writes in place into the bound storage view) | ✗ |
| `ReduceByIndex`    | histogram-style indexed reduction       | ✗ EGIR `convert_soac` rejects with `Unsupported` | n/a | ✗ (atomics not yet implemented) |

Notes:
- `Scan` consuming-input DPS is wired through Path B
  (`egir::parallelize::transform_scan_entry` reroutes phase 1 + phase 3
  writes back to the input binding when destination is `InputBuffer`).
  A view's backing buffer flows through loop block params and
  `array_with_inplace` as part of its **type** (the `Buffer(set, binding)`
  in its type's buffer slot), so `ViewIndex` recovers the storage buffer
  from `array_view_buffer(value_type)` — see View Buffer Provenance below.
- Phase 3 of parallel scan applies `op(off, elem)`, not `op(elem, off)`:
  `egir::parallelize` synthesizes a swap-args wrapper EgirFunc
  `\(a, b) -> op(b, a)` alongside the phase entries, and phase 3's Map
  routes through the wrapper. Correct for non-commutative associative
  combiners (string concat, matmul).

#### `Filter` — runtime-sized inputs and the parallelization gap

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
  scratch binding is reserved EGIR-locally (the converter's binding cursor +
  `extra_storage_bindings`, surfaced by `egir::publish`, like a `lift_gathers`
  gather buffer). A runtime `filter` reached in a *standalone* function (one
  inlining didn't fold into a compute entry) errors — only an entry owns a
  descriptor set to host the scratch buffer; `from_tlc::convert_function`
  guards this.
- **`filter` as a compute output**: `realize_outputs::retarget_filter_output`
  makes the filter compact directly into the entry's output buffer and writes
  the surviving count to a **paired `u32` length cell** (`Fixed{4}`,
  repurposed from the scratch binding) the host reads back alongside the data.

**Parallelization status.** `reduce(op, ne, filter(p, xs))` fuses in
`tlc::fusion` into a **masked redomap** `redomap(op∘mask, op, ne, xs)` with
`mask = λx. if p(x) then x else ne` (valid because `ne` is `op`'s neutral
element), so it parallelizes as an ordinary two-phase redomap — no compacted
intermediate. This fires across function boundaries via `array_semantics`
function summaries (`reduce(+, 0, evens(xs))` where `def evens = filter(…)`).

**GAP:** a *standalone* parallel `filter` (returning the compacted array —
`filter→output`/`filter→length` at an entry tail) is **not implemented**; it
runs through the serial scratch-view loop above. The parallel algorithm is
map (predicate → flags) → parallel scan (flags → offsets) → scatter
(`if flags[i] { out[offsets[i]-1] = xs[i] }`) → len (`offsets[n-1]`). Two
pieces are missing: a target-lowering `SegFilter` schedule reusing the
  parallel scan for the offsets prefix-sum,
and a guarded parallel **scatter** kernel — `EntryBuilder` only emits
straight-line SOAC phase bodies, so the conditional store needs hand-built CFG
(the existing ordered `SegHist`/scatter lowering is serial). Until then,
`reduce(filter)` is the only parallel filter path.

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
- **Parallel `Filter` → parallel `Scan` + a scatter kernel.** The scan
  prerequisite (prefix-sum over the predicate mask → write offsets) is in
  place, but the `SegFilter` schedule and the guarded scatter
  kernel are not built (see "`Filter` — runtime-sized inputs" above). Today
  `reduce(filter)` parallelizes via the masked-redomap fusion; standalone
  `filter` is serial.
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
of the type: the `Array` type's trailing **buffer** slot holds
`Buffer(set, binding)`.

- **Born at entry params.** `pin_entry_buffers` (the first TLC pass)
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
passes that share a closure-info side table. Every callable position
in the IR resolves through a single lookup that says either "direct
call" or "closure (code address + captures)".

1. **`closure_convert`** — lifts every standalone lambda to a top-level
   def. Free variables become trailing parameters on the lifted def.
   Let-bound lambdas are substituted away so no callable aliases
   survive. Transitive captures are pulled in.

2. **`hof_specialize`** — clones each user-defined higher-order
   function for every concrete callable that flows in, eliminating
   function-typed parameters. After this pass, every reachable
   top-level def has zero function-typed parameters.

3. **`closure_calls_lower`** — final global tree walk that threads
   captures into call sites. Idempotent with the pre-threading
   `hof_specialize` already did inside cloned bodies.

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
`extract_function_signature`) centralize the position queries so passes
don't pattern-match on args indices directly.

| Variant | args[0] | args[1] | args[2] | Notes |
|---|---|---|---|---|
| `Bool`, `Float(n)`, `UInt(n)`, `Int(n)` | — | — | — | Nullary scalars |
| `Unit` | — | — | — | The `()` value |
| `SideEffect` | — | — | — | "No return value, side effects only"; renders as `!()`. Used by post-`normalize_outputs` compute entry bodies and imperative builtin signatures |
| `Arrow` | param | return | — | Curry by chaining (`a → b → c` = `Arrow(a, Arrow(b, c))`) |
| `Tuple(n)` | t₁ | t₂ | … | n elements; arity in the variant tag |
| `Vec` | elem | `Size(n)` | — | n-component vector |
| `Mat` | elem | `Size(cols)` | `Size(rows)` | Column-major |
| `Array` | elem | variant | dim_0 (… dim_{rank-1}), buffer | Layout `[elem, variant, dim_0…dim_{rank-1}, buffer]`; rank is implicit (`args.len() - 3`), all arrays rank-1 today. Each dim is `Size(n)` \| `SizeVar(name)` \| `SizePlaceholder` \| `Variable`; variant is `ArrayVariantView` \| `Composite` \| `Virtual` \| `Bounded`. The trailing **buffer** is `Buffer(set, binding)` (a storage view), `NoBuffer` (a non-view array), or a variable (buffer-polymorphic) — making a view's buffer a static type property (see View Buffer Provenance) |
| `Pointer` | pointee | addrspace | — | addrspace is one of `PointerFunction` / `PointerInput` / `PointerOutput` / `PointerStorage` |
| `Unique` | inner | — | — | `*T` uniqueness marker (consumed by ownership) |
| `Record(fields)` | t₁ | t₂ | … | Field names in the variant payload (declared order); per-field types in args |
| `Sum(variants)` | — | — | — | Both names and per-variant payload types are in the `Sum` payload itself; args is empty |
| `Existential(vars)` | t | — | — | Bound size-var names in the variant payload; inner type in args[0] |
| `Named(s)`, `Size(n)`, `SizeVar(s)`, `UserVar(s)` | — | — | — | Nullary; data carried in the variant payload |
| `SizePlaceholder`, `AddressPlaceholder`, `ArrayVariant{View,Composite,Virtual,Bounded}`, `Pointer{Function,Input,Output,Storage}` | — | — | — | Nullary marker types used only as args of other variants |
| `Texture2D`, `Sampler`, `StorageTexture` | — | — | — | Nullary opaque GPU handles. Format/access for storage textures live on `EntryInput.storage_image_binding` (per-param), not on the language-level type |

## Example Program

```
-- Render a full-screen triangle.

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32) #[builtin(position)] vec4f32 =
  let verts = [@[-1.0, -1.0, 0.0, 1.0],
               @[ 3.0, -1.0, 0.0, 1.0],
               @[-1.0,  3.0, 0.0, 1.0]] in
  verts[vid]

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[target(screen)] vec4f32 =
  @[0.529, 0.808, 0.922, 1.0]  -- Sky blue
```

## Usage

```bash
# Compile to SPIR-V
cargo run --bin wyn -- compile input.wyn -o output.spv

# Compile to WGSL
cargo run --bin wyn -- compile input.wyn -o output.wgsl -t wgsl

# Type check without generating code
cargo run --bin wyn -- check input.wyn

# Output intermediate representations
cargo run --bin wyn -- compile input.wyn --output-init-ssa out.ssa   # Initial SSA
cargo run --bin wyn -- compile input.wyn --output-annotated out.ann  # Annotated source

# Visualize a SPIR-V shader
cd viz && cargo run vf ../shader.spv --vertex vertex_main --fragment fragment_main
```

## Building and Testing

```bash
cargo build --release
cargo test
```

1153 tests currently pass (9 ignored for pending features). All SPIR-V testfiles in `testfiles/` compile and validate (`bash scripts/validate_testfiles.sh`); the WGSL subset also validates (`bash scripts/validate_testfiles.sh --wgsl` — a handful skip because they depend on linked SPIR-V helpers).

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

-- Shader entry points (one entry-point keyword per stage)
#[vertex]
entry vs_main(#[builtin(vertex_index)] id: i32) #[builtin(position)] vec4f32 = ...

#[fragment]
entry fs_main(#[varying(0)] color: vec3f32) #[target(screen)] vec4f32 = ...

#[compute]
entry sum_array(#[size_hint(1024)] data: []f32) f32 =
  reduce(|a: f32, b: f32| a + b, 0.0, data)

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

## Design Choices

These are deliberate and not on a fix-it list:

- **No recursion.** Use `loop` or higher-order functions. Aligns with Futhark and GPU-targeted execution.

## Key Dependencies

- **nom** — Parser combinators
- **polytype** — Hindley-Milner type system
- **rspirv** — SPIR-V builder

For complete language details, see [SPECIFICATION.md](SPECIFICATION.md).
