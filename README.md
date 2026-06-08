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
| **TlcRegionsPinned** | `tlc::pin_entry_regions` | Each storage entry-param's concrete `Region(set, binding)` is substituted into its type, so a view's buffer is a statically-known type property that flows by unification. A distinct typestate, so the rest of the pipeline can't run without it (see View Buffer Provenance below) |
| **TlcPartialEvaled** | `tlc::partial_eval` | Constant folding and algebraic simplifications |
| **TlcSoaNormalized** | `tlc::soa` | SoA transform (`[n](A,B)` → `([n]A, [n]B)`) + Map+Zip flattening + standalone Zip elimination |
| **TlcFused** | `tlc::fusion` | SOAC fusion: map-map, interprocedural producer-consumer |
| **TlcOwnershipApplied** | `tlc::ownership` | Backward ownership-liveness analysis. Reports use-after-move; rewrites array-update operations into in-place forms when the source is mutable and dead after the call |
| **TlcOutputsNormalized** | `tlc::normalize_outputs` | Rewrites each compute entry's tail into a chain of explicit per-slot output writes. Single-output and multi-output entries share one structural shape; the entry's `def.ty` is kept in sync with its rewritten body |
| **TlcGathersLifted** | `tlc::lift_gathers` | Materializes randomly-indexed computed arrays into storage buffers by splitting the producer into its own pre-pass compute entry and rewriting indexed reads in the consumer to load from that buffer |
| **TlcDefunctionalized** | `tlc::closure_convert` → `tlc::hof_specialize` → `tlc::closure_calls_lower` | Three sequential passes: lambdas lifted to top-level defs, higher-order functions specialized away, captures threaded into call sites. Verifier-checked invariants guard each phase boundary (see Defunctionalization below) |
| **TlcMonomorphized** | `tlc::specialize`, `tlc::monomorphize` | Polymorphic intrinsics specialized; user functions monomorphized — including over a view's **region**, so a function called on two buffers yields two monomorphs (this subsumes the former `buffer_specialize` pass) |
| **TlcGeneratedLambdasFolded** | `tlc::inline` | Fold compiler-generated lambda defs back at call sites + DCE |
| **TlcSmallInlined** | `tlc::inline` | Inline small user functions and constants |
| **TlcParallelized** | `tlc::parallelize` | Per-entry SOAC parallelization analysis: pick strategy + workgroup + dispatch shape, reserve intermediate bindings, build the host pipeline descriptor, and emit a declarative parallelization plan per entry for EGIR to consume. Kernel lowering happens EGIR-side |
| **TlcReachable** | `tlc::inline` | Dead definition elimination |

### EGIR (Acyclic E-Graph IR)
| Stage | Module | Description |
|-------|--------|-------------|
| **EgirRaw** | `egir::from_tlc` | TLC → EGraph; intrinsic calls become pure nodes (with explicit arms for effectful ones). Per-slot output writes are bridged back into a tail tuple so the next stage can retarget per slot |
| **EgirOutputsAssigned** | `egir::assign_outputs` | Per-slot output retargeting: each tail-tuple slot is routed to its allocated output binding (storage view for compute, IO variable for graphics) |
| **EgirParallelized** | `egir::parallelize` | Consumes the parallelization plan from TLC and tags each planned compute entry's tail SOAC for parallel expansion. No-op when no entries are parallelized |
| **EgirSoacExpanded** | `egir::soac_expand` | Every pending SOAC is rewritten into an explicit loop subgraph. Small statically-sized maps are unrolled. Compute-entry SOACs lower to lane-indexed kernels that read the global invocation id |
| **EgirMaterialized** | `egir::materialize` | (optional, SPIR-V only) Dynamic `Index` operations are materialized into pointer-based reads and LICM-hoisted out of loops |
| **EgirSkelOptimized** | `egir::skel_opt` | Skeleton CFG rewrites: branch folding, redundant-phi elimination |

### SSA (codegen only)
| Stage | Source | Description |
|-------|--------|-------------|
| **SsaConverted** | `EgirSkelOptimized::elaborate` | EGIR chain elaborated into SSA `Program` |
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

Parallelization of compute-entry SOACs splits cleanly across the two
mid-end passes. **TLC `parallelize` is analysis-only**: it picks a
strategy + workgroup + dispatch shape per entry, reserves intermediate
bindings, and emits a declarative plan plus the host-facing pipeline
descriptor. It does not hand-roll kernel bodies — EGIR already builds
those well.

**EGIR `parallelize` consumes the plans** and rewrites entries per
strategy:

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
| `Filter`           | `filter pred xs`                        | ✓      | ✓                   | ✗ (no parallel impl) |
| `Scatter`          | `scatter dest indices values`           | ✗ EGIR `convert_soac` rejects with `Unsupported` | n/a | ✗ |
| `ReduceByIndex`    | histogram-style indexed reduction       | ✗ EGIR `convert_soac` rejects with `Unsupported` | n/a | ✗ (atomics not yet implemented) |

Notes:
- `Scan` consuming-input DPS is wired through Path B
  (`egir::parallelize::transform_scan_entry` reroutes phase 1 + phase 3
  writes back to the input binding when destination is `InputBuffer`).
  A view's backing buffer flows through loop block params and
  `array_with_inplace` as part of its **type** (the `Region(set, binding)`
  in its type's region slot), so `ViewIndex` recovers the storage buffer
  from `array_view_region(value_type)` — see View Buffer Provenance below.
- Phase 3 of parallel scan applies `op(off, elem)`, not `op(elem, off)`:
  `egir::parallelize` synthesizes a swap-args wrapper EgirFunc
  `\(a, b) -> op(b, a)` alongside the phase entries, and phase 3's Map
  routes through the wrapper. Correct for non-commutative associative
  combiners (string concat, matmul).

#### Remaining-work ordering

The unimplemented cells above have a few hard dependencies between
them, plus some softer reuse opportunities. Anything not on this list
is independent.

- **`Scatter` serial → `ReduceByIndex` serial.** Both write
  `dest[indices[k]] = …` per iteration; ReduceByIndex adds a
  read-combine-write step. They share an OOB-guarded indexed-write
  loop builder, so doing Scatter first lets ReduceByIndex reuse it
  rather than mirror the shape. Both also share a design choice on
  whether `dest: Place` accepts `BufferSlice` (entry-bound storage)
  or only `LocalArray` (function-local fixed array) initially;
  picking the same answer for both keeps the `Place` story coherent.
- **Surface parsing.** `reduce_by_index` already has a producer at
  `tlc::mod::transform_soac_reduce_by_index`. `scatter` is not in the
  SOAC names list — Scatter needs a `transform_soac_scatter` in the
  same shape before EGIR work matters.
- **Parallel `Filter` → parallel `Scan`.** Already in place;
  prefix-sum over the predicate mask gives per-element write offsets.
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
of the type: the `Array` type's trailing **region** slot holds
`Region(set, binding)`.

- **Born at entry params.** `pin_entry_regions` (the first TLC pass)
  computes each storage entry-param's binding (auto-allocated `set 0,
  0..N`, or an explicit `#[storage(set, binding)]`) and substitutes the
  param's region *variable* → `Region(set, binding)` throughout the entry.
- **Flows by unification.** A view is region-polymorphic everywhere else
  (`∀r. View[…, r]`), so a slice, a `let`, a function argument, or a SOAC
  capture inherits its region the same way it inherits its element type —
  no side-channel, no manual threading. `if c then xs else ys` over two
  different buffers fails to unify, which is the correct "can't pick a
  descriptor at runtime" error.
- **Specialized by monomorphize.** Because the region is an ordinary type
  parameter, `monomorphize` specializes a view function per region exactly
  as it does per element type — `f(xs)` and `f(ys)` over two buffers become
  two monomorphs. (This is what let the dedicated `buffer_specialize` pass
  be deleted.)
- **Read by the backends from the type.** A view is a runtime value (SPIR-V:
  a `{offset,len}` struct; WGSL: a `vec2<u32>`). `ViewIndex` recovers the
  backing buffer via `array_view_region(value_type)` →
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
| `Array` | elem | variant | dim_0 (… dim_{rank-1}), region | Layout `[elem, variant, dim_0…dim_{rank-1}, region]`; rank is implicit (`args.len() - 3`), all arrays rank-1 today. Each dim is `Size(n)` \| `SizeVar(name)` \| `SizePlaceholder` \| `Variable`; variant is `ArrayVariantView` \| `Composite` \| `Virtual` \| `Bounded`. The trailing **region** is `Region(set, binding)` (a storage view), `NoRegion` (a non-view array), or a variable (region-polymorphic) — making a view's buffer a static type property (see View Buffer Provenance) |
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
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
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

985 tests currently pass (9 ignored for pending features). All 69 SPIR-V testfiles in `testfiles/` compile and validate (`bash scripts/validate_testfiles.sh`); 67 of those also validate as WGSL (`bash scripts/validate_testfiles.sh --wgsl` — 2 skip because they depend on linked SPIR-V helpers).

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
entry fs_main(#[location(0)] color: vec3f32) #[location(0)] vec4f32 = ...

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
