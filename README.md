# Wyn

A minimal compiler for a Futhark-like programming language that generates SPIR-V, GLSL, and WGSL code for GPU shaders.

## Features

- Hindley-Milner type inference with polymorphic types
- Higher-order functions (map, reduce, zip, etc.)
- Vector and matrix types optimized for GPU operations
- Pattern matching
- SPIR-V, GLSL, and WGSL code generation for Vulkan/WebGPU shaders
- Vertex, fragment, and compute shader support
- Array operations with size tracking
- Loop constructs
- Futhark-inspired functional syntax for shader programming

## Project Structure

The project is organized as a Rust workspace:

- **`wyn-core/`** - Compiler library (lexer, parser, type checker, TLC, EGIR mid-end, SSA, SPIR-V/GLSL/WGSL backends). Includes an in-crate generic SSA framework at `ssa::framework` (blocks, values, instructions, terminators) used only for codegen.
- **`wyn/`** - Command-line executable
- **`wyn-analyzer/`** - Language server (in development)
- **`viz/`** - Visualization tool for rendering SPIR-V shaders
- **`prelude/`** - Standard library functions written in Wyn

## Compiler Architecture

The compiler uses a multi-stage pipeline with typestate-driven phases. Each stage consumes `self` and returns the next stage, enforcing valid ordering at compile time.

### Mid-End: Acyclic E-Graph

The mid-end is an **acyclic e-graph**: a sea of hash-consed pure nodes plus a skeleton CFG of effectful instructions. Most optimizations fall out of the data structure without dedicated passes.

Key structures (`egir::types`):
- **`ENode::Pure { op, operands }`** — hash-consed pure value; GVN is automatic.
- **`Skeleton`** — a CFG of side-effectful instructions (`SideEffect`) anchored in blocks with `SkeletonTerminator`s. Operands are `NodeId`s.
- **`ENode::SideEffectResult`** — unique (non-hash-consed) handle for the value produced by a skeleton instruction, consumable by the pure sea.
- **Purity is blacklisted**, not whitelisted: the only `InstKind`s kept in the skeleton are `Alloca`, `Load`, `Store`. Everything else — `Call`, `Intrinsic`, `StorageView*`, `StorageViewIndex`, `OutputPtr`, `Index`, `Project`, etc. — is hash-consed into the pure sea.

What you get "for free":
- **GVN** — `intern_pure(op, operands)` returns an existing `NodeId` when `(op, operands)` match.
- **Constant folding** — `intern_pure` consults `fold.rs` before inserting; folded results are themselves interned, so folds compose.
- **DCE** — `elaborate` is demand-driven from skeleton roots; unreached pure nodes are never emitted.
- **CSE along a domtree path** — `ScopedMap` tracks emitted nodes per dominator scope; siblings never cross-pollute.
- **LICM** — `loop_analysis` picks the outermost loop where all operands are available as the placement point.
- **Branch folding + redundant-phi elimination** — `skel_opt` rewrites the skeleton CFG before elaboration.

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
| **TlcPartialEvaled** | `tlc::partial_eval` | Constant folding and algebraic simplifications |
| **TlcSoaNormalized** | `tlc::soa` | SoA transform (`[n](A,B)` → `([n]A, [n]B)`) + Map+Zip flattening + standalone Zip elimination |
| **TlcFused** | `tlc::fusion` | SOAC fusion: map-map, interprocedural producer-consumer |
| **TlcOwnershipApplied** | `tlc::ownership` | Backward ownership-liveness analysis with fixed-point over loops/SOAC bodies. Reports use-after-move; rewrites `_w_intrinsic_array_with` → `_w_intrinsic_array_with_inplace` where the source's owner is mutable and dead-after the call |
| **TlcDefunctionalized** | `tlc::defunctionalize` (orchestration) + `tlc::closure_convert` + `tlc::hof_specialize` | Lambda lifting + HOF specialization. Captures for SOAC-position lambdas land on a `SoacBody { lam, captures }` wrapper, not on the `Lambda` struct. Three verifiers (`verify_closure_converted`, `verify_hof_specialized`, `verify_closure_calls_lowered` from `tlc::closure_calls_lower`) guard the architectural seams |
| **TlcMonomorphized** | `tlc::specialize`, `tlc::monomorphize` | Polymorphic intrinsics specialized; user functions monomorphized |
| **TlcBufferSpecialized** | `tlc::buffer_specialize` | Storage buffer parameter specialization |
| **TlcGeneratedLambdasFolded** | `tlc::inline` | Fold compiler-generated `_w_lambda_*` defs (from defunctionalization) back at call sites + DCE |
| **TlcSmallInlined** | `tlc::inline` | Inline small user functions and constants |
| **TlcParallelized** | `tlc::parallelize` | Structural parallelization of compute shader SOACs (chunked entries, pipeline descriptors) |
| **TlcReachable** | `tlc::inline` | Dead definition elimination |

### EGIR (Acyclic E-Graph IR)
| Stage | Module | Description |
|-------|--------|-------------|
| **EgirRaw** | `egir::from_tlc` | TLC term → EGraph; `_w_intrinsic_*` calls lowered as pure `PureOp::Intrinsic` (effectful intrinsics like `_w_intrinsic_storage_index` handled by explicit arms) |
| **EgirSoacExpanded** | `egir::soac_expand` | Every `PendingSoac` rewritten into an explicit loop subgraph (block-split + header/body/after, alloca/store for output arrays, view loads, SoA-aware reads). Map with a statically-sized input of ≤16 elements is unrolled via a shared `try_unroll` / `build_loop` / `expand_loop` core |
| **EgirMaterialized** | `egir::materialize` | (optional, SPIR-V only) dynamic `Index` → `Materialize` + `DynamicExtract`, then LICM-hoisted out of loops |
| **EgirSkelOptimized** | `egir::skel_opt` | Skeleton CFG rewrites: branch folding, redundant-phi elimination |

### SSA (codegen only)
| Stage | Source | Description |
|-------|--------|-------------|
| **SsaConverted** | `EgirSkelOptimized::elaborate` | EGIR chain elaborated into SSA `Program` |
| **Lowered** | `spirv` / `glsl` / `wgsl` | SSA to SPIR-V, GLSL, or WGSL |

SSA is intentionally minimal: all mid-end machinery (effect tokens, canonicalization, verification, generic transform passes) lives in EGIR. The `ssa::framework` module provides a generic CFG-with-block-params representation; `ssa::types` is the concrete instantiation (`InstKind` instructions, `Type<TypeName>` values).

Key properties:
- CFG with basic blocks and block parameters (not phi nodes)
- No effect tokens at the SSA layer — instruction order is fixed by elaboration
- `ValueDef::FunctionParam` distinct from `ValueDef::Param` (block params)
- `ControlHeader` metadata stored in a side-map on `FuncBody`, not on blocks

### SPIR-V Backend Optimizations

- **Polytype cache** — Memoizes `polytype_to_spirv` to prevent duplicate type instructions
- **Composite constant cache** — Deduplicates `OpConstantComposite` by (type, constituents)
- **Null constant cache** — Deduplicates `OpConstantNull` by type

### Defunctionalization

The defunctionalization stage spans four sibling modules under `tlc::`:
`closure_convert` (lambda lifting + free-variable analysis),
`hof_specialize` (HOF specialization runtime), `closure_calls_lower`
(post-pass verifier), and `defunctionalize` (orchestration).

What each step does:
1. **Lambda lifting** (`closure_convert`): Every standalone `TermKind::Lambda` is hoisted to a top-level `Def`, replacing the lambda site with a `Var` referring to the lifted symbol. Free variables of the lifted lambda become extra trailing parameters on the lifted def, threaded as args at every call site. The `Lambda` struct is `{ params, body, ret_ty }` — a clean function abstraction with no closure baggage.
2. **HOF specialization** (`hof_specialize`): User-defined higher-order functions get specialized per-callsite — every reachable top-level def ends with no function-typed parameters.
3. **SOAC envelope captures**: Lambdas embedded in `SoacOp` envelopes (`Map`, `Reduce`, `Scan`, …) are also lifted, but the SOAC keeps a `SoacBody { lam: Lambda, captures: Vec<(SymbolId, Type, Term)> }` payload so the loop body and its closed-over values stay together for later expansion. The same shape is used for `ArrayExpr::Generate`.

Example transformation:
```
-- Input (lambda with capture y):
map(|x| x + y, arr)

-- After defunctionalization:
_lambda_0 = λ(x, y). x + y       -- y is a regular param of the lifted def
map(SoacBody { lam: _lambda_0, captures: [(y_sym, ty, Var(y))] }, arr)
```

Three verifier-checked invariants run between phases:
`verify_closure_converted` (no `TermKind::Lambda` outside SOAC envelopes),
`verify_hof_specialized` (no reachable top-level def has a function-typed
parameter), and `verify_closure_calls_lowered` (every `App.func` is a `Var`
or operator, every call is fully arity-matched — including intrinsic
arities derived from `builtins::intrinsic_arity`).

## Example Program

```futhark
-- Simple shader that renders a full-screen triangle

def positions: [3]vec4f32 =
  [vec4 (-1.0f32) (-1.0f32) 0.0f32 1.0f32,
   vec4 3.0f32 (-1.0f32) 0.0f32 1.0f32,
   vec4 (-1.0f32) 3.0f32 0.0f32 1.0f32]

#[vertex]
def vertex_main(
    #[builtin(vertex_index)] vertex_id: i32
): #[builtin(position)] vec4f32 =
  positions[vertex_id]

#[fragment]
def fragment_main(): #[location(0)] vec4f32 =
  vec4 0.529f32 0.808f32 0.922f32 1.0f32  -- Sky blue
```

## Usage

```bash
# Compile to SPIR-V
cargo run --bin wyn -- compile input.wyn -o output.spv

# Compile to GLSL
cargo run --bin wyn -- compile input.wyn -o output.glsl -t glsl

# Compile to Shadertoy GLSL
cargo run --bin wyn -- compile input.wyn -o output.glsl -t shadertoy

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

552 tests currently pass (5 ignored for pending features). All 24 end-to-end testfiles in `testfiles/` compile and validate (`bash scripts/validate_testfiles.sh`).

## Language Overview

### Types

- **Primitives**: `i32`, `u32`, `f32`, `bool`
- **Arrays**: `[N]T` for fixed size, `[]T` for inferred size
- **Vectors**: `vec2f32`, `vec3f32`, `vec4f32` (SPIR-V types)
- **Matrices**: `mat2f32`, `mat3f32`, `mat4f32`
- **Tuples**: `(T1, T2, ...)`
- **Functions**: `T1 -> T2`

### Key Syntax

```futhark
-- Function definitions
def add x y = x + y
def reverse [n] (arr: [n]i32): [n]i32 = ...

-- Shader entry points with attributes
#[vertex]
def vs_main(#[builtin(vertex_index)] id: i32): #[builtin(position)] vec4f32 = ...

#[fragment]
def fs_main(#[location(0)] color: vec3f32): #[location(0)] vec4f32 = ...

-- Lambdas and pattern matching
\x -> x + 1
\(x, y) -> x + y

-- Loops (no recursion allowed)
loop (acc, i) = (0, 0) while i < n do (acc + arr[i], i + 1)

-- Higher-order functions
map (\x -> x * 2) arr
reduce (+) 0 arr
```

### Type Inference

```futhark
def identity x = x
-- Inferred: ∀a. a -> a

def zip_arrays xs ys = zip xs ys
-- Inferred: ∀n t1 t2. [n]t1 -> [n]t2 -> [n](t1, t2)
```

## Current Limitations

- Module system is partially implemented
- No recursion — use `loop` or higher-order functions instead
- `match` expressions parsed but not fully implemented in code generation

## Dependencies

- **nom** — Parser combinators
- **polytype** — Hindley-Milner type system
- **rspirv** — SPIR-V builder
- **slotmap** — Arena allocation for SSA IDs
- **thiserror** — Error handling
- **clap** — CLI parsing

For complete language details, see [SPECIFICATION.md](SPECIFICATION.md).
