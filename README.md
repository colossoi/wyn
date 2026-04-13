# Wyn

A minimal compiler for a Futhark-like programming language that generates SPIR-V and GLSL code for GPU shaders.

## Features

- Hindley-Milner type inference with polymorphic types
- Higher-order functions (map, reduce, zip, etc.)
- Vector and matrix types optimized for GPU operations
- Pattern matching
- SPIR-V and GLSL code generation for Vulkan/WebGPU shaders
- Vertex, fragment, and compute shader support
- Array operations with size tracking
- Loop constructs
- Futhark-inspired functional syntax for shader programming

## Project Structure

The project is organized as a Rust workspace:

- **`wyn-ssa/`** - Generic SSA framework library (blocks, values, instructions, terminators, optimization passes)
- **`wyn-core/`** - Compiler library (lexer, parser, type checker, TLC, SSA, SPIR-V/GLSL backends)
- **`wyn/`** - Command-line executable
- **`wyn-analyzer/`** - Language server (in development)
- **`viz/`** - Visualization tool for rendering SPIR-V shaders
- **`prelude/`** - Standard library functions written in Wyn

## Compiler Architecture

The compiler uses a multi-stage pipeline with typestate-driven phases. Each stage consumes `self` and returns the next stage, enforcing valid ordering at compile time.

### Frontend (AST)
| Stage | Module | Description |
|-------|--------|-------------|
| **Parsed** | `parser` | Tokenization and parsing into AST |
| **Desugared** | `desugar` | Range/slice expressions desugared; SOAC names rewritten to intrinsics |
| **Resolved** | `name_resolution` | Name resolution and module imports |
| **AstConstFoldedEarly** | `ast_const_fold` | Compile-time integer constant folding |
| **TypeChecked** | `types::checker` | Hindley-Milner type inference and checking |
| **AliasChecked** | `alias_checker` | Uniqueness/alias analysis for in-place updates |

### TLC (Typed Lambda Calculus)
| Stage | Module | Description |
|-------|--------|-------------|
| **TlcTransformed** | `tlc::transform` | AST converted to minimal typed lambda calculus |
| **TlcPartialEvaled** | `tlc::partial_eval` | Constant folding and algebraic simplifications |
| **TlcSoaNormalized** | `tlc::soa` | SoA transform (`[n](A,B)` → `([n]A, [n]B)`) + Map+Zip flattening + standalone Zip elimination |
| **TlcFused** | `tlc::fusion` | SOAC fusion: map-map, interprocedural producer-consumer |
| **TlcDefunctionalized** | `tlc::defunctionalize` | Lambda lifting + SOAC capture flattening |
| **TlcMonomorphized** | `tlc::specialize`, `tlc::monomorphize` | Polymorphic intrinsics specialized; user functions monomorphized |
| **TlcBufferSpecialized** | `tlc::buffer_specialize` | Storage buffer parameter specialization |
| **TlcGeneratedLambdasFolded** | `tlc::inline` | Fold compiler-generated `_w_lambda_*` defs (from defunctionalization) back at call sites + DCE |
| **TlcSmallInlined** | `tlc::inline` | Inline small user functions and constants |
| **TlcParallelized** | `tlc::parallelize` | Structural parallelization of compute shader SOACs (chunked entries, pipeline descriptors) |
| **TlcReachable** | `tlc::inline` | Dead definition elimination |

### EGIR (Acyclic E-Graph IR)

TLC lowers directly into an **acyclic e-graph** (sea of pure nodes + skeleton CFG) rather than into sequential SSA. The e-graph is the central mid-end IR: most optimizations fall out of the data structure without dedicated passes.

Key structures (`egir::types`):
- **`ENode::Pure { op, operands }`** — hash-consed pure value; GVN is automatic.
- **`Skeleton`** — a CFG of side-effectful instructions (`SideEffect`) anchored in blocks with `SkeletonTerminator`s. Operands are `NodeId`s.
- **`ENode::SideEffectResult`** — unique (non-hash-consed) handle for the value produced by a skeleton instruction, consumable by the pure sea.
- **Purity is blacklisted**, not whitelisted: an instruction flows into the pure sea unless it carries an effect token *and* is one of `Alloca`, `Load`, `Store`, `Soac`. `Call`, `Intrinsic`, `StorageView*`, `OutputPtr` are pure.

What you get "for free":
- **GVN** — `intern_pure(op, operands)` returns an existing `NodeId` when `(op, operands)` match.
- **Constant folding** — `intern_pure` consults `fold.rs` before inserting; folded results are themselves interned, so folds compose.
- **DCE** — `elaborate` is demand-driven from skeleton roots; unreached pure nodes are never emitted.
- **CSE along a domtree path** — `ScopedMap` tracks emitted nodes per dominator scope; siblings never cross-pollute.
- **LICM** — `loop_analysis` picks the outermost loop where all operands are available as the placement point.
- **Branch folding + redundant-phi elimination** — `skel_opt` rewrites the skeleton CFG before elaboration.

| Module | Description |
|--------|-------------|
| `from_tlc` | Direct TLC term → EGraph lowering (replaces the old `tlc::to_ssa`) |
| `canonicalize` | SSA `FuncBody` → EGraph (alternative entry, used for round-trip tests) |
| `elaborate` | EGraph → `FuncBody` via `FuncBuilder`, demand-driven and scoped |
| `extract` | Cost-based bottom-up selection of the best representative per node |
| `soac_expand` | Rewrites every `SideEffectKind::Pending(PendingSoac::...)` into an explicit loop subgraph (block-split + header/body/after, alloca/store for output arrays, view loads for view inputs, SoA-aware reads). Runs between `from_tlc` and `elaborate`. |
| `skel_opt` | Skeleton-level CFG rewrites (branch folding, redundant phi elim) |
| `fold` | Algebraic simplification applied during `intern_pure` |
| `domtree` | Generic dominator tree working over both SSA and skeleton CFGs |
| `loop_analysis` | Loop nesting info for LICM placement |
| `rewrite` | Trait + driver for pattern-based rewrite rules (Phase 2; current ruleset empty) |

### SSA

The SSA IR is built on the **`wyn-ssa`** crate, a generic SSA framework parameterized over instruction type, effect token type, and value type. The concrete instantiation uses `InstKind` for instructions, `EffectToken` for effects, and `Type<TypeName>` for types.

Key features of the SSA IR:
- CFG with basic blocks and block parameters (not phi nodes)
- Effect tokens tracked at framework level on `InstNode`, not inside instruction variants
- `ValueDef::FunctionParam` distinct from `ValueDef::Param` (block params)
- `ControlHeader` metadata stored in a side-map on `FuncBody`, not on blocks

| Stage | Module | Description |
|-------|--------|-------------|
| **SsaConverted** | `egir::from_tlc` + `egir::soac_expand` | TLC lowered into EGraph, SOACs expanded into explicit loops, then elaborated to SSA `FuncBody` |
| **SsaMaterialized** | `spirv::materialize` | Dynamic array indices materialized for SPIR-V (+ LICM) |
| **Lowered** | `spirv` | SSA to SPIR-V code generation |

Alternative backend:
| Stage | Module | Description |
|-------|--------|-------------|
| **GLSL** | `glsl` | SSA to GLSL source code (lowered straight from `SsaConverted`) |

### SSA Passes in `wyn-ssa` (generic framework)

- **`forward_single_pred_params`** — Forward block parameters through single-predecessor edges
- **`eliminate_empty_blocks`** — Remove empty unconditional-jump blocks
- **`inline_block_param` / `inline_entry_param`** — Replace a block parameter with a constant instruction
- **`verify_effects`** — Check effect token chain linearity

### SPIR-V Backend Optimizations

- **Polytype cache** — Memoizes `polytype_to_spirv` to prevent duplicate type instructions
- **Composite constant cache** — Deduplicates `OpConstantComposite` by (type, constituents)
- **Null constant cache** — Deduplicates `OpConstantNull` by type
- **Materialize pass** — Rewrites dynamic `Index` into `Materialize` + `DynamicExtract`, then `lift_and_merge` hoists `Materialize` out of loops

### Defunctionalization

The `tlc::defunctionalize` pass implements Futhark-style defunctionalization:
1. **Lambda lifting**: All lambdas become top-level definitions with captures as extra parameters
2. **StaticVal tracking**: Tracks which values are known function closures with captured variables
3. **SOAC capture flattening**: When a closure is passed to a SOAC (map, reduce, etc.), its captures are flattened as trailing arguments

Example transformation:
```
-- Input (lambda with capture y):
map(|x| x + y, arr)

-- After defunctionalization:
_lambda_0 = λ(y, x). x + y
map(_lambda_0, arr, y)  -- capture y flattened as trailing arg
```

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

564 tests currently pass (6 ignored for pending features). All 24 end-to-end testfiles in `testfiles/` compile and validate (`bash scripts/validate_testfiles.sh`).

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
- **slotmap** — Arena allocation for SSA IDs (via wyn-ssa)
- **thiserror** — Error handling
- **clap** — CLI parsing

For complete language details, see [SPECIFICATION.md](SPECIFICATION.md).
