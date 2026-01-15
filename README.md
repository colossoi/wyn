# Wyn

A minimal compiler for a Futhark-like programming language that generates SPIR-V code for GPU shaders.

## Features

- Hindley-Milner type inference with polymorphic types
- Higher-order functions (map, reduce, zip, etc.)
- Vector and matrix types optimized for GPU operations
- Pattern matching
- SPIR-V code generation for Vulkan/WebGPU shaders
- Graphical and compute (planned) shader kernel generation
- Array operations with size tracking
- Loop constructs
- Module system (in development)
- Futhark-inspired functional syntax for shader programming

## Project Structure

The project is organized as a Rust workspace:

- **`wyn-core/`** - Compiler library (lexer, parser, type checker, TLC, MIR, code generator)
- **`wyn/`** - Command-line executable
- **`spirv-validator/`** - SPIR-V validation tool using wgpu
- **`viz/`** - Visualization tool for rendering SPIR-V shaders
- **`prelude/`** - Standard library functions written in Wyn

## Compiler Architecture

The compiler uses a multi-stage pipeline with typestate-driven phases:

### Frontend (AST)
| Stage | Description |
|-------|-------------|
| **Parsed** | Tokenization and parsing into AST |
| **Desugared** | Range/slice expressions desugared |
| **Resolved** | Name resolution and module imports |
| **AstConstFolded** | Compile-time constant folding |
| **TypeChecked** | Hindley-Milner type inference and checking |
| **AliasChecked** | Uniqueness/alias analysis for mutable references |

### TLC (Typed Lambda Calculus)
| Stage | Description |
|-------|-------------|
| **TlcTransformed** | AST converted to minimal typed lambda calculus |
| **TlcLifted** | All lambdas lifted to top-level, captures explicit |

### Backend (MIR)
| Stage | Description |
|-------|-------------|
| **Flattened** | AST to MIR conversion with defunctionalization |
| **MaterializationsHoisted** | Duplicate materializations hoisted from branches |
| **Normalized** | A-normal form conversion |
| **Monomorphized** | Polymorphic functions specialized to concrete types |
| **Folded** | Partial evaluation and constant folding (optional) |
| **Reachable** | Dead code elimination |
| **Lifted** | Final binding lifting |
| **Lowered** | SPIR-V code generation |

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

# Type check without generating code
cargo run --bin wyn -- check input.wyn

# Output intermediate representations
cargo run --bin wyn -- compile input.wyn --output-tlc out.tlc    # Typed Lambda Calculus
cargo run --bin wyn -- compile input.wyn --output-init-mir out.mir  # Initial MIR
cargo run --bin wyn -- compile input.wyn --output-final-mir out.mir # Final MIR

# Visualize a SPIR-V shader
cd viz && cargo run vf ../shader.spv --vertex vertex_main --fragment fragment_main
```

## Building and Testing

```bash
cargo build --release
cargo test
```

All 546 tests currently pass.

## Language Overview

### Types

- **Primitives**: `i32`, `f32`, `bool`, etc.
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
- No recursion - use `loop` or higher-order functions instead
- `match` expressions parsed but not fully implemented in code generation

## Dependencies

- **nom** - Parser combinators
- **polytype** - Hindley-Milner type system
- **rspirv** - SPIR-V builder
- **thiserror** - Error handling
- **clap** - CLI parsing

For complete language details, see [SPECIFICATION.md](SPECIFICATION.md).
