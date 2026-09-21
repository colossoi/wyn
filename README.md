# Wyn

A minimal compiler for a Futhark-like programming language that generates SPIR-V and WGSL code for GPU shaders.

## Features

- Functional syntax for shader programming
- Hindley-Milner type inference with polymorphic types
- Higher-order functions (map, reduce, zip, etc.)
- Pattern matching
- SPIR-V and WGSL code generation for Vulkan/WebGPU shaders
- Vertex, fragment_value, fragment_position, fragment_front_facing, fragment_primitive_index, fragment_sample_index, and compute shader support
- Vector and matrix types optimized for GPU operations
- Array operations with size tracking
- Loop constructs

## Project Structure

The project is organized as a Rust workspace:

- **`wyn-core/`** - Compiler library (lexer, parser, type checker, TLC, egglog mid-end, SSA, SPIR-V/WGSL backends). Includes an in-crate generic SSA framework at `ssa::ir` (blocks, values, instructions, terminators) used only for codegen.
- **`wyn/`** - Command-line executable
- **`wyn-analyzer/`** - Language server (in development)
- **`extra/viz/`** - Visualization tool for rendering SPIR-V shaders
- **`prelude/`** - Standard library functions written in Wyn

For package layout, manifests, dependencies, imports, and build commands, see
[Writing Wyn Packages](PACKAGES.md).

## Compiler Architecture

The native CLI and WebAssembly compiler use the same typestate pipeline:

1. **Frontend** parses modules, resolves names and resources, and type checks
   the source. Type holes are rejected unless the CLI requests default values.
2. **TLC** specializes the typed functional program, extracts authored stages,
   pins entry buffers, normalizes SOACs and array representations, monomorphizes,
   inlines helpers, defunctionalizes, applies ownership, removes unreachable
   definitions, and infers input slice bounds.
3. **Egglog** imports TLC into typed sidecar arenas, fuses SOACs, inserts the
   expression graph, simplifies scalar expressions, places computations, and
   schedules physical kernels and resources.
4. **SSA** receives scheduled blocks and the published shader interface,
   optimizes and places pure instructions, and removes unreachable functions.
5. **Backend** validates and legalizes SSA for SPIR-V or WGSL, then emits the
   shader and its runtime contract.

### Egglog checkpoints

| Transition | Output checkpoint | Responsibility |
| --- | --- | --- |
| `egglog::from_tlc` | `Imported` | Import normalized TLC, callable bodies, types, SOACs, and source ABI |
| `egglog::fuse` | `Fused` | Derive fusion candidates and compose selected sidecar bodies |
| `egglog::insert_expressions` | `Expressions` | Build typed expression DAGs, region uses, and dependencies |
| `egglog::simplify` | `Simplified` | Fold constants and optionally apply algebraic rewrites (`-O`) |
| `egglog::place` | `Placed` | Place expressions across branches and loops |
| `egglog::schedule` | `Scheduled` | Plan stages, storage, scratch, dispatch domains, and dependencies |
| `egglog::to_ssa` | SSA `Elaborated` | Emit scheduled blocks and publish the shader/runtime ABI |

Each egglog pass owns its graph. Extracted expressions, types, scalar bodies,
and scheduled blocks pass between checkpoints in typed sidecar arenas.
Fusion sees SOAC layouts and producer/consumer, use, and effect constraints;
scalar syntax and lambda bodies remain opaque to fusion.

Maps use elementwise kernels. Reductions use chunk and combine dispatches;
scans add offset application, and filters use flags, offsets, and compaction.
Nested array work becomes local device loops. Physical kernel metadata retains
stable identities, dependencies, dispatch domains, and resource accesses.
`PipelineTopologyPolicy::AuthoredOnly` (`--direct`) prevents compiler-generated
host stages and intermediate resources.

### Backend boundary

`lower_ssa_to_spirv` and `lower_ssa_to_wgsl_with_program_and_options` perform
SSA optimization, expression placement, reachability filtering, and target
preparation. The WGSL path also adapts push-constant contracts to storage
parameter blocks. The compiler publishes a host program that records resource lifetimes, bindings,
and dispatch order. See [HOST.md](HOST.md) for WHL and Rust/WGPU output.

Tests can use `compile_thru_frontend`, `compile_thru_tlc`, `compile_thru_ssa`,
and `compile_thru_spirv` to stop at shared pipeline checkpoints.

## Example Program

```wyn
-- Render a full-screen triangle through one explicit graphics operation.

def vertex_main(vertex_index: u32, instance_index: u32, draw_index: u32) vertex<vec2f32> =
  let verts = [@[-1.0, -1.0, 0.0, 1.0],
               @[ 3.0, -1.0, 0.0, 1.0],
               @[-1.0,  3.0, 0.0, 1.0]] in
  vertex_output(verts[i32(vertex_index)], @[0.0, 0.0])

entry image(screen: render_target<vec4f32>) render_target<vec4f32> =
  let raster = rasterize_triangles(direct_draw(3u32, 1u32), vertex_main) in
  shade(screen, raster,
    |_, _, _, _, _| @[0.529, 0.808, 0.922, 1.0])  -- Sky blue
```

## Usage

All builds use egglog. `--egglog` remains accepted as a compatibility flag.

```bash
# Compile to SPIR-V
cargo run --bin wyn -- build input.wyn -o output.spv

# Compile to WGSL
cargo run --bin wyn -- build input.wyn -o output.wgsl -t wgsl

# Emit Rust/WGPU host code alongside WGSL
cargo run --bin wyn -- build input.wyn --target-double rust-wgpu -o output.wgsl

# Use the egglog route with WGSL output and an optional SSA dump
cargo run --bin wyn -- build input.wyn --egglog -t wgsl -o output.wgsl --output-mir output.ssa

# Compile a graphics program directly, without compiler-created prepasses
cargo run --bin wyn -- build input.wyn -o output.spv --graphics --direct
cargo run --bin wyn -- build input.wyn -o output.wgsl -t wgsl --graphics --direct

# Opt in to backend-local u64 emulation for WGSL
cargo run --bin wyn -- build input.wyn -o output.wgsl -t wgsl --wgsl-emulate-u64

# Type check without generating code (`--graphics` is required for graphics vocabulary)
cargo run --bin wyn -- check input.wyn --graphics

# Output intermediate representations
cargo run --bin wyn -- build input.wyn --output-tlc out.tlc --output-mir out.ssa

# Run a compiler-published SPIR-V or WGSL pipeline
cd extra/viz && cargo run -- pipeline ../../shader.wgsl
```

The egglog route constructs `map`, `reduce`, and `scan` as Scremas
in `egglog::from_tlc`. Full expressions, types, bodies, argument values, control
flow, and metadata are retained in `IdArena` sidecars. Fusion receives only a
summary: SOAC layouts, producer/consumer links, uses, and dependency/effect
constraints. Scalar expression syntax and lambda bodies are opaque to fusion.
After fusion, `egglog::insert_expressions` adds a separate typed expression DAG,
region parameters/results, call and structured-loop links, and data dependencies
through `expressions.egg`. Region-use facts associate globally interned syntax
with its use sites; they do not place computations or share runtime values across
invocations. Operation identities keep effectful executions distinct.
This route runs through scheduling and `egglog::to_ssa`, then uses the shared
backend and file output. Egglog timing goes to stderr. Within each function,
lowering walks backward from outputs and required effects, then topologically
orders only reachable operations. Dead records can remain in the source sidecar.
Each egglog pass owns its graph; extracted expressions and scheduled blocks pass
between stages in the IR. Executable blocks contain no SOACs. Functions adorn
entry blocks.
A shared pass loop analyzes a complete graph, derives fusion candidates in
`fusion.egg`, applies the selected composition to sidecar bodies, and repeats
with fresh facts. Selection is
deterministic and greedy. Fresh maps can fuse into single-input maps, scans, or
reductions across independent operations, preserving captures and logical tuple
elements. Shared live observers, effect barriers, and region boundaries prevent
absorption. Horizontal fusion and fusion across scan barriers remain unimplemented.

After expression insertion, `schedule.egg` selects execution recipes. Rust instantiates their
CFGs and scalar payloads in sidecar arenas. Maps use parallel elementwise kernels;
reductions use chunk and combine dispatches; scans add an offset application
dispatch; filters use flags, local offsets, combined offsets and compaction.
Chunks contain 64 consecutive elements and preserve operator order. Combine
dispatches use one invocation. Grid-stride loops cap launches at 65,535
workgroups. Empty inputs still initialize collective identities and lengths.
Nested array work becomes local device loops. Potentially colliding indexed
writes and effectful bodies use ordered single-invocation kernels.

The scaffold preserves host branches and loops, including conditional and
repeated dispatch sites. A launch completes and makes its writes visible before
host control continues; explicit dispatch dependencies describe stage ordering.
Buffer element types and dynamic grid formulas remain in the sidecar.

The SSA adapter publishes source inputs, planned storage, output bindings, and
static dispatch order through the shader interface and host program. Conditional or repeated
host dispatches still produce explicit unsupported-operation diagnostics.

Graphics vocabulary is opt-in. Without `--graphics`, names such as
`direct_draw`, `rasterize_triangles`, `shade`, and
`render_target` are ordinary, unreserved identifiers: user code may define
them, and otherwise receives the normal undefined-name diagnostic.
`--direct` is a backend-neutral output policy. It preserves authored graphics
stages and rejects programs that would require compiler-created prepass entry
points or intermediate storage.

## Building and Testing

```bash
cargo build --workspace --all-targets
cargo test --workspace
```

The standalone crates under `extra/` are excluded from the root workspace.
Build and test each with `--manifest-path`, including `extra/viz`, `extra/tephra`,
`extra/spirv-validator`, `extra/wyn-doc`, `extra/mdbook-wyn-highlight`,
`extra/treereduce-wyn`, and both `extra/tree-sitter-wyn` Rust manifests:

```bash
cargo build --manifest-path extra/viz/Cargo.toml --all-targets
cargo test --manifest-path extra/viz/Cargo.toml
cargo test --manifest-path wyn-wasm/Cargo.toml
```

Run `scripts/validate_testfiles.ps1` for SPIR-V testfile validation and add
`-Wgsl` for WGSL. The Bash and Nushell variants accept `--wgsl`.

`wyn-wasm` has its own lockfile and needs a separate browser-target check.
Install `wasm32-unknown-unknown`, then run `bash scripts/check_wasm.sh` or
`./scripts/check_wasm.ps1`. These use `--locked` and accept additional Cargo
arguments. The browser dependency enables Web Crypto for egglog's random-number
dependency.

## Language Overview

### Types

- **Primitives**: `i8`/`i16`/`i32`/`i64`, `u8`/`u16`/`u32`/`u64`,
  `f16`/`f32`/`f64`, `bool`
- **Arrays**: `[N]T` for fixed size, `[]T` for inferred size
- **Vectors**: `vec2T`, `vec3T`, and `vec4T`, including `vec2f16`,
  `vec3f16`, and `vec4f16`
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
def vs_main(vertex_index: u32, instance_index: u32, draw_index: u32) vertex<vec3f32> = ...
def fs_main(fragment_value: vec3f32, fragment_position: vec4f32, fragment_front_facing: bool, fragment_primitive_index: u32, fragment_sample_index: u32) vec4f32 = ...
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

- **egglog** — Fusion, scalar optimization, and scheduling
- **nom** — Parser combinators
- **polytype** — Hindley-Milner type system
- **rspirv** — SPIR-V builder

For complete language details, see [SPECIFICATION.md](SPECIFICATION.md).
