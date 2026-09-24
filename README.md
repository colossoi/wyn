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

The [native CLI](wyn/src/main.rs) and [WebAssembly compiler](wyn-wasm/src/lib.rs)
use the same pass order: frontend → typed lambda calculus (TLC) → egglog → SSA →
SPIR-V or WGSL. Typestate checkpoints enforce the order of the public passes.

### Frontend

[ParsedModules::load](wyn-core/src/frontend.rs) parses the source module graph
and loads the compiler prelude. `ParsedModules::type_check` then resolves imports,
elaborates modules, resolves names and resources, folds integer constants and
static array dimensions, resolves type placeholders and `open` declarations,
and runs type inference and stage-context validation. The resulting `TypeChecked`
AST stores inferred types, definition schemes, resolved identifiers, and warnings.

Before TLC lowering, `ast_type_holes::reject_type_holes` rejects source `???`
expressions. CLI builds with `--fill-holes` instead use `fill_type_holes` to replace
them with typed defaults where supported.

### TLC passes

These passes run in the order shown. SOACs are second-order array combinators,
such as `map`, `reduce`, and `scan`; SoA means structure of arrays.

| Pass (`tlc::`) | Responsibility |
| --- | --- |
| `lower_from_ast` | Lower the typed AST to polymorphic TLC, including pattern lowering, while retaining unified root entries and source ownership contracts. |
| `validate_ownership` | Check source consumption and aliasing rules before evaluation or inlining can erase call boundaries. This includes graphics resource ownership. |
| `partial_eval` | Evaluate known applications and constants, simplify their residual terms, and retain computations that depend on runtime values. |
| `extract_stages` | Extract compute, vertex, and fragment entries and their interfaces from root orchestration, including graphics callbacks and the compute work feeding them. |
| `pin_entry_buffers` | Substitute each storage parameter's assigned binding into its buffer-region type before specialization. |
| `normalize_soacs` | Convert arrays of tuples to tuples of arrays, normalize multi-input map parameters, and turn standalone `zip` into tuple construction. |
| `monomorphize` | Specialize intrinsic calls by type and instantiate reachable user definitions from entry points. Array representation specialization is a separate step. |
| `rep_specialize` | Specialize callees with abstract-array parameters for known producer representations, including the bounded capacity of filter results. |
| `inline_small` | Inline eligible small functions and constants. |
| `force_inline_soac_helpers` | Repeatedly inline helpers containing SOACs, array producers, or length queries so fusion and dispatch planning can see that work in the caller. |
| `renormalize_inlined_soa` | Repeat SoA and SOAC normalization for array structure exposed by inlining. |
| `canonicalize_conditional_producers` | Rewrite supported array-valued conditionals into a pointwise map with the branch inside its callback. |
| `normalize_soacs_to_anf` | Lift nested SOAC expressions into explicit let bindings, exposing producer/consumer edges for egglog. |
| `float_runtime_index_nested_producers` | Move eligible runtime-indexed producers out of nested callbacks before defunctionalization, exposing the producer and gather separately. |
| `defunctionalize` | Lift lambdas, make captures explicit, specialize higher-order functions, and lower closure applications to direct calls. |
| `fold_generated_lambdas` | Inline applications of compiler-generated lifted lambdas and remove definitions made unreachable. |
| `apply_ownership` | Promote eligible array updates to in-place updates and mark unique SOAC inputs. Egglog later decides storage reuse using the fused program. |
| `filter_reachable` | Remove definitions not reachable from entry points. |
| `infer_input_slice_bounds` | Attach minimum input-buffer sizes where every use of an input is a constant prefix slice. Other inputs still need sizes from their interface or runtime. |

`--output-tlc` captures the `BuffersPinned` checkpoint, before SoA normalization
and monomorphization. The TLC input to egglog is `InputSliceBoundsInferred`.

### Egglog checkpoints

Screma is the combined scan/reduce/map representation used for collective fusion.

| Transition | Output checkpoint | Responsibility |
| --- | --- | --- |
| `egglog::from_tlc` | `Imported` | Import normalized TLC, callable bodies, types, and source ABI into typed arenas; construct Scremas for map/reduce/scan and export structural fusion facts. |
| `egglog::fuse` | `Fused` | Complete a deterministic greedy fusion plan on a persistent graph, then construct the selected bodies in the sidecar arenas. |
| `egglog::insert_expressions` | `Expressions` | Insert a separate typed expression DAG, region interfaces and uses, structured control, and execution dependencies into egglog. |
| `egglog::simplify` | `Simplified` | Fold constants and simplify scalar expressions in that graph; optionally explore algebraic rewrites with `-O`, then extract and apply replacements. |
| `egglog::place` | `Placed` | Use Rust analysis to place safe shared and loop-invariant expressions in structured regions, including SOAC captures. Memory reads and opaque calls are not speculated. |
| `egglog::schedule` | `Scheduled` | Derive execution recipes, host/device residency, storage allocation and reuse, output routes, scratch, dispatch domains, and dependencies; instantiate executable blocks. |
| `egglog::to_ssa` | SSA `Elaborated` | Lower scheduled kernels for the selected target and publish source inputs, planned resources, outputs, and host computations through the shader/runtime ABI. |

Types, expressions, bodies, control flow, and metadata remain in typed `IdArena`
sidecars. Fusion, scalar simplification, and scheduling use separate egglog
graphs; expression insertion hands its graph directly to scalar simplification.
Interning expression syntax does not share runtime values across invocations:
region-use facts and placement determine where values are evaluated, and operation
identities keep effectful executions distinct. Dependency analysis works backward
from results and required effects and orders only live operations; unused records
can remain in the sidecar.

### Fusion

[fusion.egg](wyn-core/src/egglog/fusion/fusion.egg) plans over structural summaries:
iteration domains, SOAC layouts, producer/consumer links, uses, scalar dependency
summaries, and memory/effect constraints. Scalar expression syntax and callback
bodies remain opaque to the fusion rules. Source facts are analyzed once; the
planner contracts groups on a persistent graph and records the complete plan
before Rust constructs the composed bodies.

Supported cases include vertical producer/consumer fusion, horizontal fusion of
compatible independent maps/scans/reductions, and retention of shared producer
outputs. Maps can compose with filters and indexed operations; eligible filtered
reductions become masked collectives, and indexed demands can compute selected
map elements without materializing the whole array. Scan dependencies, captures,
region boundaries, and memory/effect barriers constrain these transformations.
For example, a scan followed by a map can fuse, while a reduction consuming the
scan's prefix values still requires a separate operation.

### Scheduling and publication

The scheduling graph combines the rules in `planning.egg`, `execution.egg`,
`schedule.egg`, `residency.egg`, `allocation.egg`, `reuse.egg`, `dispatch.egg`, and
`epilogues.egg`. It selects recipes and resources before Rust builds their control
flow and scalar bodies. Scheduled executable blocks contain no SOAC operations.

For top-level array work, the planner selects these
[execution recipes](wyn-core/src/egglog/schedule.egg), falling back to ordered
execution when the callback effects prevent parallel execution:

| Work | Physical execution |
| --- | --- |
| Maps without collectives | Parallel elementwise kernels. |
| Reductions without scans | A chunk dispatch followed by a combine dispatch using one 256-invocation workgroup and an ordered reduction tree. |
| Scans, possibly fused with reductions | Chunk processing, a single-invocation combine dispatch, then parallel offset application and post-map work. |
| Filters that still need compaction | One 64-invocation workgroup processes tiles in order, computes predicate prefixes in workgroup memory, writes stable compacted output, and publishes its length. |
| Eligible indexed reductions | Parallel atomic updates, including compare/exchange where needed. |
| Eligible bucket scatter | A count-clearing dispatch followed by parallel slot reservation and writes. |
| Scatter and other work without a parallel-safe recipe | Ordered single-invocation kernels. |
| Nested array work | Local device loops within the enclosing invocation. |

Collective chunks contain 64 consecutive elements and preserve operand order.
Grid-stride kernels cap launches at 65,535 workgroups. Empty inputs still initialize
collective identities and result lengths.

The planner derives CPU availability, bounded rematerialization costs, and scalar
kernel groups. CPU-available scalar work can run on the host; GPU-dependent work
stays on the GPU. Cheap immutable expressions can be recomputed at their consumers,
and compatible consecutive GPU scalar operations can share a kernel, storing only
results needed outside it. Mutable reads, effects, and unbounded work are excluded
from rematerialization. Allocation and reuse decisions use post-fusion liveness,
ownership candidates, and resource accesses.

Physical kernel metadata retains stable identities, dependencies, dispatch domains,
and resource accesses. The scheduled representation preserves host branches and
loops, but `to_ssa` still diagnoses conditional or repeated host dispatches as
unsupported. Published pipelines use static dispatch sites and explicit ordering.
`PipelineTopologyPolicy::AuthoredOnly` (`--direct`) preserves authored stages and
rejects programs requiring compiler-created prepasses or intermediate storage.

### SSA and backends

The [backend entry points](wyn-core/src/lib.rs), `lower_ssa_to_spirv` and
`lower_ssa_to_wgsl_with_program_and_options`, run these passes after `egglog::to_ssa`:

| Pass (`ssa::`) | Responsibility |
| --- | --- |
| `optimize` | Inline small helpers, fold constants, reuse dominating immutable expressions, and intern expressions that can safely move. |
| `place_floating` | Assign reachable floating expressions to concrete control-flow blocks. |
| `filter_reachable` | Prune functions and constants unreachable from entries. |
| `prepare_spirv` / `prepare_wgsl` | Remove dead pure instructions, publish texture-sampling requirements, and reject unresolved type representations. SPIR-V also verifies buffer layouts; WGSL first promotes constants needing addressable storage. |

SPIR-V emission lowers blocks and block parameters to SPIR-V control flow and phi
nodes. WGSL emission structurizes the CFG into statements, branches, and loops,
applies the selected `u64` policy, and adapts push-constant contracts to storage
parameter blocks. Both return shader code and a host program recording resource
lifetimes, bindings, and execution order. See [HOST.md](HOST.md) for WHL and
Rust/WGPU output.

`--output-mir` dumps `Elaborated` SSA before these cleanup and backend passes.
Constant folding and the normal pipeline run without `-O`; that flag enables
additional egglog algebraic rewrites. Tests can use `compile_thru_frontend`,
`compile_thru_tlc` (through TLC reachability), `compile_thru_ssa`, and
`compile_thru_spirv` to stop at shared checkpoints.

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

# Compile to WGSL and dump SSA before backend cleanup
cargo run --bin wyn -- build input.wyn -t wgsl -o output.wgsl --output-mir output.ssa

# Compile a graphics program directly, without compiler-created prepasses
cargo run --bin wyn -- build input.wyn -o output.spv --graphics --direct
cargo run --bin wyn -- build input.wyn -o output.wgsl -t wgsl --graphics --direct

# Opt in to backend-local u64 emulation for WGSL
cargo run --bin wyn -- build input.wyn -o output.wgsl -t wgsl --wgsl-emulate-u64

# Check types, ownership, and entry interfaces (`--graphics` enables graphics vocabulary)
cargo run --bin wyn -- check input.wyn --graphics

# Output intermediate representations
cargo run --bin wyn -- build input.wyn --output-tlc out.tlc --output-mir out.ssa

# Run a compiler-published SPIR-V or WGSL pipeline
cd extra/viz && cargo run -- pipeline ../../shader.wgsl
```

Generated Rust/WGPU modules provide `host_<entry>` to record and submit an entry.
Entries without CPU readbacks also provide `encode_<entry>`, taking
`&mut HostContext` and `&mut wgpu::CommandEncoder`. The caller can record attachment
clears, generated compute/render passes, and other commands in the same encoder,
then submit once. Reuse the context across frames and submit recorded calls in
order. If recording fails, discard the encoder because it may contain partial
commands. Entries requiring CPU readbacks submit the pending commands and readback
copy together, then resume recording after the read completes.

Graphics vocabulary is opt-in. Without `--graphics`, names such as
`direct_draw`, `rasterize_triangles`, `shade`, and
`render_target` are ordinary, unreserved identifiers: user code may define
them, and otherwise receives the normal undefined-name diagnostic.

Egglog timing goes to stderr; `--verbose` includes pass and sub-pass timings.

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
