# Wyn

Wyn is a functional, array-centric language for GPU programming. Programs compose
transformations of values using higher-order functions, immutable data, and static
type inference. Arrays provide a common model for inputs, intermediate values,
and results. The compiler specializes away function values and turns array
operations into GPU computation, with SPIR-V and WGSL output.

## Features

- Higher-order functions and array combinators such as `map`, `reduce`, `scan`, and `filter`
- Arrays as the primary data type, with multiple dimensions and sizes tracked by the type system
- Hindley-Milner type inference with polymorphic types
- Type holes (`???`) that report the type expected at an unfinished expression
- Structured data types, including tuples, records, and sum types, with pattern matching
- Graphics mode for shader rendering, combining rasterization and shading with array computation
- SPIR-V and WGSL code generation, with host code for resource management and execution
- Vector and matrix types optimized for GPU operations
- Uniqueness types that permit safe in-place array updates
- Modules, package imports, and expression-based loops

## Example

```wyn
def square(x: f32) f32 = x * x
def add(x: f32, y: f32) f32 = x + y
entry sum_squares(xs: []f32) f32 =
  let squares = map(square, xs) in
  reduce(add, 0.0, squares)
```

This program uses `map` to square each element of an input array, then `reduce`
to add the squares. The application calls `sum_squares`; for input
`[1.0, 2.0, 3.0]`, it returns `14.0`.

## Runtime Structure

A compiled Wyn program consists of device code and a companion host program.
Device code performs the GPU computation; the host program manages resources and
coordinates its execution. A single source entry can become several kernels,
with the compiler determining their intermediate storage, dispatch dimensions,
dependencies, and result handling.

The primary application output is a SPIR-V shader (`.spv`) paired with a generated
Rust/WGPU wrapper (`.rs`). The wrapper allocates resources, binds inputs, records
commands, and returns results to the application.
Generated `host_<entry>` functions execute entries; entries without CPU readbacks
also expose `encode_<entry>` for recording into an application's command encoder.

WGSL supports the same host-orchestrated approach for WebGPU. For direct shader
integration, `--direct` keeps array computation serial within each shader
invocation, lowering it to device loops without compiler-created prepasses or
intermediate host resources. This permits fully serial array evaluation inside a
WGSL shader; rendering still runs vertex and fragment invocations in parallel.
Direct mode is also available for SPIR-V, subject to the same storage constraints.

Graphics builds (`--graphics`) emit vertex and fragment shaders together with host
code for render passes and any supporting compute dispatches. The host program
coordinates resources and execution order across both graphics and compute work.

Shader format and host output are separate choices. `-t spirv` or `-t wgsl`
selects the device format, and `--target-double rust-wgpu` emits a sibling `.rs`
wrapper. The default host output is readable Wyn Host Lisp (`.wynhost`), which
describes the compiled execution plan and can be run by `extra/viz`. The current
CLI defaults to SPIR-V with WHL, or to WGSL when Rust/WGPU is selected without
an explicit shader target. See [HOST.md](HOST.md) for the host interface and output
contracts.

## Compiler Architecture

The main compiler is the `wyn-core` library. It progressively replaces source
abstractions with explicit computation, control flow, and storage. Its fixed
sequence of stages is shown below. Each stage operates on a particular program
representation through several passes, each responsible for a transformation or
analysis.

| Stage | Purpose and representation |
| --- | --- |
| Frontend / AST | Build an abstract syntax tree (AST), resolve source-level meaning, and attach inferred types and symbol identities. |
| TLC | Use typed lambda calculus to simplify functional code, specialize functions, normalize arrays, and eliminate higher-order values. |
| Egglog | Use graphs of array operations, scalar expressions, and dependencies to fuse work and plan execution and storage. |
| SSA | Express device computation as control-flow blocks with single-assignment values, ready for final cleanup and target preparation. |
| Backends | Lower the prepared program to SPIR-V or WGSL and emit its companion host program. |

Passes consume and produce typed checkpoints that enforce their ordering and
record which invariants later passes may rely on.

### Frontend passes

The frontend establishes what the source program means. Lexing and parsing build
syntax trees with source locations; name and module resolution connect uses to
definitions; type checking verifies expressions and interfaces. It also loads the
prelude and imported modules, so later stages receive a complete typed program
with diagnostics tied to the original source.

The [frontend pipeline](wyn-core/src/frontend.rs) runs these passes in order:

| Pass | Responsibility |
| --- | --- |
| Lexing (`lexer::tokenize`) | Turn source text into located tokens. |
| Parsing (`parser::parse_file`) | Build each module's AST and report imports so the complete source graph can be loaded. |
| `resolve_imports` | Replace physical source imports with syntax from the loaded module graph. |
| `elaborate_modules` | Elaborate module definitions, signatures, and applications into semantic module state. |
| `name_resolution::resolve_names` | Distinguish qualified module names from ordinary field access. |
| `resolve_resources` | Resolve resource views to concrete binding attributes. |
| `ast_const_fold::fold_constants` | Fold integer constants and expose static array dimensions before type inference. |
| `resolve_placeholders::resolve_type_placeholders` | Replace type and size placeholders with inference variables, preserving named parameters within each declaration. |
| `resolve_opens` | Resolve names imported by `open`, respecting lexical scope and reporting ambiguity. |
| `types::run::type_check` | Infer and check types, resolve identifier identities, validate stage contexts and pattern coverage, and collect warnings. |
| `ast_type_holes::reject_type_holes` / `fill_type_holes` | Report inferred types for unfinished `???` expressions, or replace them with supported typed defaults when `--fill-holes` is enabled. |

The resulting AST retains inferred types, definition schemes, resolved symbols,
and source locations for TLC lowering.

### TLC passes

Typed lambda calculus keeps functions, applications, captures, and array
operations explicit. This representation lends itself to partial evaluation,
substitution, specialization, and inlining while preserving source types and
ownership contracts. The TLC stage uses these transformations to expose array
producers and consumers and produce first-order code for the graph-based stage.

The passes below run in order. SOACs are second-order array combinators, such as
`map`, `reduce`, and `scan`; SoA means structure of arrays.

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

### Egglog passes

Egglog combines Datalog's relational reasoning with equivalence graphs (e-graphs).
This hybrid lends itself to both scheduling and arithmetic optimization: Datalog
rules derive dependencies, execution order, and storage requirements, while
e-graphs represent equivalent expressions and support exploring algebraic rewrites
through equality saturation. Wyn uses these capabilities to plan array fusion,
schedule work and storage, and simplify scalar arithmetic. Rust analyses and
builders supply facts, construct transformed bodies, and perform expression
placement.

This stage uses several cooperating representations:

- An array-operation graph groups work into execution regions and records inputs,
  captures, uses, and effects. A Screma combines scan, reduction, and map work
  in one operation, potentially producing several outputs.
- A typed expression DAG (directed acyclic graph) shares scalar syntax and
  records dependencies.
  Region parameters, results, and use sites distinguish invocations; sharing an
  expression node does not by itself share a runtime value.
- An execution and resource plan records host/device placement, kernel phases,
  buffers, lifetimes, and dispatch dependencies. It becomes executable blocks
  before the SSA handoff.

Types, bodies, and metadata live in typed arenas alongside the egglog graphs,
called sidecars. Fusion, scalar simplification, and scheduling use separate
graphs; expression insertion hands its graph directly to scalar simplification.

| Pass | Output checkpoint | Responsibility |
| --- | --- | --- |
| `egglog::from_tlc` | `Imported` | Import normalized TLC, callable bodies, types, and source ABI into typed arenas; construct Scremas for map/reduce/scan and export structural fusion facts. |
| `egglog::fuse` | `Fused` | Complete a deterministic greedy fusion plan on a persistent graph, then construct the selected bodies in the sidecar arenas. |
| `egglog::insert_expressions` | `Expressions` | Insert a separate typed expression DAG, region interfaces and uses, structured control, and execution dependencies into egglog. |
| `egglog::simplify` | `Simplified` | Fold constants and simplify scalar expressions in that graph; optionally explore algebraic rewrites with `-O`, then extract and apply replacements. |
| `egglog::place` | `Placed` | Use Rust analysis to place safe shared and loop-invariant expressions in structured regions, including SOAC captures. Memory reads and opaque calls are not speculated. |
| `egglog::schedule` | `Scheduled` | Derive execution recipes, host/device residency, storage allocation and reuse, output routes, scratch, dispatch domains, and dependencies; instantiate executable blocks. |
| `egglog::to_ssa` | SSA `Elaborated` | Lower scheduled kernels for the selected target and publish source inputs, planned resources, outputs, and host computations through the shader/runtime ABI. |

Dependency analysis works backward from results and required effects and orders
only live operations. Operation identities keep effectful executions distinct,
while placement determines where shared expressions are evaluated.

#### Fusion

Fusion is essential to making compositional array programs practical on the GPU.
Without it, a chain of operators can materialize an array and launch a kernel at
every step. Combining producers with consumers lets intermediate values stay
within an invocation and reduces memory traffic, temporary storage, and dispatch
overhead. The compiler relies on this step to turn high-level array composition
into efficient executable work.

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

#### Scheduling and publication

Scheduling turns the fused graph into an execution plan: it chooses host or device
execution, allocates or reuses storage, determines where outputs are written, and
orders dispatches. Rust then builds the selected kernels and control flow.
Scheduled executable blocks contain no SOAC operations.

The [execution recipes](wyn-core/src/egglog/schedule.egg) apply to surviving fused
operations. Maps absorbed into a reduction share its traversal; a map after a scan
runs with offset application. An eligible filtered reduction can use a masked
collective and avoid compaction altogether. Thus a source operator does not imply
its own kernel, and one fused operation can require several dispatches.

| Work | Physical execution |
| --- | --- |
| Maps | Parallel elementwise evaluation, including any fused map chain. |
| Reductions | Chunk reduction followed by an ordered workgroup reduction tree; fused array outputs can be written during chunk processing. |
| Scans | Chunk prefixes, serial combination of chunk totals, then parallel offset application and post-map work. This recipe also handles reductions fused with scans. |
| Filters | When a compacted array is needed, tiled workgroup evaluation computes predicate prefixes, writes survivors in input order, and publishes the result length. |
| Eligible indexed reductions | Parallel atomic updates, including compare/exchange where needed. |
| Eligible bucket scatter | A count-clearing dispatch followed by parallel slot reservation and writes. |
| Scatter and other work without a parallel-safe recipe | Ordered single-invocation kernels. |
| Nested array work | Local device loops within the enclosing invocation. |

Parallel recipes require compatible callback effects; otherwise execution falls
back to ordered device loops. Collective kernels preserve operand order, and empty
inputs still initialize collective identities and result lengths.

The planner derives CPU availability, bounded rematerialization costs, and scalar
kernel groups. CPU-available scalar work can run on the host; GPU-dependent work
stays on the GPU. Cheap immutable expressions can be recomputed at their consumers,
and compatible consecutive GPU scalar operations can share a kernel, storing only
results needed outside it. Mutable reads, effects, and unbounded work are excluded
from rematerialization. Allocation and reuse decisions use post-fusion liveness,
ownership candidates, and resource accesses.

Publication turns the plan into the shader interface and companion host program,
including resource accesses, dispatch dimensions, dependencies, inputs, and results.
Published pipelines currently use static dispatch sites; conditional or repeated
host dispatches remain a limitation. Direct mode retains the authored stages and
rejects programs requiring generated prepasses or intermediate host storage.

### SSA passes

Static single assignment (SSA) represents device computation as explicit
instructions and control-flow blocks. Each value is defined once, and block
parameters carry values across branches and loops. With array scheduling already
decided, this stage can clean up scalar code, place computations where they are
needed, and establish the representation invariants required by code generation.

The [shared SSA pipeline](wyn-core/src/ssa/mod.rs) runs these passes:

| Pass (`ssa::`) | Responsibility |
| --- | --- |
| `optimize` | Inline small helpers, fold constants, reuse dominating immutable expressions, and intern expressions that can safely move. |
| `place_floating` | Assign reachable floating expressions to concrete control-flow blocks. |
| `filter_reachable` | Prune functions and constants unreachable from entries. |
| `prepare_spirv` / `prepare_wgsl` | Remove dead pure instructions, publish texture-sampling requirements, and reject unresolved type representations. SPIR-V also verifies buffer layouts; WGSL first promotes constants needing addressable storage. |

`--output-mir` dumps `Elaborated` SSA before these cleanup and backend passes.
Constant folding and the normal pipeline run without `-O`; that flag enables
additional egglog algebraic rewrites. Tests can use `compile_thru_frontend`,
`compile_thru_tlc` (through TLC reachability), `compile_thru_ssa`, and
`compile_thru_spirv` to stop at shared checkpoints.

### Backends and output modes

The backends translate the prepared SSA into device code and pair it with the
published host program. They preserve the algorithm and resource plan chosen by
earlier stages while adapting types, control flow, bindings, and operations to the
target.

| Output | Selection and behavior |
| --- | --- |
| SPIR-V with Rust/WGPU | `-t spirv --target-double rust-wgpu`. The primary application backend emits a binary shader and a Rust wrapper. SPIR-V uses explicit blocks and phi nodes and retains native push constants. The wrapper requires WGPU's SPIR-V support. |
| WGSL with Rust/WGPU | `-t wgsl --target-double rust-wgpu`. Emits WebGPU shader source and a Rust wrapper using the same scheduled pipeline. Control flow becomes structured statements, branches, and loops; push constants become storage parameter blocks. |
| Direct shader output | `--direct` with either shader target. Array work stays serial inside authored shader invocations. Useful for embedding WGSL rendering stages; no compiler-created prepass or intermediate host storage is permitted. |
| WHL host output | `--target-double whl-unknown` (the default). Emits a `.wynhost` companion describing the host program for either shader format, for inspection or execution by the WHL interpreter. |

`--graphics` enables rendering vocabulary independently of these output choices.
WGSL additionally offers `--wgsl-emulate-u64` for a limited unsigned-integer subset.
See [HOST.md](HOST.md) for generated wrapper and host-program contracts.

## Project Structure

The root Rust workspace separates compiler transformations, shared infrastructure,
and host execution:

| Directory | Responsibility |
| --- | --- |
| `wyn-core/` | Compiler library: frontend, TLC, egglog, SSA, and device backends. |
| `wyn/` | Command-line compiler and artifact output. |
| `wyn-analyzer/` | Language server: diagnostics, navigation, hover, completion, and signature information. |
| `wyn-host/` | Typed host programs, shader interfaces, and WHL and Rust/WGPU emitters. |
| `wyn-host-interp/` | WHL reader and interpreter, with a WGPU execution backend. |
| `wyn-spirv/` | Typed SPIR-V builder and shared type/constant caches. |
| `wyn-base/` | Foundational arenas, interners, collections, and utilities. |
| `wyn-graph/` | Shared graph algorithms, including reachability, ordering, and dominators. |
| `wyn-module-graph/` | Package and source identities, import graphs, and source loading. |
| `wyn-package-manager/` | Package manifests, dependency materialization, and build-input preparation. |
| `wyn-diagnostics/` | Source-oriented diagnostic rendering. |

Other projects and supporting directories include:

| Directory | Contents |
| --- | --- |
| `wyn-wasm/` | WebAssembly compiler interface, built separately from the root workspace. |
| `playground/app/` | Browser playground. |
| `extra/viz/` | WGPU runner for SPIR-V/WGSL shaders and their WHL companions, including rendering and headless compute. |
| `extra/tephra/` | Vulkan runner for SPIR-V compute shaders. |
| `extra/tree-sitter-wyn/` | Tree-sitter grammar and editor bindings. |
| `extra/wyn-doc/`, `extra/mdbook-wyn-highlight/`, `docs/` | Documentation generation, syntax highlighting, and book configuration. |
| `extra/spirv-validator/`, `extra/treereduce-wyn/` | Shader validation and source-reduction tools. |
| `prelude/` | Automatically loaded standard library written in Wyn. |
| `pkg/` | Wyn library packages. |
| `testfiles/`, `tests/`, `scripts/` | Example and regression programs, integration fixtures, and build/validation scripts. |

## Usage

Build a source file or a package directory, selecting device and host output as
described above. See [Writing Wyn Packages](PACKAGES.md) for multi-file programs
and dependencies.

```bash
# SPIR-V with a Rust/WGPU wrapper
cargo run --bin wyn -- build input.wyn -t spirv --target-double rust-wgpu -o output.spv

# Direct WGSL rendering stages
cargo run --bin wyn -- build testfiles/unified_triangle.wyn -t wgsl --graphics --direct -o triangle.wgsl

# Check types, ownership, and entry interfaces without generating output
cargo run --bin wyn -- check input.wyn
```

Add `--graphics` to `check` for a rendering program. Use `--output-tlc FILE` and
`--output-mir FILE` to inspect compiler checkpoints; `--verbose` prints pass timings
to stderr. For runnable graphics and compute examples, see
[testfiles](testfiles/) and the [viz runner](extra/viz/README.md).

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

| Type family | Examples |
| --- | --- |
| Primitives | `i8`/`i16`/`i32`/`i64`, `u8`/`u16`/`u32`/`u64`, `f16`/`f32`/`f64`, `bool`; availability depends on the backend. |
| Arrays | `[N]T` for a fixed extent, `[n]T` for a named size, `[]T` for an inferred size, and `[n][m]T` for multiple dimensions. |
| Vectors and matrices | `vec2f32`, `vec3f32`, `vec4f32`, `vec2f16`, `mat2f32`, `mat3f32`, `mat4f32`. |
| Structured values | Tuples `(i32, bool)`, records `{x: f32, y: f32}`, and sum types with named constructors. |
| Functions | `T1 -> T2`; functions and lambdas may be passed to higher-order operators. |

Type annotations are optional where inference determines the type. For example,
`def identity(x) = x` is polymorphic in its argument type. `let` binds intermediate
values, `match` selects patterns, and `loop` expresses iteration with explicit
state. Modules and imports organize definitions across files and packages.
The [language specification](SPECIFICATION.md) describes the syntax and semantics
in detail.

## Current Limitations

- Patterns: nested tuple, vector, and constructor patterns are supported.
  Record destructuring and guards on `match` cases are not implemented.
- Modules: parameterized modules are supported, but explicit functor-type
  signatures are not. Module-type declarations must be at file scope.
- Host control flow: conditional or repeated GPU dispatches cannot yet be
  published through the compiler's runtime interface. This does not prevent
  branches and loops within shaders.
- Device-local storage: temporary arrays allocated within a shader invocation
  need statically known capacities; arbitrary dynamic local allocation is not
  supported.
- WGSL scalar types: native output supports `bool`, `i32`, `u32`, `f32`, and
  `f16` with device support. Other integer widths and `f64` are unavailable.
  Optional `--wgsl-emulate-u64` supports unsigned addition/subtraction, bitwise
  operations, shifts, comparisons, and `u32`/`i32` conversions. It does not emulate
  multiplication, division, remainder, exponentiation, floating-point conversions,
  signed `i64`, or 64-bit atomics.

## Design Choices

These are deliberate and not on a fix-it list:

- **No recursion.** Use `loop` or higher-order functions to express iteration on the GPU.
- **Restricted first-class functions.** Functions can be passed as arguments and
  returned from other functions, but function values are eliminated before
  execution. They cannot be stored in arrays, returned from conditional branches,
  or carried as loop parameters. These restrictions also apply to tuples and
  records containing functions. See [Higher-order Functions](SPECIFICATION.md#higher-order-functions).

## Key Dependencies

- **egglog** — Fusion, scalar optimization, and scheduling
- **nom** — Lexical parsing helpers
- **polytype** — Hindley-Milner type system
- **rspirv** — SPIR-V builder

For complete language details, see [SPECIFICATION.md](SPECIFICATION.md).
