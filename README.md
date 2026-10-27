# Wyn

Wyn is a functional, array-centric language for GPU programming. Programs compose
transformations of values using higher-order functions, immutable data, and static
type inference. Arrays provide a common model for inputs, intermediate values,
and results. The compiler specializes away function values and turns array
operations into GPU computation, with SPIR-V and WGSL output.

## Features

- Higher-order functions and array combinators such as `map`, `reduce`, `scan`, and `filter`
- Expression-based loops with explicit iteration state
- Arrays as the primary data type, with multiple dimensions and sizes tracked by the type system
- Hindley-Milner type inference with polymorphic types
- Type holes (`???`) that report the type expected at an unfinished expression
- Structured data types, including tuples, records, and sum types, with pattern matching
- Graphics mode for shader rendering, combining rasterization and shading with array computation
- SPIR-V and WGSL code generation, with host code for resource management and execution
- Vector and matrix types optimized for GPU operations
- Uniqueness types that permit safe in-place array updates
- Modules and package imports for organizing code

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
Supported counted loops over array state become host loops that repeat the
scheduled GPU dispatches, with the iteration count computed on the CPU.

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

Frontend analysis and optimization are shared across targets. Backend preparation
adapts SSA and resource interfaces to the selected shader format before device
and host code generation.

### Frontend passes

The frontend establishes what the source program means. Lexing and parsing build
syntax trees with source locations; name and module resolution connect uses to
definitions; type checking verifies expressions and interfaces. It also loads the
prelude and imported modules, so later stages receive a complete typed program
with diagnostics tied to the original source.

The frontend pipeline runs these passes in order:

| Pass | Responsibility |
| --- | --- |
| `lexer::tokenize` | Turn source text into located tokens. |
| `parser::parse_file` | Build each module's AST; the frontend adapter reports its imports to the module loader. |
| `resolve_imports` | Replace physical source imports with syntax from the loaded module graph. |
| `elaborate_modules` | Elaborate module definitions, signatures, and applications into semantic module state. |
| `name_resolution::qualify_module_accesses` | Rewrite module field accesses as qualified identifiers. Symbol classification is completed during type checking. |
| `resolve_resources` | Assign resource bindings and resolve resource views to concrete binding attributes. |
| `ast_const_fold::fold_constants` | Fold integer constants and expose static array dimensions before type inference; retain source references for diagnostics. |
| `resolve_placeholders::resolve_type_placeholders` | Replace type and size placeholders with inference variables, preserving named parameters within each declaration. |
| `resolve_opens` | Resolve names imported by `open`, respecting lexical scope and reporting ambiguity. |
| `types::run::type_check` | Run `name_resolution::resolve_names`, infer and check types and pattern coverage, materialize the typed AST, validate stage contexts, and collect warnings. |
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
| `lower_from_ast` | Lower the typed AST to polymorphic TLC, including patterns and maps whose inputs and callback parameters already correspond; retain unified root entries and source ownership contracts. |
| `validate_ownership` | Check source consumption and aliasing rules before evaluation or inlining can erase call boundaries. This includes graphics resource ownership. |
| `partial_eval` | Discover global constants, evaluate known applications and constants, simplify residual terms, and retain computations that depend on runtime values. |
| `specialize_types` | Specialize intrinsic calls and reachable user definitions while preserving unresolved buffer regions until final stage interfaces exist. |
| `inline_unified_helpers` | Expand constants and inline array-work, graphics orchestration, callback, and structural helpers. Each iteration also folds residual constants, normalizes values, and prunes unreachable definitions. Scalar helper optimization is left to egglog. |
| `extract_stages` | Extract compute, vertex, and fragment entries and their interfaces from root orchestration, including graphics callbacks and the compute work feeding them. Ordinary roots become compute entries. |
| `pin_entry_buffers` | Resolve entry binding layouts and substitute each storage parameter's assigned binding into its buffer-region type before buffer-dependent specialization. |
| `monomorphize` | Finish specialization with concrete buffer regions, instantiating reachable user definitions by type and known array-producer representation, including the bounded capacity of filter results. |
| `canonicalize_conditional_producers` | Rewrite supported array-valued conditionals into a pointwise map with the branch inside its callback. |
| `normalize_soacs` | Lower local arrays of tuples and their operations to tuples of arrays; turn standalone `zip` into tuple construction; name SOAC application arguments and flatten let chains. |
| `float_runtime_index_nested_producers` | Move eligible runtime-indexed producers out of nested callbacks before defunctionalization, exposing the producer and gather separately. |
| `defunctionalize` | Run closure conversion/lambda lifting, higher-order-function specialization, then closure-call lowering to direct calls. |
| `fold_generated_lambdas` | Inline applications of compiler-generated lifted lambdas, prune unreachable definitions, and verify that higher-order parameters have been eliminated. |
| `apply_ownership` | Export ownership permission on values without changing the functional TLC tree. Egglog decides buffer reuse after fusion and dispatch planning; SSA commits local storage and update decisions after placement. |

Stage extraction needs unified helpers exposed before it selects entry interfaces.
Those interfaces determine buffer bindings, which must be pinned before final
specialization can distinguish helpers operating on different buffers. Array work
is exposed once, before extraction; final specialization checks that it remains
visible. Intrinsic specialization shares the definition-specialization traversal,
after type substitution, instead of walking the program separately.

Specialization, helper inlining, and conditional-producer rewriting retain the
source array shapes. SoA lowering then runs once, with concrete element types
and the helper bodies exposed. It constructs operations from lowered operands
with their target types, including binder and callback annotations; later TLC
passes preserve that layout. Storage-backed arrays keep their declared layout
so lowering does not change the shader's buffer interface.

SOAC construction, helper inlining, and SoA lowering share a binding builder that
preserves evaluation order and flattens leading lets as they are emitted.
Distribution across tuple components binds each computed operand once. Branches,
loops, and callbacks retain their own evaluation scopes; moving producers across those
boundaries remains an explicit transformation. SoA lowering completes binding
normalization in the same traversal, including expressions emitted by earlier rewrites.

Ownership information lets fusion reuse an input buffer when its contents are no
longer needed. Reuse must preserve other reads, captured values, aliases, and
returned results. Observing an array's length alone does not prevent reuse of its
contents. Both backends also use liveness and ownership analysis to share storage
between local array values and eliminate unnecessary copies.

`--output-tlc` shows TLC after entry buffer bindings are resolved. Functions may
be polymorphic and higher-order at this point; final specialization, array
normalization, and defunctionalization follow. The CLI's `check` command stops at
the same point, before the remaining optimization and backend passes.

### Egglog passes

Egglog combines Datalog's relational reasoning with equivalence graphs (e-graphs).
Wyn uses structural rules to plan array fusion, execution placement, and dispatches,
then scalar rules to inline eligible helpers and simplify arithmetic. Rust provides
source summaries, graph ordering, scope placement, and direct SSA emission.

A single traversal of TLC resolves lexical bindings and imports semantic identities,
scopes, operation inputs, effects, and dependency summaries. Fusion, placement,
and scheduling share this structural graph. They operate on array operations and
execution regions rather than individual arithmetic expressions. Original TLC
bodies and opaque identity mappings remain available for source metadata and
computations outside scalar optimization.

After scheduling, the compiler creates a native scalar graph snapshot that retains
shared identities but clears structural analysis tables. It imports only demanded
scalar expressions and eligible helper templates. Context keys keep functions,
fused callbacks, and scalar dispatch groups distinct: batching their rule execution
does not let expressions in unrelated contexts optimize against one another.

The passes below run in order:

| Pass | Responsibility |
| --- | --- |
| `egglog::from_tlc` | Walk normalized TLC once, resolve bindings, and import structural summaries with source identities and scope relationships. |
| `egglog::fuse` | Run source-summary, dependency, work-cost, and effect-order analyses, then select a deterministic greedy fusion plan and record callback composition, argument routing, and result signatures. |
| `egglog::place` | Import the fused plan, order its execution graph and reject cycles, then choose execution domains, rematerialization, host regions, and scalar dispatch groups under the selected topology policy. |
| `egglog::schedule` | Resolve shapes, traversal proofs, execution recipes, representations, storage allocation/reuse, output routes, capacity bounds, and local-storage/boundary facts. |
| `egglog::prepare_interfaces` | Resolve scalar admission, contexts, demands, and host targets, then derive physical ABI, binding, and publication dependency/order facts. |
| `egglog::optimize_with_policy` / `optimize` | Optimize and extract demanded scalar DAGs under the selected optimization policy. |
| `egglog::prepare_host` | Prepare lowering queries and lower supported host scalar expressions/captures from the selected DAG. |
| `scalar::plan_evaluation` | Build operand-availability constraints and plan safe scalar hoisting, common-branch work, and evaluation order over the selected DAG. |
| `egglog::to_ssa` | Lower functions, kernels, control flow, and host interfaces using the evaluation plan. Resolve Egglog facts for sizes, resource accesses, scalar tasks, and dispatch loops during lowering; remove interface-only entries and return shader SSA with its host interface description. |

Egglog infers minimum input-buffer sizes for inputs used only in constant prefix
slices. Other inputs need sizes from their interface or runtime.

#### Scalar optimization and computation placement

| Optimization | Behavior |
| --- | --- |
| Helper expansion | Inline eligible helpers to expose arithmetic simplifications and shared expressions. |
| Constant folding and simplification | Fold constants and simplify integer identities, constant/equal-arm choices, and aggregate projections. General reassociation and expansion are not performed. |
| Common-subexpression sharing | Extract required expressions together into a shared graph, comparing alternatives by their combined computation cost. |
| Loop-invariant code motion (LICM) and branch hoisting | Move computations to the lowest legal loop depth and identify work common to branch arms. Operands must be available, and speculative evaluation must be read-only and guaranteed to terminate without trapping. Reuse requires the computed value to dominate its uses. |

`-O` enables scalar helper inlining and a broader search for shared computations.
Constant folding, algebraic simplification, aggregate cleanup, and safety analysis
run in both modes. Structural fusion and required lowering are always enabled.

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
planner contracts groups on a persistent graph and records selected composition,
argument routing, and result signatures in egglog. Executable bodies are emitted
later during SSA lowering.

Supported cases include vertical producer/consumer fusion, horizontal fusion of
compatible independent maps/scans/reductions, and retention of shared producer
outputs. Maps can compose with filters and indexed operations; eligible filtered
reductions become masked collectives, and indexed demands can compute selected
map elements without materializing the whole array. Scan dependencies, captures,
region boundaries, and memory/effect barriers constrain these transformations.
For example, a scan followed by a map can fuse, while a reduction consuming the
scan's prefix values still requires a separate operation.

#### Scheduling and publication

Execution placement chooses host or device execution and scalar dispatch groups.
Structural scheduling then selects physical recipes, allocates or reuses storage,
determines where outputs are written, and orders dispatches. Scalar optimization
follows these decisions; SSA lowering builds the selected kernels and control flow,
eliminating SOAC operations from the emitted program.

The [execution recipes](wyn-core/src/egglog/schedule.egg) apply to surviving fused
operations. Maps absorbed into a reduction share its traversal; a map after a scan
runs with offset application. An eligible filtered reduction can use a masked
collective and avoid compaction altogether. Thus a source operator does not imply
its own kernel, and one fused operation can require several dispatches.

| Work | Physical execution |
| --- | --- |
| Maps | Parallel elementwise evaluation, including any fused map chain. |
| Reductions | Workgroup reduction trees within chunks, followed by a workgroup combine pass; fused array outputs can be written during chunk processing. |
| Scans | Workgroup prefix scans within chunks, a workgroup scan of chunk totals, then parallel offset application and post-map work. This recipe also handles reductions fused with scans. |
| Filters | When a compacted array is needed, tiled workgroup evaluation computes predicate prefixes, writes survivors in input order, and publishes the result length. |
| Eligible indexed reductions | Parallel atomic updates, including compare/exchange where needed. |
| Eligible bucket scatter | A count-clearing dispatch followed by parallel slot reservation and writes. |
| Eligible scatter | Parallel indexed writes, with a separate initialization dispatch when needed. Colliding writes must have identical values; conflicting destination reads or effectful callbacks require serial execution. |
| Work without a parallel-safe recipe | Ordered single-invocation kernels. |
| Supported counted array loops | Host-controlled repetition of the scheduled body dispatches, with setup and completion phases for array state. |
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

Supported counted array loops publish repeated dispatches with a CPU-computable
iteration count. General conditional dispatch orchestration remains unsupported.
Direct mode retains the authored stages and rejects programs requiring generated
prepasses or intermediate host storage.

### SSA passes

Static single assignment (SSA) represents device computation as explicit
instructions and control-flow blocks. Each value is defined once, and block
parameters carry values across branches and loops. Array scheduling and scalar
computation placement are decided during the Egglog stage. SSA preparation
simplifies the generated instructions and assigns concrete storage for code
generation.

The passes below run in order. Names are relative to `ssa::` unless otherwise
qualified.

| Orchestration pass | Subpass | Responsibility |
| --- | --- | --- |
| `optimize` | `storage::lower_array_update` | Make authored storage-view writes explicit during concrete representation lowering. |
| `optimize` | `materialize_dynamic_index` | Introduce and share addressable storage for dynamic indexing of fixed scalar-array values, preserving existing array views. |
| `place_floating` | `ir::schedule_floating` | Assign the generated array materializations to concrete control-flow blocks. |
| `filter_reachable` | Reachability walk | Prune function definitions unreachable from entries, counting calls only in CFG-reachable blocks. |
| `prepare_wgsl` | `promote_addressable_constants` | Promote constants needing addressable storage. |
| `prepare_*` → `simplify_values` | `eliminate_dead_pure_instructions` | Remove unused pure computations. |
| `prepare_*` → `simplify_values` | `if_conversion::run` → `convert` | Convert eligible control-flow diamonds to selects, processing inner selections first and bounding added speculative work. |
| `prepare_*` → `simplify_values` | `constant_folding::fold` | Fold generated constants and simplify constant selections. |
| `prepare_*` → `simplify_values` | `eliminate_dead_pure_instructions` | Remove computations made unused by conversion and folding. |
| `prepare_*` → `simplify_values` | `ir::eliminate_single_input_params` | Replace single-input block parameters with their dominating incoming values. |
| `prepare_*` → `local_storage::apply` | `bufferize` | For bodies with eligible local array updates: split aggregate block parameters, remove unused local loads, analyze liveness, allocate/reuse storage, validate the assignment, rewrite to explicit storage operations, and remove dead instructions/value metadata. |
| `prepare_*` → `local_storage::apply` | `ownership::apply` | Use placed SSA aliases, uses, dominance, and loop scopes to elide remaining local value-update copies. Runs for both SPIR-V and WGSL. |
| `prepare_*` | `texture_sampling::publish_texture_sampling` | Publish texture-sampling requirements. |
| `prepare_*` | `backend_validation::verify_no_abstract_types` | Reject unresolved type representations. |

`--output-mir` shows SSA before cleanup and backend preparation.

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
and dependencies. Save the example above as `input.wyn` to try these commands:

```bash
# Default SPIR-V shader and WHL host program (input.spv and input.wynhost)
cargo run --bin wyn -- build input.wyn

# SPIR-V with a Rust/WGPU wrapper
cargo run --bin wyn -- build input.wyn -t spirv --target-double rust-wgpu -o output.spv

# Direct WGSL rendering stages
cargo run --bin wyn -- build testfiles/unified_triangle.wyn -t wgsl --graphics --direct -o triangle.wgsl

# Check types, ownership, and entry interfaces without generating output
cargo run --bin wyn -- check input.wyn
```

Add `--graphics` to `check` for a rendering program. Use `--output-tlc FILE` and
`--output-mir FILE` to inspect compiler checkpoints; `--verbose` prints pass timings
to stderr. Add `-O` to `build` for scalar helper expansion and extraction search;
fusion and basic cleanup also run without it. Both commands accept
`--max-warnings N` (`0` suppresses warnings). For runnable graphics and compute examples, see
[testfiles](testfiles/) and the [viz runner](extra/viz/README.md).

## Building and Testing

Use Rust and Cargo with a native toolchain for your platform. Run these commands
from the repository root:

```bash
cargo fmt --all
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
`-Wgsl` for WGSL. SPIR-V validation requires `spirv-val` on `PATH`; WGSL
validation builds and uses `extra/viz`. Add `-TrackedOnly` to the PowerShell
script to skip local, untracked experiments, and `-Release` to use release
binaries. The Bash and Nushell variants accept `--wgsl` and `--release`.
The Bash script uses Cargo metadata to locate both binaries, respecting
`CARGO_TARGET_DIR` and Cargo configuration; it requires Python 3 to read that metadata.

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
- Host control flow: dispatch loops are supported for eligible counted loops over
  array state with CPU-computable bounds. General conditional dispatches and
  arbitrary host dispatch loops are not supported. Branches and loops within
  shaders remain available.
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
