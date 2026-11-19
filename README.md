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
| Egglog | Rewrite one flat executable dataflow graph, prove transformation legality, and select logical execution and physical kernel policies. |
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
| `normalize_soacs` | Lower local arrays of tuples and their operations to tuples of arrays; turn standalone `zip` into tuple construction; name SOAC application arguments and flatten let chains. |
| `defunctionalize` | Run closure conversion/lambda lifting, higher-order-function specialization, then closure-call lowering to direct calls. |
| `fold_generated_lambdas` | Inline applications of compiler-generated lifted lambdas, prune unreachable definitions, and verify that higher-order parameters have been eliminated. |
| `apply_ownership` | Finish the validated TLC checkpoint while retaining parameter consumption diets. Egglog derives reuse legality from the imported graph; SSA emission and preparation assign concrete storage. |

Stage extraction needs unified helpers exposed before it selects entry interfaces.
Those interfaces determine buffer bindings, which must be pinned before final
specialization can distinguish helpers operating on different buffers. Array work
is exposed once, before extraction; final specialization checks that it remains
visible. Intrinsic specialization shares the definition-specialization traversal,
after type substitution, instead of walking the program separately.

Specialization and helper inlining retain the source array shapes. SoA lowering then runs once, with concrete element types
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

Egglog combines relational reasoning with equivalence graphs. Wyn's executable
representation is a flat graph of named values, ordered operation occurrences,
regions, control edges, iteration domains, and memory capabilities. Scalar
expressions, array traversals, branches, loops, and accumulator updates all use
this representation. Rust operation payloads hold attributes such as opcodes and
types; operands and executable bodies live in graph relations.

Slices use explicit views. Import records `transpose` and `flatten` as
axis transforms with known source coordinates. Placement can compose those
coordinates with a preceding rectangular traversal and eliminate its
intermediate array when the shape and effect proofs permit it.

[from_tlc.rs](wyn-core/src/egglog/from_tlc.rs) imports normalized TLC once. Fusion,
placement, and scalar simplification edit that same graph. TLC remains available
for source locations and public interfaces. Derived dependency, availability,
liveness, alias, shape, and effect facts are invalidated and recomputed after
edits. Scope and effect proofs determine where computations can be shared,
inlined, moved, or fused.

The public pass order is enforced by typed checkpoints in
[egglog/mod.rs](wyn-core/src/egglog/mod.rs):

| Pass | Responsibility |
| --- | --- |
| `egglog::from_tlc` | Import typed values, operation inputs and outputs, callable/control regions, logical shapes, and memory capabilities. |
| `egglog::fuse` | Inline required helpers using structural facts, then analyze dependencies and effects; compose compatible traversals and accumulator scopes; simplify and hoist until graph edits reach a fixed point; validate memory bindings. |
| `egglog::place` / `place_with_cost_model` | Apply the topology policy and portable cost model, canonicalize and distribute eligible nested traversals, fuse explicit axis transforms, and derive host regions and loop execution choices. |
| `egglog::schedule` / `schedule_with_cost_model` | Commit logical execution choices for surviving parallel operations. The cost-model variant also accepts native subgroup support; default scheduling uses portable collectives. |
| `egglog::prepare_interfaces` | Validate memory capabilities and bindings using placement's dataflow proofs, which scheduling preserves. Final scalar simplification refreshes analysis after inlining. |
| `egglog::optimize_with_policy` / `optimize` | Simplify authoritative statements: fold constants, forward values, share common expressions, remove dead work, and perform legal code motion. `Full` additionally inlines nonrecursive helpers; `optimize` selects `Full`. |
| `egglog::prepare_host` | Recompute final proofs, select reuse and scan epilogues, and prepare dispatch boundaries, collective metadata, host captures, and region schedules. Populate physical kernel and launch policies in egglog. |
| `egglog::to_ssa` | Instantiate the prepared execution with concrete buffer identities, allocations, copies, and SSA control flow; publish host operations, dispatch dependencies, and source result layouts. |

Egglog rules own transformation and reuse legality, speculation/retry safety,
atomic and subgroup eligibility, kernel algorithm policies, and launch grouping.
Rust supplies cost calculations, applies selected graph edits, prepares execution
metadata, lowers host expressions, and constructs SSA. Concrete allocation and
coalescing use actual storage types and buffer identities during emission.

The main implementation boundaries are:

| Responsibility | Sources |
| --- | --- |
| Executable graph and edits | [memory/schema.egg](wyn-core/src/egglog/memory/schema.egg), [dataflow/ir.rs](wyn-core/src/egglog/dataflow/ir.rs), [dataflow/edits.egg](wyn-core/src/egglog/dataflow/edits.egg), [dataflow/edit.rs](wyn-core/src/egglog/dataflow/edit.rs) |
| Dependencies, scopes, liveness, and resources | [analysis/dataflow.egg](wyn-core/src/egglog/analysis/dataflow.egg), [analysis/resources.egg](wyn-core/src/egglog/analysis/resources.egg), [flow.egg](wyn-core/src/egglog/flow.egg) |
| Fusion and scalar simplification | [fusion/graph.egg](wyn-core/src/egglog/fusion/graph.egg), [fusion/mod.rs](wyn-core/src/egglog/fusion/mod.rs), [scalar/simplify.egg](wyn-core/src/egglog/scalar/simplify.egg), [memory.rs](wyn-core/src/egglog/memory.rs) |
| Shapes, placement, and physical policies | [planning.rs](wyn-core/src/egglog/planning.rs), [planning/graph.egg](wyn-core/src/egglog/planning/graph.egg), [planning/choices.egg](wyn-core/src/egglog/planning/choices.egg), [planning/legality.egg](wyn-core/src/egglog/planning/legality.egg), [planning/execution.egg](wyn-core/src/egglog/planning/execution.egg) |
| Host preparation and SSA emission | [execution.rs](wyn-core/src/egglog/execution.rs), [host.rs](wyn-core/src/egglog/host.rs), [to_ssa/entry.rs](wyn-core/src/egglog/to_ssa/entry.rs), [to_ssa/kernels.rs](wyn-core/src/egglog/to_ssa/kernels.rs) |

Rule schedules separate candidate discovery, availability/effect proofs, legality,
and selection. Choices that depend on settled costs or blocker facts run after
their prerequisites saturate. Physical policies can be parameterized by concrete
atomic support and output publication; emission reads the applicable choice.

For a quick compile-only check, run `./scripts/smoke_egglog.ps1`. It builds the
compiler and compiles the [smoke program](testfiles/egglog_smoke.wyn) with `-O` to
WGSL, checking that a compute entry was emitted. This does not execute shaders or
replace correctness gates for other language features and targets.

#### Scalar optimization and computation placement

| Optimization | Behavior |
| --- | --- |
| Helper expansion | Inline required array/control helpers during structural simplification; `-O` additionally inlines nonrecursive helpers to expose scalar work. Expansion uses isolated inline eligibility, lexical structure, and call-graph facts. Cloning and substitution read structural metadata directly; whole-program dataflow, liveness, effects, and scalar candidate discovery run afterward, when simplification needs them. |
| Constant folding and simplification | Fold supported scalar constants, integer identities, constant/equal-arm choices, lengths, and aggregate projections. General reassociation and expansion are not performed. |
| Common-subexpression sharing | Forward equivalent repeatable operations to an available definition in a compatible scope, preserving operands and evaluation order. |
| Loop-invariant code motion and branch factoring | Move eligible work when operands are available and scope, effect, and termination proofs permit it. Sharing must preserve dependencies and dominating definitions. |

Constant folding, common-expression sharing, reducing rewrites, and safety analysis
run in both scalar modes. Structural fusion and required lowering are always enabled.

#### Fusion

Fusion lets intermediate array values stay within an invocation and reduces
materialized storage and dispatches. The rules in
[fusion/graph.egg](wyn-core/src/egglog/fusion/graph.egg) reason over executable
operations, domains, loads, result clauses, uses, regions, and resource effects.
Rust applies selected edits through the shared graph mutation API, preserving
operation order and refreshing derived facts.

Compatible maps, scans, and reductions can share a traversal, including independent
outputs and retained producer results. Filter predicates can feed guarded maps or
reductions without first materializing a packed array when legality permits;
[fusion/compaction.rs](wyn-core/src/egglog/fusion/compaction.rs) applies these stream
rewrites. Accumulator scopes support indexed writes and reductions. Captures,
scan dependencies, control boundaries, and conflicting reads/writes constrain
fusion. Conditional factoring and nested traversal distribution use the same
availability and effect proofs.

#### Scheduling and publication

Logical placement and scheduling precede final scalar simplification. Host
preparation then selects physical policies and prepares dispatch/region metadata.
SSA emission constructs the kernels and concrete storage, preserving input aliases,
snapshots, live ranges, and source result grouping.

| Work | Physical execution |
| --- | --- |
| Maps | Parallel elementwise traversal, including fused map chains. |
| Reductions | Workgroup reduction trees within chunks and a combine pass; fused array outputs can be written during chunk processing. |
| Scans | Workgroup chunk prefixes, scanned chunk totals, and offset application, including eligible post-map work and retained totals. |
| Filters | Stable predicate-prefix compaction when a packed result is required; eligible guarded consumers can avoid the intermediate packed array. |
| Eligible indexed reductions | Parallel atomic updates, including compare/exchange where needed. |
| Eligible bucket scatter | Count initialization followed by parallel slot reservation and writes. |
| Eligible scatter | Parallel indexed writes, with separate initialization when needed; effect and destination-access proofs constrain eligibility. |
| Work without a parallel-safe recipe | Ordered device loops. |
| Supported counted and while array loops | Host-controlled repetition of body dispatches, carrying buffers and scalar/shape witnesses through setup, test, and completion phases. |
| Nested array work | Local device loops or distributed ranked traversals, according to legality and cost. |

Collective kernels preserve operand order, and empty inputs still publish identities
and logical lengths. Host-known scalar expressions and control can run on the CPU;
GPU-dependent values remain on the device until an orchestration decision requires
a readback after their writer. Supported branches can select dispatch sequences.
Direct mode retains authored shader invocations and keeps array work local.

[Host emission](wyn-core/src/egglog/to_ssa/entry.rs) publishes rectangular dimensions
or segmented offset buffers alongside the flat payload. Logical lengths remain
separate from allocation capacities. Shifted result copies run after their producer
dispatch, and host allocations/readbacks are ordered around the stages supplying
their sizes. Both WHL and Rust/WGPU hosts consume this published program.

### SSA passes

Static single assignment (SSA) represents device computation as explicit
instructions and control-flow blocks. Each value is defined once, and block
parameters carry values across branches and loops. Array scheduling and source-level scalar
computation placement are decided during the Egglog stage. SSA preparation
simplifies the generated instructions and assigns concrete storage for code
generation.

The passes below run in order. Names are relative to `ssa::` unless otherwise
qualified.

| Orchestration pass | Subpass | Responsibility |
| --- | --- | --- |
| `optimize` | `storage::lower_array_update` | Make authored storage-view writes explicit during concrete representation lowering. |
| `optimize` | `forward_aggregate_reads` | Forward scalar projections from their original place within a block, stopping at possible writes to preserve snapshots. |
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
to stderr. Add `-O` to `build` for additional nonrecursive scalar helper expansion;
fusion and basic cleanup also run without it. Both commands accept
`--max-warnings N` (`0` suppresses warnings). For runnable graphics and compute examples, see
[testfiles](testfiles/) and the [viz runner](extra/viz/README.md).

## Building and Testing

Use Rust and Cargo with a native toolchain for your platform. Run these commands
from the repository root:

```bash
cargo fmt --all -- --check
cargo build --workspace --all-targets
python scripts/test_units_test.py
python scripts/test_integrations_test.py
python scripts/test_units.py
python scripts/test_integrations.py
cargo test --workspace --doc
```

The unit gate enforces the three-second limit and recorded timing exceptions.
The integration gate gives every case a five-minute process-tree deadline and
stops on failure. See [test placement and timing exceptions](TESTS.md).

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
