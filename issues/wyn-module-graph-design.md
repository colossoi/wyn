# Wyn Source Module Graph Design

Status: proposal

This document proposes a `wyn-module-graph` crate that owns the physical source
module system shared by the package manager, compiler, analyzer, and tests. It
loads a closed package plan, resolves imports, and produces an immutable graph
whose payload is supplied by the Wyn frontend.

The crate understands packages, source modules, dependency aliases, normalized
paths, source text, spans, and import edges. Wyn syntax reaches it through one
small adapter.

## Terminology

Wyn currently uses “module” for two related concepts:

- A **source module** is one imported `.wyn` file and is a node in the source
  graph.
- A **semantic module** is a language value introduced by forms such as
  `module M = { ... }`, module signatures, functors, and `open`.

`wyn-module-graph` owns source modules. `wyn-core` continues to elaborate
semantic modules because their behavior depends on Wyn syntax, types, scopes,
and name resolution.

The existing `wyn-core::semantic_modules` module implements semantic modules.
Renaming it to something such as `semantic_modules` would make this boundary
clearer once the new crate lands.

## Architectural boundary

| `wyn-module-graph` owns | `wyn-core` owns | Package manager and driver own |
| --- | --- | --- |
| Opaque package and module IDs | Lexing and parsing Wyn | Manifest and lockfile parsing |
| Package-local dependency aliases | Extracting import requests from parsed syntax | Version selection |
| Normalized package-relative paths | Semantic modules, functors, signatures, and `open` | Fetching and source verification |
| Source text and byte ranges | Visibility and name resolution | Mapping packages to verified local roots |
| Import resolution and import edges | Type checking and public API analysis | Constructing the closed package plan |
| Module caching and cycle diagnostics | Monomorphization, reachability, and code generation | Choosing the root executable module |
| Deterministic traversal and provenance | User-facing rendering of syntax errors | Filesystem implementation of source loading |

The dependency direction is:

```text
wyn-base       wyn-graph
      \         /
    wyn-module-graph
       /       \
 wyn-core    package manager
       \       /
          wyn driver
```

`wyn-module-graph` depends only on domain-independent crates. The driver composes
the package plan, source provider, and Wyn frontend adapter.

## Design principles

### Opaque session IDs, explicit stable identities

`PackageId`, `ModuleId`, and `ImportSiteId` are small integers valid within one
graph. Callers use them as opaque keys. Persisted files use stable package names,
versions, source fingerprints, and package-relative module paths.

This keeps common compiler data compact while making cache movement and arena
allocation order irrelevant to semantic identity.

### One source file per source module

Each `ModuleId` owns one source buffer. Physical spans carry that ID, while
compiler-generated syntax uses `None`:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    module: Option<ModuleId>,
    range: TextRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextRange {
    start: u32,
    end: u32,
}
```

Ranges are half-open UTF-8 byte ranges. The graph stores line starts with the
source buffer and converts byte positions to line and column only when a
diagnostic is rendered.

The lexer retains token start and end byte offsets directly. Span merging uses
the enclosing byte range and requires physical endpoints to carry the same
`ModuleId`; a generated endpoint contributes no physical range.

This makes the graph itself the source map, with `ModuleId` serving as the key
for source text and diagnostic locations.

### Package-relative paths

`ModulePath` is a normalized, UTF-8 path within one package. Its constructors
perform separator normalization, dot-segment resolution, extension handling,
and package-root confinement. Absolute cache and checkout paths stay in the
driver's source provider.

Graph diagnostics display a stable name such as:

```text
wyn/rng@v1.4.2:src/lib.wyn
```

The driver may add a local filesystem path when an editor or verbose diagnostic
needs it.

### Immutable output

The builder performs loading and graph construction. Successful construction
returns an immutable `ModuleGraph<T>`. Later compiler passes query this graph
and store their own derived data keyed by `ModuleId`.

The first implementation returns a complete graph or a structured error. This
keeps compiler control flow simple and leaves recovery policy with the analyzer.

## Package input

The package manager resolves versions and sources before module loading. It
lowers that result into a syntax-independent `PackageGraph`:

```rust
pub struct PackageGraph {
    root: ModuleKey,
    packages: IdArena<PackageId, Package>,
}

pub struct Package {
    identity: PackageIdentity,
    library_root: ModulePath,
    dependencies: Vec<Dependency>,
}

pub struct PackageIdentity {
    canonical_name: Arc<str>,
    version: Arc<str>,
    source_fingerprint: SourceFingerprint,
}

pub struct Dependency {
    alias: DependencyAlias,
    package: PackageId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModuleKey {
    pub package: PackageId,
    pub path: ModulePath,
}
```

The module crate treats version and fingerprint values as validated identity
data. SemVer ordering and MVS remain package-manager responsibilities.

`PackageGraphBuilder::new()` should allocate IDs and validate:

- one root module;
- unique canonical package identities;
- unique dependency aliases within each package;
- every dependency target belongs to the plan;
- every module path is normalized and package-relative;
- stable iteration order for packages and aliases.

The package dependency graph describes which aliases are available. Actual
source-module edges are discovered from imports.

## Source reader

The graph builder reads verified source through a narrow interface:

```rust
pub trait SourceReader {
    type Error;

    fn load(&mut self, module: &ModuleKey)
        -> Result<Arc<str>, Self::Error>;
}
```

The driver implementation holds the mapping from `PackageId` to verified local
root. An in-memory implementation supports tests and the analyzer. The graph
crate receives source text and remains independent of filesystem and network
APIs.

The graph stores each returned `Arc<str>` once. `ModuleGraph::source(ModuleId)`
and line-index methods provide the source interface used by diagnostics.

## Module parser

The module graph needs the imports contained in each source module. A frontend
adapter returns its opaque parsed payload and reports each import through the
call boundary:

```rust
pub trait ModuleParser {
    type Parsed;
    type Error;

    fn parse(
        &mut self,
        module: ModuleId,
        source: &str,
        report_import: &mut dyn FnMut(
            ImportSiteId,
            ImportTarget,
            TextRange,
        ),
    ) -> Result<Self::Parsed, Self::Error>;
}

pub enum ImportTarget {
    Local(RelativeModulePath),
    Dependency {
        alias: DependencyAlias,
        module: Option<RelativeModulePath>,
    },
}
```

A local target is resolved from the importing module's parent path. A dependency
target with `module: None` selects that package's library root; a dependency
target with a module path resolves it from the library root's parent.

The Wyn adapter performs three syntax-aware operations:

1. Parse a file into `wyn_core::parser::ParsedFile`.
2. Read the parser's source-ordered import list, including imports nested in
   semantic module expressions.
3. Translate each file-local `ImportSiteId` and decoded path into an
   `ImportTarget`.

The AST retains the same `ImportSiteId`. Later passes resolve an import with:

```rust
let target: ModuleId = graph.import_target(current_module, import_site);
```

This avoids rewriting syntax nodes with filesystem paths and keeps the module
crate independent of how Wyn spells an import.

The adapter uses Wyn's full parser once. Import reporting remains aligned with
the language grammar, while the parsed payload feeds the same syntax tree to
later compiler passes. The graph crate may use a private record while building
edges; that temporary representation is absent from the public API.

## Graph output

Successful loading produces:

```rust
pub struct ModuleGraph<T> {
    packages: PackageGraph,
    root: ModuleId,
    modules: IdArena<ModuleId, LoadedModule<T>>,
    dependency_order: Vec<ModuleId>,
    sources: SourceMap,
}

pub struct LoadedModule<T> {
    key: ModuleKey,
    syntax: T,
    imports: Vec<ImportEdge>,
}

pub struct ImportEdge {
    site: ImportSiteId,
    span: Span,
    target: ModuleId,
}
```

The public query surface should include:

```rust
impl<T> ModuleGraph<T> {
    pub fn package_graph(&self) -> &PackageGraph;
    pub fn root(&self) -> ModuleId;
    pub fn module(&self, id: ModuleId) -> Option<&LoadedModule<T>>;
    pub fn modules(&self) -> impl Iterator<Item = (ModuleId, &LoadedModule<T>)>;
    pub fn package_of(&self, id: ModuleId) -> Option<PackageId>;
    pub fn source(&self, id: ModuleId) -> Option<&str>;
    pub fn import_target(&self, from: ModuleId, site: ImportSiteId)
        -> Option<ModuleId>;
    pub fn modules_in_dependency_order(&self)
        -> impl Iterator<Item = ModuleId>;
    pub fn location(&self, span: Span) -> Result<SourceLocation, SpanError>;
    pub fn display_location(&self, span: Span)
        -> Result<impl Display + '_, SpanError>;
    pub fn snippet(&self, span: Span) -> Result<&str, SpanError>;
    pub fn erase_syntax(self) -> SourceGraph;
}
```

Fields stay private so invariants are established once by the builder. The
graph provides deterministic iteration in dependency order and source order.
Numeric IDs remain implementation details.

The core API is an in-process Rust API. Lockfile and command protocols use
separate serializable data-transfer types and lower into `PackageGraph` through
validated constructors. The graph and its materialized source reader form a
`PackagePlan`.

## Build API

The main entry point remains small:

```rust
pub struct PackagePlan<S = LocalSources> {
    package_graph: PackageGraph,
    sources: S,
}

impl<S: SourceReader> PackagePlan<S> {
    pub fn load<F: ModuleParser>(
        self,
        parser: &mut F,
    ) -> Result<ModuleGraph<F::Parsed>, BuildFailure<F::Error, S::Error>>;
}
```

The builder performs these steps:

1. Intern the root `ModuleKey` and mark it as loading.
2. Ask `SourceReader` for its source buffer.
3. Allocate its `ModuleId` before parsing so every resulting span can use it.
4. Ask `ModuleParser` for syntax and import requests.
5. Resolve each local target within the current package or each dependency
   target through the current package's alias map.
6. Reuse an existing `ModuleId` for a previously discovered `ModuleKey`.
7. Record each import edge, including its source span and `ImportSiteId`.
8. Load newly discovered targets and finish with a deterministic dependency
   order.

The builder keeps `Loading`, `Loaded`, and `Failed` states internally. A second
edge to a loaded module reuses it. An edge to a loading module produces an
ordered cycle containing the import spans along the active stack.

## Structured errors

Module-system errors carry IDs and spans, leaving final presentation to the
compiler or analyzer:

```rust
pub enum BuildError<ParseError, ProviderError> {
    Load {
        module: ModuleKey,
        trace: Box<[ImportTraceFrame]>,
        source: ProviderError,
    },
    Parse {
        module: ModuleId,
        trace: Box<[ImportTraceFrame]>,
        source: ParseError,
    },
    UnknownDependency {
        alias: DependencyAlias,
        span: Span,
        trace: Box<[ImportTraceFrame]>,
    },
    InvalidPath {
        span: Span,
        trace: Box<[ImportTraceFrame]>,
        reason: PathError,
    },
    Cycle {
        edges: Box<[ImportTraceFrame]>,
    },
}
```

`ImportTraceFrame` contains the import span and requested target. The span
already identifies the importing module. `BuildFailure` retains the closed package graph
and every source buffer loaded before failure, so its default display can name
dependency releases as `name@version:path`, attach line and column numbers, and
render “imported from” notes without exposing local cache paths.

## Wyn frontend changes

The current parser returns a whole `Program` that owns the compilation's
`NodeCounter` and `SemanticModules`. Loading several source modules requires a
file-level boundary:

```rust
pub struct ParsedFile {
    pub declarations: Vec<Declaration<ParsedFamily>>,
}

pub fn parse_file(
    module: ModuleId,
    source: &str,
    node_ids: &mut NodeCounter,
    options: CompilerOptions,
) -> Result<ParsedFile>;
```

The Wyn `ModuleParser` adapter owns a mutable reference to the compilation's
`NodeCounter` and calls `parse_file`. Compilation-wide semantic state is created
while the source closure is loaded.

The compiler boundary is:

```rust
let modules = ParsedModules::load(plan, options)?;
let typed = modules.type_check()?;
```

`ParsedModules` owns the node allocator, semantic-module environment, language
options, and compiler-provided prelude state, keeping that state inseparable
from `ModuleGraph<ParsedFile>`. After physical imports are resolved, AST
checkpoints share an `Arc<SourceGraph>`. A `CompilationFailure`
holds another cheap reference to the same graph, so semantic diagnostics retain
package identity, package-relative source paths, and source text even though
the failing pass consumed its input checkpoint. Later AST nodes retain their
`ModuleId`, and semantic symbol identities incorporate the owning `PackageId`.

## Compiler integration

1. `wyn-module-graph` owns ID arenas, paths, package graphs and plans, spans, source
   lookup, physical import resolution, and in-memory tests.
2. `Span` identifies an optional `ModuleId` and a `TextRange`.
3. The private Wyn parser produces one `ParsedFile`; `WynFrontend` reports each
   physical import with its file-local `ImportSiteId`.
4. The CLI constructs a synthetic one-package `PackagePlan` for direct source
   compilation.
5. `ParsedModules::load` produces an opaque checkpoint, and
   `ParsedModules::type_check` runs the semantic frontend.
6. Physical imports resolve through `(ModuleId, ImportSiteId)` before module
   elaboration.
7. AST checkpoints share the syntax-free `SourceGraph`, and semantic identities
   carry `PackageId` wherever cross-package collisions are possible.
8. The package manager constructs multi-package plans through the same
   validated graph and plan API.

## Test strategy

The module work has two distinct test levels. Unit tests exercise graph rules
directly. Functional tests exercise the package driver, filesystem adapter,
Wyn parser, and module graph together through the real CLI.

### Unit tests

Unit tests live beside the implementation in `wyn-module-graph`. They use an
in-memory `SourceReader` and a small fake `ModuleParser`, so a test can state
its package plan, source text, and discovered imports without constructing Wyn
syntax or touching the filesystem.

Run them with:

```text
cargo test -p wyn-module-graph --lib
```

The unit suite covers:

- validation and deterministic interning of package, module, and import-site
  IDs;
- normalization of package-relative paths and package-root confinement;
- package-local dependency alias lookup;
- two packages containing the same relative module path;
- two packages using the same dependency alias for different targets;
- a diamond import loading and parsing its shared target once;
- direct and indirect cycles with ordered import spans;
- parse and load failures retaining their complete importing chain;
- deterministic module IDs, traversal, and graph iteration;
- equivalent results from in-memory and filesystem providers; and
- a domain-independent dependency tree for the graph crate.

Any source-fetching cases use fake providers with in-memory responses. This
keeps the unit suite deterministic while still covering checksum and source
identity errors.

### Functional tests

The functional suite consists entirely of local package trees under
`tests/module-packages/`. The Cargo integration test
`wyn/tests/package_manager_functional.rs` is its test binary and launches the
real CLI through `CARGO_BIN_EXE_wyn`.

For each case, the test binary:

1. validates that all package sources are local paths;
2. copies the complete case into a fresh temporary directory;
3. invokes the package command with `--offline`;
4. checks the exit status and stable diagnostic, graph, or lockfile output; and
5. confirms that the checked-in fixture remains unchanged.

The runner treats Git, URL, and registry sources in a functional fixture as a
test-definition error. The package manager therefore receives no opportunity
to contact a network service during this suite.

The convenience scripts `scripts/test_local_packages.sh` and
`scripts/test_local_packages.ps1` run:

```text
cargo test -p wyn --test package_manager_functional -- --nocapture
```

The first functional cases should cover a single local dependency, a local
chain, a diamond, identical relative paths in different packages, aliases with
different package-local meanings, a missing alias, an import cycle, a path
escaping its package root, a major-version conflict, and deterministic repeated
resolution. Compiler execution should stop after checking or source-only
compilation so this suite remains independent of graphics hardware and external
validators.

The ready-to-copy fixture template is documented in
`tests/module-packages/README.md`. The Rust test binary and platform scripts land
with the package command they invoke.

## Research influences

- rust-analyzer separates build-system discovery from an abstract crate graph,
  uses opaque file IDs, keeps paths relative to source roots, and makes source
  text an input rather than performing I/O in semantic layers:
  [architecture](https://rust-analyzer.github.io/book/contributing/architecture.html),
  [source roots and module trees](https://rust-analyzer.github.io/book/contributing/guide.html),
  [project model](https://rust-lang.github.io/rust-analyzer/project_model/index.html).
- Go's `go/packages` exposes opaque package IDs, an import map, source files,
  and optional syntax/type payloads. The graph remains useful at several loading
  levels: [`go/packages`](https://pkg.go.dev/golang.org/x/tools/go/packages).
- Cargo's metadata format separates resolved package nodes and dependency edges
  from compiler syntax: [`cargo metadata`](https://doc.rust-lang.org/stable/cargo/commands/cargo-metadata.html).
- `codespan-reporting` demonstrates the small diagnostic-source interface:
  file identity, display name, source text, line lookup, and line ranges:
  [`Files`](https://docs.rs/codespan-reporting/latest/codespan_reporting/files/trait.Files.html).
- LLVM's source manager stores source buffers and the location that introduced
  each included buffer, which supports provenance chains:
  [`SourceMgr`](https://llvm.org/doxygen/classllvm_1_1SourceMgr.html).
- Rust's compiler source map stores source buffers, byte positions, line starts,
  and stable source identity while converting to user-facing locations on
  demand: [`rustc_span`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/),
  [`SourceFile`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_span/struct.SourceFile.html).

The common lesson is to make project structure and source identity explicit,
keep physical I/O behind an adapter, and let syntax-aware layers attach their
own payload to the resulting graph.
