# Wyn Package Manager Design

Status: proposal

This document specifies a minimal, source-only package manager for Wyn. It is
intended to be implementable in small stages while preserving whole-program
optimization and leaving room for a future registry.

## Goals

- Make a build reproducible from a manifest, a lockfile, and immutable source.
- Keep dependency selection understandable and deterministic.
- Compile every dependency from source as part of one whole program.
- Select one major version of each canonical package for the whole build.
- Keep package fetching separate from compilation.
- Encourage compatible releases by making the compatible path the easy path.
- Use Git repositories and semantic-version tags as the initial package source.

## Initial scope

- Packages contain Wyn source and declarative metadata.
- One Git repository contains one package rooted beside `wyn.toml`.
- Dependencies form one unconditional graph.
- Semantic versions express single minimum requirements.
- A content-addressed cache stores immutable source trees.
- Whole-program compilation begins after the complete source graph is ready.

## Package model

A package release has:

- a canonical package name, such as `wyn/rng`;
- a semantic version, such as `v1.4.2`;
- an immutable source location and source revision;
- a `wyn.toml` manifest at the repository root;
- one library root, initially `src/lib.wyn`;
- zero or more executable roots, initially under `src/bin/`.

The canonical name is the package's ecosystem identity. A dependency alias is
local to the depending package and is the name used in that package's source.
For example, the package named `example/noise` may refer to `wyn/rng` using the
alias `rng`.

Every exact release has a stable `PackageIdentity` containing its canonical
name, semantic version, and locked source hash. The module graph assigns compact
`PackageId` and `ModuleId` keys for one compilation. Stable module identity is
the package identity plus normalized package-relative module path, which remains
unchanged across source-cache locations.

## Version semantics

Versions use `vMAJOR.MINOR.PATCH`, optionally followed by a semantic-version
prerelease suffix. Published versions are totally ordered using SemVer rules.

- `v0.x.y` is unstable and may contain API-breaking changes.
- `v1.x.y` and later stable releases use the usual SemVer compatibility rule.
- A patch release preserves the public API while fixing defects.
- A minor release may add compatible API.
- A stable major release may make incompatible changes.

Published version spellings consist of the three numeric components and an
optional prerelease suffix. Prereleases enter a build when a manifest explicitly
names one or an add/update command is given a prerelease option.

Each dependency requirement names one minimum version. In a manifest,
`version = "v1.4.2"` means **at least `v1.4.2` within major version 1**.

### One major version per package

The first resolver maintains one selection for each canonical package name. All
requirements for that name use the same major version.

- requirements for `wyn/rng` at `v1.2.0` and `v1.7.0` select `v1.7.0`;
- requirements for `wyn/rng` at `v1.7.0` and `v2.1.0` produce a major-version
  conflict with both dependency chains;
- resolving the conflict means moving the participating packages onto one major
  version of `wyn/rng`.

Major version zero follows the same selection algorithm. Its unstable status
means that choosing the highest `v0` minimum carries an explicit compatibility
warning.

## Minimum Version Selection

The resolver operates on a graph whose nodes are exact package releases and
whose edges are minimum-version requirements. For each canonical package name,
the build list contains the highest minimum mentioned anywhere in the reachable
requirement graph.

Conceptually:

1. Start with the root package's requirements.
2. Load the manifest for every exact release named by a reachable requirement.
3. Add that release's requirements to the graph.
4. For each canonical package name, verify one common major version and retain
   the highest required version.
5. Continue until the reachable requirement graph stops changing.
6. Validate that aliases, sources, checksums, and selected releases are
   consistent, then emit the selected build list.

The implementation uses a work queue and monotonic maps. Requirements found
through a lower version remain part of the graph after a higher version is
selected.

The requirement language consists of one minimum version. Resolution is a
monotonic graph traversal. An ordinary resolve follows the versions named in
manifests; `add` and `update` perform version discovery.

This gives an important reproducibility property: a build whose graph requires
up to `v1.8.0` continues selecting `v1.8.0` after `v1.9.0` is published. An
explicit update brings the new release into the graph.

### Compatibility is a required ecosystem rule

MVS assumes that a newer release in one stable major version can replace an
older release in that major. Wyn should reinforce that assumption:

- `wyn pkg publish --check` compares the candidate's public API with the most
  recent release in its major version;
- stable patch and minor releases fail the check when they remove or
  incompatibly change public API;
- diagnostics recommend a new major version for a breaking stable release;
- the compiler owns the canonical API representation, so checks follow Wyn's
  actual type and name-resolution rules;
- behavioral compatibility remains a maintainer responsibility.

The check should begin as an explicit publishing gate and can become a registry
policy later. It is useful even before a registry exists.

## Manifest

The initial `wyn.toml` format is deliberately small:

```toml
manifest-version = 1

[package]
name = "example/noise"
version = "v1.0.0"
wyn = "v0.1.0"
library = "src/lib.wyn"

[dependencies]
rng = { package = "wyn/rng", version = "v1.4.2", git = "https://example.invalid/wyn/rng.git" }
```

Rules:

- dependency table keys are source-level aliases;
- package names are canonical ecosystem identities;
- versions are minimum requirements;
- one canonical package name resolves to one source and one selected major;
- dependencies participate in every build of the package;
- the root lockfile controls the selected build list;
- each manifest uses the fields defined by its declared manifest format.

A future registry can map package names to Git sources while preserving the
resolver's version semantics.

For local development, a dependency entry may use `path = "../rng"` in place of
`git`. The path source uses the same package name, version, and alias fields as a
Git source.

## Lockfile and reproducibility

MVS gives stable version selection because publication alone leaves the
requirement graph unchanged. `wyn.lock` adds immutable Git commits, source
verification, and reliable offline reconstruction.

The root lockfile records:

- a lockfile format version;
- the root manifest hash;
- the complete selected build list;
- canonical package name and exact selected version for each entry;
- source URL and immutable Git commit;
- a canonical source-tree checksum;
- dependency edges and the aliases visible from each package;
- local path metadata, when a dependency uses a path source.

It is checked into version control for applications and is recommended for
libraries. The root package's lockfile governs the complete consuming build.

Build behavior:

1. If `wyn.lock` matches `wyn.toml`, use the locked commits and verify hashes.
2. If it is absent or stale, run MVS from the manifests and write a new lockfile.
3. In `--locked` mode, fail instead of changing the lockfile.
4. In `--offline` mode, fail if any locked source is missing from the cache.
5. Never accept a fetched tree whose checksum differs from the lockfile.

The lockfile is generated data and should have a stable, deterministic order so
that reviews show meaningful changes.

## Source fetching and cache

The package manager is the only component allowed to access the network. The
compiler receives a closed compile plan containing local, verified source
roots.

Package preparation always follows the same boundary, including before remote
fetching is implemented:

1. Parse dependency requirements and their declared source kinds.
2. Resolve each requirement to the source that must provide it.
3. Materialize that source as a verified local package root.
4. Read the materialized package's manifest and continue until the dependency
   graph is closed.
5. Construct `PackagePlan` only when every selected package has a local
   source root.

A local path is already materialized, so the current implementation
canonicalizes it and validates its manifest identity and version. A Git source
is retained as a source requirement but preparation reports that Git
materialization is unavailable. Later cache lookup and fetching replace that
failure without changing the compiler boundary or permitting a partially
materialized `PackagePlan`.

Fetched sources live in a content-addressed, read-only cache. Cache entries are
keyed by canonical source checksum and include the resolved commit and manifest.
The cache treats package trees as inert data and stores build output separately.

The initial Git source protocol is:

- list and fetch semantic-version tags only when an explicit add or update
  command requires discovery;
- resolve a selected tag to a full commit ID;
- verify that the checked-out manifest name and version match the requested
  release;
- hash a canonical representation of tracked package files;
- copy or materialize the verified tree into the content-addressed cache;
- reject ambiguous, malformed, or moving identities rather than guessing.

The first version should require one package at the repository root. This avoids
needing tag-prefix and subdirectory conventions before they are necessary.

## Imports and package namespaces

Package source uses dependency aliases. A representative import is:

```wyn
module Rng = import "pkg:rng"
```

The exact syntax can change with Wyn's module work, but its meaning is fixed:

- `rng` is looked up in the importing package's dependency alias table;
- the alias resolves to one locked `PackageId`;
- the remainder of the import, if any, resolves inside that package's source
  root;
- the loader confines local relative imports to their owning package;
- aliases are scoped to the depending package, allowing the same alias to refer
  to different package identities in different packages;
- compiler diagnostics display both the friendly alias and canonical package
  identity when the distinction matters.

Normal source imports contain dependency aliases and module paths. Manifests and
the resolved build graph carry semantic versions.

## Whole-program compilation

After resolution and fetching, the package manager constructs one
`PackageGraph` containing:

- the root package and selected entry point;
- every selected package's stable identity and library root module;
- each package's dependency alias map;
- the compiler-version requirement;
- enough source provenance for diagnostics and reproducibility reports.

The package manager pairs this graph with a source reader over the verified
local roots to produce the `PackagePlan` accepted by the compiler.

The syntax-light `wyn-module-graph` crate loads and parses the plan through a Wyn
frontend adapter, producing `ModuleGraph<ParsedFile>`. The compiler performs
cross-package semantic-module elaboration, name resolution, and type checking on
that graph, then runs its existing whole-program monomorphization and
optimization pipeline. Package boundaries govern namespace, provenance, and
visibility, while optimization remains global.

The crate boundary and proposed API are specified in
`issues/wyn-module-graph-design.md`.

Only the root executable's entry points and explicitly requested host ABI
exports seed whole-program reachability. Dependency declarations become
reachable through ordinary calls and references.

## Commands

The first useful command surface is:

```text
wyn pkg init
wyn pkg add <source>[@<version>] [--as <alias>]
wyn pkg remove <alias>
wyn pkg resolve
wyn pkg update [<alias>]
wyn pkg graph
wyn pkg fetch
wyn build [--locked] [--offline]
wyn check [--locked] [--offline]
```

Command semantics:

- `init` writes a minimal manifest.
- `add` discovers a release only when the version is omitted, then writes the
  chosen release as an explicit minimum and resolves the graph.
- `remove` edits the root manifest and resolves again.
- `resolve` traverses explicit manifest requirements and refreshes the lockfile.
- `update <alias>` queries available releases in that dependency's selected
  major, raises the direct minimum, and shows the build-list change.
- `update` applies the same operation to all direct dependencies.
- `graph` explains why each selected version is present and which requirement
  won.
- `fetch` fills and verifies the cache as a distinct step from compilation.

## Implementation shape

Implement resolution and fetching as a library with a thin `wyn pkg` command
layer. Suggested components are:

- `manifest`: strict parsing, normalization, and semantic validation;
- `version`: SemVer ordering and major-version validation;
- `resolve`: MVS graph construction and explanation paths;
- `source`: Git discovery and immutable revision resolution;
- `checksum`: canonical source-tree hashing;
- `cache`: transactional, content-addressed source storage;
- `lockfile`: deterministic read, validation, and atomic write;
- `plan`: construction of the closed `PackageGraph` and materialized
  `PackagePlan` consumed by module loading;
- `api_check`: compiler-backed compatibility checks;
- `cli`: user-oriented edits, plans, and diagnostics.

All manifest and lockfile writes must be atomic. Fetches should stage into a
temporary directory, verify completely, then rename into the cache. Concurrent
processes should either share a completed cache entry or wait on a narrow lock;
they observe complete cache entries.

The resolver should retain a predecessor edge for every selected minimum so
errors and `wyn pkg graph` can answer “why is this version here?” directly from
the resolution result.

## Implementation stages

### Stage 1: closed local package graph

- Land the compiler changes in the companion TODO.
- Parse local-path and Git dependency sources.
- Materialize local paths and report Git requirements at the explicit
  materialization boundary.
- Build package-aware module graphs and one compile plan.
- Preserve whole-program optimization across package boundaries.
- Land unit tests with each library component and the local-package functional
  harness with the package CLI.

### Stage 2: exact Git releases and lockfile

- Add Git-tag discovery, immutable commits, checksums, and cache storage.
- Initially require exact versions at the CLI boundary while validating all
  source and lockfile behavior.
- Add locked and offline builds.

### Stage 3: MVS and update workflow

- Interpret manifest versions as minimums.
- Resolve transitive build lists using MVS.
- Add explicit version discovery to `add` and `update` only.
- Emit requirement-chain explanations for selections and major conflicts.

### Stage 4: compatibility enforcement and publishing

- Emit canonical public API descriptions from the compiler.
- Compare candidate releases with their predecessor in the same stable major.
- Add publish checks and release-tag validation.

## Testing

Implementation uses two test levels:

- unit tests exercise manifests, versions, MVS, lockfiles, checksums, package
  plans, source providers, and module-graph construction through in-memory
  inputs and fake providers;
- functional tests exercise complete local package trees through the real Wyn
  CLI.

Functional fixtures live below `tests/module-packages/` and use local `path`
dependencies exclusively. The runner verifies that restriction before invoking
Wyn. The Cargo integration test `wyn/tests/package_manager_functional.rs`
serves as the test binary. It copies each fixture to a temporary directory,
invokes `CARGO_BIN_EXE_wyn`, and checks status plus stable diagnostics, graph
output, or lockfile contents. Once Stage 2 introduces network-capable sources,
the runner also passes `--offline`.

The convenience scripts `scripts/test_local_packages.sh` and
`scripts/test_local_packages.ps1` invoke that test binary through Cargo. The
test scaffold and rules for adding cases are in
`tests/module-packages/README.md`.

## Acceptance criteria for the first complete release

- Two packages can depend on different compatible minimums and deterministically
  select the higher one.
- Publishing a newer unrequested release leaves a normal build unchanged.
- Requirements for different majors of one canonical package produce a clear
  conflict with both dependency chains.
- A locked build reconstructs and verifies the same source graph on a clean
  machine.
- An offline locked build succeeds solely from verified cache entries.
- Resolution and fetch treat dependency source as inert data.
- Duplicate imports share one module identity, and import cycles produce an
  explicit diagnostic.
- Diagnostics identify the package, version, file, and dependency chain
  involved.
- Unused dependency code is removed by whole-program reachability.
- Stable patch and minor release checks detect incompatible public API removal
  or change.
