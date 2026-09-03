# Writing Wyn Packages

A Wyn package is a directory containing a `wyn.toml` manifest and one or more
Wyn source files. The manifest gives the package a stable name and version,
selects its library root, and assigns local aliases to its dependencies.

Wyn packages contain source. A build starts at one root source file, follows
its imports, and compiles the complete reachable source graph as one program.
Dependency packages are therefore part of the same whole-program optimization
as the root package.

## Minimal package

A small library package has this layout:

```text
hello/
├── wyn.toml
├── src/
│   └── lib.wyn
└── test/
```

`test/` is optional space for package-specific test scaffolding. Its contents
are deliberately left to the package.

The minimal `wyn.toml` is:

```toml
manifest-version = 1

[package]
name = "example/hello"
version = "v0.1.0"
wyn = "v0.1.0"
library = "src/lib.wyn"
```

The library root is an ordinary Wyn source file:

```wyn
def twice(x: i32) i32 = x + x
```

From the Wyn repository, check the package with:

```sh
cargo run --release -p wyn -- check hello
```

When `wyn` is installed as a command, the equivalent is:

```sh
wyn check hello
```

## Manifest reference

The manifest is strict TOML. Every field shown in the minimal manifest is
required, and unknown fields are rejected.

| Field | Meaning |
| --- | --- |
| `manifest-version` | Manifest format version. The current format is `1`. |
| `package.name` | Canonical ecosystem name for the package. |
| `package.version` | Version of this package. |
| `package.wyn` | Earliest Wyn version that may build the package. Preparation rejects packages requiring a newer Wyn. |
| `package.library` | Package-relative path to the library root source file. |

Package names are slash-separated lowercase names such as `wyn/noise` or
`acme/rendering/geometry`. Components may contain ASCII lowercase letters,
digits, `-`, `_`, and `.`.

Versions use `vMAJOR.MINOR.PATCH` semantic-version syntax:

```text
v0.1.0
v1.4.2
v2.0.0-beta.1
```

Prerelease versions are supported. Build metadata is excluded from canonical
package versions.

The library path is relative to the package directory and remains confined to
that directory. Use `/` separators in manifests so the same package works on
every host.

## Multiple source files

The library root can import other files in the package:

```text
geometry/
├── wyn.toml
└── src/
    ├── lib.wyn
    └── shapes.wyn
```

In `src/lib.wyn`:

```wyn
module Shapes = import "shapes"
```

The `.wyn` extension is optional in source imports. Local import paths are
resolved relative to the importing file:

```wyn
import "utilities"
module Shapes = import "geometry/shapes"
```

An import can use `..` while the resolved file remains inside the package
directory. Imports that escape the package are rejected; dependencies provide
the package-to-package boundary.

Every source file is a module. A package's library root is the module presented
to dependents. Bind imported files to named modules when you want a qualified
interface:

```wyn
module Shapes = import "shapes"
```

An unqualified import brings declarations directly into scope:

```wyn
import "shapes"
```

Module bindings and module type ascription are the clearest way to give a
library a deliberate namespace and interface. See [MODULES.md](MODULES.md) and
the Modules section of [SPECIFICATION.md](SPECIFICATION.md) for the language's
module constructs.

## Local dependencies

Add dependencies under `[dependencies]`. Each table key is the alias used by
the importing package's source code:

```toml
manifest-version = 1

[package]
name = "example/noise"
version = "v0.3.0"
wyn = "v0.1.0"
library = "src/lib.wyn"

[dependencies]
rng = { package = "wyn/rng", version = "v0.1.0", path = "../rng" }
```

The fields of a local dependency are:

| Field | Meaning |
| --- | --- |
| Table key (`rng`) | Package-local import alias. |
| `package` | Canonical name expected in the dependency's manifest. |
| `version` | Minimum acceptable version within one major version. |
| `path` | Path from the importing package directory to the dependency package directory. |

The dependency path must be relative and must lead to a directory containing a
matching `wyn.toml`. Preparation recursively reads transitive dependencies and
checks the declared package names and versions.

For a local path dependency, a requirement of `v1.4.2` accepts a materialized
`v1.4.2` or later `v1` release. A `v2` release belongs to a different
compatibility range. The same major-version rule applies to `v0` packages.

Import the dependency by alias:

```wyn
import "pkg:rng"
```

That imports the dependency's library root. A suffix addresses another source
module relative to that root:

```wyn
module Distributions = import "pkg:rng/distributions"
```

Aliases belong to the package that declares them. A dependency's own aliases
do not become aliases of its consumers.

## Selecting what to check or build

The CLI accepts a package directory or a `.wyn` source file.

Passing a package directory selects the `package.library` file:

```sh
wyn check path/to/geometry
wyn build path/to/application --output application.spv
```

Passing a source file inside a package selects that file as the build root and
uses the nearest enclosing `wyn.toml`. This is useful for examples and other
package-local programs that need the package's dependency aliases:

```sh
wyn check path/to/geometry/examples/smoke.wyn
wyn build path/to/application/examples/demo.wyn --output demo.spv
```

A `.wyn` file outside any package is treated as a standalone source root. Local
file imports remain available relative to that file.

Use the directory or source path as the command input. `wyn.toml` supplies
metadata rather than serving as a command input.

## Building output

SPIR-V is the default target:

```sh
wyn build path/to/application --output application.spv
```

Select WGSL explicitly:

```sh
wyn build path/to/application --target wgsl --output application.wgsl
```

Programs that use the graphics pipeline vocabulary also pass `--graphics`:

```sh
wyn build path/to/application/examples/image.wyn --graphics --output image.spv
```

`wyn check` loads the same package and source graph and runs the frontend
validation without writing backend output:

```sh
wyn check path/to/application/examples/image.wyn --graphics
```

For graphics and compute programs, a successful build may also write a JSON
pipeline descriptor beside the shader output.

## GitHub dependencies and materialization

A GitHub repository specifier can provide a dependency:

```toml
[dependencies]
rng = { package = "wyn/rng", version = "v0.1.0", github = "github.com/example/wyn-rng" }
```

Until version resolution is implemented, a GitHub dependency's declared
version selects the exact repository tag with the same spelling. The example
above fetches tag `v0.1.0`. Package preparation:

1. looks for the unpacked tag in the local package cache;
2. downloads the GitHub source archive when the cache has no completed entry;
3. safely unpacks one package directory into a temporary cache location;
4. installs the completed source tree atomically; and
5. validates its `wyn.toml` package name and version before compilation.

`wyn build` and `wyn check` perform this preparation automatically. The cache
contains ordinary unpacked source trees rather than repository histories.

Set `WYN_PKG_CACHE` to choose the cache directory. The platform defaults
are:

| Platform | Default |
| --- | --- |
| Windows | `%LOCALAPPDATA%\wyn\packages` |
| macOS | `$HOME/Library/Caches/wyn/packages` |
| Other Unix | `$XDG_CACHE_HOME/wyn/packages`, or `$HOME/.cache/wyn/packages` |

Entries are addressed by GitHub owner, repository, and tag. Repeated builds use
the completed unpacked entry without another HTTP request.

## Example from this repository

[`pkg/noise/wyn.toml`](pkg/noise/wyn.toml) declares `rng` as a local dependency,
and [`pkg/noise/src/lib.wyn`](pkg/noise/src/lib.wyn) imports it with
`import "pkg:rng"`. The playground package gives a complete root program that
uses the package:

```sh
cargo run --release -p wyn -- build testfiles/playground/noise_demo.wyn --graphics --output noise_demo.spv
```

The compiler finds `testfiles/playground/wyn.toml`, prepares its local package
dependencies, follows the imports beginning at `noise_demo.wyn`, and optimizes
that complete source program together.
