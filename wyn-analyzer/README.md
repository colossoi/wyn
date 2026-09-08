# wyn-analyzer

The analyzer resolves `wyn.toml` packages through the same package manager as
the compiler. Definitions and references use compiler symbol identities and
physical source locations, including imported package files.

Supported navigation includes functions, parameters, local bindings, and folded
integer constant expressions. Reference searches inspect `.wyn` files in the
workspace (including unopened files) and loaded dependencies, deduplicate
locations, and honor `includeDeclaration`. Open editor buffers override disk
sources throughout each package graph. Editing or closing a dependency buffer
also refreshes diagnostics in open callers.

Imported functions provide hover and signature information. Package-member
completion handles incomplete expressions such as `Math.` by analyzing complete
top-level `module Math = import "pkg:math"` declarations independently.

Navigation currently requires successful type checking of the source being
queried. Workspace references skip files that fail analysis and are collected on
demand, so large workspaces can take longer. Type-only occurrences, including
constant array dimensions, do not yet retain individual source locations.

## Verify with a running server

From the repository root:

```sh
cargo build -p wyn-analyzer --locked
python3 scripts/check_analyzer_navigation.py target/debug/wyn-analyzer
```

The script starts the real server over stdio and creates temporary packages. It
checks cross-package definitions, unopened callers, different import aliases,
signatures, completion during an incomplete edit, unsaved dependency positions,
diagnostic refresh, separate same-named bindings, and folded constant references.
It uses only the Python standard library and removes its fixtures afterward.
