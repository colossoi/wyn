# tree-sitter-wyn

Tree-sitter grammar and query files for the Wyn language.

## Build

Install the local JavaScript tooling from this directory:

```sh
npm ci
```

Regenerate the parser artifacts after editing `grammar.js`:

```sh
npm run generate
```

The generated files are written under `src/`, including `parser.c`,
`grammar.json`, and `node-types.json`.

Useful local checks:

```sh
npm run parse -- ../../testfiles/open_module_demo.wyn
npm run test
```

## Helix setup

Helix reads language definitions from `languages.toml`. The file can live in
the Helix config directory, or in a project-local `.helix/languages.toml`.

Example configuration:

```toml
[[language]]
name = "wyn"
scope = "source.wyn"
injection-regex = "wyn"
file-types = ["wyn"]
comment-tokens = "--"
indent = { tab-width = 2, unit = "  " }
grammar = "wyn"

[[grammar]]
name = "wyn"
source = { git = "https://github.com/colossoi/wyn", rev = "<commit-or-tag>", subpath = "extra/tree-sitter-wyn" }
```

Replace `<commit-or-tag>` with the commit hash or tag that contains the grammar
version to use.

Fetch and build the grammar with Helix:

```sh
hx --grammar fetch
hx --grammar build
```

Helix also needs query files in its runtime directory. Use `hx --health` to see
the runtime directories that your installation searches. Put these files under
`queries/wyn/` in one of those runtime directories:

```text
queries/highlights.scm -> queries/wyn/highlights.scm
queries/locals.scm     -> queries/wyn/locals.scm
```

Typical user-runtime destinations are:

```text
Linux/macOS: ~/.config/helix/runtime/queries/wyn/
Windows:     %AppData%\helix\runtime\queries\wyn\
```

Check the installation:

```sh
hx --health wyn
hx ../../testfiles/open_module_demo.wyn
```

The first command should report that the Wyn grammar is installed. The second
command should open a Wyn file with Tree-sitter highlighting.
