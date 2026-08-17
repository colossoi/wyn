# Wyn EGIR Pass Inspector

A Vite single-page application for comparing EGIR programs across selectable
compiler passes. The current checkpoints are:

- `egir::optimize_semantics`: segmented semantic EGIR to optimized semantic
  EGIR, with compiler-authored operation provenance; and
- `egir::realize_outputs`: converted raw EGIR before and after output-writer
  provenance is recorded. Canonical destination-passing result bindings are
  already present in both panes because TLC-to-EGIR conversion now creates
  them directly.

The viewer's normative textual display contract is documented in
[`IR_SYNTAX.md`](IR_SYNTAX.md).

The before and after panes scroll independently. Clicking an entry, function,
or constant name brings the matching top-level definition into view in the
other pane; lower-level selections only highlight relationships.

Build the dedicated compiler WebAssembly package after Rust changes:

```sh
npm run build:wasm
```

Then start the local application:

```sh
npm run dev
```

`npm run build` type-checks and bundles the current generated WebAssembly
package. `npm run build:all` rebuilds both Rust/WASM and the web application.
