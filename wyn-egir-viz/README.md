# Wyn EGIR Pass Inspector

A Vite single-page application for comparing EGIR programs across selectable
compiler passes. The current checkpoints are:

- `egir::plan_logical_resources`: optimized semantic EGIR to target-independent
  logical resources, staged bodies, explicit resident flows, and external
  inputs;
- `egir::plan`: staged IR to the physical kernel DAG, including kernel-owned
  physical EGIR bodies, dependencies, dispatch domains, resource accesses, and
  entry and auxiliary-function boundaries constructed on their final physical
  value, view, or place ABI channels, with reconciled call arguments and
  results;
- `egir::optimize_semantic_operations`: segmented semantic EGIR through the
  shared dead-operation-elimination and fusion fixpoint, with compiler-authored
  operation provenance; and
- `egir::reify_soacs`: converted raw EGIR and the resulting segmented semantic
  EGIR. Conversion already owns complete entry routes; the semantic pane adds
  linked writer provenance and uniform SOAC publication/resource facts.

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
