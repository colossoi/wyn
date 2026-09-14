# Wyn EGIR Pass Inspector

A Vite single-page application for comparing EGIR programs immediately before
and after selectable compiler passes. The selector follows the current
portable, parallel pipeline with generated stages enabled. Every pass has a
Wyn example in [`src/passes.ts`](src/passes.ts).

The 17 selectable checkpoints, in pipeline order, are:

| Phase | Passes (all prefixed with `egir::`) |
| --- | --- |
| Reification | `reify_soacs` |
| Semantic optimization | `eliminate_dead_semantic_operations`, `fuse_semantic_operations`, `lift_stage_uniform_values` |
| Resource and residency planning | `allocate_semantic_resources`, `resolve_residency` |
| Kernel planning | `finalize_staged_ir`, `allocate_recipe_scratch`, `build_kernel_schedule`, `physicalize_kernel_schedule` |
| Physical EGIR | `lower_soacs`, `eliminate_internal_place_calls`, `partially_inline_calls`, `materialize_dynamic_extracts`, `rewrite`, `optimize_skeleton`, `erase_resources` |

Dead-operation elimination shows one whole-program step; fusion shows at most
one legal rewrite after dead operations have been eliminated. Later checkpoints
run the complete shared optimization fixpoint first.

Allocation snapshots include the authored stage bodies in the residency draft.
Residency adds generated stages and resident flows. Finalization completes the
topology and constructs projected kernel bodies, recipes, routing, capture
dependencies, and scratch requirements inside each stage. Scratch allocation
replaces those requirements with logical resource IDs.

Schedule construction shows the generated phase bodies, dependencies, launch
domains, resource accesses, and publication routes. Physicalization assigns
resource bindings and publishes interfaces. Stage and flow IDs remain stable
through finalization; stage/component IDs link recipes across allocation and
to scheduled phase families. Kernel IDs survive physicalization, while body
nodes have distinct IDs within each projection and retain semantic provenance.

Metadata differences highlight individual structured fields, including scratch
requirements becoming resource references. The last EGIR checkpoint is resource
erasure; SSA elaboration and backend passes are outside this viewer.

The WASM API also exposes aggregate `egir::optimize_semantic_operations` and
`egir::plan` inspections. The planning aggregate starts with optimized
semantic IR and includes resource allocation through physicalization.

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
package. `npm run build:all` rebuilds Rust/WASM, checks every selector example,
and builds the web application.

After rebuilding WASM, run `npm test` to compile every selector example through
its actual browser checkpoint and verify that both panes retain executable
bodies and control flow. Native snapshot regression tests run with
`cargo test --manifest-path ../wyn-egir-viz-wasm/Cargo.toml --offline`.
