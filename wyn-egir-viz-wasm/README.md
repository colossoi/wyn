# wyn-egir-viz-wasm

Dedicated WebAssembly boundary for the EGIR pass visualizer. It compiles an
editor buffer to the selected EGIR checkpoint and returns before/after
visualization models. It exposes the individual EGIR passes from reification
through resource erasure, plus aggregate semantic optimization and complete kernel planning. See the
[selector and checkpoint list](../wyn-egir-viz/README.md).

Structured snapshots retain canonical result destinations, entry-output routes,
compiler-authored semantic optimization provenance, logical resources, staged
bodies and resident flows (including the unfinished residency draft), stage-owned
recipes and scratch requirements, and scheduled and physical kernel bodies. Draft snapshots read the builder without finalizing
it, so external inputs and published outputs appear at their actual checkpoint.
Recipe metadata is structured; scratch allocation retains stage/component identity,
scheduling links emitted phases to those components, and physicalization preserves
kernel IDs. Every body scopes its node IDs; semantic IDs remain provenance.

Build the browser package with:

```sh
wasm-pack build --target web --out-dir ../wyn-egir-viz/src/wasm-pkg
```

Run native snapshot regressions with `cargo test --offline` in this directory.
After rebuilding the package, `npm test` in `../wyn-egir-viz` exercises all
selector examples through the generated WASM API.
