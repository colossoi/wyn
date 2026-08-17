# wyn-egir-viz-wasm

Dedicated WebAssembly boundary for the EGIR pass visualizer. It compiles an
editor buffer to the selected EGIR checkpoint and returns before/after
visualization models. It currently exposes semantic optimization (including
compiler-authored operation provenance) and raw-EGIR output-writer
realization. Structured snapshots retain canonical result destinations and
entry-output route metadata.

Build the browser package with:

```sh
wasm-pack build --target web --out-dir ../wyn-egir-viz/src/wasm-pkg
```
