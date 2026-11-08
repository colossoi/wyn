# Fold — standalone Wyn port

Port of mrdoob’s [Codevember 2026 / 01 / Fold](https://mrdoob.github.io/codevember-2026/01/).
The MIT source is pinned by `REVISION` and retained in `upstream/` as a reference
and numerical oracle.

All application logic is authored in Wyn. The existing Viz binary runs the
compiler-generated shaders and host program unchanged. There is no Fold Rust
application, browser host, JavaScript simulation host, or playground.

## Build and run

From the Wyn repository root:

```sh
cargo build --release -p wyn --bin wyn
python3 fold/build.py
python3 fold/run.py --size 360
```

The launcher expects the existing `extra/viz/target/release/viz` binary.
`--size` accepts a square size or `WIDTHxHEIGHT`. Additional arguments pass
through to Viz, for example `--headless --max-frames 1`.

Launches default to a five-second process-group SIGKILL deadline, including Viz
startup. Use `--deadline 30` for the user-authorized longer startup budget.
Compilation through `build.py` is separate and has no deadline.
Python only compiles and launches; it contains no Fold simulation or rendering
logic. The feedback configuration carries an opaque Wyn-authored state array.

## Source

- `src/main.wyn`: application entry, feedback, input and frame ordering.
- `src/simulation.wyn`: state, fixed stepping, feeding, integration, topology,
  retirement, moving walls, and grabbing.
- `src/constraints.wyn`: sequential Gauss–Seidel XPBD, alternating edge traversal,
  four iterations, grabbing, hashed contacts, floor friction and walls.
- `src/geometry.wyn`: Catmull–Rom interpolation, area-weighted normals, shell,
  rounded rims, topology and picking.
- `src/camera.wyn`: camera, projection and viewport policy.
- `src/settings.wyn`: defaults and keyboard controls.
- `src/material.wyn` and `src/render.wyn`: lighting, refraction, depth peeling,
  occlusion, grading, antialiasing and render-pass ordering.

Space pauses and R resets. Dragging grabs the cloth. Other controls are defined
in `settings.wyn`: +/- feed, [] stiffness, comma/period thickness, T transmission,
A occlusion, H roughness, I/O IOR, G tint, D/E exposure, C/V contrast, S/W saturation,
and J/K temperature.

## Validation status

The all-Wyn application compiles and its generated WGSL passes Naga validation.
At master `912ddff3`, scalar-output storage packing resolves the Metal binding
limit; the reduced reproducer runs successfully through unchanged Viz.
Fold reaches Metal compute-pipeline compilation. With the authorized 30-second
deadline, Metal reports `Compiler encountered an internal error` compiling
`fold__compute_0_compute`. A reduced initialization-only shader reproduces the
same error (`test/metal_array_state_init.wyn`). No simulation frame has executed.
Runtime validation remains incomplete.

Earlier native-driver performance and
numerical measurements apply to the removed mixed Rust/Wyn implementation,
not this application. Historical reports remain in `profiles/`.

The port uses binary32 arithmetic; upstream simulation uses binary64. Exact
long-run numerical identity is not claimed. Three.js infrastructure is replaced
by authored Wyn rendering passes; studio environment, iridescence, occlusion,
roughness filtering and edge antialiasing require visual review.

Compiler reproducers under `test/` have issue comments at their tops. Upstream
fixture generation is retained for numerical comparisons. The old native-driver
parity launcher has not yet been migrated to the generated full application.
