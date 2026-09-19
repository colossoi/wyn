# Fluid simulation in Wyn

A package-based port of the DFSPH physics in
[MarcVivas/fluid-simulation](https://github.com/MarcVivas/fluid-simulation),
revision `db25b1794c0d420a76436fef81d27a84d83f7742` (MIT; see LICENSE).

From the repository root:

```bash
bash fluid-simulation/run.sh
bash fluid-simulation/run.sh --skip-build --preset double-dam-break --running
bash fluid-simulation/run.sh --skip-build --max-frames 60
bash fluid-simulation/run.sh --skip-build --compile-only
```

Requires Rust/Cargo, Python 3, and a GPU supported by `extra/viz`. The launcher
follows `scripts/play.sh`, but permits generated compute stages and allocates
their scratch buffers from the compiler's descriptor. Artifacts go under
`tmp/fluid-simulation`. Extra arguments are forwarded to `viz pipeline`.

Space toggles simulation; R resets; holding the left mouse button adjusts the
orbit camera. Starts running; pass `--paused` to inspect the initial state. Presets are
`colliding-blocks`, `double-dam-break`, and `rotating-block`.

## Layout

- `src/dfsph.wyn`: poly6 density, spiky gradients, divergence and density
  pressure solves, gravity, wall repulsion, damping, integration and collision.
- `packages/presets`: the three upstream initial conditions, with reproducible
  jitter from the repository's `wyn/rng` package.
- `packages/renderer`: ray/sphere rendering, ground grid, speed colors, and
  camera math from `wyn/gfx`.
- `packages/spatial`: Hilbert keys, cornerstone octree, and neighbor lists.
- `../pkg/sort`: stable 30-bit radix sorting shared by spatial preparation.
- `src/main.wyn`: frame orchestration and position/velocity feedback.

The solver takes equal-length position/velocity arrays and an index array
`0i32..<count`. Positions store radius in `.w`; velocity `.w` is zero.
The explicit index array preserves static intermediate buffer sizes.

## Scope and differences

This is a small reference port, not the upstream million-particle implementation.
It uses exact O(N²) neighbor scans instead of the Vulkan Hilbert sort/octree and
ray/sphere rendering instead of mesh shaders. Default count is 4,320 particles;
edit `count` and `bounds` in `src/main.wyn` together for larger scenes. Rendering
also scales with particle count per pixel. No upstream performance claim applies.

The pass ordering and default one iteration per pressure solve match upstream.
The timestep is fixed at 1/90 second per rendered frame, so wall-clock playback
speed depends on frame rate. Pausing preserves state but still evaluates the
compute passes. Camera pan and wheel zoom are not implemented.

Current compiler descriptors expose generated entry names for feedback;
`fluid.viz.json` references the final position and velocity passes. The launcher
checks that these entries exist. Revisit the sidecar if changing root stage order.

## Validation

```bash
python3 fluid-simulation/test/check.py
```

Builds both shader targets and compares a headless GPU step with a Python
reference, including interacting particles and collisions at both bounds.
The 432-particle viewer was also exercised interactively after fixing its
buffer allocation and startup uniform configuration. The current 4,320-particle
configuration compiles; its interactive performance has not been measured.
