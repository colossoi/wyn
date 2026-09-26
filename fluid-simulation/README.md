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
runs the generated `.wynhost` program, which allocates intermediate buffers
and schedules compute stages. The launcher supplies initial feedback storage.
Artifacts go under
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
The viewer uses Hilbert sorting, a persistent cornerstone octree, and capped
neighbor lists; the small reference fixture also retains the all-pairs solver.
Rendering uses ray/sphere intersections instead of mesh shaders.
Default count is 4,320 particles;
keep `count`, `tree_capacity`, and `bounds` in `src/main.wyn` consistent with
the feedback storage sizes in `launch.py` when changing the scene. Rendering
also scales with particle count per pixel. No upstream performance claim applies.

The pass ordering and default one iteration per pressure solve match upstream.
The timestep is fixed at 1/90 second per rendered frame, so wall-clock playback
speed depends on frame rate. Pausing preserves state but still evaluates the
compute passes. Camera pan and wheel zoom are not implemented.

`fluid.viz.json` refers to the authored `fluid` entry and its position, velocity,
and tree results. Generated stage names are managed by the host program.
Revisit the sidecar if changing the authored result order.

## Validation

```bash
python3 fluid-simulation/test/check.py
python3 fluid-simulation/test/check_neighbors_step.py
python3 fluid-simulation/test/spatial_check.py
```

Builds both shader targets and compares a headless GPU step with a Python
reference, including interacting particles and collisions at both bounds.
The 4,320-particle viewer also completed a three-frame headless smoke test at
64×48 with finite particle positions, exercising startup and cross-frame
feedback. Its interactive performance has not been measured.

The checks use this checkout's release compiler and viewer by default. Set
`WYN` and `VIZ` to test alternate builds (for example, `target/debug/wyn` and
`extra/viz/target/debug/viz`). Readbacks use the host program's typed results.
