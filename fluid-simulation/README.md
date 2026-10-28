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

The launcher builds with `--subgroups`, enabling native GPU scan/reduction arithmetic.
It needs a GPU with subgroup support; omit that compiler flag for the portable
workgroup implementation. Direct tiled neighbor output is enabled in both modes.


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
- `packages/renderer`: culled, compacted sphere impostors using indexed
  indirect instancing, sphere depth, speed colors, and `wyn/gfx` camera math.
- `packages/spatial`: Hilbert keys, cornerstone octree, and neighbor lists.
- `../pkg/sort`: stable four-bit radix sorting (`Sort.FourBit`) for particles and live tree nodes.
- `src/main.wyn`: frame orchestration and position/velocity feedback.

The solver takes equal-length position/velocity arrays and an index array
`0i32..<count`. Positions store radius in `.w`; velocity `.w` is zero.
The explicit index array preserves static intermediate buffer sizes.

## Scope and differences

This is a small reference port, not the upstream million-particle implementation.
The viewer uses Hilbert sorting, a persistent cornerstone octree, and capped
neighbor lists; the small reference fixture also retains the all-pairs solver.
Rendering uses indexed instanced quads instead of task/mesh shaders.
Default count is 4,320 particles;
keep `count`, `tree_capacity`, and `bounds` in `src/main.wyn` consistent with
the feedback storage sizes in `launch.py` when changing the scene. Rendering
uses four vertices per visible particle. No upstream performance claim applies.

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
python3 fluid-simulation/test/check_linked_neighbors.py
python3 fluid-simulation/test/check_renderer_path.py
```

Builds both shader targets and compares a headless GPU step with a Python
reference, including interacting particles and collisions at both bounds.
Both backends completed four headless frames at 320×240 with identical
positions and velocities for 4,320 particles. A 120-frame WGSL run through
block contact retained finite positions and velocities and kept particles
inside the simulation bounds. This does not establish 100k-particle performance.

GPU tests need access to the host GPU. A sandbox without GPU devices falls
back to llvmpipe, where the 4097-slot octree can stall during pipeline creation.
Check the selected adapter before interpreting that timeout as a compiler failure.

The checks use this checkout's release compiler and viewer by default. Set
`WYN` and `VIZ` to test alternate builds (for example, `target/debug/wyn` and
`extra/viz/target/debug/viz`). Readbacks use the host program's typed results.
