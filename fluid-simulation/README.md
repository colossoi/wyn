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

Build a larger specialization without editing the source:

```bash
python fluid-simulation/build.py --particles 100000 --target wgsl
python fluid-simulation/launch.py tmp/fluid-simulation/fluid.wgsl
```

The generated `.fluid.json` supplies feedback capacities. World dimensions grow
with the preset so particle spacing stays constant. `--tree-capacity` overrides
the default allocation; the `overflow` result reports insufficient spatial
capacity. `run.sh` also accepts `--particles`, `--tree-capacity`, `--target` and
`--gpu-cost-profile`.

Tree allocation follows upstream's `24 * particles / (leaf_threshold + 1)`
reserve, rounded up to whole sibling groups plus a sentinel, with at least the
default scene's 4,097 slots. The simulation uses upstream's split-only update;
merging is available but disabled in both simulation paths. A spatial overflow
turns the particles red and stops advancing them. Rebuild older specializations
to pick up the corrected capacity; the earlier 25k allocation was too small.

Set `WYN_GPU_PROFILE=1` on a viewer invocation to inspect shader compilation,
per-pass GPU timestamps and host readback waits directly.

```bash
python3 fluid-simulation/test/check.py
python3 fluid-simulation/test/check_neighbors_step.py
python3 fluid-simulation/test/spatial_check.py
python3 fluid-simulation/test/check_linked_neighbors.py
python3 fluid-simulation/test/check_renderer_path.py
python3 fluid-simulation/test/check_build.py
```

Builds both shader targets and compares a headless GPU step with a Python
reference, including interacting particles and collisions at both bounds.
On the RX 580, the complete 640×480 simulation runs at about 20 ms per frame
for 4,320 particles and 184–196 ms for 100,000 particles across SPIR-V/WGSL.
Both targets passed particle-count, finite-value, bounds and overflow checks
through the full scaling ladder. See the
[implementation and measured results](../docs/fluid-performance-implementation.md)
for the sampling method, remaining costs and validation-layer limitation.

GPU tests need access to the host GPU. Check the selected adapter before
interpreting performance results.

The checks use this checkout's release compiler and viewer by default. Set
`WYN` and `VIZ` to test alternate builds (for example, `target/debug/wyn` and
`extra/viz/target/debug/viz`). Readbacks use the host program's typed results.
