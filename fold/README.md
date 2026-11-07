# Fold — native Wyn port

Standalone native port of mrdoob’s [Codevember 2026 / 01 / Fold](https://mrdoob.github.io/codevember-2026/01/).
The original MIT source is pinned by `REVISION` and retained in `upstream/`
only as a reference and numerical oracle. No browser host, server, JavaScript
simulation host, or playground is used.

## Build and run

From the Wyn repository root:

```sh
cargo build --release -p wyn --bin wyn
python3 fold/build.py
cargo build --release --manifest-path fold/native/Cargo.toml --target-dir extra/viz/target
python3 fold/run.py
```

Use `python3 fold/run.py --size 360` for a 360 × 360 interactive window.
Resizing the window also changes the render resolution.

`run.py` opens the native Winit/WGPU window. After shader compilation finishes,
it starts a five-second OS-signal deadline and kills the entire child process
group on expiry. Builds and CPU tests have no deadlines. For a self-managed
interactive session, the executable is `extra/viz/target/release/wyn-fold`.

Drag the cloth to grab a compliant nine-particle patch. Space pauses, R resets,
and Escape quits. Native keys replace the original web settings panel:

| Keys | Setting |
| --- | --- |
| − / + | Feed speed |
| [ / ] | Stiffness |
| , / . | Thickness |
| T | Transmission on/off |
| A | Ambient occlusion on/off |
| H | Cycle roughness |
| I / O | Index of refraction |
| G | Cycle tint |
| D / E | Exposure |
| C / V | Contrast |
| S / W | Saturation |
| J / K | Temperature |

The window supports resizing and pixel density is capped at one, as upstream.

## Implementation and fidelity

`src/constraints.wyn` implements the ordered XPBD solver: stretch, shear,
bending, compliant grabbing, hashed particle contacts, recycling floor/friction,
and viewport/depth walls. It preserves four iterations, alternating edge
traversal, contact passes on iterations one and three, and upstream constants.
Native CPU execution preserves Gauss–Seidel update ordering. Contact chains are
bounded and malformed results are rejected.

`native/src/simulation.rs` ports the original 120 Hz host lifecycle in binary64:
row feeding, inlet motion, velocity damping/integration/reconstruction, gradual
retirement, moving viewport walls, and smoothed grab anchors. Constraints run in
Wyn-generated native CPU code in binary32. The renderer remains on the GPU.
`build.py` compiles Wyn to WGSL, exports scalar Metal C++ through Naga, and
`cpu.py` adapts platform vector/builtin spellings and buffer arguments for Clang.
The emitted equations, loops, conditionals, and indexed updates are retained;
there is no separately handwritten CPU constraint solver. A Clang C++20 compiler
is required (the macOS command-line developer tools provide it).

`native/src/mesh.rs` ports the Catmull–Rom surface, 24 render columns, 570 render
rows, both shell faces, area-weighted vertex normals, and six-segment rounded
rims. Birth time and retirement coverage travel with the cloth.

The native renderer has nearest-depth/normal, peeled-color/depth, refracted
beauty, ambient occlusion, denoising/grading, and antialias passes. It retains
the original camera, rainbow pigment equation, slab refraction march and binary
hit refinement, Fresnel, GGX lighting/clearcoat, ACES matrices, and display grading.
Three.js infrastructure is replaced by native shaders. Studio environment,
thin-film iridescence, screen-space occlusion, roughness filtering, and edge
antialiasing are native implementations, not byte-identical Three.js shaders.
Visual parity is for review; bitwise rendering equivalence is not claimed.

The original upstream architecture also uses CPU simulation and GPU rendering.
The earlier one-invocation GPU port took about 112 ms per step on this machine.
The native CPU path takes about 1.28 ms per step on the same folded checkpoint,
without changing the timestep, solver iteration count, or update ordering.
A 60-frame native interactive run measured approximately 55 FPS at both
360 × 360 and 900 × 900. These are short measurements on the Radeon Pro 570 iMac,
not a guarantee for all cloth states or machines.

`--gpu-solver` retains the old serial GPU execution path for diagnostics. Its
private arrays exceed what the AMD Metal compiler can compile, so `spill.py`
assigns six distinct arrays to a storage buffer without aliasing. The default
CPU path uses the original private-array kernel and needs no storage spill.

## Validation

```sh
cargo test --release --manifest-path fold/native/Cargo.toml --target-dir extra/viz/target
python3 -m unittest discover -s fold/test -p 'test_*.py'
python3 fold/test/parity.py
```

The CPU host comparison matches upstream over 90 steps including feeding,
grabbing, window resizing, and retirement. All positions, velocities, birth
times, topology, and lifecycle scalars are checked.

Native CPU comparisons, starting from identical binary32
inputs, passed these upstream fixtures (four XPBD iterations each):

| Fixture | Particles | Largest coordinate error |
| --- | ---: | ---: |
| Hanging | 156 | 3.214e-7 |
| Contacts, grab, walls | 156 | 2.407e-5 |
| Retirement | 156 | 5.166e-6 |
| Nonbinary inverse masses with contacts | 156 | 9.597e-6 |
| Full hanging mesh | 2483 | 8.165e-6 |

These are short numerical comparisons, not proof of long-run trajectory
identity: upstream arithmetic is binary64 and the cloth is chaotic. The native
solver and full renderer have also run from reset through two simulated seconds
using short checkpointed runs.

## Snapshots and checkpoints

```sh
python3 fold/run.py --headless --steps 30 --no-render --save-state /tmp/fold.json
python3 fold/run.py --headless --steps 30 --no-render --load-state /tmp/fold.json --save-state /tmp/fold.json
python3 fold/run.py --headless --steps 0 --load-state /tmp/fold.json --size 768 --snapshot /tmp/fold.png
python3 fold/run.py --load-state /tmp/fold.json
```

Every command still has the five-second simulation deadline. Checkpoints retain
positions, previous positions, velocities, feeding, birth times, grabs, viewport
walls, and retirement state. They allow longer simulation evolution without a
long individual run. `--frames N` closes the interactive app after N frames;
`--no-ao` is available for headless render diagnostics.

## Profiling

Use `--profile` for per-step solver and per-frame mesh/render timings.
Add `--gpu-solver` to compare against the original slow GPU path.
`--profile-copy` applies only to that GPU path and bypasses the equations for a transfer baseline.
See [the profiling report](profiles/README.md) for measured bottlenecks and shader inspection.
