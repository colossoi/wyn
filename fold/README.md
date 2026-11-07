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
A serial dispatch preserves Gauss–Seidel update ordering. Contact chains are
bounded and malformed results are rejected.

`native/src/simulation.rs` ports the original 120 Hz host lifecycle in binary64:
row feeding, inlet motion, velocity damping/integration/reconstruction, gradual
retirement, moving viewport walls, and smoothed grab anchors. Constraints run in
the actual Wyn GPU kernel in binary32, not in a substitute native solver.

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

The AMD Metal compiler rejects the solver’s approximately 174 KiB of private
arrays. `spill.py` allocates six distinct arrays in one explicit storage buffer
instead, preserving declaration-time zero initialization and all scalar access
order. It rejects value copies and unexpected layouts; it introduces no array
aliases. `build.py` also audits that contact hashes are built outside neighbor
queries. This is allocation lowering after Wyn compilation, not a change to the
constraint equations.

## Validation

```sh
cargo test --release --manifest-path fold/native/Cargo.toml --target-dir extra/viz/target
python3 -m unittest discover -s fold/test -p 'test_*.py'
python3 fold/test/parity.py
```

The CPU host comparison matches upstream over 90 steps including feeding,
grabbing, window resizing, and retirement. All positions, velocities, birth
times, topology, and lifecycle scalars are checked.

Actual GPU comparisons on AMD Radeon Pro 570, starting from identical binary32
inputs, passed these upstream fixtures (four XPBD iterations each):

| Fixture | Particles | Largest coordinate error |
| --- | ---: | ---: |
| Hanging | 156 | 3.214e-7 |
| Contacts, grab, walls | 156 | 2.371e-5 |
| Retirement | 156 | 2.523e-6 |

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
