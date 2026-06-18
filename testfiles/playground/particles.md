# Particles (`particles.wyn`) — build and run

A minimal boids-style particle simulator that demonstrates storage-buffer
rasterization: a single compute pass (`sim`) evolves particle state and
`scatter`s one texel per particle into a flat framebuffer, and the fragment
shader samples that framebuffer at O(1) per pixel.

## What you need

- The `wyn` compiler (`cargo build --release -p wyn`).
- The `viz` runtime (`cd extra/viz && cargo build --release`).
- A GPU with Vulkan / DX12.

## Build the SPIR-V

From the repo root:

```bash
cargo build --release -p wyn
./target/release/wyn compile testfiles/playground/particles.wyn \
  -o testfiles/playground/particles.spv
```

`wyn compile` emits both `particles.spv` and `particles.json` — the JSON
descriptor is the runtime's source of truth for bindings, dispatch
sizes, buffer lengths, etc.

## Run interactively

```bash
(cd extra/viz && cargo build --release)
./extra/viz/target/release/viz pipeline \
  testfiles/playground/particles.spv \
  --feedback sim:prev_pos=sim_output \
  --buffer-init fb:4194304:0 \
  --buffer-init seed:8192:rng \
  --size 512x512
```

Flag by flag:

- `--feedback sim:prev_pos=sim_output` — wires the simulation feedback
  loop: `sim`'s `prev_pos` input reads the previous frame's
  `sim_output`. The host allocates two physical buffers and swaps which
  is bound each frame. Without this the simulation reads zeroes every
  frame and never evolves.
- `--buffer-init fb:4194304:0` — allocates the framebuffer `sim`
  scatters into, zero-filled: `RES*RES * vec4f32 = 512*512*16 = 4194304`
  bytes. `sim` re-clears it each frame, so moving particles don't leave
  trails. Without it `fb` is unbound.
- `--buffer-init seed:8192:rng` — allocates the `seed` buffer that
  supplies the initial particle state, filled with one uniform-random
  `vec4f32` in `[0, 1)` per particle: `N * vec4f32 = 512*16 = 8192`
  bytes. On the first frame `sim` maps each `seed` entry into a starting
  position + velocity. Without it `seed` is unbound.
- `--size 512x512` — makes the render surface match `RES` so the
  fragment's `fb[y*RES + x]` lines up 1:1 with the framebuffer. The
  shader hard-codes `RES = 512`; at any other surface size the flat
  index wraps and you get duplicated / sheared copies of the image.

`pipeline` mode auto-detects interactive vs headless from the
descriptor — `particles.json` has a graphics pipeline, so a window
opens.

### What you should see

512 particles scattered across the full framebuffer at startup, then
clumping together under a soft alignment + cohesion (boids) force as the
simulation runs. There is no mouse / keyboard interaction; the sim runs
on its own.

## Pipeline shape

| Entry        | Stage    | Role                                                    |
| ------------ | -------- | ------------------------------------------------------ |
| `sim`        | compute  | Reads `prev_pos`, evolves particle state, and `scatter`s one white texel per particle into the self-cleared `fb`; returns the new state via `sim_output`. On the first frame it maps the host-seeded `seed` buffer into the initial state. |
| `vertex_main`| vertex   | Full-screen triangle.                                  |
| `main_image` | fragment | Samples `fb` at the pixel's flat index (`y*RES + x`).  |

The resource layout is documented in the comment block at the top of
`particles.wyn`; read that first when tracing where a binding flows.

## Notes

- **Spawn randomness comes from the host.** Initial positions/velocities
  are read from the `seed` buffer, which `--buffer-init seed:8192:rng`
  fills with uniform-random `f32` in `[0, 1)`. The shader maps those draws
  into its domain, so it needs no GPU-side RNG library.
- **`scatter` lowers serially in this cut** (sequential `scatter`
  semantics); a parallel version is a pure-optimization follow-up.
