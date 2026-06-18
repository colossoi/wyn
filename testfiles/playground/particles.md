# Particles (`particles.wyn`) — build and run

A minimal boids-style particle simulator that demonstrates storage-buffer
rasterization: a compute feedback loop evolves particle state, a second
compute pass `scatter`s one texel per particle into a flat framebuffer,
and the fragment shader samples that framebuffer at O(1) per pixel.

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
  --feedback tick:prev_pos=tick_output \
  --zero-buffer fb:4194304 \
  --size 512x512
```

Flag by flag:

- `--feedback tick:prev_pos=tick_output` — wires the simulation feedback
  loop: `tick`'s `prev_pos` input reads the previous frame's
  `tick_output`. The host allocates two physical buffers and swaps which
  is bound each frame. Without this the simulation reads zeroes every
  frame and never evolves.
- `--zero-buffer fb:4194304` — allocates the framebuffer the `rasterize`
  pass scatters into: `RES*RES * vec4f32 = 512*512*16 = 4194304` bytes.
  The host zeroes it (and re-clears each frame), so moving particles
  don't leave trails. Without it `fb` is unbound.
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
| `tick`       | compute  | Pure simulation; reads `prev_pos`, writes `tick_output`. On the first frame it spawns initial state from `lib/noise`'s `fasthash` generator. |
| `rasterize`  | compute  | One pass over the particles; `scatter`s one white texel per particle into `fb` at its flattened pixel index. |
| `vertex_main`| vertex   | Full-screen triangle.                                  |
| `main_image` | fragment | Samples `fb` at the pixel's flat index (`y*RES + x`).  |

The resource layout is documented in the comment block at the top of
`particles.wyn`; read that first when tracing where a binding flows.

## Notes

- **Spawn randomness comes from `lib/noise`.** Initial positions/velocities
  are per-particle counter-based draws from `fasthash` (the PCG generator
  backing `lib/noise`). An earlier hand-rolled `fract(x*y*(x+y))` hash lost
  all f32 precision past particle ~40 and piled most of them onto pixel
  (0,0); the counter-based generator is integer math and spreads cleanly.
- **`rasterize` lowers serially in this cut** (sequential `scatter`
  semantics); a parallel version is a pure-optimization follow-up.
