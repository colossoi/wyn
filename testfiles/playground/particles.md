# Particles (`particles.wyn`) — build and run

A minimal boids-style particle simulator. A compute pass evolves particle
state, and the fragment stage displays the returned state directly.

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
  --size 512x512
```

`particles.viz.json` is discovered automatically next to the shader. It maps
authored result 0 of `particles` back to the authored `prev_pos` parameter on
the next frame. The configuration never names the compiler-generated output
buffer.

`pipeline` mode auto-detects interactive vs headless from the
descriptor — `particles.json` has a graphics pipeline, so a window
opens.

### What you should see

512 particles scattered across the full framebuffer at startup, then
clumping together under a soft alignment + cohesion (boids) force as the
simulation runs. There is no mouse / keyboard interaction; the sim runs
on its own.

## Pipeline shape

| Entry         | Stage    | Role                                                   |
| ------------- | -------- | ------------------------------------------------------ |
| `particles`   | compute  | Reads `prev_pos` and returns the next particle state.  |
| `vertex_main` | vertex   | Full-screen triangle.                                  |
| `main_image`  | fragment | Displays particles from the newly computed state.      |

The resource layout is documented in the comment block at the top of
`particles.wyn`; read that first when tracing where a binding flows.

## Notes

- **Spawn randomness comes from the host.** Initial positions/velocities
  are read from the `seed` buffer, which `--buffer-init seed:rng`
  fills with uniform-random `f32` in `[0, 1)`. The shader maps those draws
  into its domain, so it needs no GPU-side RNG library.
- **`scatter` lowers serially in this cut** (sequential `scatter`
  semantics); a parallel version is a pure-optimization follow-up.
