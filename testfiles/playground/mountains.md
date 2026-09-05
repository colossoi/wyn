# Mountains (`mountains.wyn`) — build and run

Wyn port of runevision's
[Mouse-Paint Eroded Mountains](https://www.shadertoy.com/view/sf23W1)
Shadertoy. Reference GLSL lives under `extra/mountains-src/`
(extracted from `sf23W1.json`, gitignored).

## What you need

- The `wyn` compiler (`cargo build --release -p wyn`).
- The `viz` runtime (`cd extra/viz && cargo build --release`).
- A GPU with Vulkan or DX12 support.

`spirv-val` is optional; if it's on `$PATH` you can sanity-check the
SPIR-V before launching the renderer.

## Build the SPIR-V

From the repo root:

```bash
cargo build --release -p wyn
./target/release/wyn build testfiles/playground/mountains.wyn \
  --graphics --direct \
  -o testfiles/playground/mountains.spv
```

`wyn build` emits both `mountains.spv` and `mountains.json` — the
JSON descriptor is the runtime's source of truth for bindings,
dispatch sizes, storage-texture formats, etc.

## Run interactively

```bash
(cd extra/viz && cargo build --release)
./extra/viz/target/release/viz pipeline \
  testfiles/playground/mountains.spv
```

`pipeline` mode auto-detects interactive vs headless from the
descriptor — `mountains.json` has a graphics pipeline, so a window
opens.

### Default behaviour

- Camera spins slowly (`TIME_CAM_SPIN = 1/60`) with a fixed sun direction.
- A radial island envelope and three octaves of derivative noise produce the
  height field directly in the fragment shader.
- Additional procedural noise supplies breakup detail for terrain and water.
- The example is self-contained: it needs no host textures or feedback state.

### Controls

| Input        | Effect                                                |
| ------------ | ----------------------------------------------------- |
| Mouse move   | Moves the retained brush-radius cursor                |
| Mouse button | Shows the cursor rings over the terrain               |

### What you should see

A small procedural island in the middle of a calm sea, lit by a fixed sun
from the upper-left, with a slowly rotating camera.

## Known limitations (v1)

- **No editable feedback terrain**: the original Shadertoy uses several
  cross-frame buffer passes for mouse painting and erosion. This playground
  version evaluates a static procedural terrain so direct compilation remains
  a single authored graphics pipeline.
- **Erosion helpers retained but not integrated**: the ported erosion routines
  remain in the source for reference, but the direct height sampler uses the
  cheaper three-octave field.
- **No iFrameRate uniform**: paint accumulation assumes ~60 fps. At
  very different frame rates the brush strength per stroke will look
  off; cosmetic only.
- **Dither texture replaced**: the original samples a noise texture
  for the per-pixel dither; we skip that and rely on the 16-bit
  framebuffer's natural quantisation.
- **No camera mouse-control**: the original's
  `CAMERA_MOUSE_CONTROL` define is omitted; mouse always controls
  the brush.

## Source layout

| File                                     | Role                              |
| ---------------------------------------- | --------------------------------- |
| `testfiles/playground/mountains.wyn`     | The port itself (single file).    |
| `testfiles/playground/mountains.md`      | This file.                        |
| `extra/mountains-src/{common,buffer_a,buffer_b,buffer_c,image}.glsl` | Reference GLSL extracted from the Shadertoy JSON. Gitignored. |

The Wyn file is self-contained and publishes only its Shadertoy-style uniforms
and render target.
