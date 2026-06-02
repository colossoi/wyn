# Mountains (`mountains.wyn`) — build and run

Wyn port of runevision's
[Mouse-Paint Eroded Mountains](https://www.shadertoy.com/view/sf23W1)
Shadertoy. Reference GLSL lives under `extra/mountains-src/`
(extracted from `sf23W1.json`, gitignored).

## What you need

- The `wyn` compiler (`cargo build --release -p wyn`).
- The `viz` runtime (`cd extra/viz && cargo build --release`).
- A GPU with Vulkan / DX12 + the `rgba16float` storage-texture format
  (everything since ~2017 in practice).

`spirv-val` is optional; if it's on `$PATH` you can sanity-check the
SPIR-V before launching the renderer.

## Build the SPIR-V

From the repo root:

```bash
cargo build --release -p wyn
./target/release/wyn compile testfiles/playground/mountains.wyn \
  -o testfiles/playground/mountains.spv
```

`wyn compile` emits both `mountains.spv` and `mountains.json` — the
JSON descriptor is the runtime's source of truth for bindings,
dispatch sizes, storage-texture formats, etc.

## Run interactively

```bash
(cd extra/viz && cargo build --release)
./extra/viz/target/release/viz pipeline \
  testfiles/playground/mountains.spv \
  --feedback buffer_a:prev_a=out_a
```

`--feedback ENTRY:READ=WRITE` tells the host that the `prev_a` texture
binding on the `buffer_a` compute pipeline reads the previous frame's
value of the `out_a` storage_image binding. The host allocates two
physical wgpu textures for the pair and swaps which one is "current"
each frame. Without this flag Buffer A reads zeroes every frame and
your painting never persists.

`pipeline` mode auto-detects interactive vs headless from the
descriptor — `mountains.json` has a graphics pipeline, so a window
opens.

### Default behaviour

- Camera spins slowly (`TIME_CAM_SPIN = 1/60`) — fixed sun direction.
- Buffer A initializes a soft central dome on the first two frames;
  thereafter it copies its previous output forward.
- Mouse drag accumulates brush deltas into the heightmap. Shift +
  drag inverts the brush (digs valleys).
- The erosion filter (Buffer B) is currently bypassed — the rendered
  surface is the raw painted heightmap. See "Known limitations"
  below.

### Controls

| Input        | Effect                                                |
| ------------ | ----------------------------------------------------- |
| Left drag    | Paint the heightmap up                                |
| Shift + drag | Paint the heightmap down (invert brush)               |
| Backspace    | Reset to the initial dome (held-down detection)       |
| Mouse move   | No camera effect by default — camera is auto-spinning |

### What you should see

A small island in the middle of a calm sea, lit by a fixed sun from
the upper-left, with a slowly rotating camera. Painting builds up
ridges that persist across frames. Without the erosion filter on
they look smooth — that's expected for now.

## Known limitations (v1)

- **Erosion bypassed**: the full `ErosionFilter` (Buffer B) compiles
  and runs but its output is currently discarded — `Heightmap()`
  returns the raw input unchanged. The Shadertoy original toggles
  this with the Enter key; we'll wire up that keyboard read once the
  Buffer B keyboard binding lands.
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

The Wyn file declares the resource layout up top in a comment block;
read that first if you're tracing where a particular sampler /
uniform / storage texture flows.
