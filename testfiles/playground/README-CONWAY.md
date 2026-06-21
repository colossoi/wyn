# Conway's Game of Life (`conway.wyn`) — build and run

GPU-resident Conway's Game of Life. The board lives on a 2-D storage
texture, one texel per cell, 1.0 = alive / 0.0 = dead. A compute pass
applies the rule per cell each frame; a full-screen fragment pass
renders the just-written board. Persistence frame-to-frame is
provided by viz's `--feedback` ping-pong machinery — the same path
`mountains.wyn` uses for its heightmap.

## What you need

- The `wyn` compiler (`cargo build --release -p wyn`).
- The `viz` runtime (`cd extra/viz && cargo build --release`).
- A GPU with Vulkan or DX12 and the `r32float` storage-texture format
  (anything modern in practice).

`spirv-val` is optional; if it's on `$PATH` you can sanity-check the
SPIR-V before launching the renderer.

## Build the SPIR-V

From the repo root:

```bash
cargo build --release -p wyn
./target/release/wyn compile testfiles/playground/conway.wyn \
  -o testfiles/playground/conway.spv
```

`wyn compile` emits both `conway.spv` and `conway.json` — the JSON
descriptor is the runtime's source of truth for bindings, dispatch
size, storage-texture format, etc.

## Run interactively

```bash
(cd extra/viz && cargo build --release)
./extra/viz/target/release/viz pipeline \
  testfiles/playground/conway.spv \
  --feedback step:prev_board=board
```

`--feedback ENTRY:READ=WRITE` tells the host that the `prev_board`
texture binding on the `step` compute pipeline reads the previous
frame's value of the `board` storage_image binding. viz allocates
two physical wgpu textures for the pair and swaps which one is
"current" each frame. Without this flag every frame reads zeroes,
the initial random seed never persists, and Conway stays dark.

`pipeline` mode auto-detects interactive vs. headless from the
descriptor — `conway.json` has a graphics pipeline (the
`vertex_main` + `display` pair), so a window opens.

The window size IS the board size. Resize the window and the grid
resizes with it. (The storage texture is `SameAsWindow` and the
compute dispatch derives from it, one thread per cell.) A 600×600
window is ~360k cells, which any modern GPU eats for breakfast.

## What you should see

For the first two frames, each cell is independently seeded to alive
with ~20% probability via a deterministic hash of its `(x, y)`. From
frame 2 onward, every cell follows the Conway rule:

- Alive cell with 2 or 3 alive neighbors → stays alive.
- Alive cell otherwise → dies.
- Dead cell with exactly 3 alive neighbors → comes alive.
- Dead cell otherwise → stays dead.

You'll see the noise quickly settle into a sparse field of still-lifes
(blocks, beehives), oscillators (blinkers, toads), and the occasional
glider migrating across the grid.

## Shader anatomy

The Wyn source has three entry points:

- **`step` (compute)** — one thread per cell. Reads its eight
  Moore-neighborhood cells from `prev_board: texture2d` via
  `texture_load`, counts alive neighbors, applies the rule, writes
  the new state with `image_store(board, [x, y], ...)`. The first
  two frames (`iFrame < 2`) skip the rule and write the hash-seeded
  random pattern instead, so both ping-pong slots start with the
  same seed.

- **`vertex_main` (vertex)** — a standard one-triangle full-screen
  vertex stage. Covers the viewport so the fragment runs for every
  pixel.

- **`display` (fragment)** — reads `tex_board: texture2d` at the
  same `(set=0, binding=0)` as the compute pipeline's storage_image.
  viz shares one physical wgpu texture across that slot, so the
  fragment sees the texel that `step` just wrote. White if alive,
  black if dead.

The previous-frame texture binding (`prev_board` at `(0, 1)`) and
the current-frame storage-image binding (`board` at `(0, 0)`) deliberately
sit at different `(set, binding)` slots — `--feedback` matches `READ`
to a `texture2d` binding and `WRITE` to a `storage_image` binding by
name within the entry, then aliases the read at one parity to the
write at the opposite parity.

## Tweaking the simulation

Cell density at startup is set by the `< 20u32` threshold in `step`:

```wyn
let alive_init = if (seed % 100u32) < 20u32 then 1.0 else 0.0 in
```

Bump it up for more chaos at the start, down for sparser fields. The
hash itself is deterministic, so the same window size always yields
the same initial board — handy for repro.

To reseed mid-run, close the window and relaunch (the texture is
zero-initialized on each launch and the `iFrame < 2` branch reseeds).
A keyboard-driven reset like `mountains.wyn`'s backspace path would
be a small extension — read the `keyboard` storage buffer and OR a
"reset" flag into the existing init branch.

## Known limitations

- **Workgroup size is `(64, 1, 1)`** — the compiler's current default
  for a 2-D storage-image-derived dispatch. A `(8, 8, 1)` workgroup
  would give better cache locality for the 8-neighbor read pattern,
  but at the scales above (sub-megapixel grids on a discrete GPU),
  the difference is invisible.
- **No persistence across launches** — the board is in GPU memory
  only. Closing the window discards it.
- **No interaction yet** — there's no mouse-paint, no
  pause/resume, no step-forward. Easy follow-ups, but conway.wyn
  keeps the scope tight: just the rule + ping-pong + render.
