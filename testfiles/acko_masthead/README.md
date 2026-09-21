# Acko masthead — Wyn port

A Wyn rendering of the animated ribbons in [Steven Wittens's Acko masthead](https://acko.net/).
The curve samples, palette, lighting, and endpoint shapes are based on
[Acko's implementation](https://acko.net/cache/combo.min.js?c3d6624d).
This example renders the lit scene with depth testing and an animated reveal.

## Build and run

From the repository root:

```powershell
cargo build -p wyn
cargo build --manifest-path extra/viz/Cargo.toml
node scripts/build_masthead_ends.mjs
.\target\debug\wyn.exe build testfiles/acko_masthead --graphics -t wgsl -o target/acko.wgsl
.\extra\viz\target\debug\viz.exe run target/acko.wgsl --storage-dir testfiles/acko_masthead
```

Wyn emits the shader and a sibling `.wynhost` program. Viz loads the shader you
name, interprets its WHL program, and loads the source arguments from the named
`.bin` files. The Wyn entry point supplies the index buffer and draw count.

To capture the fully revealed scene:

```powershell
.\extra\viz\target\debug\viz.exe run target/acko.wgsl --storage-dir testfiles/acko_masthead --headless --size 960x540 --push-constant time:f32=30 --max-frames 1 --dump-texture screen:target/acko.png
```

Set `time:f32=5` to inspect the reveal in progress. Without a time override,
viz advances the animation each frame and supplies the viewport resolution.

`scripts/build_scene.bat` regenerates the endpoint mesh, builds a SPIR-V shader
and WHL program in this directory, and checks the shader with `spirv-val`.
It requires Node.js, Cargo, and the Vulkan SDK tools on `PATH`.

## Wyn modules

- `scene.wyn` defines the camera, lighting, and graphics entry point. It uses
  `indexed_draw_from`, `rasterize_triangles`, and `shade_with` with the built-in
  `fragment_state` type.
- `geometry.wyn` samples the baked curve cross-sections and places the moving
  arrowheads and end caps.
- `mesh.wyn` contains the generated index count.
- `wyn.toml` selects `scene.wyn` as the package's library module.

## Endpoint geometry

Each ribbon uses an 8- or 16-vertex superellipse cross-section. Letter ends
have three extra rings that narrow along the curve's tangent to form an
arrowhead. The inner red layers use truncated tips. Other ends use a ring at
80% of the profile size and a flat cap, producing a bevel rather than a
hemisphere. The start of every ribbon uses the same beveled cap.

During the reveal, `geometry.wyn` moves each track's visible interval along its
curve. Body rings outside that interval collapse to its endpoints. Extra end
rings sample the same curve, so the caps and arrowheads follow the moving ends.
Endpoint frames are interpolated from the baked cross-sections.

## Mesh data

All `.bin` files are headerless little-endian arrays. There are 16 tracks,
67,720 vertices, and 405,456 triangle indices (135,152 triangles).

| File | Element type | Contents |
|---|---|---|
| `position.bin` | `vec4f32` | Per-vertex world position and AO factor. |
| `normal.bin` | `vec4f32` | Per-vertex normal in xyz. |
| `vertex_color.bin` | `vec4f32` | Per-vertex palette color. |
| `vert_s.bin` | `f32` | Distance along the curve relative to the track's resting start. |
| `track_meta.bin` | `vec4f32` | Per-vertex reveal speed, bend, duration, and visible length. |
| `track_vertex.bin` | `u32` | Per-vertex track, endpoint role, and profile corner. |
| `track_ranges.bin` | `vec4u32` | Per-track first sample vertex, ring count, profile count, and padding. |
| `track_shapes.bin` | Two `vec4f32` per track | Arrow width/aspect/truncation/shift, then body width/bevel depth/padding/padding. |
| `indices.bin` | `u32` | Indexed triangle list. |

The packed vertex tag uses bits 0–3 for the track, 4–7 for the endpoint role,
and 8–11 for the profile corner. Roles are 0 for body, 1–3 for start
center/bevel/outline, 4 for end outline, 5–7 for arrow base/middle/tip, and
8–9 for end bevel/center.

The first 66,808 vertices of the five position/normal/color/arc/motion files
are the committed curve samples. `scripts/build_masthead_ends.mjs` preserves
that prefix and regenerates the endpoint templates, tags, shape tables,
indices, and `mesh.wyn`. Running it repeatedly produces the same data. It
requires no downloads and does not rebake the underlying spline curves.
