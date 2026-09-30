# wyn/text

SDF, MSDF and MTSDF text primitives for Wyn, with a bundled **Aileron Regular
0.102** font atlas. Aileron is a widely distributed sans-serif by Sora Sagano
(dot colon), released under **CC0**. The font, license, atlas, metrics and an
offline regeneration script are included. No compiler or renderer modifications
are required.

## Use

Add a dependency to your application's `wyn.toml` (adjust the relative path):

```toml
[dependencies]
text = { package = "wyn/text", version = "v0.1.0", path = "../text" }
```

```wyn
import "pkg:text"          -- font-independent shader / layout functions
import "pkg:text/aileron"  -- optional bundled font metrics
```

The font module also imports the core, so importing only `pkg:text/aileron`
is sufficient when using the bundled font.

Upload `assets/aileron-mtsdf.png` as **Rgba8Unorm**, preserving all four
channels. This is numeric distance data: **do not use an sRGB texture format,
premultiply alpha, discard alpha, flip rows, or apply color management**.
Use linear filtering, clamp-to-edge, and LOD 0. Avoid ordinary color mipmaps.

The 516×516 atlas has 48 texels per em and a full distance range of 8 texels.
RGB contains MSDF; alpha contains an independently generated true SDF. Both
use the same glyph bounds, so switching algorithms does not change layout.
The decoded zero is at 0.5, with positive distance inside.

```wyn
-- Call in a fragment shader; `linear_sampler` must use linear filtering.
def text_color(atlas: texture2d, linear_sampler: sampler, uv: vec2f32) vec4f32 =
  let range = text.screen_range_fragment(
    text.unit_range(aileron.px_range, aileron.atlas_size), uv) in
  let sample = texture_sample(atlas, linear_sampler, uv, 0.0f32) in
  let opacity = text.fill(text.msdf.distance(sample.xyz), range, 0.0f32) in
  text.straight(@[1.0f32, 1.0f32, 1.0f32, 1.0f32], opacity)
```

For true SDF with the bundled texture, replace `text.msdf.distance(sample.xyz)`
with `text.sdf.distance(sample.w)`. For a separate single-channel SDF texture,
pass its red channel instead. Interpolate the three channels before taking
their median. The `text.sample_bilinear` fallback performs four clamped texel
loads when the host cannot provide a linear sampler; the demo uses this because
the current `viz` runner supplies nearest-neighbor samplers.

## API

| Function | Purpose |
| --- | --- |
| `sdf.distance(channel)` | Decode a single normalized SDF channel |
| `msdf.distance(rgb)` | Median reconstruction of three interpolated MSDF channels |
| `unit_range(px_range, atlas_size)` | Atlas-relative distance range |
| `screen_range_2d(px_range, font_size, atlas_em_size)` | Uniform, axis-aligned scale |
| `screen_range(unit_range, uv_dx, uv_dy)` | Analytic derivative form, any stage |
| `screen_range_fragment(unit_range, uv)` | Fragment derivatives, including rotated/perspective text |
| `coverage(distance_px)` | One-pixel linear antialiasing ramp |
| `fill(distance, screen_range, weight_px)` | Fill with optional outward weight adjustment |
| `outline(distance, screen_range, width_px)` | Exterior outline coverage |
| `straight(color, coverage)` | Straight alpha for Wyn `#source_over` / WGPU `ALPHA_BLENDING` |
| `premultiply(color, coverage)` / `over(fg, bg)` | Premultiplied output / premultiplied source-over |
| `quad_vertex(glyph, baseline, font_size, vertex)` | Six triangle-list vertices; pixel position and UV |
| `pixel_to_clip(position, viewport, depth)` | Top-down pixels to WebGPU/Vulkan clip coordinates |
| `layout(steps, font_size, line_height)` | Segmented scan of advances and line breaks |
| `sample_bilinear(atlas, dimensions, uv)` | Sampler-independent LOD-0 linear interpolation |

All functions above are in `text`. `glyph` contains advance in em units,
plane bounds `(left, top, right, bottom)` relative to the baseline, and UV bounds
in the same order. Plane Y, pixel Y and image rows all increase downward.
`quad_vertex` returns degenerate geometry for a space or line break. Its two
triangles become counterclockwise after `pixel_to_clip`.

`aileron.glyph(codepoint)` looks up metrics, with `?` for missing characters.
`aileron.has_glyph` reports actual atlas coverage (including space). `glyph_count`,
`atlas_size`, `atlas_em_size`, `px_range`, `ascender`, `descender` and `line_height`
describe the bundled face. The ascender is negative in top-down coordinates.

`aileron.layout(codepoints, font_size)` returns one baseline offset per Unicode
codepoint as `[n]vec2f32`. Supply Unicode scalar values as `i32`, not UTF-8 bytes.
LF resets X and advances by the font line height; CR is ignored (CRLF works);
TAB advances four spaces, and NBSP aliases space. Tabs are fixed advances,
not tab stops. The generic `text.layout` accepts custom advances, including
kerning or externally shaped advances. A line-break step ignores its advance.

The bundled subset contains **196 glyphs**: all 95 printable ASCII characters,
85 Latin-1 glyphs and 16 additional letters/punctuation. `assets/charset.txt`
is authoritative. Missing Latin-1 characters and file hashes are recorded in
`assets/provenance.json`. This font release exports no kerning pairs through
msdf-atlas-gen; the convenience layout uses unkerned advances. It does not
perform OpenType shaping, normalization, bidirectional layout, wrapping or
ligatures. Precompose accented Latin text or provide shaped glyph positions
and an appropriate atlas for other scripts.

Inputs must be finite; dimensions, em sizes, font size, viewport and distance
range must be positive; outline widths must be nonnegative. Keep outline/weight
effects within the atlas's half-range padding, leaving room for the antialias
fringe. SDF softens corners; MSDF preserves them better, but neither is an
unbounded-resolution outline renderer. Small text remains unhinted. Evaluate
fragment derivatives in uniform control flow before per-pixel branches.

The package leaves color space to the caller. In a normal scene, blend linear
colors and encode once on output. The specimen chooses display colors for
`viz`'s headless RGBA8 output. Use straight output with Wyn's `#source_over`;
premultiplied output requires a matching host blend state or `text.over`.

## Render the demo

Requires Node 18+ and existing `wyn` / `viz` binaries. There are no npm packages
to install. From the repository root in PowerShell:

```powershell
$env:WYN = (Resolve-Path 'target/release/wyn.exe').Path
$env:VIZ = (Resolve-Path 'extra/viz/target/release/viz.exe').Path
node pkg/text/tools/demo.mjs
```

This compiles the actual Wyn example and renders `pkg/text/build/demo/demo.png`
on the GPU. The specimen compares SDF and MSDF at 18, 24, 42 and 180 pixels per
em, plus an MTSDF fill/outline treatment. Every label also uses this font package.
The tool lays out the fixed specimen on the CPU from the shipped metrics and
draws instanced glyph quads. It asserts the fixed 247-glyph draw count used by
the example, as the current Wyn `direct_draw` requires literal counts.

```powershell
node pkg/text/tools/demo.mjs --interactive
node pkg/text/tools/demo.mjs path/to/output
node pkg/text/test/check.mjs
```

The tests execute numerical GPU checks on WGSL and SPIR-V for distance decoding,
coverage, outlines, rotated derivatives, all glyph metrics, fallback, multiline
layout across scan workgroups, empty/singleton layout, quad bounds, compositing,
and bilinear texture sampling (including texel centers, clamping and alpha).
Asset hashes and bounds are checked too. Tests and demo use existing tools and
never rebuild or edit the compiler. The locally available Vulkan stack emitted
explicit-layout validation warnings during the initial demo; numeric results
passed and the image rendered. This package does not alter that stack or disable
validation. Empty layout is tested inside Wyn because the current `viz` allocator
cannot bind a zero-byte runtime vector input.

## Regenerate the font assets

Download/build **msdf-atlas-gen 1.4.0 (MSDFgen 1.13.0)**, then run:

```powershell
node pkg/text/tools/generate.mjs path/to/msdf-atlas-gen.exe
```

Generation is offline and verifies the bundled OTF checksum. It uses the explicit
subset, 48 texels/em, an 8-texel range, top-down coordinates, seed 0 and one thread.
It regenerates PNG, JSON, Wyn metrics and provenance together. Byte-identical
rebuilds require the same upstream generator build; platform/library versions
can change rasterization details. No generator binary is included.

## Sources and licensing

- [Aileron's author and CC0 declaration](https://dotcolon.net/fonts/aileron/)
- [Author's 0.102 download](https://dotcolon.net/files/fonts/aileron_0102.zip)
- [MSDFgen reference reconstruction](https://github.com/Chlumsky/msdfgen)
- [MSDF Atlas Generator](https://github.com/Chlumsky/msdf-atlas-gen)
- [Requested rendering comparison](https://alphapixeldev.com/sdf-vs-msdf-vs-slug-vs-rive-gpu-text-rendering/)

Font and derived atlas data: CC0, see `assets/CC0-1.0.txt` and
`assets/FONT-LICENSE.txt`. The reconstruction follows Viktor Chlumsky's MSDFgen
reference; its MIT notice is preserved in `assets/LICENSE-msdfgen.txt`.
