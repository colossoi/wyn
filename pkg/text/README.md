# wyn/text

SDF, MSDF, MTSDF and **Slug** text rendering for Wyn, with a bundled **Aileron Regular
0.102** font atlas. Aileron is a widely distributed sans-serif by Sora Sagano
(dot colon), released under **CC0**. The font, license, atlas, metrics and an
offline regeneration script are included. No compiler or renderer modifications
are required.

Slug evaluates quadratic outlines directly on the GPU using the published
algorithm by Eric Lengyel. It includes band acceleration, nonzero winding
coverage, two-ray antialiasing and dynamic half-pixel quad dilation. It uses
the same font and glyph subset as the distance-field path. See [SLUG.md](SLUG.md)
for its buffer formats, generation and precision limits. [API.md](API.md) is the
complete annotated public reference, including SDF and Slug usage examples.

## Use

Add a dependency to your application's `wyn.toml` (adjust the relative path):

```toml
[dependencies]
text = { package = "wyn/text", version = "v0.1.0", path = "../text" }
```

```wyn
import "pkg:text"          -- shared layout, coordinates and compositing
import "pkg:text/sdf"      -- SDF/MSDF/MTSDF, plus shared helpers
import "pkg:text/slug"     -- Slug, plus shared helpers
import "pkg:text/aileron"  -- all Aileron metadata, plus both renderers
```

Importing only `pkg:text/aileron` is sufficient when using the bundled font.
The four source files are `lib.wyn`, `sdf.wyn`, `slug.wyn` and `aileron.wyn`.
Aileron has one combined table for both renderers, one character lookup and
one layout/advance policy. Texture and curve buffers remain external assets.

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
  let range = sdf.screen_range_fragment(
    sdf.unit_range(aileron.px_range, aileron.atlas_size), uv) in
  let sample = texture_sample(atlas, linear_sampler, uv, 0.0f32) in
  let opacity = sdf.fill(sdf.msdf_distance(sample.xyz), range, 0.0f32) in
  text.straight(@[1.0f32, 1.0f32, 1.0f32, 1.0f32], opacity)
```

For true SDF with the bundled texture, replace `sdf.msdf_distance(sample.xyz)`
with `sdf.distance(sample.w)`. For a separate single-channel SDF texture,
pass its red channel instead. Interpolate the three channels before taking
their median. The `sdf.sample_bilinear` fallback performs four clamped texel
loads when the host cannot provide a linear sampler; the demo uses this because
the current `viz` runner supplies nearest-neighbor samplers.

## API

See [API.md](API.md) for every public type, constant and function, with units,
return values, shader-stage restrictions and complete examples. Source definitions
have matching documentation comments. Names starting with `internal_` are
implementation details outside the supported API.

| Namespace | Public operations |
| --- | --- |
| `aileron` | `sdf_glyph`, `slug_glyph`, `advance`, `has_glyph`, `layout`, font/atlas metrics |
| `sdf` | Distance decoding, range calculation, fill/outline coverage, sampling, glyph quads |
| `slug` | Explicit/fragment coverage, glyph quads, dynamic dilation |
| `text` | Generic layout, pixel-to-clip conversion, straight/premultiplied compositing |

`aileron.sdf_glyph(codepoint)` returns padded em-space bounds and atlas UVs.
`aileron.slug_glyph(codepoint)` returns tight outline bounds and band metadata.
Both use `?` for missing characters and agree with `aileron.advance` on controls.
`aileron.has_glyph` reports actual subset coverage, including space but excluding
control/NBSP aliases. The ascender is negative in top-down coordinates.

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
on the GPU. The specimen compares SDF, MSDF and Slug at 18, 23, 36 and 136 pixels
per em, with a larger Slug text line. Every label also uses this font package.
The tool lays out the fixed specimen on the CPU from the shipped metrics and
draws instanced glyph quads. It asserts the fixed 353-glyph draw count used by
the example, as the current Wyn `direct_draw` requires literal counts.

```powershell
node pkg/text/tools/demo.mjs --interactive
node pkg/text/tools/demo.mjs path/to/output
node pkg/text/test/check.mjs
node pkg/text/test/check-slug.mjs
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

The Slug suite adds an independent f64 winding reference for 4,681 font samples
per target, synthetic edge/curve/hole tests, signed-zero root eligibility,
buffer/band integrity checks and projection-space dilation tests. Shared-font
checks cover renderer agreement for all advances, controls and fallback. The current
combined suites execute 38 GPU cases across WGSL and SPIR-V.

## Regenerate the font assets

Download/build **msdf-atlas-gen 1.4.0 (MSDFgen 1.13.0)**, then run:

```powershell
node pkg/text/tools/generate.mjs path/to/msdf-atlas-gen.exe
```

Generation is offline and verifies the bundled OTF checksum. It uses the explicit
subset, 48 texels/em, an 8-texel range, top-down coordinates, seed 0 and one thread.
It regenerates PNG, JSON and provenance, then rebuilds the unified Aileron
module using the shipped Slug metadata. The Slug generator also rebuilds this
same module after generating its buffers. Both routes check that glyph subsets
and advances agree. Run `node pkg/text/tools/generate-font.mjs` to rebuild only
the Wyn metadata from the two shipped JSON files. Byte-identical
rebuilds require the same upstream generator build; platform/library versions
can change rasterization details. No generator binary is included.

## Updating from the initial package

The renderer split deliberately changes the initial import/API names:

| Initial name | Current name |
| --- | --- |
| `text.glyph`, `text.empty_glyph` | `sdf.glyph`, `sdf.empty_glyph` |
| `text.sdf.distance` | `sdf.distance` |
| `text.msdf.distance` | `sdf.msdf_distance` |
| `text.unit_range`, `text.screen_range*` | Corresponding `sdf.*` functions |
| `text.coverage`, `text.fill`, `text.outline` | Corresponding `sdf.*` functions |
| `text.sample_bilinear`, `text.quad_vertex` | Corresponding `sdf.*` functions |
| `aileron.glyph` | `aileron.sdf_glyph` |
| `pkg:text/aileron_slug` | `pkg:text/aileron` |
| `aileron_slug.glyph`, `aileron_slug.layout` | `aileron.slug_glyph`, `aileron.layout` |

Shared layout, coordinate conversion and compositing remain in `text`.
Raw font tables, lookup indexes and Slug root/ray helpers are internal.

## Sources and licensing

- [Aileron's author and CC0 declaration](https://dotcolon.net/fonts/aileron/)
- [Author's 0.102 download](https://dotcolon.net/files/fonts/aileron_0102.zip)
- [MSDFgen reference reconstruction](https://github.com/Chlumsky/msdfgen)
- [MSDF Atlas Generator](https://github.com/Chlumsky/msdf-atlas-gen)
- [Eric Lengyel's Slug reference shaders](https://github.com/EricLengyel/Slug)
- [Requested rendering comparison](https://alphapixeldev.com/sdf-vs-msdf-vs-slug-vs-rive-gpu-text-rendering/)

Font and derived atlas data: CC0, see `assets/CC0-1.0.txt` and
`assets/FONT-LICENSE.txt`. The reconstruction follows Viktor Chlumsky's MSDFgen
reference; its MIT notice is preserved in `assets/LICENSE-msdfgen.txt`.
The Slug implementation is adapted from Eric Lengyel's reference shaders under
MIT, with credit and the license in `assets/SLUG-REFERENCE.txt` and
`assets/LICENSE-slug.txt`. Slug's font-derived curve and band data remain CC0.
