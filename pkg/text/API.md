# Public API

This is the supported public surface of `wyn/text`. Source definitions also
carry documentation comments. Names beginning with `internal_` are implementation
details, even where Wyn permits access; they are not part of this contract.
Declarations below omit bodies. All numeric inputs must be finite.

## Modules and imports

| Import | Namespace | Responsibility |
| --- | --- | --- |
| `pkg:text` | `text` | Shared layout, coordinates and compositing |
| `pkg:text/sdf` | `sdf` | SDF, MSDF and MTSDF rendering; imports `text` |
| `pkg:text/slug` | `slug` | Direct quadratic-outline rendering; imports `text` |
| `pkg:text/aileron` | `aileron` | All bundled font metadata; imports both renderers |

The corresponding files are `src/lib.wyn`, `src/sdf.wyn`, `src/slug.wyn` and
`src/aileron.wyn`. Aileron has one combined glyph table and one codepoint lookup.
The atlas image and GPU curve/band buffers remain external binary resources.
The package provides shader functions; the caller owns resource uploads and draws.

## Aileron

```wyn
-- Texture dimensions in texels; atlas density in texels/em;
-- full encoded distance range in texels, respectively.
def atlas_size: vec2f32
def atlas_em_size: f32
def px_range: f32

-- Baseline spacing in em; top-down baseline offsets in em.
-- The ascender is negative and the descender is positive.
def line_height: f32
def ascender: f32
def descender: f32

-- Number of explicitly supported codepoints, including space.
def glyph_count: i32

-- True only for supported codepoints. Control/NBSP aliases return false.
def has_glyph(codepoint: i32) bool

-- Horizontal advance in em, using the fallback/control rules below.
def advance(codepoint: i32) f32

-- Padded distance-field geometry and atlas UVs.
def sdf_glyph(codepoint: i32) sdf.glyph

-- Tight outline bounds and Slug band lookup metadata.
def slug_glyph(codepoint: i32) slug.glyph

-- One baseline per codepoint, in top-down pixels, starting at (0,0).
-- font_size is positive screen pixels/em.
def layout<[n]>(codepoints: [n]i32, font_size: f32) [n]vec2f32
```

Supply Unicode scalar values as `i32`, not UTF-8 bytes. Missing codepoints use
`?`; TAB advances four spaces; LF resets X and advances Y by the line height;
CR is ignored; NBSP aliases space. Both glyph methods return empty geometry for
space and these controls, and their advances agree with `aileron.advance`.
Tabs are fixed advances, not tab stops. Layout performs no shaping, kerning,
normalization, bidi reordering, wrapping or ligatures. See the README for the
196-character subset and font provenance.

## Distance fields: `sdf`

```wyn
type glyph = {
  advance: f32,    -- layout advance in em
  plane: vec4f32,  -- padded left,top,right,bottom in em relative to baseline
  uv: vec4f32,     -- matching normalized atlas left,top,right,bottom
}

-- Invisible, degenerate geometry with the supplied advance in em.
def empty_glyph(advance: f32) glyph

-- channel - 0.5: zero at outline, positive inside, negative outside.
-- Returns encoded distance units, not pixels. For Aileron, use atlas alpha.
def distance(channel: f32) f32

-- Median of interpolated RGB minus 0.5; same units/sign as distance().
-- Interpolate RGB before decoding. For Aileron, use atlas RGB.
def msdf_distance(rgb: vec3f32) f32

-- Full encoded range / atlas dimensions, producing a UV-unit range.
-- px_range is full range in texels, not half-range. Inputs are positive.
def unit_range(px_range: f32, atlas_size: vec2f32) vec2f32

-- Uniform, axis-aligned scale: max(px_range * font_size / atlas_em_size, 1).
-- px_range: atlas texels; font_size: screen pixels/em;
-- atlas_em_size: atlas texels/em. Returns a range in screen pixels.
def screen_range_2d(px_range: f32, font_size: f32, atlas_em_size: f32) f32

-- General transform: unit comes from unit_range(); uv_dx and uv_dy are
-- UV changes per horizontal/vertical screen pixel. Handles rotation and
-- local perspective scaling. Result is at least one screen pixel.
-- Pure arithmetic; usable in any stage with explicit gradients.
def screen_range(unit: vec2f32, uv_dx: vec2f32, uv_dy: vec2f32) f32

-- Fragment-only wrapper computing UV derivatives. Execute in uniform
-- control flow, before divergent branches.
def screen_range_fragment(unit: vec2f32, uv: vec2f32) f32

-- clamp(distance_px + 0.5, 0, 1): one-screen-pixel antialiasing ramp.
def coverage(distance_px: f32) f32

-- Filled opacity [0,1]. signed_distance comes from a decoder;
-- screen_px_range from a range function. weight_px is screen pixels:
-- zero = ordinary fill; positive = expand; negative = thin.
def fill(signed_distance: f32, screen_px_range: f32, weight_px: f32) f32

-- Exterior outline opacity [0,1], excluding normal fill.
-- Same distance/range units as fill(); width_px is nonnegative screen pixels.
-- Returns expanded coverage minus ordinary coverage. True SDF supports
-- rounded distance effects; MSDF preserves sharper corners.
def outline(signed_distance: f32, screen_px_range: f32, width_px: f32) f32

-- Four clamped texel loads, bilinearly interpolated at mip level zero.
-- dimensions must match the texture. Preserves all channels, including alpha.
-- A host-provided linear sampler usually needs fewer fetches.
def sample_bilinear(atlas: texture2d, dimensions: vec2i32, uv: vec2f32) vec4f32

-- One vertex of a six-entry triangle-list quad; vertex_index is [0,6).
-- baseline is top-down screen pixels; font_size is positive pixels/em.
-- Returns (pixel position, atlas UV). Use text.pixel_to_clip() on position.
-- Empty glyphs produce degenerate geometry.
def quad_vertex(g: glyph, baseline: vec2f32, font_size: f32,
                vertex_index: u32) (vec2f32, vec2f32)
```

All plane coordinates and atlas rows increase downward. Upload the bundled
atlas as `Rgba8Unorm`: it is numeric data, so preserve alpha, avoid sRGB decoding
and premultiplication, and use linear filtering with clamp-to-edge at LOD 0.
Do not use ordinary color mipmaps. Keep outline/weight effects within the atlas's
half-range padding, allowing room for the antialias fringe. Dimensions, sizes
and distance ranges must be positive. SDF/MSDF remain limited by atlas resolution.

## Outlines: `slug`

```wyn
type glyph = {
  advance: f32,        -- layout advance in em
  bounds: vec4f32,     -- tight left,top,right,bottom in top-down em; no padding
  band_transform: vec4f32, -- scale_x,scale_y,offset_x,offset_y
  bands: vec4i32,      -- horizontal header offset/count, vertical offset/count
}

-- Evaluate nonzero-winding coverage, returning [0,1]. Buffer formats:
-- curves: two vec4s per quadratic: (p0.x,p0.y,p1.x,p1.y), (p2.x,p2.y,0,0).
-- bands: (offset,count) into indices; glyph offsets select these headers.
-- indices: curve IDs; ID i locates curves[2*i], not curves[i].
-- coordinate: sample location in top-down em relative to baseline.
-- ems_per_pixel: positive per-axis footprint in em, normally
--   abs(dEm/dScreenX) + abs(dEm/dScreenY).
-- At 96 pixels/em without rotation: (1/96,1/96).
-- No derivatives inside; this is Slug's two-ray coverage estimate.
def coverage(curves: []vec4f32, bands: []vec2i32, indices: []i32,
             g: glyph, coordinate: vec2f32, ems_per_pixel: vec2f32) f32

-- Fragment-only wrapper: uses componentwise fwidth(coordinate).
-- Execute in uniform control flow. For divergent renderer selection,
-- compute derivatives before branching and call coverage() explicitly.
def coverage_fragment(curves: []vec4f32, bands: []vec2i32, indices: []i32,
                      g: glyph, coordinate: vec2f32) f32

-- One of six triangle-list vertices; vertex_index is [0,6).
-- m0...m3: rows mapping (em.x,em.y,0,1) to clip space, including glyph
-- baseline translation, scale, rotation and projection.
-- viewport: positive render-target width/height in screen pixels.
-- Returns (clip-space position including W, expanded em coordinate).
-- Expands the quad for the antialias fringe. Pass em unchanged to the
-- fragment shader with perspective-correct interpolation; do not clamp it.
def quad_vertex(g: glyph, vertex_index: u32, m0: vec4f32, m1: vec4f32,
                m2: vec4f32, m3: vec4f32, viewport: vec2f32) (vec4f32, vec2f32)

-- Lower-level dynamic dilation for custom convex bounding polygons.
-- position: original vertex in em. normal: nonzero outward miter normal;
-- rectangle corners use (+/-1,+/-1), not normalized diagonals.
-- Transform rows and viewport have the same meaning as quad_vertex().
-- Returns expanded em position, using the reference half-pixel dilation
-- calculation including perspective. quad_vertex() calls this automatically.
def dilate(position: vec2f32, normal: vec2f32, m0: vec4f32, m1: vec4f32,
           m3: vec4f32, viewport: vec2f32) vec2f32
```

`band_transform` maps sample coordinates to uniform bands via
`coordinate * scale + offset`, followed by floor/clamping. Both band counts
must be positive, including for an empty glyph. Buffer offsets/counts must be
valid. Contours must be closed; holes use opposite winding. Band sorting is
required for early exit. See [SLUG.md](SLUG.md) for the full geometry contract,
buffer byte layouts, conversion tolerance and filtering limits.

Projection must be nonsingular, with the glyph in front of the near plane;
the caller clips eye/near-plane crossings. Dilation requires
`sqrt(u*u+v*v) > s*t` in the reference notation documented in `SLUG.md`.

## Shared helpers: `text`

```wyn
-- advance is em; a line break ignores advance, resets X and advances Y.
type layout_step = { advance: f32, line_break: bool }

-- Exclusive baselines, starting at (0,0), returned in top-down pixels.
-- font_size is positive pixels/em; line_height is positive em.
-- Supports custom advances including externally shaped or kerned positions.
def layout<[n]>(steps: [n]layout_step, font_size: f32,
                line_height: f32) [n]vec2f32

-- Top-down pixels to WebGPU/Vulkan clip coordinates, with W=1.
-- viewport dimensions must be positive; depth must be in [0,1].
def pixel_to_clip(position: vec2f32, viewport: vec2f32, depth: f32) vec4f32

-- Apply coverage to straight color alpha; RGB is unchanged.
-- Use with Wyn #source_over / WGPU ALPHA_BLENDING.
def straight(color: vec4f32, opacity: f32) vec4f32

-- Straight input -> premultiplied output, applying opacity once.
-- Requires a matching host blend state or explicit over() composition.
def premultiply(color: vec4f32, opacity: f32) vec4f32

-- Premultiplied foreground over premultiplied background.
def over(foreground: vec4f32, background: vec4f32) vec4f32
```

Color alpha and opacity should be in `[0,1]`. Helpers do not clamp or convert
color spaces. Ordinarily blend in linear space and encode once on output.

## What quad_vertex does

The GPU draws each glyph on a rectangle made from two triangles, hence six
vertex entries. Each vertex carries a position and sampling coordinates.
The GPU interpolates these coordinates; the fragment shader determines the
letter's coverage and leaves the surrounding rectangle transparent.
Layout places baselines; `quad_vertex` places a glyph rectangle at its baseline.
SDF supplies atlas UVs. Slug supplies em coordinates and expands its rectangle
for antialiasing. These are low-level geometry helpers, not text-layout routines.

## Usage examples

Add the dependency shown in the README. Applications can import only
`pkg:text/aileron` to access all four namespaces. Complete standalone examples
are [examples/sdf.wyn](examples/sdf.wyn) and [examples/slug.wyn](examples/slug.wyn).
Their package-local imports allow compilation directly from this repository;
the equivalent application examples below use the public package import.

For SDF, upload `assets/aileron-mtsdf.png` as `Rgba8Unorm`. The example uses alpha
for true SDF; change its decoder to `sdf.msdf_distance(encoded.xyz)` for MSDF.
Both examples draw A at 96 pixels/em with its baseline at (40,140).

```wyn
import "pkg:text/aileron"

-- Draw A at a 96-pixel em size, with its baseline at (40,140).
entry sdf_example(atlas: texture2d, resolution: vec3f32,
                  screen: render_target<vec4f32>) render_target<vec4f32> =
  let triangles = rasterize_triangles(direct_draw(6u32, 1u32),
    |v: u32, _: u32, _: u32|
      let (p, uv) = sdf.quad_vertex(aileron.sdf_glyph(65),
        @[40.0f32, 140.0f32], 96.0f32, v) in
      vertex_output(text.pixel_to_clip(p, resolution.xy, 0.0f32), uv)) in
  shade_with({depth_test = #disabled, depth_write = false,
              blend = #source_over, color_write = true}, screen, triangles,
    |uv, _, _, _, _|
      let range = sdf.screen_range_fragment(
        sdf.unit_range(aileron.px_range, aileron.atlas_size), uv) in
      let encoded = sdf.sample_bilinear(atlas,
        @[i32(aileron.atlas_size.x), i32(aileron.atlas_size.y)], uv) in
      let d = sdf.distance(encoded.w) in
      -- For MSDF, use: sdf.msdf_distance(encoded.xyz)
      text.straight(@[1.0f32, 1.0f32, 1.0f32, 1.0f32],
                    sdf.fill(d, range, 0.0f32)))
```

For Slug, upload `aileron-slug-curves.bin`, `aileron-slug-bands.bin` and
`aileron-slug-indices.bin` to the correspondingly named inputs, using the buffer
types above. No atlas is needed by this example.

```wyn
import "pkg:text/aileron"

-- Draw the same A at the same size and baseline, directly from curves.
entry slug_example(curves: []vec4f32, bands: []vec2i32, indices: []i32,
                   resolution: vec3f32,
                   screen: render_target<vec4f32>) render_target<vec4f32> =
  let viewport = resolution.xy in
  let m0 = @[192.0f32 / viewport.x, 0.0f32, 0.0f32, 80.0f32 / viewport.x - 1.0f32] in
  let m1 = @[0.0f32, -192.0f32 / viewport.y, 0.0f32, 1.0f32 - 280.0f32 / viewport.y] in
  let m2 = @[0.0f32, 0.0f32, 0.0f32, 0.0f32] in
  let m3 = @[0.0f32, 0.0f32, 0.0f32, 1.0f32] in
  let triangles = rasterize_triangles(direct_draw(6u32, 1u32),
    |v: u32, _: u32, _: u32|
      let (clip, em) = slug.quad_vertex(aileron.slug_glyph(65),
        v, m0, m1, m2, m3, viewport) in
      vertex_output(clip, em)) in
  shade_with({depth_test = #disabled, depth_write = false,
              blend = #source_over, color_write = true}, screen, triangles,
    |em, _, _, _, _|
      let opacity = slug.coverage_fragment(curves, bands, indices,
        aileron.slug_glyph(65), em) in
      text.straight(@[1.0f32, 1.0f32, 1.0f32, 1.0f32], opacity))
```
