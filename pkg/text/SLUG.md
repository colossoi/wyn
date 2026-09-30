# Slug rendering with Aileron

This package implements the core **Slug algorithm by Eric Lengyel**: evaluate
quadratic Bézier crossings and nonzero winding directly in the fragment shader,
with horizontal/vertical band acceleration and analytic two-ray antialiasing.
The vertex helper expands each glyph quad by half a screen pixel, including
under perspective, and adjusts its sample coordinates consistently. This is
an implementation of the published algorithm, not the commercial Slug Library
or its text-shaping engine.

The implementation adapts the MIT-licensed reference at commit
`be3c13eb7d63f9e8aa5c583e42d92c374cb91d98`. The reference's texture indexing is
replaced by typed Wyn storage arrays. Coordinates remain full `f32`, and indices
are `i32`; there is no integer encoding in float texture channels. No compiler,
host runtime, or `viz` changes are needed.

The complete annotated public API and runnable examples are in [API.md](API.md).
All font lookup now lives in `aileron.wyn`; Slug algorithms live in `slug.wyn`.

## Imports and inputs

```wyn
import "pkg:text/slug"          -- font-independent Slug functions
import "pkg:text/aileron"  -- also imports Slug and Aileron metrics
```

Upload these immutable files as read-only storage buffers:

| File | Wyn type | Contents |
| --- | --- | --- |
| `assets/aileron-slug-curves.bin` | `[]vec4f32` | Two vec4s per quadratic: `(p0.x,p0.y,p1.x,p1.y)` then `(p2.x,p2.y,0,0)` |
| `assets/aileron-slug-bands.bin` | `[]vec2i32` | `(offset, count)` into the curve-index list |
| `assets/aileron-slug-indices.bin` | `[]i32` | Curve IDs; multiply by two to locate their first vec4 |

All values are little-endian. Vec4 stride is 16 bytes, vec2 stride is 8 bytes,
and scalar stride is 4 bytes. The buffers contain 12,445 quadratics, 2,962 band
headers, and 34,175 indices. Their combined size is 558,636 bytes. There is no
distance-field texture in this path. The comparison demo separately loads the
MTSDF texture for its SDF and MSDF columns.

`aileron.slug_glyph(codepoint)` returns:

```wyn
type glyph = {
  advance: f32,
  bounds: vec4f32,
  band_transform: vec4f32,
  bands: vec4i32,
}
```

Bounds are `(left, top, right, bottom)` in top-down em coordinates, relative to
the baseline, without distance-field padding. The transform is
`(scale_x, scale_y, offset_x, offset_y)` for selecting a band. The `bands` fields
are `(horizontal_header_offset, horizontal_count, vertical_header_offset,
vertical_count)`. Each count is positive, including space's empty band.

`aileron.layout` supplies the shared layout and fallback policy for both renderers,
so changing renderer does not change text placement. Unsupported glyphs use `?`;
spaces, CR/LF, tabs and NBSP have empty geometry. Layout still provides their
advances and newline behavior. Both glyph records agree with `aileron.advance`,
including zero advance for CR/LF and four-space advance for TAB. Use the layout
API to apply newline behavior.

## Rendering

For every glyph, pass four rows of the matrix that transforms
`(em.x, em.y, 0, 1)` to clip space, including that glyph's baseline translation.
Call:

```wyn
let (clip, em) = slug.quad_vertex(glyph, vertex_index,
                                  m0, m1, m2, m3, viewport) in
vertex_output(clip, em)
```

The vertex index is in `[0,6)`, in triangle-list order. Use perspective-correct
interpolation for `em`. The expanded `em` coordinates must be passed unchanged
to the fragment stage; recomputing them from the original bounds would dilate
the glyph itself instead of just its bounding quad. The glyph metadata must
remain constant across each primitive (flat interpolation for integer IDs,
or pass an identical float ID at every vertex and round before lookup, as the
demo does).

```wyn
let opacity = slug.coverage_fragment(curves, bands, indices, glyph, em) in
text.straight(color, opacity)
```

`coverage_fragment` computes the per-axis pixel footprint using `fwidth`.
Evaluate it in uniform control flow. If selecting the renderer per fragment,
compute derivatives before branching and use the pure variant:

```wyn
slug.coverage(curves, bands, indices, glyph, em, ems_per_pixel)
```

Here `ems_per_pixel = abs(dEm/dScreenX) + abs(dEm/dScreenY)` componentwise, as in
the reference shader. For ordinary axis-aligned text at 64 pixels/em, it is
`@[1.0f32/64.0f32, 1.0f32/64.0f32]`.

The low-level `slug.dilate(position, miter_normal, m0, m1, m3, viewport)` supports
other convex glyph bounding polygons. The normal is scaled so its endpoint
moves both adjacent edges by one unit: rectangle corners use `(±1,±1)`, not
normalized diagonal vectors. The function normalizes internally only for the
projection calculation. `quad_vertex` constructs the appropriate normals.

## Geometry contract and limits

- Curves must form closed contours. Straight segments are represented by
  `(start, end, end)`. Holes use opposite winding to their enclosing contours.
- Bands partition glyph space uniformly. Curves in a horizontal band are sorted
  by descending maximum control-point X; vertical bands sort by maximum Y.
  The renderer relies on this order for its early exit.
- Omit perfectly horizontal curves from horizontal bands and perfectly vertical
  curves from vertical bands. The generator overlaps bands by `1/1024` em to
  accommodate boundaries and round-off, as recommended by the reference.
- Rendering uses the nonzero fill rule. Optional even-odd filling, optical
  weight boosting, font hinting, shaping, ligatures and color-font layers are
  not implemented. The existing text package's layout limitations still apply.
- Slug's two-ray filter estimates pixel coverage; it is not exact area
  integration. The tests preserve the reference's behavior at exact tangencies
  and double roots, including its 0.25 coverage at the synthetic lens tangent.
- Inputs and matrix entries must be finite. Viewport dimensions and pixel
  footprints are positive. All buffer offsets/counts must be in range. The
  glyph must lie in front of the near plane with a nonsingular projection;
  dynamic dilation requires `sqrt(u*u+v*v) > s*t` in the reference's notation.
  The host must clip glyphs that cross the eye/near plane. Extremely oblique
  projections and very large coordinates remain subject to f32 precision.

Aileron Regular is a CFF/OpenType font with cubic outlines. The offline generator
uses FontTools `Cu2QuPen` to convert those to quadratics with a permitted deviation
of **1/65536 em** (about 0.03125 pixel at 2048 pixels/em), then stores f32 control
points. The GPU analytically evaluates those stored quadratics. It does not
recover exact original cubics at arbitrary magnification; both the conversion
tolerance and subsequent f32 round-off remain. Glyph bounds are computed from
the converted quadratic extrema.

The root eligibility bit table and signed zero behavior match the reference.
One degenerate guard avoids division by zero when the nearly-linear fallback
has zero slope. Dynamic dilation uses the algebraically equivalent rationalized
form `s*s / (sqrt(u*u+v*v) - s*t)` to avoid subtracting nearly equal squares.

## Regeneration and tests

```powershell
python -m pip install -r pkg/text/tools/slug-requirements.txt
python pkg/text/tools/generate-slug.py
node pkg/text/test/check-slug.mjs
node pkg/text/tools/demo.mjs
```

FontTools 4.60.1, Python 3.10+ and Node 18+ are needed for offline regeneration.
Runtime rendering and the demo do not need Python. The generator verifies the
same bundled font checksum as the SDF generator and takes the exact 196-glyph
subset from the existing atlas metadata. It regenerates all three buffers,
`aileron-slug.json`, then invokes `tools/generate-font.mjs` to rebuild the unified
`src/aileron.wyn` from both renderer metadata files. JSON includes the source
hash, conversion tolerance, band overlap, per-glyph curve ranges and buffer hashes.

The suite checks band integrity and sorting, every glyph against an independent
f64 root/winding calculation at 4,681 sample points per target, plus synthetic
lines, quadratic curves, holes, reversed winding and signed zero. It checks
dynamic dilation by projecting the result back to pixels and measuring the
half-pixel displacement under orthographic and perspective transforms. Both
WGSL and SPIR-V execute on the GPU. Set `WYN` and `VIZ` to existing tool paths
as in the main README.

## References and attribution

- Eric Lengyel, [GPU-Centered Font Rendering Directly from Glyph Outlines](https://jcgt.org/published/0006/02/02/), JCGT 2017.
- [Slug reference implementation](https://github.com/EricLengyel/Slug/tree/be3c13eb7d63f9e8aa5c583e42d92c374cb91d98).
- [A Decade of Slug](https://terathon.com/blog/decade-slug.html), including dynamic dilation and the author's patent dedication announcement.

Slug shader code Copyright 2017 by Eric Lengyel. Adapted under MIT; the full
license and source provenance are in `assets/LICENSE-slug.txt` and
`assets/SLUG-REFERENCE.txt`. Aileron and its derived outline data are CC0.
