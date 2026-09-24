# wyn/colorspace

Small, pure `f32` colour functions for Wyn shaders. No dependencies, textures,
allocations or runtime colour-space dispatcher. Import with `import "pkg:colorspace"`
after adding a `colorspace` dependency pointing to this package.

The API keeps **primaries/white point**, **transfer function**, **exposure** and
**display rendering** separate. An RGB triplet alone does not specify a colour.
Function names and namespaces state the expected representation; values remain
ordinary `vec3f32` so the package composes with existing shaders.

## Implemented core

| Function under `colorspace` | Input → output |
| --- | --- |
| `srgb.decode(rgb)` / `decode_channel(x)` | Encoded sRGB → linear-light sRGB |
| `srgb.encode(rgb)` / `encode_channel(x)` | Linear-light sRGB → encoded sRGB |
| `linear_srgb.to_xyz_d65(rgb)` | Linear-light sRGB → CIE XYZ, D65 |
| `linear_srgb.from_xyz_d65(xyz)` | CIE XYZ, D65 → linear-light sRGB |
| `linear_srgb.luminance(rgb)` | Linear-light sRGB → relative CIE Y |
| `gamma.decode(rgb, gamma)` / `decode_channel(x, gamma)` | Signed power `sign(x) * abs(x)^gamma` |
| `gamma.encode(rgb, gamma)` / `encode_channel(x, gamma)` | Signed power `sign(x) * abs(x)^(1/gamma)` |
| `hdr.expose(rgb, stops)` | Scene-linear RGB multiplied by `2^stops` |
| `hdr.relative_to_nits(rgb, reference_white_nits)` | Relative linear RGB → absolute linear RGB scaling |
| `hdr.nits_to_relative(rgb, reference_white_nits)` | Inverse reference-white scaling |

All RGB arguments/results are `vec3f32`; channel operations and luminance use
`f32`. Alpha stays separate and unchanged. Nonlinear transfers expect straight,
unpremultiplied RGB; unpremultiply first when working with premultiplied data.

Inputs must be finite, gamma and reference white strictly positive, and magnitudes
small enough that intermediate/results fit in `f32`. The shader functions do not
perform parameter validation. Invalid parameters, NaN, infinities and overflow
are outside the contract. Negative/out-of-gamut values and values above one are
preserved rather than silently clipped. Negative transfer values use the signed
extension; they do not represent negative physical light. The rounded sRGB
standard breakpoints are retained, including their tiny discontinuity.

XYZ uses D65 `(x=.3127, y=.3290)` with `Y=1` for RGB white `(1,1,1)`.
No chromatic adaptation occurs. Luminance is linear CIE Y, not nonlinear video
luma, HSL lightness or perceived brightness. After reference-white scaling,
`linear_srgb.luminance(rgb_nits)` gives cd/m² (nits). Reference-white scaling
requires calibrated relative input; it does not turn arbitrary scene radiance
into calibrated photometry by itself. There is deliberately no default white
level, display peak or exposure.

### Typical use

```wyn
import "pkg:colorspace"

def authored_color: vec3f32 =
  colorspace.srgb.decode(@[0.30f32, 0.49f32, 0.69f32])

def adjust_light(rgb: vec3f32, stops: f32) vec3f32 =
  colorspace.hdr.expose(rgb, stops)
```

Lighting, blending and exposure happen in linear light. A renderer then applies
its chosen tone/gamut mapping, followed by display encoding **once**. An sRGB
texture sampler already decodes color textures. An sRGB output attachment already
encodes linear output; do not also call `srgb.encode` there. Normal maps and
roughness are data, not encoded colours. This package does not install a renderer
display transform or change its sky palette.

Power gamma is a separate transfer model, not a replacement for the piecewise
sRGB curve. For example, encoded sRGB `0.5` decodes to approximately `0.21404114`.
Colour-grading controls called “gamma” should be a future explicitly defined
operation, rather than an ambiguous `gamma_correct` alias.

## Broader API direction (not implemented)

Grow in independently testable layers rather than a universal `convert` switch:

| Area | Proposed shape and important choices |
| --- | --- |
| RGB spaces | `linear_display_p3` and `linear_rec2020` ↔ `xyz_d65`; transfer namespaces stay separate. P3 primaries do not imply a unique transfer curve. |
| White points | Explicit `adapt.bradford_d65_to_d50` / inverse first; then a matrix builder if needed. Never adapt implicitly inside an XYZ conversion. |
| Perceptual authoring | `oklab.from_linear_srgb` / `to_linear_srgb`; later OkLCh with hue in radians, documented achromatic hue and interpolation wrap policy. Useful for authoring, not light accumulation or a claim of uniform HDR appearance. |
| HDR encoding | `pq.encode_nits` / `decode_nits` with absolute luminance units; HLG scene OETF and display EOTF as distinct APIs, explicit display peak/system gamma. Define domain/clipping policies before implementing either. |
| Display rendering | `tonemap.*` accepts a named linear input space and outputs a documented display-linear space. Name a fitted curve `aces_fit` rather than claiming a complete ACES transform. Exposure, tone mapping, gamut mapping and transfer remain separately composable. |
| Gamut handling | `gamut.in_srgb`, explicit hard clipping, then a perceptual mapper with a documented destination gamut. Never hide gamut loss in an invertible conversion. |
| Grading | Exposure first; later white balance, contrast with a named pivot, and lift/gamma/gain with defined space and parameter direction. |
| Interpolation | Linear-light RGB mixing for light; Oklab/OkLCh for perceptual authoring. Explicit straight/premultiplied alpha utilities with a zero-alpha policy. |
| Packing/interop | Explicit `srgb8` unpack/pack and rounding rules, coordinated with `wyn/packing`; packed integer colors must state whether they contain encoded or linear values. |
| Spectral work | Spectrum → XYZ via observer/illuminant integration as a separate module or offline tooling. Three wavelength samples are not automatically RGB primaries. |

Prefer explicit composition initially. If space mix-ups become common, add small
record wrappers or compile-time metadata without creating per-pixel runtime
dispatch. Conversion functions should continue accepting HDR and out-of-gamut
intermediate values where mathematically meaningful.

## Validation

Requires Wyn on PATH (or `WYN`), Cargo and a WGPU adapter:

```sh
cargo test --manifest-path pkg/colorspace/test/runner/Cargo.toml -- --nocapture
```

The runner compiles and executes the actual package through both SPIR-V and WGSL,
using runtime input buffers. It checks an 8-bit ramp, both transfer knees and
their signed counterparts, known gray/white/primary references, XYZ and transfer
round trips, negative/HDR values, exposure stops and explicit reference-white
scaling against `f64` references. Tolerance is `2e-5 * max(1, abs(expected))`.
Generated shader files go to the system temporary directory. For compile-only:

```sh
wyn build pkg/colorspace/test/colorspace.wyn -O --target wgsl -o /tmp/colorspace.wgsl
```

## Sources

- [W3C CSS Color 4 conversion code](https://www.w3.org/TR/css-color-4/#color-conversion-code):
  extended sRGB transfer and D65 RGB/XYZ matrices. The matrix coefficients above
  are rounded to `f32`; the numerical references use the rational coefficients.
- [Oklab, Björn Ottosson](https://bottosson.github.io/posts/oklab/): future perceptual layer.
- [ITU-R BT.2100](https://www.itu.int/rec/R-REC-BT.2100): future PQ/HLG work.
- [Bruneton atmosphere API](https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/model.h.html):
  useful distinction between spectral samples, linear-sRGB photometry and display encoding.
