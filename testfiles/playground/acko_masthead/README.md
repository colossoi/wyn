# Acko masthead — Wyn port

Wyn port of the music-reactive 3D track visualization from
[acko.net](https://acko.net/)'s home-page masthead. Original by
**Steven Wittens**. The original implementation lives in
`https://acko.net/cache/combo.min.js` (a minified bundle of Three.js +
Acko's app code).

The masthead at first paint shows ~20 "track" tubes — spline curves
extruded through superellipse cross-sections — lit with a
sun + zenith diffuse plus half-vector specular, pre-baked AO, and a
CRT post-process (barrel curvature + RGB sub-pixel + 16-px quantize +
3-level color quantize + scanlines + vignette + edge "tube" shade).

## Files

| File | What | Committed? |
|---|---|---|
| `positions.bin` | Per-vertex world position + AO: `N × vec4f32` `(x, y, z, ao)`, little-endian. | Yes — shipped data. |
| `normals.bin` | Per-vertex normal: `N × vec4f32` `(nx, ny, nz, 1.0)`. | Yes — shipped data. |
| `vertex_colors.bin` | Per-vertex color: `N × vec4f32` `(r, g, b, 1.0)`, each track's palette color. | Yes — shipped data. |
| `scene.wyn` | Vertex + fragment shader for the lit 3D pass. Vertex reads the three `.bin`s as storage buffers; fragment is the TrackShader port. | Yes. |
| `post.wyn` | Vertex + fragment shader for the CRT post-process pass. Reads the scene render target as a `Sampler2D`. | Yes. |
| `pipeline.json` | Two-stage graphics pipeline descriptor wiring scene → rgba16f attachment → post → swapchain. | Yes. |
| `scene.spv` / `scene.json` | `wyn compile` output. | No — `build_scene.bat` regenerates. |
| `README.md` | This file. | Yes. |

`N` = 62792 (62792 cross-section ring vertices across 16 tracks; the
4 placeholder tracks in `Acko.TrackData` with `n < 3` are degenerate
scaffolding and are skipped).

### `.bin` format

Each file is a flat little-endian float32 array, no header — `N`
entries of 4 floats each (16-byte stride). The three files are
parallel: index `i` in all three describes the same vertex. viz's
`--storage-dir` flag loads `<dir>/<binding_name>.bin` for each
`storage_buffer` binding `scene.wyn` declares, matched by name. The
vertex shader indexes them by `vertex_index`.

There's no text mirror — the `.bin`s are the committed source of
truth. To inspect by eye, `hexdump` or a three-line
`struct.unpack` loop reads them fine; the layout is fully specified
above.

## Regenerating the `.bin` files

The `.bin`s are produced by a one-time scratch tool that runs Acko's
geometry pipeline against the `Acko.TrackData` literal embedded in
`combo.min.js`, and writes `positions.bin` / `normals.bin` /
`vertex_colors.bin` directly. The tool itself **isn't checked in** (it
depends on third-party JS we don't own), but the algorithm it
implements is documented below in enough detail that it can be
re-built from scratch in any language with vec3 + mat3 arithmetic.

The committed `.bin`s were generated against `combo.min.js` version
`c3d6624d` (cache key visible in the asset URL at fetch time).

### Step 0 — get the source

```bash
curl -sL 'https://acko.net/cache/combo.min.js?c3d6624d' -o combo.min.js
```

### Step 1 — extract `Acko.TrackData` and `Acko.Palette` literals

Both are object/array literals inside `combo.min.js`. Find them with
substring search:

- `Acko.Palette={red:[...], platinum:[...], blue:[...], slate:[...], orange:[]};`
- `Acko.TrackData=[{n:8, power:..., width:..., points:[[_v(x,y,z), null, smooth_weight, tension?], ...], ...}, ...];`

Both reference a global helper `_v(x, y, z)` that constructs a
`THREE.Vector3`. For extraction purposes, stub it as
`(x, y, z) => ({ x, y, z })` and eval the literals in a scratch JS
module — they consume only `_v` and `Acko.Palette[name][index]`.

There are **20 `TrackData` entries**. Field reference is at the
bottom of this README.

### Step 2 — port the four geometry functions

All four live around offsets 510000–518000 in `combo.min.js`. They use
shared module-scope scratch `THREE.Vector3` instances (`j`, `h`, `g`,
`e`, `b`, `a`) that you can mirror as local scratch in a re-implementation.

#### `o(width, height, power, n)` — superellipse cross-section (~10 LOC)

Build `n` vertices around the tube cross-section:

```
for k in 0..n:
    θ = k * 2π/n + π/n            // staggered, not starting at 0
    v_k = (cos(θ)^power · width, sin(θ)^power · height, 0)
```

`cos(θ)^power` and `sin(θ)^power` go negative; the original `Math.pow`
returns `NaN` for negative bases with fractional exponents (apparently
either harmless visually in the original, or the negative-quadrant
vertices land outside the visible camera range). The clean re-impl is
sign-preserving: `sign(cos(θ)) · |cos(θ)|^power`. Same for sin.

Then compute per-vertex 2D normals as
`normalize(cross(v_{k+1} − v_{k−1}, ẑ))`. The cross with `ẑ` rotates
the chord 90° in-plane.

#### `m(points, detail, smooth, bank, up, relative, warp, spring, refine, simplify)` — spline curve + frame (~180 LOC)

This is the spline that turns control points into a smooth curve plus
per-sample reference normals.

**Phase 1 — expand each control point into a quarter-arc.** For each
control point `p[i]` with `prev = p[i-1]`, `next = p[i+1]`:

- Let `smooth_weight = p[i][2]` and `tension = p[i][3] || detail`.
- If `prev` and `next` exist and `smooth_weight > 0`, sample a
  `tension`-step quarter-arc:
  ```
  j = prev_pos − cur_pos
  h = next_pos − cur_pos
  af = min(smooth_weight, min(|j|, |h|) / 2)
  j *= af / |j|;  h *= af / |h|
  g = cur_pos + j + h
  for k in 0..tension:
      θ = k · (π/2) / (tension − 1)
      a = cos(θ)^0.8;  b = sin(θ)^0.8
      out_pos = g − a·h − b·j + relative
      emit(out_pos)
  ```
- Otherwise (endpoint or zero smoothing) emit a single
  `cur_pos + relative` point.

The `^0.8` weight is a tuneable curvature roll-off; it's not standard
Hermite. Verbatim port.

**Phase 1.5 — apply z-warp if `warp` is true.** Before each emit:
`out_pos.z += cos(out_pos.x / 250) · 30`.

**Phase 2 — drop near-coincident samples.** Walk the result; any
sample whose neighbor is < 1 unit away gets removed.

**Phase 3 — Laplacian smoothing.** Do `smooth` passes; each pass walks
the interior samples and moves each sample by
`spring · ((prev − cur) + (next − cur))`. `spring` defaults to 0.3.

**Phase 4 — adaptive refinement (if `refine`).** Two passes. At each
sample, compute the turn angle `asin(|tangent_in × tangent_out|)`. If
the angle exceeds **9° (in radians)**, insert two Catmull-Rom
interpolated points at parameters 0.5 and (originally 0.5 again — yes,
verbatim there's a `for(an=1/2; an<1; an+=1/2)` loop that runs once).
Catmull-Rom is:
```
B = (A − D) · 0.5;  y = (z − C) · 0.5
interp(t) = t³·(2(C−A) + B + y) + t²·(−3(C−A) − 2B − y) + t·B + C
```
where `D, C, A, z` are the four neighbor positions and `t ∈ (0, 1)`.

**Phase 4.5 — adaptive simplification (if `simplify`).** Drop samples
whose accumulated turn angle is below **1° (radians)**.

**Phase 5 — parallel-transport frame (if `bank`).** Initialize
`normals[0] = up.normalized()`. Then for each interior sample compute
the rotation axis between `tangent_in` and `tangent_out`:
```
axis = normalize(tangent_in × tangent_out)
θ    = asin(|tangent_in × tangent_out|)
```
Apply Rodrigues' rotation matrix (3×3) to the running `normal`, store
result. Then two Laplacian smoothing passes with spring 0.3 over the
normals (each pass also renormalizes).

#### `getMatrices(t)` — frame at curve parameter `t` (~30 LOC)

Given the sample arrays `path.vertices[]` and `path.normals[]` (output
of `m`), produce the 4×4 transform at floating-point parameter `t`:

```
E = floor(clamp(t, 0, len − 1.0001))
x = t − E                                 // fractional part
// tangent: a 3-tap centered Catmull-Rom-style finite diff, lerped
if (E == 0 && x ≤ 0.5) || (E == len − 2 && x > 0.5):
    tangent = path.vertices[E+1] − path.vertices[E]
else if x > 0.5:
    tangent = lerp(path.vertices[E+1] − path.vertices[E],
                   path.vertices[E+2] − path.vertices[E+1],
                   (x + 0.5) − 1)
else:
    tangent = lerp(path.vertices[E]   − path.vertices[E-1],
                   path.vertices[E+1] − path.vertices[E],
                   x + 0.5)
ref_normal = lerp(path.normals[E], path.normals[E+1], x).normalized()
position   = lerp(path.vertices[E], path.vertices[E+1], x)
tangent   = tangent.normalized()
binormal  = cross(tangent, ref_normal)
true_normal = cross(tangent, binormal)        // re-orthogonalize

mat4 columns = (binormal, true_normal, tangent, position)
mat3         = (binormal, true_normal, tangent)
```

`mat3` rotates 2D-profile normals into world space; `mat4` transforms
2D-profile positions.

#### `getOutline(C, M, x, G, F, dst_row)` — write one ring (~25 LOC)

Builds one cross-section ring at curve parameter `C`:

```
mat4, mat3 = getMatrices(C)
tangent = (mat3[6], mat3[7], mat3[8])     // 3rd column = Z axis
scale = (G, F, 1)                         // (width_scale, height_scale, 1)
for k in 0..n:
    pos_2d = profile.vertices[k] · scale     // componentwise
    pos_2d.z += M                            // optional thickness offset
    world_pos = mat4 · (pos_2d, 1)
    world_normal = (x ? sign(x)·tangent : mat3 · profile.normals[k])
    write {world_pos.xyz, ao_factor, world_normal.xyz, 1.0} to dst_row
```

`M` is a thickness offset (used for caps), `x ∈ {-1, 0, +1}` chooses
between profile-normal (0) and ±tangent-as-normal (caps). `G` and `F`
are scale multipliers (typically `1, 1`; arrows and caps shrink them).

For the static masthead bring-up we only emit `outline` rings — no
caps, no arrows. Walk `t` from `track.travel` to `track.travel +
track.length`, stepping `1` unit at a time (use the curve's arc-length
parameterization).

#### AO factor

Tracks with `castAO` non-false generate per-segment occlusion discs
(`generateDiscs` / `AmbientOcclusion`) and bake the result into each
vertex's `pos.w` slot. The discs are pre-computed once per page load.

A faithful port can either:
- bake AO into `vertex_colors` (or a parallel `ao_factor` const array)
  during extraction, or
- skip and pass `ao_factor = 1.0` initially, accepting flatter shading.

Skipping is fine for the first cut; revisit in a follow-up if the
masthead silhouette reads too flat.

### Step 3 — emit the `.bin` files

After running the geometry pipeline for each track, flatten all
vertices into three parallel `N`-entry arrays and write each as a
flat little-endian float32 file (see **`.bin` format** above):

- `positions.bin` — `vec4f32` `(x, y, z, ao)` per vertex
- `normals.bin` — `vec4f32` `(nx, ny, nz, 1.0)` per vertex
- `vertex_colors.bin` — `vec4f32` `(r, g, b, 1.0)`, the track's
  palette color (0–255 → 0–1) copied to every vertex of that track

`N` is `sum over all rendered tracks of (ring_samples · profile_n)` —
62792 for the committed data. Geometry goes in storage buffers, not
SPIR-V constants: a 62k-vertex const array compiles to a ~4.7 MB
module that hangs the GPU driver on shader load. As `.bin` files
uploaded to storage buffers, the `scene.wyn` module stays ~4.5 KB
regardless of vertex count.

## TrackData field reference

| Field | Type | Meaning |
|---|---|---|
| `n` | int | Cross-section vertex count (8 for all tracks). |
| `power` | float | Superellipse exponent: smaller = squarer cross-section. |
| `width`, `height` | float | Cross-section half-extents. |
| `relative` | vec3 | Constant offset added to every curve position. |
| `points` | array | Control points: `[vec3, null, smooth_weight, tension?]` per entry. |
| `smooth` | int | Laplacian smoothing pass count. |
| `detail` | int | Default tension (arc subdivisions per control-point quarter-arc). |
| `up` | vec3 | Reference up vector for parallel-transport frame. |
| `bank` | bool | Build frame normals (parallel-transport). Default true. |
| `warp` | bool | Apply z-warp `cos(x/250)·30`. Default true. |
| `refine` | bool | Adaptive curvature refinement. Default true. |
| `simplify` | bool | Adaptive curvature simplification. Default true. |
| `spring` | float | Laplacian step factor. Default 0.3. |
| `color` | `[r,g,b]` | Per-track RGB ∈ [0, 255] from `Acko.Palette.{red,platinum,blue,slate}[i]`. |
| `castAO`, `receiveAO` | float\|false | AO disc generation/receipt factor; false disables. |
| `AOStep` | int | AO disc spacing along the curve. |
| `travel`, `length` | float | Visible-segment start and span in curve units. |
| `arrow`, `aspect`, `truncate` | float | Arrowhead shape (out of scope for static port). |
| `edge` | int | Outline width hint (rendering knob). |
| `shift` | float | Visible-window phase shift. |

## Pipeline (for context, when reading `pipeline.json`)

Original acko.net chain: `scene` → `SSAO` → `SSAOUpsample` → `EightBit (CRT)` → `FXAA`.

We do: `scene` → `EightBit (CRT)` only. `SSAO` and `FXAA` are
deferred follow-ups; both are additive once render-target
infrastructure is in place.
