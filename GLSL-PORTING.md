# Porting GLSL Shadertoy shaders to Wyn

A reference for translating fragment-shader GLSL (especially
Shadertoy-style) into Wyn. Covers the traps that produce a shader
which compiles and validates but renders wrong.

## File skeleton

```wyn
open f32

#[uniform(set=1, binding=0)] def iResolution: vec3f32
#[uniform(set=1, binding=1)] def iTime: f32
#[uniform(set=1, binding=5)] def iMouse: vec4f32   -- if needed

def verts: [3]vec4f32 =
  [@[-1.0, -1.0, 0.0, 1.0],
   @[3.0,  -1.0, 0.0, 1.0],
   @[-1.0,  3.0, 0.0, 1.0]]

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vertex_id: i32)
  #[builtin(position)] vec4f32 = verts[vertex_id]

#[fragment]
entry fragment_main(#[builtin(position)] fragCoord: vec4f32)
  #[location(0)] vec4f32 =
  let fc = @[fragCoord.x, iResolution.y - fragCoord.y] in   -- Y FLIP
  ...
```

## Y-flip (critical)

Vulkan's framebuffer Y points **down** (top of screen is `y=0`),
Shadertoy's GLSL `fragCoord.y` points **up** (bottom of screen is
`y=0`). Compute `fc = vec2(fragCoord.x, iResolution.y - fragCoord.y)`
**before** anything else in `fragment_main`. Use `fc` throughout —
not raw `fragCoord`. Forgetting this renders the shader upside-down.

```wyn
let fc = @[fragCoord.x, iResolution.y - fragCoord.y] in
let uv = (2.0 * fc - iResolution.xy) / iResolution.y in   -- or whatever the original GLSL did
```

## Matrix literals: outer arrays are columns, not rows

This is the most subtle gotcha. Wyn's parser comment at
`parser.rs:2255` says "elements are row arrays," but at SPIR-V
emission each outer array becomes an `OpCompositeConstruct` operand —
which SPIR-V interprets as a **column-vector**. Net effect:

```wyn
@[[a, b], [c, d]]
```

produces the matrix

```
[a   c]
[b   d]
```

**not** `[[a, b], [c, d]]` as the row-major intuition suggests.

Practical consequence: the natural GLSL→Wyn translation
`mat2(c, s, -s, c)` → `@[[c, s], [0.0 - s, c]]` is **correct**. Both
languages happen to coincide because both treat the constructor /
literal as "list of columns," even though Wyn's syntax visually looks
like rows.

**Don't try to "fix" this** to `@[[c, -s], [s, c]]` thinking outer =
rows — you'll transpose every matrix, including all rotations, which
reverses their direction.

**Verification recipe** if in doubt: write a minimal dot-orbit
shader where a dot at `(0.5, 0)` is rotated by `rot(iTime)`. Watch
the screen direction:

```glsl
// GLSL
mat2 rot(float a) { float c=cos(a),s=sin(a); return mat2(c,s,-s,c); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5*iResolution.xy) / iResolution.y;
    vec2 p = vec2(0.5, 0.0) * rot(iTime);
    float d = length(uv - p) - 0.05;
    fragColor = vec4(vec3(smoothstep(0.0, 0.005, -d)), 1.0);
}
```

```wyn
-- Wyn (with the Y-flip)
def rot(a: f32) mat2f32 =
  let c = cos(a) in let s = sin(a) in
  @[[c, s], [0.0 - s, c]]
#[fragment]
entry fragment_main(#[builtin(position)] fragCoord: vec4f32) #[location(0)] vec4f32 =
  let fc = @[fragCoord.x, iResolution.y - fragCoord.y] in
  let uv = (fc - 0.5 * iResolution.xy) / iResolution.y in
  let p = @[0.5, 0.0] * rot(iTime) in
  let d = magnitude(uv - p) - 0.05 in
  let i = smoothstep(0.0, 0.005, 0.0 - d) in
  @[i, i, i, 1.0]
```

At `iTime ≈ π/2` the dot should be at the **bottom** of the screen in
both languages (motion is clockwise as iTime increases).

## `mod`: matches GLSL

Wyn `vec.mod(x, y)` and scalar `f32.mod(x, y)` both lower to SPIR-V
`OpFMod` / WGSL `x - y * floor(x/y)` — the sign of the result matches
the **sign of `y`**. Same as GLSL `mod`. Safe to use directly:

```wyn
let pos_tiled = vec.mod(pos - @[2.0, 2.0, 2.0], @[4.0, 4.0, 4.0]) - @[2.0, 2.0, 2.0] in
```

**Don't** confuse this with C-style `%` (sign of dividend), which
would produce a discontinuity at `x = 0` and visible flicker on tiled
SDFs.

## Vector operations

| GLSL                  | Wyn                           |
|-----------------------|-------------------------------|
| `length(v)`           | `magnitude(v)`                |
| `abs(v)` (scalar)     | `abs(x)`                      |
| `abs(v)` (vector)     | `vec.abs(v)`                  |
| `mod(v, m)`           | `vec.mod(v, m)`               |
| `fract(v)` (vector)   | `vec.fract(v)`                |
| `floor(v)` (vector)   | `vec.floor(v)`                |
| `sign(v)` (vector)    | `vec.sign(v)`                 |
| `max(a, b)` (vec×vec) | `vec.max(a, b)`               |
| `atan(y, x)` (2-arg)  | `atan2(y, x)`                 |

Implicit scalar→vector broadcasts in GLSL (`v + 1.0`) don't exist in
Wyn — write `v + @[1.0, 1.0, 1.0]` (or whatever shape matches).
Scalar-scalar ops work fine; `vec * scalar` and `scalar * vec` also
work without a broadcast.

## Loops: no `break`

Wyn has two bounded loop forms (`SPECIFICATION.md:843`+):

```
loop pat = init for x < n do body      -- foldl over 0..<n, n known
loop pat = init for x in arr do body   -- foldl over an array
loop pat = init while cond do body     -- conditional, manual counter
```

You cannot combine them. There is no `break`. To bound iteration AND
allow early exit (e.g. raymarcher hit detection), use the manual-
counter `while` form with a `done: bool` in the state and an `&&
!done` in the condition:

```wyn
let (_, _, t_final, _) =
  loop (i, p, t, done) = (0, ro, 0.0, false)
  while i < 100 && !done do
    let d = scene_map(p) in
    if d < 0.001 then (i + 1, p, t, true)
    else (i + 1, p + rd * d, t + d, false)
in
```

If no early exit is needed, the cleaner `for i < n` form drops the
counter from the state tuple:

```wyn
let acc =
  loop a = init for i < n do
    body_using_a_and_i
in
```

## `with .xy *= rot(...)` swizzle-assign

Wyn supports the GLSL `v.xy *= mat2` idiom directly via `with` syntax:

```wyn
let v_rotated = v with .xy *= rot(angle) in
let v_step2  = v_rotated with .yz *= other_rot in
```

This is the natural translation of GLSL `v.xy *= rot(a); v.yz *= other;`.

## `lookat` mat3

The GLSL idiom `mat3(rt, cross(rt, dir), dir)` is column-major:
columns are `rt`, `up2`, `dir`. The Wyn translation:

```wyn
def lookat(dir: vec3f32) mat3f32 =
  let rt = normalize(cross(dir, @[0.0, 1.0, 0.0])) in
  let up2 = cross(rt, dir) in
  @[[rt.x,  rt.y,  rt.z ],
    [up2.x, up2.y, up2.z],
    [dir.x, dir.y, dir.z]]
```

is correct — same numerical matrix as the GLSL via the
columns-as-outer-arrays convention above. Apply as `v * lookat(dir)`
to match GLSL's `lookat(dir) * v`.

## Type conversion

`f32.i32(i)` casts `i32` → `f32`. `i32.f32(x)` truncates `f32` → `i32`.
The integer loop counter from `for i < n` is i32; use `f32.i32(i)` to
get a float for math.

## Uniforms

The playground convention (see `creation.wyn`, `seascape.wyn`, etc.):

```wyn
#[uniform(set=1, binding=0)] def iResolution: vec3f32
#[uniform(set=1, binding=1)] def iTime: f32
#[uniform(set=1, binding=5)] def iMouse: vec4f32
```

These three are populated by the playground host. `iTimeDelta` isn't
available — fall back to a fixed `dt = 0.0` or restructure away from
delta-time motion.

## What's not portable

Texture inputs (`iChannelN`, `texture()`, `textureLod()`, audio
samplers): no direct equivalent. Drop the texture-dependent paths or
substitute with procedural noise. If the original has a `#define
disable_sound_texture_sampling` ifdef that turns texture sampling
into zero, follow that branch.

Anti-aliasing supersampling, stereo mode, `iTimeDelta` motion-blur
jitter: drop them.

Variable-length arrays: Wyn has fixed-size `[N]T` only. If a GLSL
shader uses `vec3[]` or dynamically-sized arrays, refactor to a
fixed-size buffer or unroll.

GLSL `out` parameters: translate to tuple returns. `void f(vec3 p,
out vec3 a, out vec3 b, out vec3 c)` becomes
`def f(p: vec3f32) (vec3f32, vec3f32, vec3f32)`.

GLSL mutable globals (e.g. `gTime` that's written by `mainImage` and
read by `de()`): thread the value explicitly as a function argument.

## Diagnosing visual differences

When the output looks wrong:

- **Upside down**: missing Y-flip. Add the `fc` rewrite at the top
  of `fragment_main`.
- **Spinning the wrong way**: probably **not** a matrix-convention
  mismatch (those coincide between Wyn and GLSL). Suspect a sign on
  an angle argument, or a typo in a rotation direction in your
  own code.
- **Tiled-SDF flicker**: check that `vec.mod` is what you used, not
  C-style `%`-style modulo. Discontinuity at cell boundaries causes
  high-frequency flicker.
- **Sphere/SDF clipping through walls**: trace the offset math
  carefully. Often a `+ (fparam - (2 - fparam))` style expression
  with `fparam = 0` collapses to a non-zero offset that's easy to
  miss.
- **Missing surface features**: check whether the GLSL has an
  `if (length(p) < r)` block that mutates the SDF (`sdf += pattern
  * 0.1`). Such conditional thickening is easy to drop in a port.
- **Different camera angle**: verify the `lookat` direction conventions
  and the sign of forward-direction computation (`adv - hpos` vs
  `hpos - adv`).
