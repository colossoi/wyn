# viz

GPU shader runner for Wyn-compiled SPIR-V and WGSL modules.

## Commands

```
viz pipeline <shader.spv>    # Run from the pipeline descriptor JSON
                             # (defaults to the sibling <shader>.json that
                             # `wyn compile` writes; -p overrides).
                             # Graphics descriptor -> interactive window;
                             # compute-only -> headless with --input/--output.
viz run <shader.spv>         # Alias for `pipeline`
viz validate <shader.spv>    # Validate a SPIR-V module (headless naga)
viz info                     # Show GPU device info
viz testpattern              # Render a built-in test pattern
```

## Cross-frame feedback

Interactive pipelines can feed an authored entry result back into one of that
entry's inputs on the next frame. Put that host policy in a versioned
`<shader>.viz.json` sidecar:

```json
{
  "version": 1,
  "feedback": [
    {
      "entry": "particles",
      "input": "prev_pos",
      "result": 0,
      "initial": { "kind": "zero" }
    }
  ]
}
```

Here, result 0 of the authored `particles` entry becomes `prev_pos` on the
following frame. `initial.kind` may be `zero`, `rng`, or `file`; file initial
state uses `{ "kind": "file", "path": "seed.bin" }`, with paths relative to
the sidecar.

`viz pipeline shader.spv` automatically loads `shader.viz.json` when present.
Use `--config FILE` when the sidecar lives elsewhere, or `--no-config` to skip
automatic loading. Feedback selectors deliberately do not use generated
descriptor binding names such as `particles_output`.

## Host-provided images

`--image NAME:FILE` (repeatable) uploads a PNG/JPEG once at startup as
the texture binding named `NAME`. The shader side is a plain entry
parameter — no `resource` declaration:

```wyn
def fragment_main(fragment: fragment_invocation<vec2f32>,
                  input_image: texture2d, samp: sampler) vec4f32 =
  texture_sample(input_image, samp, fragment.value, 0.0)

entry image(input_image: texture2d, samp: sampler,
            screen: render_target<vec4f32>) render_target<vec4f32> =
  let raster = rasterize_triangles(direct_draw(3u32, 1u32), vertex_main) in
  shade(screen, raster,
    |fragment| fragment_main(fragment, input_image, samp))
```

```
viz pipeline shader.spv --image input_image:photo.png
```

The texture is Rgba8Unorm at the file's native size; sample with
normalized UVs to fit any window. Raw file bytes are uploaded as-is
(sRGB stays sRGB-encoded, Shadertoy-style). Interactive mode only.

## Uniform block values

`--uniform NAME.MEMBER:TYPE=VALUE` (repeatable) writes one member of a
uniform block once at startup, placed by the descriptor's published
member layout (`Binding::Uniform { size, members }`). Value syntax
matches `--push-constant`; use `NAME:TYPE=VALUE` for a bare
scalar/vector uniform. Block buffers are zero-initialized, so unset
members read as zero.

```
viz pipeline shader.spv --uniform c.radius:f32=0.35 --uniform c.tint:f32x2=0.9,0.2
```

The Shadertoy names (`iResolution`/`iTime`/`iMouse`/`iFrame`) keep
their per-frame automatic values. See `testfiles/uniform_block_smoke.wyn`
for a record-uniform pipeline driven this way.

## Texture snapshots

`--dump-texture NAME:FILE` (repeatable) reads back the storage texture
whose binding is named `NAME` (the storage-write view's parameter name)
and writes it as a PNG when the run ends — pair with `--max-frames N`
for a deterministic snapshot without watching the window:

```
viz pipeline shader.spv --max-frames 30 --dump-texture ao_out:ao.png
```

Float formats are clamped to [0, 1] before 8-bit quantization;
`r32float` dumps as grayscale. See `pkg/gtao/test/gtao_demo.wyn` for a
multi-pass pipeline whose AO output is snapshotted this way.

## Miner

The Bitcoin miner moved to **tephra** (the Vulkan compute runner) — see
`extra/tephra/README.md` for the `tephra mine` subcommand, genesis-block
verification, and benchmarking.
