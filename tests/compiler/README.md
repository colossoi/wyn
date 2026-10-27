# Compiler fixtures

See [test placement and the three-second unit-test limit](../../TESTS.md).

Use these fixtures for broader compiler scenarios and CLI/backend behavior. Keep
very small input/error examples in table-driven Rust unit tests, using a shared
helper to check a collection of sources without spawning a process for each.

Add a `.wyn` file under `cases/`. No Rust changes or test registration are needed.
The file is ordinary Wyn source with a leading `--!` block containing TOML expectations:

```wyn
--! command = "check"
--! exit = 2
--! warnings = ["Hole of type i32"]
--! errors = ["type hole inferred as `i32`"]

def value = 5i32 + ???
```

Run all fixtures:

```sh
cargo test -p wyn --test compiler_fixtures
```

Select filenames containing a substring and show per-case output:

```sh
WYN_TEST_FILTER=type_hole cargo test -p wyn --test compiler_fixtures -- --nocapture
```

An empty selection fails. The runner reports failing filenames and continues through
other cases. It runs one worker per available CPU (logical core), capped at 24, with a shared
queue so workers take the next fixture as they finish. If the CPU count is unavailable,
it uses one worker. Fixtures can also be
passed directly to `wyn check` or `wyn build`; the expectation block is just comments.

## Expectations

The block must start on the first line. Every metadata line starts with `--!`;
a blank line ends the block. Unknown fields are rejected.

| Field | Meaning |
| --- | --- |
| `command` | Required: `"check"` or `"build"`. |
| `exit` | Required exact process exit code, including `2` for type holes. |
| `targets` | Required nonempty list for builds: `"spirv"`, `"wgsl"`, or both. Forbidden for checks. |
| `args` | Optional additional CLI arguments, e.g. `["--graphics"]`. Warnings default to limit 100; `--max-warnings` here overrides that limit. |
| `source_path` | Optional relative path to copy the source to inside the isolated working directory, e.g. `"source files/scene.wyn"`. The CLI receives that relative path. |
| `output_path` | Optional relative shader output path, e.g. `"generated shaders/compiled graphics.shader"`. Defaults to `"output.{target}"`. Parent directories are created. |
| `emit_mir` | Optional `true` on builds to request `output.ssa` alongside the shader. |
| `artifacts` | Optional array of artifact expectations, described below. |
| `errors` | Ordered list of expected diagnostic-message prefixes, excluding `error: `. Count must match exactly. Defaults to `[]`, requiring no errors. |
| `warnings` | Same for `warning: `. `[]` requires no warnings. Omission leaves warnings unconstrained. |
| `contains` | Additional required stderr substrings, such as omitted-warning counts or source locations. |
| `excludes` | Forbidden stderr substrings. |

Match diagnostic categories and the wording relevant to the regression. Prefer a
short `errors` prefix plus `contains` for the specific reason; avoid module IDs,
debug token/type representations, and unrelated wording. Error counts and order
remain exact.

Every successful build must emit a nonempty shader, pass validation, and have at
least one affirmative artifact-content expectation for each target. Existing
fixtures check the declared entry names in the emitted host program; this also
catches missing or mismatched source entries even when the shader is valid. Empty or whitespace-only
shader output fails even if a validator would accept it. SPIR-V uses
`spirv-val` when available, otherwise Naga parsing and validation. A validator
rejection or a launch error other than “not found” fails the case. WGSL uses Naga.
An unsupported Naga feature fails rather than silently skipping validation.

Combine related positive examples where that retains coverage. Keep independent
negative examples separate so one failure cannot hide another. Build examples need
reachable entries using their results. Build success and validation do not prove
numeric results or internal optimization properties.

`range_bounds.wyn` and `slice_bounds.wyn` use `check` for frontend acceptance.
Related range, slice-borrow, array-helper and specialization regressions retain
`build` coverage. Use `build` when the behavior depends on lowering, a backend,
output artifacts, or build-specific CLI handling; otherwise prefer `check`.

## Artifact expectations

Use `[[artifacts]]` tables inside the same leading comment block when a regression
checks more than compilation and validation:

```wyn
--! command = "build"
--! exit = 0
--! targets = ["spirv", "wgsl"]
--! [[artifacts]]
--! path = "output.wynhost"
--! contains = ["(gpu-dispatch "]
--! [[artifacts]]
--! path = "output.json"
--! absent = true
--! [[artifacts]]
--! path = "output.{target}"
--! targets = ["wgsl"]
--! contains = ["@compute"]

entry main(xs: []i32) []i32 = map(|x| x + 1, xs)
```

`path` is relative to the isolated output directory. `{target}` expands to
`spirv` or `wgsl`; the shader itself is `output.{target}`. By default the file
must exist. `absent = true` requires it not to exist and cannot be combined with
content assertions. `contains` and `excludes` are required/forbidden text substrings. `starts_with`
checks an exact file prefix, such as the host-program header.
An optional `targets` list limits the assertion to those fixture targets; unknown
or inapplicable target names fail. Each target has a fresh output directory, so
one target's sidecar cannot satisfy the other target's expectation.

Group related positive examples in a single source file, with distinct reachable
entries when necessary. Each such entry should appear in an affirmative host-output
check. Type holes may share a source because the compiler reports all of them;
assert the full expected diagnostic count. Keep unrelated early-failing negatives
separate. Keep option variants separate where they intentionally test different
compiler modes. Only `cli_graphics_paths_optimized.wyn` requests `-O`; the runner
never adds it implicitly. Default compilation still performs required lowering,
partial evaluation and cheap cleanup.
