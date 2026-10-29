# Compiler fixtures

See [test placement and the three-second unit-test limit](../../TESTS.md).

Use these fixtures for broader compiler scenarios and CLI/backend behavior. Keep
very small input/error examples in table-driven Rust unit tests, using a shared
helper to check a collection of sources without spawning a process for each.

Add a `.wyn` file under `cases/`. No Rust changes or test registration are needed.
The runner also discovers `pkg/sort/test/compiler/` for sort-package fixtures,
which can import the actual implementation within its package boundary.
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
compiler modes. `cli_graphics_paths_optimized.wyn` and `scalar_policy_optimized.wyn` request
`-O`; the runner never adds it implicitly. The scalar-policy pair checks helper
retention/expansion and cheap subtraction cleanup in emitted MIR. Array-layout
and runtime-branch fixtures use the default policy on both shader backends;
they assert compilation and validity, not numerical results. Default compilation still performs required lowering,
partial evaluation and cheap cleanup.

## Host contracts and shader structure

Tests that run substantially the whole compiler belong here even when they inspect
allocation, optimization, or instruction details. The runner now supports these
additional successful-build expectations (using the default output filenames):

- `compare_host = "inputs/alternative.wyn"`: build a second source with the same
  options and output names, validate its shader, and require byte-for-byte equal
  WHL. Both sources must use the same entry names. The comparison starts without
  the first build's shader/host files, so missing output cannot reuse stale files.
- `rust_host = true`: also build with `--target-double rust-wgpu`, validate the fresh
  shader, and require a nonempty Rust host artifact.
- `[[host]]`: select an `entry` and inspect its public WHL declarations or execute
  its host function with a tracing backend. `result_min_bytes = [16, ...]` checks
  each published result capacity; `parameters = N` checks the parameter count.
  Capacity-only checks do not execute GPU-dependent host control flow.
- For host execution, `allocations = [12, 20]` checks the multiset of allocated
  buffer sizes. `inputs = { indices = [0, 1, 2] }` supplies 32-bit words for named
  buffer/host-buffer parameters or named fields in packed host layouts; unspecified buffers are zero-filled to their
  declared minimum size. `scalar_arguments = [3]` supplies scalar parameters in
  declaration order. Texture inputs are opaque handles, not actual textures.
  `draws = ["(:indexed allocation-12-0 :u32 3 2 0 0 0)"]` checks exact executed
  draw operands. `allocation-BYTES-ORDINAL` identifies an allocation among buffers
  of that size; `input-NAME` identifies a supplied buffer (underscores become
  hyphens). Generated draw buffers must be live and written by an earlier dispatch.
- `[[spirv_array_loads]]` selects a source `entry` and positive `minimum`. It counts
  whole-array loads in that entry's generated functions and reachable callees,
  excluding unrelated entries. These checks apply to SPIR-V; WGSL still validates.

The trace executes WHL, **not shaders**. It verifies host allocation, resource
identity, draw parameters, and ordering, but does not prove GPU numerical results
or sorting correctness. Unknown backend operations fail. The harness self-tests
reject missing writers, wrong draw resources, wrong allocation sizes, and array
loads that only occur in unrelated entry points.

Examples: `graphics_draw_buffers*.wyn`, `graphics_aliases*.wyn`,
`array_updates.wyn`, and `array_snapshots*.wyn`. The alias companion lives under
`cases/inputs/`, outside standalone discovery. Group related scenarios while
preserving each contract; keep cases separate where combining them exposes an
unrelated compiler failure rather than weakening validation to make them pass.

For output determinism, `repeat_builds = 3` compares shader bytes from three
independent CLI builds (2 is also supported). The runner removes the previous
shader before each repeat and requires identical error/warning lines (elapsed-time
status messages are excluded). The first shader
is validated normally; byte equality ensures repeats have the same contents.
`spirv_min_block_phis = 2` requires at least two phi instructions in one block,
so `scan_determinism.wyn` cannot pass after losing the structure it exercises.

`vector_constant_math.wyn` groups shader coverage for the constant-folding
examples; their expected numerical constants remain unit assertions.
`graphics_buffer_types.wyn` groups the source-level indexing/helper regressions;
both backends separately unit-test descriptor-slot type isolation without
running the frontend or optimizer.

Additional `[[host]]` checks can inspect declarations without executing the host:
`calls` and `minimum_calls` map WHL operation names to exact/minimum counts within
that entry's function; `host_inputs` checks whether CPU dependencies exist (and
that their byte spans are aligned); `result_ownership`, `result_names`,
`first_result_owned`, `no_buffer_inputs`, and `texture_types` check the corresponding
public declaration properties. `result_layouts` contains quoted WHL layout forms.

Execution checks include `result_bytes`, `result_at_least_bytes`, `result_words`
(32-bit bit patterns), `dispatches`, and `no_device_reads`. `captured_results` is a
boolean per returned tuple element indicating whether a draw reads that resource.
`texture_draw_chain` checks two draws writing the returned scene, a later dispatch
reading that scene and writing the returned sample buffer, and a final draw reading
that buffer. Buffer views retain their underlying resource identity in these checks.
With `rust_host`, these graphics checks also compare the order of emitted Rust
render/compute calls with WHL and require one host function per source entry.

`[[shader]]` selects a source `entry` and optionally `stage` (`compute`, `vertex`,
`fragment`). It can require `workgroup_size`, reachable `workgroup_storage`, a
`barrier`, an `atomic` operation, sorted `loop_depths`, or
`guarded_loops = [total, unguarded]`. Structural checks include only matching entry
points and reachable helpers, so grouping cases cannot hide a missing operation
behind an unrelated entry. These assertions supplement normal shader validation.

`tests/compiler/library/*.wyn` supplies public-library integration cases to
`wyn-core/tests/dispatch_grid.rs` and `host_contracts.rs`. Run them with
`cargo test -p wyn-core --test dispatch_grid --test host_contracts`.
Explicit grids currently require the library API; the CLI has no grid attribute.
Host-contract cases inspect the published loop-completion interface and generated
Rust allocation/context behavior. Their shared setup lives under
`wyn-core/tests/support/`, outside the unit suite; source preparation is reused
across backend/grid variants.

## Instruction and representation checks

`[[spirv]]` selects an `entry` and its reachable helpers. `counts` and `maximum`
map SPIR-V opcode names to exact/maximum counts; `present` and `absent` require or
forbid opcodes. `ext_inst = { Sin = 0 }` checks extended instructions.
`array_traffic = [allocations, loads, stores]` counts function-local whole-array
traffic. `array_phis`, `array_loads`, and `minimum_array_allocations` provide
individual checks. `spirv_helpers` counts non-entry functions across the module;
keep separate fixtures when helper sharing across entries is the property tested.
`spirv_round_trip = true` re-emits each entry with Naga and rejects explicitly
laid-out function-local storage. `shader_entries` checks the SPIR-V entry count.

`[[shader]]` also accepts `targets`, `entry_count`, `binary` and `math` operation
counts, `selects_present`, `ifs_present`, and `local_array_traffic`. For WGSL,
`parameter_loads = { "0" = 1 }` checks loads through kernel parameter zero,
resolving its actual binding through emitted WHL, and
`workgroup_array_scalar = "u32"` requires reachable workgroup arrays with that
scalar element type. All these checks are scoped to the named entry and reachable
helpers, so unrelated entries cannot satisfy them.

`wyn-core/tests/emitted_contracts.rs` uses external `.wyn` sources for public host
layout, Rust-signature, direct-capacity, and render-target resource/use-chain
contracts that need more than the text fixture assertions. It shares frontend/SSA
work across backends. The fluid-neighbor case reads the actual package sources.

`[[shader]]` can also require `parallel = true` (some matching entry has more than
one invocation per workgroup), `subgroup_inclusive_add = true/false` on reachable
shader functions, `workgroup_arrays_scalar = true`, and
`maximum_entry_loops = N`. The last check counts loops in each matching entry body.
`[[spirv]]` supports `minimum_phi_inputs`, `structured_merges = true` (each
selection merge precedes a conditional branch or switch), and
`aggregate_forwarding = true` (no repeated aggregate operations within a block or
extract directly from a constructed aggregate). Extended instruction counts
support `Normalize` as well as `Sin`.

`wyn-core/tests/host_execution.rs` executes emitted WHL with a tracing backend for
runtime allocation/length, dispatch, aliasing, and scalar-result contracts. Sources
live in `tests/compiler/library/` or reuse existing testfiles. Backend variants
share source preparation and SSA; multiple input values reuse one emitted program.
It does not execute GPU shaders. `host_stage_contracts.wyn` groups ten related
storage and dispatch cases under the existing emitted-artifact harness.
