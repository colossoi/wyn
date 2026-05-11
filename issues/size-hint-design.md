# `#[size_hint(N)]` — design notes for an unused attribute

## Current state (2026-05-11)

The `#[size_hint(N)]` parameter attribute is fully plumbed but has no
consumer. The chain is:

1. **Parser** (`parser.rs:839`) — accepts `#[size_hint(N)]`, produces
   `ast::Attribute::SizeHint(u32)`.
2. **Type checker** — passes the attribute through unchanged.
3. **TLC → EGIR** (`egir/from_tlc.rs:355` via
   `extract_size_hint` at `egir/from_tlc.rs:1976`) — extracts the value
   and stores it on `ssa::EntryInput.size_hint: Option<u32>`.
4. **Backends** (`spirv/mod.rs`, `wgsl/ssa_lowering.rs`,
   `glsl/ssa_lowering.rs`) — never read the field.

It's used at two source sites:

- `testfiles/reduce_compute.wyn:6` — `#[size_hint(1024)] data: []f32`
- `testfiles/scan_compute.wyn:6` — same shape

Both rely entirely on its surface presence; removing the attribute
would not change compiled output.

## What it could mean

Two distinct semantic interpretations are plausible. They unlock very
different optimizations and require very different scaffolding to
implement.

### Interpretation A: dispatch hint (informational)

The compiler treats `size_hint(N)` as a **default** for the host
runtime — a suggested dispatch grid size when the host doesn't supply
one. The actual runtime length is still dynamic; nothing in compiled
output assumes the array has length N.

What this unlocks:

- **Default `default_dispatch_count` in the pipeline descriptor.**
  Today the parallelizer emits `DispatchSize::DerivedFromInputLength
  { workgroup_size: 64 }` (`tlc/parallelize.rs:448`) and the host must
  compute `ceil(actual_length / 64)` workgroups itself. With a hint,
  the descriptor can carry a `default_total_threads: Option<u32>` so a
  thin host runtime can dispatch without inspecting buffer length.
- **Workgroup-size selection.** Currently `LOCAL_SIZE = (64, 1, 1)` is
  baked in for every compute entry (`parallelize.rs:448-449`). With a
  hint:
  - Small inputs (e.g. `size_hint(128)`) → single workgroup,
    workgroup_size = N, skip the two-phase reduce entirely.
  - Medium inputs → keep workgroup=64, scale partials buffer.
  - Large inputs (`size_hint(1M)`) → workgroup=256 for better
    occupancy; size partials buffer accordingly.
- **Partials buffer sizing.** The two-phase reduce hardcodes
  `partials[TOTAL_THREADS=64]` (`parallelize.rs:1108`). With a hint,
  this can scale to the expected workload.

Scope: small, mostly inside `tlc/parallelize.rs`. The pipeline
descriptor needs one new `Option<u32>` field. No language-level
changes. Backends untouched.

### Interpretation B: size assertion (load-bearing)

The compiler treats `size_hint(N)` as a **guarantee** from the user
that the array will always be exactly N elements at dispatch time. The
runtime length becomes effectively static.

What this unlocks:

- **Constant-fold the length** into loop bounds throughout the body.
  Reduce/scan phase loops become bounded with a compile-time constant
  → unrollable, vectorizable.
- **Type-level reconciliation.** A `[]f32` (size-variable) param with
  `#[size_hint(1024)]` could unify with `[1024]f32` at the type level.
  But then why not just write `[1024]f32` and skip the attribute?
- **Fixed-size SPIR-V arrays.** Lower to `OpTypeArray` (composite,
  fixed size) instead of `OpTypeRuntimeArray` (variable, requires
  `StorageBuffer` storage class with decorations). Same in
  WGSL/GLSL — fixed-size `array<f32, 1024>` vs unsized `array<f32>`.

Scope: very large. Touches every backend's array-type lowering, the
storage-binding logic in `egir::from_tlc`, possibly the type checker
if size_hint is intended to unify with a size-variable.

The **key tension** with interpretation B: if the user wants a static
size, the type system already supports it via `[1024]f32`. The
attribute would only be useful for partially-static cases (e.g.
"declare the type as `[]f32` to keep it polymorphic, but optimize
hot-path assuming 1024"). That's a niche, and probably not the
intended use given the existing testfile usage.

## Why this matters / who would benefit

The two existing users — `reduce_compute.wyn` and `scan_compute.wyn` —
read like canonical "compute over a known-size buffer" cases. They
demonstrate the form but get no measurable benefit from it today.

Adoption: zero. If we're going to keep the attribute, it should do
**something** before the next external user sees it; otherwise it's
documentation that lies.

## Recommendation: pursue Interpretation A; rename attribute

I'd implement **Interpretation A** (dispatch hint) and rename the
attribute to make the semantics explicit:

- `#[default_dispatch(N)]` or `#[expected_size(N)]` — same shape,
  honest name. Communicates "default for host runtime" rather than
  "the compiler will optimize assuming this size."

For full Interpretation B semantics, write `[1024]f32` and let the
existing type-level path do the work. There's no good reason to have
a second, weaker mechanism that says "trust me, it's 1024" when
`[1024]f32` says it more cleanly.

## Proposed implementation plan (Interpretation A)

1. **Pipeline descriptor.** Add `default_total_threads: Option<u32>`
   to `ComputePipeline` in `wyn-pipeline-descriptor`. Host runtime
   uses it when the application doesn't override.
2. **Parallelizer workgroup selection.** Replace the
   `LOCAL_SIZE = 64` constant with a function
   `pick_workgroup_size(size_hint: Option<u32>) -> (u32, u32, u32)`:
   - `size_hint < 64` → `(size_hint.next_power_of_two(), 1, 1)`
   - `size_hint in 64..=64K` → `(64, 1, 1)` (current behavior)
   - `size_hint > 64K` → `(256, 1, 1)`
   - `None` → `(64, 1, 1)`
3. **Partials buffer.** Size partials to
   `total_threads / workgroup_size` workgroups instead of hardcoded
   64. Falls out of (2) automatically once
   `LOCAL_SIZE`/`TOTAL_THREADS` become per-entry derived.
4. **Pass size_hint from EntryInput through to the parallelizer.**
   Currently the parallelizer doesn't see `EntryInput`; it works at
   TLC level before EGIR. So either: (a) parse the attribute earlier
   and thread it through TLC, or (b) move parallelization decisions
   to after EGIR conversion. (a) is the smaller change.
5. **Validation.** When multiple input params have `size_hint`,
   require them to agree on N (or define a precedence rule). Emit a
   warning when an entry has size_hint but no dispatch-derived SOAC
   (e.g. a vertex shader with size_hint — meaningless).
6. **Rename.** Decide on `#[default_dispatch(N)]` vs
   `#[expected_size(N)]` vs keep `#[size_hint(N)]`. Update parser,
   docs, the two testfiles.
7. **Tests.** Add a parallelize unit test pinning the workgroup-size
   selection for each size bucket. Add an end-to-end test that
   compiles `reduce_compute.wyn` and inspects the emitted
   `local_size_x` decoration.

## What this is *not*

Not in scope of this design:

- Type-level size unification (Interpretation B).
- Backend changes to fixed-size arrays.
- A new variant of `[]A` that's "almost-static."

Those are larger changes and the user model is murkier; defer them
until there's actual demand.

## Open questions

- Is the rename worth the source breakage in two testfiles? (Yes —
  the name is currently wrong-by-implication.)
- Should the parallelizer fall back to `LOCAL_SIZE = 64` for
  graphical entries (vertex/fragment) that get lifted to compute?
  Those don't have a meaningful input length.
- What about workgroup sizes > 1024? Vulkan / WGSL both cap at
  1024 in some contexts; the size buckets above should respect that.
