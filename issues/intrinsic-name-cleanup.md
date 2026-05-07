# Eliminate `_w_intrinsic_*` String Dispatch

## Context

`_w_intrinsic_*` is the historical compiler-internal name prefix for
catalog builtins (`_w_intrinsic_uninit`, `_w_intrinsic_array_with`,
`_w_intrinsic_storage_index`, …). Every IR pass and backend used to
dispatch on these strings. The catalog now exposes `BuiltinId`-based
identity (`crate::builtins::catalog().known().uninit`, etc.), and
`InstKind::Intrinsic` already carries a `BuiltinId` (not a name).
Migration is half-finished — the goal is to retire string-keyed
dispatch outside `builtins/`, shrinking `_w_intrinsic_*` to just the
catalog-internal name registry where it has genuine semantic meaning
(name mangling, error blame, debug printing).

## Status

- **Commit 1** — *done* (master `ac819a0`): deleted dead Category-5
  intrinsics — `_w_intrinsic_filter`, `_w_intrinsic_scatter`,
  `_w_intrinsic_hist_1d` catalog entries; `convert_soac_filter` in
  EGIR; the `INTRINSIC_FILTER` constant; the three dead
  `*_scheme` builders. Added `testfiles/filter_demo.wyn` as a
  reminder that filter is broken end-to-end (unresolved `?k.` size
  variable panics SPIR-V codegen). `SoacOp::Filter` and its TLC
  passthrough sites stay — separate concern.
- **Commit 2** — *pending*: migrate string-keyed dispatch sites to
  `BuiltinId` comparison.
- **Commit 3** — *pending*: prune unused `INTRINSIC_*` constants and
  the `var_term_matches_name` / `var_term_canonical_name` helpers.

Survey of remaining `_w_intrinsic_` usage in `wyn-core/src/`
(post-commit-1):

- ~180 total `_w_intrinsic_` substring hits.
- ~75 string literals.
- 28 `INTRINSIC_*` constants in `builtins/names.rs`.
- Of those constants, **9** are referenced from non-catalog code:
  `array_with`, `array_with_inplace`, `length`, `slice`,
  `storage_index`, `storage_len`, `storage_store`, `thread_id`,
  `uninit`. All 9 already have `KnownBuiltinIds` fields. **Zero new
  fields needed.**

## Why these are intrinsics — categorisation

Every catalog entry exists for a structural reason that prevents it
from being a regular Wyn function. Understanding which structural
category each falls into shapes the remaining migration:

| Category | What it is | Members | Field needed? |
|---|---|---|---|
| **1. GPU instruction wrappers** | Polymorphic stub for a single GLSL.std.450 / SPIR-V / WGSL builtin op. Lowers via `BuiltinLowering::PrimOp(GlslExt(N))`. No IR-level dispatch needed; the catalog tells the backend what to emit. | `abs`, `ceil`, `floor`, `fract`, `cos`, `clamp`, `mix`, `smoothstep`, `magnitude`, `normalize`, `dot`, `cross`, `distance`, `outer`, `reflect`, `refract`, `determinant`, `inverse` | None |
| **2. Environment / GPU dispatch state** | Reads non-program state from the GPU runtime. | `thread_id` | Already in `KnownBuiltinIds` |
| **3. Storage-buffer protocol (resource boundary)** | Crosses the SPIR-V resource boundary. set/binding constants flow to the backend; effects need ordering. | `storage_index`, `storage_store`, `storage_len` | Already in `KnownBuiltinIds` |
| **4. Array-variant–dispatched primitives** | Wyn arrays have three variants (Composite/View/Virtual); each lowers differently per variant. `array_with`/`array_with_inplace` is coupled to ownership analysis. | `length`, `slice`, `uninit`, `array_with`, `array_with_in_place`, `replicate` | First five already in `KnownBuiltinIds`. `replicate` not externally dispatched — none |
| **5. SOACs that escape the SoacOp envelope** | *Garbage — deleted in commit 1.* | ~~`filter`, `hist_1d`, `scatter`~~ | n/a |
| **6. Bit-level primitives without a Wyn operator** | Needs special expansion at the backend (no single GLSL.std.450 op); polymorphic over `u32` only. | `rotr32` | Not externally dispatched — none |

## Commit 2 — Migrate string-dispatch sites to `BuiltinId`

For every site using `var_term_matches_name(term, syms, INTRINSIC_FOO)`
or `lookup_by_any_name(INTRINSIC_FOO)`, rewrite to compare a
`BuiltinId` against `catalog.known().foo`. Two patterns:

- **At a TLC `Term::App { func, args }` where `func` is a `Var`:**
  Add a small helper in `tlc/mod.rs`:
  ```rust
  pub fn var_term_builtin_id(term: &Term, symbols: &SymbolTable) -> Option<BuiltinId>
  ```
  that returns the `BuiltinId` for both `VarRef::Builtin { id, .. }`
  (fast path) and `VarRef::Symbol` (catalog name lookup via
  `lookup_by_any_name`). Callers compare to
  `catalog.known().<field>` directly.

- **At an SSA `InstKind::Intrinsic { id, .. }`:** already a
  `BuiltinId`; just compare directly.

Files to touch:

- `tlc/ownership.rs` — `ALIASING_INTRINSICS`,
  `FRESH_PRODUCER_INTRINSICS` string slices. Convert to `&[BuiltinId]`
  populated via `known()`.
- `tlc/buffer_specialize.rs` — ~15 hits, all `var_term_matches_name`.
- `tlc/parallelize.rs` — ~7 hits, mostly `intrinsic_term(INTRINSIC_*, …)`
  factory calls. Switch the helper to take a `BuiltinId` and emit
  `Var(VarRef::Builtin { id, .. })`.
- `tlc/soa.rs` — 8 hits, similar shape.
- `tlc/mod.rs` — 6 hits in dispatch helpers.
- `egir/from_tlc.rs` — `INTRINSIC_STORAGE_INDEX`/`STORAGE_STORE`/
  `STORAGE_LEN`/`LENGTH` match arms. Convert to a `BuiltinId` match
  against `catalog.known()`. Error-message strings stay.
- `wgsl/ssa_lowering.rs` — `INTRINSIC_ARRAY_WITH`/`ARRAY_WITH_INPLACE`
  comparisons. Convert to `id == known.array_with` etc. Error-message
  strings stay.

After this commit: `grep -rn 'INTRINSIC_[A-Z]' wyn-core/src/ | grep -v
/builtins/` returns zero hits.

## Commit 3 — Prune unused constants and string helpers

- Remove the 19 Category-1/2/4-`replicate`/6 `INTRINSIC_*` constants
  from `names.rs` that exist only as `defs.rs` registration tokens.
  Either inline as literals in `defs.rs` or keep `names.rs` as the
  canonical name registry. Recommendation: inline, since the strings
  are themselves the catalog-internal protocol and there's no second
  consumer.
- Delete `var_term_matches_name` and `var_term_canonical_name` from
  `tlc/mod.rs` if no callers remain.
- Spot-check that `module_manager::is_intrinsic`'s
  `name.starts_with("_w_intrinsic_")` AST-layer check still holds.
  This pre-resolution check stays — `BuiltinId` isn't available at
  the module-manager boundary. **Out of scope, deferred.**

## What stays as `_w_intrinsic_*` strings (intentional)

- `builtins/defs.rs` — registers names in the catalog. The strings
  *are* the registry.
- `builtins/names.rs` — narrowed to the catalog-seed set (or eliminated
  entirely if commit 3 inlines them).
- Error/diagnostic message strings — user-visible.
- `builtins/lowering.rs::dispatch_name()` — accessor for backends'
  unimplemented-fallback errors.
- Backend mangling (`glsl_mangle`, `wgsl_mangle`) — input is the
  registered name.
- `module_manager::is_intrinsic` — operates on AST `Name`s before
  catalog resolution. Deferred.
- Tests asserting on rendered output (e.g.
  `glsl_mangle("_w_intrinsic_foo")`).

## Critical files

- `wyn-core/src/builtins/names.rs` — pruned in commit 3.
- `wyn-core/src/tlc/mod.rs` — new `var_term_builtin_id` helper in
  commit 2; old string helpers deleted in commit 3.
- `wyn-core/src/tlc/{ownership,buffer_specialize,parallelize,soa}.rs`
  — bulk of the migration in commit 2.
- `wyn-core/src/egir/from_tlc.rs` — `INTRINSIC_*` lookups migrated in
  commit 2.
- `wyn-core/src/wgsl/ssa_lowering.rs` — small migration in commit 2.

## Verification

After each commit:
1. `cargo test --lib -p wyn-core` — 769 pass.
2. `bash scripts/validate_testfiles.sh` — 34/34 + 1 skipped
   (`filter_demo`).
3. `cargo fmt --all` clean.

After commit 3:

4. `grep -rn 'INTRINSIC_[A-Z_]*' wyn-core/src/` returns hits only
   inside `wyn-core/src/builtins/` (or zero, if `names.rs` inlined).
5. `grep -rn 'var_term_matches_name\|var_term_canonical_name'
   wyn-core/src/` returns zero.
6. `grep -rn '"_w_intrinsic_' wyn-core/src/` shows only catalog
   registrations, error-message templates, and tests.

## Risks

- **`var_term_builtin_id` cost** — one extra `HashMap` lookup per
  call site, only on the legacy `VarRef::Symbol` path. Negligible.
- **`module_manager::is_intrinsic` left alone** — relies on the
  `_w_intrinsic_` prefix convention. If we ever rename the prefix,
  this is the holdout. Out of scope.
