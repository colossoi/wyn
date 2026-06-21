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
- **Commit 2 (Phase A)** — *pending*: migrate **emitters** so every
  catalog-builtin call site emits `Var(VarRef::Builtin{id, 0})` and
  no catalog name string survives in synthesised TLC.
- **Commit 3 (Phase B)** — *pending*: migrate **dispatchers** to
  compare `BuiltinId` directly, since Phase A makes the legacy
  `Var(VarRef::Symbol("_w_intrinsic_*"))` form unreachable.
- **Commit 4** — *pending*: prune unused `INTRINSIC_*` constants and
  the `var_term_matches_name` / `var_term_canonical_name` helpers.

### Pipeline boundary for name → BuiltinId resolution

1. **Lex/Parse → AST.** Pure strings.
2. **Type-check + name resolution.** `NameResolution` maps each
   `Var`-position `NodeId` → `ResolvedValueRef`; for catalog entries
   that's `Builtin { id, overload_idx }`.
3. **`to_tlc` (AST → TLC) — the resolution boundary.** User-source
   `Var`s emit `VarRef::Builtin` from the resolution side table.
   *Compiler-synthesised* intrinsic calls flow through three emitter
   helpers — `tlc::build_call`, `parallelize::intrinsic_term`,
   `buffer_specialize::make_app` — each of which already does its
   own `lookup_by_any_name(name)` and emits `Var(Builtin{id, 0})`.
4. **After Phase 3:** every catalog reference inside a `Term` is
   `VarRef::Builtin{id}`. The remaining `VarRef::Symbol` references
   with `_w_*` prefixes are TLC-internal special forms that aren't
   catalog citizens (`_w_index`, `_w_tuple_proj`, `_w_array_lit`,
   `_w_vec_lit`, `_w_tuple`).

### The legacy outlier

`tlc/soa.rs::mk_array_with` (line 1051) calls
`resolve_or_alloc(INTRINSIC_ARRAY_WITH)` and emits
`Var(VarRef::Symbol(aw_sym))` rather than `Var(VarRef::Builtin)`.
This violates the Phase-3 invariant and is the *reason* the
ownership rewriter at `tlc/ownership.rs:1136` has to "match by name
so the rewrite fires whether the call site is a
`Var(Symbol(_w_intrinsic_array_with))` (older synthesised paths) or
a `Var(Builtin(ARRAY_WITH_ID))`". Phase A fixes the emission;
Phase B then deletes the name fallback.

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

## Commit 2 — Phase A: emitters take `BuiltinId`

The three emitter helpers that synthesise intrinsic calls in TLC
already do `lookup_by_any_name(name)` internally and emit
`Var(Builtin{id, 0})`. The `&str` parameter is just a lookup key —
make it a `BuiltinId` directly.

- **`tlc/mod.rs::build_call`, `build_intrinsic_call`** — keep the
  `&str` overload for TLC-internal special forms (`_w_array_lit`,
  `_w_vec_lit`, `_w_index`, `_w_tuple_proj`, `_w_tuple`); add
  `build_builtin_call(id: BuiltinId, ...)` for catalog calls.
  Migrate the three catalog callers
  (`INTRINSIC_ARRAY_WITH`/`LENGTH`/`SLICE`).
- **`tlc/parallelize.rs::intrinsic_term`** — switch to taking
  `BuiltinId` (no special-form callers in this file). Migrate the
  ~7 callers (`THREAD_ID`/`STORAGE_INDEX`/`STORE`/`LEN`).
- **`tlc/buffer_specialize.rs::make_app`** — switch to taking
  `BuiltinId` (audit for any special-form callers; carve a separate
  helper if needed). Migrate dispatch-site emitters that synthesise
  `STORAGE_*`/`LENGTH`/`SLICE` calls.
- **`tlc/soa.rs::mk_array_with`** — *the legacy outlier.* Replace
  the `resolve_or_alloc(INTRINSIC_ARRAY_WITH)` →
  `Var(VarRef::Symbol(...))` emission with a
  `Var(VarRef::Builtin{id: known.array_with, overload_idx: 0})`
  emission. Remove `resolve_or_alloc` if no other callers remain.

After Phase A: any synthesised TLC `Var` referring to a catalog
intrinsic is `VarRef::Builtin{id}`. The `Var(Symbol)` arm of every
downstream `var_term_matches_name(_, _, INTRINSIC_*)` becomes dead.

## Commit 3 — Phase B: dispatchers compare `BuiltinId`

Add helper in `tlc/mod.rs`:
```rust
pub fn var_term_builtin_id(term: &Term, symbols: &SymbolTable) -> Option<BuiltinId>
```
that returns the id for `VarRef::Builtin` (fast path) and falls back
to a catalog name lookup for `VarRef::Symbol` (defensive — should be
unreachable for catalog names after Phase A).

For every site using `var_term_matches_name(term, syms, INTRINSIC_FOO)`
or `lookup_by_any_name(INTRINSIC_FOO)`, rewrite to compare a
`BuiltinId` against `catalog.known().foo`.

Files to touch:

- `tlc/ownership.rs` — split mixed `ALIASING_INTRINSICS` /
  `FRESH_PRODUCER_INTRINSICS` lists: TLC-special-form members
  (`_w_index`, `_w_tuple_proj`, `_w_array_lit`, `_w_vec_lit`,
  `_w_tuple`) stay string-keyed; catalog members
  (`_w_intrinsic_array_with`, `_w_intrinsic_array_with_inplace`)
  switch to `BuiltinId` compare via `known()`. Migrate the
  array_with → array_with_inplace rewriter at line 1140.
- `tlc/buffer_specialize.rs` — ~15 dispatcher hits.
- `tlc/parallelize.rs` — dispatcher hits (most of parallelize's
  hits were emitters, handled in Phase A).
- `tlc/soa.rs` — dispatcher hits in `transform_term` (the
  `INTRINSIC_ARRAY_WITH | INTRINSIC_ARRAY_WITH_INPLACE` match arm
  at line 339, etc.).
- `tlc/mod.rs` — remaining dispatcher hits (the 3 emitter sites
  were migrated in Phase A).
- `egir/from_tlc.rs` — `INTRINSIC_STORAGE_INDEX`/`STORAGE_STORE`/
  `STORAGE_LEN`/`LENGTH` match arms. Convert to a `BuiltinId` match
  against `catalog.known()`. Error-message strings stay.
- `wgsl/ssa_lowering.rs` — `INTRINSIC_ARRAY_WITH`/`ARRAY_WITH_INPLACE`
  comparisons. Convert to `id == known.array_with` etc. Error-message
  strings stay.

After this commit: `grep -rn 'INTRINSIC_[A-Z]' wyn-core/src/ | grep
-v /builtins/` returns zero hits.

## Commit 4 — Prune unused constants and string helpers

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

- `wyn-core/src/builtins/names.rs` — pruned in commit 4.
- `wyn-core/src/tlc/mod.rs` — new `build_builtin_call` emitter in
  commit 2; new `var_term_builtin_id` dispatcher helper in commit 3;
  old string helpers deleted in commit 4.
- `wyn-core/src/tlc/{parallelize,buffer_specialize,soa}.rs` —
  emitters migrated in commit 2; dispatchers migrated in commit 3.
- `wyn-core/src/tlc/ownership.rs` — dispatcher migration in commit 3.
- `wyn-core/src/egir/from_tlc.rs` — `INTRINSIC_*` lookups migrated in
  commit 3.
- `wyn-core/src/wgsl/ssa_lowering.rs` — small migration in commit 3.

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
