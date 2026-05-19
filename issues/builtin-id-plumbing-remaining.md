# BuiltinId / Overload-Index Plumbing — Remaining Work

## Status

Test status at the head of `master`: 769/769 lib tests pass, 35/35
testfiles validate.

End-to-end identity is now: AST identifier → `NameResolution::Builtin
{ id, overload_idx }` → `tlc::VarRef::Builtin { id, overload_idx }`
→ `egir::PureOp::Intrinsic { id, overload_idx }` → `ssa::InstKind::
Intrinsic { id, overload_idx, args }` → backend dispatches on
`catalog.get(id)` and `catalog.known()`.

The Intrinsic enum has been deleted; `BuiltinLowering` now has a
`ByBuiltinId` marker variant and backends dispatch on
`catalog.known()` BuiltinIds (`uninit`, `array_with`, `length`,
`slice`, `storage_len`, `thread_id`, `storage_index`, `storage_store`).
`ExtInstSplat { ext, splat_args }` keeps its inline data because that
varies per catalog entry.

The remaining work below is *cleanup* — the architecture is correct
end-to-end; these are loose ends.

## What's done (since the last revision of this doc)

- **Phase 3 — converting `compiler_internal!` entries:** done.
  `INTRINSIC_STORAGE_INDEX` / `INTRINSIC_STORAGE_STORE` promoted to
  typed `Intrinsic::*` variants and structurally dispatched at the
  EGIR layer (commit `a29245d`). The Intrinsic-enum unification then
  replaced those typed variants with `ByBuiltinId` + a `known.*` id
  comparison (commit `ea1dea5`). `INTRINSIC_COS` left in place;
  vestigial but no synthesizers, harmless.
- **Synthesised `Call`s of catalog names:** done.
  `buffer_specialize::make_app` (`a29245d`) and
  `parallelize::intrinsic_term` (`1f79102`) now emit `Var(Builtin {
  id, overload_idx: 0 })` for catalog names. Single-overload assert
  mirrors `tlc::build_call`. Every IR-level `App` whose func is a
  catalog name resolves through `BuiltinId` at construction.
- **Bonus, not originally in the doc — Intrinsic enum eliminated**
  (commit `ea1dea5`). `BuiltinLowering::Intrinsic(Intrinsic::*)` is
  gone; the variants collapsed into a single `ByBuiltinId` marker.
  Trade-off: no Rust exhaustiveness on the intrinsic dispatch set; a
  new id surfaces as a runtime "ByBuiltinId dispatch: unknown
  builtin id" bail. `BuiltinCatalog::build` panics on missing
  `known.*` entries at startup as a safety net.

## Phase 4 — sequencing the `resolve_value_name` cleanup

**Partially done; the deeper deletion is blocked, and the right
order matters.**

The original premise was: "after Phase 1, every catalog match has a
`NameResolution` entry, so the `lookup_intrinsic` /
`lookup_module_scheme` fallbacks should be dead — except for
synthetic callers that pass `node_id: None`." We attempted to
remove the FieldAccess `node_id: None` caller and tighten
`resolve_value_name` to require a `NodeId`, then delete the
fallbacks. Removing the FieldAccess caller is safe (modules are
rewritten to qualified `Identifier` by name resolution before
type-check; only genuine record field accesses reach the
`FieldAccess` arm). But deleting the fallbacks broke the prelude.

**Why the fallbacks are load-bearing right now:**
`build_name_resolution` deliberately skips prelude module decls (see
the comment in `name_resolution.rs::build_name_resolution`: "their
existing type-check path is load-bearing"). Identifiers inside
prelude bodies — `f32.pi` referencing `pi`'s sibling, `log10`
calling `log`, the auto-generated `f32.*` /  `f32.+` / `f32.cos`
references after `resolve_opens` qualifies them — therefore reach
the type checker with **no `NameResolution` entry**. The
`lookup_intrinsic` and `lookup_module_scheme` chain is the only path
for these. Deleting it strands the entire prelude.

### Required order

1. **Make `NameResolution` cover every AST body that can be lowered
   to TLC** — including prelude module function bodies, not just
   user modules. This is the prerequisite for everything else; the
   `user_module_names` filter at `name_resolution.rs:354` is the
   load-bearing exception that has to go. (Functor instantiation
   NodeId collision — separate issue file — must land first; that's
   the structural blocker.)
2. **Assert in TLC lowering that every builtin-looking identifier
   has a `BuiltinId`** — no late catalog lookup by string. With
   coverage from step 1 plus the `var_term_builtin_id` migration
   that's already done, this should be enforceable. The assertion
   catches any unconverted path before it silently falls back.
3. **Then, and only then, delete or shrink `lookup_intrinsic`.**
   The catalog escape hatch can come out once step 2 proves nothing
   reaches it for builtin identity.

### `lookup_module_scheme` stays longer

`lookup_module_scheme` is **not** the same category as builtin
identity. It's the checker-level module/type-environment operation
that resolves `M.f` against a module's exposed type bindings —
modules expose user-defined functions and type aliases, not just
catalog builtins. Keep it past Phase 4. Removing it is a separate
question about whether modules-as-type-environments should be
reified into `NameResolution` at all, and the answer there isn't
obviously yes.

**What's actually been deleted:**
- The FieldAccess `node_id: None` caller in `resolve_value_name`'s
  caller list. Verified safe — no test regressions. Not yet
  committed; the work was reverted to keep `master` clean while the
  prelude-coverage extension is sequenced separately. Recreate by
  removing the small block at the top of the `ExprKind::FieldAccess`
  arm of `infer_expression` that calls `try_extract_qual_name`/
  `resolve_value_name(..None)`, then remove `try_extract_qual_name`
  and the unused `ResolvedValue::scheme_for_table` field.

**What's still in place:**
- The `is_qualified` parameter and the `lookup_intrinsic` chain.
  The escape hatch stays until steps 1+2 above are complete.
- `lookup_module_scheme` — likely staying past Phase 4 entirely.

## Structural audit — string-keyed identifier resolution should happen in one place

Stronger invariant than Phase 4: identifier strings should be
resolved to a `BuiltinId` (or a user-defined `SymbolId`) exactly
once, in name resolution / the type checker. Every pass downstream
of TypeChecked — TLC, EGIR, SSA, every backend — should operate on
the structural form (`VarRef::Builtin { id, overload_idx }`,
`PureOp::Intrinsic { id, overload_idx }`, `InstKind::Intrinsic { id,
overload_idx }`) and never match an identifier string against a
catalog name.

**Progress:** synthesised TLC calls (`buffer_specialize::make_app`,
`parallelize::intrinsic_term`) now emit `Var(Builtin)` directly.
The Intrinsic enum is gone; backend dispatch on `BuiltinId` against
`catalog.known()`.

**Still to audit:**
- `wyn-core/src/spirv/mod.rs` — `lower_intrinsic` byname-dispatch
  (and the `InstKind::Intrinsic` else-fallthrough that routes to
  it). Mostly residual after Phase 3.5; some entries still string-
  key.
- `wyn-core/src/wgsl/ssa_lowering.rs` — `if name == INTRINSIC_*`
  guards inside `InstKind::Call` and `InstKind::Intrinsic` arms.
- `wyn-core/src/glsl/ssa_lowering.rs` — `try_emit_array_intrinsic`
  was rewritten to dispatch by `catalog.known()` ids, but the
  generic `lower_intrinsic` and the `InstKind::Call` builtin
  dispatch still funnel through `lookup_by_any_name(name)`. That's
  legitimate (the SSA `Call.func` is a string), but the body should
  switch on the `BuiltinId` rather than re-checking the name.

Pair this with the optional shim cleanup below — most violations
sit at the same backend dispatch sites.

## Optional cleanup — remove transitional shims

After Phase 3 finished and the Intrinsic enum was deleted, several
items rephrase:

- The `BuiltinLowering::NotLowered` arm of SPIR-V's
  `lower_builtin_call` (currently bails with a diagnostic) is dead
  for any builtin emitted through the current synthesis paths.
  Only `hof_intrinsic!` and `compiler_internal!(INTRINSIC_COS, …)`
  produce `NotLowered` today; both are intentional sentinels. Leave
  the bail in place.
- `compiler_internal!`'s 1-arg / 2-arg distinction is moot — both
  produce `BuiltinLowering::ByBuiltinId`. Could collapse to one
  form.
- The else-branch in SPIR-V's `InstKind::Intrinsic` dispatch that
  falls through to `lower_intrinsic` by name is the same string-
  keyed residue that the structural audit covers.

## Potentially-broken multi-overload story for `mul`

`mul` has three overloads with three different `PrimOp`s
(`MatrixTimesMatrix`, `MatrixTimesVector`, `VectorTimesMatrix`).
With overload-index dispatch in place, calling `mul(M, v)` now
correctly picks the `MatrixTimesVector` lowering — but no testfile
exercises this. Add a testfile or unit test that exercises all
three overloads, just to lock in the behaviour we now claim to
support. Cheap, defensive.

## Sequencing

1. **Make `NameResolution` cover every AST body that lowers to TLC**,
   including prelude module bodies. The blocker is the functor
   instantiation NodeId collision (separate issue file) — landing
   that prerequisite is what unsticks this step. Then drop the
   `user_module_names` filter in `name_resolution::
   build_name_resolution`.
2. **Assert in TLC lowering that every builtin-looking identifier
   has a `BuiltinId`.** No late catalog lookup by string. Catches
   any unconverted path before it silently reaches the escape hatch.
3. **Delete the FieldAccess `node_id: None` caller in
   `resolve_value_name`**, tighten the signature to
   `node_id: NodeId`, drop the `is_qualified` parameter.
4. **Now shrink/remove `lookup_intrinsic`.** Only after steps 1–2
   prove nothing reaches it for builtin identity. Keep
   `lookup_module_scheme` — it's a separate concern (module/type
   environment, not builtin identity), and may stay indefinitely.
5. **Structural audit** of downstream string-keyed identifier
   resolution. Most violations sit at the same backend dispatch
   sites; pair with the optional cleanups below.
6. **Optional cleanups** (collapse `compiler_internal!` arity,
   prune SPIR-V's `lower_intrinsic` to just its still-string-keyed
   arms, etc.).
7. **`mul` multi-overload regression test.**

## Verification (each step)

- `cargo build` — clean.
- `cargo test --lib -p wyn-core` — full suite passes.
- `bash scripts/validate_testfiles.sh` — 35/35.
