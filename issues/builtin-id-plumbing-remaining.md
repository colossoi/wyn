# BuiltinId / Overload-Index Plumbing — Remaining Work

## Status

Phases 1–3.5 of the plan in `~/.claude/plans/greedy-baking-elephant.md`
are committed (commits `f35ddef`, `78e0ed5`, `f958443`, `2ba3e4f`).

End-to-end identity is now: AST identifier → `NameResolution::Builtin
{ id, overload_idx }` → `tlc::VarRef::Builtin { id, overload_idx }`
→ `egir::PureOp::Intrinsic { id, overload_idx }` → `ssa::InstKind::
Intrinsic { id, overload_idx, args }` → backend dispatches on
`def.overloads()[overload_idx].lowering`.

Test status at the head of `master`: 769/769 lib tests pass, 34/34
testfiles validate.

The remaining work below is all *cleanup* — the architecture is
correct end-to-end; these are loose ends that were tracked but not
finished.

## Phase 3 — finish converting `compiler_internal!` entries

Three `compiler_internal!(NAME, Purity::Pure)` (NotLowered) entries
remain in `wyn-core/src/builtins/defs.rs`. Each conversion follows
the recipe used for `slice` / `storage_len` / `thread_id`:

1. Add `Intrinsic::X` variant in `builtins/lowering.rs`.
2. Switch the catalog entry to the 3-arg `compiler_internal!` form
   (`compiler_internal!(NAME, Purity::Pure, Intrinsic::X)`).
3. Move the body from `lower_intrinsic`'s `"x"` arm in
   `spirv/mod.rs` into `lower_builtin_call`'s `Intrinsic::X` arm.
4. Add `Intrinsic::X` to the `typed_dispatch` `matches!` in the
   `InstKind::Intrinsic` arm so it routes through `lower_builtin_call`.
5. Delete the `"x"` arm from `lower_intrinsic`.
6. Add `Intrinsic::X` exhaustiveness arms in `glsl/ssa_lowering.rs`
   (and `wgsl/ssa_lowering.rs` if it has a typed `Intrinsic` match —
   currently it dispatches by name only).

### `INTRINSIC_COS`

Vestigial. The user-facing `cos` is registered via
`polymorphic_intrinsic!("cos", INTRINSIC_COS, scalar_unary,
PrimOp::GlslExt(14))`, and the per-type `f32.cos` / `f64.cos` /
`vec.cos` entries are also `PrimOp(GlslExt(14))`. The
`compiler_internal!(INTRINSIC_COS, Purity::Pure)` entry only matters
if some pass synthesises `_w_intrinsic_cos` directly (rather than
calling the polymorphic one).

**Action:** grep for `INTRINSIC_COS` emission sites. If unused,
delete the `compiler_internal!` entry. If used, promote to a typed
variant — but it should just be `PrimOp(GlslExt(14))` since it has
no special operand handling.

### `INTRINSIC_STORAGE_INDEX`

Special-cased in `convert_named_app` (`egir/from_tlc.rs:~1081`):
when seen with three args, emits an `InstKind::Load` side-effect
through a `ViewIndex` place. So the catalog entry's `BuiltinLowering`
is never consulted by the backend along the normal path.

**Action:** verify with grep that `_w_intrinsic_storage_index` is
*only* emitted by paths that go through that special arm. If so,
the `compiler_internal!` entry has no backend code path and can
either:
- (a) Stay as `NotLowered` with a comment explaining why (the
  `BuiltinId` is still useful for `lookup_by_any_name` parity).
- (b) Get promoted to a typed `Intrinsic::StorageIndexErased` variant
  whose backend arm bails with a clear message — defensive, surfaces
  any future regression where the special arm misses.

(b) is consistent with the rest of the conversion work but adds a
typed variant that should never fire. (a) is honest about the
status. Either is fine; (a) is the smaller change.

### `INTRINSIC_STORAGE_STORE`

Same shape as `STORAGE_INDEX`: special-cased in `convert_named_app`
to emit a `Store` side-effect. Same analysis and same options.

## Latent bug — synthesised `Call`s of catalog names

While debugging `array_param_view_slice.wyn`, MIR showed *both*
forms for `_w_intrinsic_storage_len`:

```
%5 = intrinsic @_w_intrinsic_storage_len#0(...)   # the Phase 3.5 path
%3 = call      @_w_intrinsic_storage_len(...)     # synthesised path
```

The `Call` form is emitted by `buffer_specialize::make_app` (which
constructs an `App` whose func is `Var(Symbol(...))`). It currently
*works* because `lookup_lowering(name)` returns
`Intrinsic(StorageLen)` (a single-overload entry, so `[0]` is
always correct), and SPIR-V's `InstKind::Call` arm routes through
`lower_builtin_call` which handles `Intrinsic(StorageLen)` correctly.

For multi-overload catalog names, the same synthesised path would
silently dispatch overload `[0]` regardless of intent — same class
of bug as the pre-Phase-3.5 `lookup_lowering` issue.

**Action:** audit `buffer_specialize::make_app` and
`parallelize::make_intrinsic_term` (and similar synthesised-call
helpers) to emit `Var(Builtin { id, overload_idx })` directly when
the name resolves to a catalog entry. Mirror what `tlc::build_call`
does. After this, every IR-level `InstKind::Call` whose `func` is a
catalog name should be a bug.

## Phase 4 — remove string fallbacks in `resolve_value_name`

`types/checker.rs` `resolve_value_name` falls back to
`lookup_intrinsic(full_name)` and `lookup_module_scheme(full_name)`
when `name_resolution.get(node_id)` returns `None`. After Phase 1,
every catalog match has a `NameResolution` entry, so these fallbacks
should be dead — *except* for synthetic callers that pass `node_id:
None`.

### Audit synthetic `node_id: None` callers

Grep `resolve_value_name(...None)` and identify each call site.
Likely candidates:
- `FieldAccess` chain recovery (re-resolves a qualified name without
  a backing `Identifier` AST node).
- Other recovery paths in error reporting.

For each, either:
- Wire a real `NodeId` through, or
- Replace the call with a more direct catalog lookup that doesn't
  pretend to be name-resolving an absent identifier.

### Delete the fallbacks

Once `node_id: None` callers are eliminated:

- `lookup_intrinsic(full_name)` — delete.
- `lookup_module_scheme(full_name)` — for per-type ops whose schemes
  come from `module_schemes`, route through the catalog using the
  `BuiltinId` instead of the surface-name string. Either:
  - **(a)** Have `lookup_module_scheme` accept a `BuiltinId` and key
    on the catalog's `surface_name`. Cheap rename.
  - **(b)** Move per-type-op scheme construction into the catalog
    itself. Deeper; decouples scheme source from name keying.

  Pick (a) for this phase; (b) is a separate cleanup.

Files: `wyn-core/src/types/checker.rs`.

## Optional cleanup — remove transitional shims

After Phase 3 finishes (no `NotLowered` reachable at backend
dispatch):

- The `BuiltinLowering::NotLowered` arm of SPIR-V's
  `lower_builtin_call` (currently bails with a diagnostic) becomes
  unreachable. It can either stay as a defensive bail or be replaced
  with `unreachable!()`.
- The `else` branch in SPIR-V's `InstKind::Intrinsic` dispatch that
  falls through to `lower_intrinsic` by name disappears — every
  variant now goes through `typed_dispatch`. The `lower_intrinsic`
  function itself can shrink to just its still-string-keyed arms
  (whatever's left after the conversions above), or be deleted
  entirely if nothing remains.
- `BuiltinLowering::NotLowered` can be deleted entirely if no
  `compiler_internal!` 2-arg form remains.
- The 2-arg form of `compiler_internal!` can be deleted.

## Potentially-broken multi-overload story for `mul`

`mul` has three overloads with three different `PrimOp`s
(`MatrixTimesMatrix`, `MatrixTimesVector`, `VectorTimesMatrix`).
With overload-index dispatch in place, calling `mul(M, v)` now
correctly picks the `MatrixTimesVector` lowering — but no testfile
exercises this. Add a testfile or unit test that exercises all
three overloads, just to lock in the behaviour we now claim to
support. Cheap, defensive.

## Sequencing

The remaining items are independent. Suggested order:

1. **Synthesised-call audit** (`buffer_specialize::make_app`,
   `parallelize::*`). One mole-style sweep similar to the
   `var_term_matches_name` sweep in commit `2ba3e4f`. Eliminates a
   latent multi-overload bug class.
2. **`INTRINSIC_COS`** — likely just delete.
3. **`INTRINSIC_STORAGE_INDEX` / `STORAGE_STORE`** — option (a):
   document why they stay `NotLowered`.
4. **Phase 4** fallback removal.
5. **Optional cleanups** (NotLowered shim, 2-arg compiler_internal!,
   `lower_intrinsic` shrink/delete).
6. **`mul` multi-overload regression test.**

## Verification (each step)

- `cargo build` — clean.
- `cargo test --lib -p wyn-core` — full suite passes.
- `bash scripts/validate_testfiles.sh` — 34/34.
