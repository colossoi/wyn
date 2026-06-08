# View-array provenance through SOAC captures

## Status

**DONE — Tier 1 complete.** The two view-op walker families are unified into
one (`rewrite_specialized_body` + helpers); the `rewrite_term` family is
deleted. There is a single provenance environment `SymbolId → ViewProv
{ binding, offset: Term, len: Term }`, no `buffer_map`, and **no fabricated
`BindingRef::new(0, n)`** — an unspecialized view-param function is rewritten
against an empty environment, so an unresolvable view op fails downstream
rather than silently reading descriptor `(0,0)`. Gate held throughout: 1029
wyn-core tests, 5 rspirv provenance guards, SPIR-V 73/1 (pre-existing
`conversion_constructor_demo`), WGSL 72/0/2.

**No `ViewSource` enum.** The earlier reverted attempt concluded we'd need a
`Params | WholeBuffer` tag — but that tag just re-encodes the entry/function
split the refactor exists to delete. Storing `offset`/`len` as plain `Term`s
makes the distinction disappear: an entry seeds `offset = 0` / `len =
storage_len(binding)`, a specialized function seeds `offset = Var(offset_sym)`
/ `len = Var(len_sym)`. The one place the split is still observable —
"does the view symbol survive as a value?" — is read off the offset directly
(`is_int_lit_zero`): offset `0` ⟺ entry whole-buffer (symbol survives, leave
the `Var` for egir/parallelize), offset `Var` ⟺ specialized param (symbol
replaced, reconstitute an explicit `StorageView`). The same literal-0 check
suppresses the `0 + i` index form so the entry body's normal form stays
byte-identical to what `parallelize`/size-hint pattern-match.

Slice→composite (the original failing case) remains fixed:
`rewrite_specialized_body` lowers `_w_intrinsic_slice(view, s, e)` with a
composite result to an N-element `ArrayExpr::Literal` of
`storage_index(binding, offset + k)` reads.

### How the reverted-attempt obstacles were resolved

1. **Duplicated SOAC-body handlers.** `try_specialize_soac_view_captures` now
   resolves captures via `try_resolve_view_expr` against the live
   `view_params` and is called from the single merged SOAC-body handler.
2. **Non-transparent restructuring.** Whole-buffer offset/len are inlined at
   the use sites (no `let`-wrapping of the entry body), and the `0 + i` /
   index-coercion forms are suppressed for offset-0 views — so downstream
   pattern-matching sees identical structure.

## Symptom

```wyn
def gather3(arr: [3]f32) f32 = arr[0] + arr[1] + arr[2]

#[compute]
entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) []f32 =
  map(|_:i32| gather3(xs[0..3]), 0i32..<3)
```

…fails at SPIR-V emission:

```
in function '_w_lambda_13': slice_to_composite: no buffer provenance for value ValueId(2v1)
```

`ValueId(2v1)` is the lifted lambda's second parameter — the captured
`xs` view. The slice-to-composite handler in `spirv::mod.rs` looks up
`view_buffer_id[%2v1]` and finds nothing, because `view_buffer_id` is
per-function and the caller never propagated the buffer id across
the function-call boundary.

## Bigger picture: the bug isn't just slice

Replacing `xs[0..3]` with the whole view `xs` makes the program
"validate" cleanly. **But the resulting SPIR-V is wrong** — every
`OpAccessChain` reads from descriptor `(set=0, binding=0)` (the
compiler-allocated output buffer) instead of `(set=2, binding=0)`
(`xs`). Validation passes only because `(0, 0)` is a declared
descriptor. The lifted lambda is silently reading from the wrong
buffer.

Confirmed with `spirv-dis`:

```
OpDecorate %75 DescriptorSet 2  Binding 0   <- xs (input)
OpDecorate %76 DescriptorSet 0  Binding 0   <- output
%98 = OpAccessChain ... %76 ...             <- wrong! reads from output
%104 = OpAccessChain ... %76 ...
%111 = OpAccessChain ... %76 ...
```

So the bug is broader than "the slice loses provenance" — the lifted
lambda's view param has lost provenance even without slicing. The
slice case just fails loudly because `slice_view_to_composite`
needs the buffer id to emit the loads; the non-slice case silently
emits accesses into the wrong descriptor.

## Architecture of the loss

After defunctionalize, a `map`'s lambda becomes a `SoacBody`:

```
SoacBody {
  lam.body = Var(_w_lambda_13),                    // ref to the lifted def
  captures = [(xs_sym, [?]f32, Var(xs))],          // values appended to call args
}
```

The lifted def is a plain `DefMeta::LiftedLambda` with params
`[loop_param, captured_view_param]`. The SOAC body itself doesn't
encode the call — egir/`soac_expand` later emits
`Call(_w_lambda_13, [loop_i, captured_view])`.

`buffer_specialize`'s job is to ensure no plain function or lifted
lambda ends up with a view-typed parameter — view params should be
replaced by `(offset: u32, len: u32)` pairs, with the binding baked
into the body. It does this for plain `App` call sites via
`specialize_call`: when `tick`'s body has `App(f, [xs])` and `xs` is
buffer-backed, it creates `f_b0s2b0` (a per-buffer specialized copy)
and rewrites the call.

**SOAC capture lists go through a different code path.** The SOAC's
captured `Var(xs)` is walked via `rewrite_term`, which sees a bare
`Var` and passes it through unchanged. So the lifted def's view
param stays view-typed, and `specialize_function_body` falls back to
seeding the lifted def's body with a **fake** `BindingRef::new(0, N)`
placeholder. That fake binding ends up baked into the body — which
is why the non-slice case silently reads from `(0, 0)`.

## Two latent bugs in `buffer_specialize`

1. **`process_entry_point` ignores explicit-storage params.**
   `compute_entry_binding_layout` pushes `None` into
   `entry.param_bindings` for params with an explicit
   `#[storage(set, binding)]` attribute, because the auto-binding
   counter has to skip them. `process_entry_point` then iterates
   `entry.param_bindings.iter().flatten()` and never seeds
   `buffer_map` for explicit-storage params. So a SOAC inside an
   entry with explicit-storage view params never gets the
   provenance the layout knows about.

2. **SOAC capture lists don't trigger `specialize_call`.** Even
   when `buffer_map` *does* know `xs → (set, binding)`, walking the
   SoacBody's captures (just `rewrite_term` on each capture term)
   doesn't recognize the bare `Var(xs)` as a buffer arg the way
   `rewrite_term`'s App arm does for function arguments.

The current WIP fix addresses both by (1) seeding `buffer_map`
from explicit-storage attrs as well, and (2) adding
`try_specialize_soac_view_captures` which detects view-typed SOAC
captures and routes through `create_specialized_def` the same way
`specialize_call` does.

## Why the WIP fix is ~70 lines instead of trivial

The user's intuition was that this should be small: "provenance
tracks the ultimate source — xs and xs[0..3] should have the same
provenance, so let's give the slice the missing provenance." That's
right *conceptually* — but the slice in question is inside
`_w_lambda_13`, not inside `tick`. Inside the lifted def, `xs` is a
function parameter, not a view that any buffer-tracker has provenance
for. There's no provenance to "attach" without first solving the
cross-function transfer.

Two ways to make the slice see real provenance:

1. **Inline the lifted def into the SoacBody**, substituting captures.
   Then the slice's `xs` is the entry-level `xs` directly. *But*
   `rewrite_term` (the body-walker buffer_specialize uses for entries)
   deliberately doesn't handle slice→composite — only
   `rewrite_specialized_body` does, because that path is the only one
   where we have a *real* `(set, binding)` to bake into the emitted
   storage ops. So inlining alone doesn't fix it; you also have to
   extend `rewrite_term` with slice→composite handling that uses
   `buffer_map` directly.

2. **Per-buffer specialize the lifted def** — the current WIP. Reuses
   the existing slice→composite handling in `rewrite_specialized_body`,
   mirrors what App already does via `specialize_call`. ~70 lines of
   bookkeeping: identify view captures, map them to lifted-def param
   positions (captures appended after lam.params), call
   `create_specialized_def`, get back the offset/len param syms,
   rebuild the captures list with `(0, storage_len)` replacements
   at the right positions, share the `(orig_sym, spec_key) →
   spec_sym` cache with the App path.

Both approaches end up touching a similar amount of code. The
per-buffer path is the smaller diff and more consistent with the
existing architecture; the inline path is new machinery and exists
in tension with the "rewrite_term doesn't handle slice→composite"
design call.

## State of the WIP

- Test pinned (failing).
- Fix 1 (`process_entry_point` seeds explicit-storage params) — implemented,
  verified `buffer_map` is now populated correctly.
- Fix 2 (`try_specialize_soac_view_captures`) — implemented, fires
  correctly, calls `create_specialized_def`. The call site in `tick`
  now correctly calls `_w_lambda_13_b1s2b0` with `(loop_i, 0,
  storage_len(2, 0))`.
- **Outstanding**: the specialized lifted def's body still contains a
  bare `Var(xs)` (the original lifted def's param symbol), which
  `rewrite_specialized_body` *should* be rewriting via the
  `view_param_map`. At SPIR-V emission it becomes `global @xs` and
  errors with `Unknown global: xs`. Likely a hole in
  `rewrite_specialized_body`'s handling of `slice(Var(view_param))`
  consumed by composite `Index`, or in how the `view_param_map` is
  threaded into the right scope. Needs further investigation.

## Next steps

When picking this up:

1. Decide between approach (1) and (2) above. The WIP is on (2). If
   we'd rather pursue (1), this branch is the wrong starting point.

2. If continuing with (2), dump the TLC for `_w_lambda_13_b1s2b0`'s
   body right after `create_specialized_def` returns and confirm
   whether `rewrite_specialized_body` left a bare `Var(xs)` behind.
   The likely culprit is the `Index(slice(Var(view_param)), i)`
   shape: `try_resolve_view_expr` recognizes the slice, the Index
   arm rewrites it to a storage_index, but the slice's operand
   (`Var(view_param)`) may also be reached on a path that doesn't
   consult `view_param_map`.

3. Once that's resolved, also verify the q1 case
   (`gather3(xs)` without slicing) — its current "validates but
   reads wrong buffer" behaviour should turn into a correct read
   from `(set=2, binding=0)`. Add an integration test asserting the
   `OpAccessChain` decoration matches the entry-param's
   `#[storage(...)]`.
