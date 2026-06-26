# Typestate Backlog: Invariants That Could Become Unrepresentable

*Theme: the ANF commit (`c06bf458`) made "a SOAC is a direct argument of
another SOAC" unrepresentable by tightening a field type, retiring a runtime
validator. These are the remaining invariant-level candidates in that vein —
things enforced by a validation function, by `unreachable!`, or by nothing at
all. Two items from the original survey are being worked separately (the
`soa.rs` guard/extract collapse and the SOAC operand-layout restructure); this
file is the backlog of the rest.*

Ordered roughly by value ÷ cost.

---

## 1. Every reachable `SymbolId` is registered — assumed by 30+ `.expect()`

**Invariant:** any `SymbolId` reachable from a `Def` body has an entry in the
`SymbolTable`. **Enforced by:** nothing — relied on by 30+
`.expect("BUG: symbol not in table")` (`egir/from_tlc.rs:162,185,1088,1140,1754`;
`tlc/closure_convert.rs:192,1243`; `tlc/hof_specialize.rs:1269`; …) and surfaced
as a `ConvertError::Internal` arm. Nothing stops a pass minting a fresh
`SymbolId` without registering it.

**Fix:** make `SymbolId` allocatable only through `SymbolTable::alloc(...)` so a
constructed id is provably in the table (private constructor + the table is the
only mint). Lookups that walk reachable code become total — the 30+ `.expect()`s
and the internal-error arm delete. **Blast radius:** medium-high but localized to
the `SymbolTable` API + every symbol-minting site. High readability payoff.

## 2. Phase-scoped variants enforced by `unreachable!` *(the big one)*

**Invariant:** `ArrayExpr::Zip` is valid only pre-SoA; `ArrayExpr::StorageView`
only post-`lift_gathers`/post-mono; `TermKind::OutputSlotStore` only
post-`normalize_outputs`. **Enforced by:** scattered `unreachable!()`
(`tlc/soa.rs:360`) + pass ordering + a doc comment that already proposes the fix
(`tlc/mod.rs:489-510`). Every pass must *remember* an `unreachable!` arm.

**Fix:** phase-split the enums (`PreSoaArrayExpr`/`PostSoaArrayExpr`, etc.),
threading a phase type parameter through `Term`/`TermKind`/`SoacOp`/`Lambda`. The
same move also makes structural the validator postconditions in
`verify_closure_converted` (no `Lambda` nodes survive, `closure_convert.rs:501`)
and `verify_hof_specialized` (no arrow-typed params, `hof_specialize.rs:39`).
**Blast radius:** large — cascades across the whole TLC pipeline. The prize, but
a deliberate standalone effort; the comment notes it's "fine until it pays for
itself."

## 3. Flat-chain `Let` — `verify_flattened` could be a type

**Invariant:** a `let`'s rhs is never itself a `let` (single flat spine).
**Enforced by:** `tlc/normalize.rs:77` `verify_flattened`, a `debug_assert`-only
postcondition (no-op in release); the fusion driver depends on it. **Fix:**
`TermKind::Let { rhs: NonLetTerm, .. }` where `NonLetTerm` is a newtype excluding
`Let`. **Blast radius:** large (every `Let` constructor), but conceptually clean
and self-contained.

## 4. Flat-apps — `assert_flat_apps` could be a type

**Invariant:** no `App` sits in the func position of another `App` (fully
applied calls only). **Enforced by:** `tlc/mod.rs:809` `assert_flat_apps`
(debug-panic) and `closure_calls_lower.rs:43` `verify_closure_calls_lowered`
(post-pass `Result`). **Fix:** split `TermKind::App` into `DirectCall { func:
SymbolId, args }` / `BuiltinCall { op, args }` — nested-App becomes
unconstructable and call dispatch simplifies across backends. **Blast radius:**
large (every App site). Pairs naturally with the arity check
(`verify_closure_calls_lowered`).

## 5. Rank-1 array invariant — comment-only, *no* check

**Invariant:** every array in EGIR is rank-1: type args `[elem, variant, size,
region]` (4 args). **Enforced by:** a comment (`egir/soac_expand.rs:100`) and
code that pattern-matches exactly 4 args; `is_plain_array_source` silently skips
non-rank-1. No validator. **Fix:** an `Array1` newtype wrapping `(elem, variant,
size, region)` used at/after the TLC→EGIR boundary, so a non-rank-1 array is
unrepresentable in EGIR. **Blast radius:** medium, confined to EGIR + the
boundary. Also subsumes the `soa.rs` array-field `.expect("Array has …")` cluster
(`soa.rs:35-38`) via an `as_array() -> Option<ArrayType>` accessor.

## 6. `realize_outputs` postcondition — debug-only validator → proof token

**Invariant:** after `realize_outputs`, no runtime-sized `Composite` array is
reachable from entry outputs/operands. **Enforced by:** `realize_outputs/verify.rs:38`
`check()`, gated on `cfg!(debug_assertions)`. **Fix:** a marker/newtype on the
graph proving the pass ran (constructor only reachable through `realize_outputs`).
**Blast radius:** small, localized post-pass.

## 7. CFG reachability — `domtree` `.expect()` with no skeleton validator

**Invariant:** every reachable non-entry block has a reachable predecessor.
**Enforced by:** nothing; relied on by `egir/domtree.rs:100`
`.expect("reachable non-entry block has a predecessor")`. **Fix:** a validated
`Skeleton` newtype whose constructor checks reachability once, or a builder that
can't produce a danging block. **Blast radius:** medium (skeleton construction).

---

### Also noted (trivial)
- `tlc/lift_gathers.rs:518` re-destructures the same `TermKind::Soac` it just
  matched, with an `unreachable!()` else — collapse to one `if let`.
