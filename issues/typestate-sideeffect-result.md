# Make `SideEffect.result` Kind-Discriminated (kill the `Option`)

*Part of the "make invalid states unrepresentable" theme — same spirit as
`c06bf458` (locked SOAC inputs to ANF via a tightened `ArrayExpr`).*

## Context

`egir/types.rs:115` defines every skeleton instruction uniformly:

```rust
pub struct SideEffect {
    pub kind: SideEffectKind,            // Inst(InstKind) | Pending(PendingSoac)
    pub operand_nodes: SmallVec<[NodeId; 4]>,
    pub result: Option<NodeId>,          // "Result value, if this produces one"
    pub effects: Option<(EffectToken, EffectToken)>,
    pub span: Option<Span>,
}
```

`result` is `Option<NodeId>` regardless of `kind`. But whether an
instruction produces a value is a *property of its kind*, not a free
choice: a `Load` / `Alloca` / `OutputSlot` / expanded Screma always
produces one; a `Store` / `ControlBarrier` never does.

## Invalid states currently representable

- `Store` / `ControlBarrier` with `result: Some(_)` — a dangling SSA value.
- `Load` / `Alloca` / value-producing `PendingSoac` with `result: None` —
  a missing value that every consumer must `.expect()` back.

## How it's enforced today (runtime)

Six-plus `.expect()` calls reconstruct the invariant the type erased:

- `egir/soac_expand.rs` — `se.result.expect("Screma/Filter/Scatter has a result")`
  at ~5 sites (Screma, Filter, Scatter expansion).
- `egir/elaborate.rs:327` — `se.result.expect("Alloca side-effect must carry a result")`.

## The tightening

Discriminate `result` by kind. Cleanest is to split `SideEffect` (or the
`SideEffectKind` wrapper) into value-producing vs pure-effect:

```rust
pub enum SideEffect {
    Value {                              // Load, Alloca, OutputSlot, expanded SOAC
        kind: ValueKind,
        operand_nodes: SmallVec<[NodeId; 4]>,
        result: NodeId,                  // required, no Option
        effects: Option<(EffectToken, EffectToken)>,
        span: Option<Span>,
    },
    Effect {                             // Store, ControlBarrier
        kind: EffectKind,
        operand_nodes: SmallVec<[NodeId; 4]>,
        effects: Option<(EffectToken, EffectToken)>,
        span: Option<Span>,
    },
}
```

A lighter variant (less churn) keeps one struct but moves `result` into the
two `SideEffectKind` arms so the field only exists where it is legal.

## Blast radius

~40 match sites: `soac_expand.rs` (the 5 `.expect`s + bulk `expand_one`),
`elaborate.rs`, `graph_ops.rs` (emit_store/load/alloca), light touches in
`skel_opt`, `realize_outputs`. Medium. Removes every `.expect` on the field
and makes "did this instruction yield a value?" a compile-time fact.

## Verification

1. `cargo test -p wyn-core`.
2. `scripts/validate_testfiles.sh` and `--wgsl` — both unchanged.
3. `grep -n 'result.expect' wyn-core/src/egir/` → zero on this field.
