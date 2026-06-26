# Split `SoacOp::Filter`'s `map_lam: Option` Into Two Variants

*Part of the "make invalid states unrepresentable" theme — same spirit as
`c06bf458`. Small, isolated, clean.*

## Context

`tlc/mod.rs:702`:

```rust
Filter {
    map_lam: Option<SoacBody>,   // None = plain filter, Some(f) = map fused into filter
    pred: SoacBody,
    input: ArrayExpr,
    destination: SoacDestination,
}
```

`map_lam` is an `Option` standing in for a two-case *mode*. When `Some(f)`, the
filter's output element type is `f`'s return type, not the input element type —
so the `Option`'s state silently changes the node's typing. Fusion eligibility
is gated by `.map_lam.is_none()` checks (`array_semantics.rs:227`, `fusion.rs`).

## Invalid states currently representable

Nothing crashes today, but the `Option` smell means every consumer re-derives
"is this a plain or fused filter?" from `is_none()`, and the output-element-type
rule is a comment, not a structural fact.

## The tightening

```rust
enum FilterKind {
    Plain    { pred, input, destination },                 // out elem = input elem
    FusedMap { map_lam: SoacBody, pred, input, destination }, // out elem = map_lam ret
}
```

(or keep `Filter` as one variant but make the mode an inner two-arm enum). The
"already fused?" question becomes a pattern match; the output-element-type rule
attaches to the variant that owns it.

## Blast radius

Low (~10 construct/match sites, all in filter handling: `array_semantics`,
`fusion`, `from_tlc`, `parallelize`). Self-contained — a good standalone commit.

## Verification

1. `cargo test -p wyn-core` (the map→filter fusion tests cover both modes).
2. `scripts/validate_testfiles.sh` and `--wgsl` — both unchanged
   (`particles.wyn` exercises the fused path).
3. `grep -n 'map_lam.is_none\|map_lam.is_some' wyn-core/src/tlc/` → zero.
