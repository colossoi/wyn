# Replace Stringly-Typed Operators With Enums

*Part of the "make invalid states unrepresentable" theme — same spirit as
`c06bf458`. Sibling to `issues/intrinsic-name-cleanup.md` (the other half of
killing string dispatch).*

## Context

Three AST nodes carry operators as `String`, gated only by the parser. Every
downstream consumer (type checker, TLC lowering, SPIR-V backend, WGSL backend)
re-matches the string and needs a fallthrough / `unreachable!` arm for the
operators the parser "can't" produce.

- `ast.rs:447` — `pub struct BinaryOp { pub op: String }`
- `ast.rs:452` — `pub struct UnaryOp { pub op: String }`  (`// "-" or "!"`)
- `ast.rs:393` — `VecWith { … op: Option<String>, … }`
  (`None` = `=`; `Some` of `"*"`/`"+"`/`"-"`/`"/"`)

## Invalid states currently representable

`BinaryOp { op: "wat".into() }`, `UnaryOp { op: "~".into() }`,
`VecWith { op: Some("%".into()), .. }` — all constructable, all caught (if at
all) far from the parser by a match that silently falls through or panics.

## How it's enforced today

The parser alone: `get_operator_precedence()` (`parser.rs:1838`) for binary,
the `-`/`!` parse arm (`parser.rs:2083`) for unary, the compound-op parse arm
(`parser.rs:1948`) for `VecWith`. Nothing downstream re-validates; the alphabet
is a convention each backend re-decodes.

## The tightening

```rust
pub enum BinaryOp { Add, Sub, Mul, Div, Eq, Ne, Lt, Le, Gt, Ge, And, Or, Pipe, … }
pub enum UnaryOp  { Negate, Not }
pub enum VecWithOp { Assign, AddAssign, SubAssign, MulAssign, DivAssign }
```

The parser maps tokens → enum once; every backend match becomes exhaustive and
loses its impossible-operator arm. (Keep operator *spelling* for diagnostics via
a `fn symbol(&self) -> &'static str`.)

## Blast radius

Broad but mechanical — each site is a one-line swap from string compare to enum
match. Touches `parser.rs`, `types/checker.rs`, `tlc/` lowering, `spirv/`,
`wgsl/`. The payoff is exhaustiveness across all four backends. Worth doing as
one focused "operators" commit, separate from the intrinsic-name migration.

## Bonus (same flavor, tiny): `RecordWith.path`

`ast.rs:406` — `path: Vec<String>` documented "Length ≥ 1", but `tlc/mod.rs:3206`
does `&path[0]`: an empty path is a real latent panic. Replace with a non-empty
pair `(String, Vec<String>)` (head + tail) so emptiness is unconstructable.
`MatchExpr.cases` (`ast.rs:482`) is similarly "≥ 1 by convention" — same fix
with a shared `NonEmpty<T>`.

## Verification

1. `cargo test -p wyn-core`.
2. `scripts/validate_testfiles.sh` and `--wgsl` — both unchanged.
3. `grep -n '"\\+"\\|"=="\\|op ==' wyn-core/src/` across backends → no
   string-keyed operator dispatch remains.
