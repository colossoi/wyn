//! TLC pattern lowering: `match` expressions and pattern-driven let
//! bindings. Generates sequential test chains (no decision-tree
//! compilation — see `crate::types::patterns::coverage` for rationale).

pub mod match_lowering;
