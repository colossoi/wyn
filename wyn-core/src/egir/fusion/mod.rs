//! Semantic SegOp fusion (EGIR milestone 5).
//!
//! The horizontal and vertical passes here are driven by `egir::semantic_opt`
//! and rest on two primitives: provenance-based `SegSpace` equality (`space`),
//! and `egir::semantic_graph::SemanticGraph`, the query layer over the semantic
//! dependency DAG. That oracle owns the invariant fusion rests on: never move
//! an operation across resource or effect ordering.
//!
//! Horizontal fusion combines independent siblings. Vertical fusion composes
//! callable regions for a pure, single-consumer `SegMap` producer and its
//! same-space consumer. Multi-consumer producers deliberately survive for the
//! allocation pass to materialize once.

pub(crate) mod horizontal;
pub(crate) mod space;
pub(crate) mod vertical;
