//! Semantic SegOp fusion (EGIR milestone 5).
//!
//! Two reusable primitives — provenance-based `SegSpace` equality (`space`) and
//! a read-only query layer over the existing semantic dependency DAG
//! (`legality`) — underpin the horizontal and vertical passes that
//! `egir::semantic_opt` drives. The legality oracle owns the invariant fusion
//! rests on: never move an operation across resource or effect ordering.
//!
//! Horizontal fusion combines independent siblings. Vertical fusion composes
//! callable regions for a pure, single-consumer `SegMap` producer and its
//! same-space consumer. Multi-consumer producers deliberately survive for the
//! allocation pass to materialize once.

pub(crate) mod horizontal;
pub(crate) mod legality;
pub(crate) mod space;
pub(crate) mod vertical;
