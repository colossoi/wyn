//! Semantic SegOp fusion (EGIR milestone 5).
//!
//! Two reusable primitives — provenance-based `SegSpace` equality (`space`) and
//! a read-only query layer over the existing semantic dependency DAG
//! (`legality`) — underpin the graph-rewriting fusion passes that
//! `egir::semantic_opt` drives. The legality oracle owns the single invariant
//! the whole milestone rests on: never fuse or reorder two ops that conflict on
//! a resource or effect.

pub(crate) mod horizontal;
pub(crate) mod legality;
pub(crate) mod space;
pub(crate) mod vertical;
