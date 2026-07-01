//! Semantic SegOp fusion (EGIR milestone 5).
//!
//! Two reusable primitives — provenance-based `SegSpace` equality (`space`) and
//! a read-only query layer over the existing semantic dependency DAG
//! (`legality`) — underpin the `horizontal` fusion pass that `egir::semantic_opt`
//! drives. The legality oracle owns the invariant fusion rests on: never fuse or
//! reorder two ops that alias (share a binding under a non-Read access).
//!
//! Only *horizontal* (same-space sibling) fusion lives here. Producer→consumer
//! fusion is a source-level optimization performed in `tlc::fusion`
//! (force-inline + `fuse_maps`); by the time SegOps exist there are no
//! single-consumer producer chains left to inline, so EGIR does not repeat it.

pub(crate) mod horizontal;
pub(crate) mod legality;
pub(crate) mod space;
