//! Structural "same iteration space" comparison for SegOp fusion.
//!
//! Two SegOps may be fused only over an equal `SegSpace`. Equality here is
//! *provenance*-based, not `NodeId` identity: sibling SegOps over the same input
//! array intern their extent nodes separately, so we compare the host-dispatch
//! identity of each dimension (a fixed length, a push-constant offset, a
//! resource length's binding+stride) rather than the node. `SegExtent`'s derived
//! `Eq` is node-sensitive and deliberately not used here. A conservative
//! false-negative only declines a legal fusion; a false-positive would fuse
//! genuinely different spaces, so the comparison errs toward inequality.

use crate::egir::types::{SegExtent, SegSpace};

/// True iff two iteration spaces are provably the same for fusion purposes.
pub fn seg_space_fusable(a: &SegSpace, b: &SegSpace) -> bool {
    a.level == b.level
        && a.dims.len() == b.dims.len()
        && a.dims.iter().zip(&b.dims).all(|(x, y)| seg_extent_fusable(x, y))
}

/// True iff two extents denote the same dispatch dimension.
pub fn seg_extent_fusable(a: &SegExtent, b: &SegExtent) -> bool {
    match (a, b) {
        (SegExtent::Fixed(x), SegExtent::Fixed(y)) => x == y,
        // The node is a re-interned `FuncParam`; the offset is the host
        // dispatch identity (matches `domain_from_space`).
        (SegExtent::PushConstant { offset: x, .. }, SegExtent::PushConstant { offset: y, .. }) => x == y,
        // Same buffer + element stride ⇒ same length, regardless of which
        // re-interned length node each op holds.
        (
            SegExtent::ResourceLength {
                binding: ba,
                elem_bytes: ea,
                ..
            },
            SegExtent::ResourceLength {
                binding: bb,
                elem_bytes: eb,
                ..
            },
        ) => ba == bb && ea == eb,
        // A non-host-dispatchable runtime length: only equal when the ops
        // already share the (hash-consed) node. Conservative.
        (SegExtent::Value(x), SegExtent::Value(y)) => x == y,
        _ => false,
    }
}

#[cfg(test)]
#[path = "space_tests.rs"]
mod space_tests;
