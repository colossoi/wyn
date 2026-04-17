//! EGIR materialize pass.
//!
//! Rewrites every pure `Index(arr, i)` whose index isn't a compile-time
//! constant into `DynamicExtract(Materialize(arr), i)`. The SPIR-V backend
//! needs this because `OpCompositeExtract` requires literal indices —
//! anything else has to spill the composite to a memory-backed handle and
//! `OpAccessChain` into it.
//!
//! In the SSA version of this pass we also had to LICM `Materialize` out of
//! loops manually. In EGIR that's free: `Materialize` only depends on the
//! array operand, so `elaborate`'s scoped/loop-aware placement emits it at
//! the deepest dominator of the array — outside the loop whenever the array
//! is loop-invariant.
//!
//! Two `Index` nodes with the same array share a single `Materialize` node
//! via hash-consing, so we don't need a separate dedup step either.

use smallvec::smallvec;

use crate::ssa::types::ConstantValue;

use super::types::{EGraph, ENode, NodeId, PureOp};

/// Rewrite all dynamic Index nodes in the e-graph to Materialize +
/// DynamicExtract. Called by the typestate transition
/// `EGraphSoacExpanded::materialize`.
///
/// Skipped for arrays whose variant is `ArrayVariantOwnedView`: those live
/// at runtime as a `{buffer: [N]T, valid_len: i32}` struct, and the
/// backend's Index lowering already drills through the struct to access
/// the inner buffer. A naive `Materialize` would spill the whole struct,
/// after which `DynamicExtract`'s `AccessChain(struct_ptr, idx)` would try
/// to select a struct field by a non-literal index — invalid SPIR-V.
pub(crate) fn run(graph: &mut EGraph) {
    // Snapshot first; we'll mutate node entries and add new Materialize nodes.
    let targets: Vec<(NodeId, NodeId, NodeId)> = graph
        .nodes
        .iter()
        .filter_map(|(nid, node)| match node {
            ENode::Pure {
                op: PureOp::Index,
                operands,
            } if operands.len() == 2 => {
                let arr = operands[0];
                let idx = operands[1];
                if is_const_int(graph, idx) || is_owned_view_array(graph, arr) {
                    None
                } else {
                    Some((nid, arr, idx))
                }
            }
            _ => None,
        })
        .collect();

    for (index_nid, arr_nid, idx_nid) in targets {
        let arr_ty = graph.types[&arr_nid].clone();

        // Materialize is hash-consed: two Index(arr, _) share the same
        // Materialize(arr) handle automatically.
        let mat_nid = graph.intern_pure(PureOp::Materialize, smallvec![arr_nid], arr_ty);

        // Replace the original Index node in place with DynamicExtract(mat, idx).
        // The NodeId stays the same so all consumers continue to resolve through it.
        // graph.types[index_nid] is unchanged (still elem_ty).
        graph.nodes[index_nid] = ENode::Pure {
            op: PureOp::DynamicExtract,
            operands: smallvec![mat_nid, idx_nid],
        };
    }
}

/// Is this NodeId a compile-time integer constant? Includes both the inline
/// `ENode::Constant(ConstantValue::I32|U32)` form and `ENode::Pure(PureOp::Int|Uint)`.
fn is_const_int(graph: &EGraph, nid: NodeId) -> bool {
    match &graph.nodes[nid] {
        ENode::Constant(ConstantValue::I32(_) | ConstantValue::U32(_)) => true,
        ENode::Pure {
            op: PureOp::Int(_) | PureOp::Uint(_),
            ..
        } => true,
        _ => false,
    }
}

/// Does this NodeId carry an `ArrayVariantOwnedView` array type?
fn is_owned_view_array(graph: &EGraph, nid: NodeId) -> bool {
    use crate::types::TypeExt;
    graph
        .types
        .get(&nid)
        .and_then(|t| t.array_variant())
        .map(crate::types::is_array_variant_owned_view)
        .unwrap_or(false)
}
