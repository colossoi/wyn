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
//!
//! Storage-view arrays are exempt: they are memory-backed already, so
//! `lower_index` reads them with a dynamic `OpAccessChain` into the backing
//! buffer (`lower_view_index`). Spilling a view to a Function-local composite
//! and `DynamicExtract`ing it would both be wrong (a runtime-sized view has no
//! in-register form) and invalid SPIR-V (a dynamic index into that spilled
//! struct). Only in-register composites need the rewrite.

use smallvec::smallvec;

use polytype::Type;

use crate::ast::TypeName;
use crate::ssa::types::ConstantValue;
use crate::types::TypeExt;

use super::program::PhysicalProgram;
use super::types::{EGraph, ENode, NodeId, PureOp};

/// Run `run_one_body` on every function and entry point in the program.
pub(crate) fn run(inner: &mut PhysicalProgram) {
    for f in &mut inner.functions {
        run_one_body(&mut f.graph);
    }
    for e in &mut inner.entry_points {
        run_one_body(&mut e.graph);
    }
}

/// Rewrite all dynamic Index nodes in the e-graph to Materialize +
/// DynamicExtract.
fn run_one_body<R: super::types::GraphResource>(graph: &mut EGraph<R>) {
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
                if is_const_int(graph, idx) || is_view(graph, arr) {
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
        graph.replace_pure_node(index_nid, PureOp::DynamicExtract, smallvec![mat_nid, idx_nid]);
    }
}

/// Is `nid`'s array type a storage view? `lower_index` reads a view with a
/// native dynamic `OpAccessChain`, so it must not be spilled to a composite.
fn is_view<R>(graph: &EGraph<R>, nid: NodeId) -> bool {
    graph.types.get(&nid).is_some_and(|ty| {
        matches!(
            ty.array_variant(),
            Some(Type::Constructed(TypeName::ArrayVariantView, _))
        )
    })
}

/// Is this NodeId a compile-time integer constant? Includes both the inline
/// `ENode::Constant(ConstantValue::I32|U32)` form and `ENode::Pure(PureOp::Int|Uint)`.
fn is_const_int<R>(graph: &EGraph<R>, nid: NodeId) -> bool {
    match &graph.nodes[nid] {
        ENode::Constant(ConstantValue::I32(_) | ConstantValue::U32(_)) => true,
        ENode::Pure {
            op: PureOp::Int(_) | PureOp::Uint(_),
            ..
        } => true,
        _ => false,
    }
}
