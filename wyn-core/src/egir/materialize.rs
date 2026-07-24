//! EGIR materialize pass.
//!
//! Rewrites every pure `Index(arr, i)` whose index isn't a compile-time
//! constant into `DynamicExtract(Materialize(arr), i)`. The SPIR-V backend
//! needs this because `OpCompositeExtract` requires literal indices —
//! anything else has to spill the composite to a memory-backed handle and
//! `OpAccessChain` into it.
//!
//! `Materialize` depends only on the array operand, so `elaborate`'s
//! scoped, loop-aware placement emits it at the deepest dominator of the
//! array—outside the loop whenever the array is loop-invariant.
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

/// Physical EGIR with dynamic composite extraction made explicit.
#[derive(Debug, Clone, Copy, Default)]
pub struct Materialized;

impl super::ir::Stage for Materialized {
    type Family = super::types::Physical;
    type ResourceDecl = crate::interface::StorageBindingDecl;
    type OutputRoute = super::ir::RealizedOutputRoute;
    type ProgramData = super::program::CoreProgramData;
    type GlobalContext = super::program::PlannedGlobal;
}

use super::program::Program;
use super::types::{EGraph, ENode, Family, NodeId, PureOp};

/// Make dynamic composite extraction explicit in every body.
pub fn run(program: Program<super::soac_expand::SoacsExpanded>) -> Program<Materialized> {
    program
        .map_graphs(|_, mut graph| {
            run_one_body(&mut graph);
            graph
        })
        .into_stage()
}

/// Rewrite all dynamic Index nodes in the e-graph to Materialize +
/// DynamicExtract.
fn run_one_body<P: Family>(graph: &mut EGraph<P>) {
    // Snapshot first; we'll mutate node entries and add new Materialize nodes.
    let targets: Vec<(NodeId, NodeId, NodeId)> = graph
        .nodes
        .iter()
        .filter_map(|(nid, node)| match &node.kind {
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
        let arr_ty = graph.nodes[arr_nid].ty.clone();

        // Materialize is hash-consed: two Index(arr, _) share the same
        // Materialize(arr) handle automatically.
        let mat_nid = graph.intern_pure(PureOp::Materialize, smallvec![arr_nid], arr_ty, None);

        // Replace the original Index node in place with DynamicExtract(mat, idx).
        // The NodeId stays the same so all consumers continue to resolve through it.
        // The node's stored type is unchanged (still elem_ty).
        graph.replace_pure_node(index_nid, PureOp::DynamicExtract, smallvec![mat_nid, idx_nid]);
    }
}

/// Is `nid`'s array type a storage view? `lower_index` reads a view with a
/// native dynamic `OpAccessChain`, so it must not be spilled to a composite.
fn is_view<P: Family>(graph: &EGraph<P>, nid: NodeId) -> bool {
    graph.nodes.get(nid).is_some_and(|node| {
        matches!(
            node.ty.array_variant(),
            Some(Type::Constructed(TypeName::ArrayVariantView, _))
        )
    })
}

/// Is this NodeId a compile-time integer constant? Includes both the inline
/// `ENode::Constant(ConstantValue::I32|U32)` form and `ENode::Pure(PureOp::Int|Uint)`.
fn is_const_int<P: Family>(graph: &EGraph<P>, nid: NodeId) -> bool {
    match &graph.nodes[nid].kind {
        ENode::Constant(ConstantValue::I32(_) | ConstantValue::U32(_)) => true,
        ENode::Pure {
            op: PureOp::Int(_) | PureOp::Uint(_),
            ..
        } => true,
        _ => false,
    }
}
