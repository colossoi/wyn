//! Cost-based extraction: pick the cheapest enode per eclass.
//!
//! In Phase 1 (no unions), this is trivially the identity: every node
//! is its own best representative. When union nodes exist, bottom-up DP
//! picks the cheaper child at each union.

use crate::LookupMap;

use super::types::{EGraph, ENode, NodeId, PureOp};

/// Cost of a node. Lower is better.
pub type Cost = u32;

/// Compute the best (cheapest) representative for each NodeId.
///
/// Returns a map from NodeId -> best concrete NodeId (the chosen representative).
/// For non-union nodes, this maps to themselves.
/// For union nodes, this maps to the best leaf of the union tree.
pub fn extract(graph: &EGraph) -> LookupMap<NodeId, NodeId> {
    let mut best_cost: LookupMap<NodeId, Cost> = LookupMap::new();
    let mut best_node: LookupMap<NodeId, NodeId> = LookupMap::new();

    // Topological sort of the acyclic graph.
    let topo = topological_sort(graph);

    // Bottom-up: compute cost for each node.
    for &nid in &topo {
        let node = &graph.nodes[nid];
        match node {
            ENode::Union { left, right } => {
                let lc = best_cost.get(left).copied().unwrap_or(Cost::MAX);
                let rc = best_cost.get(right).copied().unwrap_or(Cost::MAX);
                if lc <= rc {
                    best_cost.insert(nid, lc);
                    best_node.insert(nid, best_node.get(left).copied().unwrap_or(*left));
                } else {
                    best_cost.insert(nid, rc);
                    best_node.insert(nid, best_node.get(right).copied().unwrap_or(*right));
                }
            }
            ENode::Pure { op, operands } => {
                let child_sum: Cost = operands
                    .iter()
                    .map(|c| best_cost.get(c).copied().unwrap_or(0))
                    .fold(0u32, |a, b| a.saturating_add(b));
                let cost = op_cost(op).saturating_add(child_sum);
                best_cost.insert(nid, cost);
                best_node.insert(nid, nid);
            }
            ENode::Constant(_) | ENode::FuncParam { .. } | ENode::BlockParam { .. } => {
                best_cost.insert(nid, 0);
                best_node.insert(nid, nid);
            }
            ENode::SideEffectResult => {
                // Side-effect results have zero cost (they're mandatory).
                best_cost.insert(nid, 0);
                best_node.insert(nid, nid);
            }
        }
    }

    best_node
}

/// Static cost per operation kind.
fn op_cost(op: &PureOp) -> Cost {
    match op {
        PureOp::ResourceLen(_) => 0,
        // Leaves / free operations:
        PureOp::Int(_)
        | PureOp::Uint(_)
        | PureOp::Float(_)
        | PureOp::Bool(_)
        | PureOp::Unit
        | PureOp::Global(_)
        | PureOp::Extern(_)
        | PureOp::Project { .. } => 0,

        // Single-instruction operations:
        PureOp::BinOp(_)
        | PureOp::UnaryOp(_)
        | PureOp::Index
        | PureOp::Tuple(_)
        | PureOp::Vector(_)
        | PureOp::ArrayLit(_)
        | PureOp::ArrayRange { .. } => 1,

        // More expensive:
        PureOp::Matrix { .. } => 2,
        PureOp::DynamicExtract => 2,
        PureOp::Materialize => 3,
        PureOp::OutputSlot { .. } => 0,
        PureOp::StorageView(_) => 2,
        PureOp::ViewIndex => 1,
        PureOp::PlaceIndex => 1,
        PureOp::StorageViewLen => 0,
        PureOp::StorageImageLoad(_) | PureOp::StorageImageStore(_) => 2,
        PureOp::Intrinsic { .. } => 2,
        PureOp::Call(_) => 3,
    }
}

/// Dependency-order sort of the acyclic e-graph.
///
/// Extraction's bottom-up DP reads each node's children before the node, which
/// only holds for a topological order. A cycle means an earlier pass interned a
/// node into its own operand tree; there is no order to fall back on, so say so
/// rather than hand the DP an arbitrary one.
fn topological_sort(graph: &EGraph) -> Vec<NodeId> {
    wyn_graph::topo_sort_by_dependencies(graph.nodes.keys(), |node, dependencies| {
        dependencies.extend(graph.nodes[node].children());
    })
    .unwrap_or_else(|err| {
        panic!(
            "EGraph extraction requires an acyclic graph; {} node(s) lie on or behind a cycle: {:?}",
            err.remaining().len(),
            err.remaining()
        )
    })
}
