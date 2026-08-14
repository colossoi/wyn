//! Cost-based extraction: pick the cheapest enode per eclass.
//!
//! Union nodes are created by the rewrite pass (`egir::rewrite`) run on the
//! physical program; a graph without them extracts trivially to the
//! identity. Union alternatives are compared by the total cost of their
//! *distinct* reachable nodes, so a subgraph shared by both sides cancels
//! out instead of penalizing the side that references it more often.

use std::collections::HashSet;

use crate::op::BinaryOperator;
use crate::LookupMap;

use super::types::{EGraph, Family, PureOp, ValueId, ValueKind};

/// Cost of a node. Lower is better.
pub type Cost = u32;

/// Compute the best (cheapest) representative for each ValueId.
///
/// Returns a map from ValueId -> best concrete ValueId (the chosen representative).
/// For non-union nodes, this maps to themselves.
/// For union nodes, this maps to the best leaf of the union tree.
pub fn extract<P: Family>(graph: &EGraph<P>) -> LookupMap<ValueId, ValueId> {
    let mut best_cost: LookupMap<ValueId, Cost> = LookupMap::new();
    let mut best_node: LookupMap<ValueId, ValueId> = LookupMap::new();

    // Topological sort of the acyclic graph.
    let topo = topological_sort(graph);

    // Bottom-up: compute cost for each node.
    for &nid in &topo {
        let node = &graph.nodes[nid].kind;
        match node {
            ValueKind::Union { left, right } => {
                // A plain subtree-sum comparison would double-count operands
                // shared by both sides — e.g. the base of a pow-vs-multiply-
                // chain union — and mis-pick whenever the shared value is
                // expensive. Closure costs count each distinct node once, so
                // shared subgraphs contribute equally and cancel.
                let lc = closure_cost(graph, *left, &best_node);
                let rc = closure_cost(graph, *right, &best_node);
                let chosen = if lc <= rc { left } else { right };
                best_cost.insert(nid, best_cost.get(chosen).copied().unwrap_or(Cost::MAX));
                best_node.insert(nid, best_node.get(chosen).copied().unwrap_or(*chosen));
            }
            ValueKind::Pure { op, operands } => {
                let child_sum: Cost = operands
                    .iter()
                    .map(|c| best_cost.get(c).copied().unwrap_or(0))
                    .fold(0u32, |a, b| a.saturating_add(b));
                let cost = op_cost(op).saturating_add(child_sum);
                best_cost.insert(nid, cost);
                best_node.insert(nid, nid);
            }
            ValueKind::Constant(_) | ValueKind::FuncParam { .. } | ValueKind::BlockParam { .. } => {
                best_cost.insert(nid, 0);
                best_node.insert(nid, nid);
            }
            ValueKind::SideEffectResult => {
                // Side-effect results have zero cost (they're mandatory).
                best_cost.insert(nid, 0);
                best_node.insert(nid, nid);
            }
        }
    }

    best_node
}

/// Total op cost of the distinct nodes reachable from `root`, following each
/// union to its already-chosen representative (children precede parents in
/// the topo order, so nested unions are resolved by the time this runs).
fn closure_cost<P: Family>(graph: &EGraph<P>, root: ValueId, best: &LookupMap<ValueId, ValueId>) -> Cost {
    let mut seen = HashSet::new();
    let mut stack = vec![root];
    let mut total: Cost = 0;
    while let Some(nid) = stack.pop() {
        let nid = best.get(&nid).copied().unwrap_or(nid);
        if !seen.insert(nid) {
            continue;
        }
        if let ValueKind::Pure { op, operands } = &graph.nodes[nid].kind {
            total = total.saturating_add(op_cost(op));
            stack.extend(operands.iter().copied());
        }
    }
    total
}

/// Modeled cost of the backend's `**` lowering: GLSL.std.450 `Pow` (an
/// exp/log sequence) for floats, an exponentiation-by-squaring helper for
/// ints. A multiply chain proposed by `rewrite::PowToMulChain` wins while
/// it needs fewer multiplies than this.
const POW_COST: Cost = 8;

/// Static cost per operation kind.
fn op_cost<R>(op: &PureOp<R>) -> Cost {
    match op {
        PureOp::BinOp(BinaryOperator::Power) => POW_COST,
        PureOp::ResourceLen(_) => 0,
        // Leaves / free operations:
        PureOp::Int(_)
        | PureOp::Uint(_)
        | PureOp::Float(_)
        | PureOp::Bool(_)
        | PureOp::Unit
        | PureOp::Global(_)
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
fn topological_sort<P: Family>(graph: &EGraph<P>) -> Vec<ValueId> {
    wyn_graph::topo_sort_by_dependencies(graph.nodes.keys(), |node, dependencies| {
        dependencies.extend(graph.nodes[node].kind.children());
    })
    .unwrap_or_else(|err| {
        panic!(
            "EGraph extraction requires an acyclic graph; {} node(s) lie on or behind a cycle: {:?}",
            err.remaining().len(),
            err.remaining()
        )
    })
}
