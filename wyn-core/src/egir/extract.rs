//! Cost-based extraction: pick the cheapest enode per eclass.
//!
//! In Phase 1 (no unions), this is trivially the identity: every node
//! is its own best representative. When union nodes exist, bottom-up DP
//! picks the cheaper child at each union.

use std::collections::HashMap;

use super::types::{EGraph, ENode, NodeId, PureOp};

/// Cost of a node. Lower is better.
pub type Cost = u32;

/// Compute the best (cheapest) representative for each NodeId.
///
/// Returns a map from NodeId → best concrete NodeId (the chosen representative).
/// For non-union nodes, this maps to themselves.
/// For union nodes, this maps to the best leaf of the union tree.
pub fn extract(graph: &EGraph) -> HashMap<NodeId, NodeId> {
    let mut best_cost: HashMap<NodeId, Cost> = HashMap::new();
    let mut best_node: HashMap<NodeId, NodeId> = HashMap::new();

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
        PureOp::StorageViewLen => 0,
        PureOp::Intrinsic { .. } => 2,
        PureOp::Call(_) => 3,
    }
}

/// Kahn's algorithm for topological sort on the acyclic e-graph.
fn topological_sort(graph: &EGraph) -> Vec<NodeId> {
    let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
    for (nid, _) in &graph.nodes {
        in_degree.entry(nid).or_insert(0);
    }

    // Count incoming edges.
    for (_, node) in &graph.nodes {
        for child in node.children() {
            *in_degree.entry(child).or_insert(0) += 1;
        }
    }

    // Wait — this counts *uses* as in-degree, but we want to process
    // leaves first (nodes with no children, i.e., no dependencies).
    // For Kahn's algorithm on a DAG, in-degree means "number of
    // predecessors/dependencies", not "number of users".
    //
    // Actually, we need to process nodes in dependency order: a node
    // must be processed after all its children (operands). So the edges
    // go from child → parent (operand → user). in_degree = number of
    // operands that haven't been processed yet.

    // Recompute correctly: in_degree = number of children (operands).
    in_degree.clear();
    for (nid, node) in &graph.nodes {
        in_degree.insert(nid, node.children().len());
    }

    // Build reverse adjacency: child → list of parents that depend on it.
    let mut users: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for (nid, node) in &graph.nodes {
        for child in node.children() {
            users.entry(child).or_default().push(nid);
        }
    }

    // Start with leaves (nodes with 0 children = 0 in-degree).
    let mut queue: Vec<NodeId> =
        in_degree.iter().filter(|(_, &deg)| deg == 0).map(|(&nid, _)| nid).collect();

    let mut result = Vec::with_capacity(graph.nodes.len());

    while let Some(nid) = queue.pop() {
        result.push(nid);
        if let Some(parent_list) = users.get(&nid) {
            for &parent in parent_list {
                let deg = in_degree.get_mut(&parent).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push(parent);
                }
            }
        }
    }

    result
}
