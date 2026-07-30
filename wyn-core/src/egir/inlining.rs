//! Context-independent EGIR call-inlining machinery.
//!
//! This module owns the mechanics of cloning a callee value DAG, substituting
//! call operands for function parameters, and replacing the call. It contains
//! no profitability or placement policy; callers decide which calls to inline.

use crate::LookupMap;

use super::graph_ops::{clone_value_subgraph, ConstantCopy};
use super::ir::{Family, Func};
use super::types::{EGraph, ENode, NodeId, PureOp, SkeletonTerminator, WynLanguage};

#[cfg(test)]
#[path = "inlining_tests.rs"]
mod inlining_tests;

/// Return the result root of a function whose body is a single, effect-free
/// block. Such a body is a pure value DAG and can be cloned into a caller
/// without reconstructing control flow or effect ordering.
pub(crate) fn inlineable_return_root<P: Family>(function: &Func<P, WynLanguage>) -> Option<NodeId> {
    if function.graph.skeleton.blocks.len() != 1
        || function.graph.skeleton.blocks.iter().any(|(_, block)| block.control_header.is_some())
        || function.graph.nodes.iter().any(|(_, node)| node.alias.is_some())
    {
        return None;
    }
    let block = &function.graph.skeleton.blocks[function.graph.skeleton.entry];
    if !block.side_effects.is_empty() || !block.params.is_empty() {
        return None;
    }
    match block.term {
        SkeletonTerminator::Return(Some(result)) => Some(result),
        _ => None,
    }
}

/// Number of callee nodes cloned by [`inline_pure_call`] before caller-side
/// hash-consing. Returns `None` for bodies the generic inliner cannot clone.
pub(crate) fn inlineable_node_count<P: Family>(function: &Func<P, WynLanguage>) -> Option<usize> {
    let root = inlineable_return_root(function)?;
    Some(
        wyn_graph::reachable_from_ordered([root], wyn_graph::WalkOrder::DepthFirst, |node, out| {
            out.extend(function.graph.nodes[node].kind.children())
        })
        .len(),
    )
}

/// Inline one pure call by cloning the callee's value DAG into the caller and
/// substituting each `FuncParam` with the corresponding call operand.
///
/// Hash-consing in `caller` provides CSE while the clone is built. The original
/// call is subsumed so every existing use follows the inlined value without a
/// whole-graph reference rewrite.
pub(crate) fn inline_pure_call<P: Family>(
    caller: &mut EGraph<P>,
    call: NodeId,
    callee: &Func<P, WynLanguage>,
) -> Result<NodeId, String> {
    let (called_function, operands) = match caller.nodes.get(call).map(|node| &node.kind) {
        Some(ENode::Pure {
            op: PureOp::Call(function),
            operands,
        }) => (*function, operands.clone()),
        _ => return Err(format!("inline_pure_call: node {call:?} is not a pure call")),
    };
    if called_function != callee.region {
        return Err(format!(
            "inline_pure_call: call targets {:?} but callee `{}` has identity {:?}",
            called_function, callee.name, callee.region
        ));
    }
    if operands.len() != callee.params.len() {
        return Err(format!(
            "inline_pure_call: `{}` has {} call operands but {} parameters",
            callee.name,
            operands.len(),
            callee.params.len()
        ));
    }
    for (index, (operand, (param_ty, _))) in operands.iter().zip(&callee.params).enumerate() {
        let operand_ty = caller
            .nodes
            .get(*operand)
            .map(|node| &node.ty)
            .ok_or_else(|| format!("inline_pure_call: operand {index} has no type"))?;
        if operand_ty != param_ty {
            return Err(format!(
                "inline_pure_call: operand {index} of `{}` has type {operand_ty:?}, expected {param_ty:?}",
                callee.name
            ));
        }
    }

    let root = inlineable_return_root(callee).ok_or_else(|| {
        format!(
            "inline_pure_call: `{}` is not a pure single-block value DAG",
            callee.name
        )
    })?;
    let mut memo = LookupMap::new();
    let reachable =
        wyn_graph::reachable_from_ordered([root], wyn_graph::WalkOrder::DepthFirst, |node, out| {
            out.extend(callee.graph.nodes[node].kind.children())
        });
    for node in reachable {
        let definition = &callee.graph.nodes[node].kind;
        if let ENode::FuncParam { index } = definition {
            let replacement = operands.get(*index).copied().ok_or_else(|| {
                format!(
                    "inline_pure_call: `{}` contains out-of-range FuncParam {index}",
                    callee.name
                )
            })?;
            memo.insert(node, replacement);
        }
    }

    let inlined = clone_value_subgraph(&callee.graph, caller, root, &mut memo, ConstantCopy::Intern, true)?;
    if inlined == call {
        return Err(format!(
            "inline_pure_call: inlining `{}` reproduced the original call",
            callee.name
        ));
    }
    let result_ty = caller
        .nodes
        .get(call)
        .map(|node| &node.ty)
        .ok_or_else(|| format!("inline_pure_call: call {call:?} has no result type"))?;
    let inlined_ty = caller
        .nodes
        .get(inlined)
        .map(|node| &node.ty)
        .ok_or_else(|| format!("inline_pure_call: inlined root {inlined:?} has no type"))?;
    if result_ty != inlined_ty {
        return Err(format!(
            "inline_pure_call: `{}` inlined result has type {inlined_ty:?}, call expects {result_ty:?}",
            callee.name
        ));
    }

    caller.subsume_pure_in_place(call, inlined);
    Ok(inlined)
}
