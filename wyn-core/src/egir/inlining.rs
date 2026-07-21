//! Context-independent EGIR call-inlining machinery.
//!
//! This module owns the mechanics of cloning a callee value DAG, substituting
//! call operands for function parameters, and replacing the call. It contains
//! no profitability or placement policy; callers decide which calls to inline.

use crate::LookupMap;

use super::graph_ops::{clone_value_subgraph, ConstantCopy};
use super::ir::Func;
use super::types::{EGraph, ENode, EgirPhase, NodeId, PureOp, SkeletonTerminator, WynLanguage};

#[cfg(test)]
#[path = "inlining_tests.rs"]
mod inlining_tests;

/// Return the result root of a function whose body is a single, effect-free
/// block. Such a body is a pure value DAG and can be cloned into a caller
/// without reconstructing control flow or effect ordering.
pub(crate) fn inlineable_return_root<P: EgirPhase>(function: &Func<P, WynLanguage>) -> Option<NodeId> {
    if function.graph.skeleton.blocks.len() != 1
        || !function.control_headers.is_empty()
        || !function.aliases.is_empty()
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

/// Inline one pure call by cloning the callee's value DAG into the caller and
/// substituting each `FuncParam` with the corresponding call operand.
///
/// Hash-consing in `caller` provides CSE while the clone is built. The original
/// call is subsumed so every existing use follows the inlined value without a
/// whole-graph reference rewrite.
pub(crate) fn inline_pure_call<P: EgirPhase>(
    caller: &mut EGraph<P>,
    call: NodeId,
    callee: &Func<P, WynLanguage>,
) -> Result<NodeId, String> {
    let (called_name, operands) = match caller.nodes.get(call) {
        Some(ENode::Pure {
            op: PureOp::Call(name),
            operands,
        }) => (name.clone(), operands.clone()),
        _ => return Err(format!("inline_pure_call: node {call:?} is not a pure call")),
    };
    if called_name != callee.name {
        return Err(format!(
            "inline_pure_call: call targets `{called_name}` but callee is `{}`",
            callee.name
        ));
    }
    if operands.len() != callee.params.len() {
        return Err(format!(
            "inline_pure_call: `{called_name}` has {} call operands but {} parameters",
            operands.len(),
            callee.params.len()
        ));
    }
    for (index, (operand, (param_ty, _))) in operands.iter().zip(&callee.params).enumerate() {
        let operand_ty = caller
            .types
            .get(operand)
            .ok_or_else(|| format!("inline_pure_call: operand {index} has no type"))?;
        if operand_ty != param_ty {
            return Err(format!(
                "inline_pure_call: operand {index} of `{called_name}` has type {operand_ty:?}, expected {param_ty:?}"
            ));
        }
    }

    let root = inlineable_return_root(callee)
        .ok_or_else(|| format!("inline_pure_call: `{called_name}` is not a pure single-block value DAG"))?;
    let mut memo = LookupMap::new();
    for (node, definition) in &callee.graph.nodes {
        if let ENode::FuncParam { index } = definition {
            let replacement = operands.get(*index).copied().ok_or_else(|| {
                format!("inline_pure_call: `{called_name}` contains out-of-range FuncParam {index}")
            })?;
            memo.insert(node, replacement);
        }
    }

    let inlined = clone_value_subgraph(&callee.graph, caller, root, &mut memo, ConstantCopy::Intern, true)?;
    if inlined == call {
        return Err(format!(
            "inline_pure_call: inlining `{called_name}` reproduced the original call"
        ));
    }
    let result_ty = caller
        .types
        .get(&call)
        .ok_or_else(|| format!("inline_pure_call: call {call:?} has no result type"))?;
    let inlined_ty = caller
        .types
        .get(&inlined)
        .ok_or_else(|| format!("inline_pure_call: inlined root {inlined:?} has no type"))?;
    if result_ty != inlined_ty {
        return Err(format!(
            "inline_pure_call: `{called_name}` inlined result has type {inlined_ty:?}, call expects {result_ty:?}"
        ));
    }

    caller.subsume_pure_in_place(call, inlined);
    Ok(inlined)
}
