//! Context-independent EGIR call-inlining machinery.
//!
//! This module owns the mechanics of cloning a callee value DAG, substituting
//! call operands for function parameters, and replacing the call. It contains
//! no profitability or placement policy; callers decide which calls to inline.

use std::collections::VecDeque;

use crate::{LookupMap, LookupSet};

use super::graph_ops::{clone_value_subgraph, ConstantCopy, PureCopy};
use super::ir::{Family, Func};
use super::types::{EGraph, ENode, NodeId, PureOp, SkeletonTerminator, WynLanguage};
use crate::flow::{BlockId, ControlHeader};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InlineCost {
    pub nodes: usize,
    pub blocks: usize,
}

#[derive(Clone, Debug)]
struct StructuredInlineSummary {
    blocks: Vec<BlockId>,
    nodes: usize,
}

/// Return the bounded cloning cost for a call at one caller block. DAG calls
/// need no skeleton growth. Structured calls additionally require their value
/// to be observed by exactly this block's terminator so CFG splicing preserves
/// evaluation scope.
pub(crate) fn inlineable_call_cost_at_block<P: Family>(
    caller: &EGraph<P>,
    call: NodeId,
    block: BlockId,
    callee: &Func<P, WynLanguage>,
) -> Option<InlineCost> {
    if let Some(nodes) = inlineable_node_count(callee) {
        return Some(InlineCost { nodes, blocks: 0 });
    }
    if caller.skeleton.blocks.get(block)?.control_header.is_some()
        || uniquely_observing_terminator(caller, call) != Some(block)
    {
        return None;
    }
    let (function, operands) = match &caller.nodes.get(call)?.kind {
        ENode::Pure {
            op: PureOp::Call(function),
            operands,
        } => (*function, operands),
        _ => return None,
    };
    if function != callee.region
        || caller.nodes[call].ty != callee.return_ty
        || validate_operands(caller, operands, callee).is_err()
    {
        return None;
    }
    let summary = structured_inline_summary(callee)?;
    Some(InlineCost {
        nodes: summary.nodes,
        // Every callee block is cloned and one continuation receives returns.
        blocks: summary.blocks.len() + 1,
    })
}

/// Inline a call at the block whose terminator uniquely observes its result.
/// Single-block value DAGs retain their cheaper graph-only path; structured
/// selection CFGs are spliced before the observing terminator.
pub(crate) fn inline_call_at_block<P: Family>(
    caller: &mut EGraph<P>,
    call: NodeId,
    block: BlockId,
    callee: &Func<P, WynLanguage>,
) -> Result<NodeId, String> {
    if inlineable_return_root(callee).is_some() {
        inline_pure_call(caller, call, callee)
    } else {
        inline_structured_call_before_terminator(caller, call, block, callee)
    }
}

fn structured_inline_summary<P: Family>(
    function: &Func<P, WynLanguage>,
) -> Option<StructuredInlineSummary> {
    let graph = &function.graph;
    if graph.skeleton.blocks.is_empty()
        || graph.nodes.iter().any(|(_, node)| node.alias.is_some())
        || graph.skeleton.verify_branch_arities().is_err()
    {
        return None;
    }

    let blocks = wyn_graph::reachable_from_ordered(
        [graph.skeleton.entry],
        wyn_graph::WalkOrder::DepthFirst,
        |block, out| out.extend(graph.skeleton.blocks[block].term.successors()),
    );
    if blocks.len() != graph.skeleton.blocks.len() {
        return None;
    }
    let reachable: LookupSet<BlockId> = blocks.iter().copied().collect();
    let mut returns = 0usize;
    for block in &blocks {
        let body = &graph.skeleton.blocks[*block];
        if !body.side_effects.is_empty()
            || matches!(body.control_header, Some(ControlHeader::Loop { .. }))
            || matches!(
                body.term,
                SkeletonTerminator::Return(None) | SkeletonTerminator::Unreachable
            )
        {
            return None;
        }
        if let Some(ControlHeader::Selection { merge }) = body.control_header {
            if !reachable.contains(&merge) {
                return None;
            }
        }
        if let SkeletonTerminator::Return(Some(result)) = body.term {
            if graph.nodes.get(result)?.ty != function.return_ty {
                return None;
            }
            returns += 1;
        }
    }
    if returns == 0 || !graph.skeleton.blocks[graph.skeleton.entry].params.is_empty() {
        return None;
    }

    // Acyclicity is a semantic capability boundary independent of structured
    // metadata: malformed or headerless backedges must not enter this path.
    let mut indegree: LookupMap<BlockId, usize> = blocks.iter().map(|block| (*block, 0)).collect();
    for block in &blocks {
        for successor in graph.skeleton.blocks[*block].term.successors() {
            *indegree.get_mut(&successor)? += 1;
        }
    }
    let mut ready: VecDeque<BlockId> =
        blocks.iter().copied().filter(|block| indegree[block] == 0).collect();
    let mut visited = 0usize;
    while let Some(block) = ready.pop_front() {
        visited += 1;
        for successor in graph.skeleton.blocks[block].term.successors() {
            let degree = indegree.get_mut(&successor)?;
            *degree -= 1;
            if *degree == 0 {
                ready.push_back(successor);
            }
        }
    }
    if visited != blocks.len() {
        return None;
    }

    let roots = blocks.iter().flat_map(|block| graph.skeleton.blocks[*block].term.referenced_nodes());
    let values = wyn_graph::reachable_from_ordered(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        if let Some(definition) = graph.nodes.get(node) {
            out.extend(definition.children());
        }
    });
    for value in &values {
        match &graph.nodes.get(*value)?.kind {
            ENode::FuncParam { index } => {
                if graph.nodes[*value].ty != function.params.get(*index)?.0 {
                    return None;
                }
            }
            ENode::BlockParam { block, index } => {
                if graph.skeleton.blocks.get(*block)?.params.get(*index) != Some(value) {
                    return None;
                }
            }
            ENode::SideEffectResult => return None,
            ENode::Pure { .. } | ENode::Union { .. } | ENode::Constant(_) => {}
        }
    }
    Some(StructuredInlineSummary {
        blocks,
        nodes: values.len(),
    })
}

fn roots_reach<P: Family>(
    graph: &EGraph<P>,
    roots: impl IntoIterator<Item = NodeId>,
    target: NodeId,
) -> bool {
    wyn_graph::reachable_from_ordered(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        if let Some(definition) = graph.nodes.get(node) {
            out.extend(definition.children());
        }
    })
    .contains(&target)
}

fn uniquely_observing_terminator<P: Family>(graph: &EGraph<P>, call: NodeId) -> Option<BlockId> {
    let mut observer = None;
    for (block, body) in &graph.skeleton.blocks {
        if body
            .side_effects
            .iter()
            .any(|effect| roots_reach(graph, effect.operand_nodes.iter().copied(), call))
        {
            return None;
        }
        if roots_reach(graph, body.term.referenced_nodes(), call) {
            if observer.replace(block).is_some() {
                return None;
            }
        }
    }
    observer
}

fn inline_structured_call_before_terminator<P: Family>(
    caller: &mut EGraph<P>,
    call: NodeId,
    block: BlockId,
    callee: &Func<P, WynLanguage>,
) -> Result<NodeId, String> {
    let summary = structured_inline_summary(callee).ok_or_else(|| {
        format!(
            "inline_structured_call_before_terminator: `{}` is not a bounded effect-free selection CFG",
            callee.name
        )
    })?;
    if caller.skeleton.blocks.get(block).is_none() {
        return Err(format!(
            "inline_structured_call_before_terminator: missing caller block {block:?}"
        ));
    }
    if caller.skeleton.blocks[block].control_header.is_some() {
        return Err(format!(
            "inline_structured_call_before_terminator: caller block {block:?} is a structured header"
        ));
    }
    if uniquely_observing_terminator(caller, call) != Some(block) {
        return Err(format!(
            "inline_structured_call_before_terminator: call {call:?} is not uniquely observed by {block:?}'s terminator"
        ));
    }

    let (called_function, operands) = match caller.nodes.get(call).map(|node| &node.kind) {
        Some(ENode::Pure {
            op: PureOp::Call(function),
            operands,
        }) => (*function, operands.clone()),
        _ => {
            return Err(format!(
                "inline_structured_call_before_terminator: node {call:?} is not a pure call"
            ));
        }
    };
    if called_function != callee.region {
        return Err(format!(
            "inline_structured_call_before_terminator: call targets {:?} but callee `{}` has identity {:?}",
            called_function, callee.name, callee.region
        ));
    }
    validate_operands(caller, &operands, callee)?;
    let call_ty = caller.nodes[call].ty.clone();
    if call_ty != callee.return_ty {
        return Err(format!(
            "inline_structured_call_before_terminator: `{}` call has type {call_ty:?}, function returns {:?}",
            callee.name, callee.return_ty
        ));
    }

    let mut block_map = LookupMap::new();
    for source in &summary.blocks {
        block_map.insert(*source, caller.skeleton.create_block());
    }
    let continuation = caller.skeleton.create_block();
    let result = caller.add_block_param(continuation, call_ty);

    let mut memo = LookupMap::new();
    for (source, definition) in &callee.graph.nodes {
        if let ENode::FuncParam { index } = definition.kind {
            let replacement = operands.get(index).copied().ok_or_else(|| {
                format!(
                    "inline_structured_call_before_terminator: `{}` contains out-of-range FuncParam {index}",
                    callee.name
                )
            })?;
            memo.insert(source, replacement);
        }
    }
    for source_block in &summary.blocks {
        let target_block = block_map[source_block];
        for source_param in &callee.graph.skeleton.blocks[*source_block].params {
            let target_param =
                caller.add_block_param(target_block, callee.graph.nodes[*source_param].ty.clone());
            memo.insert(*source_param, target_param);
        }
    }

    let clone_value = |caller: &mut EGraph<P>, memo: &mut LookupMap<NodeId, NodeId>, value| {
        clone_value_subgraph(
            &callee.graph,
            caller,
            value,
            memo,
            ConstantCopy::Intern,
            true,
            PureCopy::Fold,
        )
    };
    for source_block in &summary.blocks {
        let source = &callee.graph.skeleton.blocks[*source_block];
        let target_block = block_map[source_block];
        let term = match &source.term {
            SkeletonTerminator::Return(Some(value)) => SkeletonTerminator::Branch {
                target: continuation,
                args: vec![clone_value(caller, &mut memo, *value)?],
            },
            SkeletonTerminator::Branch { target, args } => SkeletonTerminator::Branch {
                target: block_map[target],
                args: args
                    .iter()
                    .map(|value| clone_value(caller, &mut memo, *value))
                    .collect::<Result<_, _>>()?,
            },
            SkeletonTerminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => SkeletonTerminator::CondBranch {
                cond: clone_value(caller, &mut memo, *cond)?,
                then_target: block_map[then_target],
                then_args: then_args
                    .iter()
                    .map(|value| clone_value(caller, &mut memo, *value))
                    .collect::<Result<_, _>>()?,
                else_target: block_map[else_target],
                else_args: else_args
                    .iter()
                    .map(|value| clone_value(caller, &mut memo, *value))
                    .collect::<Result<_, _>>()?,
            },
            SkeletonTerminator::Return(None) | SkeletonTerminator::Unreachable => {
                return Err(format!(
                    "inline_structured_call_before_terminator: `{}` changed after eligibility analysis",
                    callee.name
                ));
            }
        };
        caller.skeleton.blocks[target_block].term = term;
        caller.skeleton.blocks[target_block].control_header =
            source.control_header.as_ref().map(|header| header.remap(&|source| block_map[&source]));
    }

    let mut old_term = std::mem::replace(
        &mut caller.skeleton.blocks[block].term,
        SkeletonTerminator::Branch {
            target: block_map[&callee.graph.skeleton.entry],
            args: Vec::new(),
        },
    );
    // The continuation parameter is only in scope after the cloned callee.
    // Rewrite the observing value graph and terminator explicitly instead of
    // unioning the original call with that parameter: extraction may otherwise
    // choose the block parameter while elaborating a value placed before its
    // defining continuation.
    caller.replace_node_references(call, result);
    old_term.visit_nodes_mut(|node| {
        if *node == call {
            *node = result;
        }
    });
    caller.skeleton.blocks[continuation].term = old_term;
    caller.skeleton.verify_branch_arities()?;
    caller.verify_hash_cons()?;
    Ok(result)
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
    let inlined = clone_callee_result(caller, &operands, callee)?;
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

    if inlined == call {
        return Err(format!(
            "inline_pure_call: inlining `{}` reproduced the original call",
            callee.name
        ));
    }
    caller.subsume_pure_in_place(call, inlined);
    fold_project_consumers(caller, call, inlined);
    Ok(inlined)
}

/// Revisit projections that already existed in the caller before an aggregate
/// call was inlined. Substitution can expose a tuple beneath the call, but
/// those consumers are not rebuilt by [`clone_value_subgraph`], so eagerly
/// propagate their selected components and any nested projections here.
fn fold_project_consumers<P: Family>(graph: &mut EGraph<P>, source: NodeId, replacement: NodeId) {
    let mut pending = vec![(source, replacement)];
    while let Some((source, replacement)) = pending.pop() {
        let consumers = graph
            .nodes
            .iter()
            .filter_map(|(node, definition)| match &definition.kind {
                ENode::Pure {
                    op: PureOp::Project { index },
                    operands,
                } if operands.as_slice() == [source] => Some((node, *index, definition.ty.clone())),
                _ => None,
            })
            .collect::<Vec<_>>();
        for (project, index, ty) in consumers {
            let Some(selected) = graph.try_algebraic_fold(&PureOp::Project { index }, &[replacement], &ty)
            else {
                continue;
            };
            if selected == project {
                continue;
            }
            graph.subsume_pure_in_place(project, selected);
            pending.push((project, selected));
        }
    }
}

/// Replace the result of an effect-classified call with a cloned pure callee
/// DAG. Resource erasure conservatively anchors calls that carry storage
/// views, even when the specialized callee contains only index operations.
/// Removing that wrapper lets a containing fixed-array helper become a pure
/// value DAG and participate in ordinary call inlining.
pub(crate) fn inline_effect_call_to_pure_callee<P: Family>(
    caller: &mut EGraph<P>,
    result: NodeId,
    operands: &[NodeId],
    callee: &Func<P, WynLanguage>,
) -> Result<NodeId, String> {
    if !matches!(
        caller.nodes.get(result).map(|node| &node.kind),
        Some(ENode::SideEffectResult)
    ) {
        return Err(format!(
            "inline_effect_call_to_pure_callee: result {result:?} is not a side-effect result"
        ));
    }
    let inlined = clone_callee_result(caller, operands, callee)?;
    let result_ty = &caller.nodes[result].ty;
    let inlined_ty = &caller.nodes[inlined].ty;
    if result_ty != inlined_ty {
        return Err(format!(
            "inline_effect_call_to_pure_callee: `{}` inlined result has type {inlined_ty:?}, effect expects {result_ty:?}",
            callee.name
        ));
    }
    caller.nodes[result].kind = ENode::Union {
        left: inlined,
        right: inlined,
    };
    Ok(inlined)
}

fn clone_callee_result<P: Family>(
    caller: &mut EGraph<P>,
    operands: &[NodeId],
    callee: &Func<P, WynLanguage>,
) -> Result<NodeId, String> {
    validate_operands(caller, operands, callee)?;

    let root = inlineable_return_root(callee).ok_or_else(|| {
        format!(
            "clone_callee_result: `{}` is not a pure single-block value DAG",
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

    let inlined = clone_value_subgraph(
        &callee.graph,
        caller,
        root,
        &mut memo,
        ConstantCopy::Intern,
        true,
        PureCopy::Fold,
    )?;
    let inlined_ty = caller
        .nodes
        .get(inlined)
        .map(|node| &node.ty)
        .ok_or_else(|| format!("clone_callee_result: inlined root {inlined:?} has no type"))?;
    if &callee.return_ty != inlined_ty {
        return Err(format!(
            "clone_callee_result: `{}` inlined result has type {inlined_ty:?}, function returns {:?}",
            callee.name, callee.return_ty
        ));
    }
    Ok(inlined)
}

fn validate_operands<P: Family>(
    caller: &EGraph<P>,
    operands: &[NodeId],
    callee: &Func<P, WynLanguage>,
) -> Result<(), String> {
    if operands.len() != callee.params.len() {
        return Err(format!(
            "inline call: `{}` has {} call operands but {} parameters",
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
            .ok_or_else(|| format!("inline call: operand {index} has no type"))?;
        if operand_ty != param_ty {
            return Err(format!(
                "inline call: operand {index} of `{}` has type {operand_ty:?}, expected {param_ty:?}",
                callee.name
            ));
        }
    }
    Ok(())
}
