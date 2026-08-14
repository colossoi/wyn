//! Context-independent EGIR call-inlining machinery.
//!
//! This module owns the mechanics of cloning a callee value DAG, substituting
//! call operands for function parameters, and replacing the call. It contains
//! no profitability or placement policy; callers decide which calls to inline.

use std::collections::VecDeque;

use crate::{LookupMap, LookupSet};

use super::graph_ops::{clone_value_subgraph, ConstantCopy, PureCopy};
use super::ir::{CallEffects, CallSiteId, Family, Func, OperandRef, ResultBinding};
use super::types::{EGraph, PureOp, SkeletonTerminator, ValueId, ValueKind, WynLanguage};
use crate::flow::{BlockId, ControlHeader};

#[cfg(test)]
#[path = "inlining_tests.rs"]
mod inlining_tests;

/// Return the result root of a function whose body is a single, effect-free
/// block. Such a body is a pure value DAG and can be cloned into a caller
/// without reconstructing control flow or effect ordering.
pub(crate) fn inlineable_return_root<P: Family>(function: &Func<P, WynLanguage>) -> Option<ValueId> {
    inlineable_return_binding(function)?.single_value()
}

fn inlineable_return_binding<P: Family>(
    function: &Func<P, WynLanguage>,
) -> Option<&ResultBinding<polytype::Type<crate::ast::TypeName>>> {
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
    match &block.term {
        SkeletonTerminator::Return(Some(result)) if result.ty() == function.result().ty() => Some(result),
        _ => None,
    }
}

/// Number of callee nodes cloned by [`inline_pure_call`] before caller-side
/// hash-consing. Returns `None` for bodies the generic inliner cannot clone.
pub(crate) fn inlineable_node_count<P: Family>(function: &Func<P, WynLanguage>) -> Option<usize> {
    let roots = inlineable_return_binding(function)?.values();
    Some(
        wyn_graph::reachable_from_ordered(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
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
    call: ValueId,
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
    let site = call_site_id(caller, call)?;
    if validate_call(caller, site, callee).is_err()
        || caller.call(site).result().single_value() != Some(call)
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
    call: ValueId,
    block: BlockId,
    callee: &Func<P, WynLanguage>,
) -> Result<ValueId, String> {
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
        if let SkeletonTerminator::Return(Some(result)) = &body.term {
            let value = result.single_value()?;
            if graph.nodes.get(value)?.ty != *function.result().ty() {
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
            ValueKind::FuncParam { parameter } => {
                if graph.nodes[*value].ty != *function.params().get(parameter.index())?.ty() {
                    return None;
                }
            }
            ValueKind::BlockParam { block, index } => {
                if graph.skeleton.blocks.get(*block)?.params.get(*index).map(|value| value.value())
                    != Some(*value)
                {
                    return None;
                }
            }
            ValueKind::SideEffectResult | ValueKind::CallResult { .. } | ValueKind::PlaceLength { .. } => {
                return None;
            }
            ValueKind::Pure { .. } | ValueKind::Union { .. } | ValueKind::Constant(_) => {}
        }
    }
    Some(StructuredInlineSummary {
        blocks,
        nodes: values.len(),
    })
}

fn roots_reach<P: Family>(
    graph: &EGraph<P>,
    roots: impl IntoIterator<Item = ValueId>,
    target: ValueId,
) -> bool {
    wyn_graph::reachable_from_ordered(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        if let Some(definition) = graph.nodes.get(node) {
            out.extend(definition.children());
        }
    })
    .contains(&target)
}

fn uniquely_observing_terminator<P: Family>(graph: &EGraph<P>, call: ValueId) -> Option<BlockId> {
    let mut observer = None;
    for (block, body) in &graph.skeleton.blocks {
        if body
            .side_effects
            .iter()
            .any(|effect| roots_reach(graph, graph.effect_boundary_value_dependencies(effect), call))
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
    call: ValueId,
    block: BlockId,
    callee: &Func<P, WynLanguage>,
) -> Result<ValueId, String> {
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

    let site = call_site_id(caller, call).ok_or_else(|| {
        format!("inline_structured_call_before_terminator: value {call:?} is not a call result")
    })?;
    validate_call(caller, site, callee)?;
    let call_site = caller.call(site);
    if call_site.result().single_value() != Some(call) {
        return Err("structured inlining requires one by-value call result".into());
    }
    let operands = call_site.arguments().to_vec();
    let call_ty = caller.nodes[call].ty.clone();
    if &call_ty != callee.result().ty() {
        return Err(format!(
            "inline_structured_call_before_terminator: `{}` call has type {call_ty:?}, function returns {:?}",
            callee.name,
            callee.result().ty()
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
        if let ValueKind::FuncParam { parameter } = definition.kind {
            let replacement = operands
                .get(parameter.index())
                .and_then(|operand| operand.value())
                .ok_or_else(|| {
                format!(
                    "inline_structured_call_before_terminator: `{}` parameter {} is not a value argument",
                    callee.name,
                    parameter.index()
                )
            })?;
            memo.insert(source, replacement);
        }
    }
    for source_block in &summary.blocks {
        let target_block = block_map[source_block];
        for source_param in &callee.graph.skeleton.blocks[*source_block].params {
            let source_value = source_param.value();
            let target_param =
                caller.add_block_param(target_block, callee.graph.nodes[source_value].ty.clone());
            memo.insert(source_value, target_param);
        }
    }

    let clone_value = |caller: &mut EGraph<P>, memo: &mut LookupMap<ValueId, ValueId>, value| {
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
    let clone_flow = |caller: &mut EGraph<P>, memo: &mut LookupMap<ValueId, ValueId>, value| {
        let value = clone_value(caller, memo, value)?;
        Ok::<_, String>(caller.admit_flow_value(value))
    };
    for source_block in &summary.blocks {
        let source = &callee.graph.skeleton.blocks[*source_block];
        let target_block = block_map[source_block];
        let term = match &source.term {
            SkeletonTerminator::Return(Some(binding)) => SkeletonTerminator::Branch {
                target: continuation,
                args: vec![clone_flow(
                    caller,
                    &mut memo,
                    binding
                        .single_value()
                        .ok_or_else(|| "structured inlining requires scalar returns".to_string())?,
                )?],
            },
            SkeletonTerminator::Branch { target, args } => SkeletonTerminator::Branch {
                target: block_map[target],
                args: args
                    .iter()
                    .map(|value| {
                        clone_flow(caller, &mut memo, value.value())
                    })
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
                    .map(|value| {
                        clone_flow(caller, &mut memo, value.value())
                    })
                    .collect::<Result<_, _>>()?,
                else_target: block_map[else_target],
                else_args: else_args
                    .iter()
                    .map(|value| {
                        clone_flow(caller, &mut memo, value.value())
                    })
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
    old_term.visit_values_mut(|node| {
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
    result: ValueId,
    callee: &Func<P, WynLanguage>,
) -> Result<ValueId, String> {
    let site = call_site_id(caller, result)
        .ok_or_else(|| format!("inline_pure_call: value {result:?} is not a call result"))?;
    let call_results = caller.call(site).result().values();
    let requested = call_results
        .iter()
        .position(|value| *value == result)
        .ok_or_else(|| "call result is absent from its call-site binding".to_string())?;
    let inlined = clone_callee_results(caller, site, callee)?;
    if call_results.len() != inlined.len() {
        return Err("callee return binding does not match its call-site binding".into());
    }
    for (call_result, replacement) in call_results.into_iter().zip(inlined.iter().copied()) {
        if caller.nodes[call_result].ty != caller.nodes[replacement].ty {
            return Err(format!(
                "inline_pure_call: `{}` produced a result type that differs from its call boundary",
                callee.name
            ));
        }
        caller.subsume_pure_in_place(call_result, replacement);
        fold_project_consumers(caller, call_result, replacement);
    }
    Ok(inlined[requested])
}

/// Revisit projections that already existed in the caller before an aggregate
/// call was inlined. Substitution can expose a tuple beneath the call, but
/// those consumers are not rebuilt by [`clone_value_subgraph`], so eagerly
/// propagate their selected components and any nested projections here.
fn fold_project_consumers<P: Family>(graph: &mut EGraph<P>, source: ValueId, replacement: ValueId) {
    let mut pending = vec![(source, replacement)];
    while let Some((source, replacement)) = pending.pop() {
        let consumers = graph
            .nodes
            .iter()
            .filter_map(|(node, definition)| match &definition.kind {
                ValueKind::Pure {
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

fn clone_callee_results<P: Family>(
    caller: &mut EGraph<P>,
    site: CallSiteId,
    callee: &Func<P, WynLanguage>,
) -> Result<Vec<ValueId>, String> {
    validate_call(caller, site, callee)?;
    if !matches!(callee.effects(), CallEffects::Pure) {
        return Err(format!("inline_pure_call: `{}` is not pure", callee.name));
    }

    let roots = inlineable_return_binding(callee).ok_or_else(|| {
        format!(
            "clone_callee_result: `{}` is not a pure single-block value DAG",
            callee.name
        )
    })?
    .values();
    let arguments = caller.call(site).arguments().to_vec();
    let mut memo = LookupMap::new();
    let reachable = wyn_graph::reachable_from_ordered(
        roots.iter().copied(),
        wyn_graph::WalkOrder::DepthFirst,
        |node, out| out.extend(callee.graph.nodes[node].kind.children()),
    );
    for node in reachable {
        let definition = &callee.graph.nodes[node].kind;
        if let ValueKind::FuncParam { parameter } = definition {
            let replacement = arguments
                .get(parameter.index())
                .and_then(|argument| argument.value())
                .ok_or_else(|| {
                format!(
                    "inline_pure_call: `{}` parameter {} is not a value argument",
                    callee.name,
                    parameter.index()
                )
            })?;
            memo.insert(node, replacement);
        }
    }

    let mut inlined = Vec::with_capacity(roots.len());
    for root in roots {
        inlined.push(clone_value_subgraph(
            &callee.graph,
            caller,
            root,
            &mut memo,
            ConstantCopy::Intern,
            true,
            PureCopy::Fold,
        )?);
    }
    Ok(inlined)
}

fn call_site_id<P: Family>(graph: &EGraph<P>, result: ValueId) -> Option<CallSiteId> {
    match graph.nodes.get(result)?.kind() {
        ValueKind::CallResult { call, .. } => Some(*call),
        _ => None,
    }
}

fn validate_call<P: Family>(
    caller: &EGraph<P>,
    site: CallSiteId,
    callee: &Func<P, WynLanguage>,
) -> Result<(), String> {
    let call = caller.call(site);
    if call.callee() != callee.region {
        return Err(format!(
            "inline call targets {:?} but callee `{}` has identity {:?}",
            call.callee(),
            callee.name,
            callee.region
        ));
    }
    if call.effects() != callee.effects() {
        return Err(format!("inline call to `{}` has inconsistent effects", callee.name));
    }
    if call.result().ty() != callee.result().ty() {
        return Err(format!("inline call to `{}` has an inconsistent result tree", callee.name));
    }
    if call.arguments().len() != callee.params().len() {
        return Err(format!(
            "inline call: `{}` has {} call operands but {} parameters",
            callee.name,
            call.arguments().len(),
            callee.params().len()
        ));
    }
    for (index, (argument, parameter)) in call.arguments().iter().zip(callee.params()).enumerate() {
        let matches = match (argument, parameter.representation()) {
            (OperandRef::Value(value), super::ir::OperandType::Value(ty)) => {
                caller.nodes.get(*value).is_some_and(|node| node.ty() == ty)
            }
            (OperandRef::View(view), super::ir::OperandType::View(ty)) => {
                caller.nodes.get(view.value()).is_some_and(|node| node.ty() == &ty.array)
            }
            (OperandRef::Place(place), super::ir::OperandType::Place(ty)) => {
                caller.places().get(*place).is_some_and(|place| {
                    place.ty().pointee == ty.pointee && place.ty().access == ty.access
                })
            }
            _ => false,
        };
        if !matches {
            return Err(format!(
                "inline call: operand {index} of `{}` does not match its parameter representation",
                callee.name,
            ));
        }
    }
    Ok(())
}
