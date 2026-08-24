//! Context-independent EGIR call-inlining machinery.
//!
//! This module owns the mechanics of cloning a callee value DAG, substituting
//! call operands for function parameters, and replacing the call. It contains
//! no profitability or placement policy; callers decide which calls to inline.

use crate::ast;
use std::collections::VecDeque;
use wyn_base::IdSource;

use crate::{LookupMap, LookupSet};

use super::graph_ops::{clone_body_substituting, clone_value_subgraph, ConstantCopy, PureCopy};
use super::ir::{
    CallEffects, CallSiteId, Family, Func, OperandRef, PlaceDestination, ResultBinding, ResultDestination,
    SideEffectSite,
};
use super::types::{
    EGraph, EffectOp, EffectToken, SideEffectKind, SkeletonTerminator, ValueId, ValueKind, WynLanguage,
};
use crate::flow::{BlockId, ControlHeader};

#[cfg(test)]
#[path = "inlining_tests.rs"]
mod inlining_tests;

fn inlineable_return_binding<P: Family>(
    function: &Func<P, WynLanguage>,
) -> Option<&ResultBinding<polytype::Type<ast::TypeName>>> {
    if function.graph.skeleton.blocks.len() != 1
        || function.graph.skeleton.blocks.iter().any(|(_, block)| block.control_header.is_some())
    {
        return None;
    }
    let block = &function.graph.skeleton.blocks[function.graph.skeleton.entry];
    if !block.params.is_empty()
        || block.side_effects.iter().any(|effect| {
            let SideEffectKind::Effect(EffectOp::Call { site }) = effect.kind() else {
                return true;
            };
            function.graph.call(*site).effects() != CallEffects::Pure
        })
    {
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
    if !matches!(function.effects(), CallEffects::Pure) {
        return None;
    }
    let roots = inlineable_return_binding(function)?.values();
    Some(
        wyn_graph::reachable_from_ordered(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
            out.extend(function.graph.value_dependencies(node))
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
    let site = call_site_id(caller, call)?;
    if caller.skeleton.blocks.get(block)?.control_header.is_some()
        || uniquely_observing_call_terminator(caller, site) != Some(block)
    {
        return None;
    }
    if validate_call(caller, site, callee).is_err() {
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
    if inlineable_node_count(callee).is_some() {
        inline_pure_call(caller, call, callee)
    } else {
        inline_structured_call_before_terminator(caller, call, block, callee)
    }
}

/// Inline one explicitly sequenced call by cloning its whole physical body
/// and splicing every cloned return into the call continuation.
pub(crate) fn inline_effectful_call<P: Family>(
    caller: &mut EGraph<P>,
    effect: SideEffectSite,
    callee: &Func<P, WynLanguage>,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<InlineCost, String> {
    let site = effect_call_site(caller, effect)
        .ok_or_else(|| format!("inline_effectful_call: {effect:?} is not an explicit call effect"))?;
    validate_call(caller, site, callee)?;
    complete_body_summary(callee).map_err(|reason| {
        format!(
            "inline_effectful_call: `{}` has an invalid physical body: {reason}",
            callee.name
        )
    })?;

    let boundary = caller.call(site).clone();
    let call_results = boundary.result().values();
    let continuation = caller.skeleton.split_block_before_effect(effect.block, effect.index);
    let continuation_call = SideEffectSite {
        block: continuation,
        index: 0,
    };
    let removed = caller.skeleton.remove_effect_splicing_dependencies(continuation_call);
    debug_assert!(matches!(
        removed.kind(),
        SideEffectKind::Effect(EffectOp::Call { site: removed_site }) if *removed_site == site
    ));
    let cloned = clone_body_substituting(
        &callee.graph,
        caller,
        boundary.argument_bindings(),
        &[],
        effect_ids,
    )?;
    let cost = InlineCost {
        nodes: cloned.node_count,
        blocks: cloned.block_count + 1,
    };
    if cloned.returns.is_empty() {
        return Err(format!(
            "inline_effectful_call: `{}` has no returning path",
            callee.name
        ));
    }
    for (_, result) in &cloned.returns {
        validate_cloned_return(result, boundary.result())?;
    }

    let replacements = if cloned.returns.len() == 1 {
        cloned.returns[0].1.values()
    } else {
        call_results
            .iter()
            .map(|result| caller.add_block_param(continuation, caller.nodes[*result].ty.clone()))
            .collect::<Vec<_>>()
    };
    if replacements.len() != call_results.len() {
        return Err(format!(
            "inline_effectful_call: `{}` returned {} values for {} call results",
            callee.name,
            replacements.len(),
            call_results.len()
        ));
    }

    for (return_block, binding) in &cloned.returns {
        let args =
            if cloned.returns.len() == 1 { Vec::new() } else { caller.admit_flow_values(binding.values()) };
        caller.skeleton.blocks[*return_block].term = SkeletonTerminator::Branch {
            target: continuation,
            args,
        };
    }
    caller.skeleton.blocks[effect.block].term = SkeletonTerminator::Branch {
        target: cloned.entry,
        args: Vec::new(),
    };

    for (&call_result, &replacement) in call_results.iter().zip(&replacements) {
        if caller.nodes[call_result].ty != caller.nodes[replacement].ty {
            return Err(format!(
                "inline_effectful_call: `{}` produced result type {:?} for call-boundary type {:?}",
                callee.name, caller.nodes[replacement].ty, caller.nodes[call_result].ty,
            ));
        }
        caller.replace_value_references(call_result, replacement);
        caller.install_aliases([(call_result, replacement)]);
    }
    super::graph_ops::fold_exposed_projections(caller);
    for &replacement in &replacements {
        super::graph_ops::materialize_place_backed_projections(
            caller,
            replacement,
            continuation,
            effect_ids,
        );
    }
    caller.skeleton.verify_branch_arities()?;
    caller.verify_hash_cons()?;
    Ok(cost)
}

fn effect_call_site<P: Family>(graph: &EGraph<P>, effect: SideEffectSite) -> Option<CallSiteId> {
    match graph.skeleton.get_effect(effect)?.kind() {
        SideEffectKind::Effect(EffectOp::Call { site }) => Some(*site),
        _ => None,
    }
}

fn remove_explicit_call_site<P: Family>(
    graph: &mut EGraph<P>,
    site: CallSiteId,
    effect: SideEffectSite,
) -> Result<(), String> {
    let removed = graph.skeleton.remove_effect_splicing_dependencies(effect);
    if !matches!(
        removed.kind(),
        SideEffectKind::Effect(EffectOp::Call { site: removed_site }) if *removed_site == site
    ) {
        return Err(format!("call {site:?} lost its explicit skeleton site"));
    }
    Ok(())
}

fn validate_cloned_return<Ty: PartialEq + Clone>(
    returned: &ResultBinding<Ty>,
    boundary: &ResultBinding<Ty>,
) -> Result<(), String> {
    if returned.ty() != boundary.ty() || returned.destination_count() != boundary.destination_count() {
        return Err("cloned return does not match its call result tree".into());
    }
    for (returned, boundary) in returned.destination_leaves().into_iter().zip(boundary.destination_leaves())
    {
        let Some((returned_ty, returned)) = returned.single_destination() else {
            unreachable!()
        };
        let Some((boundary_ty, boundary)) = boundary.single_destination() else {
            unreachable!()
        };
        let matches = returned_ty == boundary_ty
            && match (returned, boundary) {
                (ResultDestination::ReturnValue(_), ResultDestination::ReturnValue(_)) => true,
                (
                    ResultDestination::Place(PlaceDestination::Fixed(left)),
                    ResultDestination::Place(PlaceDestination::Fixed(right)),
                ) => left == right,
                (
                    ResultDestination::Place(PlaceDestination::Bounded {
                        storage: left_storage,
                        length: left_length,
                    }),
                    ResultDestination::Place(PlaceDestination::Bounded {
                        storage: right_storage,
                        length: right_length,
                    }),
                ) => left_storage == right_storage && left_length == right_length,
                _ => false,
            };
        if !matches {
            return Err("cloned return changes a physical result route".into());
        }
    }
    Ok(())
}

fn structured_inline_summary<P: Family>(
    function: &Func<P, WynLanguage>,
) -> Option<StructuredInlineSummary> {
    let graph = &function.graph;
    if graph.skeleton.blocks.is_empty() || graph.skeleton.verify_branch_arities().is_err() {
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
            if result.ty() != function.result().ty()
                || result.destination_count() != function.result().destination_count()
                || result.destination_count() != result.values().len()
                || result.destination_leaves().iter().any(|leaf| {
                    leaf.single_value()
                        .and_then(|value| graph.nodes.get(value))
                        .is_none_or(|value| value.ty() != leaf.ty())
                })
            {
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
        out.extend(graph.value_dependencies(node));
    });
    for value in &values {
        match &graph.nodes.get(*value)?.kind {
            ValueKind::FuncParam { parameter } => {
                if graph.nodes[*value].ty != *function.params().get(*parameter)?.ty() {
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
            ValueKind::SideEffectResult
            | ValueKind::CallResult { .. }
            | ValueKind::PlaceLength { .. }
            | ValueKind::PlaceView { .. } => {
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

fn complete_body_summary<P: Family>(
    function: &Func<P, WynLanguage>,
) -> Result<StructuredInlineSummary, String> {
    let graph = &function.graph;
    graph.skeleton.verify_branch_arities()?;
    let blocks = wyn_graph::reachable_from_ordered(
        [graph.skeleton.entry],
        wyn_graph::WalkOrder::DepthFirst,
        |block, out| out.extend(graph.skeleton.blocks[block].term.successors()),
    );
    if blocks.len() != graph.skeleton.blocks.len()
        || !graph.skeleton.blocks[graph.skeleton.entry].params.is_empty()
    {
        return Err("the body contains unreachable blocks or an entry block parameter".into());
    }

    let expected_values = function
        .result()
        .destination_leaves()
        .into_iter()
        .filter(|leaf| {
            matches!(
                leaf.single_destination(),
                Some((_, ResultDestination::ReturnValue(_)))
            )
        })
        .count();
    let expected_places = function
        .result()
        .destination_leaves()
        .into_iter()
        .map(|leaf| match leaf.single_destination() {
            Some((_, ResultDestination::Place(PlaceDestination::Fixed(_)))) => 1,
            Some((_, ResultDestination::Place(PlaceDestination::Bounded { .. }))) => 2,
            _ => 0,
        })
        .sum::<usize>();
    let mut returns = 0usize;
    for block in &blocks {
        let body = &graph.skeleton.blocks[*block];
        if body.side_effects.iter().any(|effect| matches!(effect.kind(), SideEffectKind::Soac(_)))
            || matches!(body.term, SkeletonTerminator::Return(None))
        {
            return Err(format!(
                "block {block:?} retains a SOAC or has no physical return"
            ));
        }
        if let SkeletonTerminator::Return(Some(result)) = &body.term {
            if result.ty() != function.result().ty()
                || result.destination_count() != function.result().destination_count()
                || result.values().len() != expected_values
                || result.places().len() != expected_places
            {
                return Err(format!(
                    "block {block:?} returns type {:?}, {} destinations, {} values, and {} places; the callable ABI requires type {:?}, {} destinations, {expected_values} values, and {expected_places} places",
                    result.ty(),
                    result.destination_count(),
                    result.values().len(),
                    result.places().len(),
                    function.result().ty(),
                    function.result().destination_count(),
                ));
            }
            returns += 1;
        }
    }
    if returns == 0 {
        return Err("the body has no returning path".into());
    }
    Ok(StructuredInlineSummary {
        blocks,
        nodes: graph.nodes.len(),
    })
}

fn roots_reach_any<P: Family>(
    graph: &EGraph<P>,
    roots: impl IntoIterator<Item = ValueId>,
    targets: &LookupSet<ValueId>,
) -> bool {
    wyn_graph::reachable_from_ordered(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        out.extend(graph.value_dependencies(node));
    })
    .into_iter()
    .any(|value| targets.contains(&value))
}

fn uniquely_observing_call_terminator<P: Family>(graph: &EGraph<P>, site: CallSiteId) -> Option<BlockId> {
    let results = graph.call(site).result().values().into_iter().collect::<LookupSet<_>>();
    if results.is_empty() {
        return None;
    }
    let mut observer = None;
    for (block, body) in &graph.skeleton.blocks {
        if body.side_effects.iter().any(|effect| {
            roots_reach_any(graph, graph.effect_boundary_value_dependencies(effect), &results)
        }) {
            return None;
        }
        if roots_reach_any(graph, body.term.referenced_nodes(), &results) {
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
    let site = call_site_id(caller, call).ok_or_else(|| {
        format!("inline_structured_call_before_terminator: value {call:?} is not a call result")
    })?;
    if uniquely_observing_call_terminator(caller, site) != Some(block) {
        return Err(format!(
            "inline_structured_call_before_terminator: call {call:?} is not uniquely observed by {block:?}'s terminator"
        ));
    }

    validate_call(caller, site, callee)?;
    let anchor = caller
        .side_effect_index()
        .call_site(site)
        .ok_or_else(|| format!("call {site:?} has no explicit skeleton site"))?;
    remove_explicit_call_site(caller, site, anchor)?;
    let call_site = caller.call(site);
    let call_results = call_site.result().values();
    if call_site.result().destination_count() != call_results.len() {
        return Err("structured inlining requires by-value call results".into());
    }
    let requested = call_results
        .iter()
        .position(|result| *result == call)
        .ok_or_else(|| "structured inlining trigger is absent from its call boundary".to_string())?;
    let operands = call_site.argument_bindings().clone();

    let mut block_map = LookupMap::new();
    for source in &summary.blocks {
        block_map.insert(*source, caller.skeleton.create_block());
    }
    let continuation = caller.skeleton.create_block();
    let results = call_results
        .iter()
        .map(|result| caller.add_block_param(continuation, caller.nodes[*result].ty.clone()))
        .collect::<Vec<_>>();

    let mut memo = LookupMap::new();
    for (source, definition) in &callee.graph.nodes {
        if let ValueKind::FuncParam { parameter } = definition.kind {
            let replacement =
                operands.get(&parameter).and_then(|operand| operand.value()).ok_or_else(|| {
                    format!(
                    "inline_structured_call_before_terminator: `{}` parameter {parameter:?} is not a value argument",
                    callee.name,
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
                args: binding
                    .values()
                    .into_iter()
                    .map(|value| clone_flow(caller, &mut memo, value))
                    .collect::<Result<_, _>>()?,
            },
            SkeletonTerminator::Branch { target, args } => SkeletonTerminator::Branch {
                target: block_map[target],
                args: args
                    .iter()
                    .map(|value| clone_flow(caller, &mut memo, value.value()))
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
                    .map(|value| clone_flow(caller, &mut memo, value.value()))
                    .collect::<Result<_, _>>()?,
                else_target: block_map[else_target],
                else_args: else_args
                    .iter()
                    .map(|value| clone_flow(caller, &mut memo, value.value()))
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
    for (&call_result, &result) in call_results.iter().zip(&results) {
        caller.replace_node_references(call_result, result);
    }
    old_term.visit_values_mut(|node| {
        if let Some(index) = call_results.iter().position(|result| result == node) {
            *node = results[index];
        }
    });
    caller.skeleton.blocks[continuation].term = old_term;
    caller.skeleton.verify_branch_arities()?;
    caller.verify_hash_cons()?;
    Ok(results[requested])
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
    let anchor = caller
        .side_effect_index()
        .call_site(site)
        .ok_or_else(|| format!("call {site:?} has no explicit skeleton site"))?;
    let existing_calls = caller.calls().keys().collect::<LookupSet<_>>();
    let inlined = clone_callee_results(caller, site, callee)?;
    let cloned_calls =
        caller.calls().keys().filter(|candidate| !existing_calls.contains(candidate)).collect::<Vec<_>>();
    let cloned_call_effects = cloned_calls
        .into_iter()
        .map(|site| {
            if caller.call(site).effects() != CallEffects::Pure {
                return Err(format!(
                    "inline_pure_call: `{}` contains a non-pure nested call",
                    callee.name
                ));
            }
            Ok(super::ir::SideEffect::new(
                SideEffectKind::Effect(EffectOp::Call { site }),
                smallvec::smallvec![],
                None,
                None,
                None,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
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
        caller.replace_value_references(call_result, replacement);
        caller.install_aliases([(call_result, replacement)]);
    }
    remove_explicit_call_site(caller, site, anchor)?;
    caller.skeleton.blocks[anchor.block]
        .side_effects
        .splice(anchor.index..anchor.index, cloned_call_effects);
    super::graph_ops::fold_exposed_projections(caller);
    Ok(inlined[requested])
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

    let roots = inlineable_return_binding(callee)
        .ok_or_else(|| {
            format!(
                "clone_callee_result: `{}` is not a pure single-block value DAG",
                callee.name
            )
        })?
        .values();
    let arguments = caller.call(site).argument_bindings().clone();
    let mut memo = LookupMap::new();
    let reachable = wyn_graph::reachable_from_ordered(
        roots.iter().copied(),
        wyn_graph::WalkOrder::DepthFirst,
        |node, out| out.extend(callee.graph.value_dependencies(node)),
    );
    for node in reachable {
        let definition = &callee.graph.nodes[node].kind;
        if let ValueKind::FuncParam { parameter } = definition {
            let replacement =
                arguments.get(parameter).and_then(|argument| argument.value()).ok_or_else(|| {
                    format!(
                        "inline_pure_call: `{}` parameter {parameter:?} is not a value argument",
                        callee.name,
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
        return Err(format!(
            "inline call to `{}` has inconsistent effects",
            callee.name
        ));
    }
    if call.result().ty() != callee.result().ty() {
        return Err(format!(
            "inline call to `{}` has an inconsistent result tree",
            callee.name
        ));
    }
    if call.arguments().len() != callee.params().len() {
        return Err(format!(
            "inline call: `{}` has {} call operands but {} parameters",
            callee.name,
            call.arguments().len(),
            callee.params().len()
        ));
    }
    for (index, (parameter_id, parameter)) in callee.params().iter_with_ids().enumerate() {
        let argument = call.argument(parameter_id).ok_or_else(|| {
            format!(
                "inline call: `{}` has no binding for parameter {parameter_id:?}",
                callee.name
            )
        })?;
        let matches = match (&argument, parameter.representation()) {
            (OperandRef::Value(value), super::ir::OperandType::Value(ty)) => {
                caller.nodes.get(*value).is_some_and(|node| node.ty() == ty)
            }
            (OperandRef::View(view), super::ir::OperandType::View(ty)) => {
                caller.nodes.get(view.value()).is_some_and(|node| {
                    <WynLanguage as super::ir::Language>::view_argument_matches(&ty.array, node.ty())
                })
            }
            (OperandRef::Place(place), super::ir::OperandType::Place(ty)) => {
                caller.places().get(*place).is_some_and(|place| {
                    <WynLanguage as super::ir::Language>::view_argument_matches(
                        &ty.pointee,
                        &place.ty().pointee,
                    ) && ty.access.accepts(place.ty().access)
                })
            }
            _ => false,
        };
        if !matches {
            let argument_definition = argument
                .value()
                .and_then(|value| caller.nodes.get(value))
                .map(|value| (value.ty(), value.kind()));
            return Err(format!(
                "inline call: operand {index} ({argument:?}, {argument_definition:?}) of `{}` does not match parameter {:?}",
                callee.name,
                parameter.representation(),
            ));
        }
    }
    Ok(())
}
