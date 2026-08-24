//! Partial-inlining policy for calls that hide profitable placement or copy
//! elimination opportunities.
//!
//! The inlining mechanism itself is context-independent; this module supplies
//! the profitability decision. Calls that pass a concrete composite array are
//! inlined so the caller can propagate that value into the callee instead of
//! transporting the complete array through a by-value function parameter.
//! Calls repeatedly evaluated by an explicit CFG loop are also inlined when
//! they mix invariant and varying operands: EGIR elaboration hoists every pure
//! node, including a `PureOp::Call`, when all operands are invariant, but a
//! mixed call hides invariant work inside the callee until it is inlined.

/// Physical EGIR after copy- and placement-sensitive partial inlining.
#[derive(Debug, Clone, Copy)]
pub enum PartiallyInlinedTag {}
pub type PartiallyInlined = super::program::Program<
    PartiallyInlinedTag,
    super::ir::ProgramFamily<
        Physical,
        interface::StorageBindingDecl,
        super::ir::RealizedOutputRoute,
        super::program::ResourceProgramData,
    >,
    super::program::PlannedGlobal,
>;

use crate::ast;
use crate::flow;
use crate::interface;
use crate::types;
use crate::types::TypeExt;
use crate::FunctionId;
use crate::LookupMap;
use wyn_base::IdSource;

use super::inlining;
use super::ir::Language;
use super::loop_analysis::{LoopAnalysis, LoopInvariance};
use super::program::Func;
use super::types::{
    EGraph, EffectOp, EffectToken, Physical, SideEffectKind, SideEffectSite, ValueId, ValueKind,
};

#[cfg(test)]
#[path = "partial_inline_tests.rs"]
mod partial_inline_tests;

/// A single inline may expose at most this many callee nodes. This is an
/// upper bound before caller-side hash-consing, so actual growth is often less.
const MAX_CALLEE_NODES: usize = 128;
/// A structured callee may add at most this many blocks, including its return
/// continuation. Ordinary DAG inlining adds zero blocks.
const MAX_CALLEE_BLOCKS: usize = 8;
/// Aggregate per-body upper bound across the fixpoint.
const MAX_INLINED_NODES: usize = 512;
/// Aggregate skeleton-growth bound across the fixpoint.
const MAX_INLINED_BLOCKS: usize = 32;
/// Independent guard against a long chain of tiny wrappers.
const MAX_INLINES: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct InliningStats {
    calls_inlined: usize,
    node_budget: usize,
    block_budget: usize,
}

#[derive(Clone, Debug)]
struct Candidate {
    call: ValueId,
    block: flow::BlockId,
    callee: FunctionId,
    callee_nodes: usize,
    callee_blocks: usize,
}

#[derive(Clone, Copy, Debug)]
struct EffectfulCandidate {
    effect: SideEffectSite,
    callee: FunctionId,
}

/// Inline profitable mixed-variance calls in every physical body. The ordinary
/// scoped elaborator then performs CSE and LICM on the exposed DAG.
pub fn partially_inline_calls(
    program: super::soac_expand::SoacsExpanded,
) -> Result<PartiallyInlined, String> {
    // Snapshot callable bodies so callers can be rewritten without aliasing
    // `program.functions`. A caller-local fixpoint handles calls revealed by a
    // clone, so snapshots do not need to be refreshed after each body.
    let callees: LookupMap<FunctionId, Func<Physical>> =
        program.functions.iter().map(|function| (function.region, function.clone())).collect();
    program
        .try_map_graphs_with_state(|site, mut graph, _, context| {
            inline_body(&mut graph, &callees, &mut context.effect_ids)
                .map_err(|error| format!("partial inlining in {site:?} failed: {error}"))?;
            Ok(graph)
        })
        .map(|program| program.retag())
}

fn inline_body(
    graph: &mut EGraph<Physical>,
    callees: &LookupMap<FunctionId, Func<Physical>>,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<InliningStats, String> {
    let mut stats = InliningStats::default();
    let mut effectful_inlines = 0usize;
    while let Some(candidate) = find_effectful_candidate(graph, callees) {
        if effectful_inlines == 1024 {
            return Err("effectful call inlining exceeded the acyclic call-graph bound".into());
        }
        let callee = &callees[&candidate.callee];
        let cost = inlining::inline_effectful_call(graph, candidate.effect, callee, effect_ids)
            .map_err(|error| format!("while inlining `{}`: {error}", callee.name))?;
        effectful_inlines += 1;
        stats.calls_inlined += 1;
        stats.node_budget += cost.nodes;
        stats.block_budget += cost.blocks;
    }
    while stats.calls_inlined < MAX_INLINES
        && stats.node_budget < MAX_INLINED_NODES
        && stats.block_budget < MAX_INLINED_BLOCKS
    {
        let remaining_nodes = MAX_INLINED_NODES - stats.node_budget;
        let remaining_blocks = MAX_INLINED_BLOCKS - stats.block_budget;
        let Some(candidate) = find_candidate(graph, callees, remaining_nodes, remaining_blocks) else {
            break;
        };
        let callee = &callees[&candidate.callee];
        inlining::inline_call_at_block(graph, candidate.call, candidate.block, callee)
            .map_err(|error| format!("while inlining `{}`: {error}", callee.name))?;
        stats.calls_inlined += 1;
        stats.node_budget += candidate.callee_nodes;
        stats.block_budget += candidate.callee_blocks;
    }
    Ok(stats)
}

fn find_effectful_candidate(
    graph: &EGraph<Physical>,
    callees: &LookupMap<FunctionId, Func<Physical>>,
) -> Option<EffectfulCandidate> {
    for (block, body) in &graph.skeleton.blocks {
        for (index, effect) in body.side_effects.iter().enumerate() {
            let SideEffectKind::Effect(EffectOp::Call { site }) = effect.kind() else {
                continue;
            };
            let call = graph.call(*site);
            if call.effects() == super::types::CallEffects::Pure
                && call.arguments().all(|argument| argument.place().is_none())
                && call.result().places().is_empty()
            {
                continue;
            }
            let callee_id = call.callee();
            if !callees.contains_key(&callee_id) {
                continue;
            }
            let effect = SideEffectSite { block, index };
            return Some(EffectfulCandidate {
                effect,
                callee: callee_id,
            });
        }
    }
    None
}

fn find_candidate(
    graph: &EGraph<Physical>,
    callees: &LookupMap<FunctionId, Func<Physical>>,
    remaining_nodes: usize,
    remaining_blocks: usize,
) -> Option<Candidate> {
    // A composite fixed array is an SSA value, so leaving it behind a call
    // boundary passes the complete aggregate by value. Map kernels make this
    // especially costly: every lane calls the helper, which commonly
    // materializes the parameter again for a dynamic index. Inline these
    // calls even without an explicit CFG loop so substitution exposes the
    // original array directly to placement and copy elimination.
    for (block_id, block) in &graph.skeleton.blocks {
        let roots = block
            .side_effects
            .iter()
            .flat_map(|effect| graph.effect_boundary_value_dependencies(effect))
            .chain(block.term.referenced_nodes());
        let reachable =
            wyn_graph::reachable_from_ordered(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
                out.extend(graph.value_dependencies(node));
            });
        for node in reachable {
            let ValueKind::CallResult { call, .. } = &graph.nodes[node].kind else {
                continue;
            };
            let boundary = graph.call(*call);
            let callee_name = boundary.callee();
            let operands = boundary.arguments();
            let Some(callee) = callees.get(&callee_name) else {
                continue;
            };
            if operands.len() != callee.params().len()
                || !callee.params().iter().any(|parameter| is_fixed_composite_array(parameter.ty()))
            {
                continue;
            }
            let Some(cost) = inlining::inlineable_call_cost_at_block(graph, node, block_id, callee) else {
                continue;
            };
            if cost.nodes <= MAX_CALLEE_NODES
                && cost.nodes <= remaining_nodes
                && cost.blocks <= MAX_CALLEE_BLOCKS
                && cost.blocks <= remaining_blocks
            {
                return Some(Candidate {
                    call: node,
                    block: block_id,
                    callee: callee_name,
                    callee_nodes: cost.nodes,
                    callee_blocks: cost.blocks,
                });
            }
        }
    }

    let loops = LoopAnalysis::build(&graph.skeleton);

    // Iterate in skeleton order for deterministic code growth. Recompute after
    // every inline: the clone can reveal another mixed call, or can make an
    // older candidate unreachable through subsumption.
    for (header, _) in &graph.skeleton.blocks {
        if !loops.is_header(header) {
            continue;
        }
        let mut invariance = LoopInvariance::new(graph, &loops, header);
        for (block_id, block) in &graph.skeleton.blocks {
            if !loops.is_in_loop(block_id, header) {
                continue;
            }
            let roots = block
                .side_effects
                .iter()
                .flat_map(|effect| graph.effect_boundary_value_dependencies(effect))
                .chain(block.term.referenced_nodes());
            let reachable =
                wyn_graph::reachable_from_ordered(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
                    out.extend(graph.value_dependencies(node));
                });
            for node in reachable {
                let ValueKind::CallResult { call, .. } = &graph.nodes[node].kind else {
                    continue;
                };
                let boundary = graph.call(*call);
                let callee_name = boundary.callee();
                let operands = boundary.arguments();
                let Some(callee) = callees.get(&callee_name) else {
                    continue;
                };
                if operands.len() != callee.params().len() {
                    continue;
                }
                let invariant_args = operands
                    .map(|operand| operand.value().map(|value| invariance.is_invariant(value)))
                    .collect::<Option<Vec<_>>>();
                let Some(invariant_args) = invariant_args else {
                    continue;
                };
                if !invariant_args.iter().any(|value| *value) || !invariant_args.iter().any(|value| !*value)
                {
                    continue;
                }
                let Some(cost) = inlining::inlineable_call_cost_at_block(graph, node, block_id, callee)
                else {
                    continue;
                };
                if cost.nodes > MAX_CALLEE_NODES
                    || cost.nodes > remaining_nodes
                    || cost.blocks > MAX_CALLEE_BLOCKS
                    || cost.blocks > remaining_blocks
                {
                    continue;
                }
                return Some(Candidate {
                    call: node,
                    block: block_id,
                    callee: callee_name,
                    callee_nodes: cost.nodes,
                    callee_blocks: cost.blocks,
                });
            }
        }
    }
    None
}

fn is_fixed_composite_array(ty: &polytype::Type<ast::TypeName>) -> bool {
    (ty.array_variant().is_some_and(types::is_array_variant_composite)
        && matches!(
            ty.array_size(),
            Some(polytype::Type::Constructed(ast::TypeName::Size(_), _))
        ))
        || super::types::WynLanguage::product_fields(ty)
            .is_some_and(|fields| fields.iter().any(is_fixed_composite_array))
}
