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
        crate::interface::StorageBindingDecl,
        super::ir::RealizedOutputRoute,
        super::program::CoreProgramData,
    >,
    super::program::PlannedGlobal,
>;

use crate::types::TypeExt;
use crate::LookupMap;

use super::inlining;
use super::ir::{EffectOp, SideEffectKind};
use super::loop_analysis::{LoopAnalysis, LoopInvariance};
use super::program::PhysicalFunc;
use super::types::{EGraph, Physical, PureOp, ValueId, ValueKind};

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
    block: crate::flow::BlockId,
    callee: crate::FunctionId,
    callee_nodes: usize,
    callee_blocks: usize,
}

/// Inline profitable mixed-variance calls in every physical body. The ordinary
/// scoped elaborator then performs CSE and LICM on the exposed DAG.
pub fn partially_inline_calls(
    program: super::materialize::Materialized,
) -> Result<PartiallyInlined, String> {
    // Snapshot callable bodies so callers can be rewritten without aliasing
    // `program.functions`. A caller-local fixpoint handles calls revealed by a
    // clone, so snapshots do not need to be refreshed after each body.
    let callees: LookupMap<crate::FunctionId, PhysicalFunc> =
        program.functions.iter().map(|function| (function.region, function.clone())).collect();
    let callees = prepare_fixed_array_callees(callees)?;
    program
        .try_map_graphs(|site, mut graph| {
            inline_prerequisite_effect_calls(&mut graph, &callees, MAX_INLINED_NODES)
                .map_err(|error| format!("effect-call inlining in {site:?} failed: {error}"))?;
            inline_body(&mut graph, &callees)
                .map_err(|error| format!("partial inlining in {site:?} failed: {error}"))?;
            Ok(graph)
        })
        .map(|program| program.retag())
}

/// Resource erasure conservatively anchors a call that carries a storage view
/// in the effect skeleton. If that call targets a pure indexing DAG, inline it
/// first so a surrounding helper with a by-value fixed-array parameter can
/// itself become an ordinary pure inlining candidate.
fn prepare_fixed_array_callees(
    mut callees: LookupMap<crate::FunctionId, PhysicalFunc>,
) -> Result<LookupMap<crate::FunctionId, PhysicalFunc>, String> {
    for _ in 0..MAX_INLINES {
        let snapshot = callees.clone();
        let mut changed = false;
        for callee in callees.values_mut() {
            changed |=
                inline_prerequisite_effect_calls(&mut callee.graph, &snapshot, MAX_INLINED_NODES)? > 0;
        }
        if !changed {
            return Ok(callees);
        }
    }
    Ok(callees)
}

fn inline_prerequisite_effect_calls(
    graph: &mut EGraph<Physical>,
    callees: &LookupMap<crate::FunctionId, PhysicalFunc>,
    node_budget: usize,
) -> Result<usize, String> {
    if !graph
        .nodes
        .values()
        .any(|node| matches!(node.kind, ValueKind::FuncParam { .. }) && is_fixed_composite_array(&node.ty))
    {
        return Ok(0);
    }

    let mut inlined_nodes = 0;
    loop {
        let candidate = graph.skeleton.blocks.iter().find_map(|(block, body)| {
            body.side_effects.iter().enumerate().find_map(|(index, effect)| {
                let SideEffectKind::Effect(EffectOp::Op {
                    tag: PureOp::Call(callee),
                }) = &effect.kind
                else {
                    return None;
                };
                let result = effect.result?;
                let target = callees.get(callee)?;
                let nodes = inlining::inlineable_node_count(target)?;
                (nodes <= MAX_CALLEE_NODES && nodes <= node_budget - inlined_nodes)
                    .then(|| (block, index, *callee, effect.operand_nodes.clone(), result, nodes))
            })
        });
        let Some((block, index, callee, operands, result, nodes)) = candidate else {
            break;
        };
        inlining::inline_effect_call_to_pure_callee(graph, result, &operands, &callees[&callee])?;
        let removed_effects = graph.skeleton.blocks[block].side_effects[index].effects;
        graph.skeleton.blocks[block].side_effects.remove(index);
        if let Some((input, output)) = removed_effects {
            for effect in &mut graph.skeleton.blocks[block].side_effects[index..] {
                if let Some((effect_input, _)) = &mut effect.effects {
                    if *effect_input == output {
                        *effect_input = input;
                        break;
                    }
                }
            }
        }
        inlined_nodes += nodes;
        if inlined_nodes == node_budget {
            break;
        }
    }
    Ok(inlined_nodes)
}

fn inline_body(
    graph: &mut EGraph<Physical>,
    callees: &LookupMap<crate::FunctionId, PhysicalFunc>,
) -> Result<InliningStats, String> {
    let mut stats = InliningStats::default();
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
        inlining::inline_call_at_block(graph, candidate.call, candidate.block, callee)?;
        stats.calls_inlined += 1;
        stats.node_budget += candidate.callee_nodes;
        stats.block_budget += candidate.callee_blocks;
    }
    Ok(stats)
}

fn find_candidate(
    graph: &EGraph<Physical>,
    callees: &LookupMap<crate::FunctionId, PhysicalFunc>,
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
            .flat_map(|effect| effect.operand_nodes.iter().copied())
            .chain(block.term.referenced_nodes());
        let reachable =
            wyn_graph::reachable_from_ordered(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
                if let Some(definition) = graph.nodes.get(node) {
                    out.extend(definition.children());
                }
            });
        for node in reachable {
            let ValueKind::Pure {
                op: PureOp::Call(callee_name),
                operands,
            } = &graph.nodes[node].kind
            else {
                continue;
            };
            let Some(callee) = callees.get(callee_name) else {
                continue;
            };
            if operands.len() != callee.params.len()
                || !callee.params.iter().any(|(ty, _)| is_fixed_composite_array(ty))
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
                    callee: *callee_name,
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
                .flat_map(|effect| effect.operand_nodes.iter().copied())
                .chain(block.term.referenced_nodes());
            let reachable =
                wyn_graph::reachable_from_ordered(roots, wyn_graph::WalkOrder::DepthFirst, |node, out| {
                    if let Some(definition) = graph.nodes.get(node) {
                        out.extend(definition.children());
                    }
                });
            for node in reachable {
                let ValueKind::Pure {
                    op: PureOp::Call(callee_name),
                    operands,
                } = &graph.nodes[node].kind
                else {
                    continue;
                };
                let Some(callee) = callees.get(callee_name) else {
                    continue;
                };
                if operands.len() != callee.params.len() {
                    continue;
                }
                let invariant_args =
                    operands.iter().map(|operand| invariance.is_invariant(*operand)).collect::<Vec<_>>();
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
                    callee: *callee_name,
                    callee_nodes: cost.nodes,
                    callee_blocks: cost.blocks,
                });
            }
        }
    }
    None
}

fn is_fixed_composite_array(ty: &polytype::Type<crate::ast::TypeName>) -> bool {
    ty.array_variant().is_some_and(crate::types::is_array_variant_composite)
        && matches!(
            ty.array_size(),
            Some(polytype::Type::Constructed(crate::ast::TypeName::Size(_), _))
        )
}
