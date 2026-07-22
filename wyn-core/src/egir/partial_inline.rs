//! Partial-inlining policy for calls in repeated CFG regions.
//!
//! The inlining mechanism itself is context-independent; this module supplies
//! only the profitability decision for calls repeatedly evaluated by an
//! explicit CFG loop. EGIR elaboration already hoists every pure node,
//! including a `PureOp::Call`, when all of its operands are loop-invariant. A
//! mixed-variance call is opaque, however, so invariant work inside the callee
//! cannot reach the preheader until the call is inlined.

use crate::flow::{BlockId, ControlHeader};
use crate::LookupMap;

use super::inlining;
use super::ir::RegionId;
use super::loop_analysis::{LoopAnalysis, LoopInvariance};
use super::program::{PhysicalFunc, PhysicalProgram, Program, RegionInterner};
use super::types::{EGraph, ENode, NodeId, Physical, PureOp};

#[cfg(test)]
#[path = "partial_inline_tests.rs"]
mod partial_inline_tests;

/// A single inline may expose at most this many callee nodes. This is an
/// upper bound before caller-side hash-consing, so actual growth is often less.
const MAX_CALLEE_NODES: usize = 128;
/// Aggregate per-body upper bound across the fixpoint.
const MAX_INLINED_NODES: usize = 512;
/// Independent guard against a long chain of tiny wrappers.
const MAX_INLINES: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct InliningStats {
    calls_inlined: usize,
    node_budget: usize,
}

#[derive(Clone, Debug)]
struct Candidate {
    call: NodeId,
    callee: RegionId,
    callee_nodes: usize,
}

/// Inline profitable mixed-variance calls in every physical function and
/// entry. The ordinary scoped elaborator then performs CSE and LICM on the
/// exposed DAG.
pub fn run(program: &mut PhysicalProgram) -> Result<(), String> {
    let program: &mut Program<Physical> = program;
    // Snapshot callable bodies so callers can be rewritten without aliasing
    // `program.functions`. A caller-local fixpoint handles calls revealed by a
    // clone, so snapshots do not need to be refreshed after each body.
    let callees: LookupMap<RegionId, PhysicalFunc> =
        program.iter_regions().map(|(id, function)| (id, function.clone())).collect();
    let region_interner = &program.region_interner;

    for function in &mut program.functions {
        inline_body(
            &mut function.graph,
            &function.control_headers,
            region_interner,
            &callees,
        )
        .map_err(|error| format!("partial inlining in function `{}` failed: {error}", function.name))?;
    }
    for entry in &mut program.entry_points {
        inline_body(
            &mut entry.graph,
            &entry.control_headers,
            region_interner,
            &callees,
        )
        .map_err(|error| format!("partial inlining in entry `{}` failed: {error}", entry.name))?;
    }
    Ok(())
}

fn inline_body(
    graph: &mut EGraph<Physical>,
    control_headers: &LookupMap<BlockId, ControlHeader>,
    region_interner: &RegionInterner,
    callees: &LookupMap<RegionId, PhysicalFunc>,
) -> Result<InliningStats, String> {
    let mut stats = InliningStats::default();
    while stats.calls_inlined < MAX_INLINES && stats.node_budget < MAX_INLINED_NODES {
        let remaining = MAX_INLINED_NODES - stats.node_budget;
        let Some(candidate) = find_candidate(graph, control_headers, region_interner, callees, remaining)
        else {
            break;
        };
        let callee = &callees[&candidate.callee];
        inlining::inline_pure_call(graph, candidate.call, callee)?;
        stats.calls_inlined += 1;
        stats.node_budget += candidate.callee_nodes;
    }
    Ok(stats)
}

fn find_candidate(
    graph: &EGraph<Physical>,
    control_headers: &LookupMap<BlockId, ControlHeader>,
    region_interner: &RegionInterner,
    callees: &LookupMap<RegionId, PhysicalFunc>,
    remaining_budget: usize,
) -> Option<Candidate> {
    let loops = LoopAnalysis::build(&graph.skeleton, control_headers);

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
                let ENode::Pure {
                    op: PureOp::Call(callee_name),
                    operands,
                } = &graph.nodes[node]
                else {
                    continue;
                };
                let Some(callee_id) = region_interner.get(callee_name) else {
                    continue;
                };
                let Some(callee) = callees.get(&callee_id) else {
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
                let Some(callee_nodes) = inlining::inlineable_node_count(callee) else {
                    continue;
                };
                if callee_nodes > MAX_CALLEE_NODES || callee_nodes > remaining_budget {
                    continue;
                }
                return Some(Candidate {
                    call: node,
                    callee: callee_id,
                    callee_nodes,
                });
            }
        }
    }
    None
}
