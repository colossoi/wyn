//! Required lowering for call boundaries that cannot be represented by the
//! SSA call channel.
//!
//! Physical call ABI selection may route arguments or results through EGIR
//! places. Internal calls with those routes are expanded to a fixpoint. This
//! is a correctness transition, not a profitability decision: its accounting
//! is deliberately independent from bounded partial inlining.

use crate::{FunctionId, LookupMap};
use wyn_base::IdSource;

use super::inlining;
use super::program::Func;
use super::types::{EGraph, EffectOp, EffectToken, Physical, SideEffectKind, SideEffectSite};

/// Physical EGIR whose remaining calls can all use the SSA value channel.
#[derive(Debug, Clone, Copy)]
pub enum CallsPlaceFreeTag {}
pub type CallsPlaceFree = super::program::PhysicalProgram<CallsPlaceFreeTag>;

/// Mandatory expansion has no profitability limit. This guard only turns a
/// recursive or otherwise non-converging internal call graph into a diagnostic.
const MAX_REQUIRED_INLINES: usize = 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RequiredCallInliningStats {
    pub calls: usize,
    pub nodes: usize,
    pub blocks: usize,
}

/// Per-body accounting for mandatory place-call lowering. These counters do
/// not participate in any optional inlining budget.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RequiredCallInliningTrace {
    pub bodies: Vec<(super::ir::BodySite, RequiredCallInliningStats)>,
}

#[derive(Clone, Copy, Debug)]
struct PlaceCallCandidate {
    effect: SideEffectSite,
    callee: FunctionId,
}

/// Eliminate internal calls with place arguments or destination-passed
/// results, then prove that every remaining call is directly SSA-callable.
pub fn eliminate_internal_place_calls(
    program: super::soac_expand::SoacsExpanded,
) -> Result<CallsPlaceFree, String> {
    eliminate_internal_place_calls_with_trace(program).map(|(program, _)| program)
}

/// Required call lowering with separate per-body accounting for diagnostics.
pub fn eliminate_internal_place_calls_with_trace(
    program: super::soac_expand::SoacsExpanded,
) -> Result<(CallsPlaceFree, RequiredCallInliningTrace), String> {
    let callees: LookupMap<FunctionId, Func<Physical>> =
        program.functions.iter().map(|function| (function.region, function.clone())).collect();
    let mut trace = RequiredCallInliningTrace::default();
    let program = program
        .try_map_graphs_with_state(|site, mut graph, _, context| {
            let stats = eliminate_body(&mut graph, &callees, &mut context.effect_ids)
                .map_err(|error| format!("place-call elimination in {site:?} failed: {error}"))?;
            trace.bodies.push((site, stats));
            Ok::<_, String>(graph)
        })?
        .retag_physical();
    verify_ssa_lowerable_calls(&program)?;
    Ok((program, trace))
}

fn eliminate_body(
    graph: &mut EGraph<Physical>,
    callees: &LookupMap<FunctionId, Func<Physical>>,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<RequiredCallInliningStats, String> {
    let mut stats = RequiredCallInliningStats::default();
    while let Some(candidate) = find_place_call_candidate(graph, callees) {
        if stats.calls == MAX_REQUIRED_INLINES {
            return Err("place-call inlining exceeded the acyclic call-graph bound".into());
        }
        let callee = &callees[&candidate.callee];
        let cost = inlining::inline_effectful_call(graph, candidate.effect, callee, effect_ids)
            .map_err(|error| format!("while inlining `{}`: {error}", callee.name))?;
        stats.calls += 1;
        stats.nodes += cost.nodes;
        stats.blocks += cost.blocks;
    }
    Ok(stats)
}

fn find_place_call_candidate(
    graph: &EGraph<Physical>,
    callees: &LookupMap<FunctionId, Func<Physical>>,
) -> Option<PlaceCallCandidate> {
    for (block, body) in &graph.skeleton.blocks {
        for (index, effect) in body.side_effects.iter().enumerate() {
            let SideEffectKind::Effect(EffectOp::Call { site }) = effect.kind() else {
                continue;
            };
            let call = graph.call(*site);
            if call.arguments().all(|argument| argument.value().is_some())
                && call.result().places().is_empty()
            {
                continue;
            }
            let callee = call.callee();
            if !callees.contains_key(&callee) {
                continue;
            }
            return Some(PlaceCallCandidate {
                effect: SideEffectSite { block, index },
                callee,
            });
        }
    }
    None
}

/// Verify the call-boundary postcondition consumed by SSA elaboration.
///
/// The canonical physical ABI verifier owns argument ordering, types, result
/// trees, callee identity, and effect metadata. This verifier adds the exact
/// value-channel and explicit-placement requirements of `elaborate_call`.
pub fn verify_ssa_lowerable_calls<Tag>(
    program: &super::program::PhysicalProgram<Tag>,
) -> Result<(), String> {
    let boundaries = super::physical_call_abi::callable_boundaries(&program.functions, &program.externs);
    for function in &program.functions {
        verify_graph_calls(&function.graph, &function.name, &boundaries)?;
    }
    for entry in &program.entry_points {
        verify_graph_calls(&entry.graph, &entry.name, &boundaries)?;
    }
    for constant in &program.constants {
        verify_graph_calls(&constant.graph, &constant.name, &boundaries)?;
    }
    Ok(())
}

fn verify_graph_calls(
    graph: &EGraph<Physical>,
    owner: &str,
    boundaries: &LookupMap<FunctionId, super::physical_call_abi::CallableBoundary>,
) -> Result<(), String> {
    for (site, placement) in graph.side_effect_index().calls() {
        let call = graph.call(site);
        let boundary = boundaries.get(&call.callee()).ok_or_else(|| {
            format!(
                "physical body `{owner}` calls {:?} without a stable callable ABI",
                call.callee()
            )
        })?;
        graph
            .verify_call_boundary(site, &boundary.0, &boundary.1, boundary.2)
            .map_err(|error| format!("physical body `{owner}`: {error}"))?;

        if let Some((index, _)) =
            call.arguments().enumerate().find(|(_, argument)| argument.value().is_none())
        {
            return Err(format!(
                "physical body `{owner}` has non-SSA argument {index} at call {site:?} in {:?}",
                placement.block
            ));
        }
        if !call.result().places().is_empty() {
            return Err(format!(
                "physical body `{owner}` has a place destination at call {site:?} in {:?}",
                placement.block
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "eliminate_call_places_tests.rs"]
mod tests;
