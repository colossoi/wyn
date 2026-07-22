//! Lift stage-uniform work out of parallel semantic regions.
//!
//! Each transformed [`SegBody`](super::types::SegBody) use receives a private
//! specialized region. Mixed-stage pure calls are first exposed with the
//! context-independent inliner. Maximal stage-uniform, loop-independent pure
//! values are then cloned into the enclosing entry graph and passed back as
//! one aggregate capture. Existing scalar residency decides whether that
//! capture is profitable and legal to materialize in a singleton prepass.

use polytype::Type;
use smallvec::smallvec;

use crate::ast::TypeName;
use crate::flow::{BlockId, ControlHeader};
use crate::{LookupSet, SortedSet};

use super::graph_ops::{self, ConstantCopy};
use super::inlining;
use super::program::{SemanticFunc, SemanticProgram};
use super::stage_variance::{entry_parameter_dependences, StageDependence, StageDependenceAnalysis};
use super::types::{
    EGraph, ENode, NodeId, PureOp, RegionId, SegBody, SideEffectKind, SideEffectSite, SoacEffect,
};

#[cfg(test)]
#[path = "stage_lift_tests.rs"]
mod stage_lift_tests;

const MAX_INLINED_NODES: usize = 512;

#[derive(Debug, thiserror::Error)]
pub(crate) enum StageLiftError {
    #[error("stage-dependence analysis for `{scope}` failed: {reason}")]
    Analysis {
        scope: String,
        reason: String,
    },
    #[error("stage lifting cannot resolve region {0}")]
    MissingRegion(RegionId),
    #[error("stage lifting lost inline candidate region {0}")]
    MissingInlineRegion(RegionId),
    #[error("stage lifting lost its repeated body site")]
    MissingBodySite,
    #[error("stage-lift rewrite failed: {0}")]
    Rewrite(String),
}

impl From<String> for StageLiftError {
    fn from(error: String) -> Self {
        Self::Rewrite(error)
    }
}

type Result<T> = std::result::Result<T, StageLiftError>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StageLiftStats {
    pub(crate) bodies_specialized: usize,
    pub(crate) calls_inlined: usize,
    pub(crate) values_lifted: usize,
}

#[derive(Clone, Copy)]
struct BodySite {
    entry: usize,
    effect: SideEffectSite,
    body: usize,
}

struct StageLiftCandidate {
    function: SemanticFunc,
    original_body: SegBody,
    frontier: Vec<NodeId>,
    calls_inlined: usize,
}

pub(crate) fn run(program: &mut SemanticProgram) -> Result<StageLiftStats> {
    let mut stats = StageLiftStats::default();
    stats.calls_inlined += inline_direct_entry_calls(program)?;
    loop {
        let Some((site, prepared)) = find_next_candidate(program)? else {
            break;
        };
        let scope = program.entry_points[site.entry].name.clone();
        let name = fresh_region_name(
            program,
            &format!("{}_{}_stage_lift", scope, prepared.function.name),
        );
        let frontier_count = prepared.frontier.len();
        let calls_inlined = prepared.calls_inlined;
        let (mut function, mut captures) = {
            let graph = &mut program.entry_points[site.entry].graph;
            apply_lift(graph, prepared)?
        };
        function.name = name;
        let region = program.define_region(function);
        captures.region = region;
        let body = program
            .entry_seg_body_mut(site.entry, site.effect, site.body)
            .ok_or(StageLiftError::MissingBodySite)?;
        *body = captures;

        stats.bodies_specialized += 1;
        stats.calls_inlined += calls_inlined;
        stats.values_lifted += frontier_count;
    }
    Ok(stats)
}

/// Expose invariant subgraphs hidden inside mixed-stage calls made directly
/// by shader entries. Scalar residency subsequently decides whether those
/// exposed values are worth a separate stage invocation.
fn inline_direct_entry_calls(program: &mut SemanticProgram) -> Result<usize> {
    let mut total = 0;
    for entry_index in 0..program.entry_points.len() {
        let (mut graph, control_headers, parameter_dependences, scope) = {
            let entry = &program.entry_points[entry_index];
            (
                entry.graph.clone(),
                entry.control_headers.clone(),
                entry_parameter_dependences(entry),
                entry.name.clone(),
            )
        };
        let calls = inline_mixed_calls_in_graph(
            program,
            &mut graph,
            &control_headers,
            &parameter_dependences,
            &scope,
        )?;
        if calls != 0 {
            program.entry_points[entry_index].graph = graph;
            total += calls;
        }
    }
    Ok(total)
}

fn find_next_candidate(program: &SemanticProgram) -> Result<Option<(BodySite, StageLiftCandidate)>> {
    for (entry_index, entry) in program.entry_points.iter().enumerate() {
        let enclosing =
            StageDependenceAnalysis::for_entry(entry).map_err(|reason| StageLiftError::Analysis {
                scope: entry.name.clone(),
                reason,
            })?;
        for (block, skeleton_block) in &entry.graph.skeleton.blocks {
            for (effect_index, effect) in skeleton_block.side_effects.iter().enumerate() {
                let SideEffectKind::Soac(SoacEffect(_, soac)) = &effect.kind else {
                    continue;
                };
                if soac.scheduling_space().is_none() {
                    continue;
                }
                for (body_index, body) in soac.seg_bodies().into_iter().enumerate() {
                    let Some(prepared) = prepare_lift(program, &enclosing, body)? else {
                        continue;
                    };
                    return Ok(Some((
                        BodySite {
                            entry: entry_index,
                            effect: SideEffectSite {
                                block,
                                index: effect_index,
                            },
                            body: body_index,
                        },
                        prepared,
                    )));
                }
            }
        }
    }
    Ok(None)
}

fn prepare_lift(
    program: &SemanticProgram,
    enclosing: &StageDependenceAnalysis,
    body: &SegBody,
) -> Result<Option<StageLiftCandidate>> {
    let mut function =
        program.region(body.region).ok_or(StageLiftError::MissingRegion(body.region))?.clone();
    let parameter_dependences =
        StageDependenceAnalysis::seg_body_parameter_dependences(function.params.len(), enclosing, body)
            .map_err(|reason| StageLiftError::Analysis {
                scope: function.name.clone(),
                reason,
            })?;
    let calls_inlined = inline_mixed_calls(program, &mut function, &parameter_dependences)?;
    let analysis = StageDependenceAnalysis::for_graph(
        &function.graph,
        &function.control_headers,
        &parameter_dependences,
    )
    .map_err(|reason| StageLiftError::Analysis {
        scope: function.name.clone(),
        reason,
    })?;
    let frontier = invariant_frontier(
        &function.graph,
        &analysis,
        function.params.len(),
        body.captures.len(),
        calls_inlined != 0,
    );
    if frontier.is_empty() {
        return Ok(None);
    }
    Ok(Some(StageLiftCandidate {
        function,
        original_body: body.clone(),
        frontier,
        calls_inlined,
    }))
}

fn inline_mixed_calls(
    program: &SemanticProgram,
    function: &mut SemanticFunc,
    parameter_dependences: &[StageDependence],
) -> Result<usize> {
    inline_mixed_calls_in_graph(
        program,
        &mut function.graph,
        &function.control_headers,
        parameter_dependences,
        &function.name,
    )
}

fn inline_mixed_calls_in_graph(
    program: &SemanticProgram,
    graph: &mut EGraph,
    control_headers: &crate::LookupMap<BlockId, ControlHeader>,
    parameter_dependences: &[StageDependence],
    scope: &str,
) -> Result<usize> {
    let mut calls_inlined = 0;
    let mut node_budget = 0;
    while node_budget < MAX_INLINED_NODES {
        let analysis = StageDependenceAnalysis::for_graph(graph, control_headers, parameter_dependences)
            .map_err(|reason| StageLiftError::Analysis {
                scope: scope.to_string(),
                reason,
            })?;
        let remaining = MAX_INLINED_NODES - node_budget;
        let candidate = graph_ops::reachable_execution_values(graph).into_iter().find_map(|node| {
            let call = analysis.call_arguments(graph, node)?;
            if !call.has_mixed_stage_variance() {
                return None;
            }
            let region = program.region_interner.get(&call.callee)?;
            let callee = program.region(region)?;
            if callee.params.len() != call.arguments.len() {
                return None;
            }
            let nodes = inlining::inlineable_node_count(callee)?;
            (nodes <= remaining).then_some((node, region, nodes))
        });
        let Some((call, region, nodes)) = candidate else {
            break;
        };
        let callee = program.region(region).ok_or(StageLiftError::MissingInlineRegion(region))?;
        inlining::inline_pure_call(graph, call, callee)?;
        calls_inlined += 1;
        node_budget += nodes;
    }
    Ok(calls_inlined)
}

fn invariant_frontier(
    graph: &EGraph,
    analysis: &StageDependenceAnalysis,
    parameter_count: usize,
    capture_count: usize,
    exposed_by_mixed_call: bool,
) -> Vec<NodeId> {
    let leading = parameter_count.saturating_sub(capture_count);
    graph_ops::maximal_execution_frontier(graph, |node| is_liftable(graph, analysis, node, leading))
        .into_iter()
        .filter(|node| exposed_by_mixed_call || subgraph_contains_call(graph, *node))
        .collect()
}

fn is_liftable(
    graph: &EGraph,
    analysis: &StageDependenceAnalysis,
    node: NodeId,
    leading_parameters: usize,
) -> bool {
    if !matches!(
        graph.nodes.get(node),
        Some(ENode::Pure { op, .. })
            if !matches!(
                op,
                PureOp::Project { .. }
                    | PureOp::ViewIndex
                    | PureOp::PlaceIndex
                    | PureOp::OutputSlot { .. }
            )
    ) {
        return false;
    }
    let dependence = analysis.dependence(node);
    let Some(ty) = graph.types.get(&node) else {
        return false;
    };
    dependence.is_stage_invariant()
        && !dependence.is_compile_time_constant()
        && dependence.loop_dependencies().is_empty()
        && !crate::types::TypeExt::is_array(ty)
        && crate::ssa::layout::storage_elem_stride(ty).is_some()
        && cloneable_from_captures(graph, node, leading_parameters, &mut LookupSet::new())
}

fn subgraph_contains_call(graph: &EGraph, root: NodeId) -> bool {
    wyn_graph::reachable_from_ordered([root], wyn_graph::WalkOrder::DepthFirst, |node, out| {
        if let Some(definition) = graph.nodes.get(node) {
            out.extend(definition.children());
        }
    })
    .into_iter()
    .any(|node| {
        matches!(
            graph.nodes.get(node),
            Some(ENode::Pure {
                op: PureOp::Call(_),
                ..
            })
        )
    })
}

fn cloneable_from_captures(
    graph: &EGraph,
    node: NodeId,
    leading_parameters: usize,
    visiting: &mut LookupSet<NodeId>,
) -> bool {
    if !visiting.insert(node) {
        return true;
    }
    let cloneable = match graph.nodes.get(node) {
        Some(ENode::Constant(_)) => true,
        Some(ENode::FuncParam { index }) => *index >= leading_parameters,
        Some(ENode::Pure { operands, .. }) => operands
            .iter()
            .all(|operand| cloneable_from_captures(graph, *operand, leading_parameters, visiting)),
        Some(ENode::Union { left, right }) => {
            cloneable_from_captures(graph, *left, leading_parameters, visiting)
                && cloneable_from_captures(graph, *right, leading_parameters, visiting)
        }
        Some(ENode::BlockParam { .. } | ENode::SideEffectResult) | None => false,
    };
    visiting.remove(&node);
    cloneable
}

fn apply_lift(enclosing: &mut EGraph, mut prepared: StageLiftCandidate) -> Result<(SemanticFunc, SegBody)> {
    let mut body = prepared.original_body;
    let mut memo = body.capture_bindings(&prepared.function)?;

    let mut cloned = Vec::with_capacity(prepared.frontier.len());
    let mut types = Vec::with_capacity(prepared.frontier.len());
    for &root in &prepared.frontier {
        cloned.push(graph_ops::clone_value_subgraph(
            &prepared.function.graph,
            enclosing,
            root,
            &mut memo,
            ConstantCopy::Intern,
            true,
        )?);
        types.push(prepared.function.graph.types[&root].clone());
    }

    let (capture, capture_ty) = if cloned.len() == 1 {
        (cloned[0], types[0].clone())
    } else {
        let ty = Type::Constructed(TypeName::Tuple(types.len()), types.clone());
        let tuple = enclosing.intern_pure(PureOp::Tuple(cloned.len()), cloned.into(), ty.clone(), None);
        (tuple, ty)
    };
    let parameter = prepared.function.push_seg_body_capture(
        &mut body,
        capture,
        capture_ty,
        "stage_uniform_capture".into(),
    );

    for (index, (&root, ty)) in prepared.frontier.iter().zip(types).enumerate() {
        let replacement = if prepared.frontier.len() == 1 {
            parameter
        } else {
            prepared.function.graph.intern_pure(
                PureOp::Project { index: index as u32 },
                smallvec![parameter],
                ty,
                None,
            )
        };
        graph_ops::replace_all_references(&mut prepared.function.graph, root, replacement);
    }

    prune_dead_captures(&mut prepared.function, &mut body)?;
    Ok((prepared.function, body))
}

/// Compact only the trailing capture portion of the region ABI. Leading
/// lane/element parameters are fixed by the SOAC, even when the body ignores
/// one. Alias-bearing bodies are left unchanged because aliases may preserve
/// incidental demands outside the skeleton roots.
fn prune_dead_captures(function: &mut SemanticFunc, body: &mut SegBody) -> Result<()> {
    if !function.aliases.is_empty() {
        return Ok(());
    }
    let leading_parameters = body.leading_parameter_count(function)?;
    let parameter_count = function.params.len();
    let live = graph_ops::reachable_execution_values(&function.graph);
    let mut retained_captures = SortedSet::new();
    for node in live {
        if let Some(ENode::FuncParam { index }) = function.graph.nodes.get(node) {
            if *index >= parameter_count {
                return Err(StageLiftError::Rewrite(format!(
                    "region `{}` has out-of-range parameter {index}",
                    function.name
                )));
            }
            if *index >= leading_parameters {
                retained_captures.insert(*index - leading_parameters);
            }
        }
    }
    function.retain_seg_body_captures(body, &retained_captures)?;
    Ok(())
}

fn fresh_region_name(program: &SemanticProgram, base: &str) -> String {
    if program.region_interner.get(base).is_none() {
        return base.to_string();
    }
    for suffix in 1.. {
        let candidate = format!("{base}_{suffix}");
        if program.region_interner.get(&candidate).is_none() {
            return candidate;
        }
    }
    unreachable!()
}
