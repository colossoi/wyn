//! Lift stage-uniform work out of parallel semantic regions.
//!
//! Each transformed [`SegBody`](super::types::SegBody) use receives a private
//! specialized region. Mixed-stage pure calls are first exposed with the
//! context-independent inliner. Maximal stage-uniform, loop-independent pure
//! values are then cloned into the enclosing entry graph and passed back as
//! one aggregate capture. Existing scalar residency decides whether that
//! capture is profitable and legal to materialize in a singleton prepass.

use crate::ssa;
use crate::types;
use polytype::Type;
use smallvec::smallvec;

use crate::ast::TypeName;
use crate::{FunctionId, LookupMap, LookupSet, SortedSet};

use super::graph_ops::{self, ConstantCopy, PureCopy};
use super::inlining;
use super::ir::{Body, BodySite as ProgramBodySite};
use super::program::{fresh_region_name, Func, SemanticProgramData};
use super::reify::Segmented;
use super::stage_variance::{entry_parameter_dependences, StageDependence, StageDependenceAnalysis};
use super::types::{
    EGraph, PureOp, SegBody, Semantic, SideEffectKind, SideEffectSite, SoacEffect, ValueId, ValueKind,
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
    MissingRegion(FunctionId),
    #[error("stage lifting lost inline candidate region {0}")]
    MissingInlineRegion(FunctionId),
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
struct SegBodySite {
    entry: usize,
    effect: SideEffectSite,
    body: usize,
}

struct StageLiftCandidate {
    function: Func<Semantic>,
    original_body: SegBody,
    frontier: Vec<ValueId>,
    calls_inlined: usize,
}

pub(crate) fn lift_stage_uniform_values(program: Segmented) -> Result<Segmented> {
    run_with_stats(program).map(|(program, _)| program)
}

fn run_with_stats(mut program: Segmented) -> Result<(Segmented, StageLiftStats)> {
    let mut stats = StageLiftStats::default();
    while let Some(patch) = analyze_direct_entry_calls(&program)? {
        stats.calls_inlined += patch.calls_inlined;
        program = apply_direct_entry_calls(program, patch);
    }
    loop {
        let Some((site, prepared)) = find_next_candidate(&program)? else {
            break;
        };
        let scope = program.entry_points[site.entry].name.clone();
        let mut identities = program.data.identities.clone();
        let name = fresh_region_name(
            &identities,
            &format!("{}_{}_stage_lift", scope, prepared.function.name),
        );
        let region = identities.alloc_function(name.clone());
        let frontier_count = prepared.frontier.len();
        let calls_inlined = prepared.calls_inlined;
        let mut specialized = None;
        program = program.try_rewrite_body(ProgramBodySite::Entry(site.entry), |body| match body {
            Body::Entry(mut entry) => {
                let (mut function, mut captures) = apply_lift(&mut entry.graph, prepared)?;
                function.region = region;
                function.name = name;
                captures.region = region;
                let body = entry
                    .graph
                    .skeleton
                    .get_effect_mut(site.effect)
                    .and_then(|effect| effect.seg_body_mut(site.body))
                    .ok_or(StageLiftError::MissingBodySite)?;
                *body = captures;
                specialized = Some(function);
                Ok::<_, StageLiftError>(Body::Entry(entry))
            }
            _ => unreachable!("stage lifting only targets entry points"),
        })?;
        program = program
            .extend_functions([specialized.expect("stage-lift rewrite did not produce its region")])
            .map_data(|data| SemanticProgramData { identities, ..data });

        stats.bodies_specialized += 1;
        stats.calls_inlined += calls_inlined;
        stats.values_lifted += frontier_count;
    }
    Ok((program, stats))
}

/// Expose invariant subgraphs hidden inside mixed-stage calls made directly
/// by shader entries. Scalar residency subsequently decides whether those
/// exposed values are worth a separate stage invocation.
struct DirectEntryCallsPatch {
    entry: usize,
    graph: EGraph,
    calls_inlined: usize,
}

fn analyze_direct_entry_calls(program: &Segmented) -> Result<Option<DirectEntryCallsPatch>> {
    for (entry_index, entry) in program.entry_points.iter().enumerate() {
        let mut graph = entry.graph.clone();
        let calls = inline_mixed_calls_in_graph(
            program,
            &mut graph,
            &super::stage_variance::bind_parameter_dependences(
                &entry.params,
                &entry_parameter_dependences(entry),
            ),
            &entry.name,
        )?;
        if calls != 0 {
            return Ok(Some(DirectEntryCallsPatch {
                entry: entry_index,
                graph,
                calls_inlined: calls,
            }));
        }
    }
    Ok(None)
}

fn apply_direct_entry_calls(program: Segmented, patch: DirectEntryCallsPatch) -> Segmented {
    program.rewrite_body(ProgramBodySite::Entry(patch.entry), |body| match body {
        Body::Entry(mut entry) => {
            entry.graph = patch.graph;
            Body::Entry(entry)
        }
        _ => unreachable!("direct entry-call patch targeted a non-entry body"),
    })
}

fn find_next_candidate(program: &Segmented) -> Result<Option<(SegBodySite, StageLiftCandidate)>> {
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
                        SegBodySite {
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
    program: &Segmented,
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
    let bound_dependences =
        super::stage_variance::bind_parameter_dependences(&function.params, &parameter_dependences);
    let analysis =
        StageDependenceAnalysis::for_graph(&function.graph, &bound_dependences).map_err(|reason| {
            StageLiftError::Analysis {
                scope: function.name.clone(),
                reason,
            }
        })?;
    let leading = function.params.len().saturating_sub(body.captures.len());
    let capture_parameters = function.params.ids().skip(leading).collect::<LookupSet<_>>();
    let frontier = invariant_frontier(
        &function.graph,
        &analysis,
        &capture_parameters,
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
    program: &Segmented,
    function: &mut Func<Semantic>,
    parameter_dependences: &[StageDependence],
) -> Result<usize> {
    inline_mixed_calls_in_graph(
        program,
        &mut function.graph,
        &super::stage_variance::bind_parameter_dependences(&function.params, parameter_dependences),
        &function.name,
    )
}

fn inline_mixed_calls_in_graph(
    program: &Segmented,
    graph: &mut EGraph,
    parameter_dependences: &LookupMap<super::types::ParameterId, StageDependence>,
    scope: &str,
) -> Result<usize> {
    let mut calls_inlined = 0;
    let mut node_budget = 0;
    while node_budget < MAX_INLINED_NODES {
        let analysis =
            StageDependenceAnalysis::for_graph(graph, parameter_dependences).map_err(|reason| {
                StageLiftError::Analysis {
                    scope: scope.to_string(),
                    reason,
                }
            })?;
        let remaining = MAX_INLINED_NODES - node_budget;
        let candidate = graph_ops::reachable_execution_values(graph).into_iter().find_map(|node| {
            let call = analysis.call_arguments(graph, node)?;
            if !call.has_mixed_stage_variance() {
                return None;
            }
            let region = call.callee;
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
    capture_parameters: &LookupSet<super::types::ParameterId>,
    exposed_by_mixed_call: bool,
) -> Vec<ValueId> {
    graph_ops::maximal_execution_frontier(graph, |node| {
        is_liftable(graph, analysis, node, capture_parameters)
    })
    .into_iter()
    .filter(|node| exposed_by_mixed_call || subgraph_contains_call(graph, *node))
    .collect()
}

fn is_liftable(
    graph: &EGraph,
    analysis: &StageDependenceAnalysis,
    node: ValueId,
    capture_parameters: &LookupSet<super::types::ParameterId>,
) -> bool {
    if !matches!(
        graph.nodes.get(node).map(|node| &node.kind),
        Some(ValueKind::Pure { op, .. }) if !matches!(op, PureOp::Project { .. })
    ) {
        return false;
    }
    let dependence = analysis.dependence(node);
    let Some(ty) = graph.nodes.get(node).map(|node| &node.ty) else {
        return false;
    };
    dependence.is_stage_invariant()
        && !dependence.is_compile_time_constant()
        && dependence.loop_dependencies().is_empty()
        && !types::TypeExt::is_array(ty)
        && ssa::layout::storage_elem_stride(ty).is_some()
        && cloneable_from_captures(graph, node, capture_parameters, &mut LookupSet::new())
}

fn subgraph_contains_call(graph: &EGraph, root: ValueId) -> bool {
    wyn_graph::reachable_from_ordered([root], wyn_graph::WalkOrder::DepthFirst, |node, out| {
        if let Some(definition) = graph.nodes.get(node) {
            out.extend(definition.kind.children());
        }
    })
    .into_iter()
    .any(|node| {
        matches!(
            graph.nodes.get(node).map(|node| &node.kind),
            Some(ValueKind::CallResult { .. })
        )
    })
}

fn cloneable_from_captures(
    graph: &EGraph,
    node: ValueId,
    capture_parameters: &LookupSet<super::types::ParameterId>,
    visiting: &mut LookupSet<ValueId>,
) -> bool {
    if !visiting.insert(node) {
        return true;
    }
    let cloneable = match graph.nodes.get(node).map(|node| &node.kind) {
        Some(ValueKind::Constant(_)) => true,
        Some(ValueKind::FuncParam { parameter }) => capture_parameters.contains(parameter),
        Some(ValueKind::Pure { operands, .. }) => operands
            .iter()
            .all(|operand| cloneable_from_captures(graph, *operand, capture_parameters, visiting)),
        Some(ValueKind::Union { left, right }) => {
            cloneable_from_captures(graph, *left, capture_parameters, visiting)
                && cloneable_from_captures(graph, *right, capture_parameters, visiting)
        }
        Some(
            ValueKind::BlockParam { .. }
            | ValueKind::CallResult { .. }
            | ValueKind::PlaceLength { .. }
            | ValueKind::PlaceView { .. }
            | ValueKind::SideEffectResult,
        )
        | None => false,
    };
    visiting.remove(&node);
    cloneable
}

fn apply_lift(
    enclosing: &mut EGraph,
    mut prepared: StageLiftCandidate,
) -> Result<(Func<Semantic>, SegBody)> {
    let mut body = prepared.original_body;
    let mut memo = body
        .capture_bindings(&prepared.function)?
        .into_iter()
        .filter_map(|(parameter, capture)| capture.value().map(|capture| (parameter, capture)))
        .collect();

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
            PureCopy::Preserve,
        )?);
        types.push(prepared.function.graph.nodes[root].ty.clone());
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
        enclosing.operand_ref(capture),
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
        prepared.function.graph.replace_value_references(root, replacement);
    }

    prune_dead_captures(&mut prepared.function, &mut body)?;
    Ok((prepared.function, body))
}

/// Compact only the trailing capture portion of the region ABI. Leading
/// lane/element parameters are fixed by the SOAC, even when the body ignores
/// one.
fn prune_dead_captures(function: &mut Func<Semantic>, body: &mut SegBody) -> Result<()> {
    let leading_parameters = body.leading_parameter_count(function)?;
    let live = graph_ops::reachable_execution_values(&function.graph);
    let mut retained_captures = SortedSet::new();
    for node in live {
        if let Some(ValueKind::FuncParam { parameter }) =
            function.graph.nodes.get(node).map(|node| &node.kind)
        {
            let Some(position) = function.params().abi_position(*parameter) else {
                return Err(StageLiftError::Rewrite(format!(
                    "region `{}` has undeclared parameter {parameter:?}",
                    function.name
                )));
            };
            if position >= leading_parameters {
                retained_captures.insert(position - leading_parameters);
            }
        }
    }
    function.retain_seg_body_captures(body, &retained_captures)?;
    Ok(())
}
