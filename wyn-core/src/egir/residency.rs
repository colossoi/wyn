//! Semantic residency planning for arrays and cross-dispatch scalars.
//!
//! The pass recognizes shared producers, runtime gathers, invariant scalar
//! reductions, and cost-eligible preludes of parallel operations after output
//! realization and semantic fusion. It records each decision as a typed
//! materialization plan, allocates its logical handoff resource, and rewires
//! consumers to explicit storage views or loads. Target lowering only chooses
//! and schedules the physical kernel recipe.

use std::collections::{HashMap, HashSet};

use polytype::Type;

use super::graph_ops;
use super::program::{
    AllocatedProgram, CompilerResource, CompilerResourceKind, LogicalSize, MaterializationRequirement,
    OutputWriter, ResourceId, SemanticOpId, SemanticResourceDecl, SemanticResourceRef,
};
use super::soac::{filter, screma};
use super::types::{
    EGraph, ENode, EffectToken, NodeId, PureOp, ResourceAccess, SegExtent, SegResourceAccess, SegSpace,
    SideEffectKind, SideEffectSite, Soac, SoacDestination, SoacEffect, SoacPlacement,
};
use crate::ast::TypeName;
use crate::flow::{BlockId, ExecutionModel};
use crate::interface::StorageRole;
use crate::types::TypeExt;

enum MaterializationPlan {
    FixedOperation {
        entry: usize,
        kind: FixedMaterializationKind,
        operation: FixedOperationMaterialization,
    },
    RuntimeArray {
        entry: usize,
        operation: RuntimeArrayMaterialization,
    },
    StagePrelude {
        entry: usize,
        prelude: StagePreludeMaterialization,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FixedMaterializationKind {
    SharedArray,
    Gather,
    Scalar,
}

impl FixedMaterializationKind {
    fn is_scalar(self) -> bool {
        self == Self::Scalar
    }
}

struct FixedOperationMaterialization {
    result: NodeId,
    producer: SemanticOpId,
    source_site: SideEffectSite,
    projected_site: SideEffectSite,
    projected_result: NodeId,
    projection: super::graph_projector::GraphProjection,
    space: SegSpace<SemanticResourceRef>,
    outputs: Vec<OutputSpec>,
}

struct RuntimeArrayMaterialization {
    result: NodeId,
    producer: SemanticOpId,
    source_site: SideEffectSite,
    projected_site: SideEffectSite,
    projection: super::graph_projector::GraphProjection,
    space: SegSpace<SemanticResourceRef>,
    /// Variable-cardinality array represented by capacity storage plus a
    /// separately stored logical length.
    scratch: ResourceId,
    elem_ty: Type<TypeName>,
    result_ty: Type<TypeName>,
    size: LogicalSize,
}

struct RuntimeArrayHandoff {
    data: ResourceId,
    length: ResourceId,
    elem_ty: Type<TypeName>,
    result_ty: Type<TypeName>,
    size: LogicalSize,
}

struct ParallelPrelude {
    root: NodeId,
    consumers: Vec<SemanticOpId>,
}

struct StagePreludeMaterialization {
    root: NodeId,
    insertion_site: Option<SideEffectSite>,
    recipe: super::graph_projector::ProjectedValueRecipe,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
    live_outs: Vec<ParallelPreludeLiveOut>,
}

struct ParallelPreludeLiveOut {
    source: NodeId,
    projected: NodeId,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
}

#[derive(Clone)]
struct OutputSpec {
    elem_ty: Type<TypeName>,
    result_ty: Type<TypeName>,
    size: LogicalSize,
}

struct InputReplacement {
    project: NodeId,
    view: NodeId,
    view_ty: Type<TypeName>,
    resource: ResourceId,
}

pub fn run(
    inner: &mut AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    select_in_place_destinations(inner);
    loop {
        super::semantic_graph::rebuild_dependencies(inner);
        let Some(plan) = next_materialization_plan(inner)? else {
            break;
        };
        apply_materialization(inner, plan, effect_ids)?;
    }
    super::semantic_graph::rebuild_dependencies(inner);
    super::realize_outputs::reconcile::run(inner)
        .map_err(|error| format!("materialized storage-view reconciliation failed: {error}"))?;
    if cfg!(debug_assertions) {
        super::realize_outputs::verify::check(inner).map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// Resolve TLC's uniqueness-only candidates from the final semantic use graph.
/// Output realization and fusion have already run, so `InputBuffer` here is a
/// physical choice rather than a source-level prediction.
fn select_in_place_destinations(inner: &mut AllocatedProgram) {
    for entry in &mut inner.entry_points {
        select_in_place_in_graph(&mut entry.graph);
    }
    for function in &mut inner.functions {
        select_in_place_in_graph(&mut function.graph);
    }
}

fn select_in_place_in_graph(graph: &mut EGraph) {
    // Multi-block liveness needs block-parameter substitution. Stay sound and
    // conservative until that representation is needed by a reuse candidate.
    if graph.skeleton.blocks.len() != 1 {
        clear_unique_input_candidates(graph);
        return;
    }
    let block_id = graph.skeleton.entry;
    let effect_count = graph.skeleton.blocks[block_id].side_effects.len();
    for effect_index in 0..effect_count {
        let (operands, map_inputs, operator_inputs, map_destinations, acc_destinations, filter_candidate) = {
            let effect = &graph.skeleton.blocks[block_id].side_effects[effect_index];
            match &effect.kind {
                SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => (
                    effect.operand_nodes.to_vec(),
                    op.lanes().maps.iter().map(|map| map.input_indices.clone()).collect(),
                    op.operators()
                        .into_iter()
                        .map(|operator| operator.input_indices.clone())
                        .collect::<Vec<_>>(),
                    op.lanes().maps.iter().map(|map| map.destination).collect(),
                    op.operators().into_iter().map(|operator| operator.destination).collect(),
                    false,
                ),
                SideEffectKind::Soac(SoacEffect(
                    _,
                    Soac::Filter(filter::Op {
                        state:
                            filter::SemanticState {
                                storage: filter::Output::Local { destination, .. },
                                ..
                            },
                        ..
                    }),
                )) => (
                    effect.operand_nodes.to_vec(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    destination.is_unplaced_unique_input(),
                ),
                _ => continue,
            }
        };

        let mut input_use_counts = HashMap::<usize, usize>::new();
        for &input in map_inputs.iter().flatten().chain(operator_inputs.iter().flatten()) {
            *input_use_counts.entry(input.index()).or_default() += 1;
        }
        let mut claimed = HashSet::<usize>::new();
        let resolve = |input: usize, claimed: &mut HashSet<usize>| {
            input_use_counts.get(&input) == Some(&1)
                && claimed.insert(input)
                && operands.get(input).is_some_and(|&node| {
                    reusable_input_type(&graph.types[&node])
                        && input_is_dead_after(graph, block_id, effect_index, node)
                })
        };

        let new_maps: Vec<_> = map_destinations
            .iter()
            .enumerate()
            .map(|(lane, destination)| {
                if !destination.is_unplaced_unique_input() {
                    return *destination;
                }
                map_inputs
                    .get(lane)
                    .and_then(|inputs| inputs.first())
                    .copied()
                    .filter(|input| resolve(input.index(), &mut claimed))
                    .map_or_else(SoacDestination::fresh, |_| {
                        destination.placed(SoacPlacement::InputBuffer)
                    })
            })
            .collect();
        let new_accs: Vec<_> = acc_destinations
            .iter()
            .enumerate()
            .map(|(operator, destination)| {
                if !destination.is_unplaced_unique_input() {
                    return *destination;
                }
                operator_inputs
                    .get(operator)
                    .and_then(|inputs| inputs.first())
                    .copied()
                    .filter(|input| resolve(input.index(), &mut claimed))
                    .map_or_else(SoacDestination::fresh, |_| {
                        destination.placed(SoacPlacement::InputBuffer)
                    })
            })
            .collect();

        let filter_destination = filter_candidate.then(|| {
            if reusable_input_type(&graph.types[&operands[0]])
                && input_is_dead_after(graph, block_id, effect_index, operands[0])
            {
                SoacDestination::unique_input().placed(SoacPlacement::InputBuffer)
            } else {
                SoacDestination::fresh()
            }
        });
        let effect = &mut graph.skeleton.blocks[block_id].side_effects[effect_index];
        match &mut effect.kind {
            SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                for (map, destination) in op.lanes_mut().maps.iter_mut().zip(new_maps) {
                    map.destination = destination;
                }
                for (operator, destination) in op.operators_mut().into_iter().zip(new_accs) {
                    operator.destination = destination;
                }
            }
            SideEffectKind::Soac(SoacEffect(
                _,
                Soac::Filter(filter::Op {
                    state:
                        filter::SemanticState {
                            storage: filter::Output::Local { destination, .. },
                            ..
                        },
                    ..
                }),
            )) => {
                if let Some(resolved) = filter_destination {
                    *destination = resolved;
                }
            }
            _ => {}
        }
        retype_input_buffer_results(graph, block_id, effect_index);
    }
}

fn retype_input_buffer_results(graph: &mut EGraph, block: BlockId, effect_index: usize) {
    let effect = &graph.skeleton.blocks[block].side_effects[effect_index];
    let Some(result) = effect.result else {
        return;
    };
    let projections: Vec<_> = graph
        .nodes
        .iter()
        .filter_map(|(node, definition)| match definition {
            ENode::Pure {
                op: PureOp::Project { index },
                operands,
            } if operands.as_slice() == [result] => Some((node, *index as usize)),
            _ => None,
        })
        .collect();
    let (result_types, changed) = {
        let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
            return;
        };
        let mut retyped = op.result_types();
        // Output realization may already have changed a projected field to a
        // storage view. Preserve those later EGIR decisions while changing
        // only fields whose uniqueness candidate became a physical reuse.
        for (projection, field) in &projections {
            retyped[*field] = graph.types[projection].clone();
        }
        let mut changed = false;
        for (lane, map) in op.lanes().maps.iter().enumerate() {
            if map.destination.is_input_buffer() {
                if let Some(input) = map.input_indices.first() {
                    retyped[lane] = op.lanes().inputs[input.index()].array.clone();
                    changed = true;
                }
            }
        }
        for (operator_index, operator) in op.operators().into_iter().enumerate() {
            if operator.destination.is_input_buffer() {
                if let Some(input) = operator.input_indices.first() {
                    retyped[op.lanes().maps.len() + operator_index] =
                        op.lanes().inputs[input.index()].array.clone();
                    changed = true;
                }
            }
        }
        (retyped, changed)
    };
    if !changed {
        return;
    }
    if let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) =
        &mut graph.skeleton.blocks[block].side_effects[effect_index].kind
    {
        op.set_result_types(&result_types);
    }
    graph.retype_node(
        result,
        Type::Constructed(TypeName::Tuple(result_types.len()), result_types.clone()),
    );
    for (projection, field) in projections {
        if let Some(ty) = result_types.get(field) {
            graph.retype_node(projection, ty.clone());
        }
    }
}

fn clear_unique_input_candidates(graph: &mut EGraph) {
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            match &mut effect.kind {
                SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                    for map in &mut op.lanes_mut().maps {
                        if map.destination.is_unplaced_unique_input() {
                            map.destination.make_fresh();
                        }
                    }
                    for operator in op.operators_mut() {
                        if operator.destination.is_unplaced_unique_input() {
                            operator.destination.make_fresh();
                        }
                    }
                }
                SideEffectKind::Soac(SoacEffect(
                    _,
                    Soac::Filter(filter::Op {
                        state:
                            filter::SemanticState {
                                storage: filter::Output::Local { destination, .. },
                                ..
                            },
                        ..
                    }),
                )) if destination.is_unplaced_unique_input() => {
                    destination.make_fresh();
                }
                _ => {}
            }
        }
    }
}

fn input_is_dead_after(graph: &EGraph, block: BlockId, index: usize, input: NodeId) -> bool {
    let body = &graph.skeleton.blocks[block];
    body.side_effects[index + 1..]
        .iter()
        .flat_map(|effect| effect.referenced_nodes())
        .chain(body.term.referenced_nodes())
        .all(|root| !graph_ops::pure_depends_on(graph, root, input))
}

fn reusable_input_type(ty: &Type<TypeName>) -> bool {
    match ty.array_variant() {
        Some(Type::Constructed(TypeName::ArrayVariantVirtual, _)) => return false,
        Some(Type::Constructed(TypeName::ArrayVariantView, _)) => return true,
        _ => {}
    }
    let runtime_sized =
        ty.array_size().is_some_and(|size| !matches!(size, Type::Constructed(TypeName::Size(_), _)));
    !runtime_sized || crate::types::array_view_buffer(ty).is_some()
}

fn next_materialization_plan(inner: &AllocatedProgram) -> Result<Option<MaterializationPlan>, String> {
    if let Some(plan) = plan_operation_result(inner)? {
        return Ok(Some(plan));
    }
    Ok(plan_parallel_prelude(inner).or_else(|| plan_direct_stage_prelude(inner)))
}

fn apply_materialization(
    inner: &mut AllocatedProgram,
    plan: MaterializationPlan,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    match plan {
        MaterializationPlan::FixedOperation {
            entry,
            kind,
            operation,
        } => {
            materialize_operation_result(inner, entry, kind, operation, effect_ids)?;
        }
        MaterializationPlan::RuntimeArray { entry, operation } => {
            materialize_runtime_array_result(inner, entry, operation, effect_ids)?;
        }
        MaterializationPlan::StagePrelude { entry, prelude } => {
            materialize_stage_prelude(inner, entry, prelude, effect_ids);
        }
    }
    Ok(())
}

fn plan_operation_result(inner: &AllocatedProgram) -> Result<Option<MaterializationPlan>, String> {
    let dependencies = super::semantic_graph::SemanticGraph::new(&inner.semantic_dependencies);
    for (entry_index, entry) in inner.entry_points.iter().enumerate() {
        for (block_id, block) in &entry.graph.skeleton.blocks {
            for (effect_index, effect) in block.side_effects.iter().enumerate() {
                let Some(result) = effect.result else {
                    continue;
                };
                let Some(&id) = effect.kind.soac_id() else {
                    continue;
                };
                let semantic_consumers = dependencies.value_consumers(&id).collect::<HashSet<_>>();
                let semantic_consumers = Some(&semantic_consumers);
                let source_site = SideEffectSite {
                    block: block_id,
                    index: effect_index,
                };
                match &effect.kind {
                    SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                        let Some(kind) = operation_result_residency(
                            entry,
                            op,
                            result,
                            source_site,
                            semantic_consumers,
                            inner.array_residency_demands.contains(&id),
                        ) else {
                            continue;
                        };
                        let Some(plan) =
                            operation_result_plan(entry_index, entry, op, result, id, source_site, kind)?
                        else {
                            continue;
                        };
                        return Ok(Some(plan));
                    }
                    SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) => {
                        if let Some(plan) = filter_runtime_array_plan(
                            entry_index,
                            entry,
                            op,
                            result,
                            id,
                            source_site,
                            semantic_consumers,
                        )? {
                            return Ok(Some(plan));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(None)
}

fn filter_runtime_array_plan(
    entry_index: usize,
    entry: &super::program::SemanticEntry,
    op: &filter::Op<super::types::Semantic>,
    result: NodeId,
    producer: SemanticOpId,
    source_site: SideEffectSite,
    consumers: Option<&HashSet<SemanticOpId>>,
) -> Result<Option<MaterializationPlan>, String> {
    let filter::SemanticState {
        space,
        storage:
            filter::Output::Runtime {
                scratch,
                length: filter::RuntimeLength::ViewOnly,
            },
    } = &op.state
    else {
        return Ok(None);
    };
    if !has_parallel_consumer(entry, consumers) {
        return Ok(None);
    }
    let elem_ty = op.body.output_element_type().clone();
    crate::ssa::layout::storage_elem_stride(&elem_ty).ok_or_else(|| {
        format!("runtime-array producer {producer:?} has no legal storage element layout")
    })?;
    let result_ty = entry.graph.types[&result].clone();
    let projection = super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
        .selected_operation_recipe(HashSet::from([source_site]))
        .map_err(|error| format!("could not project runtime-array producer {producer:?}: {error}"))?;
    let projected_site = projection
        .effect_site(source_site)
        .ok_or_else(|| format!("runtime-array projection omitted producer site for {producer:?}"))?;
    let size = LogicalSize::for_space(space, &elem_ty)
        .ok_or_else(|| format!("runtime-array producer {producer:?} has no legal logical storage size"))?;
    Ok(Some(MaterializationPlan::RuntimeArray {
        entry: entry_index,
        operation: RuntimeArrayMaterialization {
            result,
            producer,
            source_site,
            projected_site,
            projection,
            space: space.clone(),
            scratch: scratch.0,
            size,
            elem_ty,
            result_ty,
        },
    }))
}

fn operation_result_residency(
    entry: &super::program::SemanticEntry,
    op: &screma::Op<super::types::Semantic>,
    result: NodeId,
    site: SideEffectSite,
    consumers: Option<&HashSet<SemanticOpId>>,
    requires_array_storage: bool,
) -> Option<FixedMaterializationKind> {
    let screma::SemanticState::Segmented { resources, .. } = op.semantic_state() else {
        return None;
    };
    let cloneable = op.lanes().maps.iter().all(|map| map.destination.is_unplaced_fresh())
        && op.operators().into_iter().all(|operator| operator.destination.is_unplaced_fresh())
        && resources.iter().all(|resource| {
            resource.access == ResourceAccess::Read
                || entry
                    .outputs
                    .iter()
                    .filter_map(|output| output.resource)
                    .any(|output| output == resource.resource)
        });
    let dependencies = dependency_effects(&entry.graph, site.block, site.index);
    let upstream =
        dependencies.iter().copied().filter(|index| *index != site.index).collect::<HashSet<_>>();
    if !cloneable || !dependencies_are_cloneable(&entry.graph, site.block, &upstream) {
        return None;
    }

    match op {
        screma::Op::Map { lanes, .. } if !lanes.maps.is_empty() => {
            array_result_residency(entry, result, consumers, requires_array_storage)
        }
        screma::Op::Scan { .. } => array_result_residency(entry, result, consumers, requires_array_storage),
        screma::Op::Reduce { operators, .. }
            if operators.rest.is_empty()
                && (has_segmented_screma_consumer(entry, consumers)
                    || !entry.execution_model.is_compute())
                && scalar_result_is_used(&entry.graph, result, site.block, site.index)
                && invocation_invariant(entry, site.block, &dependencies) =>
        {
            Some(FixedMaterializationKind::Scalar)
        }
        _ => None,
    }
}

fn array_result_residency(
    entry: &super::program::SemanticEntry,
    result: NodeId,
    consumers: Option<&HashSet<SemanticOpId>>,
    requires_array_storage: bool,
) -> Option<FixedMaterializationKind> {
    if consumers.map_or(0, HashSet::len) >= 2 {
        Some(FixedMaterializationKind::SharedArray)
    } else if entry.graph.types[&result].contains_runtime_sized_composite_array() && requires_array_storage
    {
        Some(FixedMaterializationKind::Gather)
    } else {
        None
    }
}

fn operation_result_plan(
    entry_index: usize,
    entry: &super::program::SemanticEntry,
    op: &screma::Op<super::types::Semantic>,
    result: NodeId,
    producer: SemanticOpId,
    source_site: SideEffectSite,
    kind: FixedMaterializationKind,
) -> Result<Option<MaterializationPlan>, String> {
    let screma::SemanticState::Segmented { space, .. } = op.semantic_state() else {
        return Err(format!("materialization producer {producer:?} is not segmented"));
    };
    let projection = match super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
        .selected_operation_recipe(HashSet::from([source_site]))
    {
        Ok(projection) => projection,
        Err(_) => {
            // Projection feasibility is part of the materialization policy:
            // a producer depending on a loop/selection boundary parameter
            // cannot become an entry prepass and remains in its source graph.
            return Ok(None);
        }
    };
    let output_specs = output_specs(&entry.graph, kind, space, op)
        .ok_or_else(|| format!("materialization producer {producer:?} has an unsupported output layout"))?;
    let projected_result = projection
        .node(result)
        .ok_or_else(|| format!("materialization projection omitted result for {producer:?}"))?;
    let projected_site = projection
        .effect_site(source_site)
        .ok_or_else(|| format!("materialization projection omitted producer site for {producer:?}"))?;
    Ok(Some(MaterializationPlan::FixedOperation {
        entry: entry_index,
        kind,
        operation: FixedOperationMaterialization {
            result,
            producer,
            source_site,
            projected_site,
            projected_result,
            projection,
            space: space.clone(),
            outputs: output_specs,
        },
    }))
}

fn plan_parallel_prelude(inner: &AllocatedProgram) -> Option<MaterializationPlan> {
    for (entry_index, entry) in inner.entry_points.iter().enumerate() {
        let dependencies = super::semantic_graph::SemanticGraph::with_operation_captures(
            &inner.semantic_dependencies,
            &entry.graph,
        );
        for prelude in parallel_preludes(entry, &dependencies) {
            let ty = &entry.graph.types[&prelude.root];
            let Some(stride) = crate::ssa::layout::storage_elem_stride(ty) else {
                continue;
            };
            if ty.is_array() {
                continue;
            }
            let Some(consumer_sites) = operation_sites(&dependencies, &prelude.consumers) else {
                continue;
            };
            let Some(consumer_block) = consumer_sites.first().map(|site| site.block) else {
                continue;
            };
            if consumer_sites.iter().any(|site| site.block != consumer_block)
                || !consumer_sites.iter().all(|site| supports_parallel_prefix_consumer(entry, *site))
            {
                continue;
            }
            let consumer_site_set = consumer_sites.iter().copied().collect::<HashSet<_>>();
            if !source_is_observed_only_by_consumers_or_outputs(entry, prelude.root, &consumer_site_set) {
                continue;
            }
            let Some(insertion_site) = consumer_sites.iter().min_by_key(|site| site.index).copied() else {
                continue;
            };
            let Ok(recipe) =
                super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
                    .captured_value_recipe(prelude.root, insertion_site)
            else {
                continue;
            };
            let Some(live_outs) = parallel_prelude_live_outs(entry, prelude.root, &recipe) else {
                continue;
            };
            let Some(analysis) = super::residency_cost::analyze_prelude(inner, entry, &recipe) else {
                continue;
            };
            let invocations = launched_consumer_invocations(entry, &dependencies, &prelude.consumers);
            if !analysis.should_materialize(invocations) {
                continue;
            }
            return Some(MaterializationPlan::StagePrelude {
                entry: entry_index,
                prelude: StagePreludeMaterialization {
                    root: prelude.root,
                    insertion_site: Some(insertion_site),
                    recipe,
                    elem_ty: ty.clone(),
                    size: LogicalSize::FixedBytes(u64::from(stride)),
                    live_outs,
                },
            });
        }
    }
    None
}

/// Vertex and fragment invocation counts are selected by draw state, outside
/// the shader module. Price direct stage lifting against one modest batch so
/// only substantial uniform work clears the singleton-launch overhead.
const DIRECT_GRAPHICS_INVOCATIONS: u64 = 64;

fn plan_direct_stage_prelude(inner: &AllocatedProgram) -> Option<MaterializationPlan> {
    for (entry_index, entry) in inner.entry_points.iter().enumerate() {
        if matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
            continue;
        }
        let Ok(analysis) = super::stage_variance::StageDependenceAnalysis::for_entry(entry) else {
            continue;
        };
        let frontier = graph_ops::maximal_execution_frontier(&entry.graph, |node| {
            direct_stage_value_is_liftable(entry, &analysis, node)
        });
        for root in frontier {
            let ty = &entry.graph.types[&root];
            let Some(stride) = crate::ssa::layout::storage_elem_stride(ty) else {
                continue;
            };
            let Ok(recipe) =
                super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
                    .entry_value_recipe(root)
            else {
                continue;
            };
            let Some(live_outs) = parallel_prelude_live_outs(entry, root, &recipe) else {
                continue;
            };
            let Some(analysis) = super::residency_cost::analyze_prelude(inner, entry, &recipe) else {
                continue;
            };
            if !analysis.should_materialize(DIRECT_GRAPHICS_INVOCATIONS) {
                continue;
            }
            return Some(MaterializationPlan::StagePrelude {
                entry: entry_index,
                prelude: StagePreludeMaterialization {
                    root,
                    insertion_site: None,
                    recipe,
                    elem_ty: ty.clone(),
                    size: LogicalSize::FixedBytes(u64::from(stride)),
                    live_outs,
                },
            });
        }
    }
    None
}

fn direct_stage_value_is_liftable(
    entry: &super::program::SemanticEntry,
    analysis: &super::stage_variance::StageDependenceAnalysis,
    node: NodeId,
) -> bool {
    let Some(ENode::Pure { op, .. }) = entry.graph.nodes.get(node) else {
        return false;
    };
    if matches!(
        op,
        PureOp::Project { .. } | PureOp::ViewIndex | PureOp::PlaceIndex | PureOp::OutputSlot { .. }
    ) {
        return false;
    }
    let dependence = analysis.dependence(node);
    let Some(ty) = entry.graph.types.get(&node) else {
        return false;
    };
    dependence.is_stage_invariant()
        && !dependence.is_compile_time_constant()
        && dependence.loop_dependencies().is_empty()
        && !ty.is_array()
        && crate::ssa::layout::storage_elem_stride(ty).is_some()
}

/// Values produced by effects that move into a prepass may also feed retained
/// consumers without being dependencies of the primary captured boundary.
/// Publish those live-outs beside the primary handoff before removing their
/// source effects.
fn parallel_prelude_live_outs(
    entry: &super::program::SemanticEntry,
    root: NodeId,
    recipe: &super::graph_projector::ProjectedValueRecipe,
) -> Option<Vec<ParallelPreludeLiveOut>> {
    let producer_effects = recipe.projection.source_effects();
    let retained_terminators = retained_prelude_terminator_blocks(entry, &recipe.source);
    let mut candidates = Vec::new();
    for (block_id, block) in &entry.graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            if producer_effects.contains(&SideEffectSite {
                block: block_id,
                index,
            }) {
                candidates.extend(effect.result);
            }
        }
        candidates.extend(
            block
                .params
                .iter()
                .copied()
                .filter(|value| *value != root && recipe.projection.node(*value).is_some()),
        );
    }
    candidates.sort_unstable();
    candidates.dedup();

    let mut live_outs = Vec::new();
    for source in candidates {
        if source == root
            || !value_is_observed_outside(entry, source, producer_effects, &retained_terminators)
        {
            continue;
        }
        let elem_ty = entry.graph.types[&source].clone();
        let stride = crate::ssa::layout::storage_elem_stride(&elem_ty)?;
        live_outs.push(ParallelPreludeLiveOut {
            source,
            projected: recipe.projection.node(source)?,
            elem_ty,
            size: LogicalSize::FixedBytes(u64::from(stride)),
        });
    }
    Some(live_outs)
}

fn value_is_observed_outside(
    entry: &super::program::SemanticEntry,
    value: NodeId,
    producer_effects: &HashSet<SideEffectSite>,
    retained_terminators: &HashSet<BlockId>,
) -> bool {
    entry.graph.skeleton.blocks.iter().any(|(block_id, block)| {
        block.side_effects.iter().enumerate().any(|(index, effect)| {
            !producer_effects.contains(&SideEffectSite {
                block: block_id,
                index,
            }) && effect
                .referenced_nodes()
                .any(|reference| graph_ops::pure_depends_on(&entry.graph, reference, value))
        }) || retained_terminators.contains(&block_id)
            && block
                .term
                .referenced_nodes()
                .into_iter()
                .any(|reference| graph_ops::pure_depends_on(&entry.graph, reference, value))
    }) || entry
        .output_routes
        .iter()
        .any(|route| graph_ops::pure_depends_on(&entry.graph, route.source.value, value))
}

fn retained_prelude_terminator_blocks(
    entry: &super::program::SemanticEntry,
    source: &super::graph_projector::ValueRecipeSource,
) -> HashSet<BlockId> {
    match source {
        super::graph_projector::ValueRecipeSource::EntryBlock => {
            entry.graph.skeleton.blocks.keys().collect()
        }
        super::graph_projector::ValueRecipeSource::StructuredPrefix { continuation } => {
            let mut retained = HashSet::new();
            let mut pending = vec![*continuation];
            while let Some(block) = pending.pop() {
                if !retained.insert(block) {
                    continue;
                }
                pending.extend(entry.graph.skeleton.blocks[block].term.successors());
            }
            retained
        }
    }
}

fn parallel_preludes(
    entry: &super::program::SemanticEntry,
    dependencies: &super::semantic_graph::SemanticGraph,
) -> Vec<ParallelPrelude> {
    let mut preludes = Vec::<ParallelPrelude>::new();
    let mut by_root = HashMap::<NodeId, usize>::new();
    for capture in dependencies.captured_values() {
        for operation in dependencies.capture_consumers(capture) {
            let Some(site) = dependencies.operation_site(&operation) else {
                continue;
            };
            let SideEffectKind::Soac(SoacEffect(_, soac)) = &entry.graph.skeleton.effect(site).kind else {
                continue;
            };
            if soac.scheduling_space().is_none() {
                continue;
            }
            let root = parallel_prelude_boundary_root(entry, site, capture);
            if let Some(index) = by_root.get(&root).copied() {
                if !preludes[index].consumers.contains(&operation) {
                    preludes[index].consumers.push(operation);
                }
            } else {
                by_root.insert(root, preludes.len());
                preludes.push(ParallelPrelude {
                    root,
                    consumers: vec![operation],
                });
            }
        }
    }
    preludes
}

/// Captures in a structured continuation commonly project fields from its one
/// boundary value.  Schedule that value as a unit: the projector can then
/// detach the complete prefix once, while consumers keep using their existing
/// field projections after the boundary value is replaced by a handoff load.
fn parallel_prelude_boundary_root(
    entry: &super::program::SemanticEntry,
    consumer: SideEffectSite,
    capture: NodeId,
) -> NodeId {
    let params = &entry.graph.skeleton.blocks[consumer.block].params;
    let mut roots =
        params.iter().copied().filter(|param| graph_ops::pure_depends_on(&entry.graph, capture, *param));
    match (roots.next(), roots.next()) {
        (Some(root), None) => root,
        _ => capture,
    }
}

fn operation_sites(
    dependencies: &super::semantic_graph::SemanticGraph,
    operations: &[SemanticOpId],
) -> Option<Vec<SideEffectSite>> {
    operations.iter().map(|operation| dependencies.operation_site(operation)).collect()
}

fn supports_parallel_prefix_consumer(entry: &super::program::SemanticEntry, site: SideEffectSite) -> bool {
    matches!(
        &entry.graph.skeleton.effect(site).kind,
        SideEffectKind::Soac(SoacEffect(_, Soac::Screma(screma::Op::Map {
            lanes,
            state: screma::SemanticState::Segmented { .. },
        }))) if !lanes.maps.is_empty()
    )
}

fn source_is_observed_only_by_consumers_or_outputs(
    entry: &super::program::SemanticEntry,
    root: NodeId,
    consumers: &HashSet<SideEffectSite>,
) -> bool {
    // Realized output stores are valid additional observers: residency
    // rewrites their value dependencies to the same handoff load. Other
    // serial effects and terminators still keep the prefix in place.
    let output_effects = entry
        .output_routes
        .iter()
        .flat_map(|route| &route.writers)
        .filter_map(|writer| match writer {
            OutputWriter::Effect(effect) => Some(*effect),
            OutputWriter::Value(_) => None,
        })
        .collect::<HashSet<_>>();
    for (block_id, block) in &entry.graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            if effect
                .referenced_nodes()
                .any(|reference| graph_ops::pure_depends_on(&entry.graph, reference, root))
                && !consumers.contains(&SideEffectSite {
                    block: block_id,
                    index,
                })
                && !effect.effects.is_some_and(|(_, output)| output_effects.contains(&output))
            {
                return false;
            }
        }
        if block
            .term
            .referenced_nodes()
            .into_iter()
            .any(|reference| graph_ops::pure_depends_on(&entry.graph, reference, root))
        {
            return false;
        }
    }
    true
}

fn launched_consumer_invocations(
    entry: &super::program::SemanticEntry,
    dependencies: &super::semantic_graph::SemanticGraph,
    consumers: &[SemanticOpId],
) -> u64 {
    let workgroup = match entry.execution_model {
        ExecutionModel::Compute { local_size } => u64::from(local_size.0)
            .saturating_mul(u64::from(local_size.1))
            .saturating_mul(u64::from(local_size.2))
            .max(1),
        ExecutionModel::Vertex | ExecutionModel::Fragment => 1,
    };
    consumers.iter().fold(0u64, |total, consumer| {
        let Some(site) = dependencies.operation_site(consumer) else {
            return total;
        };
        let SideEffectKind::Soac(SoacEffect(_, soac)) = &entry.graph.skeleton.effect(site).kind else {
            return total;
        };
        let Some(space) = soac.scheduling_space() else {
            return total;
        };
        let logical = space.dims().iter().try_fold(1u64, |count, extent| match extent {
            SegExtent::Fixed(length) => count.checked_mul(u64::from(*length)),
            _ => None,
        });
        let launched = logical.map_or(workgroup, |count| count.div_ceil(workgroup) * workgroup);
        total.saturating_add(launched)
    })
}

fn has_parallel_consumer(
    entry: &super::program::SemanticEntry,
    consumers: Option<&HashSet<SemanticOpId>>,
) -> bool {
    has_matching_consumer(entry, consumers, |soac| soac.scheduling_space().is_some())
}

fn has_segmented_screma_consumer(
    entry: &super::program::SemanticEntry,
    consumers: Option<&HashSet<SemanticOpId>>,
) -> bool {
    has_matching_consumer(entry, consumers, |soac| {
        matches!(
            soac,
            Soac::Screma(op)
                if matches!(op.semantic_state(), screma::SemanticState::Segmented { .. })
        )
    })
}

fn has_matching_consumer(
    entry: &super::program::SemanticEntry,
    consumers: Option<&HashSet<SemanticOpId>>,
    mut supports: impl FnMut(&Soac<super::types::Semantic>) -> bool,
) -> bool {
    let Some(consumers) = consumers else {
        return false;
    };
    entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects).any(|effect| {
        matches!(&effect.kind, SideEffectKind::Soac(SoacEffect(id, soac)) if supports(soac) && consumers.contains(id))
    })
}

fn scalar_result_is_used(
    graph: &EGraph,
    result: NodeId,
    producer_block: BlockId,
    producer_index: usize,
) -> bool {
    graph.skeleton.blocks.iter().any(|(block_id, block)| {
        block.side_effects.iter().enumerate().any(|(index, effect)| {
            (block_id != producer_block || index != producer_index)
                && effect.referenced_nodes().any(|node| graph_ops::value_depends_on(graph, node, result))
        }) || block
            .term
            .clone()
            .try_map::<()>(
                |node| {
                    if graph_ops::value_depends_on(graph, node, result) {
                        Err(())
                    } else {
                        Ok(node)
                    }
                },
                Ok,
            )
            .is_err()
    })
}

fn invocation_invariant(
    entry: &super::program::SemanticEntry,
    block_id: BlockId,
    effects: &HashSet<usize>,
) -> bool {
    let Ok(dependence) = super::stage_variance::StageDependenceAnalysis::for_entry(entry) else {
        return false;
    };
    let block = &entry.graph.skeleton.blocks[block_id];
    let mut roots = Vec::new();
    for &index in effects {
        roots.extend(block.side_effects[index].referenced_nodes());
    }
    let reachable = graph_ops::execution_value_producer_closure(&entry.graph, roots).nodes;
    reachable.into_iter().all(|node| {
        let Some(ENode::FuncParam { index }) = entry.graph.nodes.get(node) else {
            return true;
        };
        dependence.dependence(node).is_stage_invariant()
            && super::residency_cost::entry_parameter_is_scalar_relocatable(entry, *index)
    })
}

fn dependencies_are_cloneable(graph: &EGraph, block_id: BlockId, effects: &HashSet<usize>) -> bool {
    let block = &graph.skeleton.blocks[block_id];
    effects.iter().all(|&index| {
        matches!(
            &block.side_effects[index].kind,
            SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op)))
                if matches!(op.semantic_state(), screma::SemanticState::Segmented { output_slots, resources, .. }
                    if output_slots.is_empty()
                        && op
                            .lanes()
                            .maps
                            .iter()
                            .all(|map| map.destination.is_unplaced_fresh())
                        && op
                            .operators()
                            .into_iter()
                            .all(|operator| operator.destination.is_unplaced_fresh())
                        && resources.iter().all(|resource| resource.access == ResourceAccess::Read))
        )
    })
}

fn materialize_operation_result(
    inner: &mut AllocatedProgram,
    entry_index: usize,
    kind: FixedMaterializationKind,
    operation: FixedOperationMaterialization,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    let FixedOperationMaterialization {
        result,
        producer: producer_id,
        source_site,
        projected_site,
        projected_result,
        projection,
        space,
        outputs: output_specs,
    } = operation;
    let entry = &inner.entry_points[entry_index];
    let source_output_resources =
        entry.outputs.iter().filter_map(|output| output.resource.map(|resource| resource.0)).collect();
    let producer_resources = entry.resources_referenced_by_projection(&projection);
    let producer_storage = entry.resource_declarations_for(&producer_resources);
    let execution_model = match entry.execution_model {
        ExecutionModel::Compute { local_size } => ExecutionModel::Compute { local_size },
        ExecutionModel::Vertex | ExecutionModel::Fragment => ExecutionModel::Compute {
            local_size: (64, 1, 1),
        },
    };
    let name_suffix = match kind {
        FixedMaterializationKind::SharedArray => "materialize_shared",
        FixedMaterializationKind::Gather => "gather_materialize",
        FixedMaterializationKind::Scalar => "prepass_scalar",
    };
    let compact_inputs = !entry.execution_model.is_compute();
    let mut producer_entry = projected_materialization_entry(
        inner,
        entry,
        name_suffix,
        execution_model,
        producer_storage,
        projection,
    );
    if compact_inputs {
        producer_entry.compact_interface();
    }
    let producer_owner = producer_id;
    let resource_kind = match kind {
        FixedMaterializationKind::SharedArray => CompilerResourceKind::MultiConsumerArray,
        FixedMaterializationKind::Gather => CompilerResourceKind::GatherHandoff,
        FixedMaterializationKind::Scalar => CompilerResourceKind::ScalarHandoff,
    };
    let output_resources = output_specs
        .iter()
        .enumerate()
        .map(|(slot, output)| {
            inner.alloc_compiler_resource(
                CompilerResource::new(resource_kind, Some(producer_owner), slot),
                output.elem_ty.clone(),
                output.size.clone(),
            )
        })
        .collect::<Vec<_>>();
    configure_operation_materialization(
        &mut producer_entry,
        kind,
        projected_site,
        projected_result,
        &output_resources,
        &output_specs,
        &source_output_resources,
        effect_ids,
    )?;

    rewrite_materialized_operation_source(
        &mut inner.entry_points[entry_index],
        kind,
        result,
        source_site,
        &output_resources,
        &output_specs,
        effect_ids,
    )?;
    let producer = match kind {
        FixedMaterializationKind::SharedArray => MaterializationRequirement::SharedArray {
            space,
            entry: producer_entry,
        },
        FixedMaterializationKind::Gather => MaterializationRequirement::Gather {
            space,
            entry: producer_entry,
        },
        FixedMaterializationKind::Scalar => MaterializationRequirement::Scalar {
            entry: producer_entry,
        },
    };
    inner.materializations.alloc(producer);
    Ok(())
}

fn materialize_runtime_array_result(
    inner: &mut AllocatedProgram,
    entry_index: usize,
    operation: RuntimeArrayMaterialization,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    let RuntimeArrayMaterialization {
        result,
        producer: producer_id,
        source_site,
        projected_site,
        projection,
        space,
        scratch,
        elem_ty,
        result_ty,
        size,
    } = operation;
    let entry = &inner.entry_points[entry_index];
    let producer_resources = entry.resources_referenced_by_projection(&projection);
    let producer_storage = entry.resource_declarations_for(&producer_resources);
    let execution_model = match entry.execution_model {
        ExecutionModel::Compute { local_size } => ExecutionModel::Compute { local_size },
        ExecutionModel::Vertex | ExecutionModel::Fragment => ExecutionModel::Compute {
            local_size: (64, 1, 1),
        },
    };
    let mut producer_entry = projected_materialization_entry(
        inner,
        entry,
        "materialize_filter",
        execution_model,
        producer_storage,
        projection,
    );
    let length = inner.alloc_compiler_resource(
        CompilerResource::new(CompilerResourceKind::FilterLenCell, Some(producer_id), 1),
        Type::Constructed(TypeName::UInt(32), vec![]),
        LogicalSize::FixedBytes(4),
    );
    let handoff = RuntimeArrayHandoff {
        data: scratch,
        length,
        elem_ty,
        result_ty,
        size,
    };
    producer_entry.set_resource_declaration(
        handoff.data,
        StorageRole::Output,
        &handoff.elem_ty,
        &handoff.size,
    );
    producer_entry.set_resource_declaration(
        handoff.length,
        StorageRole::Output,
        &Type::Constructed(TypeName::UInt(32), vec![]),
        &LogicalSize::FixedBytes(4),
    );
    let effect = producer_entry.graph.skeleton.effect_mut(projected_site);
    let SideEffectKind::Soac(SoacEffect(
        _,
        Soac::Filter(filter::Op {
            state: filter::SemanticState { storage, .. },
            ..
        }),
    )) = &mut effect.kind
    else {
        return Err("runtime-array materialization projection did not retain a filter".to_string());
    };
    *storage = filter::Output::Runtime {
        scratch: SemanticResourceRef(handoff.data),
        length: filter::RuntimeLength::Stored(SemanticResourceRef(handoff.length)),
    };
    producer_entry.compact_interface();

    rewrite_runtime_array_source(
        &mut inner.entry_points[entry_index],
        result,
        source_site,
        &handoff,
        effect_ids,
    )?;
    inner.materializations.alloc(MaterializationRequirement::RuntimeArray {
        space,
        entry: producer_entry,
    });
    Ok(())
}

fn rewrite_runtime_array_source(
    entry: &mut super::program::SemanticEntry,
    result: NodeId,
    source_site: SideEffectSite,
    handoff: &RuntimeArrayHandoff,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    entry.set_resource_declaration(handoff.data, StorageRole::Input, &handoff.elem_ty, &handoff.size);
    entry.set_resource_declaration(
        handoff.length,
        StorageRole::Input,
        &u32_ty,
        &LogicalSize::FixedBytes(4),
    );
    let length_view =
        graph_ops::intern_resource_view(&mut entry.graph, handoff.length, u32_ty.clone(), None);
    let (survivor_count, load_effect) =
        detached_scalar_handoff_load(&mut entry.graph, length_view, &u32_ty, effect_ids);
    let zero = graph_ops::intern_u32(&mut entry.graph, 0, None);
    let view = graph_ops::intern_chunked_resource_view(
        &mut entry.graph,
        handoff.data,
        zero,
        survivor_count,
        handoff.elem_ty.clone(),
        None,
    );
    let view_ty = entry.graph.types[&view].clone();
    retarget_input_metadata(
        &mut entry.graph,
        &[InputReplacement {
            project: result,
            view,
            view_ty,
            resource: handoff.data,
        }],
    )?;
    graph_ops::replace_all_references(&mut entry.graph, result, view);
    entry.graph.retype_node(result, handoff.result_ty.clone());
    for route in &mut entry.output_routes {
        if route.source.value == result {
            route.source.value = view;
        }
    }
    let block = &mut entry.graph.skeleton.blocks[source_site.block];
    block.side_effects.remove(source_site.index);
    block.side_effects.insert(source_site.index, load_effect);
    refresh_resource_reads_for_values(&mut entry.graph, &[survivor_count, view]);
    super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph);
    entry.compact_interface();
    Ok(())
}

fn configure_operation_materialization(
    producer: &mut super::program::SemanticEntry,
    kind: FixedMaterializationKind,
    producer_site: SideEffectSite,
    producer_result: NodeId,
    output_resources: &[ResourceId],
    output_specs: &[OutputSpec],
    source_output_resources: &HashSet<ResourceId>,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    let mut output_views = Vec::new();
    for (&resource, output) in output_resources.iter().zip(output_specs) {
        output_views.push(producer.declare_resource_view(
            resource,
            StorageRole::Output,
            &output.elem_ty,
            &output.size,
        ));
    }

    configure_materialized_soac(
        &mut producer.graph,
        kind,
        producer_site,
        &output_views,
        output_resources,
        source_output_resources,
    )?;
    configure_materialized_result(
        &mut producer.graph,
        kind,
        producer_result,
        &output_views,
        output_specs,
        effect_ids,
    );
    Ok(())
}

fn configure_materialized_soac(
    graph: &mut EGraph,
    kind: FixedMaterializationKind,
    producer_site: SideEffectSite,
    output_views: &[NodeId],
    output_resources: &[ResourceId],
    source_output_resources: &HashSet<ResourceId>,
) -> Result<(), String> {
    let producer_effect = graph.skeleton.effect_mut(producer_site);
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &mut producer_effect.kind else {
        return Err("fixed materialization projection did not retain a Screma operation".to_string());
    };
    let screma::SemanticState::Segmented {
        placement,
        output_slots,
        resources,
        ..
    } = op.semantic_state_mut()
    else {
        return Err("fixed materialization Screma was not segmented".to_string());
    };
    *placement = screma::Placement::Kernel;
    *output_slots = Vec::new();
    resources.retain(|access| {
        access.access == ResourceAccess::Read || !source_output_resources.contains(&access.resource.0)
    });
    if kind.is_scalar() {
        return Ok(());
    }
    for map in &mut op.lanes_mut().maps {
        map.destination.place(SoacPlacement::OutputView);
    }
    for operator in op.operators_mut() {
        operator.destination.place(SoacPlacement::OutputView);
    }
    producer_effect.operand_nodes.extend(output_views.iter().copied());
    let screma::SemanticState::Segmented { resources, .. } = op.semantic_state_mut() else {
        return Err("configured materialization Screma lost segmented state".to_string());
    };
    resources.extend(output_resources.iter().map(|resource| SegResourceAccess {
        resource: SemanticResourceRef(*resource),
        access: ResourceAccess::Write,
    }));
    resources.sort_by_key(|access| access.resource);
    Ok(())
}

fn configure_materialized_result(
    graph: &mut EGraph,
    kind: FixedMaterializationKind,
    result: NodeId,
    output_views: &[NodeId],
    output_specs: &[OutputSpec],
    effect_ids: &mut crate::IdSource<EffectToken>,
) {
    if kind.is_scalar() {
        if let Some((&output_view, output)) = output_views.first().zip(output_specs.first()) {
            let result_ty = graph.types[&result].clone();
            let value = if result_ty == output.result_ty {
                result
            } else {
                graph.intern_pure(
                    PureOp::Project { index: 0 },
                    smallvec::smallvec![result],
                    output.result_ty.clone(),
                    None,
                )
            };
            emit_scalar_handoff_store(
                graph,
                graph.skeleton.entry,
                output_view,
                value,
                &output.elem_ty,
                effect_ids,
            );
        }
        return;
    }

    let original_ty = graph.types[&result].clone();
    let view_types = output_views.iter().map(|view| graph.types[view].clone()).collect::<Vec<_>>();
    let preserve_single =
        output_specs.first().is_some_and(|output| view_types.len() == 1 && original_ty == output.result_ty);
    let materialized_ty = if preserve_single {
        view_types[0].clone()
    } else {
        Type::Constructed(TypeName::Tuple(view_types.len()), view_types)
    };
    graph.retype_node(result, materialized_ty);
}

fn rewrite_materialized_operation_source(
    entry: &mut super::program::SemanticEntry,
    kind: FixedMaterializationKind,
    result: NodeId,
    producer_site: SideEffectSite,
    output_resources: &[ResourceId],
    output_specs: &[OutputSpec],
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    let (block_id, effect_index) = (producer_site.block, producer_site.index);
    let mut array_replacements = Vec::new();
    let mut replacements = Vec::new();
    let mut scalar_effects = Vec::new();
    for (lane, (&resource, output)) in output_resources.iter().zip(output_specs).enumerate() {
        let view =
            graph_ops::intern_resource_view(&mut entry.graph, resource, output.elem_ty.clone(), None);
        // Pure projections are hash-consed, so interning is both the lookup and
        // the fallback construction. Do not rescan the whole node sea.
        let project = entry.graph.intern_pure(
            PureOp::Project { index: lane as u32 },
            smallvec::smallvec![result],
            output.result_ty.clone(),
            None,
        );
        let value = if kind.is_scalar() {
            let (loaded, load_effect) =
                detached_scalar_handoff_load(&mut entry.graph, view, &output.elem_ty, effect_ids);
            scalar_effects.push(load_effect);
            loaded
        } else {
            array_replacements.push(InputReplacement {
                project,
                view,
                view_ty: entry.graph.types[&view].clone(),
                resource,
            });
            view
        };
        replacements.push((project, value, resource));
        entry.set_resource_declaration(resource, StorageRole::Input, &output.elem_ty, &output.size);
    }
    retarget_input_metadata(&mut entry.graph, &array_replacements)?;
    for &(project, value, _) in &replacements {
        graph_ops::replace_all_references(&mut entry.graph, project, value);
        let value_ty = entry.graph.types[&value].clone();
        entry.graph.retype_node(project, value_ty);
    }
    let result_ty = entry.graph.types[&result].clone();
    let replacement_values = replacements.iter().map(|(_, value, _)| *value).collect::<Vec<_>>();
    let replacement_result = if replacement_values.len() == 1 && result_ty == output_specs[0].result_ty {
        replacement_values[0]
    } else {
        let replacement_ty = Type::Constructed(
            TypeName::Tuple(replacement_values.len()),
            replacement_values.iter().map(|value| entry.graph.types[value].clone()).collect(),
        );
        entry.graph.intern_pure(
            PureOp::Tuple(replacement_values.len()),
            replacement_values.into(),
            replacement_ty,
            None,
        )
    };
    graph_ops::replace_all_references(&mut entry.graph, result, replacement_result);
    let replacement_result_ty = entry.graph.types[&replacement_result].clone();
    entry.graph.retype_node(result, replacement_result_ty);
    for route in &mut entry.output_routes {
        if let Some((_, value, _)) =
            replacements.iter().find(|(project, _, _)| *project == route.source.value)
        {
            route.source.value = *value;
        }
    }
    entry.graph.skeleton.blocks[block_id].side_effects.remove(effect_index);
    for (offset, effect) in scalar_effects.into_iter().enumerate() {
        entry.graph.skeleton.blocks[block_id].side_effects.insert(effect_index + offset, effect);
    }
    if kind.is_scalar() {
        let loaded_values = replacements.iter().map(|(_, value, _)| *value).collect::<Vec<_>>();
        refresh_resource_reads_for_values(&mut entry.graph, &loaded_values);
    }
    super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph);
    Ok(())
}

fn materialize_stage_prelude(
    inner: &mut AllocatedProgram,
    entry_index: usize,
    prelude: StagePreludeMaterialization,
    effect_ids: &mut crate::IdSource<EffectToken>,
) {
    let StagePreludeMaterialization {
        root,
        insertion_site,
        recipe,
        elem_ty,
        size,
        live_outs,
    } = prelude;
    let super::graph_projector::ProjectedValueRecipe {
        projection,
        value: projected_root,
        result_block,
        source,
    } = recipe;
    let producer_effects = projection.source_effects().clone();
    let producer_entry = {
        let entry = &inner.entry_points[entry_index];
        let producer_resources = entry.resources_referenced_by_projection(&projection);
        projected_materialization_entry(
            inner,
            entry,
            "prepass_scalar",
            ExecutionModel::Compute {
                local_size: (1, 1, 1),
            },
            entry.resource_declarations_for(&producer_resources),
            projection,
        )
    };
    let values = std::iter::once(ParallelPreludeLiveOut {
        source: root,
        projected: projected_root,
        elem_ty,
        size,
    })
    .chain(live_outs)
    .collect::<Vec<_>>();
    let handoffs = values
        .into_iter()
        .enumerate()
        .map(|(slot, value)| {
            let resource = inner.alloc_compiler_resource(
                CompilerResource::new(CompilerResourceKind::ScalarHandoff, None, slot),
                value.elem_ty.clone(),
                value.size.clone(),
            );
            (resource, value)
        })
        .collect::<Vec<_>>();
    let mut producer_entry = producer_entry;
    for (resource, value) in &handoffs {
        let output_view = producer_entry.declare_resource_view(
            *resource,
            StorageRole::Output,
            &value.elem_ty,
            &value.size,
        );
        emit_scalar_handoff_store(
            &mut producer_entry.graph,
            result_block,
            output_view,
            value.projected,
            &value.elem_ty,
            effect_ids,
        );
    }
    producer_entry.compact_interface();

    let entry = &mut inner.entry_points[entry_index];
    let mut loaded_values = Vec::new();
    let mut load_effects = Vec::new();
    for (resource, value) in &handoffs {
        let view = entry.declare_resource_view(*resource, StorageRole::Input, &value.elem_ty, &value.size);
        let (loaded, load_effect) =
            detached_scalar_handoff_load(&mut entry.graph, view, &value.elem_ty, effect_ids);
        graph_ops::replace_all_references(&mut entry.graph, value.source, loaded);
        loaded_values.push(loaded);
        load_effects.push(load_effect);
    }
    let loaded_root = loaded_values[0];
    match source {
        super::graph_projector::ValueRecipeSource::EntryBlock => {
            if let Some(insertion_site) = insertion_site {
                replace_prelude_effects_with_load(entry, &producer_effects, insertion_site, load_effects);
            } else {
                replace_entry_prelude_with_load(entry, &producer_effects, load_effects);
            }
        }
        super::graph_projector::ValueRecipeSource::StructuredPrefix { continuation } => {
            replace_structured_prefix_with_load(
                entry,
                &producer_effects,
                continuation,
                loaded_root,
                load_effects,
            )
        }
    }
    refresh_resource_reads_for_values(&mut entry.graph, &loaded_values);
    super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph);
    entry.compact_interface();

    inner.materializations.alloc(MaterializationRequirement::Scalar {
        entry: producer_entry,
    });
}

fn projected_materialization_entry(
    program: &AllocatedProgram,
    source: &super::program::SemanticEntry,
    name_suffix: &str,
    execution_model: ExecutionModel,
    resource_declarations: Vec<SemanticResourceDecl>,
    projection: super::graph_projector::GraphProjection,
) -> super::program::SemanticEntry {
    let aliases = projection.remap_aliases(&source.aliases);
    super::program::SemanticEntry {
        name: fresh_entry_name(program, &format!("{}_{name_suffix}", source.name)),
        span: source.span,
        execution_model,
        inputs: source.inputs.clone(),
        outputs: Vec::new(),
        resource_declarations,
        params: source.params.clone(),
        return_ty: Type::Constructed(TypeName::Unit, vec![]),
        graph: projection.graph,
        control_headers: projection.control_headers,
        aliases,
        output_routes: Vec::new(),
    }
}

fn emit_scalar_handoff_store(
    graph: &mut EGraph,
    block: BlockId,
    output_view: NodeId,
    value: NodeId,
    elem_ty: &Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
) {
    let zero = graph_ops::intern_u32(graph, 0, None);
    graph_ops::emit_storage_store(
        graph,
        block,
        output_view,
        zero,
        value,
        elem_ty.clone(),
        effect_ids,
        None,
    );
}

fn detached_scalar_handoff_load(
    graph: &mut EGraph,
    view: NodeId,
    elem_ty: &Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> (NodeId, super::types::SideEffect) {
    let zero = graph_ops::intern_u32(graph, 0, None);
    let place = graph.intern_pure(
        PureOp::ViewIndex,
        smallvec::smallvec![view, zero],
        elem_ty.clone(),
        None,
    );
    graph_ops::detached_load(graph, place, elem_ty.clone(), effect_ids, None)
}

fn replace_prelude_effects_with_load(
    entry: &mut super::program::SemanticEntry,
    producer_effects: &HashSet<SideEffectSite>,
    insertion_site: SideEffectSite,
    load_effects: Vec<super::types::SideEffect>,
) {
    let mut removed = producer_effects.iter().map(|site| site.index).collect::<Vec<_>>();
    removed.sort_unstable();
    removed.dedup();
    let removed_before_consumer = removed.iter().filter(|index| **index < insertion_site.index).count();
    for index in removed.iter().rev() {
        entry.graph.skeleton.blocks[insertion_site.block].side_effects.remove(*index);
    }
    let insertion_index = insertion_site.index - removed_before_consumer;
    for (offset, load_effect) in load_effects.into_iter().enumerate() {
        entry.graph.skeleton.blocks[insertion_site.block]
            .side_effects
            .insert(insertion_index + offset, load_effect);
    }
}

fn replace_entry_prelude_with_load(
    entry: &mut super::program::SemanticEntry,
    producer_effects: &HashSet<SideEffectSite>,
    load_effects: Vec<super::types::SideEffect>,
) {
    entry.graph.skeleton.remove_effect_sites(producer_effects.iter().copied());
    let block = &mut entry.graph.skeleton.blocks[entry.graph.skeleton.entry];
    for (index, load) in load_effects.into_iter().enumerate() {
        block.side_effects.insert(index, load);
    }
}

fn replace_structured_prefix_with_load(
    entry: &mut super::program::SemanticEntry,
    producer_effects: &HashSet<SideEffectSite>,
    continuation: BlockId,
    loaded: NodeId,
    load_effects: Vec<super::types::SideEffect>,
) {
    entry.graph.skeleton.remove_effect_sites(producer_effects.iter().copied());
    let source_entry = entry.graph.skeleton.entry;
    entry.graph.skeleton.blocks[source_entry].side_effects.extend(load_effects);
    entry.graph.skeleton.blocks[source_entry].term = super::types::SkeletonTerminator::Branch {
        target: continuation,
        args: vec![loaded],
    };
    let aliases = super::skel_opt::run_one_body(&mut entry.graph);
    entry.aliases.extend(aliases);
    entry.retain_live_control_headers();
}

fn output_specs(
    graph: &EGraph,
    materialization: FixedMaterializationKind,
    space: &super::types::SegSpace,
    op: &screma::Op<super::types::Semantic>,
) -> Option<Vec<OutputSpec>> {
    let result_types = op.result_types();
    let output_elem_types = match (materialization, op) {
        (FixedMaterializationKind::Scalar, screma::Op::Reduce { .. }) => result_types.clone(),
        (_, screma::Op::Map { lanes, .. }) => {
            lanes.maps.iter().map(|map| map.output_element_type.clone()).collect()
        }
        (_, screma::Op::Scan { lanes, operators, .. }) => lanes
            .maps
            .iter()
            .map(|map| map.output_element_type.clone())
            .chain(
                std::iter::once(&operators.first)
                    .chain(&operators.rest)
                    .map(|operator| graph.types[&operator.neutral].clone()),
            )
            .collect(),
        _ => return None,
    };
    if output_elem_types.len() != result_types.len() {
        return None;
    }
    output_elem_types
        .into_iter()
        .zip(&result_types)
        .map(|(elem_ty, result_ty)| {
            let size = if materialization.is_scalar() {
                LogicalSize::FixedBytes(u64::from(crate::ssa::layout::storage_elem_stride(&elem_ty)?))
            } else {
                LogicalSize::for_space(space, &elem_ty)?
            };
            Some(OutputSpec {
                size,
                elem_ty,
                result_ty: result_ty.clone(),
            })
        })
        .collect()
}

fn refresh_resource_reads_for_values(graph: &mut EGraph, values: &[NodeId]) {
    let mut sites = Vec::<SideEffectSite>::new();
    for (block_id, block) in &graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            if matches!(&effect.kind, SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) if matches!(op.semantic_state(), screma::SemanticState::Segmented { .. }))
                && effect
                    .referenced_nodes()
                    .any(|node| values.iter().any(|value| graph_ops::value_depends_on(graph, node, *value)))
            {
                sites.push(SideEffectSite {
                    block: block_id,
                    index,
                });
            }
        }
    }
    for site in sites {
        let reads = {
            let effect = graph.skeleton.effect(site);
            super::semantic_graph::read_resources(graph, effect)
        };
        let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) =
            &mut graph.skeleton.effect_mut(site).kind
        else {
            continue;
        };
        let screma::SemanticState::Segmented { resources, .. } = op.semantic_state_mut() else {
            continue;
        };
        resources.retain(|access| access.access != ResourceAccess::Read);
        for read in reads {
            if let Some(existing) = resources.iter_mut().find(|access| access.resource == read.resource) {
                if existing.access == ResourceAccess::Write {
                    existing.access = ResourceAccess::ReadWrite;
                }
            } else {
                resources.push(read);
            }
        }
        resources.sort_by_key(|access| access.resource);
    }
}

/// Return the transitive semantic producer closure needed to compute one
/// effect.  Materialization is an entry prepass, so an internal producer chain
/// must move with the multi-consumer map instead of leaving dangling Project
/// nodes in the cloned graph.
fn dependency_effects(graph: &EGraph, block_id: BlockId, root: usize) -> HashSet<usize> {
    let block = &graph.skeleton.blocks[block_id];
    let producers: HashMap<NodeId, usize> = block
        .side_effects
        .iter()
        .enumerate()
        .filter_map(|(index, effect)| effect.result.map(|result| (result, index)))
        .collect();
    let mut retained = HashSet::from([root]);
    let mut effects = vec![root];
    while let Some(effect_index) = effects.pop() {
        let mut nodes: Vec<_> = block.side_effects[effect_index].referenced_nodes().collect();
        let mut visited = HashSet::new();
        while let Some(node) = nodes.pop() {
            if !visited.insert(node) {
                continue;
            }
            // Stop at a node that is another effect's result: that producer
            // is retained whole, and its own operands belong to it.
            if let Some(&producer) = producers.get(&node) {
                if retained.insert(producer) {
                    effects.push(producer);
                }
                continue;
            }
            if let Some(definition) = graph.nodes.get(node) {
                nodes.extend(definition.children());
            }
        }
    }
    retained
}

fn retarget_input_metadata(graph: &mut EGraph, replacements: &[InputReplacement]) -> Result<(), String> {
    let mut result_retypes = Vec::new();
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            match &mut effect.kind {
                SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                    let mut new_resources = Vec::new();
                    let mut domain_input = None;
                    for (input, input_type) in op.lanes_mut().inputs.iter_mut().enumerate() {
                        if let Some(replacement) = replacements
                            .iter()
                            .find(|replacement| effect.operand_nodes[input] == replacement.project)
                        {
                            input_type.array = replacement.view_ty.clone();
                            new_resources.push(replacement.resource);
                            if input == 0 {
                                let elem_ty = input_type.element();
                                let elem_bytes = crate::ssa::layout::storage_elem_stride(&elem_ty)
                                    .ok_or_else(|| {
                                        format!(
                                            "cannot retarget Screma input with non-storable element type {elem_ty:?}"
                                        )
                                    })?;
                                domain_input = Some((replacement.view, replacement.resource, elem_bytes));
                            }
                        }
                    }
                    {
                        let screma::SemanticState::Segmented { space, resources, .. } =
                            op.semantic_state_mut()
                        else {
                            continue;
                        };
                        replace_space_references(space, replacements);
                        if let Some((view, resource, elem_bytes)) = domain_input {
                            space.retarget_single_resource_length(
                                view,
                                SemanticResourceRef(resource),
                                elem_bytes,
                            );
                        }
                        for resource in new_resources {
                            if !resources.iter().any(|access| access.resource.0 == resource) {
                                resources.push(SegResourceAccess {
                                    resource: SemanticResourceRef(resource),
                                    access: ResourceAccess::Read,
                                });
                            }
                        }
                    }
                    let input_arrays =
                        op.lanes().inputs.iter().map(|input| input.array.clone()).collect::<Vec<_>>();
                    for map in &mut op.lanes_mut().maps {
                        if map.destination.is_input_buffer() {
                            if let Some(input) = map.input_indices.first() {
                                map.result_type = input_arrays[input.index()].clone();
                            }
                        }
                    }
                    for operator in op.operators_mut() {
                        if operator.destination.is_input_buffer() {
                            if let Some(input) = operator.input_indices.first() {
                                operator.result_type = input_arrays[input.index()].clone();
                            }
                        }
                    }
                    if let Some(result) = effect.result {
                        result_retypes.push((result, op.result_types()));
                    }
                }
                SideEffectKind::Soac(SoacEffect(_, Soac::Filter(filter::Op { body, state }))) => {
                    let mut domain_input = None;
                    if let Some(replacement) = replacements
                        .iter()
                        .find(|replacement| effect.operand_nodes[0] == replacement.project)
                    {
                        let input = match &mut body.input {
                            filter::Input::Plain(input) | filter::Input::Mapped { input, .. } => input,
                        };
                        input.array = replacement.view_ty.clone();
                        let elem_ty = input.element();
                        let elem_bytes = crate::ssa::layout::storage_elem_stride(&elem_ty)
                            .ok_or_else(|| {
                                format!(
                                    "cannot retarget filter input with non-storable element type {elem_ty:?}"
                                )
                            })?;
                        domain_input = Some((replacement.view, replacement.resource, elem_bytes));
                    }
                    replace_space_references(&mut state.space, replacements);
                    if let Some((view, resource, elem_bytes)) = domain_input {
                        state.space.retarget_single_resource_length(
                            view,
                            SemanticResourceRef(resource),
                            elem_bytes,
                        );
                    }
                }
                _ => {}
            }
        }
    }
    for (result, result_types) in result_retypes {
        let current = graph.types[&result].clone();
        let retyped = match current {
            Type::Constructed(TypeName::Tuple(_), _) => {
                Type::Constructed(TypeName::Tuple(result_types.len()), result_types)
            }
            _ if result_types.len() == 1 => result_types[0].clone(),
            _ => Type::Constructed(TypeName::Tuple(result_types.len()), result_types),
        };
        graph.retype_node(result, retyped);
    }
    Ok(())
}

fn replace_space_references(space: &mut super::types::SegSpace, replacements: &[InputReplacement]) {
    for replacement in replacements {
        space.replace_reference(
            replacement.project,
            replacement.view,
            SemanticResourceRef(replacement.resource),
        );
    }
}

fn fresh_entry_name(inner: &AllocatedProgram, base: &str) -> String {
    let available = |name: &str| {
        inner.entry_points.iter().all(|entry| entry.name != name)
            && inner.materializations.values().all(|requirement| requirement.entry().name != name)
    };
    if available(base) {
        return base.to_string();
    }
    for suffix in 1.. {
        let candidate = format!("{base}_{suffix}");
        if available(&candidate) {
            return candidate;
        }
    }
    unreachable!()
}

#[cfg(test)]
#[path = "residency_tests.rs"]
mod residency_tests;
