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
    AllocatedProgram, CompilerResource, CompilerResourceKind, LogicalSize, MaterializationId,
    MaterializationRequirement, OutputWriter, ResourceId, SemanticOpId, SemanticResourceDecl,
    SemanticResourceRef,
};
use super::soac::{filter, screma};
use super::types::{
    EGraph, ENode, EffectToken, NodeId, PureOp, ResourceAccess, SegExtent, SegResourceAccess, SegSpace,
    SideEffectKind, SideEffectSite, Soac, SoacEffect, SoacPlacement,
};
use crate::ast::TypeName;
use crate::flow::{BlockId, ExecutionModel};
use crate::interface::StorageRole;
use crate::types::TypeExt;

enum MaterializationPlan {
    FixedOperation {
        entry: usize,
        kind: FixedMaterializationKind,
        operation: ProjectedOperation,
        projected_result: NodeId,
        outputs: Vec<OutputSpec>,
    },
    RuntimeArray {
        entry: usize,
        operation: ProjectedOperation,
        /// Variable-cardinality array represented by capacity storage plus a
        /// separately stored logical length.
        scratch: ResourceId,
        elem_ty: Type<TypeName>,
        result_ty: Type<TypeName>,
        size: LogicalSize,
    },
    StagePrelude {
        entry: usize,
        insertion_site: Option<SideEffectSite>,
        recipe: super::graph_projector::ProjectedValueRecipe,
        outputs: Vec<StagePreludeOutput>,
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

struct ProjectedOperation {
    result: NodeId,
    producer: SemanticOpId,
    source_site: SideEffectSite,
    projected_site: SideEffectSite,
    projection: super::graph_projector::GraphProjection,
    space: SegSpace<SemanticResourceRef>,
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

struct StagePreludeOutput {
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
    program: &mut AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    loop {
        super::semantic_graph::rebuild_dependencies(program);
        let Some(plan) = next_materialization_plan(program)? else {
            break;
        };
        apply_materialization(program, plan, effect_ids)?;
    }
    super::semantic_graph::rebuild_dependencies(program);
    super::realize_outputs::reconcile::run(program)
        .map_err(|error| format!("materialized storage-view reconciliation failed: {error}"))?;
    if cfg!(debug_assertions) {
        super::realize_outputs::verify::check(program).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn next_materialization_plan(program: &AllocatedProgram) -> Result<Option<MaterializationPlan>, String> {
    if let Some(plan) = plan_operation_result(program)? {
        return Ok(Some(plan));
    }
    Ok(plan_parallel_prelude(program).or_else(|| plan_direct_stage_prelude(program)))
}

fn apply_materialization(
    program: &mut AllocatedProgram,
    plan: MaterializationPlan,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    match plan {
        MaterializationPlan::FixedOperation {
            entry,
            kind,
            operation,
            projected_result,
            outputs,
        } => {
            materialize_operation_result(
                program,
                entry,
                kind,
                operation,
                projected_result,
                outputs,
                effect_ids,
            )?;
        }
        MaterializationPlan::RuntimeArray {
            entry,
            operation,
            scratch,
            elem_ty,
            result_ty,
            size,
        } => {
            materialize_runtime_array_result(
                program, entry, operation, scratch, elem_ty, result_ty, size, effect_ids,
            )?;
        }
        MaterializationPlan::StagePrelude {
            entry,
            insertion_site,
            recipe,
            outputs,
        } => {
            materialize_stage_prelude(program, entry, insertion_site, recipe, outputs, effect_ids);
        }
    }
    Ok(())
}

fn plan_operation_result(program: &AllocatedProgram) -> Result<Option<MaterializationPlan>, String> {
    let dependencies = super::semantic_graph::SemanticGraph::new(&program.semantic_dependencies);
    for (entry_index, entry) in program.entry_points.iter().enumerate() {
        let uses = graph_ops::ValueUseIndex::build(&entry.graph);
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
                            program.array_residency_demands.contains(&id),
                            &uses,
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
        operation: ProjectedOperation {
            result,
            producer,
            source_site,
            projected_site,
            projection,
            space: space.clone(),
        },
        scratch: scratch.0,
        size,
        elem_ty,
        result_ty,
    }))
}

fn operation_result_residency(
    entry: &super::program::SemanticEntry,
    op: &screma::Op<super::types::Semantic>,
    result: NodeId,
    site: SideEffectSite,
    consumers: Option<&HashSet<SemanticOpId>>,
    requires_array_storage: bool,
    uses: &graph_ops::ValueUseIndex,
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
    let dependencies = dependency_effects(&entry.graph, site)?;
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
                && scalar_result_is_used(uses, result, site)
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
        operation: ProjectedOperation {
            result,
            producer,
            source_site,
            projected_site,
            projection,
            space: space.clone(),
        },
        projected_result,
        outputs: output_specs,
    }))
}

fn plan_parallel_prelude(program: &AllocatedProgram) -> Option<MaterializationPlan> {
    for (entry_index, entry) in program.entry_points.iter().enumerate() {
        let dependencies = super::semantic_graph::SemanticGraph::with_operation_captures(
            &program.semantic_dependencies,
            &entry.graph,
        );
        for prelude in parallel_preludes(entry, &dependencies) {
            let ty = &entry.graph.types[&prelude.root];
            if crate::ssa::layout::storage_elem_stride(ty).is_none() {
                continue;
            }
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
            let projector =
                super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers);
            if !source_is_observed_only_by_consumers_or_outputs(
                entry,
                projector.use_index(),
                prelude.root,
                &consumer_site_set,
            ) {
                continue;
            }
            let Some(insertion_site) = consumer_sites.iter().min_by_key(|site| site.index).copied() else {
                continue;
            };
            let Ok(recipe) = projector.captured_value_recipe_with_retained_values(
                prelude.root,
                insertion_site,
                entry.output_routes.iter().map(|route| route.source.value),
            ) else {
                continue;
            };
            let Some(outputs) = stage_prelude_outputs(entry, [prelude.root], &recipe) else {
                continue;
            };
            let Some(analysis) = super::residency_cost::analyze_prelude(program, entry, &recipe) else {
                continue;
            };
            let invocations = launched_consumer_invocations(entry, &dependencies, &prelude.consumers);
            if !analysis.should_materialize(invocations) {
                continue;
            }
            return Some(MaterializationPlan::StagePrelude {
                entry: entry_index,
                insertion_site: Some(insertion_site),
                recipe,
                outputs,
            });
        }
    }
    None
}

/// Direct entry invocation counts are selected by draw or dispatch state,
/// outside the shader module. Price direct stage lifting against one modest
/// batch so only substantial uniform work clears the singleton-launch overhead.
const DIRECT_STAGE_INVOCATIONS: u64 = 64;

fn plan_direct_stage_prelude(program: &AllocatedProgram) -> Option<MaterializationPlan> {
    for (entry_index, entry) in program.entry_points.iter().enumerate() {
        let Ok(analysis) = super::stage_variance::StageDependenceAnalysis::for_entry(entry) else {
            continue;
        };
        let frontier = graph_ops::maximal_execution_frontier(&entry.graph, |node| {
            direct_stage_value_is_liftable(entry, &analysis, node)
        });
        if frontier.is_empty() {
            continue;
        }
        let Ok(recipe) = super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
            .entry_values_recipe_with_retained_values(
                frontier.iter().copied(),
                entry.output_routes.iter().map(|route| route.source.value),
            )
        else {
            continue;
        };
        let Some(outputs) = stage_prelude_outputs(entry, frontier, &recipe) else {
            continue;
        };
        let Some(analysis) = super::residency_cost::analyze_prelude(program, entry, &recipe) else {
            continue;
        };
        if !analysis.should_materialize(DIRECT_STAGE_INVOCATIONS) {
            continue;
        }
        return Some(MaterializationPlan::StagePrelude {
            entry: entry_index,
            insertion_site: None,
            recipe,
            outputs,
        });
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
fn stage_prelude_outputs(
    entry: &super::program::SemanticEntry,
    roots: impl IntoIterator<Item = NodeId>,
    recipe: &super::graph_projector::ProjectedValueRecipe,
) -> Option<Vec<StagePreludeOutput>> {
    let mut sources = roots.into_iter().collect::<Vec<_>>();
    sources.extend(recipe.live_outs());
    let mut seen = HashSet::new();
    sources.retain(|source| seen.insert(*source));
    let mut outputs = Vec::with_capacity(sources.len());
    for source in sources {
        let elem_ty = entry.graph.types[&source].clone();
        let stride = crate::ssa::layout::storage_elem_stride(&elem_ty)?;
        outputs.push(StagePreludeOutput {
            source,
            projected: recipe.projection.node(source)?,
            elem_ty,
            size: LogicalSize::FixedBytes(u64::from(stride)),
        });
    }
    Some(outputs)
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
    uses: &graph_ops::ValueUseIndex,
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
    let observers = uses.pure_observers(root);
    observers.effect_sites().all(|site| {
        consumers.contains(&site)
            || entry
                .graph
                .skeleton
                .effect(site)
                .effects
                .is_some_and(|(_, output)| output_effects.contains(&output))
    }) && observers.terminator_blocks().next().is_none()
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
    uses: &graph_ops::ValueUseIndex,
    result: NodeId,
    producer: SideEffectSite,
) -> bool {
    let observers = uses.value_observers(result);
    observers.effect_sites().any(|site| site != producer) || observers.terminator_blocks().next().is_some()
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
    program: &mut AllocatedProgram,
    entry_index: usize,
    kind: FixedMaterializationKind,
    operation: ProjectedOperation,
    projected_result: NodeId,
    output_specs: Vec<OutputSpec>,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    let ProjectedOperation {
        result,
        producer: producer_id,
        source_site,
        projected_site,
        projection,
        space,
    } = operation;
    let materialization = program.materializations.alloc_id();
    let entry = &program.entry_points[entry_index];
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
        materialization,
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
            program.alloc_compiler_resource(
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
        &mut program.entry_points[entry_index],
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
    program.materializations.insert(materialization, producer);
    Ok(())
}

fn materialize_runtime_array_result(
    program: &mut AllocatedProgram,
    entry_index: usize,
    operation: ProjectedOperation,
    scratch: ResourceId,
    elem_ty: Type<TypeName>,
    result_ty: Type<TypeName>,
    size: LogicalSize,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    let ProjectedOperation {
        result,
        producer: producer_id,
        source_site,
        projected_site,
        projection,
        space,
    } = operation;
    let materialization = program.materializations.alloc_id();
    let entry = &program.entry_points[entry_index];
    let producer_resources = entry.resources_referenced_by_projection(&projection);
    let producer_storage = entry.resource_declarations_for(&producer_resources);
    let execution_model = match entry.execution_model {
        ExecutionModel::Compute { local_size } => ExecutionModel::Compute { local_size },
        ExecutionModel::Vertex | ExecutionModel::Fragment => ExecutionModel::Compute {
            local_size: (64, 1, 1),
        },
    };
    let mut producer_entry = projected_materialization_entry(
        materialization,
        entry,
        "materialize_filter",
        execution_model,
        producer_storage,
        projection,
    );
    let length = program.alloc_compiler_resource(
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
        &mut program.entry_points[entry_index],
        result,
        source_site,
        &handoff,
        effect_ids,
    )?;
    program.materializations.insert(
        materialization,
        MaterializationRequirement::RuntimeArray {
            space,
            entry: producer_entry,
        },
    );
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
    program: &mut AllocatedProgram,
    entry_index: usize,
    insertion_site: Option<SideEffectSite>,
    recipe: super::graph_projector::ProjectedValueRecipe,
    outputs: Vec<StagePreludeOutput>,
    effect_ids: &mut crate::IdSource<EffectToken>,
) {
    if outputs.is_empty() {
        return;
    }
    let super::graph_projector::ProjectedValueRecipe {
        projection,
        result_block,
        source,
        ..
    } = recipe;
    let materialization = program.materializations.alloc_id();
    let producer_effects = projection.source_effects().clone();
    let producer_entry = {
        let entry = &program.entry_points[entry_index];
        let producer_resources = entry.resources_referenced_by_projection(&projection);
        projected_materialization_entry(
            materialization,
            entry,
            "prepass_scalar",
            ExecutionModel::Compute {
                local_size: (1, 1, 1),
            },
            entry.resource_declarations_for(&producer_resources),
            projection,
        )
    };
    let handoffs = outputs
        .into_iter()
        .enumerate()
        .map(|(slot, value)| {
            let resource = program.alloc_compiler_resource(
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

    let entry = &mut program.entry_points[entry_index];
    let mut loaded_values = Vec::with_capacity(handoffs.len());
    let mut load_effects = Vec::with_capacity(handoffs.len());
    for (resource, value) in &handoffs {
        let view = entry.declare_resource_view(*resource, StorageRole::Input, &value.elem_ty, &value.size);
        let (loaded, load_effect) =
            detached_scalar_handoff_load(&mut entry.graph, view, &value.elem_ty, effect_ids);
        graph_ops::replace_all_references(&mut entry.graph, value.source, loaded);
        loaded_values.push(loaded);
        load_effects.push(load_effect);
    }
    let loaded_primary = loaded_values[0];
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
                loaded_primary,
                load_effects,
            )
        }
    }
    refresh_resource_reads_for_values(&mut entry.graph, &loaded_values);
    super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph);
    entry.compact_interface();

    program.materializations.insert(
        materialization,
        MaterializationRequirement::Scalar {
            entry: producer_entry,
        },
    );
}

fn projected_materialization_entry(
    materialization: MaterializationId,
    source: &super::program::SemanticEntry,
    name_suffix: &str,
    execution_model: ExecutionModel,
    resource_declarations: Vec<SemanticResourceDecl>,
    projection: super::graph_projector::GraphProjection,
) -> super::program::SemanticEntry {
    let aliases = projection.remap_aliases(&source.aliases);
    super::program::SemanticEntry {
        name: materialization.entry_name(&source.name, name_suffix),
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
fn dependency_effects(graph: &EGraph, root: SideEffectSite) -> Option<HashSet<usize>> {
    let closure =
        graph_ops::value_producer_closure(graph, graph.skeleton.get_effect(root)?.referenced_nodes());
    if closure.effects.iter().any(|site| site.block != root.block) {
        return None;
    }
    Some(closure.effects.into_iter().map(|site| site.index).chain([root.index]).collect())
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
