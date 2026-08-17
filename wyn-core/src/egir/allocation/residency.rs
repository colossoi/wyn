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

use super::super::graph_ops;
use super::super::graph_projector::{
    GraphProjection, GraphProjector, ProjectedValueRecipe, ValueRecipeSource,
};
use super::super::program::{
    CompilerResource, CompilerResourceKind, Entry, LogicalSize, MaterializationId,
    MaterializationRequirement, OutputWriter, Program, RealizedOutputRoute, ResourceId, SemanticOpId,
    SemanticResourceDecl, SemanticResourceRef, SlotSource,
};
use super::super::semantic_graph::SemanticGraph;
use super::super::soac::{filter, screma};
use super::super::stage_variance::StageDependenceAnalysis;
use super::super::types::{
    EGraph, EffectToken, PureOp, ResourceAccess, ResultBinding, SegExtent, SegResourceAccess, SegSpace,
    Semantic, SideEffect, SideEffectKind, SideEffectSite, SkeletonTerminator, Soac, SoacEffect, ValueId,
    ValueKind, ViewId, WynLanguage,
};
use super::ResourcesAllocated;
use crate::ast::TypeName;
use crate::flow::{BlockId, ExecutionModel};
use crate::interface::StorageRole;
use crate::pipeline_descriptor::{DispatchSize, Pipeline, StorageTextureSize};
use crate::types::TypeExt;

enum MaterializationPlan {
    FixedOperation {
        entry: usize,
        kind: FixedMaterializationKind,
        operation: ProjectedOperation,
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
        recipe: ProjectedValueRecipe,
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
    result: ResultBinding<Type<TypeName>>,
    projected_result: ResultBinding<Type<TypeName>>,
    producer: SemanticOpId,
    source_site: SideEffectSite,
    projected_site: SideEffectSite,
    projection: GraphProjection,
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
    root: ValueId,
    consumers: Vec<SemanticOpId>,
}

struct StagePreludeOutput {
    source: ValueId,
    projected: ValueId,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputStorage {
    Array,
    Scalar,
}

#[derive(Clone)]
struct OutputSpec {
    field: usize,
    storage: OutputStorage,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
}

struct InputReplacement {
    project: ValueId,
    view: ViewId,
    view_ty: Type<TypeName>,
    resource: ResourceId,
}

pub fn resolve_residency(mut program: ResourcesAllocated) -> Result<ResourcesAllocated, String> {
    loop {
        let Some(plan) = next_materialization_plan(&program)? else {
            break;
        };
        program = apply_materialization(program, plan)?;
    }
    if cfg!(debug_assertions) {
        super::super::realize_outputs::verify::check(&program).map_err(|error| error.to_string())?;
    }
    Ok(program)
}

fn next_materialization_plan(program: &ResourcesAllocated) -> Result<Option<MaterializationPlan>, String> {
    let dependencies = super::super::semantic_graph::dependencies(program);
    let array_residency_demands = super::super::semantic_graph::array_residency_demands(program);
    if let Some(plan) = plan_operation_result(program, &dependencies, &array_residency_demands)? {
        return Ok(Some(plan));
    }
    Ok(plan_parallel_prelude(program, &dependencies).or_else(|| plan_direct_stage_prelude(program)))
}

fn apply_materialization(
    program: ResourcesAllocated,
    plan: MaterializationPlan,
) -> Result<ResourcesAllocated, String> {
    match plan {
        MaterializationPlan::FixedOperation {
            entry,
            kind,
            operation,
            outputs,
        } => materialize_operation_result(program, entry, kind, operation, outputs),
        MaterializationPlan::RuntimeArray {
            entry,
            operation,
            scratch,
            elem_ty,
            result_ty,
            size,
        } => materialize_runtime_array_result(program, entry, operation, scratch, elem_ty, result_ty, size),
        MaterializationPlan::StagePrelude {
            entry,
            insertion_site,
            recipe,
            outputs,
        } => Ok(materialize_stage_prelude(
            program,
            entry,
            insertion_site,
            recipe,
            outputs,
        )),
    }
}

fn plan_operation_result(
    program: &ResourcesAllocated,
    dependency_edges: &[super::super::semantic_graph::SemanticDependency],
    array_residency_demands: &HashSet<SemanticOpId>,
) -> Result<Option<MaterializationPlan>, String> {
    let dependencies = SemanticGraph::new(dependency_edges);
    for (entry_index, entry) in program.entry_points.iter().enumerate() {
        let uses = graph_ops::ValueUseIndex::build(&entry.graph);
        for (block_id, block) in &entry.graph.skeleton.blocks {
            for (effect_index, effect) in block.side_effects.iter().enumerate() {
                let Some(result) = effect.result.as_ref() else {
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
                            array_residency_demands.contains(&id),
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
    entry: &Entry<Semantic>,
    op: &filter::Op<Semantic>,
    result: &ResultBinding<Type<TypeName>>,
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
    let result_ty = result.ty().clone();
    let projection = GraphProjector::new(&entry.graph)
        .selected_operation_recipe(HashSet::from([source_site]))
        .map_err(|error| format!("could not project runtime-array producer {producer:?}: {error}"))?;
    let projected_site = projection
        .effect_site(source_site)
        .ok_or_else(|| format!("runtime-array projection omitted producer site for {producer:?}"))?;
    let projected_result = projection
        .result(result)
        .map_err(|error| format!("runtime-array projection omitted result for {producer:?}: {error}"))?;
    let size = LogicalSize::for_space(space, &elem_ty)
        .ok_or_else(|| format!("runtime-array producer {producer:?} has no legal logical storage size"))?;
    Ok(Some(MaterializationPlan::RuntimeArray {
        entry: entry_index,
        operation: ProjectedOperation {
            result: result.clone(),
            projected_result,
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
    entry: &Entry<Semantic>,
    op: &screma::Op<Semantic>,
    result: &ResultBinding<Type<TypeName>>,
    site: SideEffectSite,
    consumers: Option<&HashSet<SemanticOpId>>,
    requires_array_storage: bool,
    uses: &graph_ops::ValueUseIndex,
) -> Option<FixedMaterializationKind> {
    let screma::SemanticState::Segmented { resources, .. } = op.semantic_state() else {
        return None;
    };
    let cloneable =
        op.result_state.iter().all(|result| result.ownership == crate::types::SoacOwnership::Fresh)
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

    if !op.form.post.result_types.is_empty() {
        array_result_residency(entry, result, consumers, requires_array_storage)
    } else if op.is_reduce()
        && op.form.reductions.len() == 1
        && (has_segmented_screma_consumer(entry, consumers) || !entry.execution_model.is_compute())
        && result.single_value().is_some_and(|value| scalar_result_is_used(uses, value, site))
        && invocation_invariant(entry, site.block, &dependencies)
    {
        Some(FixedMaterializationKind::Scalar)
    } else {
        None
    }
}

fn array_result_residency(
    _entry: &Entry<Semantic>,
    result: &ResultBinding<Type<TypeName>>,
    consumers: Option<&HashSet<SemanticOpId>>,
    requires_array_storage: bool,
) -> Option<FixedMaterializationKind> {
    if consumers.map_or(0, HashSet::len) >= 2 {
        Some(FixedMaterializationKind::SharedArray)
    } else if result.ty().contains_runtime_sized_composite_array() && requires_array_storage {
        Some(FixedMaterializationKind::Gather)
    } else {
        None
    }
}

fn operation_result_plan(
    entry_index: usize,
    entry: &Entry<Semantic>,
    op: &screma::Op<Semantic>,
    result: &ResultBinding<Type<TypeName>>,
    producer: SemanticOpId,
    source_site: SideEffectSite,
    kind: FixedMaterializationKind,
) -> Result<Option<MaterializationPlan>, String> {
    let screma::SemanticState::Segmented { space, .. } = op.semantic_state() else {
        return Err(format!("materialization producer {producer:?} is not segmented"));
    };
    let projection =
        match GraphProjector::new(&entry.graph).selected_operation_recipe(HashSet::from([source_site])) {
            Ok(projection) => projection,
            Err(_) => {
                // Projection feasibility is part of the materialization policy:
                // a producer depending on a loop/selection boundary parameter
                // cannot become an entry prepass and remains in its source graph.
                return Ok(None);
            }
        };
    let output_specs = output_specs(result, kind, space, op)
        .ok_or_else(|| format!("materialization producer {producer:?} has an unsupported output layout"))?;
    let projected_result = projection
        .result(result)
        .map_err(|error| format!("materialization projection omitted result for {producer:?}: {error}"))?;
    let projected_site = projection
        .effect_site(source_site)
        .ok_or_else(|| format!("materialization projection omitted producer site for {producer:?}"))?;
    Ok(Some(MaterializationPlan::FixedOperation {
        entry: entry_index,
        kind,
        operation: ProjectedOperation {
            result: result.clone(),
            projected_result,
            producer,
            source_site,
            projected_site,
            projection,
            space: space.clone(),
        },
        outputs: output_specs,
    }))
}

fn plan_parallel_prelude(
    program: &ResourcesAllocated,
    dependency_edges: &[super::super::semantic_graph::SemanticDependency],
) -> Option<MaterializationPlan> {
    for (entry_index, entry) in program.entry_points.iter().enumerate() {
        let dependencies = SemanticGraph::with_operation_captures(dependency_edges, &entry.graph);
        for prelude in parallel_preludes(entry, &dependencies) {
            let ty = &entry.graph.nodes[prelude.root].ty;
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
            let projector = GraphProjector::new(&entry.graph);
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
                entry.routes().map(|route| route.source.value),
            ) else {
                continue;
            };
            let Some(outputs) = stage_prelude_outputs(entry, [prelude.root], &recipe) else {
                continue;
            };
            let Some(analysis) = super::cost::analyze_prelude(program, entry, &recipe) else {
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

/// Dynamic draw and dispatch sizes are unavailable during shader compilation.
/// Price those direct stages against one modest batch so only substantial
/// uniform work clears the singleton-launch overhead.
const DIRECT_STAGE_INVOCATION_FALLBACK: u64 = 64;

fn plan_direct_stage_prelude(program: &ResourcesAllocated) -> Option<MaterializationPlan> {
    for (entry_index, entry) in program.entry_points.iter().enumerate() {
        let Ok(analysis) = StageDependenceAnalysis::for_entry(entry) else {
            continue;
        };
        let frontier = graph_ops::maximal_execution_frontier(&entry.graph, |node| {
            direct_stage_value_is_liftable(entry, &analysis, node)
        });
        if frontier.is_empty() {
            continue;
        }
        let Ok(recipe) = GraphProjector::new(&entry.graph).entry_values_recipe_with_retained_values(
            frontier.iter().copied(),
            entry.routes().map(|route| route.source.value),
        ) else {
            continue;
        };
        let Some(outputs) = stage_prelude_outputs(entry, frontier, &recipe) else {
            continue;
        };
        let Some(analysis) = super::cost::analyze_prelude(program, entry, &recipe) else {
            continue;
        };
        if !analysis.should_materialize(direct_stage_invocations(program, entry)) {
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

fn direct_stage_invocations(program: &ResourcesAllocated, entry: &Entry<Semantic>) -> u64 {
    let ExecutionModel::Compute { local_size } = &entry.execution_model else {
        return DIRECT_STAGE_INVOCATION_FALLBACK;
    };
    let workgroup = u64::from(local_size.0.max(1))
        .saturating_mul(u64::from(local_size.1.max(1)))
        .saturating_mul(u64::from(local_size.2.max(1)));
    let dispatch =
        program.data.core.pipeline.pipelines.iter().enumerate().find_map(|(pipeline_index, pipeline)| {
            match pipeline {
                Pipeline::Compute(compute) => {
                    compute.stages.iter().enumerate().find_map(|(stage_index, stage)| {
                        (program
                            .data
                            .core
                            .stage_entries
                            .get(pipeline_index)
                            .and_then(|entries| entries.get(stage_index))
                            == Some(&entry.id))
                        .then_some(&stage.dispatch_size)
                    })
                }
                Pipeline::Graphics(_) => None,
            }
        });
    if let Some(DispatchSize::Fixed {
        x,
        y,
        z,
        explicit: true,
    }) = dispatch
    {
        return u64::from(*x)
            .saturating_mul(u64::from(*y))
            .saturating_mul(u64::from(*z))
            .saturating_mul(workgroup);
    }

    let image_domain_is_inferred = matches!(
        dispatch,
        Some(DispatchSize::Fixed {
            x: 1,
            y: 1,
            z: 1,
            explicit: false
        })
    );
    let fixed_image = if image_domain_is_inferred {
        entry.inputs.iter().find_map(|input| {
            let (_, _, _, size) = input.storage_image_binding()?;
            match size {
                StorageTextureSize::Fixed { width, height } => Some((width, height)),
                StorageTextureSize::SameAsWindow => None,
            }
        })
    } else {
        None
    };
    if let Some((width, height)) = fixed_image {
        let groups_x = u64::from(width).div_ceil(u64::from(local_size.0.max(1)));
        let groups_y = u64::from(height).div_ceil(u64::from(local_size.1.max(1)));
        return groups_x.saturating_mul(groups_y).saturating_mul(workgroup);
    }

    DIRECT_STAGE_INVOCATION_FALLBACK
}

fn direct_stage_value_is_liftable(
    entry: &Entry<Semantic>,
    analysis: &StageDependenceAnalysis,
    node: ValueId,
) -> bool {
    let Some(ValueKind::Pure { op, .. }) = entry.graph.nodes.get(node).map(|node| &node.kind) else {
        return false;
    };
    if matches!(op, PureOp::Project { .. }) {
        return false;
    }
    let dependence = analysis.dependence(node);
    let Some(ty) = entry.graph.nodes.get(node).map(|node| &node.ty) else {
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
    entry: &Entry<Semantic>,
    roots: impl IntoIterator<Item = ValueId>,
    recipe: &ProjectedValueRecipe,
) -> Option<Vec<StagePreludeOutput>> {
    let mut sources = roots.into_iter().collect::<Vec<_>>();
    sources.extend(recipe.live_outs());
    let mut seen = HashSet::new();
    sources.retain(|source| seen.insert(*source));
    let mut outputs = Vec::with_capacity(sources.len());
    for source in sources {
        let elem_ty = entry.graph.nodes[source].ty.clone();
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

fn parallel_preludes(entry: &Entry<Semantic>, dependencies: &SemanticGraph) -> Vec<ParallelPrelude> {
    let mut preludes = Vec::<ParallelPrelude>::new();
    let mut by_root = HashMap::<ValueId, usize>::new();
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
    entry: &Entry<Semantic>,
    consumer: SideEffectSite,
    capture: ValueId,
) -> ValueId {
    let params = &entry.graph.skeleton.blocks[consumer.block].params;
    let mut roots = params
        .iter()
        .map(|param| param.value())
        .filter(|param| graph_ops::pure_depends_on(&entry.graph, capture, *param));
    match (roots.next(), roots.next()) {
        (Some(root), None) => root,
        _ => capture,
    }
}

fn operation_sites(
    dependencies: &SemanticGraph,
    operations: &[SemanticOpId],
) -> Option<Vec<SideEffectSite>> {
    operations.iter().map(|operation| dependencies.operation_site(operation)).collect()
}

fn supports_parallel_prefix_consumer(entry: &Entry<Semantic>, site: SideEffectSite) -> bool {
    matches!(
        &entry.graph.skeleton.effect(site).kind,
        SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op)))
            if op.is_map()
                && op.form.post.is_identity()
                && !op.form.post.result_types.is_empty()
                && matches!(op.semantic_state(), screma::SemanticState::Segmented { .. })
    )
}

fn source_is_observed_only_by_consumers_or_outputs(
    entry: &Entry<Semantic>,
    uses: &graph_ops::ValueUseIndex,
    root: ValueId,
    consumers: &HashSet<SideEffectSite>,
) -> bool {
    // Realized output stores are valid additional observers: residency
    // rewrites their value dependencies to the same handoff load. Other
    // serial effects and terminators still keep the prefix in place.
    let output_effects = entry
        .routes()
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
    entry: &Entry<Semantic>,
    dependencies: &SemanticGraph,
    consumers: &[SemanticOpId],
) -> u64 {
    let workgroup = match &entry.execution_model {
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

fn has_parallel_consumer(entry: &Entry<Semantic>, consumers: Option<&HashSet<SemanticOpId>>) -> bool {
    has_matching_consumer(entry, consumers, |soac| soac.scheduling_space().is_some())
}

fn has_segmented_screma_consumer(
    entry: &Entry<Semantic>,
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
    entry: &Entry<Semantic>,
    consumers: Option<&HashSet<SemanticOpId>>,
    mut supports: impl FnMut(&Soac<Semantic>) -> bool,
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
    result: ValueId,
    producer: SideEffectSite,
) -> bool {
    let observers = uses.value_observers(result);
    observers.effect_sites().any(|site| site != producer) || observers.terminator_blocks().next().is_some()
}

fn invocation_invariant(entry: &Entry<Semantic>, block_id: BlockId, effects: &HashSet<usize>) -> bool {
    let Ok(dependence) = StageDependenceAnalysis::for_entry(entry) else {
        return false;
    };
    let block = &entry.graph.skeleton.blocks[block_id];
    let mut roots = Vec::new();
    for &index in effects {
        roots.extend(graph_ops::effect_value_inputs(
            &entry.graph,
            &block.side_effects[index],
        ));
    }
    let reachable = graph_ops::execution_value_producer_closure(&entry.graph, roots).nodes;
    reachable.into_iter().all(|node| {
        let Some(ValueKind::FuncParam { parameter }) = entry.graph.nodes.get(node).map(|node| &node.kind)
        else {
            return true;
        };
        dependence.dependence(node).is_stage_invariant()
            && super::cost::entry_parameter_is_scalar_relocatable(entry, parameter.index())
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
                            .result_state
                            .iter()
                            .all(|result| result.ownership == crate::types::SoacOwnership::Fresh)
                        && resources.iter().all(|resource| resource.access == ResourceAccess::Read))
        )
    })
}

fn materialize_operation_result(
    program: ResourcesAllocated,
    entry_index: usize,
    kind: FixedMaterializationKind,
    operation: ProjectedOperation,
    output_specs: Vec<OutputSpec>,
) -> Result<ResourcesAllocated, String> {
    let Program {
        functions,
        externs,
        mut entry_points,
        constants,
        mut data,
        mut global_context,
        state: _,
    } = program;
    let ProjectedOperation {
        result,
        projected_result,
        producer: producer_id,
        source_site,
        projected_site,
        projection,
        space,
    } = operation;
    let materialization = data.materializations.alloc_id();
    let entry = &entry_points[entry_index];
    let routed_output_resources = output_specs
        .iter()
        .map(|output| {
            result
                .field(output.field)
                .and_then(|field| entry.resource_for_result(&field))
                .map(|resource| resource.0)
        })
        .collect::<Vec<_>>();
    let source_output_resources =
        entry.outputs.iter().filter_map(|output| output.resource.map(|resource| resource.0)).collect();
    let producer_resources = entry.resources_referenced_by_projection(&projection);
    let producer_storage = entry.resource_declarations_for(&producer_resources);
    let execution_model = match &entry.execution_model {
        ExecutionModel::Compute { local_size } => ExecutionModel::Compute {
            local_size: *local_size,
        },
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
        &mut data.core.identities,
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
    let array_resource_kind = match kind {
        FixedMaterializationKind::SharedArray => CompilerResourceKind::MultiConsumerArray,
        FixedMaterializationKind::Gather => CompilerResourceKind::GatherHandoff,
        FixedMaterializationKind::Scalar => CompilerResourceKind::ScalarHandoff,
    };
    let output_resources = output_specs
        .iter()
        .zip(routed_output_resources)
        .enumerate()
        .map(|(slot, (output, routed))| {
            if let Some(resource) = routed {
                return resource;
            }
            let resource_kind = match output.storage {
                OutputStorage::Array => array_resource_kind,
                OutputStorage::Scalar => CompilerResourceKind::ScalarHandoff,
            };
            data.alloc_compiler_resource(
                CompilerResource::new(resource_kind, Some(producer_owner), slot),
                output.elem_ty.clone(),
                output.size.clone(),
            )
        })
        .collect::<Vec<_>>();
    configure_operation_materialization(
        &mut producer_entry,
        projected_site,
        &projected_result,
        &output_resources,
        &output_specs,
        &source_output_resources,
        &mut global_context.effect_ids,
    )?;

    rewrite_materialized_operation_source(
        &mut entry_points[entry_index],
        &result,
        source_site,
        &output_resources,
        &output_specs,
        &mut global_context.effect_ids,
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
    data.materializations.insert(materialization, producer);
    Ok(Program::from_parts(
        functions,
        externs,
        entry_points,
        constants,
        data,
        global_context,
    ))
}

fn materialize_runtime_array_result(
    program: ResourcesAllocated,
    entry_index: usize,
    operation: ProjectedOperation,
    scratch: ResourceId,
    elem_ty: Type<TypeName>,
    result_ty: Type<TypeName>,
    size: LogicalSize,
) -> Result<ResourcesAllocated, String> {
    let Program {
        functions,
        externs,
        mut entry_points,
        constants,
        mut data,
        mut global_context,
        state: _,
    } = program;
    let ProjectedOperation {
        result,
        projected_result: _,
        producer: producer_id,
        source_site,
        projected_site,
        projection,
        space,
    } = operation;
    let materialization = data.materializations.alloc_id();
    let entry = &entry_points[entry_index];
    let producer_resources = entry.resources_referenced_by_projection(&projection);
    let producer_storage = entry.resource_declarations_for(&producer_resources);
    let execution_model = match &entry.execution_model {
        ExecutionModel::Compute { local_size } => ExecutionModel::Compute {
            local_size: *local_size,
        },
        ExecutionModel::Vertex | ExecutionModel::Fragment => ExecutionModel::Compute {
            local_size: (64, 1, 1),
        },
    };
    let mut producer_entry = projected_materialization_entry(
        &mut data.core.identities,
        materialization,
        entry,
        "materialize_filter",
        execution_model,
        producer_storage,
        projection,
    );
    let length = data.alloc_compiler_resource(
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
        &mut entry_points[entry_index],
        result
            .single_value()
            .ok_or_else(|| "runtime-array materialization requires one result value".to_string())?,
        source_site,
        &handoff,
        &mut global_context.effect_ids,
    )?;
    data.materializations.insert(
        materialization,
        MaterializationRequirement::RuntimeArray {
            space,
            entry: producer_entry,
        },
    );
    Ok(Program::from_parts(
        functions,
        externs,
        entry_points,
        constants,
        data,
        global_context,
    ))
}

fn rewrite_runtime_array_source(
    entry: &mut Entry<Semantic>,
    result: ValueId,
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
    let view_ty = entry.graph.nodes[view].ty.clone();
    let view_id = entry.graph.view_id(view);
    retarget_input_metadata(
        &mut entry.graph,
        &[InputReplacement {
            project: result,
            view: view_id,
            view_ty,
            resource: handoff.data,
        }],
    )?;
    entry.graph.replace_value_references(result, view);
    entry.graph.retype_node(result, handoff.result_ty.clone());
    for route in entry.routes_mut() {
        if route.source.value == result {
            route.source.value = view;
        }
    }
    let block = &mut entry.graph.skeleton.blocks[source_site.block];
    block.side_effects.remove(source_site.index);
    block.side_effects.insert(source_site.index, load_effect);
    refresh_resource_reads_for_values(&mut entry.graph, &[survivor_count, view]);
    let route_values = entry.routes().flat_map(|route| route.referenced_values()).collect::<Vec<_>>();
    super::super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph, route_values);
    entry.compact_interface();
    Ok(())
}

fn configure_operation_materialization(
    producer: &mut Entry<Semantic>,
    producer_site: SideEffectSite,
    producer_result: &ResultBinding<Type<TypeName>>,
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
        let field = producer_result.field(output.field).expect("materialized output field exists");
        let source = field.single_value().expect("materialized output has one value source");
        producer.internal_results.push(super::super::ir::InternalResultRoute {
            resource: SemanticResourceRef(resource),
            route: RealizedOutputRoute {
                source: SlotSource {
                    block: producer_site.block,
                    value: source,
                },
                writers: vec![OutputWriter::Value(source)],
            },
        });
    }

    configure_materialized_soac(
        &mut producer.graph,
        producer_site,
        output_resources,
        output_specs,
        source_output_resources,
    )?;
    let replacements = configure_materialized_result(
        &mut producer.graph,
        producer_site.block,
        producer_result,
        &output_views,
        output_specs,
        effect_ids,
    )?;
    for route in producer.routes_mut() {
        route.replace_values(&replacements);
    }
    Ok(())
}

fn configure_materialized_soac(
    graph: &mut EGraph,
    producer_site: SideEffectSite,
    output_resources: &[ResourceId],
    output_specs: &[OutputSpec],
    source_output_resources: &HashSet<ResourceId>,
) -> Result<(), String> {
    let producer_effect = graph.skeleton.effect_mut(producer_site);
    let SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))),
        ..
    } = producer_effect
    else {
        return Err("fixed materialization projection did not retain a Screma operation".to_string());
    };
    let array_outputs = output_resources
        .iter()
        .zip(output_specs)
        .filter_map(|(&resource, output)| {
            (output.storage == OutputStorage::Array).then_some((output.field, resource))
        })
        .collect::<Vec<_>>();

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
    *output_slots = (0..array_outputs.len()).map(super::super::ir::OutputSlotId).collect();
    resources.retain(|access| {
        access.access == ResourceAccess::Read || !source_output_resources.contains(&access.resource.0)
    });
    resources.extend(array_outputs.into_iter().map(|(_, resource)| SegResourceAccess {
        resource: SemanticResourceRef(resource),
        access: ResourceAccess::Write,
    }));
    resources.sort_by_key(|access| access.resource);
    Ok(())
}

fn configure_materialized_result(
    graph: &mut EGraph,
    block: BlockId,
    result: &ResultBinding<Type<TypeName>>,
    output_views: &[ValueId],
    output_specs: &[OutputSpec],
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<Vec<(ValueId, ValueId)>, String> {
    let mut replacements = Vec::new();
    for (&output_view, output) in output_views.iter().zip(output_specs) {
        let field = result
            .field(output.field)
            .ok_or_else(|| format!("materialized output {} has no result field", output.field))?;
        if output.storage == OutputStorage::Array {
            let destination = graph_ops::bind_result_to_view(graph, &field, output_view)?;
            replacements.extend(graph_ops::rebind_result_value_references(
                graph,
                &field,
                &destination,
            )?);
        } else {
            let value = field.single_value().ok_or_else(|| {
                format!(
                    "materialized scalar output {} is not one result leaf",
                    output.field
                )
            })?;
            emit_scalar_handoff_store(graph, block, output_view, value, &output.elem_ty, effect_ids);
        }
    }
    Ok(replacements)
}

fn rewrite_materialized_operation_source(
    entry: &mut Entry<Semantic>,
    result: &ResultBinding<Type<TypeName>>,
    producer_site: SideEffectSite,
    output_resources: &[ResourceId],
    output_specs: &[OutputSpec],
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), String> {
    let (block_id, effect_index) = (producer_site.block, producer_site.index);
    let mut array_replacements = Vec::new();
    let mut replacements = Vec::new();
    let mut scalar_effects = Vec::new();
    for (&resource, output) in output_resources.iter().zip(output_specs) {
        let view =
            graph_ops::intern_resource_view(&mut entry.graph, resource, output.elem_ty.clone(), None);
        let source = result
            .field(output.field)
            .and_then(|field| field.single_value())
            .ok_or_else(|| format!("materialized output {} is not one result leaf", output.field))?;
        let value = if output.storage == OutputStorage::Scalar {
            let (loaded, load_effect) =
                detached_scalar_handoff_load(&mut entry.graph, view, &output.elem_ty, effect_ids);
            scalar_effects.push(load_effect);
            loaded
        } else {
            let view = entry.graph.view_id(view);
            array_replacements.push(InputReplacement {
                project: source,
                view,
                view_ty: entry.graph.nodes[view.value()].ty.clone(),
                resource,
            });
            view.value()
        };
        replacements.push((source, value, resource));
        entry.set_resource_declaration(resource, StorageRole::Input, &output.elem_ty, &output.size);
    }
    retarget_input_metadata(&mut entry.graph, &array_replacements)?;
    for &(source, value, _) in &replacements {
        entry.graph.replace_value_references(source, value);
        let value_ty = entry.graph.nodes[value].ty.clone();
        entry.graph.retype_node(source, value_ty);
    }
    for route in entry.routes_mut() {
        if let Some((_, value, _)) =
            replacements.iter().find(|(source, _, _)| *source == route.source.value)
        {
            route.source.value = *value;
        }
    }
    entry.graph.skeleton.blocks[block_id].side_effects.remove(effect_index);
    for (offset, effect) in scalar_effects.into_iter().enumerate() {
        entry.graph.skeleton.blocks[block_id].side_effects.insert(effect_index + offset, effect);
    }
    let loaded_values = replacements
        .iter()
        .zip(output_specs)
        .filter_map(|((_, value, _), output)| (output.storage == OutputStorage::Scalar).then_some(*value))
        .collect::<Vec<_>>();
    refresh_resource_reads_for_values(&mut entry.graph, &loaded_values);
    let route_values = entry.routes().flat_map(|route| route.referenced_values()).collect::<Vec<_>>();
    super::super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph, route_values);
    Ok(())
}

fn materialize_stage_prelude(
    program: ResourcesAllocated,
    entry_index: usize,
    insertion_site: Option<SideEffectSite>,
    recipe: ProjectedValueRecipe,
    outputs: Vec<StagePreludeOutput>,
) -> ResourcesAllocated {
    if outputs.is_empty() {
        return program;
    }
    let Program {
        functions,
        externs,
        mut entry_points,
        constants,
        mut data,
        mut global_context,
        state: _,
    } = program;
    let ProjectedValueRecipe {
        projection,
        result_block,
        source,
        ..
    } = recipe;
    let materialization = data.materializations.alloc_id();
    let producer_effects = projection.source_effects().clone();
    let producer_entry = {
        let entry = &entry_points[entry_index];
        let producer_resources = entry.resources_referenced_by_projection(&projection);
        projected_materialization_entry(
            &mut data.core.identities,
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
            let resource = data.alloc_compiler_resource(
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
            &mut global_context.effect_ids,
        );
    }
    producer_entry.compact_interface();

    let entry = &mut entry_points[entry_index];
    let mut loaded_values = Vec::with_capacity(handoffs.len());
    let mut load_effects = Vec::with_capacity(handoffs.len());
    for (resource, value) in &handoffs {
        let view = entry.declare_resource_view(*resource, StorageRole::Input, &value.elem_ty, &value.size);
        let (loaded, load_effect) = detached_scalar_handoff_load(
            &mut entry.graph,
            view,
            &value.elem_ty,
            &mut global_context.effect_ids,
        );
        entry.graph.replace_value_references(value.source, loaded);
        loaded_values.push(loaded);
        load_effects.push(load_effect);
    }
    let loaded_primary = loaded_values[0];
    match source {
        ValueRecipeSource::EntryBlock => {
            if let Some(insertion_site) = insertion_site {
                replace_prelude_effects_with_load(entry, &producer_effects, insertion_site, load_effects);
            } else {
                replace_entry_prelude_with_load(entry, &producer_effects, load_effects);
            }
        }
        ValueRecipeSource::StructuredPrefix { continuation } => replace_structured_prefix_with_load(
            entry,
            &producer_effects,
            continuation,
            loaded_primary,
            load_effects,
        ),
    }
    refresh_resource_reads_for_values(&mut entry.graph, &loaded_values);
    let route_values = entry.routes().flat_map(|route| route.referenced_values()).collect::<Vec<_>>();
    super::super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph, route_values);
    entry.compact_interface();

    data.materializations.insert(
        materialization,
        MaterializationRequirement::Scalar {
            entry: producer_entry,
        },
    );
    Program::from_parts(functions, externs, entry_points, constants, data, global_context)
}

fn projected_materialization_entry(
    identities: &mut crate::egir::program::ProgramIdentities,
    materialization: MaterializationId,
    source: &Entry<Semantic>,
    name_suffix: &str,
    execution_model: ExecutionModel,
    resource_declarations: Vec<SemanticResourceDecl>,
    projection: GraphProjection,
) -> Entry {
    let name = materialization.entry_name(&source.name, name_suffix);
    let id = identities.alloc_entry(name.clone());
    Entry {
        id,
        name,
        span: source.span,
        execution_model,
        inputs: source.inputs.clone(),
        parameter_inputs: source.parameter_inputs.clone(),
        outputs: Vec::new(),
        internal_results: Vec::new(),
        resource_declarations,
        params: source.params.clone(),
        result: super::super::types::by_value_function_result::<WynLanguage>(Type::Constructed(
            TypeName::Unit,
            vec![],
        )),
        graph: projection.graph,
    }
}

fn emit_scalar_handoff_store(
    graph: &mut EGraph,
    block: BlockId,
    output_view: ValueId,
    value: ValueId,
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
    view: ValueId,
    elem_ty: &Type<TypeName>,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> (ValueId, SideEffect) {
    let zero = graph_ops::intern_u32(graph, 0, None);
    let view = graph.view_id(view);
    let place = graph.add_view_index_place(view, zero, elem_ty.clone(), None);
    graph_ops::detached_load(graph, place, elem_ty.clone(), effect_ids, None)
}

fn replace_prelude_effects_with_load(
    entry: &mut Entry<Semantic>,
    producer_effects: &HashSet<SideEffectSite>,
    insertion_site: SideEffectSite,
    load_effects: Vec<SideEffect>,
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
    entry: &mut Entry<Semantic>,
    producer_effects: &HashSet<SideEffectSite>,
    load_effects: Vec<SideEffect>,
) {
    entry.graph.skeleton.remove_effect_sites(producer_effects.iter().copied());
    let block = &mut entry.graph.skeleton.blocks[entry.graph.skeleton.entry];
    for (index, load) in load_effects.into_iter().enumerate() {
        block.side_effects.insert(index, load);
    }
}

fn replace_structured_prefix_with_load(
    entry: &mut Entry<Semantic>,
    producer_effects: &HashSet<SideEffectSite>,
    continuation: BlockId,
    loaded: ValueId,
    load_effects: Vec<SideEffect>,
) {
    entry.graph.skeleton.remove_effect_sites(producer_effects.iter().copied());
    let source_entry = entry.graph.skeleton.entry;
    entry.graph.skeleton.blocks[source_entry].side_effects.extend(load_effects);
    entry.graph.skeleton.blocks[source_entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: entry.graph.admit_flow_values([loaded]),
    };
    let aliases = super::super::skel_opt::run_one_body(&mut entry.graph);
    entry.graph.install_aliases(aliases);
    entry.retain_live_control_headers();
}

fn output_specs(
    result: &ResultBinding<Type<TypeName>>,
    materialization: FixedMaterializationKind,
    space: &SegSpace,
    op: &screma::Op<Semantic>,
) -> Option<Vec<OutputSpec>> {
    if op.result_count() != result.field_count() {
        return None;
    }
    (0..op.result_count())
        .map(|field| {
            let field_result = result.field(field)?;
            field_result.single_value()?;
            let elem_ty = op.form.result_element_type(field)?.clone();
            let storage = match op.form.result_id(field)? {
                screma::ResultId::Reduction { .. } => OutputStorage::Scalar,
                screma::ResultId::Post(_) if !materialization.is_scalar() => OutputStorage::Array,
                screma::ResultId::Post(_) => return None,
            };
            let size = match storage {
                OutputStorage::Scalar => {
                    LogicalSize::FixedBytes(u64::from(crate::ssa::layout::storage_elem_stride(&elem_ty)?))
                }
                OutputStorage::Array => LogicalSize::for_space(space, &elem_ty)?,
            };
            Some(OutputSpec {
                field,
                storage,
                size,
                elem_ty,
            })
        })
        .collect()
}

fn refresh_resource_reads_for_values(graph: &mut EGraph, values: &[ValueId]) {
    let mut sites = Vec::<SideEffectSite>::new();
    for (block_id, block) in &graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            if matches!(&effect.kind, SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) if matches!(op.semantic_state(), screma::SemanticState::Segmented { .. }))
                && graph_ops::effect_value_inputs(graph, effect)
                    .into_iter()
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
            super::super::semantic_graph::read_resources(graph, effect)
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
    let effect = graph.skeleton.get_effect(root)?;
    let closure = graph_ops::value_producer_closure(graph, graph_ops::effect_value_inputs(graph, effect));
    if closure.effects.iter().any(|site| site.block != root.block) {
        return None;
    }
    Some(closure.effects.into_iter().map(|site| site.index).chain([root.index]).collect())
}

fn retarget_input_metadata(graph: &mut EGraph, replacements: &[InputReplacement]) -> Result<(), String> {
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            match &mut effect.kind {
                SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                    let mut new_resources = Vec::new();
                    let mut domain_input = None;
                    for (input, input_type) in op.inputs.iter_mut().enumerate() {
                        if let Some(replacement) = replacements
                            .iter()
                            .find(|replacement| effect.operands[input].value() == Some(replacement.project))
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
                }
                SideEffectKind::Soac(SoacEffect(_, Soac::Filter(filter::Op { body, state }))) => {
                    let mut domain_input = None;
                    if let Some(replacement) = replacements
                        .iter()
                        .find(|replacement| effect.operands[0].value() == Some(replacement.project))
                    {
                        let Some(input) = body.inputs.first_mut() else {
                            continue;
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
    Ok(())
}

fn replace_space_references(space: &mut SegSpace, replacements: &[InputReplacement]) {
    for replacement in replacements {
        space.replace_reference(
            replacement.project,
            replacement.view.value(),
            SemanticResourceRef(replacement.resource),
        );
    }
}
