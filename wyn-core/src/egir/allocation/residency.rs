//! Semantic residency planning for arrays and cross-dispatch scalars.
//!
//! The residency fixpoint recognizes shared producers, runtime gathers,
//! invariant scalar reductions, and cost-eligible preludes of parallel
//! operations after output realization and semantic fusion. Each iteration
//! rebuilds its analyses, selects at most one materialization in priority
//! order, applies that rewrite, and restarts. Target lowering only chooses and
//! schedules the physical kernel recipe.

use crate::egir;
use crate::egir::ir::BodySite;
use crate::egir::soac::SegmentedMetadata;
use crate::ssa;
use crate::types;
use std::collections::HashSet;
use wyn_base::IdSource;

use polytype::Type;

use super::super::analysis::GraphAnalysis;
use super::super::from_tlc::ConvertError;
use super::super::graph_ops;
use super::super::graph_projector::{
    GraphProjection, GraphProjector, ProjectedValueRecipe, ValueRecipeSource,
};
use super::super::program::{
    AllocatedEntry, CompilerResource, CompilerResourceKind, GeneratedStageKind, LogicalSize, OutputWriter,
    RealizedOutputRoute, ResidentStorage, ResourceId, SemanticOpId, SemanticResourceDecl,
    SemanticResourceRef, SlotSource, StageOrigin,
};
use super::super::semantic_graph::{SemanticGraph, SourceValue};
use super::super::soac::{filter, screma};
use super::super::stage_variance::StageDependenceAnalysis;
use super::super::types::{
    EGraph, EffectToken, PureOp, ResourceAccess, ResultBinding, SegExtent, SegResourceAccess, SegSpace,
    Semantic as SemanticFamily, SideEffect, SideEffectKind, SideEffectSite, SkeletonTerminator, Soac,
    SoacEffect, ValueId, ValueKind, ViewId, WynLanguage,
};
use super::ResidencyDraft;
use crate::ast::TypeName;
use crate::flow::{BlockId, ExecutionModel};
use crate::interface::StorageRole;
use crate::pipeline_descriptor::{DispatchSize, Pipeline, StorageTextureSize};
use crate::types::TypeExt;
use crate::PipelineTopologyPolicy;
use wyn_staged_ir::StageId;

#[cfg(test)]
#[path = "residency_tests.rs"]
mod residency_tests;

type AllocatedSemantic = SemanticFamily<SemanticResourceRef>;
type AllocatedGraph = EGraph<AllocatedSemantic>;
type AllocatedSideEffect = SideEffect<AllocatedSemantic>;

/// Entry-local facts for one immutable stage snapshot. Callable cost and
/// dependence analysis remain with their respective planners.
fn residency_facts(program: &ResidencyDraft) -> (Vec<GraphAnalysis<'_, AllocatedSemantic>>, SemanticGraph) {
    let entries = program
        .data
        .stages
        .stages()
        .map(|(_, _, entry)| GraphAnalysis::new(&entry.graph))
        .collect::<Vec<_>>();
    let dependencies = SemanticGraph::for_bodies(
        entries.iter().enumerate().map(|(index, analysis)| (BodySite::Entry(index), analysis)),
    );
    (entries, dependencies)
}

enum OperationMaterializationPlan {
    FixedOperation {
        entry: StageId,
        kind: FixedMaterializationKind,
        operation: ProjectedOperation,
        outputs: Vec<OutputSpec>,
    },
    RuntimeArray {
        entry: StageId,
        operation: ProjectedOperation,
        output: RuntimeArrayOutput,
    },
}

struct StagePreludePlan {
    entry: StageId,
    edit: PreludeEdit,
    recipe: ProjectedValueRecipe<SemanticResourceRef>,
    outputs: Vec<OutputSpec>,
    producer_resources: HashSet<ResourceId>,
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
    producer: SemanticOpId,
    source_site: SideEffectSite,
    projected_site: SideEffectSite,
    projection: GraphProjection<SemanticResourceRef>,
    space: SegSpace<SemanticResourceRef>,
    producer_resources: HashSet<ResourceId>,
}

struct RuntimeArrayOutput {
    /// Variable-cardinality array represented by capacity storage plus a
    /// separately stored logical length.
    backing: Option<ResourceId>,
    length: Option<ResourceId>,
    source: ValueId,
    elem_ty: Type<TypeName>,
    result_ty: Type<TypeName>,
    size: LogicalSize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputStorage {
    Array,
    Scalar,
}

struct OutputSpec {
    source: ValueId,
    projected: ResultBinding<Type<TypeName>>,
    routed: Option<ResourceId>,
    storage: OutputStorage,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
}

struct BoundOutput {
    resource: ResourceId,
    spec: OutputSpec,
}

impl OutputSpec {
    fn bind(
        self,
        resources: &mut super::super::program::LogicalResourceArena,
        kind: CompilerResourceKind,
        owner: Option<SemanticOpId>,
        slot: usize,
    ) -> BoundOutput {
        let resource = self.routed.unwrap_or_else(|| {
            resources.allocate_compiler(
                CompilerResource::new(
                    if self.storage == OutputStorage::Scalar {
                        CompilerResourceKind::ScalarHandoff
                    } else {
                        kind
                    },
                    owner,
                    slot,
                ),
                self.elem_ty.clone(),
                self.size.clone(),
            )
        });
        BoundOutput { resource, spec: self }
    }
}

enum PreludeEdit {
    Before(SideEffectSite),
    Entry,
    Structured {
        continuation: BlockId,
        primary: ValueId,
    },
}

impl StagePreludePlan {
    fn prepare(
        program: &ResidencyDraft,
        entry: &AllocatedEntry,
        analysis: &GraphAnalysis<'_, AllocatedSemantic>,
        stage: StageId,
        recipe: ProjectedValueRecipe<SemanticResourceRef>,
        insertion_site: Option<SideEffectSite>,
        invocations: u64,
    ) -> Option<Self> {
        let outputs = recipe
            .projection
            .output_sources()
            .map(|source| {
                let elem_ty = entry.graph.nodes[source].ty.clone();
                let size = LogicalSize::FixedBytes(u64::from(ssa::layout::storage_elem_stride(&elem_ty)?));
                Some(OutputSpec {
                    source,
                    projected: recipe.projection.graph.value_result(recipe.projection.node(source)?),
                    routed: None,
                    storage: OutputStorage::Scalar,
                    elem_ty,
                    size,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let edit = match recipe.source {
            ValueRecipeSource::EntryBlock => insertion_site.map_or(PreludeEdit::Entry, PreludeEdit::Before),
            ValueRecipeSource::StructuredPrefix { continuation } => PreludeEdit::Structured {
                continuation,
                primary: outputs.first()?.source,
            },
        };
        if !super::cost::should_materialize_prelude(program, entry, &recipe, invocations)? {
            return None;
        }
        Some(Self {
            entry: stage,
            edit,
            producer_resources: entry.resources_referenced_by_projection(analysis, &recipe.projection),
            recipe,
            outputs,
        })
    }
}

struct InputReplacement {
    project: ValueId,
    view: ViewId,
    view_ty: Type<TypeName>,
    resource: ResourceId,
    elem_bytes: u32,
}

impl InputReplacement {
    fn new(graph: &AllocatedGraph, project: ValueId, view: ValueId) -> Result<Self, String> {
        let resource = graph_ops::extract_storage_view_source(graph, view)
            .ok_or("resident input is not a storage-backed view")?
            .0;
        let view_ty = graph.nodes[view].ty.clone();
        let elem_ty = view_ty.elem_type().ok_or("resident input is not an array view")?;
        let elem_bytes =
            ssa::layout::storage_elem_stride(elem_ty).ok_or("resident input has a non-storable element")?;
        Ok(Self {
            project,
            view: graph.view_id(view),
            view_ty,
            resource,
            elem_bytes,
        })
    }
}

pub fn resolve_residency(program: ResidencyDraft) -> Result<ResidencyDraft, String> {
    resolve_residency_with_policy(program, PipelineTopologyPolicy::AllowGenerated).map_err(|error| {
        match error {
            ConvertError::Internal(message) => message,
            error => error.to_string(),
        }
    })
}

pub(super) fn resolve_residency_with_policy(
    mut program: ResidencyDraft,
    topology: PipelineTopologyPolicy,
) -> Result<ResidencyDraft, ConvertError> {
    loop {
        // Required handoffs take priority over optional preludes. Every rewrite
        // restarts this read phase with fresh facts for the new graph.
        let (analyses, dependencies) = residency_facts(&program);
        let operation =
            plan_required_residency(&program, &dependencies, &analyses).map_err(ConvertError::Internal)?;
        if let Some(plan) = operation {
            if topology == PipelineTopologyPolicy::AuthoredOnly {
                return Err(ConvertError::PipelineTopology(
                    "authored-only lowering cannot represent an operation result that requires a compiler-created stage or handoff resource".into(),
                ));
            }
            program = match plan {
                OperationMaterializationPlan::FixedOperation {
                    entry,
                    kind,
                    operation,
                    outputs,
                } => materialize_operation_result(program, entry, kind, operation, outputs),
                OperationMaterializationPlan::RuntimeArray {
                    entry,
                    operation,
                    output,
                } => materialize_runtime_array_result(program, entry, operation, output),
            }
            .map_err(ConvertError::Internal)?;
            continue;
        }
        if topology == PipelineTopologyPolicy::AuthoredOnly {
            break;
        }
        let Some(plan) = select_stage_prelude_candidate(&program, &dependencies, &analyses) else {
            break;
        };
        program = materialize_stage_prelude(program, plan).map_err(ConvertError::Internal)?;
    }
    Ok(program)
}
fn select_stage_prelude_candidate(
    program: &ResidencyDraft,
    dependencies: &SemanticGraph,
    analyses: &[GraphAnalysis<'_, AllocatedSemantic>],
) -> Option<StagePreludePlan> {
    plan_parallel_prelude(program, dependencies, analyses)
        .or_else(|| plan_direct_stage_prelude(program, analyses))
}

fn plan_required_residency(
    program: &ResidencyDraft,
    dependencies: &SemanticGraph,
    analyses: &[GraphAnalysis<'_, AllocatedSemantic>],
) -> Result<Option<OperationMaterializationPlan>, String> {
    let mut scalar_candidates = Vec::new();
    for (entry_index, (stage, _, entry)) in program.data.stages.stages().enumerate() {
        let analysis = &analyses[entry_index];
        let mut reductions = Vec::new();
        for (block_id, block) in &entry.graph.skeleton.blocks {
            for (effect_index, effect) in block.side_effects.iter().enumerate() {
                let Some(result) = effect.result.as_ref() else {
                    continue;
                };
                let Some(&id) = effect.kind.soac_id() else {
                    continue;
                };
                let semantic_consumers = dependencies.value_consumers(&id).collect::<HashSet<_>>();
                let source_site = SideEffectSite {
                    block: block_id,
                    index: effect_index,
                };
                match &effect.kind {
                    SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                        if op.form.post.result_types.is_empty() {
                            reductions.push((source_site, id, op, result, semantic_consumers));
                            continue;
                        }
                        let Some(kind) = array_result_residency(
                            result,
                            &semantic_consumers,
                            dependencies.array_residency_demands.contains(&id),
                        ) else {
                            continue;
                        };
                        let Some(plan) = operation_result_plan(
                            stage,
                            entry,
                            analysis,
                            op,
                            result,
                            id,
                            source_site,
                            kind,
                        )?
                        else {
                            continue;
                        };
                        return Ok(Some(plan));
                    }
                    SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) => {
                        if let Some(plan) = filter_runtime_array_plan(
                            dependencies,
                            stage,
                            entry,
                            analysis,
                            op,
                            result,
                            id,
                            source_site,
                            &semantic_consumers,
                        )? {
                            return Ok(Some(plan));
                        }
                    }
                    _ => {}
                }
            }
        }
        scalar_candidates.push((stage, entry, analysis, reductions));
    }
    // Structural residency has global priority. Defer scalar legality and
    // projection until every structural candidate has been exhausted, including
    // candidates in later entries. Each entry shares one lazy invariance check.
    for (stage, entry, analysis, reductions) in scalar_candidates {
        let invariant = std::cell::OnceCell::new();
        for (source_site, id, op, result, semantic_consumers) in reductions {
            if !scalar_result_requires_handoff(
                dependencies,
                entry,
                analysis,
                &invariant,
                op,
                result,
                source_site,
                &semantic_consumers,
            ) {
                continue;
            }
            let Some(plan) = operation_result_plan(
                stage,
                entry,
                analysis,
                op,
                result,
                id,
                source_site,
                FixedMaterializationKind::Scalar,
            )?
            else {
                continue;
            };
            return Ok(Some(plan));
        }
    }
    Ok(None)
}

fn filter_runtime_array_plan(
    dependencies: &SemanticGraph,
    entry_index: StageId,
    entry: &AllocatedEntry,
    analysis: &GraphAnalysis<'_, AllocatedSemantic>,
    op: &filter::Op<AllocatedSemantic>,
    result: &ResultBinding<Type<TypeName>>,
    producer: SemanticOpId,
    source_site: SideEffectSite,
    consumers: &HashSet<SemanticOpId>,
) -> Result<Option<OperationMaterializationPlan>, String> {
    let filter::SemanticState {
        segment: SegmentedMetadata { space, .. },
        output: filter::Output::Runtime(runtime),
        ..
    } = &op.state
    else {
        return Ok(None);
    };
    if !has_parallel_consumer(entry, dependencies, consumers) {
        return Ok(None);
    }
    let elem_ty = op.body.output_element_type().clone();
    let result_ty = result.ty().clone();
    let projection = GraphProjector::new(analysis)
        .selected_operation_recipe(HashSet::from([source_site]))
        .map_err(|error| format!("could not project runtime-array producer {producer:?}: {error}"))?;
    let projected_site = projection
        .effect_site(source_site)
        .ok_or_else(|| format!("runtime-array projection omitted producer site for {producer:?}"))?;
    let source = result.single_value().ok_or("runtime-array materialization requires one result value")?;
    projection.node(source).ok_or("runtime-array projection omitted its result")?;
    let size = super::filter_capacity_size(producer, space, &elem_ty)?;
    Ok(Some(OperationMaterializationPlan::RuntimeArray {
        entry: entry_index,
        operation: ProjectedOperation {
            producer,
            source_site,
            projected_site,
            producer_resources: entry.resources_referenced_by_projection(analysis, &projection),
            projection,
            space: space.clone(),
        },
        output: RuntimeArrayOutput {
            source,
            backing: match runtime.backing {
                filter::RuntimeBacking::Deferred => None,
                filter::RuntimeBacking::Bound(resource) => Some(resource.0),
            },
            length: match runtime.length {
                filter::RuntimeLength::Implicit => None,
                filter::RuntimeLength::Stored(resource) => Some(resource.0),
            },
            size,
            elem_ty,
            result_ty,
        },
    }))
}

fn scalar_result_requires_handoff(
    dependencies: &SemanticGraph,
    entry: &AllocatedEntry,
    analysis: &GraphAnalysis<'_, AllocatedSemantic>,
    invariant: &std::cell::OnceCell<bool>,
    op: &screma::Op<AllocatedSemantic>,
    result: &ResultBinding<Type<TypeName>>,
    site: SideEffectSite,
    consumers: &HashSet<SemanticOpId>,
) -> bool {
    if !op.form.post.result_types.is_empty()
        || !op.is_reduce()
        || op.form.reductions.len() != 1
        || !(has_segmented_screma_consumer(entry, dependencies, consumers)
            || !entry.execution_model.is_compute())
        || !result.single_value().is_some_and(|value| scalar_result_is_used(analysis.slice(), value, site))
    {
        return false;
    }
    *invariant.get_or_init(|| invocation_invariant(entry, analysis))
}

fn array_result_residency(
    result: &ResultBinding<Type<TypeName>>,
    consumers: &HashSet<SemanticOpId>,
    requires_array_storage: bool,
) -> Option<FixedMaterializationKind> {
    if consumers.len() >= 2 {
        Some(FixedMaterializationKind::SharedArray)
    } else if result.ty().contains_runtime_sized_composite_array() && requires_array_storage {
        Some(FixedMaterializationKind::Gather)
    } else {
        None
    }
}

fn operation_result_plan(
    entry_index: StageId,
    entry: &AllocatedEntry,
    analysis: &GraphAnalysis<'_, AllocatedSemantic>,
    op: &screma::Op<AllocatedSemantic>,
    result: &ResultBinding<Type<TypeName>>,
    producer: SemanticOpId,
    source_site: SideEffectSite,
    kind: FixedMaterializationKind,
) -> Result<Option<OperationMaterializationPlan>, String> {
    let screma::SemanticState::Segmented(SegmentedMetadata { space, resources, .. }) = op.semantic_state()
    else {
        return Ok(None);
    };
    if !op.result_state.iter().all(|result| result.ownership == types::SoacOwnership::Fresh)
        || !resources.iter().all(|resource| {
            resource.access == ResourceAccess::Read
                || entry
                    .outputs
                    .iter()
                    .filter_map(|output| output.resource)
                    .any(|output| output == resource.resource)
        })
    {
        return Ok(None);
    }
    let projection =
        match GraphProjector::new(analysis).selected_operation_recipe(HashSet::from([source_site])) {
            Ok(projection) => projection,
            Err(_) => {
                // Projection feasibility is part of the materialization policy:
                // a producer depending on a loop/selection boundary parameter
                // cannot become an entry prepass and remains in its source graph.
                return Ok(None);
            }
        };
    // Projection already checked the complete producer closure, block-parameter
    // boundary, and retained observers. Apply cloneability to that selection.
    if !projection.source_effects().iter().filter(|site| **site != source_site).all(|site| {
        matches!(&entry.graph.skeleton.effect(*site).kind,
            SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op)))
                if matches!(op.semantic_state(), screma::SemanticState::Segmented(SegmentedMetadata { output_slots, resources, .. })
                    if output_slots.is_empty()
                        && op.result_state.iter().all(|result| result.ownership == types::SoacOwnership::Fresh)
                        && resources.iter().all(|resource| resource.access == ResourceAccess::Read)))
    }) {
        return Ok(None);
    }
    let output_specs = output_specs(entry, result, kind, space, op, &projection)
        .ok_or_else(|| format!("materialization producer {producer:?} has an unsupported output layout"))?;
    let projected_site = projection
        .effect_site(source_site)
        .ok_or_else(|| format!("materialization projection omitted producer site for {producer:?}"))?;
    Ok(Some(OperationMaterializationPlan::FixedOperation {
        entry: entry_index,
        kind,
        operation: ProjectedOperation {
            producer,
            source_site,
            projected_site,
            producer_resources: entry.resources_referenced_by_projection(analysis, &projection),
            projection,
            space: space.clone(),
        },
        outputs: output_specs,
    }))
}

fn plan_parallel_prelude(
    program: &ResidencyDraft,
    dependencies: &SemanticGraph,
    analyses: &[GraphAnalysis<'_, AllocatedSemantic>],
) -> Option<StagePreludePlan> {
    for (entry_index, (stage, _, entry)) in program.data.stages.stages().enumerate() {
        for (root, consumers) in parallel_preludes(entry, dependencies, BodySite::Entry(entry_index)) {
            let ty = &entry.graph.nodes[root].ty;
            if ssa::layout::storage_elem_stride(ty).is_none() {
                continue;
            }
            if ty.is_array() {
                continue;
            }
            let Some(consumer_sites) = operation_sites(&dependencies, &consumers) else {
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
            let projector = GraphProjector::new(&analyses[entry_index]);
            if !source_is_observed_only_by_consumers_or_outputs(
                entry,
                analyses[entry_index].slice(),
                root,
                &consumer_site_set,
            ) {
                continue;
            }
            let Some(insertion_site) = consumer_sites.iter().min_by_key(|site| site.index).copied() else {
                continue;
            };
            let Ok(recipe) = projector.captured_value_recipe_with_retained_values(
                root,
                insertion_site,
                entry.routes().map(|route| route.source.value),
            ) else {
                continue;
            };
            let invocations = launched_consumer_invocations(entry, &consumer_sites);
            if let Some(plan) = StagePreludePlan::prepare(
                program,
                entry,
                &analyses[entry_index],
                stage,
                recipe,
                Some(insertion_site),
                invocations,
            ) {
                return Some(plan);
            }
        }
    }
    None
}

/// Dynamic draw and dispatch sizes are unavailable during shader compilation.
/// Price those direct stages against one modest batch so only substantial
/// uniform work clears the singleton-launch overhead.
const DIRECT_STAGE_INVOCATION_FALLBACK: u64 = 64;

fn plan_direct_stage_prelude(
    program: &ResidencyDraft,
    analyses: &[GraphAnalysis<'_, AllocatedSemantic>],
) -> Option<StagePreludePlan> {
    for (entry_index, (stage, _, entry)) in program.data.stages.stages().enumerate() {
        let Ok(analysis) = StageDependenceAnalysis::for_entry(entry, &analyses[entry_index]) else {
            continue;
        };
        let frontier = graph_ops::maximal_execution_frontier(&entry.graph, |node| {
            direct_stage_value_is_liftable(entry, &analysis, node)
        });
        if frontier.is_empty() {
            continue;
        }
        let Ok(recipe) = GraphProjector::new(&analyses[entry_index])
            .entry_values_recipe_with_retained_values(
                frontier.iter().copied(),
                entry.routes().map(|route| route.source.value),
            )
        else {
            continue;
        };
        if let Some(plan) = StagePreludePlan::prepare(
            program,
            entry,
            &analyses[entry_index],
            stage,
            recipe,
            None,
            direct_stage_invocations(program, entry),
        ) {
            return Some(plan);
        }
    }
    None
}

fn direct_stage_invocations(program: &ResidencyDraft, entry: &AllocatedEntry) -> u64 {
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
    entry: &AllocatedEntry,
    analysis: &StageDependenceAnalysis,
    node: ValueId,
) -> bool {
    let Some(definition) = entry.graph.nodes.get(node) else {
        return false;
    };
    let ValueKind::Pure { op, .. } = &definition.kind else {
        return false;
    };
    if matches!(op, PureOp::Project { .. }) {
        return false;
    }
    let dependence = analysis.dependence(node);
    let ty = &definition.ty;
    dependence.is_stage_invariant()
        && !dependence.is_compile_time_constant()
        && dependence.loop_dependencies().is_empty()
        && !ty.is_array()
        && ssa::layout::storage_elem_stride(ty).is_some()
}

fn parallel_preludes(
    entry: &AllocatedEntry,
    dependencies: &SemanticGraph,
    body: BodySite,
) -> crate::StableMap<ValueId, Vec<SemanticOpId>> {
    let mut preludes = crate::StableMap::<ValueId, Vec<SemanticOpId>>::new();
    for capture in dependencies.captured_values(body) {
        for operation in dependencies.capture_consumers(SourceValue { body, value: capture }) {
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
            let consumers = preludes.entry(root).or_default();
            if !consumers.contains(&operation) {
                consumers.push(operation);
            }
        }
    }
    preludes
}

fn parallel_prelude_boundary_root(
    entry: &AllocatedEntry,
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

fn supports_parallel_prefix_consumer(entry: &AllocatedEntry, site: SideEffectSite) -> bool {
    matches!(
        &entry.graph.skeleton.effect(site).kind,
        SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op)))
            if op.is_map()
                && op.form.post.is_identity()
                && !op.form.post.result_types.is_empty()
                && matches!(op.semantic_state(), screma::SemanticState::Segmented(_))
    )
}

fn source_is_observed_only_by_consumers_or_outputs(
    entry: &AllocatedEntry,
    uses: &graph_ops::SliceFacts,
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

fn launched_consumer_invocations(entry: &AllocatedEntry, consumers: &[SideEffectSite]) -> u64 {
    let workgroup = match &entry.execution_model {
        ExecutionModel::Compute { local_size } => u64::from(local_size.0)
            .saturating_mul(u64::from(local_size.1))
            .saturating_mul(u64::from(local_size.2))
            .max(1),
        ExecutionModel::Vertex | ExecutionModel::Fragment => 1,
    };
    consumers.iter().fold(0u64, |total, &site| {
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
    entry: &AllocatedEntry,
    dependencies: &SemanticGraph,
    consumers: &HashSet<SemanticOpId>,
) -> bool {
    has_matching_consumer(entry, dependencies, consumers, |soac| {
        soac.scheduling_space().is_some()
    })
}

fn has_segmented_screma_consumer(
    entry: &AllocatedEntry,
    dependencies: &SemanticGraph,
    consumers: &HashSet<SemanticOpId>,
) -> bool {
    has_matching_consumer(entry, dependencies, consumers, |soac| {
        matches!(
            soac,
            Soac::Screma(op)
                if matches!(op.semantic_state(), screma::SemanticState::Segmented(_))
        )
    })
}

fn has_matching_consumer(
    entry: &AllocatedEntry,
    dependencies: &SemanticGraph,
    consumers: &HashSet<SemanticOpId>,
    mut supports: impl FnMut(&Soac<AllocatedSemantic>) -> bool,
) -> bool {
    consumers.iter().filter_map(|id| dependencies.operation_site(id)).any(|site| {
        matches!(&entry.graph.skeleton.effect(site).kind, SideEffectKind::Soac(SoacEffect(_, soac)) if supports(soac))
    })
}

fn scalar_result_is_used(uses: &graph_ops::SliceFacts, result: ValueId, producer: SideEffectSite) -> bool {
    let observers = uses.value_observers(result);
    observers.effect_sites().any(|site| site != producer) || observers.terminator_blocks().next().is_some()
}

fn invocation_invariant(entry: &AllocatedEntry, analysis: &GraphAnalysis<'_, AllocatedSemantic>) -> bool {
    let Ok(dependence) = StageDependenceAnalysis::for_entry(entry, analysis) else {
        return false;
    };
    let Ok(slice) = analysis.slice().select(
        graph_ops::execution_value_roots(&entry.graph),
        [],
        [],
        |_| true,
        |_| false,
        &[],
    ) else {
        return false;
    };
    slice.values().iter().copied().all(|node| {
        let Some(ValueKind::FuncParam { parameter }) = entry.graph.nodes.get(node).map(|node| &node.kind)
        else {
            return true;
        };
        dependence.dependence(node).is_stage_invariant()
            && entry
                .params()
                .abi_position(*parameter)
                .is_some_and(|position| super::cost::entry_parameter_is_scalar_relocatable(entry, position))
    })
}

fn materialize_operation_result(
    mut program: ResidencyDraft,
    entry_index: StageId,
    kind: FixedMaterializationKind,
    operation: ProjectedOperation,
    outputs: Vec<OutputSpec>,
) -> Result<ResidencyDraft, String> {
    let data = &mut program.data;
    let effect_ids = &mut program.global_context.effect_ids;
    let ProjectedOperation {
        producer,
        source_site,
        projected_site,
        projection,
        space,
        producer_resources,
    } = operation;
    let entry = data.stages.stage_body(entry_index).expect("planned consumer stage");
    let source_outputs =
        entry.outputs.iter().filter_map(|output| output.resource.map(|resource| resource.0)).collect();
    let (name, resource_kind, generated_kind) = match kind {
        FixedMaterializationKind::SharedArray => (
            "materialize_shared",
            CompilerResourceKind::MultiConsumerArray,
            GeneratedStageKind::SharedArray,
        ),
        FixedMaterializationKind::Gather => (
            "gather_materialize",
            CompilerResourceKind::GatherHandoff,
            GeneratedStageKind::Gather,
        ),
        FixedMaterializationKind::Scalar => (
            "prepass_scalar",
            CompilerResourceKind::ScalarHandoff,
            GeneratedStageKind::Scalar,
        ),
    };
    let mut producer_entry = projected_materialization_entry(
        &mut data.core.identities,
        data.stages.stage_count(),
        entry,
        name,
        materialization_execution_model(entry),
        entry.resource_declarations_for(&producer_resources),
        projection,
    );
    if !entry.execution_model.is_compute() {
        producer_entry.compact_interface();
    }
    let outputs = outputs
        .into_iter()
        .enumerate()
        .map(|(slot, output)| output.bind(&mut data.core.resources, resource_kind, Some(producer), slot))
        .collect::<Vec<_>>();
    configure_operation_materialization(
        &mut producer_entry,
        projected_site,
        &outputs,
        &source_outputs,
        effect_ids,
    )?;
    rewrite_materialized_operation_source(
        data.stages.stage_body_mut(entry_index).expect("planned consumer stage"),
        source_site,
        &outputs,
        effect_ids,
    )?;
    let producer = data
        .stages
        .add_stage(
            StageOrigin::Generated {
                kind: generated_kind,
                space: (!kind.is_scalar()).then_some(space),
            },
            producer_entry,
        )
        .map_err(|error| error.to_string())?;
    for output in outputs {
        data.connect_resident_flow(
            producer,
            entry_index,
            output.spec.projected.ty().clone(),
            ResidentStorage {
                data: output.resource,
                length: None,
            },
        )?;
    }
    Ok(program)
}

fn materialization_execution_model(entry: &AllocatedEntry) -> ExecutionModel {
    match &entry.execution_model {
        ExecutionModel::Compute { local_size } => ExecutionModel::Compute {
            local_size: *local_size,
        },
        ExecutionModel::Vertex | ExecutionModel::Fragment => ExecutionModel::Compute {
            local_size: (64, 1, 1),
        },
    }
}

fn materialize_runtime_array_result(
    mut program: ResidencyDraft,
    entry_index: StageId,
    operation: ProjectedOperation,
    output: RuntimeArrayOutput,
) -> Result<ResidencyDraft, String> {
    let data = &mut program.data;
    let global_context = &mut program.global_context;
    let ProjectedOperation {
        producer: producer_id,
        source_site,
        projected_site,
        producer_resources,
        projection,
        space,
    } = operation;
    let stage_number = data.stages.stage_count();
    let entry = data.stages.stage_body(entry_index).expect("planned consumer stage");
    let consumer_stage = entry_index;
    let producer_storage = entry.resource_declarations_for(&producer_resources);
    let execution_model = materialization_execution_model(entry);
    let mut producer_entry = projected_materialization_entry(
        &mut data.core.identities,
        stage_number,
        entry,
        "materialize_filter",
        execution_model,
        producer_storage,
        projection,
    );
    let storage = super::bind_filter_storage(
        &mut data.core.resources,
        producer_id,
        output.elem_ty.clone(),
        output.size.clone(),
        output.backing,
        output.length,
    )?;
    producer_entry.set_resource_declaration(storage.data, StorageRole::Output);
    producer_entry.set_resource_declaration(storage.length, StorageRole::Output);
    let effect = producer_entry.graph.skeleton.effect_mut(projected_site);
    let SideEffectKind::Soac(SoacEffect(
        _,
        Soac::Filter(filter::Op {
            state:
                filter::SemanticState {
                    segment:
                        SegmentedMetadata {
                            output_slots,
                            resources,
                            ..
                        },
                    output: filter_output,
                    ..
                },
            ..
        }),
    )) = &mut effect.kind
    else {
        return Err("runtime-array materialization projection did not retain a filter".to_string());
    };
    let filter::Output::Runtime(runtime) = filter_output else {
        return Err("runtime-array materialization projected a local filter".to_string());
    };
    runtime.backing = filter::RuntimeBacking::Bound(SemanticResourceRef(storage.data));
    runtime.length = filter::RuntimeLength::Stored(SemanticResourceRef(storage.length));
    *output_slots = vec![super::super::ir::OutputSlotId(0)];
    *resources = SegResourceAccess::merge(
        resources,
        &[storage.data, storage.length].map(|resource| SegResourceAccess {
            resource: SemanticResourceRef(resource),
            access: ResourceAccess::Write,
        }),
    );
    producer_entry.compact_interface();

    rewrite_runtime_array_source(
        data.stages.stage_body_mut(entry_index).expect("planned consumer stage"),
        source_site,
        &output,
        storage,
        &mut global_context.effect_ids,
    )?;
    let producer_stage = data
        .stages
        .add_stage(
            StageOrigin::Generated {
                kind: GeneratedStageKind::RuntimeArray,
                space: Some(space),
            },
            producer_entry,
        )
        .map_err(|error| error.to_string())?;
    data.connect_resident_flow(
        producer_stage,
        consumer_stage,
        output.result_ty,
        ResidentStorage {
            data: storage.data,
            length: Some(storage.length),
        },
    )?;
    Ok(program)
}

fn rewrite_runtime_array_source(
    entry: &mut AllocatedEntry,
    source_site: SideEffectSite,
    output: &RuntimeArrayOutput,
    storage: filter::RuntimeStorage<ResourceId>,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    let result = output.source;
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    entry.set_resource_declaration(storage.data, StorageRole::Input);
    entry.set_resource_declaration(storage.length, StorageRole::Input);
    let length_view =
        graph_ops::intern_resource_view(&mut entry.graph, storage.length, u32_ty.clone(), None);
    let (survivor_count, load_effect) =
        scalar_handoff_load(&mut entry.graph, length_view, &u32_ty, effect_ids).into_parts();
    let zero = graph_ops::intern_u32(&mut entry.graph, 0, None);
    let view = graph_ops::intern_chunked_resource_view(
        &mut entry.graph,
        storage.data,
        zero,
        survivor_count,
        output.elem_ty.clone(),
        None,
    );
    let replacement = InputReplacement::new(&entry.graph, result, view)?;
    retarget_input_metadata(&mut entry.graph, &[replacement]);
    replace_entry_values(entry, &[(result, view)]);
    entry.graph.retype_node(result, output.result_ty.clone());
    let block = &mut entry.graph.skeleton.blocks[source_site.block];
    block.side_effects.remove(source_site.index);
    block.side_effects.insert(source_site.index, load_effect);
    finish_consumer_rewrite(entry, &[survivor_count, view]);
    entry.compact_interface();
    Ok(())
}

fn configure_operation_materialization(
    producer: &mut AllocatedEntry,
    site: SideEffectSite,
    outputs: &[BoundOutput],
    source_outputs: &HashSet<ResourceId>,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) =
        &mut producer.graph.skeleton.effect_mut(site).kind
    else {
        return Err("fixed materialization projection did not retain a Screma operation".into());
    };
    let screma::SemanticState::Segmented(SegmentedMetadata {
        output_slots,
        resources,
        ..
    }) = op.semantic_state_mut()
    else {
        return Err("fixed materialization Screma was not segmented".into());
    };
    let writes = outputs
        .iter()
        .filter(|output| output.spec.storage == OutputStorage::Array)
        .map(|output| SegResourceAccess {
            resource: SemanticResourceRef(output.resource),
            access: ResourceAccess::Write,
        })
        .collect::<Vec<_>>();
    *output_slots = (0..writes.len()).map(super::super::ir::OutputSlotId).collect();
    resources.retain(|access| {
        access.access == ResourceAccess::Read || !source_outputs.contains(&access.resource.0)
    });
    *resources = SegResourceAccess::merge(resources, &writes);
    for output in outputs {
        let spec = &output.spec;
        let source = spec.projected.single_value().expect("prepared single result");
        let view = producer.declare_resource_view(output.resource, StorageRole::Output, &spec.elem_ty);
        producer.internal_results.push(super::super::ir::InternalResultRoute {
            resource: SemanticResourceRef(output.resource),
            route: RealizedOutputRoute {
                source: SlotSource {
                    block: site.block,
                    value: source,
                },
                writers: vec![OutputWriter::Value(source)],
            },
        });
        if spec.storage == OutputStorage::Array {
            let destination = graph_ops::bind_result_to_view(&mut producer.graph, &spec.projected, view)?;
            let replacements = graph_ops::rebind_result_value_references(
                &mut producer.graph,
                &spec.projected,
                &destination,
            )?;
            for route in producer.routes_mut() {
                route.replace_values(&replacements);
            }
        } else {
            scalar_handoff_store(&mut producer.graph, view, source, &spec.elem_ty, effect_ids)
                .append_to(&mut producer.graph.skeleton, site.block);
        }
    }
    Ok(())
}

fn rewrite_materialized_operation_source(
    entry: &mut AllocatedEntry,
    site: SideEffectSite,
    outputs: &[BoundOutput],
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    let mut arrays = Vec::new();
    let mut replacements = Vec::new();
    let mut effects = Vec::new();
    let mut loaded = Vec::new();
    for output in outputs {
        let spec = &output.spec;
        let view = entry.declare_resource_view(output.resource, StorageRole::Input, &spec.elem_ty);
        let value = if spec.storage == OutputStorage::Scalar {
            let (value, effect) =
                scalar_handoff_load(&mut entry.graph, view, &spec.elem_ty, effect_ids).into_parts();
            effects.push(effect);
            loaded.push(value);
            value
        } else {
            arrays.push(InputReplacement::new(&entry.graph, spec.source, view)?);
            view
        };
        replacements.push((spec.source, value));
    }
    retarget_input_metadata(&mut entry.graph, &arrays);
    replace_entry_values(entry, &replacements);
    for &(source, value) in &replacements {
        entry.graph.retype_node(source, entry.graph.nodes[value].ty.clone());
    }
    entry.graph.skeleton.blocks[site.block].side_effects.splice(site.index..=site.index, effects);
    finish_consumer_rewrite(entry, &loaded);
    Ok(())
}

fn replace_entry_values(entry: &mut AllocatedEntry, replacements: &[(ValueId, ValueId)]) {
    for &(source, value) in replacements {
        entry.graph.replace_value_references(source, value);
    }
    for route in entry.routes_mut() {
        route.replace_values(replacements);
    }
}

fn finish_consumer_rewrite(entry: &mut AllocatedEntry, values: &[ValueId]) {
    refresh_resource_reads_for_values(&mut entry.graph, values);
    let routes = entry.routes().flat_map(|route| route.referenced_values()).collect::<Vec<_>>();
    super::super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph, routes);
}

fn materialize_stage_prelude(
    mut program: ResidencyDraft,
    plan: StagePreludePlan,
) -> Result<ResidencyDraft, String> {
    let StagePreludePlan {
        entry: consumer,
        edit,
        recipe,
        outputs,
        producer_resources,
    } = plan;
    let data = &mut program.data;
    let effect_ids = &mut program.global_context.effect_ids;
    let producer_effects = recipe.projection.source_effects().clone();
    let entry = data.stages.stage_body(consumer).expect("planned consumer stage");
    let mut producer_entry = projected_materialization_entry(
        &mut data.core.identities,
        data.stages.stage_count(),
        entry,
        "prepass_scalar",
        ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        entry.resource_declarations_for(&producer_resources),
        recipe.projection,
    );
    let outputs = outputs
        .into_iter()
        .enumerate()
        .map(|(slot, output)| {
            output.bind(
                &mut data.core.resources,
                CompilerResourceKind::ScalarHandoff,
                None,
                slot,
            )
        })
        .collect::<Vec<_>>();
    for output in &outputs {
        let spec = &output.spec;
        let view =
            producer_entry.declare_resource_view(output.resource, StorageRole::Output, &spec.elem_ty);
        scalar_handoff_store(
            &mut producer_entry.graph,
            view,
            spec.projected.single_value().expect("prepared scalar output"),
            &spec.elem_ty,
            effect_ids,
        )
        .append_to(&mut producer_entry.graph.skeleton, recipe.result_block);
    }
    producer_entry.compact_interface();
    let entry = data.stages.stage_body_mut(consumer).expect("planned consumer stage");
    let mut replacements = Vec::with_capacity(outputs.len());
    let mut loads = Vec::with_capacity(outputs.len());
    for output in &outputs {
        let view = entry.declare_resource_view(output.resource, StorageRole::Input, &output.spec.elem_ty);
        let (value, effect) =
            scalar_handoff_load(&mut entry.graph, view, &output.spec.elem_ty, effect_ids).into_parts();
        replacements.push((output.spec.source, value));
        loads.push(effect);
    }
    replace_entry_values(entry, &replacements);
    match edit {
        PreludeEdit::Before(site) => {
            replace_prelude_effects_with_load(entry, &producer_effects, site, loads)
        }
        PreludeEdit::Entry => replace_entry_prelude_with_load(entry, &producer_effects, loads),
        PreludeEdit::Structured {
            continuation,
            primary,
        } => {
            let loaded = replacements
                .iter()
                .find(|(source, _)| *source == primary)
                .expect("prepared boundary output")
                .1;
            replace_structured_prefix_with_load(entry, &producer_effects, continuation, loaded, loads);
        }
    }
    finish_consumer_rewrite(
        entry,
        &replacements.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
    );
    entry.compact_interface();
    let producer = data
        .stages
        .add_stage(
            StageOrigin::Generated {
                kind: GeneratedStageKind::Scalar,
                space: None,
            },
            producer_entry,
        )
        .map_err(|error| error.to_string())?;
    for output in outputs {
        data.connect_resident_flow(
            producer,
            consumer,
            output.spec.elem_ty,
            ResidentStorage {
                data: output.resource,
                length: None,
            },
        )?;
    }
    Ok(program)
}

fn projected_materialization_entry(
    identities: &mut egir::program::ProgramIdentities,
    stage_number: usize,
    source: &AllocatedEntry,
    name_suffix: &str,
    execution_model: ExecutionModel,
    resource_declarations: Vec<SemanticResourceDecl>,
    projection: GraphProjection<SemanticResourceRef>,
) -> AllocatedEntry {
    let name = format!("{}_{}_{}", source.name, name_suffix, stage_number);
    let id = identities.alloc_entry(name.clone());
    AllocatedEntry {
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

fn scalar_handoff_store(
    graph: &mut AllocatedGraph,
    output_view: ValueId,
    value: ValueId,
    elem_ty: &Type<TypeName>,
    effect_ids: &mut IdSource<EffectToken>,
) -> graph_ops::PendingEffect<AllocatedSemantic, EffectToken> {
    let zero = graph_ops::intern_u32(graph, 0, None);
    let view = graph.view_id(output_view);
    let place = graph.add_view_index_place(view, zero, elem_ty.clone(), None);
    graph_ops::store(place, value, effect_ids, None)
}

fn scalar_handoff_load(
    graph: &mut AllocatedGraph,
    view: ValueId,
    elem_ty: &Type<TypeName>,
    effect_ids: &mut IdSource<EffectToken>,
) -> graph_ops::PendingEffect<AllocatedSemantic, ValueId> {
    let zero = graph_ops::intern_u32(graph, 0, None);
    let view = graph.view_id(view);
    let place = graph.add_view_index_place(view, zero, elem_ty.clone(), None);
    graph_ops::load(graph, place, elem_ty.clone(), effect_ids, None)
}

fn replace_prelude_effects_with_load(
    entry: &mut AllocatedEntry,
    producer_effects: &HashSet<SideEffectSite>,
    site: SideEffectSite,
    loads: Vec<AllocatedSideEffect>,
) {
    let effects = &mut entry.graph.skeleton.blocks[site.block].side_effects;
    let mut replacement = Vec::new();
    let mut loads = loads.into_iter();
    for (index, effect) in std::mem::take(effects).into_iter().enumerate() {
        if index == site.index {
            replacement.extend(&mut loads);
        }
        if !producer_effects.contains(&SideEffectSite {
            block: site.block,
            index,
        }) {
            replacement.push(effect);
        }
    }
    *effects = replacement;
}

fn replace_entry_prelude_with_load(
    entry: &mut AllocatedEntry,
    producer_effects: &HashSet<SideEffectSite>,
    load_effects: Vec<AllocatedSideEffect>,
) {
    entry.graph.skeleton.remove_effect_sites(producer_effects.iter().copied());
    let block = &mut entry.graph.skeleton.blocks[entry.graph.skeleton.entry];
    for (index, load) in load_effects.into_iter().enumerate() {
        block.side_effects.insert(index, load);
    }
}

fn replace_structured_prefix_with_load(
    entry: &mut AllocatedEntry,
    producer_effects: &HashSet<SideEffectSite>,
    continuation: BlockId,
    loaded: ValueId,
    load_effects: Vec<AllocatedSideEffect>,
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
    entry: &AllocatedEntry,
    result: &ResultBinding<Type<TypeName>>,
    materialization: FixedMaterializationKind,
    space: &SegSpace<SemanticResourceRef>,
    op: &screma::Op<AllocatedSemantic>,
    projection: &GraphProjection<SemanticResourceRef>,
) -> Option<Vec<OutputSpec>> {
    if op.result_count() != result.field_count() {
        return None;
    }
    (0..op.result_count())
        .map(|field| {
            let result = result.field(field)?;
            let source = result.single_value()?;
            let elem_ty = op.form.result_element_type(field)?.clone();
            let storage = match op.form.result_id(field)? {
                screma::ResultId::Reduction { .. } => OutputStorage::Scalar,
                screma::ResultId::Post(_) if !materialization.is_scalar() => OutputStorage::Array,
                screma::ResultId::Post(_) => return None,
            };
            let size = match storage {
                OutputStorage::Scalar => {
                    LogicalSize::FixedBytes(u64::from(ssa::layout::storage_elem_stride(&elem_ty)?))
                }
                OutputStorage::Array => LogicalSize::for_space(space, &elem_ty)?,
            };
            Some(OutputSpec {
                source,
                projected: projection.result(&result).ok()?,
                routed: entry.resource_for_result(&result).map(|resource| resource.0),
                storage,
                elem_ty,
                size,
            })
        })
        .collect()
}

fn refresh_resource_reads_for_values(graph: &mut AllocatedGraph, values: &[ValueId]) {
    let updates = {
        let analysis = GraphAnalysis::new(graph);
        let sites = values
            .iter()
            .flat_map(|value| analysis.slice().value_observers(*value).effect_sites().collect::<Vec<_>>())
            .filter(|site| {
                matches!(&graph.skeleton.effect(*site).kind,
                SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op)))
                    if matches!(op.semantic_state(), screma::SemanticState::Segmented(_)))
            })
            .collect::<HashSet<_>>();
        sites
            .into_iter()
            .map(|site| {
                (
                    site,
                    super::super::semantic_graph::read_resources(&analysis, graph.skeleton.effect(site)),
                )
            })
            .collect::<Vec<_>>()
    };
    for (site, reads) in updates {
        let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) =
            &mut graph.skeleton.effect_mut(site).kind
        else {
            continue;
        };
        let screma::SemanticState::Segmented(SegmentedMetadata { resources, .. }) = op.semantic_state_mut()
        else {
            continue;
        };
        resources.retain(|access| access.access != ResourceAccess::Read);
        *resources = SegResourceAccess::merge(resources, &reads);
    }
}

fn retarget_input_metadata(graph: &mut AllocatedGraph, replacements: &[InputReplacement]) {
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            let (inputs, segment) = match &mut effect.kind {
                SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                    let segment = match &mut op.state {
                        screma::SemanticState::Segmented(SegmentedMetadata {
                            space, resources, ..
                        }) => Some((space, resources)),
                        screma::SemanticState::Serial => None,
                    };
                    (&mut op.inputs, segment)
                }
                SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) => (
                    &mut op.body.inputs,
                    Some((&mut op.state.segment.space, &mut op.state.segment.resources)),
                ),
                _ => continue,
            };
            let mut reads = Vec::new();
            let mut domain = None;
            for (index, input) in inputs.iter_mut().enumerate() {
                if let Some(replacement) =
                    replacements.iter().find(|r| effect.operands[index].value() == Some(r.project))
                {
                    input.array = replacement.view_ty.clone();
                    reads.push(SegResourceAccess {
                        resource: SemanticResourceRef(replacement.resource),
                        access: ResourceAccess::Read,
                    });
                    if index == 0 {
                        domain = Some(replacement);
                    }
                }
            }
            if let Some((space, resources)) = segment {
                for replacement in replacements {
                    space.replace_reference(
                        replacement.project,
                        replacement.view.value(),
                        SemanticResourceRef(replacement.resource),
                    );
                }
                if let Some(replacement) = domain {
                    space.retarget_single_resource_length(
                        replacement.view,
                        SemanticResourceRef(replacement.resource),
                        replacement.elem_bytes,
                    );
                }
                *resources = SegResourceAccess::merge(resources, &reads);
            }
        }
    }
}
