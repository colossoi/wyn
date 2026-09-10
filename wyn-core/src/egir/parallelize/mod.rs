//! Target-aware recipe selection for allocated semantic EGIR.
//!
//! Reification lives in `egir::reify`. This module consumes those semantic
//! segmented operations after optimization and residency planning, selects
//! executable kernel recipes, introduces recipe-owned scratch, and produces a
//! first-class schedule. Unsupported shapes fall back before their graphs are
//! mutated.
//!
//! Canonical planning invariants:
//!
//! - every physical entry owns at most one selected kernel operation;
//! - `OutputRoute` metadata is the authority for semantic output ownership;
//! - compiler resources are allocated deterministically by endpoint,
//!   operation, kind, and slot; their numeric ids are not an external ABI;
//! - candidate analysis completes before graph mutation, so an unsupported
//!   recipe can select serial lowering without rolling back partial rewrites;
//! - host-provided bindings are ABI identities and target planning never
//!   renumbers or replaces them.
//!
//! Organization follows ownership rather than pass chronology: `model` owns
//! policy, checked errors, and immutable indexes; `planning` constructs
//! graph-local recipes directly and assigns scratch; `projection` owns
//! entry and route projection; `kernel` owns shared graph-building utilities;
//! `reduce`, `scan`, and `filter` own their algorithms (including scan-phase
//! builders shared by scan and filter); `prepare` converts selected semantic
//! operations to scheduled form; and `schedule` owns phase ordering,
//! publication, and physical construction.

#![deny(clippy::expect_used, clippy::unwrap_used)]

/// EGIR rebuilt from a validated target-specific kernel plan.
#[derive(Debug, Clone, Copy)]
pub enum PlannedTag {}
pub type Planned = super::program::PhysicalProgram<PlannedTag>;

mod capabilities;
mod filter;
mod hist;
mod kernel;
mod model;
mod planning;
pub(super) mod prepare;
mod projection;
mod reduce;
mod scan;
mod schedule;

use crate::egir;
use crate::pipeline_descriptor;
use filter::analyze_filter_candidate;
use kernel::{
    can_chunk_view, can_clone_pure_subgraph, chunk_soac_inputs, chunk_view_like, emit_chunk_arithmetic,
    synthesize_swap_wrapper, synthesize_u32_add_function,
};
use model::{CandidateSelection, ParallelizeError, Result as ParallelizeResult};
use planning::{make_screma_serial, LocatedScrema, SerialScremaRecipe};
use projection::{
    partition_entry_output_domains, project_kernel_body, project_single_effect_body, ProjectionSpec,
};
use reduce::{analyze_reduce_candidate, BoundReduce};
use scan::{analyze_scan_candidate, BoundScan, ScanPhase2Spec, ScanPhase3Spec, ScanScratch};
pub use schedule::{KernelDomain, KernelId, OutputRouteProjection, PhysicalKernel, PhysicalKernelGraph};
use std::collections::{HashMap, HashSet};
use wyn_base::IdSource;

use crate::interface::StorageAccess;
use crate::{EntryId, FunctionId, LookupMap, ResourceAccess};

use polytype::Type;
use smallvec::smallvec;

use super::allocation::ResourcesAllocated;
use super::from_tlc::ConvertError;
use super::graph_ops;
use super::program::{
    CompilerResourceKind, Func, LogicalResourceArena, OutputWriter, ResourceId, SemanticOpId,
    SemanticResourceDecl, SemanticResourceRef, StageOrigin, StagedProgram,
};
use wyn_staged_ir::StageId;

impl Planned {
    /// Logical resources after target recipe selection has installed only the
    /// work buffers required by the selected recipes.
    pub fn logical_resources(&self) -> &[super::program::LogicalResource] {
        &self.data.resources
    }
}
use super::soac::screma;
use super::types::{
    EGraph as FamilyGraph, EffectOp, EffectToken, PureOp as FamilyPureOp, SegBody, SegResourceAccess,
    SegSpace as FamilySegSpace, Semantic as SemanticFamily, SideEffect as FamilySideEffect, SideEffectKind,
    SideEffectSite, SkeletonTerminator, Soac, SoacEffect, ValueId, ValueKind,
};
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::flow::{BlockId, ControlHeader, ExecutionModel};
use crate::{LoweringProfile, PipelineTopologyPolicy, SchedulePolicy};

type Semantic = SemanticFamily<SemanticResourceRef>;
type EGraph = FamilyGraph<Semantic>;
type PureOp = FamilyPureOp<SemanticResourceRef>;
type SegSpace = FamilySegSpace<SemanticResourceRef>;
type SideEffect = FamilySideEffect<Semantic>;

/// A generated body kept together with the exact accesses established while
/// that body was built. Scheduling consumes this pair without inspecting the
/// graph or repairing missing facts.
struct BuiltPhase {
    body: super::program::PlannedEntry,
    resources: Vec<SegResourceAccess<ResourceId>>,
}

impl BuiltPhase {
    fn from_declarations(body: super::program::PlannedEntry) -> Self {
        let resources = declared_resources(&body.resource_declarations);
        Self { body, resources }
    }

    fn new(body: super::program::PlannedEntry, resources: Vec<SegResourceAccess<ResourceId>>) -> Self {
        Self { body, resources }
    }

    fn for_segment(
        body: super::program::PlannedEntry,
        segment: &screma::Segmented<SemanticResourceRef>,
    ) -> Self {
        let resources = merge_scheduled_resources(
            &declared_input_resources(&body.resource_declarations),
            &segmented_resources(segment),
        );
        Self { body, resources }
    }

    fn compute(self, dispatch: schedule::KernelDispatch, label: &'static str) -> schedule::PhaseSpec {
        schedule::PhaseSpec::compute(self.body, dispatch, label).with_resources(self.resources)
    }

    fn hist(
        self,
        dispatch: schedule::KernelDispatch,
        owner: SemanticOpId,
        operations: Vec<egir::soac::hist::AtomicUpdate>,
    ) -> schedule::PhaseSpec {
        schedule::PhaseSpec::hist(self.body, dispatch, owner, operations).with_resources(self.resources)
    }

    fn bucket(
        self,
        dispatch: schedule::KernelDispatch,
        owner: SemanticOpId,
        stage: egir::soac::hist::ParallelStage,
        topology: Option<egir::soac::hist::DispatchTopology>,
        storage: egir::soac::hist::BucketStorage<SemanticResourceRef>,
    ) -> schedule::PhaseSpec {
        schedule::PhaseSpec::bucket(self.body, dispatch, owner, stage, topology, storage)
            .with_resources(self.resources)
    }
    fn filter(
        self,
        dispatch: schedule::KernelDispatch,
        stage: super::soac::filter::ParallelStage,
        config: super::soac::filter::ParallelConfig<SemanticResourceRef>,
        storage: super::soac::filter::RuntimeStorage<SemanticResourceRef>,
    ) -> schedule::PhaseSpec {
        schedule::PhaseSpec::filter(self.body, dispatch, stage, config, storage)
            .with_resources(self.resources)
    }
}

impl From<ParallelizeError> for ConvertError {
    fn from(error: ParallelizeError) -> Self {
        Self::Internal(error.to_string())
    }
}

/// Allocated EGIR after mapped outputs have explicit destination places.
pub struct OutputDestinationsBound {
    program: ResourcesAllocated,
}

impl OutputDestinationsBound {
    pub fn program(&self) -> &ResourcesAllocated {
        &self.program
    }
}

/// An immutable recipe analysis paired with the program it describes.
pub struct KernelRecipesAnalyzed {
    program: ResourcesAllocated,
    analysis: planning::AnalyzedPlan,
    profile: LoweringProfile,
}

impl KernelRecipesAnalyzed {
    pub fn program(&self) -> &ResourcesAllocated {
        &self.program
    }
}

/// Recipe-owned scratch has been allocated and recipe handles are bound.
pub struct RecipeScratchAllocated {
    program: ResourcesAllocated,
    recipes: planning::RecipeIndex,
    profile: LoweringProfile,
}

impl RecipeScratchAllocated {
    pub fn program(&self) -> &ResourcesAllocated {
        &self.program
    }
}

/// A complete kernel schedule awaiting physical layout publication.
pub struct KernelScheduleBuilt {
    program: ResourcesAllocated,
    schedule: schedule::KernelPlan,
    profile: LoweringProfile,
}

impl KernelScheduleBuilt {
    pub fn program(&self) -> &ResourcesAllocated {
        &self.program
    }
}

pub fn bind_mapped_output_destinations(
    mut program: ResourcesAllocated,
) -> Result<OutputDestinationsBound, ConvertError> {
    let stage_ids = program.data.stages.stages().map(|(stage, _)| stage).collect::<Vec<_>>();
    for stage in stage_ids {
        let entry = program.data.stages.stage_body_mut(stage).ok_or_else(|| {
            ConvertError::Internal(format!("staged body {stage:?} disappeared during planning"))
        })?;
        entry.bind_mapped_output_destinations().map_err(ConvertError::Internal)?;
    }
    Ok(OutputDestinationsBound { program })
}

pub fn analyze_kernel_recipes(
    input: OutputDestinationsBound,
    profile: LoweringProfile,
) -> Result<KernelRecipesAnalyzed, ConvertError> {
    if profile.topology == PipelineTopologyPolicy::AuthoredOnly {
        validate_authored_only_input(&input.program)?;
        debug_assert_authored_only_input(&input.program);
    }
    if profile.schedule == SchedulePolicy::Serial {
        verify_serial_policy(&input.program)?;
    }
    let analysis = planning::analyze(&input.program, profile.schedule)?;
    Ok(KernelRecipesAnalyzed {
        program: input.program,
        analysis,
        profile,
    })
}

pub fn allocate_recipe_scratch(
    input: KernelRecipesAnalyzed,
) -> Result<RecipeScratchAllocated, ConvertError> {
    let KernelRecipesAnalyzed {
        program,
        analysis,
        profile,
    } = input;
    let (program, recipes) = analysis.allocate_scratch(program)?;
    Ok(RecipeScratchAllocated {
        program,
        recipes,
        profile,
    })
}

pub fn build_kernel_schedule(input: RecipeScratchAllocated) -> Result<KernelScheduleBuilt, ConvertError> {
    let RecipeScratchAllocated {
        mut program,
        recipes,
        profile,
    } = input;
    let builder = KernelPlanBuilder::new(
        &program.data.core.resources,
        &program.data.core.pipeline,
        &program.data.core.stage_entries,
        &program.data.stages,
        &program.functions,
        recipes,
        &mut program.global_context.semantic_ids,
        &mut program.global_context.effect_ids,
        program.data.core.identities.clone(),
    )?;
    let built = builder.build_schedule(profile.schedule == SchedulePolicy::Serial)?;
    let (schedule, generated_callables, identities) = built;
    if profile.topology == PipelineTopologyPolicy::AuthoredOnly {
        schedule
            .debug_assert_authored_only(program.data.stages.stages().count(), generated_callables.len());
    }
    let program = install_generated_callables(program, generated_callables, identities);
    Ok(KernelScheduleBuilt {
        program,
        schedule,
        profile,
    })
}

pub fn finalize_kernel_schedule(input: KernelScheduleBuilt) -> Result<Planned, ConvertError> {
    input.schedule.finalize(input.program, input.profile)
}

pub fn plan(program: ResourcesAllocated, profile: LoweringProfile) -> Result<Planned, ConvertError> {
    let program = bind_mapped_output_destinations(program)?;
    let program = analyze_kernel_recipes(program, profile)?;
    let program = allocate_recipe_scratch(program)?;
    let program = build_kernel_schedule(program)?;
    finalize_kernel_schedule(program)
}

fn verify_serial_policy(program: &ResourcesAllocated) -> ParallelizeResult<()> {
    let has_bucket_scatter = program.data.stages.stages().any(|(_, stage)| {
        let entry = stage.body();
        entry.graph.skeleton.blocks.iter().any(|(_, block)| {
            block.side_effects.iter().any(|effect| {
                let super::types::SideEffectKind::Soac(super::types::SoacEffect(
                    _,
                    super::types::Soac::Hist(op),
                )) = &effect.kind
                else {
                    return false;
                };
                op.form.operations.iter().any(|operation| {
                    matches!(operation.update, super::soac::hist::Update::BucketInsert { .. })
                })
            })
        })
    });
    if has_bucket_scatter {
        return Err(ParallelizeError::Invalid(
            "bucket_scatter requires its init/insert/finish pipeline and cannot be compiled with serial scheduling"
                .into(),
        ));
    }
    Ok(())
}

fn validate_authored_only_input(program: &ResourcesAllocated) -> Result<(), ConvertError> {
    if program
        .data
        .stages
        .stages()
        .any(|(_, stage)| matches!(stage.origin(), StageOrigin::Generated { .. }))
    {
        return Err(ConvertError::PipelineTopology(
            "authored-only lowering cannot schedule a compiler-generated stage".into(),
        ));
    }

    if program
        .data
        .core
        .resources
        .iter()
        .any(|resource| matches!(resource.origin(), super::program::ResourceOrigin::Compiler { .. }))
    {
        return Err(ConvertError::PipelineTopology(
            "authored-only lowering cannot schedule a compiler-owned resource".into(),
        ));
    }

    let authored_bindings = program
        .data
        .stages
        .stages()
        .flat_map(|(_, stage)| {
            stage
                .body()
                .inputs
                .iter()
                .filter_map(|input| input.resource)
                .chain(stage.body().outputs.iter().filter_map(|output| output.resource))
        })
        .map(|resource| resource.0)
        .collect::<HashSet<_>>();
    if let Some(resource) = program.data.stages.flows().find_map(|(_, flow)| {
        [Some(flow.storage().data), flow.storage().length]
            .into_iter()
            .flatten()
            .find(|resource| !authored_bindings.contains(resource))
    }) {
        return Err(ConvertError::PipelineTopology(format!(
            "authored-only lowering cannot carry cross-stage resource {resource:?} without an authored input or result binding"
        )));
    }
    Ok(())
}

fn debug_assert_authored_only_input(program: &ResourcesAllocated) {
    debug_assert!(
        !program
            .data
            .stages
            .stages()
            .any(|(_, stage)| matches!(stage.origin(), StageOrigin::Generated { .. })),
        "authored-only residency unexpectedly produced a generated stage"
    );
    debug_assert!(
        !program
            .data
            .core
            .resources
            .iter()
            .any(|resource| matches!(resource.origin(), super::program::ResourceOrigin::Compiler { .. })),
        "authored-only residency unexpectedly produced a compiler-owned resource"
    );
}

fn install_generated_callables(
    program: ResourcesAllocated,
    generated_callables: Vec<Func<Semantic>>,
    identities: super::program::ProgramIdentities,
) -> ResourcesAllocated {
    program.extend_functions(generated_callables).map_data(|mut data| {
        data.core.identities = identities;
        data
    })
}

struct KernelPlanBuilder<'effects> {
    schedule: schedule::ScheduleBuilder,
    stages: &'effects StagedProgram,
    recipes: planning::RecipeIndex,
    semantic_ids: &'effects mut super::program::SemanticOpIdSource,
    effect_ids: &'effects mut IdSource<EffectToken>,
    generated_callables: Vec<Func<Semantic>>,
    callables: LookupMap<FunctionId, Func<Semantic>>,
    identities: super::program::ProgramIdentities,
}

type BuiltPlan = (
    schedule::KernelPlan,
    Vec<Func<Semantic>>,
    super::program::ProgramIdentities,
);

impl planning::PlannedKernel {
    fn lower(
        self,
        lowering: &mut KernelPlanBuilder<'_>,
        stage: StageId,
        kernel: schedule::KernelId,
    ) -> ParallelizeResult<schedule::PreparedRecipe> {
        let (body, output_projection, recipe) = self.into_parts();
        let prepared = match recipe {
            planning::PlannedRecipe::Hist(candidate) => {
                lowering.lower_parallel_hist(body, kernel, candidate, output_projection)?
            }
            planning::PlannedRecipe::Filter(candidate) => {
                lowering.lower_parallel_filter(body, kernel, candidate, output_projection)?
            }
            planning::PlannedRecipe::Reduce(candidate) => {
                lowering.lower_parallel_reduce(body, kernel, candidate, output_projection)?
            }
            planning::PlannedRecipe::Scan(candidate) => {
                lowering.lower_parallel_scan(body, kernel, candidate, output_projection)?
            }
            planning::PlannedRecipe::Map(segment) => {
                let domain = schedule::domain_from_space(&segment.space)
                    .unwrap_or(schedule::KernelDomain::Fixed { x: 1, y: 1, z: 1 });
                let phase = BuiltPhase::for_segment(body, &segment)
                    .compute(schedule::KernelDispatch::inferred(domain), "serial_compute")
                    .with_output_projection(output_projection);
                schedule::PreparedRecipe::single(kernel, phase)?
            }
            planning::PlannedRecipe::Serial(recipe) => {
                lowering.prepare_serial_kernel(body, kernel, recipe, output_projection)?
            }
            planning::PlannedRecipe::Unchanged if output_projection.is_some() => {
                schedule::PreparedRecipe::single(
                    kernel,
                    schedule::PhaseSpec::compute(
                        body,
                        schedule::KernelDispatch::inferred(schedule::KernelDomain::Fixed {
                            x: 1,
                            y: 1,
                            z: 1,
                        }),
                        "serial_compute",
                    )
                    .with_output_projection(output_projection),
                )?
            }
            planning::PlannedRecipe::Unchanged => {
                schedule::PreparedRecipe::unchanged(kernel, body, lowering.schedule.stage_metadata(stage))?
            }
        };
        Ok(prepared)
    }
}

impl<'effects> KernelPlanBuilder<'effects> {
    fn into_plan(self, serial: bool) -> ParallelizeResult<BuiltPlan> {
        Ok((
            self.schedule.finish(self.stages, serial)?,
            self.generated_callables,
            self.identities,
        ))
    }

    fn define_callable(
        &mut self,
        name: String,
        build: impl FnOnce(FunctionId, String) -> ParallelizeResult<Func<Semantic>>,
    ) -> ParallelizeResult<FunctionId> {
        if self.identities.function_names().any(|existing| existing == name) {
            return Err(ParallelizeError::Invalid(format!(
                "planner-generated callable `{}` collides with an existing callable",
                name
            )));
        }
        let id = self.identities.alloc_function(name.clone());
        let function = build(id, name)?;
        if function.region != id {
            return Err("planner-generated callable did not retain its reserved region".into());
        }
        if function.name != self.identities.function_name(id) {
            return Err("planner-generated callable did not retain its reserved name".into());
        }
        self.callables.insert(id, function.clone());
        self.generated_callables.push(function);
        Ok(id)
    }

    fn callable(&self, region: FunctionId) -> ParallelizeResult<&Func<Semantic>> {
        let Some(callable) = self.callables.get(&region) else {
            return Err(ParallelizeError::Invalid(format!(
                "parallel lowering references missing callable {region:?}"
            )));
        };
        Ok(callable)
    }

    fn new(
        resources: &LogicalResourceArena,
        descriptor: &pipeline_descriptor::PipelineDescriptor,
        stage_entries: &[Vec<EntryId>],
        stages: &'effects StagedProgram,
        functions: &[Func<Semantic>],
        recipes: planning::RecipeIndex,
        semantic_ids: &'effects mut super::program::SemanticOpIdSource,
        effect_ids: &'effects mut IdSource<EffectToken>,
        identities: super::program::ProgramIdentities,
    ) -> ParallelizeResult<Self> {
        let mut schedule =
            schedule::ScheduleBuilder::from_descriptor(descriptor, stage_entries, resources, stages)?;
        for (stage, staged) in stages.stages() {
            if matches!(staged.origin(), StageOrigin::Authored) || staged.origin().space().is_some() {
                if let Some(count) = recipes.required_elements(stage) {
                    schedule.set_required_elements(stage, Some(count));
                }
            }
        }
        Ok(Self {
            schedule,
            stages,
            recipes,
            semantic_ids,
            effect_ids,
            generated_callables: Vec::new(),
            callables: functions.iter().map(|function| (function.region, function.clone())).collect(),
            identities,
        })
    }

    fn build_schedule(mut self, serial: bool) -> ParallelizeResult<BuiltPlan> {
        let stages = self.stages.stages().map(|(stage, _)| stage).collect::<Vec<_>>();
        for stage in stages {
            self.lower_endpoint(stage)?;
        }
        self.into_plan(serial)
    }

    fn lower_endpoint(&mut self, stage: StageId) -> ParallelizeResult<()> {
        let kernel = self.schedule.primary_kernel(stage);
        let plan = self.recipes.take_endpoint(stage)?;
        let (primary, siblings) = plan.into_parts();
        let primary = primary.lower(self, stage, kernel)?;
        if siblings.is_empty() {
            self.schedule.install_stage(stage, primary)?;
            return Ok(());
        }
        let mut components = vec![primary];
        for mut sibling in siblings {
            sibling.assign_entry_id(self.identities.alloc_entry(sibling.entry_name().to_owned()));
            let sibling_kernel = self.schedule.allocate_kernel();
            components.push(sibling.lower(self, stage, sibling_kernel)?);
        }
        let recipe = schedule::PreparedRecipe::parallel(components, kernel)?;
        self.schedule.install_stage(stage, recipe)?;
        Ok(())
    }

    fn lower_parallel_hist(
        &mut self,
        body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
        candidate: hist::BoundHistCandidate,
        output_projection: Option<Vec<usize>>,
    ) -> ParallelizeResult<schedule::PreparedRecipe> {
        match candidate {
            hist::BoundHistCandidate::Atomic(candidate) => {
                let domain = schedule::domain_from_space(&candidate.space)
                    .unwrap_or(schedule::KernelDomain::Fixed { x: 1, y: 1, z: 1 });
                let phase = BuiltPhase::from_declarations(body)
                    .hist(
                        schedule::KernelDispatch::inferred(domain),
                        candidate.owner,
                        candidate.operations,
                    )
                    .with_output_projection(output_projection);
                Ok(schedule::PreparedRecipe::single(kernel, phase)?)
            }
            hist::BoundHistCandidate::Bucket(candidate) => {
                self.lower_parallel_bucket(body, kernel, candidate, output_projection)
            }
        }
    }
    fn lower_parallel_reduce(
        &mut self,
        body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
        candidate: BoundReduce,
        output_projection: Option<Vec<usize>>,
    ) -> ParallelizeResult<schedule::PreparedRecipe> {
        use schedule::KernelDomain;

        let domain = schedule::domain_from_space(&candidate.segment().space)
            .unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
        let (phase1, phases) = self.emit_reduce_entry(body, candidate)?;
        let recipe = phase1
            .compute(schedule::KernelDispatch::inferred(domain), "reduce_phase1")
            .with_output_projection(output_projection);
        let after: Vec<_> = phases
            .into_iter()
            .map(|phase| {
                phase.compute(
                    schedule::KernelDispatch::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                    "reduce_combine",
                )
            })
            .collect();
        let mut specs = vec![(kernel, recipe)];
        specs.extend(after.into_iter().map(|spec| (self.schedule.allocate_kernel(), spec)));
        Ok(schedule::PreparedRecipe::sequence(specs, kernel)?)
    }

    fn lower_parallel_scan(
        &mut self,
        body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
        candidate: BoundScan,
        output_projection: Option<Vec<usize>>,
    ) -> ParallelizeResult<schedule::PreparedRecipe> {
        use schedule::KernelDomain;

        let phase1_domain = schedule::domain_from_space(&candidate.segment().space)
            .unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
        let [phase1, block_scan, apply_offsets] = self.emit_scan_entry(body, candidate)?;
        let recipe = phase1
            .compute(
                schedule::KernelDispatch::inferred(phase1_domain.clone()),
                "scan_phase1",
            )
            .with_output_projection(output_projection);
        let block_scan = block_scan.compute(
            schedule::KernelDispatch::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
            "scan_block",
        );
        let apply_offsets = apply_offsets.compute(
            schedule::KernelDispatch::explicit(phase1_domain),
            "scan_apply_offsets",
        );
        let block = self.schedule.allocate_kernel();
        let apply = self.schedule.allocate_kernel();
        Ok(schedule::PreparedRecipe::sequence(
            vec![(kernel, recipe), (block, block_scan), (apply, apply_offsets)],
            kernel,
        )?)
    }

    fn prepare_serial_kernel(
        &mut self,
        mut body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
        recipe: SerialScremaRecipe,
        output_projection: Option<Vec<usize>>,
    ) -> ParallelizeResult<schedule::PreparedRecipe> {
        make_screma_serial(&mut body.graph, recipe);
        let recipe = schedule::PhaseSpec::compute(
            body,
            schedule::KernelDispatch::inferred(schedule::KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
            "serial_compute",
        )
        .with_output_projection(output_projection);
        Ok(schedule::PreparedRecipe::single(kernel, recipe)?)
    }
}

fn merge_scheduled_resources(
    left: &[SegResourceAccess<ResourceId>],
    right: &[SegResourceAccess<ResourceId>],
) -> Vec<SegResourceAccess<ResourceId>> {
    egir::ir::SegResourceAccess::merge(left, right)
}

fn segmented_resources(
    segment: &screma::Segmented<SemanticResourceRef>,
) -> Vec<SegResourceAccess<ResourceId>> {
    segment
        .resources
        .iter()
        .map(|resource| SegResourceAccess::<ResourceId> {
            resource: resource.resource.0,
            access: resource.access,
        })
        .collect()
}

fn declared_resources(declarations: &[SemanticResourceDecl]) -> Vec<SegResourceAccess<ResourceId>> {
    let mut accesses: HashMap<ResourceId, ResourceAccess> = HashMap::new();
    for declaration in declarations {
        let access = ResourceAccess::from(StorageAccess::from(declaration.role));
        accesses.entry(declaration.resource.0).and_modify(|old| *old = old.merge(access)).or_insert(access);
    }

    let mut resources = accesses
        .into_iter()
        .map(|(resource, access)| SegResourceAccess::<ResourceId> { resource, access })
        .collect::<Vec<_>>();
    resources.sort_by_key(|resource| resource.resource);
    resources
}

fn declared_input_resources(declarations: &[SemanticResourceDecl]) -> Vec<SegResourceAccess<ResourceId>> {
    declarations
        .iter()
        .filter(|declaration| declaration.role.reads())
        .map(|declaration| SegResourceAccess::<ResourceId> {
            resource: declaration.resource.0,
            access: ResourceAccess::Read,
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests;
