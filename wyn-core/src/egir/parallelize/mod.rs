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
pub type Planned = super::program::Program<
    PlannedTag,
    super::ir::ProgramFamily<
        super::types::Physical,
        interface::StorageBindingDecl,
        super::ir::RealizedOutputRoute,
        super::program::ResourceProgramData,
    >,
    super::program::PlannedGlobal,
>;

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
use crate::interface;
use crate::pipeline_descriptor;
use crate::IdArena;
use crate::IdSource;
use filter::analyze_filter_candidate;
use kernel::{
    apply_manifest_resource_sizes, can_chunk_view, can_clone_pure_subgraph, chunk_soac_inputs,
    chunk_view_like, dispatch_worker_logical_size, emit_chunk_arithmetic, synthesize_swap_wrapper,
    synthesize_u32_add_function,
};
use model as error;
use model::{CandidateSelection, DisjointSets};
use planning::{make_screma_serial, LocatedScrema, SerialScremaRecipe};
use projection::{
    partition_entry_output_domains, project_kernel_body, project_single_effect_body, ProjectionSpec,
};
use reduce::{analyze_reduce_candidate, BoundReduce};
use scan::{analyze_scan_candidate, BoundScan, ScanPhase2Spec, ScanPhase3Spec, ScanScratch};
pub use schedule::{KernelDomain, KernelId, KernelPhaseSummary, KernelPlanSummary, OutputRouteProjection};
use std::collections::{HashMap, HashSet};

use crate::interface::StorageAccess;
use crate::{EntryId, FunctionId, LookupMap, ResourceAccess};

use polytype::Type;
use smallvec::smallvec;

use super::allocation::{self, CompilerFlowEndpoint, ResourcesAllocated};
use super::from_tlc::ConvertError;
use super::graph_ops;
use super::program::{
    CompilerResourceKind, Func, LogicalResourceArena, MaterializationId, MaterializationRequirement,
    OutputWriter, ResourceId, SemanticOpId, SemanticResourceDecl, SemanticResourceRef,
};

impl Planned {
    /// Logical resources after target recipe selection has installed only the
    /// work buffers required by the selected recipes.
    pub fn logical_resources(&self) -> &[super::program::LogicalResource] {
        &self.data.resources
    }

    pub fn kernel_plan(&self) -> &KernelPlanSummary {
        &self.global_context.kernel_plan
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
use crate::{LoweringProfile, SchedulePolicy};

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

impl From<error::ParallelizeError> for ConvertError {
    fn from(error: error::ParallelizeError) -> Self {
        Self::Internal(error.to_string())
    }
}

pub fn plan(mut program: ResourcesAllocated, profile: LoweringProfile) -> Result<Planned, ConvertError> {
    for entry in &mut program.entry_points {
        entry.bind_mapped_output_destinations().map_err(ConvertError::Internal)?;
    }
    let (program, kernel_plan) = match profile.schedule {
        SchedulePolicy::Parallel => build_parallel_plan(program),
        SchedulePolicy::Serial => build_serial_plan(program),
    }?;
    kernel_plan.finalize(program, profile)
}

/// Analyze target recipes, allocate their scratch resources, and build the
/// executable parallel kernel plan.
fn build_parallel_plan(
    program: ResourcesAllocated,
) -> error::Result<(ResourcesAllocated, schedule::KernelPlan)> {
    let analysis = planning::analyze(&program)?;
    let (mut program, recipes) = analysis.allocate_scratch(program)?;
    let flows = allocation::resource_flows(&program);
    let built = KernelPlanBuilder::new(
        &program.data.core.resources,
        &program.data.core.pipeline,
        &program.data.core.stage_entries,
        &program.entry_points,
        &program.functions,
        flows,
        recipes,
        &mut program.global_context.semantic_ids,
        &mut program.global_context.effect_ids,
        program.data.core.identities.clone(),
    )?
    .build_parallel_schedule(&program.data.materializations)?;
    let (schedule, generated_callables, identities) = built.into_plan();
    let program = install_generated_callables(program, generated_callables, identities);
    Ok((program, schedule))
}

/// Build a kernel plan that selects serial recipes without allocating
/// algorithm-specific parallel scratch resources.
fn build_serial_plan(
    mut program: ResourcesAllocated,
) -> error::Result<(ResourcesAllocated, schedule::KernelPlan)> {
    let has_bucket_scatter = program.entry_points.iter().any(|entry| {
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
        return Err(error::ParallelizeError::Invalid(
            "bucket_scatter requires its init/insert/finish pipeline and cannot be compiled with --single-stage"
                .into(),
        ));
    }
    let recipes = planning::analyze(&program)?.serial_recipes();
    let flows = allocation::resource_flows(&program);
    let built = KernelPlanBuilder::new(
        &program.data.core.resources,
        &program.data.core.pipeline,
        &program.data.core.stage_entries,
        &program.entry_points,
        &program.functions,
        flows,
        recipes,
        &mut program.global_context.semantic_ids,
        &mut program.global_context.effect_ids,
        program.data.core.identities.clone(),
    )?
    .build_serial_schedule(&program.data.materializations)?;
    let (schedule, generated_callables, identities) = built.into_plan();
    let program = install_generated_callables(program, generated_callables, identities);
    Ok((program, schedule))
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

struct KernelPlanBuilder<'resources, 'effects> {
    schedule: schedule::KernelPlan,
    resources: &'resources LogicalResourceArena,
    flows: model::ResourceFlowIndex,
    recipes: planning::RecipeIndex,
    semantic_ids: &'effects mut super::program::SemanticOpIdSource,
    effect_ids: &'effects mut IdSource<EffectToken>,
    generated_callables: Vec<Func<Semantic>>,
    callables: LookupMap<FunctionId, Func<Semantic>>,
    entry_ids: Vec<EntryId>,
    identities: super::program::ProgramIdentities,
}

type BuiltPlan = (
    schedule::KernelPlan,
    Vec<Func<Semantic>>,
    super::program::ProgramIdentities,
);

impl planning::PlannedKernel {
    /// Consume the selected body and its graph-local recipe as one operation.
    /// No caller can retain a recipe handle while independently mutating the
    /// graph it addresses.
    fn lower(
        self,
        lowering: &mut KernelPlanBuilder<'_, '_>,
        kernel: schedule::KernelId,
    ) -> error::Result<()> {
        let (body, output_projection, recipe) = self.into_parts();
        match recipe {
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
                lowering.schedule.commit_kernel(kernel, phase)?;
            }
            planning::PlannedRecipe::Serial(recipe) => {
                lowering.commit_serial_kernel(body, kernel, recipe, output_projection)?
            }
            planning::PlannedRecipe::Unchanged if output_projection.is_some() => {
                lowering.schedule.commit_kernel(
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
                )?;
            }
            planning::PlannedRecipe::Unchanged => {}
        }
        Ok(())
    }
}

impl<'resources, 'effects> KernelPlanBuilder<'resources, 'effects> {
    fn into_plan(self) -> BuiltPlan {
        (self.schedule, self.generated_callables, self.identities)
    }

    fn define_callable(
        &mut self,
        name: String,
        build: impl FnOnce(FunctionId, String) -> Func<Semantic>,
    ) -> error::Result<FunctionId> {
        if self.identities.function_names().any(|existing| existing == name) {
            return Err(error::ParallelizeError::Invalid(format!(
                "planner-generated callable `{}` collides with an existing callable",
                name
            )));
        }
        let id = self.identities.alloc_function(name.clone());
        let function = build(id, name);
        assert_eq!(
            function.region, id,
            "planner-generated callable did not retain its reserved region"
        );
        assert_eq!(
            &function.name,
            self.identities.function_name(id),
            "planner-generated callable did not retain its reserved name"
        );
        self.generated_callables.push(function);
        self.callables.insert(id, self.generated_callables.last().unwrap().clone());
        Ok(id)
    }

    fn callable(&self, region: FunctionId) -> &Func<Semantic> {
        self.callables.get(&region).expect("parallel lowering callable boundary")
    }

    fn new(
        resources: &'resources LogicalResourceArena,
        descriptor: &pipeline_descriptor::PipelineDescriptor,
        stage_entries: &[Vec<EntryId>],
        entries: &[super::program::AllocatedEntry],
        functions: &[Func<Semantic>],
        flows: Vec<(ResourceId, allocation::CompilerResourceFlow)>,
        recipes: planning::RecipeIndex,
        semantic_ids: &'effects mut super::program::SemanticOpIdSource,
        effect_ids: &'effects mut IdSource<EffectToken>,
        identities: super::program::ProgramIdentities,
    ) -> error::Result<Self> {
        let flows = model::ResourceFlowIndex::new(flows);
        let mut schedule =
            schedule::KernelPlan::from_descriptor(descriptor, stage_entries, resources, entries)?;
        for entry in entries {
            let source = entry.id;
            let endpoint = CompilerFlowEndpoint::Entry(source);
            if let Some(count) = recipes.required_elements(endpoint) {
                schedule.set_required_elements(endpoint, count);
            }
        }
        Ok(Self {
            schedule,
            resources,
            flows,
            recipes,
            semantic_ids,
            effect_ids,
            generated_callables: Vec::new(),
            callables: functions.iter().map(|function| (function.region, function.clone())).collect(),
            entry_ids: entries.iter().map(|entry| entry.id).collect(),
            identities,
        })
    }

    fn build_parallel_schedule(
        mut self,
        materializations: &IdArena<MaterializationId, MaterializationRequirement>,
    ) -> error::Result<Self> {
        self.attach_materializations(materializations)?;
        self.schedule_entries()?;
        self.schedule.coalesce_resource_flows(self.flows.flows())?;
        Ok(self)
    }

    fn build_serial_schedule(
        mut self,
        materializations: &IdArena<MaterializationId, MaterializationRequirement>,
    ) -> error::Result<Self> {
        self.attach_materializations(materializations)?;
        self.schedule.make_serial()?;
        self.schedule.coalesce_resource_flows(self.flows.flows())?;
        Ok(self)
    }

    fn schedule_entries(&mut self) -> error::Result<()> {
        for source in self.entry_ids.clone() {
            let kernel = self.schedule.primary_kernel(source);
            self.lower_endpoint(CompilerFlowEndpoint::Entry(source), kernel)?;
        }
        Ok(())
    }

    /// Attach allocation-created producer entries in compiler-flow order and
    /// immediately lower the recipe owned by each new physical kernel.
    fn attach_materializations(
        &mut self,
        materializations: &IdArena<MaterializationId, MaterializationRequirement>,
    ) -> error::Result<()> {
        let mut ready = std::collections::BTreeSet::new();
        for (_, flow) in self.flows.flows() {
            for consumer in &flow.consumers {
                if self.schedule.contains_flow_source(*consumer) {
                    ready.insert((flow.producer, *consumer));
                }
            }
        }

        while let Some((producer_id, consumer_id)) = ready.pop_first() {
            if self.schedule.contains_flow_source(producer_id) {
                continue;
            }
            let consumer = self.schedule.kernel_for_flow_source(consumer_id).ok_or_else(|| {
                error::ParallelizeError::Invalid(format!(
                    "scheduled flow consumer {consumer_id:?} has no kernel handle"
                ))
            })?;
            let CompilerFlowEndpoint::Materialization(id) = producer_id else {
                return Err(error::ParallelizeError::Invalid(
                    "typed entry/prepass producer was omitted while seeding the kernel plan".into(),
                ));
            };
            let requirement = materializations.get(id).ok_or_else(|| {
                error::ParallelizeError::Invalid(format!(
                    "materialization flow references missing requirement {id:?}"
                ))
            })?;
            let kernel = self.schedule.add_materialization_before(consumer, id, requirement)?;
            self.lower_endpoint(CompilerFlowEndpoint::Materialization(id), kernel)?;
            for upstream in self.flows.incoming(producer_id) {
                ready.insert((*upstream, producer_id));
            }
        }
        Ok(())
    }

    fn lower_endpoint(
        &mut self,
        endpoint: CompilerFlowEndpoint,
        kernel: schedule::KernelId,
    ) -> error::Result<()> {
        let Some(plan) = self.recipes.take_endpoint(endpoint)? else {
            return Ok(());
        };
        let (primary, siblings) = plan.into_parts();
        primary.lower(self, kernel)?;
        for mut sibling in siblings {
            let seed = sibling.seed_body();
            let id = self.identities.alloc_entry(seed.name.clone());
            sibling.assign_entry_id(id);
            let phase = schedule::PhaseSpec::compute(
                sibling.seed_body(),
                schedule::KernelDispatch::inferred(schedule::KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                "serial_compute",
            );
            let sibling_kernel = self.schedule.add_sibling(kernel, phase)?;
            sibling.lower(self, sibling_kernel)?;
        }
        Ok(())
    }

    fn lower_parallel_hist(
        &mut self,
        body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
        candidate: hist::BoundHistCandidate,
        output_projection: Option<Vec<usize>>,
    ) -> error::Result<()> {
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
                self.schedule.commit_kernel(kernel, phase)?;
                Ok(())
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
    ) -> error::Result<()> {
        use schedule::KernelDomain;

        let domain = schedule::domain_from_space(&candidate.segment().space)
            .unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
        let (phase1, phases) = self.emit_reduce_entry(body, candidate)?;
        let recipe = phase1
            .compute(schedule::KernelDispatch::inferred(domain), "reduce_phase1")
            .with_output_projection(output_projection);
        let after = phases
            .into_iter()
            .map(|phase| {
                phase.compute(
                    schedule::KernelDispatch::explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                    "reduce_combine",
                )
            })
            .collect();
        self.schedule.replace_chain(kernel, Vec::new(), recipe, after)?;
        Ok(())
    }

    fn lower_parallel_scan(
        &mut self,
        body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
        candidate: BoundScan,
        output_projection: Option<Vec<usize>>,
    ) -> error::Result<()> {
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
        self.schedule.replace_chain(kernel, Vec::new(), recipe, vec![block_scan, apply_offsets])?;
        Ok(())
    }

    fn commit_serial_kernel(
        &mut self,
        mut body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
        recipe: SerialScremaRecipe,
        output_projection: Option<Vec<usize>>,
    ) -> error::Result<()> {
        make_screma_serial(&mut body.graph, recipe);
        let recipe = schedule::PhaseSpec::compute(
            body,
            schedule::KernelDispatch::inferred(schedule::KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
            "serial_compute",
        )
        .with_output_projection(output_projection);
        self.schedule.commit_kernel(kernel, recipe)?;
        Ok(())
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
        .filter(|declaration| declaration.role == interface::StorageRole::Input)
        .map(|declaration| SegResourceAccess::<ResourceId> {
            resource: declaration.resource.0,
            access: ResourceAccess::Read,
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests;
