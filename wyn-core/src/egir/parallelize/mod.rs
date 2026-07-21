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

mod filter;
mod kernel;
mod model;
mod planning;
pub(super) mod prepare;
mod projection;
mod reduce;
mod scan;
mod schedule;

use filter::analyze_filter_candidate;
use kernel::{
    apply_manifest_resource_sizes, can_chunk_view, can_clone_pure_subgraph, chunk_soac_inputs,
    chunk_view_like, dispatch_worker_logical_size, emit_chunk_arithmetic, synthesize_swap_wrapper,
    synthesize_u32_add_function, ChunkInputKind,
};
use model as error;
use model::{CandidateSelection, DisjointSets};
use planning::{make_screma_serial, LocatedScrema, SerialScremaRecipe};
use projection::{partition_entry_output_domains, project_kernel_body, ProjectionSpec};
use reduce::{analyze_reduce_candidate, BoundReduce};
use scan::{analyze_scan_candidate, BoundScan, ScanPhase2Spec, ScanPhase3Spec, ScanScratch};
pub(super) use schedule::KernelPlan;
pub use schedule::{
    KernelDomain, KernelId, KernelPhaseSummary, KernelPlanSummary, OutputRouteProjection, ScheduledResource,
};
use std::collections::{HashMap, HashSet};

use crate::LookupMap;

use polytype::Type;
use smallvec::smallvec;

use super::from_tlc::ConvertError;
use super::graph_ops;
use super::program::{
    AllocatedProgram, CompilerFlowEndpoint, CompilerResourceKind, LogicalResourceArena, OutputWriter,
    PhysicalProgram, ResourceId, SemanticEntry, SemanticFunc, SemanticOpId, SemanticResourceDecl,
    SemanticResourceRef,
};
use super::soac::screma;
use super::types::{
    EGraph, ENode, EffectOp, EffectToken, NodeId, PureOp, RegionId, SegBody, SegSpace, SideEffect,
    SideEffectKind, SideEffectSite, SkeletonTerminator, Soac, SoacDestination, SoacEffect,
};
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::flow::{BlockId, ControlHeader, ExecutionModel};
use crate::types::TypeExt;
use crate::{IdSource, LoweringProfile, SchedulePolicy};

/// A generated body kept together with the exact accesses established while
/// that body was built. Scheduling consumes this pair without inspecting the
/// graph or repairing missing facts.
struct BuiltPhase {
    body: super::program::PlannedEntry,
    resources: Vec<schedule::ScheduledResource>,
}

impl BuiltPhase {
    fn from_declarations(body: super::program::PlannedEntry) -> Self {
        let resources = declared_resources(&body.resource_declarations);
        Self { body, resources }
    }

    fn new(body: super::program::PlannedEntry, resources: Vec<schedule::ScheduledResource>) -> Self {
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

pub(crate) fn plan(
    mut inner: AllocatedProgram,
    binding_ids: &mut IdSource<u32>,
    effect_ids: &mut IdSource<EffectToken>,
    profile: LoweringProfile,
) -> Result<(PhysicalProgram, KernelPlanSummary), ConvertError> {
    let kernel_plan = match profile.schedule {
        SchedulePolicy::Parallel => build_parallel_plan(&mut inner, effect_ids),
        SchedulePolicy::Serial => build_serial_plan(&mut inner, effect_ids),
    }?;

    kernel_plan.finalize(inner, binding_ids, profile)
}

/// Analyze target recipes, allocate their scratch resources, and build the
/// executable parallel kernel plan.
fn build_parallel_plan(
    inner: &mut AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<schedule::KernelPlan> {
    let analysis = planning::analyze(inner)?;
    let recipes = analysis.allocate_scratch(inner)?;
    let built = KernelPlanBuilder::new(inner, recipes, effect_ids)?.build_parallel_schedule(inner)?;
    let (schedule, generated_callables, region_interner) = built.into_plan();
    inner.functions.extend(generated_callables);
    inner.region_interner = region_interner;
    Ok(schedule)
}

/// Build a kernel plan that selects serial recipes without allocating
/// algorithm-specific parallel scratch resources.
fn build_serial_plan(
    inner: &mut AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<schedule::KernelPlan> {
    let recipes = planning::analyze(inner)?.serial_recipes();
    let built = KernelPlanBuilder::new(inner, recipes, effect_ids)?.build_serial_schedule(inner)?;
    let (schedule, generated_callables, region_interner) = built.into_plan();
    inner.functions.extend(generated_callables);
    inner.region_interner = region_interner;
    Ok(schedule)
}

struct KernelPlanBuilder<'resources, 'effects> {
    schedule: schedule::KernelPlan,
    resources: &'resources LogicalResourceArena,
    flows: model::ResourceFlowIndex,
    recipes: planning::RecipeIndex,
    effect_ids: &'effects mut crate::IdSource<EffectToken>,
    generated_callables: Vec<SemanticFunc>,
    region_interner: super::program::RegionInterner,
}

type BuiltPlan = (
    schedule::KernelPlan,
    Vec<SemanticFunc>,
    super::program::RegionInterner,
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
        (self.schedule, self.generated_callables, self.region_interner)
    }

    fn intern_callable(&mut self, name: impl AsRef<str>) -> RegionId {
        self.region_interner.intern(name.as_ref())
    }

    fn callable_name(&self, id: RegionId) -> &str {
        self.region_interner.resolve(id)
    }

    fn define_callable(&mut self, function: SemanticFunc) -> error::Result<RegionId> {
        if self.region_interner.get(&function.name).is_some() {
            return Err(error::ParallelizeError::Invalid(format!(
                "planner-generated callable `{}` collides with an existing callable",
                function.name
            )));
        }
        let id = self.region_interner.intern(&function.name);
        self.generated_callables.push(function);
        Ok(id)
    }

    fn new(
        inner: &'resources AllocatedProgram,
        recipes: planning::RecipeIndex,
        effect_ids: &'effects mut crate::IdSource<EffectToken>,
    ) -> error::Result<Self> {
        let resources = &inner.resources;
        let flows = model::ResourceFlowIndex::new(&inner.resources);
        let mut schedule = schedule::KernelPlan::from_descriptor(&inner.pipeline, &inner.semantic)?;
        for source in inner.semantic.entry_ids() {
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
            effect_ids,
            generated_callables: Vec::new(),
            region_interner: inner.region_interner.clone(),
        })
    }

    fn build_parallel_schedule(mut self, inner: &'resources AllocatedProgram) -> error::Result<Self> {
        self.attach_materializations(inner)?;
        self.schedule_entries(inner)?;
        self.schedule.coalesce_resource_flows(self.flows.flows())?;
        Ok(self)
    }

    fn build_serial_schedule(mut self, inner: &'resources AllocatedProgram) -> error::Result<Self> {
        self.attach_materializations(inner)?;
        self.schedule.make_serial()?;
        self.schedule.coalesce_resource_flows(self.flows.flows())?;
        Ok(self)
    }

    fn schedule_entries(&mut self, inner: &'resources AllocatedProgram) -> error::Result<()> {
        for source in inner.semantic.entry_ids() {
            let kernel = self.schedule.primary_kernel(source);
            self.lower_endpoint(CompilerFlowEndpoint::Entry(source), kernel)?;
        }
        Ok(())
    }

    /// Attach allocation-created producer entries in compiler-flow order and
    /// immediately lower the recipe owned by each new physical kernel.
    fn attach_materializations(&mut self, inner: &'resources AllocatedProgram) -> error::Result<()> {
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
            let crate::egir::program::CompilerFlowEndpoint::Materialization(id) = producer_id else {
                return Err(error::ParallelizeError::Invalid(
                    "typed entry/prepass producer was omitted while seeding the kernel plan".into(),
                ));
            };
            let requirement = inner.materializations.get(id).ok_or_else(|| {
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
        for sibling in siblings {
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
    left: &[schedule::ScheduledResource],
    right: &[schedule::ScheduledResource],
) -> Vec<schedule::ScheduledResource> {
    crate::egir::ir::SegResourceAccess::merge(left, right)
}

fn segmented_resources(
    segment: &screma::Segmented<SemanticResourceRef>,
) -> Vec<schedule::ScheduledResource> {
    segment
        .resources
        .iter()
        .map(|resource| schedule::ScheduledResource {
            resource: resource.resource.0,
            access: resource.access,
        })
        .collect()
}

fn declared_resources(declarations: &[SemanticResourceDecl]) -> Vec<schedule::ScheduledResource> {
    let mut accesses: HashMap<ResourceId, crate::ResourceAccess> = HashMap::new();
    for declaration in declarations {
        let access = match declaration.role {
            crate::interface::StorageRole::Input => crate::ResourceAccess::Read,
            crate::interface::StorageRole::Output => crate::ResourceAccess::Write,
            crate::interface::StorageRole::Intermediate => crate::ResourceAccess::ReadWrite,
        };
        accesses.entry(declaration.resource.0).and_modify(|old| *old = old.merge(access)).or_insert(access);
    }

    let mut resources = accesses
        .into_iter()
        .map(|(resource, access)| schedule::ScheduledResource { resource, access })
        .collect::<Vec<_>>();
    resources.sort_by_key(|resource| resource.resource);
    resources
}

fn declared_input_resources(declarations: &[SemanticResourceDecl]) -> Vec<schedule::ScheduledResource> {
    declarations
        .iter()
        .filter(|declaration| declaration.role == crate::interface::StorageRole::Input)
        .map(|declaration| schedule::ScheduledResource {
            resource: declaration.resource.0,
            access: crate::ResourceAccess::Read,
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests;
