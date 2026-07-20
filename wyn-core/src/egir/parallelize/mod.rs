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
//! policy, checked errors, and immutable indexes; `planning` recognizes
//! graph-local recipes, selects them, and assigns scratch; `projection` owns
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
#[cfg(test)]
mod test_support;

use filter::analyze_filter_candidates;
use kernel::{
    apply_manifest_resource_sizes, can_chunk_view, can_clone_pure_subgraph, chunk_soac_inputs,
    chunk_view_like, dispatch_worker_logical_size, emit_chunk_arithmetic, synthesize_swap_wrapper,
    synthesize_u32_add_function, ChunkInputKind,
};
use model as error;
use model::{FallbackReason, ParallelPolicy, RecipeSelection};
use planning::{
    make_screma_serial, parallel_recipe_effect, ParallelReduce, ParallelScan, SerialScremaRecipe,
};
#[cfg(test)]
use projection::side_effect_output_slots_from_routes;
use projection::{project_kernel_body, split_multidomain_seg_maps, ProjectionSpec, UnionFind};
use reduce::{analyze_reduce_candidate, BoundReduce};
use scan::{analyze_scan_candidate, BoundScan, ScanPhase2Spec, ScanPhase3Spec, ScanScratch};
pub(super) use schedule::KernelPlan;
pub use schedule::{
    EntryAbiProjection, KernelDomain, KernelId, KernelKind, KernelPhaseSummary, KernelPlacement,
    KernelPlanSummary, OutputRouteProjection, ScheduledResource,
};
#[cfg(test)]
pub(crate) use test_support::{planned_callable_names, FILTER_SCAN_GROUPS, REDUCE_PHASE1_WIDTH};

use std::collections::HashSet;

use crate::LookupMap;

use polytype::Type;
use smallvec::smallvec;

use super::from_tlc::ConvertError;
use super::graph_ops;
use super::program::{
    AllocatedProgram, CompilerFlowEndpoint, CompilerResourceKind, LogicalResourceArena, OutputWriter,
    PhysicalProgram, ResourceId, SemanticEntry, SemanticEntryId, SemanticFunc, SemanticOpId,
    SemanticResourceDecl, SemanticResourceRef,
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
        SchedulePolicy::SingleStage => build_sequential_plan(&inner, effect_ids),
    }?;

    kernel_plan.finalize(inner, binding_ids, profile)
}

/// Analyze target recipes, allocate their scratch resources, and build the
/// executable parallel kernel plan.
fn build_parallel_plan(
    inner: &mut AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<schedule::KernelPlan> {
    let policy = ParallelPolicy::default();
    let analysis = planning::analyze(inner, policy)?;
    let recipes = analysis.allocate_scratch(inner)?;
    let builder = KernelPlanBuilder::new(inner, recipes, policy, effect_ids)?;
    builder.build_parallel_schedule(inner)
}

/// Build a kernel plan that selects serial recipes without allocating
/// algorithm-specific parallel scratch resources.
fn build_sequential_plan(
    inner: &AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<schedule::KernelPlan> {
    let policy = ParallelPolicy::default();
    let recipes = planning::RecipeIndex::sequential();
    let builder = KernelPlanBuilder::new(inner, recipes, policy, effect_ids)?;
    builder.build_sequential_schedule(inner)
}

struct KernelPlanBuilder<'resources, 'effects> {
    schedule: schedule::KernelPlan,
    seeded: schedule::SeededKernels,
    resources: &'resources LogicalResourceArena,
    flows: model::ResourceFlowIndex,
    recipes: planning::RecipeIndex,
    policy: ParallelPolicy,
    effect_ids: &'effects mut crate::IdSource<EffectToken>,
}

impl planning::PlannedKernel {
    /// Consume the selected body and its graph-local recipe as one operation.
    /// No caller can retain a recipe handle while independently mutating the
    /// graph it addresses.
    fn lower(
        self,
        lowering: &mut KernelPlanBuilder<'_, '_>,
        kernel: schedule::KernelId,
        split_outputs: bool,
    ) -> error::Result<()> {
        let (body, semantic_slots, recipe) = self.into_parts();
        match recipe {
            planning::PlannedRecipe::Filter(candidate) => {
                lowering.lower_parallel_filter(body, kernel, candidate)?
            }
            planning::PlannedRecipe::Reduce(candidate) => {
                lowering.lower_parallel_reduce(body, kernel, candidate)?
            }
            planning::PlannedRecipe::Scan(candidate) => {
                lowering.lower_parallel_scan(body, kernel, candidate)?
            }
            planning::PlannedRecipe::Map => {
                lowering.schedule.commit_kernel(
                    kernel,
                    schedule::KernelRecipeSpec::compute(body, schedule::ComputeKernelKind::Serial),
                )?;
            }
            planning::PlannedRecipe::Serial(recipe) => {
                lowering.commit_serial_kernel(body, kernel, recipe)?
            }
            planning::PlannedRecipe::Unchanged if split_outputs => {
                lowering.schedule.commit_kernel(
                    kernel,
                    schedule::KernelRecipeSpec::compute(body, schedule::ComputeKernelKind::Serial),
                )?;
            }
            planning::PlannedRecipe::Unchanged => {}
        }
        if split_outputs {
            lowering.schedule.set_output_projection(
                kernel,
                semantic_slots.into_iter().map(super::program::OutputSlotId).collect(),
            )?;
        }
        Ok(())
    }
}

impl<'resources, 'effects> KernelPlanBuilder<'resources, 'effects> {
    fn new(
        inner: &'resources AllocatedProgram,
        recipes: planning::RecipeIndex,
        policy: ParallelPolicy,
        effect_ids: &'effects mut crate::IdSource<EffectToken>,
    ) -> error::Result<Self> {
        let resources = &inner.resources;
        let flows = model::ResourceFlowIndex::new(&inner.resources);
        let (schedule, seeded) = schedule::KernelPlan::seed(
            &inner.pipeline,
            &inner.entry_points,
            &inner.resources,
            &inner.region_interner,
        )?;
        Ok(Self {
            schedule,
            seeded,
            resources,
            flows,
            recipes,
            policy,
            effect_ids,
        })
    }

    fn build_parallel_schedule(
        mut self,
        inner: &'resources AllocatedProgram,
    ) -> error::Result<schedule::KernelPlan> {
        self.attach_materializations(inner)?;
        self.schedule_entries(inner)?;
        self.schedule.coalesce_resource_flows(self.flows.flows())?;
        Ok(self.schedule)
    }

    fn build_sequential_schedule(
        mut self,
        inner: &'resources AllocatedProgram,
    ) -> error::Result<schedule::KernelPlan> {
        self.attach_materializations(inner)?;
        self.schedule.select_sequential_recipes()?;
        self.schedule.coalesce_resource_flows(self.flows.flows())?;
        Ok(self.schedule)
    }

    fn schedule_entries(&mut self, inner: &'resources AllocatedProgram) -> error::Result<()> {
        for (index, _) in inner.entry_points.iter().enumerate() {
            let source = SemanticEntryId(index as u32);
            let kernel = self.seeded.entry(source)?;
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
        let (split_outputs, primary, siblings) = plan.into_parts();
        primary.lower(self, kernel, split_outputs)?;
        for sibling in siblings {
            let phase = schedule::PhaseSpec::compute(
                sibling.seed_body(),
                schedule::DomainSelection::Inferred(schedule::KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                schedule::ComputeKernelKind::Serial,
            );
            let sibling_kernel = self.schedule.add_sibling(kernel, phase)?;
            sibling.lower(self, sibling_kernel, split_outputs)?;
        }
        Ok(())
    }

    fn lower_parallel_reduce(
        &mut self,
        mut body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
        candidate: BoundReduce,
    ) -> error::Result<()> {
        use schedule::KernelDomain;

        let phases = self.emit_reduce_entry(&mut body, candidate)?;
        let recipe = schedule::KernelRecipeSpec::compute(body, schedule::ComputeKernelKind::ReducePhase1);
        let after = phases
            .into_iter()
            .map(|phase| {
                schedule::PhaseSpec::compute(
                    phase,
                    schedule::DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                    schedule::ComputeKernelKind::ReduceCombine,
                )
            })
            .collect();
        self.schedule.install_chain(kernel, Vec::new(), recipe, after)?;
        Ok(())
    }

    fn lower_parallel_scan(
        &mut self,
        mut body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
        candidate: BoundScan,
    ) -> error::Result<()> {
        use schedule::KernelDomain;

        let phases = self.emit_scan_entry(&mut body, candidate)?;
        let phase1_domain = self.schedule.domain_of(kernel)?;
        let recipe = schedule::KernelRecipeSpec::compute(body, schedule::ComputeKernelKind::ScanPhase1);
        let after = phases
            .into_iter()
            .enumerate()
            .map(|(phase_index, phase)| {
                schedule::PhaseSpec::compute(
                    phase,
                    schedule::DomainSelection::Explicit(if phase_index == 0 {
                        KernelDomain::Fixed { x: 1, y: 1, z: 1 }
                    } else {
                        phase1_domain.clone()
                    }),
                    if phase_index == 0 {
                        schedule::ComputeKernelKind::ScanBlock
                    } else {
                        schedule::ComputeKernelKind::ScanApplyOffsets
                    },
                )
            })
            .collect();
        self.schedule.install_chain(kernel, Vec::new(), recipe, after)?;
        Ok(())
    }

    fn commit_serial_kernel(
        &mut self,
        mut body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
        recipe: SerialScremaRecipe,
    ) -> error::Result<()> {
        make_screma_serial(&mut body.graph, recipe);
        let recipe = schedule::KernelRecipeSpec::compute(body, schedule::ComputeKernelKind::Serial);
        self.schedule.commit_kernel(kernel, recipe)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "../parallelize_tests.rs"]
mod parallelize_tests;
