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
//! policy, checked errors, and immutable indexes; `planning` owns recipe
//! analysis and scratch assignment; `facts` owns short-lived graph
//! observations; `projection` owns entry and route projection; `kernel` owns
//! shared graph-building utilities; `prefix_scan` owns builders shared by scan
//! and filter recipes; `reduce`, `scan`, and `filter` own their algorithms;
//! `prepare` converts selected semantic operations to scheduled form; and
//! `schedule` owns phase ordering, validation, publication, and physical
//! construction.

#![deny(clippy::expect_used, clippy::unwrap_used)]

mod facts;
mod filter;
mod kernel;
mod model;
mod planning;
mod prefix_scan;
pub(super) mod prepare;
mod projection;
mod reduce;
mod scan;
mod schedule;
#[cfg(test)]
mod test_support;

use facts::{
    make_screma_serial, parallel_recipe_effect, reduce_recipe_eligibility, scan_recipe_eligibility,
    segmented_recipe, semantic_effect, semantic_effect_mut, semantic_node_type, LocatedReduce, LocatedScan,
    SegmentedRecipe,
};
use filter::analyze_filter_candidates;
use kernel::{
    apply_manifest_resource_sizes, can_chunk_view, can_clone_pure_subgraph, chunk_soac_inputs,
    chunk_view_like, dispatch_worker_logical_size, emit_chunk_arithmetic, synthesize_swap_wrapper,
    synthesize_u32_add_function, ChunkInputKind,
};
use model as error;
use model::{FallbackReason, ParallelPolicy, RecipeSelection};
use prefix_scan::{ScanPhase2Spec, ScanPhase3Spec, ScanScratch};
#[cfg(test)]
use projection::side_effect_output_slots_from_routes;
use projection::{project_kernel_body, split_multidomain_seg_maps, ProjectionSpec, UnionFind};
use reduce::{analyze_reduce_candidate, BoundReduce};
use scan::{analyze_scan_candidate, BoundScan};
pub(super) use schedule::KernelPlan;
pub use schedule::{
    EntryAbiProjection, KernelDomain, KernelId, KernelKind, KernelPhaseSummary, KernelPlacement,
    KernelPlanSummary, OutputRouteProjection, ScheduledResource,
};
#[cfg(test)]
pub(crate) use test_support::{
    planned_callable_names, preflight_fallback_reasons, FILTER_SCAN_GROUPS, REDUCE_PHASE1_WIDTH,
};

use std::collections::HashSet;

use crate::LookupMap;

use polytype::Type;
use smallvec::smallvec;

use super::from_tlc::ConvertError;
use super::graph_ops;
use super::program::{
    AllocatedProgram, CompilerFlowEndpoint, CompilerResourceKind, OutputWriter, PhysicalProgram,
    ResourceId, SemanticEntry, SemanticEntryId, SemanticFunc, SemanticOpId, SemanticResourceDecl,
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

pub(crate) fn plan(
    mut inner: AllocatedProgram,
    binding_ids: &mut IdSource<u32>,
    effect_ids: &mut IdSource<EffectToken>,
    profile: LoweringProfile,
) -> Result<(PhysicalProgram, KernelPlanSummary), ConvertError> {
    let descriptor = inner.pipeline.clone();
    let kernel_plan = if profile.schedule == SchedulePolicy::Parallel {
        build_parallel_plan(&mut inner, effect_ids)
    } else {
        build_sequential_plan(&inner, effect_ids)
    }
    .map_err(|error| ConvertError::Internal(error.to_string()))?;

    kernel_plan.finalize(inner, binding_ids, profile, descriptor)
}

/// Analyze target recipes, allocate their scratch resources, and build the
/// executable parallel kernel plan.
fn build_parallel_plan(
    inner: &mut AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<schedule::KernelPlan> {
    let policy = ParallelPolicy::default();
    let plan = planning::analyze(inner, policy)?;
    let candidates = plan.allocate(inner)?;
    let builder = KernelPlanBuilder::new(inner, candidates, policy, effect_ids)?;
    builder.build_parallel_schedule(inner)
}

/// Build a kernel plan that selects serial recipes without allocating
/// algorithm-specific parallel scratch resources.
fn build_sequential_plan(
    inner: &AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<schedule::KernelPlan> {
    let policy = ParallelPolicy::default();
    let candidates = planning::CandidateIndex::sequential();
    let builder = KernelPlanBuilder::new(inner, candidates, policy, effect_ids)?;
    builder.build_sequential_schedule(inner)
}

struct KernelPlanBuilder<'resources, 'effects> {
    schedule: schedule::KernelPlan,
    seeded: schedule::SeededKernels,
    resources: model::ResourceIndex<'resources>,
    flows: model::ResourceFlowIndex,
    candidates: planning::CandidateIndex,
    policy: ParallelPolicy,
    effect_ids: &'effects mut crate::IdSource<EffectToken>,
}

impl<'resources, 'effects> KernelPlanBuilder<'resources, 'effects> {
    fn new(
        inner: &'resources AllocatedProgram,
        candidates: planning::CandidateIndex,
        policy: ParallelPolicy,
        effect_ids: &'effects mut crate::IdSource<EffectToken>,
    ) -> error::Result<Self> {
        let resources = model::ResourceIndex::new(&inner.resources)?;
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
            candidates,
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
            let requirement = inner
                .materializations
                .get(id.0 as usize)
                .filter(|requirement| requirement.id == id)
                .ok_or_else(|| {
                    error::ParallelizeError::Invalid(format!(
                        "materialization flow references missing requirement {id:?}"
                    ))
                })?;
            let kernel = self.schedule.add_materialization_before(consumer, requirement)?;
            self.lower_endpoint(CompilerFlowEndpoint::Materialization(requirement.id), kernel)?;
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
        let Some(plan) = self.candidates.take_endpoint(endpoint)? else {
            return Ok(());
        };
        let (split_outputs, primary, siblings) = plan.into_parts();
        primary.lower(self, kernel, split_outputs)?;
        for sibling in siblings {
            let phase = schedule::PhaseSpec::new(
                sibling.seed_body(),
                schedule::DomainSelection::Inferred(schedule::KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                schedule::KernelKind::SerialCompute,
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
        let recipe = schedule::KernelRecipeSpec::new(body, schedule::KernelKind::ReducePhase1);
        let after = phases
            .into_iter()
            .map(|phase| {
                schedule::PhaseSpec::new(
                    phase,
                    schedule::DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                    schedule::KernelKind::ReduceCombine,
                )
            })
            .collect();
        self.schedule.install_chain(kernel, schedule::KernelChainSpec::new(recipe).with_after(after))?;
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
        let recipe = schedule::KernelRecipeSpec::new(body, schedule::KernelKind::ScanPhase1);
        let after = phases
            .into_iter()
            .enumerate()
            .map(|(phase_index, phase)| {
                schedule::PhaseSpec::new(
                    phase,
                    schedule::DomainSelection::Explicit(if phase_index == 0 {
                        KernelDomain::Fixed { x: 1, y: 1, z: 1 }
                    } else {
                        phase1_domain.clone()
                    }),
                    if phase_index == 0 {
                        schedule::KernelKind::ScanBlock
                    } else {
                        schedule::KernelKind::ScanApplyOffsets
                    },
                )
            })
            .collect();
        self.schedule.install_chain(kernel, schedule::KernelChainSpec::new(recipe).with_after(after))?;
        Ok(())
    }

    fn commit_serial_kernel(
        &mut self,
        mut body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
        recipe: facts::SerialScremaRecipe,
    ) -> error::Result<()> {
        make_screma_serial(&mut body.graph, recipe);
        let recipe = schedule::KernelRecipeSpec::new(body, schedule::KernelKind::SerialCompute);
        self.schedule.commit_kernel(kernel, recipe)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "../parallelize_tests.rs"]
mod parallelize_tests;
