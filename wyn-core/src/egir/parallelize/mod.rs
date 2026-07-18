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
//! Organization follows ownership rather than pass chronology: `planning`
//! owns policy, checked errors, indexes, selections, and scratch assignment;
//! `facts` owns short-lived graph observations; `projection` owns entry and
//! route projection; `kernel` owns shared graph-building utilities; `reduce`,
//! `scan`, and `filter` own their complete recipes; `prepare` converts selected
//! semantic operations to scheduled form; and `schedule` owns phase ordering,
//! dependency validation, and publication.

#![deny(clippy::expect_used, clippy::unwrap_used)]

mod facts;
mod filter;
mod kernel;
mod planning;
pub(super) mod prepare;
mod projection;
mod reduce;
mod scan;
mod schedule;
mod target;
#[cfg(test)]
mod test_support;

pub(super) use schedule::ValidatedKernelPlan;
pub use schedule::{
    EntryAbiProjection, KernelDomain, KernelId, KernelKind, KernelPlacement, OutputRouteProjection,
    PublishedKernel, PublishedKernelPlan, ScheduledResource,
};
pub(crate) use target::plan;

use facts::{
    make_screma_serial, parallel_recipe_effect, seg_recipe_family, segmented_recipe_effect,
    semantic_effect, semantic_effect_mut, semantic_node_type, SegScratchFamily,
};
use filter::analyze_filter_candidates;
use kernel::{
    apply_manifest_resource_sizes, can_chunk_view, can_clone_pure_subgraph, chunk_soac_inputs,
    chunk_view_like, dispatch_worker_logical_size, emit_chunk_arithmetic, synthesize_swap_wrapper,
    synthesize_u32_add_function, ChunkInputKind,
};
use planning as error;
use planning::FallbackReason;
use planning::{ParallelPolicy, RecipeSelection};
#[cfg(test)]
use projection::side_effect_output_slots_from_routes;
use projection::{project_kernel_body, split_multidomain_seg_maps, ProjectionSpec, SplitEntry, UnionFind};
use reduce::{analyze_reduce_candidate, ReduceCandidate};
use scan::{analyze_scan_candidate, ScanCandidate, ScanPhase2Spec, ScanPhase3Spec, ScanScratch};
#[cfg(test)]
pub(crate) use test_support::{
    planned_callable_names, preflight_fallback_reasons, FILTER_SCAN_GROUPS, REDUCE_PHASE1_WIDTH,
};

use std::collections::HashSet;

use crate::LookupMap;

use polytype::Type;
use smallvec::smallvec;

use super::graph_ops;
use super::program::{
    AllocatedProgram, CompilerFlowEndpoint, CompilerResourceKind, OutputWriter, ResourceId, SemanticEntry,
    SemanticEntryId, SemanticFunc, SemanticOpId, SemanticResourceDecl, SemanticResourceRef,
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

/// Lower semantic segmented operations into executable kernel entries.
/// Pointwise `SegMap`s remain for `soac_expand`; `SegRed`s become a chunked
/// phase 1 plus a synthesized tree reduction; `SegScan`s become chunk scans,
/// an exclusive scan of block sums, and offset-application phases.
pub(super) fn lower(
    inner: &mut AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<schedule::KernelPlan> {
    use schedule::KernelPlan;

    let policy = ParallelPolicy::default();
    let candidates = planning::allocate_parallel_scratch(inner, policy)?;
    let resources = planning::ResourceIndex::new(&inner.resources)?;
    let flows = planning::ResourceFlowIndex::new(&inner.resources);

    let (schedule, seeded) = KernelPlan::seed(
        &inner.pipeline,
        &inner.entry_points,
        &inner.resources,
        &inner.region_interner,
    );
    let mut lowering = ParallelLowering::new(schedule, resources, candidates, policy, effect_ids);
    lowering.attach_materializations(inner, &flows)?;
    lowering.lower_entries(inner, &seeded)?;
    lowering.schedule.coalesce_resource_flows(flows.flows());
    Ok(lowering.schedule)
}

pub(super) fn lower_sequential(
    inner: &AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<schedule::KernelPlan> {
    let resources = planning::ResourceIndex::new(&inner.resources)?;
    let flows = planning::ResourceFlowIndex::new(&inner.resources);
    let candidates = planning::CandidateIndex::sequential();
    let (plan, _) = schedule::KernelPlan::seed(
        &inner.pipeline,
        &inner.entry_points,
        &inner.resources,
        &inner.region_interner,
    );
    let mut lowering =
        ParallelLowering::new(plan, resources, candidates, ParallelPolicy::default(), effect_ids);
    lowering.attach_materializations(inner, &flows)?;
    lowering.schedule.select_sequential_recipes();
    lowering.schedule.coalesce_resource_flows(flows.flows());
    Ok(lowering.schedule)
}

struct ParallelLowering<'resources, 'effects> {
    schedule: schedule::KernelPlan,
    resources: planning::ResourceIndex<'resources>,
    candidates: planning::CandidateIndex,
    policy: ParallelPolicy,
    effect_ids: &'effects mut crate::IdSource<EffectToken>,
}

enum LoweringSource<'a> {
    Entry {
        id: SemanticEntryId,
        entry: &'a SemanticEntry,
    },
    Materialization(&'a crate::egir::program::MaterializationRequirement),
}

impl LoweringSource<'_> {
    fn entry(&self) -> &SemanticEntry {
        match self {
            Self::Entry { entry, .. } => entry,
            Self::Materialization(requirement) => &requirement.entry,
        }
    }

    fn is_entry(&self) -> bool {
        matches!(self, Self::Entry { .. })
    }

    fn endpoint(&self) -> CompilerFlowEndpoint {
        match self {
            Self::Entry { id, .. } => CompilerFlowEndpoint::Entry(*id),
            Self::Materialization(requirement) => CompilerFlowEndpoint::Materialization(requirement.id),
        }
    }
}

impl<'resources, 'effects> ParallelLowering<'resources, 'effects> {
    fn new(
        schedule: schedule::KernelPlan,
        resources: planning::ResourceIndex<'resources>,
        candidates: planning::CandidateIndex,
        policy: ParallelPolicy,
        effect_ids: &'effects mut crate::IdSource<EffectToken>,
    ) -> Self {
        Self {
            schedule,
            resources,
            candidates,
            policy,
            effect_ids,
        }
    }

    fn lower_entries(
        &mut self,
        inner: &'resources AllocatedProgram,
        seeded: &schedule::SeededKernels,
    ) -> error::Result<()> {
        for (index, entry) in inner.entry_points.iter().enumerate() {
            let source = SemanticEntryId(index as u32);
            let Some(kernel) = seeded.entry(source) else {
                continue;
            };
            self.lower_kernel(LoweringSource::Entry { id: source, entry }, kernel)?;
        }
        Ok(())
    }

    /// Attach allocation-created producer entries in compiler-flow order and
    /// immediately lower the recipe owned by each new physical kernel.
    fn attach_materializations(
        &mut self,
        inner: &'resources AllocatedProgram,
        flows: &planning::ResourceFlowIndex,
    ) -> error::Result<()> {
        let mut ready = std::collections::BTreeSet::new();
        for (_, flow) in flows.flows() {
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
            self.lower_kernel(LoweringSource::Materialization(requirement), kernel)?;
            for upstream in flows.incoming(producer_id) {
                ready.insert((*upstream, producer_id));
            }
        }
        Ok(())
    }

    fn lower_kernel(
        &mut self,
        source: LoweringSource<'resources>,
        kernel: schedule::KernelId,
    ) -> error::Result<()> {
        let endpoint = source.endpoint();
        if self.lower_filter_if_selected(source.entry(), kernel, source.is_entry())? {
            return Ok(());
        }
        if let Some(prepared) = self.candidates.take_prepared(endpoint) {
            return self.lower_prepared_segmented(prepared, kernel);
        }
        let body = super::program::PlannedEntry::project(source.entry())?;
        if !source.is_entry()
            && !segmented_recipe_effect(&body).is_some_and(|located| {
                matches!(
                    &located.effect.kind,
                    SideEffectKind::Soac(SoacEffect(
                        _,
                        Soac::Screma(screma::Op::Reduce { .. } | screma::Op::Scan { .. })
                    ))
                )
            })
        {
            return Ok(());
        }
        self.lower_segmented_kernel_body(body, kernel)
    }

    fn lower_prepared_segmented(
        &mut self,
        prepared: planning::PreparedSegmented,
        kernel: schedule::KernelId,
    ) -> error::Result<()> {
        match prepared {
            planning::PreparedSegmented::Reduce { body, candidate } => {
                self.lower_parallel_reduce(body, kernel, candidate)
            }
            planning::PreparedSegmented::Scan { body, candidate } => {
                self.lower_parallel_scan(body, kernel, candidate)
            }
        }
    }

    fn lower_segmented_kernel_body(
        &mut self,
        body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
    ) -> error::Result<()> {
        use schedule::KernelDomain;
        let Some(located) = segmented_recipe_effect(&body) else {
            return Ok(());
        };
        let SideEffectKind::Soac(SoacEffect(owner, Soac::Screma(op))) = &located.effect.kind else {
            return Err("segmented effect changed kind before recipe selection".into());
        };
        let owner = *owner;
        match op {
            screma::Op::Map { .. } => {
                if let Some(split) = split_multidomain_seg_maps(&body)? {
                    let primary_slots = split.primary_slots;
                    if !split.entries.is_empty() {
                        let recipe = schedule::KernelRecipeSpec::new(
                            split.primary,
                            schedule::KernelKind::OutputDomainProjection,
                        );
                        self.schedule.commit_kernel(kernel, recipe)?;
                    }
                    self.schedule.set_output_projection(
                        kernel,
                        primary_slots.iter().copied().map(super::program::OutputSlotId).collect(),
                    )?;
                    for projected in split.entries {
                        let phase = schedule::PhaseSpec::new(
                            projected.entry,
                            schedule::DomainSelection::Inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                            schedule::KernelKind::OutputDomainProjection,
                        );
                        let projected_kernel = self.schedule.add_sibling(kernel, phase)?;
                        self.schedule.set_output_projection(
                            projected_kernel,
                            projected
                                .semantic_slots
                                .iter()
                                .copied()
                                .map(super::program::OutputSlotId)
                                .collect(),
                        )?;
                    }
                } else {
                    let recipe = schedule::KernelRecipeSpec::new(body, schedule::KernelKind::SerialCompute);
                    self.schedule.commit_kernel(kernel, recipe)?;
                }
            }
            screma::Op::Reduce { .. } => {
                if matches!(self.candidates.reduce(owner)?, RecipeSelection::Parallel(())) {
                    let candidate = match analyze_reduce_candidate(&body, &self.resources)? {
                        RecipeSelection::Parallel(candidate) => candidate,
                        RecipeSelection::Serial(reason) => {
                            return Err(error::ParallelizeError::Invalid(format!(
                                "preflight reduce candidate {owner:?} changed before emission: {reason:?}"
                            )));
                        }
                    };
                    self.lower_parallel_reduce(body, kernel, candidate)?;
                } else {
                    self.commit_serial_kernel(body, kernel)?;
                }
            }
            screma::Op::Scan { .. } => {
                if matches!(self.candidates.scan(owner)?, RecipeSelection::Parallel(())) {
                    let candidate = match analyze_scan_candidate(&body)? {
                        RecipeSelection::Parallel(candidate) => candidate,
                        RecipeSelection::Serial(reason) => {
                            return Err(error::ParallelizeError::Invalid(format!(
                                "preflight scan candidate {owner:?} changed before emission: {reason:?}"
                            )));
                        }
                    };
                    self.lower_parallel_scan(body, kernel, candidate)?;
                } else {
                    self.commit_serial_kernel(body, kernel)?;
                }
            }
            screma::Op::Composite { .. } => self.commit_serial_kernel(body, kernel)?,
        }
        Ok(())
    }

    fn lower_parallel_reduce(
        &mut self,
        mut body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
        candidate: ReduceCandidate,
    ) -> error::Result<()> {
        use schedule::KernelDomain;

        let phases = self.emit_reduce_entry(&mut body, candidate)?;
        let recipe = schedule::KernelRecipeSpec::new(body, schedule::KernelKind::ReducePhase1);
        let mut predecessor = self.schedule.commit_kernel(kernel, recipe)?;
        for phase in phases {
            let phase = schedule::PhaseSpec::new(
                phase,
                schedule::DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                schedule::KernelKind::ReduceCombine,
            );
            predecessor = self.schedule.add_phase_after(predecessor, phase)?;
        }
        Ok(())
    }

    fn lower_parallel_scan(
        &mut self,
        mut body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
        candidate: ScanCandidate,
    ) -> error::Result<()> {
        use schedule::KernelDomain;

        let phases = self.emit_scan_entry(&mut body, candidate)?;
        let phase1_domain =
            self.schedule.domain_of(kernel).unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
        let recipe = schedule::KernelRecipeSpec::new(body, schedule::KernelKind::ScanPhase1);
        let mut predecessor = self.schedule.commit_kernel(kernel, recipe)?;
        for (phase_index, phase) in phases.into_iter().enumerate() {
            let phase = schedule::PhaseSpec::new(
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
            );
            predecessor = self.schedule.add_phase_after(predecessor, phase)?;
        }
        Ok(())
    }

    fn commit_serial_kernel(
        &mut self,
        mut body: super::program::PlannedEntry,
        kernel: schedule::KernelId,
    ) -> error::Result<()> {
        let site = segmented_recipe_effect(&body).map(|located| located.site).ok_or_else(|| {
            error::ParallelizeError::Invalid("serial recipe has no pending kernel SegOp".into())
        })?;
        make_screma_serial(&mut body.graph, site)?;
        let recipe = schedule::KernelRecipeSpec::new(body, schedule::KernelKind::SerialCompute);
        self.schedule.commit_kernel(kernel, recipe)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "../parallelize_tests.rs"]
mod parallelize_tests;
