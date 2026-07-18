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

#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

mod chunk;
mod error;
mod facts;
mod filter_recipe;
mod kernel;
mod materialization;
mod planning;
mod policy;
pub(crate) mod prepare;
mod projection;
mod reduce;
mod scan;
pub mod schedule;

use chunk::{can_chunk_view, can_clone_pure_subgraph, chunk_soac_inputs, chunk_view_like, ChunkInputKind};
pub(crate) use facts::parallel_recipe_effect;
use facts::{
    make_screma_serial, project_root_index, seg_recipe_family, segmented_recipe_effect, semantic_effect,
    semantic_effect_mut, semantic_node_type, storage_resource_under, SegScratchFamily,
};
use filter_recipe::{analyze_filter_candidates, lower_materialized_filters, lower_runtime_filters};
use kernel::{
    apply_manifest_resource_sizes, dispatch_worker_logical_size, emit_chunk_arithmetic, emit_resource_len,
    emit_semantic_resource_len, required_resource, synthesize_swap_wrapper, synthesize_u32_add_function,
};
use materialization::attach_materializations;
#[cfg(test)]
pub(crate) use planning::preflight_fallback_reasons;
pub(crate) use planning::FallbackReason;
use planning::RecipeSelection;
use policy::ParallelPolicy;
pub use policy::PHASE2_WIDTH;
#[cfg(test)]
pub(crate) use policy::{FILTER_SCAN_GROUPS, REDUCE_PHASE1_WIDTH};
#[cfg(test)]
use projection::side_effect_output_slots_from_routes;
use projection::{project_kernel_body, split_multidomain_seg_maps, SplitEntry, UnionFind};
use reduce::{analyze_reduce_candidate, bind_reduce_candidate, emit_reduce_entry};
use scan::{
    analyze_scan_candidate, bind_scan_candidate, emit_scan_entry, synthesize_phase2_scan,
    synthesize_phase3_scan,
};

use std::collections::HashSet;

use crate::LookupMap;

use polytype::Type;
use smallvec::smallvec;

use super::graph_ops;
use super::program::{
    AllocatedProgram, CompilerResourceKind, OutputWriter, ResourceId, SemanticEntry, SemanticEntryId,
    SemanticFunc, SemanticOpId, SemanticResourceDecl, SemanticResourceRef,
};
use super::soac::{filter, screma};
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
pub(crate) fn lower(
    inner: &mut AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<schedule::KernelPlan> {
    use schedule::KernelPlan;

    let policy = ParallelPolicy::default();
    let candidates = planning::allocate_parallel_scratch(inner, policy)?;
    let resources = planning::ResourceIndex::new(&inner.resources)?;
    let flows = planning::ResourceFlowIndex::new(&inner.resources);

    let (mut schedule, seeded) = KernelPlan::seed(
        &inner.pipeline,
        &inner.entry_points,
        &inner.resources,
        &inner.region_interner,
    );
    attach_materializations(
        inner,
        &mut schedule,
        &resources,
        &flows,
        &candidates,
        policy,
        effect_ids,
    )?;
    lower_materialized_filters(inner, &mut schedule, &resources, &candidates, policy, effect_ids)?;
    let lowered_filters = lower_runtime_filters(
        inner,
        &seeded,
        &mut schedule,
        &resources,
        &candidates,
        policy,
        effect_ids,
    )?;
    for (index, entry) in inner.entry_points.iter().enumerate() {
        let source = SemanticEntryId(index as u32);
        if lowered_filters.contains(&source) {
            continue;
        }
        let Some(kernel) = seeded.entry(source) else {
            continue;
        };
        let body = super::program::PlannedEntry::project(entry)?;
        plan_segmented_kernel_body(
            body,
            kernel,
            &mut schedule,
            &resources,
            &candidates,
            policy,
            effect_ids,
        )?;
    }
    schedule.coalesce_resource_flows(flows.flows());
    Ok(schedule)
}

pub(crate) fn lower_sequential(
    inner: &AllocatedProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<schedule::KernelPlan> {
    let resources = planning::ResourceIndex::new(&inner.resources)?;
    let flows = planning::ResourceFlowIndex::new(&inner.resources);
    let candidates = planning::CandidateIndex::sequential();
    let (mut plan, _) = schedule::KernelPlan::seed(
        &inner.pipeline,
        &inner.entry_points,
        &inner.resources,
        &inner.region_interner,
    );
    attach_materializations(
        inner,
        &mut plan,
        &resources,
        &flows,
        &candidates,
        ParallelPolicy::default(),
        effect_ids,
    )?;
    plan.select_sequential_recipes();
    plan.coalesce_resource_flows(flows.flows());
    Ok(plan)
}

fn plan_segmented_kernel_body(
    mut body: super::program::PlannedEntry,
    kernel: schedule::KernelId,
    schedule: &mut schedule::KernelPlan,
    resources: &planning::ResourceIndex<'_>,
    candidates: &planning::CandidateIndex,
    policy: ParallelPolicy,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<()> {
    use schedule::KernelDomain;
    let Some((_, _, effect)) = segmented_recipe_effect(&body) else {
        return Ok(());
    };
    let SideEffectKind::Soac(SoacEffect(owner, Soac::Screma(op))) = &effect.kind else {
        return Err("segmented effect changed kind before recipe selection".into());
    };
    let owner = *owner;
    match op {
        screma::Op::Map { .. } => {
            if let Some(split) = split_multidomain_seg_maps(&body)? {
                let primary_slots = split.primary_slots;
                if !split.entries.is_empty() {
                    schedule.commit_kernel(
                        kernel,
                        split.primary,
                        schedule::KernelKind::OutputDomainProjection,
                    )?;
                }
                schedule.set_output_projection(
                    kernel,
                    primary_slots.iter().copied().map(super::program::OutputSlotId).collect(),
                )?;
                for projected in split.entries {
                    let projected_kernel = schedule.add_sibling(
                        kernel,
                        projected.entry,
                        schedule::DomainSelection::Inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                        schedule::KernelKind::OutputDomainProjection,
                    )?;
                    schedule.set_output_projection(
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
                schedule.commit_kernel(kernel, body, schedule::KernelKind::SerialCompute)?;
            }
        }
        screma::Op::Reduce { .. } => {
            if matches!(candidates.reduce(owner)?, RecipeSelection::Parallel(())) {
                let candidate = match analyze_reduce_candidate(&body, resources)? {
                    RecipeSelection::Parallel(candidate) => candidate,
                    RecipeSelection::Serial(reason) => {
                        return Err(error::ParallelizeError::Invalid(format!(
                            "preflight reduce candidate {owner:?} changed before emission: {reason:?}"
                        )));
                    }
                };
                let plan = bind_reduce_candidate(candidate, resources)?;
                let phases = emit_reduce_entry(&mut body, plan, schedule, resources, policy, effect_ids)?;
                let mut predecessor =
                    schedule.commit_kernel(kernel, body, schedule::KernelKind::ReducePhase1)?;
                for phase in phases {
                    predecessor = schedule.add_phase_after(
                        predecessor,
                        phase,
                        schedule::DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                        schedule::KernelKind::ReduceCombine,
                    )?;
                }
            } else {
                commit_serial_kernel(body, kernel, schedule)?;
            }
        }
        screma::Op::Scan { .. } => {
            if matches!(candidates.scan(owner)?, RecipeSelection::Parallel(())) {
                let candidate = match analyze_scan_candidate(&body)? {
                    RecipeSelection::Parallel(candidate) => candidate,
                    RecipeSelection::Serial(reason) => {
                        return Err(error::ParallelizeError::Invalid(format!(
                            "preflight scan candidate {owner:?} changed before emission: {reason:?}"
                        )));
                    }
                };
                let plan = bind_scan_candidate(candidate, resources)?;
                let phases = emit_scan_entry(&mut body, plan, schedule, resources, policy, effect_ids)?;
                let phase1_domain =
                    schedule.domain_of(kernel).unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
                let mut predecessor =
                    schedule.commit_kernel(kernel, body, schedule::KernelKind::ScanPhase1)?;
                for (phase_index, phase) in phases.into_iter().enumerate() {
                    predecessor = schedule.add_phase_after(
                        predecessor,
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
                    )?;
                }
            } else {
                commit_serial_kernel(body, kernel, schedule)?;
            }
        }
        screma::Op::Composite { .. } => {
            commit_serial_kernel(body, kernel, schedule)?;
        }
    }
    Ok(())
}

fn commit_serial_kernel(
    mut body: super::program::PlannedEntry,
    kernel: schedule::KernelId,
    schedule: &mut schedule::KernelPlan,
) -> error::Result<()> {
    let (block, effect, _) = segmented_recipe_effect(&body).ok_or_else(|| {
        error::ParallelizeError::Invalid("serial recipe has no pending kernel SegOp".into())
    })?;
    make_screma_serial(&mut body.graph, block, effect)?;
    schedule.commit_kernel(kernel, body, schedule::KernelKind::SerialCompute)?;
    Ok(())
}

#[cfg(test)]
#[path = "../parallelize_tests.rs"]
mod parallelize_tests;
