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
mod materialization;
mod planning;
mod policy;
pub(crate) mod prepare;
mod projection;
pub mod schedule;

use chunk::{can_chunk_view, can_clone_pure_subgraph, chunk_soac_inputs, chunk_view_like, ChunkInputKind};
use materialization::attach_materializations;
use policy::ParallelPolicy;
pub use policy::PHASE2_WIDTH;
#[cfg(test)]
pub(crate) use policy::{FILTER_SCAN_GROUPS, REDUCE_PHASE1_WIDTH};
#[cfg(test)]
use projection::side_effect_output_slots_from_routes;
use projection::{project_kernel_body, split_multidomain_seg_maps, SplitEntry, UnionFind};

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
    let candidates = planning::CandidateIndex::default();
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
            if candidates.reduce(owner) {
                let plan = analyze_reduce_entry(&body, resources)?.ok_or_else(|| {
                    error::ParallelizeError::Invalid(format!(
                        "preflight reduce candidate {owner:?} changed before emission"
                    ))
                })?;
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
            if candidates.scan(owner) {
                let plan = analyze_scan_entry(&body, resources)?.ok_or_else(|| {
                    error::ParallelizeError::Invalid(format!(
                        "preflight scan candidate {owner:?} changed before emission"
                    ))
                })?;
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

fn lower_runtime_filters(
    inner: &AllocatedProgram,
    seeded: &schedule::SeededKernels,
    schedule: &mut schedule::KernelPlan,
    resources: &planning::ResourceIndex<'_>,
    candidates: &planning::CandidateIndex,
    policy: ParallelPolicy,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<HashSet<SemanticEntryId>> {
    use schedule::KernelDomain;
    let mut lowered = HashSet::new();
    for (index, entry) in inner.entry_points.iter().enumerate() {
        let source = SemanticEntryId(index as u32);
        let Some(seeded_kernel) = seeded.entry(source) else {
            continue;
        };
        let Some(candidate) = analyze_filter_candidate(entry) else {
            continue;
        };
        let semantic_id = candidate.semantic_id;
        if !candidates.filter(semantic_id) {
            continue;
        }
        let analysis = analyze_filter_entry(entry, resources)?
            .ok_or_else(|| format!("preflight filter candidate {semantic_id:?} changed before emission"))?;
        let projected = super::program::PlannedEntry::project(entry)?;
        let groups = if let Some(split) = split_multidomain_seg_maps(&projected)? {
            let mut groups = vec![SplitEntry {
                entry: split.primary,
                semantic_slots: split.primary_slots,
                semantic_ops: split.primary_semantic_ops,
            }];
            groups.extend(split.entries);
            groups
        } else {
            vec![SplitEntry {
                entry: projected,
                semantic_slots: (0..entry.outputs.len()).collect(),
                semantic_ops: [semantic_id].into_iter().collect(),
            }]
        };
        let mut filter_group = None;
        for group in groups {
            let kernel = if group.entry.name == entry.name {
                seeded_kernel
            } else {
                schedule.add_sibling(
                    seeded_kernel,
                    group.entry.clone(),
                    schedule::DomainSelection::Inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                    schedule::KernelKind::SerialCompute,
                )?
            };
            if group.semantic_ops.contains(&semantic_id) {
                filter_group = Some((kernel, group.entry, group.semantic_slots));
            } else {
                plan_segmented_kernel_body(
                    group.entry,
                    kernel,
                    schedule,
                    resources,
                    candidates,
                    policy,
                    effect_ids,
                )?;
                schedule.set_output_projection(
                    kernel,
                    group.semantic_slots.iter().copied().map(super::program::OutputSlotId).collect(),
                )?;
            }
        }
        let Some((kernel, filter_entry, filter_slots)) = filter_group else {
            return Err(format!("runtime filter {semantic_id:?} was lost during output splitting").into());
        };
        lower_filter_kernel(
            filter_entry,
            kernel,
            analysis,
            schedule,
            resources,
            policy,
            effect_ids,
        )?;
        schedule.set_output_projection(
            kernel,
            filter_slots.iter().copied().map(super::program::OutputSlotId).collect(),
        )?;
        lowered.insert(source);
    }
    Ok(lowered)
}

fn lower_filter_kernel(
    filter_entry: super::program::PlannedEntry,
    kernel: schedule::KernelId,
    analysis: FilterAnalysis,
    schedule: &mut schedule::KernelPlan,
    resources: &planning::ResourceIndex<'_>,
    policy: ParallelPolicy,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<()> {
    let family =
        build_filter_kernel_family(filter_entry, analysis, schedule, resources, policy, effect_ids)?;
    install_filter_kernel_family(kernel, family, schedule)
}

struct FilterKernelFamily {
    domain: schedule::KernelDomain,
    work: filter::WorkBuffers,
    flags: super::program::PlannedEntry,
    scan: super::program::PlannedEntry,
    combine: super::program::PlannedEntry,
    apply_offsets: super::program::PlannedEntry,
    scatter: super::program::PlannedEntry,
    scan_workgroup_width: u32,
    scan_groups: u32,
}

fn build_filter_kernel_family(
    filter_entry: super::program::PlannedEntry,
    analysis: FilterAnalysis,
    schedule: &mut schedule::KernelPlan,
    resources: &planning::ResourceIndex<'_>,
    policy: ParallelPolicy,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<FilterKernelFamily> {
    use crate::interface::StorageRole;
    use schedule::KernelDomain;

    let FilterAnalysis { candidate, work } = analysis;
    let FilterCandidate { space, len_out, .. } = candidate;
    let domain = schedule::domain_from_space(&space).unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let declaration = |resource, role| filter_resource_declaration(resources, resource, role, &u32_ty);

    let mut flags_storage = filter_entry
        .resource_declarations
        .iter()
        .filter(|declaration| declaration.role == StorageRole::Input)
        .cloned()
        .collect::<Vec<_>>();
    flags_storage.push(declaration(work.flags, StorageRole::Output)?);
    let flags = project_kernel_body(
        &filter_entry,
        format!("{}_filter_flags", filter_entry.name),
        filter_entry.execution_model.clone(),
        Vec::new(),
        Vec::new(),
        flags_storage,
        Type::Constructed(TypeName::Unit, vec![]),
    )?;
    let scan_storage = [
        (work.flags, StorageRole::Input),
        (work.offsets, StorageRole::Output),
        (work.block_sums, StorageRole::Output),
    ]
    .into_iter()
    .map(|(resource, role)| declaration(resource, role))
    .collect::<Result<Vec<_>, _>>()?;
    let mut scan = project_kernel_body(
        &filter_entry,
        format!("{}_filter_scan", filter_entry.name),
        ExecutionModel::Compute {
            local_size: (policy.reduce_phase1_width, 1, 1),
        },
        Vec::new(),
        Vec::new(),
        scan_storage,
        Type::Constructed(TypeName::Unit, vec![]),
    )?;
    let zero = graph_ops::intern_u32(&mut scan.graph, 0, None);

    let add_name = format!("{}_filter_scan_add", filter_entry.name);
    let add_fn = synthesize_u32_add_function(add_name.clone(), filter_entry.span);
    schedule.define_callable(add_fn);
    let mut combine = synthesize_phase2_scan(
        &scan.name,
        add_name.clone(),
        u32_ty.clone(),
        &scan.graph,
        zero,
        required_resource(work.block_sums),
        required_resource(work.block_offsets),
        Some(required_resource(len_out)),
        effect_ids,
    )
    .map_err(|error| {
        format!(
            "failed to synthesize filter scan for `{}`: {error}",
            filter_entry.name
        )
    })?;
    apply_manifest_resource_sizes(&mut combine, resources)?;
    let swap_wrapper_name = format!("{}_filter_scan_add_offsets", filter_entry.name);
    let swap_wrapper =
        synthesize_swap_wrapper(swap_wrapper_name, add_name, u32_ty.clone(), filter_entry.span);
    let swap_region = schedule.define_callable(swap_wrapper);
    let mut apply_offsets = synthesize_phase3_scan(
        &scan.name,
        swap_region,
        Type::Constructed(TypeName::UInt(32), vec![]),
        required_resource(work.offsets),
        required_resource(work.block_offsets),
        policy.reduce_phase1_width,
        effect_ids,
    )?;
    apply_manifest_resource_sizes(&mut apply_offsets, resources)?;

    let mut scatter_resources = filter_entry.resource_declarations.clone();
    for declaration in &mut scatter_resources {
        if declaration.resource == len_out {
            declaration.role = StorageRole::Input;
        }
    }
    scatter_resources.push(declaration(work.flags, StorageRole::Input)?);
    scatter_resources.push(declaration(work.offsets, StorageRole::Input)?);
    scatter_resources.push(declaration(work.block_offsets, StorageRole::Input)?);
    let scatter = project_kernel_body(
        &filter_entry,
        filter_entry.name.clone(),
        filter_entry.execution_model.clone(),
        filter_entry.outputs.iter().map(|output| output.inner.clone()).collect(),
        filter_entry.output_routes.clone(),
        scatter_resources,
        filter_entry.return_ty.clone(),
    )?;
    Ok(FilterKernelFamily {
        domain,
        work,
        flags,
        scan,
        combine,
        apply_offsets,
        scatter,
        scan_workgroup_width: policy.reduce_phase1_width,
        scan_groups: policy.filter_scan_groups,
    })
}

fn install_filter_kernel_family(
    kernel: schedule::KernelId,
    family: FilterKernelFamily,
    schedule: &mut schedule::KernelPlan,
) -> error::Result<()> {
    use schedule::KernelDomain;

    let FilterKernelFamily {
        domain,
        work,
        flags,
        scan,
        combine,
        apply_offsets,
        scatter,
        scan_workgroup_width,
        scan_groups,
    } = family;
    schedule.commit_filter_kernel(
        kernel,
        scatter,
        schedule::KernelKind::FilterScatter,
        filter::Plan::Scatter(filter::ParallelConfig {
            buffers: work,
            scan_workgroup_width,
        }),
    )?;
    schedule.add_filter_phase_before(
        kernel,
        flags,
        schedule::DomainSelection::Explicit(domain.clone()),
        schedule::KernelKind::FilterFlags,
        filter::Plan::Flags(filter::ParallelConfig {
            buffers: work,
            scan_workgroup_width,
        }),
    )?;
    // The scan runs a fixed worker grid so each worker scans a large chunk;
    // flags and scatter remain one-thread-per-input-element.
    schedule.add_filter_phase_before(
        kernel,
        scan,
        schedule::DomainSelection::Explicit(KernelDomain::Fixed {
            x: scan_groups,
            y: 1,
            z: 1,
        }),
        schedule::KernelKind::FilterScan,
        filter::Plan::Scan(filter::ParallelConfig {
            buffers: work,
            scan_workgroup_width,
        }),
    )?;
    schedule.add_phase_before(
        kernel,
        combine,
        schedule::DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
        schedule::KernelKind::FilterCombine,
    )?;
    schedule.add_phase_before(
        kernel,
        apply_offsets,
        schedule::DomainSelection::Explicit(domain),
        schedule::KernelKind::FilterScan,
    )?;
    Ok(())
}

fn filter_resource_declaration(
    resources: &planning::ResourceIndex<'_>,
    reference: SemanticResourceRef,
    role: crate::interface::StorageRole,
    elem_ty: &Type<TypeName>,
) -> error::Result<SemanticResourceDecl> {
    let resource = reference.0;
    let logical = resources.get(resource)?;
    Ok(SemanticResourceDecl {
        resource: reference,
        role,
        elem_ty: elem_ty.clone(),
        size: logical.size.clone(),
    })
}

fn lower_materialized_filters(
    inner: &AllocatedProgram,
    schedule: &mut schedule::KernelPlan,
    resources: &planning::ResourceIndex<'_>,
    candidates: &planning::CandidateIndex,
    policy: ParallelPolicy,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<()> {
    for requirement in &inner.materializations {
        if requirement.kind != super::program::MaterializationKind::RuntimeArray {
            continue;
        }
        let endpoint = super::program::CompilerFlowEndpoint::Materialization(requirement.id);
        let kernel = schedule.kernel_for_flow_source(endpoint).ok_or_else(|| {
            format!(
                "runtime-array materialization {:?} was not scheduled",
                requirement.id
            )
        })?;
        let Some(candidate) = analyze_filter_candidate(&requirement.entry) else {
            continue;
        };
        if !candidates.filter(candidate.semantic_id) {
            continue;
        }
        let analysis = analyze_filter_entry(&requirement.entry, resources)?.ok_or_else(|| {
            format!(
                "preflight filter candidate {:?} changed before materialization emission",
                candidate.semantic_id
            )
        })?;
        let body = super::program::PlannedEntry::project(&requirement.entry)?;
        lower_filter_kernel(body, kernel, analysis, schedule, resources, policy, effect_ids)?;
    }
    Ok(())
}

pub(super) struct FilterCandidate {
    pub semantic_id: SemanticOpId,
    pub space: SegSpace,
    pub len_out: SemanticResourceRef,
}

struct FilterAnalysis {
    candidate: FilterCandidate,
    work: filter::WorkBuffers,
}

pub(super) fn analyze_filter_candidate(entry: &SemanticEntry) -> Option<FilterCandidate> {
    let mut analysis = None;
    for effect in entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects) {
        let SideEffectKind::Soac(SoacEffect(
            semantic_id,
            Soac::Filter(filter::Op {
                state:
                    filter::SemanticState {
                        space,
                        storage: filter::Output::Runtime { length, .. },
                    },
                ..
            }),
        )) = &effect.kind
        else {
            continue;
        };
        let filter::RuntimeLength::Stored(len_out) = length else {
            return None;
        };
        let semantic_id = *semantic_id;
        if analysis
            .replace(FilterCandidate {
                semantic_id,
                space: space.clone(),
                len_out: *len_out,
            })
            .is_some()
        {
            return None;
        }
    }
    analysis
}

fn analyze_filter_entry(
    entry: &SemanticEntry,
    resources: &planning::ResourceIndex<'_>,
) -> error::Result<Option<FilterAnalysis>> {
    let Some(candidate) = analyze_filter_candidate(entry) else {
        return Ok(None);
    };
    let work = filter_work_buffers(candidate.semantic_id, resources)?;
    Ok(Some(FilterAnalysis { candidate, work }))
}

fn filter_work_buffers(
    owner: SemanticOpId,
    resources: &planning::ResourceIndex<'_>,
) -> error::Result<filter::WorkBuffers> {
    let resource_id = |kind, slot| {
        resources.exactly_one_at(owner, kind, slot).map(|resource| SemanticResourceRef(resource.id))
    };
    Ok(filter::WorkBuffers {
        flags: resource_id(CompilerResourceKind::FilterFlags, 0)?,
        offsets: resource_id(CompilerResourceKind::FilterOffsets, 1)?,
        block_sums: resource_id(CompilerResourceKind::FilterScanBlockSums, 2)?,
        block_offsets: resource_id(CompilerResourceKind::FilterScanBlockOffsets, 3)?,
    })
}

fn required_resource(reference: SemanticResourceRef) -> ResourceId {
    reference.0
}

fn apply_manifest_resource_sizes(
    entry: &mut super::program::PlannedEntry,
    resources: &planning::ResourceIndex<'_>,
) -> error::Result<()> {
    for declaration in &mut entry.resource_declarations {
        let resource = declaration.resource.0;
        let logical = resources.get(resource)?;
        declaration.size = logical.size.clone();
    }
    Ok(())
}

/// Workgroup width for the single-workgroup tree-reduce phase 2: `W` threads
/// grid-stride the `T` partials into shared memory, then reduce in-shared with
/// a log-`W` tree. Kept modest so `W * sizeof(elem)` stays within the
/// workgroup shared-memory budget (256 × a 36-byte tuple ≈ 9 KB). The phase2
/// The published compute stage must dispatch this same width.
#[allow(clippy::too_many_arguments)]
fn build_tree_reduce_phase2(
    b: &mut super::builder::EntryBuilder,
    op_func: String,
    elem_ty: Type<TypeName>,
    init_nid: NodeId,
    partials_resource: ResourceId,
    phase1_graph: &super::types::EGraph,
    accumulator_value: NodeId,
    output_stores: &[(NodeId, NodeId)],
    width: u32,
) -> Result<(), String> {
    let w = width;
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let view_arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(0),
            // resource stamped by `intern_resource_view`.
            crate::types::no_buffer(),
        ],
    );

    // ---- entry block: lid, partials view + length, shared view, result view ----
    let entry_bid = b.graph_mut().skeleton.entry;
    let (graph, control_headers, eff) = b.construction_parts_mut();

    let lid = graph_ops::intern_intrinsic(
        graph,
        catalog().known().local_id,
        smallvec![],
        u32_ty.clone(),
        None,
    );
    let partials_view =
        graph_ops::intern_resource_view(graph, partials_resource, view_arr_ty.clone(), None);
    let len = emit_resource_len(graph, partials_resource);
    // Workgroup-shared `array<elem, W>` (id 0 within this entry).
    let shared_view = graph_ops::emit_workgroup_view(graph, 0, w, view_arr_ty.clone(), None);
    let w_nid = graph_ops::intern_u32(graph, w, None);
    let zero_u32 = graph_ops::intern_u32(graph, 0, None);

    // Contiguous per-thread chunk over `partials` (not strided): thread `lid`
    // reduces `partials[start .. end)`, so the tree combines `shared[0..W]` in
    // global order and the reduction stays valid for associative,
    // non-commutative operators.
    //   chunk = ceil(len / W);  start = lid * chunk;  end = min(start+chunk, len)
    let w_minus_1 = graph_ops::intern_u32(graph, w - 1, None);
    let len_plus = graph_ops::intern_binop(graph, "+", len, w_minus_1, u32_ty.clone(), None);
    let chunk = graph_ops::intern_binop(graph, "/", len_plus, w_nid, u32_ty.clone(), None);
    let start = graph_ops::intern_binop(graph, "*", lid, chunk, u32_ty.clone(), None);
    let start_plus = graph_ops::intern_binop(graph, "+", start, chunk, u32_ty.clone(), None);
    let u32_min = catalog()
        .lookup_by_any_name("u32.min")
        .ok_or_else(|| "required builtin `u32.min` is missing from the catalog".to_string())?;
    let end = graph_ops::intern_intrinsic(
        graph,
        u32_min.id,
        smallvec![start_plus, len],
        u32_ty.clone(),
        None,
    );

    // ---- blocks ----
    let grid_header = graph.skeleton.create_block();
    let grid_body = graph.skeleton.create_block();
    let grid_cont = graph.skeleton.create_block();
    let grid_after = graph.skeleton.create_block();
    let tree_header = graph.skeleton.create_block();
    let tree_body = graph.skeleton.create_block();
    let tree_then = graph.skeleton.create_block();
    let tree_sel_merge = graph.skeleton.create_block();
    let tree_cont = graph.skeleton.create_block();
    let tree_after = graph.skeleton.create_block();
    let write_blk = graph.skeleton.create_block();
    let end_blk = graph.skeleton.create_block();

    // grid_header params: (acc, i)
    let acc_in = graph.add_block_param(grid_header, elem_ty.clone());
    let i_in = graph.add_block_param(grid_header, u32_ty.clone());

    // entry → grid_header(init, start)
    graph.skeleton.blocks[entry_bid].term = SkeletonTerminator::Branch {
        target: grid_header,
        args: vec![init_nid, start],
    };

    // grid_header: i < end ? grid_body : grid_after(acc)
    let grid_cond = graph_ops::intern_binop(graph, "<", i_in, end, bool_ty.clone(), None);
    graph.skeleton.blocks[grid_header].term = SkeletonTerminator::CondBranch {
        cond: grid_cond,
        then_target: grid_body,
        then_args: vec![],
        else_target: grid_after,
        else_args: vec![acc_in],
    };
    control_headers.insert(
        grid_header,
        ControlHeader::Loop {
            merge: grid_after,
            continue_block: grid_cont,
        },
    );

    // grid_body: acc' = op(acc, partials[i]); → grid_cont(acc')
    let elem_i =
        graph_ops::emit_view_load(graph, grid_body, partials_view, i_in, elem_ty.clone(), eff, None);
    let acc_next = graph.intern_pure(
        PureOp::Call(op_func.clone()),
        smallvec![acc_in, elem_i],
        elem_ty.clone(),
        None,
    );
    graph.skeleton.blocks[grid_body].term = SkeletonTerminator::Branch {
        target: grid_cont,
        args: vec![acc_next],
    };

    // grid_cont(acc_c): i_next = i + W; → grid_header(acc_c, i_next)
    let acc_c = graph.add_block_param(grid_cont, elem_ty.clone());
    let one_u32 = graph_ops::intern_u32(graph, 1, None);
    let i_next = graph_ops::intern_binop(graph, "+", i_in, one_u32, u32_ty.clone(), None);
    graph.skeleton.blocks[grid_cont].term = SkeletonTerminator::Branch {
        target: grid_header,
        args: vec![acc_c, i_next],
    };

    // grid_after(acc_final): shared[lid] = acc_final; barrier; → tree_header(1)
    let acc_final = graph.add_block_param(grid_after, elem_ty.clone());
    graph_ops::emit_storage_store(
        graph,
        grid_after,
        shared_view,
        lid,
        acc_final,
        elem_ty.clone(),
        eff,
        None,
    );
    graph_ops::emit_workgroup_barrier(graph, grid_after, eff);
    graph.skeleton.blocks[grid_after].term = SkeletonTerminator::Branch {
        target: tree_header,
        args: vec![one_u32],
    };

    // Grow an adjacent-pair tree from stride 1. This preserves source order
    // for associative, non-commutative operators.
    let stride_in = graph.add_block_param(tree_header, u32_ty.clone());
    let stride_cond = graph_ops::intern_binop(graph, "<", stride_in, w_nid, bool_ty.clone(), None);
    graph.skeleton.blocks[tree_header].term = SkeletonTerminator::CondBranch {
        cond: stride_cond,
        then_target: tree_body,
        then_args: vec![],
        else_target: tree_after,
        else_args: vec![],
    };
    control_headers.insert(
        tree_header,
        ControlHeader::Loop {
            merge: tree_after,
            continue_block: tree_cont,
        },
    );

    // Only the first lane in each adjacent pair combines the two runs.
    let two = graph_ops::intern_u32(graph, 2, None);
    let pair_width = graph_ops::intern_binop(graph, "*", stride_in, two, u32_ty.clone(), None);
    let lane_in_pair = graph_ops::intern_binop(graph, "%", lid, pair_width, u32_ty.clone(), None);
    let active = graph_ops::intern_binop(graph, "==", lane_in_pair, zero_u32, bool_ty.clone(), None);
    graph.skeleton.blocks[tree_body].term = SkeletonTerminator::CondBranch {
        cond: active,
        then_target: tree_then,
        then_args: vec![],
        else_target: tree_sel_merge,
        else_args: vec![],
    };
    control_headers.insert(
        tree_body,
        ControlHeader::Selection {
            merge: tree_sel_merge,
        },
    );

    // tree_then: shared[lid] = op(shared[lid], shared[lid+stride]); → tree_sel_merge
    let a = graph_ops::emit_view_load(graph, tree_then, shared_view, lid, elem_ty.clone(), eff, None);
    let lid_plus = graph_ops::intern_binop(graph, "+", lid, stride_in, u32_ty.clone(), None);
    let bb = graph_ops::emit_view_load(
        graph,
        tree_then,
        shared_view,
        lid_plus,
        elem_ty.clone(),
        eff,
        None,
    );
    let combined = graph.intern_pure(
        PureOp::Call(op_func.clone()),
        smallvec![a, bb],
        elem_ty.clone(),
        None,
    );
    graph_ops::emit_storage_store(
        graph,
        tree_then,
        shared_view,
        lid,
        combined,
        elem_ty.clone(),
        eff,
        None,
    );
    graph.skeleton.blocks[tree_then].term = SkeletonTerminator::Branch {
        target: tree_sel_merge,
        args: vec![],
    };

    // tree_sel_merge → tree_cont   (selection merge; barrier lives past it)
    graph.skeleton.blocks[tree_sel_merge].term = SkeletonTerminator::Branch {
        target: tree_cont,
        args: vec![],
    };

    // tree_cont: barrier; stride_next = stride*2; → tree_header(stride_next)
    graph_ops::emit_workgroup_barrier(graph, tree_cont, eff);
    let stride_next = graph_ops::intern_binop(graph, "*", stride_in, two, u32_ty.clone(), None);
    graph.skeleton.blocks[tree_cont].term = SkeletonTerminator::Branch {
        target: tree_header,
        args: vec![stride_next],
    };

    // tree_after: lid == 0 ? write_blk : end_blk   (selection)
    let is_zero = graph_ops::intern_binop(graph, "==", lid, zero_u32, bool_ty.clone(), None);
    graph.skeleton.blocks[tree_after].term = SkeletonTerminator::CondBranch {
        cond: is_zero,
        then_target: write_blk,
        then_args: vec![],
        else_target: end_blk,
        else_args: vec![],
    };
    control_headers.insert(tree_after, ControlHeader::Selection { merge: end_blk });

    // write_blk: combined = shared[0]; replay each captured output store reading
    // `combined` in place of the per-thread accumulator value. A scalar reduce
    // has one store (`out[0] = combined`); a tuple-element reduce decomposes
    // across one store per field.
    let s0 = graph_ops::emit_view_load(
        graph,
        write_blk,
        shared_view,
        zero_u32,
        elem_ty.clone(),
        eff,
        None,
    );
    for &(place, value) in output_stores {
        let cloned_place = graph_ops::clone_pure_subgraph(phase1_graph, graph, place)?;
        let cloned_value = graph_ops::clone_pure_subgraph_substituting(
            phase1_graph,
            graph,
            value,
            &[(accumulator_value, s0)],
        )?;
        graph_ops::emit_store(graph, write_blk, cloned_place, cloned_value, eff, None);
    }
    graph.skeleton.blocks[write_blk].term = SkeletonTerminator::Branch {
        target: end_blk,
        args: vec![],
    };

    // end_blk is the exit; `build()` finalizes it with Return(None).
    b.set_current_block(end_blk);
    Ok(())
}

/// Emit the chunk-arithmetic preamble (`tid`, `chunk_start`,
/// `chunk_len`) as pure nodes in `graph`. Caller supplies the
/// `input_len` NodeId (typed `u32`) — for StorageView inputs that's a
/// `_w_intrinsic_storage_len(set, binding)` call; for Range inputs
/// it's the Range's own `len` operand. Returns
/// `(tid, chunk_start, chunk_len)`.
fn emit_chunk_arithmetic(
    graph: &mut super::types::EGraph,
    total_threads: u32,
    input_len: NodeId,
) -> Result<(NodeId, NodeId, NodeId), String> {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    // The chunk arithmetic runs in the input's *index* type: storage-view
    // inputs index in u32 (`_w_intrinsic_storage_len`), Range inputs in the
    // range's own element type (typically i32). Computing in u32 and feeding
    // a u32 `chunk_start`/`chunk_len` into an i32 Range produced an
    // `OpCompositeConstruct` whose constituents didn't match the i32
    // `{start, step, len}` struct (spirv-val rejected it). Derive the index
    // type from `input_len` and emit all arithmetic there.
    let index_ty = graph
        .types
        .get(&input_len)
        .cloned()
        .ok_or_else(|| format!("chunk input length {input_len:?} has no type"))?;
    let is_u32 = index_ty == u32_ty;

    // `tid`/`num_workgroups` are u32 intrinsics. The returned `tid` stays u32
    // (callers use it as a `partials[tid]` storage index); the index-typed
    // copies feed the chunk math.
    let tid = graph_ops::intern_intrinsic(
        graph,
        catalog().known().thread_id,
        smallvec![],
        u32_ty.clone(),
        None,
    );
    let nwg = graph_ops::intern_intrinsic(
        graph,
        catalog().known().num_workgroups,
        smallvec![],
        u32_ty.clone(),
        None,
    );
    let tid_idx = cast_u32_to_index(graph, tid, &index_ty)?;
    let nwg_idx = cast_u32_to_index(graph, nwg, &index_ty)?;

    // Runtime total thread count = num_workgroups.x * workgroup width. With a
    // `derived_from_input_length` dispatch (~ceil(n / width) workgroups) this
    // makes chunk_size ≈ 1, so each thread reduces ~one element — a saturating
    // grid rather than a fixed `total_threads`-wide one. `total_threads` is the
    // compile-time per-workgroup width.
    let wg_width = intern_index_lit(graph, total_threads, &index_ty);
    let total = graph_ops::intern_binop(graph, "*", nwg_idx, wg_width, index_ty.clone(), None);
    let one = intern_index_lit(graph, 1, &index_ty);
    let total_minus_one = graph_ops::intern_binop(graph, "-", total, one, index_ty.clone(), None);
    let len_plus = graph_ops::intern_binop(graph, "+", input_len, total_minus_one, index_ty.clone(), None);
    let chunk_size = graph_ops::intern_binop(graph, "/", len_plus, total, index_ty.clone(), None);
    let raw_chunk_start = graph_ops::intern_binop(graph, "*", tid_idx, chunk_size, index_ty.clone(), None);
    let min_name = if is_u32 { "u32.min" } else { "i32.min" };
    let min_op =
        catalog().lookup_by_any_name(min_name).ok_or_else(|| format!("{} not in catalog", min_name))?;
    // Clamp idle workers to the end before subtraction. For n < workers this
    // produces `(start=n,len=0)` instead of underflowing `n-start`.
    let chunk_start = graph_ops::intern_intrinsic(
        graph,
        min_op.id,
        smallvec![raw_chunk_start, input_len],
        index_ty.clone(),
        None,
    );
    let remaining = graph_ops::intern_binop(graph, "-", input_len, chunk_start, index_ty.clone(), None);
    let chunk_len =
        graph_ops::intern_intrinsic(graph, min_op.id, smallvec![chunk_size, remaining], index_ty, None);
    Ok((tid, chunk_start, chunk_len))
}

/// Integer literal `n` typed as `index_ty` (`u32` → `PureOp::Uint`, else
/// `PureOp::Int`).
fn intern_index_lit(graph: &mut super::types::EGraph, n: u32, index_ty: &Type<TypeName>) -> NodeId {
    let op = match index_ty {
        Type::Constructed(TypeName::UInt(32), _) => super::types::PureOp::Uint(n.to_string()),
        _ => super::types::PureOp::Int(n.to_string()),
    };
    graph.intern_pure(op, smallvec![], index_ty.clone(), None)
}

/// Cast a u32 value into `index_ty`: identity for u32, else the per-type
/// bitcast intrinsic (`i32.u32`).
fn cast_u32_to_index(
    graph: &mut super::types::EGraph,
    v: NodeId,
    index_ty: &Type<TypeName>,
) -> Result<NodeId, String> {
    match index_ty {
        Type::Constructed(TypeName::UInt(32), _) => Ok(v),
        Type::Constructed(TypeName::Int(32), _) => {
            let conv = catalog()
                .lookup_by_any_name("i32.u32")
                .ok_or_else(|| "i32.u32 not in catalog".to_string())?;
            Ok(graph_ops::intern_intrinsic(
                graph,
                conv.id,
                smallvec![v],
                index_ty.clone(),
                None,
            ))
        }
        other => Err(format!("chunk arithmetic: unsupported index type {:?}", other)),
    }
}

fn emit_semantic_resource_len(graph: &mut super::types::EGraph, resource: SemanticResourceRef) -> NodeId {
    emit_resource_len(graph, resource.0)
}

fn emit_resource_len(graph: &mut super::types::EGraph, resource: ResourceId) -> NodeId {
    graph.intern_pure(
        PureOp::ResourceLen(SemanticResourceRef(resource)),
        smallvec![],
        Type::Constructed(TypeName::UInt(32), vec![]),
        None,
    )
}

fn dispatch_worker_logical_size(elem_ty: &Type<TypeName>) -> super::program::LogicalSize {
    crate::ssa::layout::type_byte_size(elem_ty).map_or(super::program::LogicalSize::Unspecified, |bytes| {
        super::program::LogicalSize::SameAsDispatch {
            elem_bytes: bytes as u32,
        }
    })
}

/// Programmatic phase 2 synthesis where the neutral element is a
/// (possibly compound) pure subgraph cloned from phase 1. Used by the
/// Screma reduce path for any NE shape (scalar literal, tuple, array,
/// etc.).
/// Synthesize a reduce phase-2 combine entry. Its `partials` buffer is typed as
/// the (possibly tuple) accumulator element; the workgroup tree reduces them to
/// one combined value and replays the accumulator's captured output stores
/// (`output_stores`, `(place, value)` nodes from `phase1_graph`) against it,
/// substituting `accumulator_value` for the combined result. `output_decls`
/// declares the output bindings this entry writes. Screma's multi-accumulator
/// path passes a `_phase2_combine_{i}` `full_name` per combiner.
fn synthesize_phase2_reduce_cloning_ne_named(
    full_name: String,
    op_func: String,
    elem_ty: Type<TypeName>,
    phase1_graph: &super::types::EGraph,
    phase1_ne_nid: NodeId,
    partials_resource: ResourceId,
    accumulator_value: NodeId,
    output_stores: &[(NodeId, NodeId)],
    output_decls: &[(ResourceId, Type<TypeName>, super::program::LogicalSize)],
    phase2_width: u32,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<super::program::PlannedEntry, String> {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(full_name, (phase2_width, 1, 1), effect_ids);
    b.declare_intermediate_storage_sized(
        partials_resource,
        elem_ty.clone(),
        dispatch_worker_logical_size(&elem_ty),
    );
    for (resource, ty, size) in output_decls {
        b.declare_output_storage_sized(*resource, ty.clone(), size.clone());
    }

    let init_nid = graph_ops::clone_pure_subgraph(phase1_graph, b.graph_mut(), phase1_ne_nid)?;
    build_tree_reduce_phase2(
        &mut b,
        op_func,
        elem_ty,
        init_nid,
        partials_resource,
        phase1_graph,
        accumulator_value,
        output_stores,
        phase2_width,
    )?;
    Ok(b.build())
}

fn segmented_screma_effect(graph: &EGraph) -> Option<(BlockId, usize, &SideEffect)> {
    graph.skeleton.blocks.iter().find_map(|(block, contents)| {
        contents.side_effects.iter().enumerate().find_map(|(index, effect)| {
            matches!(
                &effect.kind,
                SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op)))
                    if matches!(
                        op.semantic_state(),
                        screma::SemanticState::Segmented {
                            placement: screma::Placement::Kernel,
                            ..
                        }
                    )
            )
            .then_some((block, index, effect))
        })
    })
}

fn segmented_recipe_effect(entry: &super::program::PlannedEntry) -> Option<(BlockId, usize, &SideEffect)> {
    if let Some(effect) = segmented_screma_effect(&entry.graph) {
        return Some(effect);
    }
    if !matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
        return None;
    }
    let mut promoted = None;
    for (block, contents) in &entry.graph.skeleton.blocks {
        for (index, effect) in contents.side_effects.iter().enumerate() {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                continue;
            };
            if matches!(
                op.semantic_state(),
                screma::SemanticState::Segmented {
                    placement: screma::Placement::LaneLocal,
                    output_slots,
                    ..
                } if !output_slots.is_empty()
            ) && matches!(op, screma::Op::Reduce { .. } | screma::Op::Scan { .. })
            {
                if promoted.is_some() {
                    return None;
                }
                promoted = Some((block, index, effect));
            }
        }
    }
    promoted
}

pub(crate) fn parallel_recipe_effect(
    entry: &super::program::PlannedEntry,
) -> Option<(BlockId, usize, &SideEffect)> {
    segmented_recipe_effect(entry).or_else(|| {
        entry.graph.skeleton.blocks.iter().find_map(|(block, contents)| {
            contents.side_effects.iter().enumerate().find_map(|(index, effect)| {
                matches!(&effect.kind, SideEffectKind::Soac(SoacEffect(_, Soac::Filter(_))))
                    .then_some((block, index, effect))
            })
        })
    })
}

fn make_screma_serial(graph: &mut EGraph, block_id: BlockId, index: usize) -> error::Result<()> {
    let effect = graph
        .skeleton
        .blocks
        .get_mut(block_id)
        .and_then(|block| block.side_effects.get_mut(index))
        .ok_or_else(|| {
            error::ParallelizeError::Invalid(format!("stale segmented effect site {block_id:?}:{index}"))
        })?;
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &mut effect.kind else {
        return Err("segmented effect site no longer contains a Screma operation".into());
    };
    *op.semantic_state_mut() = screma::SemanticState::Serial;
    Ok(())
}

fn semantic_effect(graph: &EGraph, block: BlockId, index: usize) -> error::Result<&SideEffect> {
    graph.skeleton.blocks.get(block).and_then(|contents| contents.side_effects.get(index)).ok_or_else(
        || error::ParallelizeError::Invalid(format!("stale semantic effect site {block:?}:{index}")),
    )
}

fn semantic_effect_mut(graph: &mut EGraph, block: BlockId, index: usize) -> error::Result<&mut SideEffect> {
    graph
        .skeleton
        .blocks
        .get_mut(block)
        .and_then(|contents| contents.side_effects.get_mut(index))
        .ok_or_else(|| {
            error::ParallelizeError::Invalid(format!("stale semantic effect site {block:?}:{index}"))
        })
}

fn semantic_node_type(graph: &EGraph, node: NodeId) -> error::Result<Type<TypeName>> {
    graph
        .types
        .get(&node)
        .cloned()
        .ok_or_else(|| error::ParallelizeError::Invalid(format!("semantic node {node:?} has no type")))
}

fn project_root_index(graph: &super::types::EGraph, value: NodeId, root: NodeId) -> Option<u32> {
    let mut cur = value;
    let mut last_index = None;
    loop {
        if cur == root {
            return last_index;
        }
        match graph.nodes.get(cur) {
            Some(super::types::ENode::Pure {
                op: super::types::PureOp::Project { index },
                operands,
            }) => {
                last_index = Some(*index);
                cur = *operands.first()?;
            }
            _ => return None,
        }
    }
}

fn storage_resource_under(graph: &super::types::EGraph, root: NodeId) -> Option<SemanticResourceRef> {
    wyn_graph::find_map_reachable(
        [root],
        wyn_graph::WalkOrder::DepthFirst,
        |node, out| {
            if let Some(value) = graph.nodes.get(node) {
                out.extend(value.children());
            }
        },
        |node| graph_ops::extract_storage_view_source(graph, node),
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SegScratchFamily {
    Reduce,
    Scan,
}

/// Parse the eligibility gates shared by candidate selection and lowering.
fn seg_recipe_family(se: &SideEffect) -> Option<SegScratchFamily> {
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &se.kind else {
        return None;
    };
    let valid_placement = match op.semantic_state() {
        screma::SemanticState::Segmented {
            placement: screma::Placement::Kernel,
            ..
        } => true,
        screma::SemanticState::Segmented {
            placement: screma::Placement::LaneLocal,
            output_slots,
            ..
        } => !output_slots.is_empty() && matches!(op, screma::Op::Reduce { .. } | screma::Op::Scan { .. }),
        screma::SemanticState::Serial => false,
    };
    if !valid_placement {
        return None;
    }
    let lanes = op.lanes();
    let operators = op.operators();
    let maps_are_output_views = lanes.maps.iter().all(|map| map.destination.is_output_view());
    match op {
        screma::Op::Reduce { .. } => {
            if operators.iter().any(|op| !op.combine.captures.is_empty())
                || lanes.inputs.is_empty()
                || !maps_are_output_views
                || !operators.iter().all(|op| op.destination.is_unplaced_fresh())
            {
                return None;
            }
            Some(SegScratchFamily::Reduce)
        }
        screma::Op::Scan { .. } => {
            if operators.len() != 1
                || !operators[0].combine.captures.is_empty()
                || lanes.inputs.len() != 1
                || !maps_are_output_views
                || !operators.iter().all(|op| op.destination.is_output_view())
            {
                return None;
            }
            Some(SegScratchFamily::Scan)
        }
        screma::Op::Map { .. } | screma::Op::Composite { .. } => None,
    }
}

pub(super) struct ReduceCandidate {
    pub block: BlockId,
    pub effect: usize,
    pub owner: SemanticOpId,
    pub scratch_types: Vec<Type<TypeName>>,
    combine_regions: Vec<RegionId>,
    input_views: Vec<(NodeId, Type<TypeName>)>,
    map_output_view_operands: Vec<usize>,
    map_count: usize,
    neutral_values: Vec<NodeId>,
    result: NodeId,
    stores: Vec<Vec<ReduceOutputStore>>,
    outputs: Vec<Vec<(ResourceId, Type<TypeName>, super::program::LogicalSize)>>,
}

struct ReduceAnalysis {
    candidate: ReduceCandidate,
    partials: Vec<ResourceId>,
}

struct ReduceOutputStore {
    location: (BlockId, usize),
    place: NodeId,
    value: NodeId,
    writer: Option<super::types::EffectToken>,
}

pub(super) fn analyze_reduce_candidate(
    entry: &super::program::PlannedEntry,
    resources: &planning::ResourceIndex<'_>,
) -> Option<ReduceCandidate> {
    let (block, effect, _) = segmented_recipe_effect(entry)?;
    let side_effect = entry.graph.skeleton.blocks.get(block)?.side_effects.get(effect)?;
    if seg_recipe_family(side_effect)? != SegScratchFamily::Reduce {
        return None;
    }
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(screma::Op::Reduce { lanes, operators, .. }))) =
        &side_effect.kind
    else {
        return None;
    };
    let operators = operators.iter().collect::<Vec<_>>();
    let n_inputs = lanes.inputs.len();
    let n_accs = operators.len();
    let n_maps = lanes.maps.len();
    let operand = |index| side_effect.operand_nodes.get(index).copied();
    if !(0..n_inputs).all(|index| {
        operand(index)
            .is_some_and(|view| can_chunk_view(&entry.graph, view, ChunkInputKind::StorageOrRange))
    }) {
        return None;
    }
    let map_base = n_inputs;
    if !(0..n_maps).all(|index| {
        operand(map_base + index)
            .is_some_and(|view| can_chunk_view(&entry.graph, view, ChunkInputKind::StorageOnly))
    }) {
        return None;
    }
    let result = side_effect.result?;
    let owner = *side_effect.kind.soac_id()?;
    if operators.iter().any(|operator| !can_clone_pure_subgraph(&entry.graph, operator.neutral, &[])) {
        return None;
    }
    let scratch_types = operators
        .iter()
        .map(|operator| entry.graph.types.get(&operator.neutral).cloned())
        .collect::<Option<Vec<_>>>()?;
    if scratch_types.iter().any(|ty| crate::ssa::layout::type_byte_size(ty).is_none()) {
        return None;
    }
    let input_views = (0..n_inputs)
        .map(|index| {
            let view = operand(index)?;
            Some((view, entry.graph.types.get(&view)?.clone()))
        })
        .collect::<Option<Vec<_>>>()?;
    let combine_regions = operators.iter().map(|operator| operator.combine.region).collect();
    let neutral_values = operators.iter().map(|operator| operator.neutral).collect();
    let map_output_view_operands = (0..n_maps).map(|index| map_base + index).collect();
    let mut stores = (0..n_accs).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut outputs: Vec<Vec<(ResourceId, Type<TypeName>, super::program::LogicalSize)>> =
        vec![Vec::new(); n_accs];
    for (block_id, block) in &entry.graph.skeleton.blocks {
        for (effect_index, effect) in block.side_effects.iter().enumerate() {
            if !matches!(effect.kind, SideEffectKind::Effect(EffectOp::Store)) {
                continue;
            }
            let (Some(&place), Some(&value)) = (effect.operand_nodes.first(), effect.operand_nodes.get(1))
            else {
                continue;
            };
            let Some(root) = project_root_index(&entry.graph, value, result)
                .or_else(|| (value == result && n_maps + n_accs == 1).then_some(0))
            else {
                continue;
            };
            let accumulator = root as usize;
            if accumulator < n_maps || accumulator - n_maps >= n_accs {
                continue;
            }
            let accumulator = accumulator - n_maps;
            if !can_clone_pure_subgraph(&entry.graph, place, &[])
                || !can_clone_pure_subgraph(&entry.graph, value, &[result])
            {
                return None;
            }
            stores[accumulator].push(ReduceOutputStore {
                location: (block_id, effect_index),
                place,
                value,
                writer: effect.effects.map(|(_, writer)| writer),
            });
            if let Some(resource) = storage_resource_under(&entry.graph, place).map(|resource| resource.0) {
                let logical = resources.get(resource).ok()?;
                let output = entry.resource_declarations.iter().find(|declaration| {
                    declaration.role == crate::interface::StorageRole::Output
                        && declaration.resource.0 == resource
                });
                if let Some(output) = output {
                    if !outputs[accumulator].iter().any(|(candidate, _, _)| *candidate == resource) {
                        outputs[accumulator].push((resource, output.elem_ty.clone(), logical.size.clone()));
                    }
                }
            }
        }
    }
    (0..n_accs).all(|index| !stores[index].is_empty() && !outputs[index].is_empty()).then_some(
        ReduceCandidate {
            block,
            effect,
            owner,
            scratch_types,
            combine_regions,
            input_views,
            map_output_view_operands,
            map_count: n_maps,
            neutral_values,
            result,
            stores,
            outputs,
        },
    )
}

fn analyze_reduce_entry(
    entry: &super::program::PlannedEntry,
    resources: &planning::ResourceIndex<'_>,
) -> error::Result<Option<ReduceAnalysis>> {
    let Some(candidate) = analyze_reduce_candidate(entry, resources) else {
        return Ok(None);
    };
    let partials = resources
        .ordered_slots(
            candidate.owner,
            CompilerResourceKind::ReducePartial,
            0,
            candidate.scratch_types.len(),
        )?
        .iter()
        .map(|resource| resource.id)
        .collect();
    Ok(Some(ReduceAnalysis { candidate, partials }))
}

fn emit_reduce_entry(
    entry: &mut super::program::PlannedEntry,
    analysis: ReduceAnalysis,
    schedule: &schedule::KernelPlan,
    resources: &planning::ResourceIndex<'_>,
    policy: ParallelPolicy,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<Vec<super::program::PlannedEntry>> {
    let ReduceAnalysis {
        candidate,
        partials: partial_resources,
    } = analysis;
    let ReduceCandidate {
        block: block_id,
        effect: idx,
        scratch_types: elem_tys,
        combine_regions,
        input_views: input_view_data,
        map_output_view_operands,
        map_count: n_maps,
        neutral_values: init_nids,
        result: screma_result_nid,
        stores,
        outputs: acc_output_decls,
        ..
    } = candidate;
    debug_assert_eq!(
        segmented_recipe_effect(entry).map(|(block, effect, _)| (block, effect)),
        Some((block_id, idx))
    );
    let total_threads = policy.reduce_phase1_width;
    let n_accs = stores.len();
    let mut acc_stores = (0..n_accs).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut drop_locations = Vec::new();
    let mut dropped_writers = std::collections::HashSet::new();
    for (accumulator, stores) in stores.into_iter().enumerate() {
        for store in stores {
            acc_stores[accumulator].push((store.place, store.value));
            drop_locations.push(store.location);
            dropped_writers.extend(store.writer);
        }
    }
    let reduce_funcs = schedule.callable_names(combine_regions);

    debug_assert_eq!(n_accs, elem_tys.len());
    // 3. Chunk all input views and every map output view; swap them back
    // into the Screma operand list.
    let chunked = chunk_soac_inputs(
        &mut entry.graph,
        &input_view_data,
        total_threads,
        ChunkInputKind::StorageOrRange,
        "SegRed",
    )?;
    let chunk_start = chunked.chunk_start;
    let chunk_len = chunked.chunk_len;
    {
        let se = semantic_effect_mut(&mut entry.graph, block_id, idx)?;
        for (i, &new_view) in chunked.views.iter().enumerate() {
            let operand = se.operand_nodes.get_mut(i).ok_or_else(|| {
                error::ParallelizeError::Invalid(format!("reduce recipe is missing input operand {i}"))
            })?;
            *operand = new_view;
        }
    }
    for (map_index, operand_index) in map_output_view_operands.iter().enumerate() {
        let orig_view = *semantic_effect(&entry.graph, block_id, idx)?
            .operand_nodes
            .get(*operand_index)
            .ok_or_else(|| {
                error::ParallelizeError::Invalid(format!(
                    "reduce recipe is missing map output operand {operand_index}"
                ))
            })?;
        let view_ty = semantic_node_type(&entry.graph, orig_view)?;
        let chunked_view = chunk_view_like(
            &mut entry.graph,
            orig_view,
            view_ty,
            chunk_start,
            chunk_len,
            ChunkInputKind::StorageOnly,
            &format!("SegRed map output {map_index}"),
        )?;
        let operand = semantic_effect_mut(&mut entry.graph, block_id, idx)?
            .operand_nodes
            .get_mut(*operand_index)
            .ok_or_else(|| {
                error::ParallelizeError::Invalid(format!(
                    "reduce recipe lost map output operand {operand_index}"
                ))
            })?;
        *operand = chunked_view;
    }

    // 5. Phase 1 stores each thread's whole accumulator value to `partials[tid]`
    // and no longer writes the outputs. `accumulator_value` is the hash-consed
    // `Project{acc_pos}(screma_result)` node — phase 2 substitutes it for the
    // combined result when replaying the captured stores.
    let accumulator_values: Vec<NodeId> = (0..n_accs)
        .map(|acc_i| {
            entry.graph.intern_pure(
                super::types::PureOp::Project {
                    index: (n_maps + acc_i) as u32,
                },
                smallvec![screma_result_nid],
                elem_tys[acc_i].clone(),
                None,
            )
        })
        .collect();
    // Drop the decomposed output stores (highest index first per block).
    drop_locations.sort_by(|a, b| b.1.cmp(&a.1));
    for (bid, sx) in drop_locations {
        let effects = &mut entry
            .graph
            .skeleton
            .blocks
            .get_mut(bid)
            .ok_or_else(|| {
                error::ParallelizeError::Invalid(format!(
                    "reduce output store references stale block {bid:?}"
                ))
            })?
            .side_effects;
        if sx >= effects.len() {
            return Err(error::ParallelizeError::Invalid(format!(
                "reduce output store references stale effect {bid:?}:{sx}"
            )));
        }
        effects.remove(sx);
    }
    for route in &mut entry.output_routes {
        route.writers.retain(
            |writer| !matches!(writer, OutputWriter::Effect(effect) if dropped_writers.contains(effect)),
        );
    }
    for acc_i in 0..n_accs {
        let elem_ty = elem_tys[acc_i].clone();
        let arr_ty = Type::Constructed(
            TypeName::Array,
            vec![
                elem_ty.clone(),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
                Type::Variable(0),
                crate::types::no_buffer(),
            ],
        );
        let partials_view =
            graph_ops::intern_resource_view(&mut entry.graph, partial_resources[acc_i], arr_ty, None);
        graph_ops::emit_storage_store(
            &mut entry.graph,
            block_id,
            partials_view,
            chunked.tid,
            accumulator_values[acc_i],
            elem_ty,
            effect_ids,
            None,
        );
        // Clear the moved output bindings from phase 1; register partials.
        for (resource, _, _) in &acc_output_decls[acc_i] {
            let logical = resources.get(*resource)?;
            if let Some(binding) = logical.host_binding() {
                for output in &mut entry.outputs {
                    if output.storage_binding() == Some(binding) {
                        output.make_storage_internal();
                    }
                }
            }
        }
        entry.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(partial_resources[acc_i]),
            role: crate::interface::StorageRole::Intermediate,
            elem_ty: elem_tys[acc_i].clone(),
            size: resources.get(partial_resources[acc_i])?.size.clone(),
        });
    }
    // A moved output binding may also carry an Output storage declaration (e.g. a
    // hoisted prepass result). Phase 1 no longer writes it; phase 2 owns it.
    let moved: std::collections::HashSet<ResourceId> =
        acc_output_decls.iter().flatten().map(|(b, _, _)| *b).collect();
    entry.resource_declarations.retain(|declaration| {
        declaration.role != crate::interface::StorageRole::Output
            || !moved.contains(&declaration.resource.0)
    });

    // 6. Synthesize one phase 2 entry per accumulator. Dropping the phase-1
    // stores leaves their pure place/value subgraphs available for projection.
    let mut phase2s = Vec::with_capacity(n_accs);
    for acc_i in 0..n_accs {
        let phase2_name = if n_accs == 1 {
            format!("{}_phase2_combine", entry.name)
        } else {
            format!("{}_phase2_combine_{}", entry.name, acc_i)
        };
        let phase2 = synthesize_phase2_reduce_cloning_ne_named(
            phase2_name,
            reduce_funcs[acc_i].clone(),
            elem_tys[acc_i].clone(),
            &entry.graph,
            init_nids[acc_i],
            partial_resources[acc_i],
            accumulator_values[acc_i],
            &acc_stores[acc_i],
            &acc_output_decls[acc_i],
            policy.reduce_phase2_width,
            effect_ids,
        )?;
        phase2s.push(phase2);
    }
    // Scheduling consumed the semantic SegRed. Phase 1 is now an ordinary
    // per-invocation Screma over the thread's chunk; `soac_expand` lowers that
    // local loop while the synthesized phase-2 entries combine its partials.
    make_screma_serial(&mut entry.graph, block_id, idx)?;
    Ok(phase2s)
}

pub(super) struct ScanCandidate {
    pub block: BlockId,
    pub effect: usize,
    pub owner: SemanticOpId,
    pub scratch_type: Type<TypeName>,
    step_region: RegionId,
    combine_region: RegionId,
    step_captures: Vec<NodeId>,
    neutral: NodeId,
    input_view: NodeId,
    input_view_type: Type<TypeName>,
    map_output_view_operands: Vec<usize>,
    scan_output_view_operand: usize,
    scan_output_storage: SemanticResourceRef,
    scan_output_view_type: Type<TypeName>,
}

struct ScanAnalysis {
    candidate: ScanCandidate,
    block_sums: ResourceId,
    block_offsets: ResourceId,
}

pub(super) fn analyze_scan_candidate(entry: &super::program::PlannedEntry) -> Option<ScanCandidate> {
    let (block, effect, _) = segmented_recipe_effect(entry)?;
    let side_effect = entry.graph.skeleton.blocks.get(block)?.side_effects.get(effect)?;
    if seg_recipe_family(side_effect)? != SegScratchFamily::Scan {
        return None;
    }
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(screma::Op::Scan { lanes, operators, .. }))) =
        &side_effect.kind
    else {
        return None;
    };
    let operator = &operators.first;
    if !can_clone_pure_subgraph(&entry.graph, operator.neutral, &[]) {
        return None;
    }
    let input = *side_effect.operand_nodes.first()?;
    if !can_chunk_view(&entry.graph, input, ChunkInputKind::StorageOrRange) {
        return None;
    }
    let output_base = lanes.inputs.len();
    if !(0..lanes.maps.len()).all(|index| {
        side_effect
            .operand_nodes
            .get(output_base + index)
            .is_some_and(|view| can_chunk_view(&entry.graph, *view, ChunkInputKind::StorageOnly))
    }) {
        return None;
    }
    let scan_output_view_operand = output_base + lanes.maps.len();
    let scan_output = *side_effect.operand_nodes.get(scan_output_view_operand)?;
    let scan_output_storage = graph_ops::extract_storage_view_source(&entry.graph, scan_output)?;
    let owner = *side_effect.kind.soac_id()?;
    let scratch_type = entry.graph.types.get(&operator.neutral)?.clone();
    crate::ssa::layout::type_byte_size(&scratch_type)?;
    let input_view_type = entry.graph.types.get(&input)?.clone();
    let scan_output_view_type = entry.graph.types.get(&scan_output)?.clone();
    Some(ScanCandidate {
        block,
        effect,
        owner,
        scratch_type,
        step_region: operator.step.region,
        combine_region: operator.combine.region,
        step_captures: operator.step.captures.clone(),
        neutral: operator.neutral,
        input_view: input,
        input_view_type,
        map_output_view_operands: (0..lanes.maps.len()).map(|index| output_base + index).collect(),
        scan_output_view_operand,
        scan_output_storage,
        scan_output_view_type,
    })
}

fn analyze_scan_entry(
    entry: &super::program::PlannedEntry,
    resources: &planning::ResourceIndex<'_>,
) -> error::Result<Option<ScanAnalysis>> {
    let Some(candidate) = analyze_scan_candidate(entry) else {
        return Ok(None);
    };
    let block_sums = resources.exactly_one_at(candidate.owner, CompilerResourceKind::ScanBlockSums, 0)?.id;
    let block_offsets =
        resources.exactly_one_at(candidate.owner, CompilerResourceKind::ScanBlockOffsets, 1)?.id;
    Ok(Some(ScanAnalysis {
        candidate,
        block_sums,
        block_offsets,
    }))
}

fn emit_scan_entry(
    entry: &mut super::program::PlannedEntry,
    analysis: ScanAnalysis,
    schedule: &mut schedule::KernelPlan,
    resources: &planning::ResourceIndex<'_>,
    policy: ParallelPolicy,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<Vec<super::program::PlannedEntry>> {
    debug_assert_eq!(
        segmented_recipe_effect(entry).map(|(block, effect, _)| (block, effect)),
        Some((analysis.candidate.block, analysis.candidate.effect))
    );
    let total_threads = policy.reduce_phase1_width;
    let ScanCandidate {
        block: block_id,
        effect: idx,
        scratch_type: elem_ty,
        step_region,
        combine_region,
        step_captures: step_capture_nodes,
        neutral: init_nid,
        input_view: input_view_nid,
        input_view_type: input_view_ty,
        map_output_view_operands: map_output_view_ops,
        scan_output_view_operand: scan_output_view_op,
        scan_output_storage,
        scan_output_view_type: orig_scan_output_view_ty,
        ..
    } = analysis.candidate;
    let (block_sums_resource, block_offsets_resource) = (analysis.block_sums, analysis.block_offsets);
    let op_func = schedule.callable_name(step_region).to_string();
    let reduce_func = schedule.callable_name(combine_region).to_string();

    // Chunk the input and the scan output view; swap them into the operand list.
    let chunked = chunk_soac_inputs(
        &mut entry.graph,
        &[(input_view_nid, input_view_ty.clone())],
        total_threads,
        ChunkInputKind::StorageOrRange,
        "SegScan",
    )?;
    let chunk_start = chunked.chunk_start;
    let chunk_len = chunked.chunk_len;
    let chunked_input_nid = chunked.views[0];
    {
        let operand = semantic_effect_mut(&mut entry.graph, block_id, idx)?
            .operand_nodes
            .first_mut()
            .ok_or_else(|| error::ParallelizeError::Invalid("scan recipe lost its input operand".into()))?;
        *operand = chunked_input_nid;
    }
    for (map_index, operand_index) in map_output_view_ops.iter().enumerate() {
        let original = *semantic_effect(&entry.graph, block_id, idx)?
            .operand_nodes
            .get(*operand_index)
            .ok_or_else(|| {
                error::ParallelizeError::Invalid(format!(
                    "scan recipe is missing map output operand {operand_index}"
                ))
            })?;
        let view_ty = semantic_node_type(&entry.graph, original)?;
        let chunked_view = chunk_view_like(
            &mut entry.graph,
            original,
            view_ty,
            chunk_start,
            chunk_len,
            ChunkInputKind::StorageOnly,
            &format!("SegScan map output {map_index}"),
        )?;
        let operand = semantic_effect_mut(&mut entry.graph, block_id, idx)?
            .operand_nodes
            .get_mut(*operand_index)
            .ok_or_else(|| {
                error::ParallelizeError::Invalid(format!(
                    "scan recipe lost map output operand {operand_index}"
                ))
            })?;
        *operand = chunked_view;
    }
    let chunked_scan_output = graph_ops::intern_chunked_resource_view(
        &mut entry.graph,
        scan_output_storage.0,
        chunk_start,
        chunk_len,
        orig_scan_output_view_ty,
        None,
    );
    {
        let operand = semantic_effect_mut(&mut entry.graph, block_id, idx)?
            .operand_nodes
            .get_mut(scan_output_view_op)
            .ok_or_else(|| {
                error::ParallelizeError::Invalid("scan recipe lost its output-view operand".into())
            })?;
        *operand = chunked_scan_output;
    }

    // Append a chunked reduce over the same input that stores each thread's
    // final accumulator to `block_sums[tid]`.
    {
        let next_semantic_op = entry
            .graph
            .skeleton
            .blocks
            .iter()
            .flat_map(|(_, block)| &block.side_effects)
            .filter_map(|effect| effect.kind.soac_id())
            .map(|id| id.0)
            .max()
            .map_or(0, |id| id + 1);
        // `[chunked_input, init]` — the step captures live on the SegBody below.
        let reduce_operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![chunked_input_nid, init_nid];
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![elem_ty.clone()]);
        let screma_nid = graph_ops::emit_pending_soac(
            &mut entry.graph,
            block_id,
            SemanticOpId(next_semantic_op),
            Soac::Screma(screma::Op::Reduce {
                lanes: screma::Lanes {
                    inputs: vec![super::types::SoacInputType { array: input_view_ty }],
                    maps: vec![],
                },
                operators: screma::NonEmpty {
                    first: screma::Operator {
                        step: SegBody {
                            region: schedule.intern_callable(&op_func),
                            captures: step_capture_nodes,
                        },
                        combine: SegBody {
                            region: schedule.intern_callable(&op_func),
                            captures: vec![],
                        },
                        input_indices: vec![screma::InputId(0)],
                        neutral: init_nid,
                        shape: Vec::new(),
                        commutative: false,
                        destination: SoacDestination::fresh(),
                        result_type: elem_ty.clone(),
                    },
                    rest: Vec::new(),
                },
                state: screma::SemanticState::Serial,
            }),
            reduce_operands,
            tuple_ty,
            effect_ids,
            None,
        );
        let result_nid = entry.graph.intern_pure(
            super::types::PureOp::Project { index: 0 },
            smallvec![screma_nid],
            elem_ty.clone(),
            None,
        );
        let arr_ty = Type::Constructed(
            TypeName::Array,
            vec![
                elem_ty.clone(),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
                Type::Variable(0),
                crate::types::no_buffer(),
            ],
        );
        let block_sums_view =
            graph_ops::intern_resource_view(&mut entry.graph, block_sums_resource, arr_ty, None);
        graph_ops::emit_storage_store(
            &mut entry.graph,
            block_id,
            block_sums_view,
            chunked.tid,
            result_nid,
            elem_ty.clone(),
            effect_ids,
            None,
        );
    }

    // Both intermediates are declared on phase 1 (block_sums is written here,
    // block_offsets is read by phase 3) so the verifiers and `realize_outputs`
    // see a consistent interface.
    for resource in [block_sums_resource, block_offsets_resource] {
        entry.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(resource),
            role: crate::interface::StorageRole::Intermediate,
            elem_ty: elem_ty.clone(),
            size: resources.get(resource)?.size.clone(),
        });
    }

    let mut phase2 = synthesize_phase2_scan(
        &entry.name,
        reduce_func.clone(),
        elem_ty.clone(),
        &entry.graph,
        init_nid,
        block_sums_resource,
        block_offsets_resource,
        None,
        effect_ids,
    )?;
    apply_manifest_resource_sizes(&mut phase2, resources)?;
    let swap_wrapper_name = format!("{}_scan_op_swap", entry.name);
    let swap_wrapper = synthesize_swap_wrapper(
        swap_wrapper_name.clone(),
        reduce_func,
        elem_ty.clone(),
        entry.span,
    );
    let swap_region = schedule.define_callable(swap_wrapper);
    let mut phase3 = synthesize_phase3_scan(
        &entry.name,
        swap_region,
        elem_ty,
        required_resource(scan_output_storage),
        block_offsets_resource,
        total_threads,
        effect_ids,
    )?;
    apply_manifest_resource_sizes(&mut phase3, resources)?;

    // Phase 1 is now a per-invocation Screma scan over the thread's chunk plus
    // the appended block-sum reduce; `soac_expand` lowers both.
    make_screma_serial(&mut entry.graph, block_id, idx)?;
    Ok(vec![phase2, phase3])
}

/// Synthesize phase 2 of a parallel scan: a single-invocation sequential
/// exclusive scan over `block_sums`. `block_offsets[i]` is the prefix of
/// blocks strictly before `i`, which phase 3 can safely prepend to chunk `i`.
fn synthesize_phase2_scan(
    entry_name: &str,
    op_func: String,
    elem_ty: Type<TypeName>,
    phase1_graph: &super::types::EGraph,
    phase1_ne_nid: NodeId,
    block_sums_resource: ResourceId,
    block_offsets_resource: ResourceId,
    len_out: Option<ResourceId>,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<super::program::PlannedEntry, String> {
    use super::builder::EntryBuilder;
    let mut b =
        EntryBuilder::new_compute(format!("{}_phase2_scan_sums", entry_name), (1, 1, 1), effect_ids);
    let scratch_len = dispatch_worker_logical_size(&elem_ty);
    b.declare_intermediate_storage_sized(block_sums_resource, elem_ty.clone(), scratch_len.clone());
    b.declare_intermediate_storage_sized(block_offsets_resource, elem_ty.clone(), scratch_len);
    if let Some(len_out) = len_out {
        b.declare_output_storage_sized(
            len_out,
            elem_ty.clone(),
            super::program::LogicalSize::FixedBytes(4),
        );
    }

    let init_nid = graph_ops::clone_pure_subgraph(phase1_graph, b.graph_mut(), phase1_ne_nid)?;
    let phase2 = build_exclusive_scan_phase2(
        &mut b,
        op_func,
        elem_ty.clone(),
        init_nid,
        block_sums_resource,
        block_offsets_resource,
        len_out.is_some(),
    );
    // A runtime filter publishes the scan's grand total (its survivor count)
    // into the length cell. The generic scan builder above stays oblivious to
    // this; only the bridge that knows the filter's `len_out` wires it up.
    if let (Some(len_out), Some(total)) = (len_out, phase2.total) {
        let (graph, _, effect_ids) = b.construction_parts_mut();
        let len_view = graph_ops::intern_resource_view(graph, len_out, elem_ty.clone(), None);
        graph_ops::emit_storage_store(
            graph,
            phase2.after,
            len_view,
            phase2.zero,
            total,
            elem_ty,
            effect_ids,
            None,
        );
    }
    Ok(b.build())
}

/// What an exclusive-scan phase-2 loop hands back to a caller that wants to
/// append work (e.g. a runtime filter storing the survivor count) to the
/// post-loop `after` block. The loop itself is generic — it knows nothing
/// about where a total is stored.
struct ExclusiveScanPhase2 {
    /// The grand total of all block sums, exposed as an `after` block param.
    /// `Some` only when `want_total` was requested.
    total: Option<NodeId>,
    /// The post-loop block (also left as the builder's current block).
    after: BlockId,
    /// The interned `0` node, reusable as a store index.
    zero: NodeId,
}

fn build_exclusive_scan_phase2(
    b: &mut super::builder::EntryBuilder,
    op_func: String,
    elem_ty: Type<TypeName>,
    init_nid: NodeId,
    block_sums_resource: ResourceId,
    block_offsets_resource: ResourceId,
    want_total: bool,
) -> ExclusiveScanPhase2 {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(0),
            crate::types::no_buffer(),
        ],
    );
    let entry_block = b.graph_mut().skeleton.entry;
    let (graph, control_headers, effect_ids) = b.construction_parts_mut();
    let sums = graph_ops::intern_resource_view(graph, block_sums_resource, arr_ty.clone(), None);
    let offsets = graph_ops::intern_resource_view(graph, block_offsets_resource, arr_ty, None);
    let len = emit_resource_len(graph, block_sums_resource);
    let zero = graph_ops::intern_u32(graph, 0, None);
    let one = graph_ops::intern_u32(graph, 1, None);

    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let cont = graph.skeleton.create_block();
    let after = graph.skeleton.create_block();
    let acc = graph.add_block_param(header, elem_ty.clone());
    let index = graph.add_block_param(header, u32_ty.clone());
    graph.skeleton.blocks[entry_block].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![init_nid, zero],
    };
    let condition = graph_ops::intern_binop(graph, "<", index, len, bool_ty, None);
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond: condition,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args: if want_total { vec![acc] } else { vec![] },
    };
    control_headers.insert(
        header,
        ControlHeader::Loop {
            merge: after,
            continue_block: cont,
        },
    );

    graph_ops::emit_storage_store(
        graph,
        body,
        offsets,
        index,
        acc,
        elem_ty.clone(),
        effect_ids,
        None,
    );
    let value = graph_ops::emit_view_load(graph, body, sums, index, elem_ty.clone(), effect_ids, None);
    let next_acc = graph.intern_pure(
        PureOp::Call(op_func),
        smallvec![acc, value],
        elem_ty.clone(),
        None,
    );
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: cont,
        args: vec![next_acc],
    };
    let continued_acc = graph.add_block_param(cont, graph.types[&acc].clone());
    let next_index = graph_ops::intern_binop(graph, "+", index, one, u32_ty, None);
    graph.skeleton.blocks[cont].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![continued_acc, next_index],
    };
    // Expose the grand total as an `after` block param when requested; the
    // caller (a filter compaction) appends its own store. The generic scan
    // never touches a length cell.
    let total = if want_total {
        let total = graph.add_block_param(after, elem_ty.clone());
        Some(total)
    } else {
        None
    };
    b.set_current_block(after);
    ExclusiveScanPhase2 { total, after, zero }
}

/// Synthesize phase 3 of a parallel scan: a chunked compute entry where each
/// thread reads `off = block_offsets[tid]` and applies `op(off, output[i])` to
/// every element of its chunk of `output`. Map's call convention is
/// `func(elem, ...captures)`, so phase 3 routes through `swap_wrapper_name`
/// (`\(elem, off) -> op(off, elem)`) to keep `off` in the accumulator slot for
/// non-commutative ops.
fn synthesize_phase3_scan(
    entry_name: &str,
    swap_region: RegionId,
    elem_ty: Type<TypeName>,
    output_resource: ResourceId,
    block_offsets_resource: ResourceId,
    total_threads: u32,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<super::program::PlannedEntry, String> {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(
        format!("{}_phase3_add_offsets", entry_name),
        (total_threads, 1, 1),
        effect_ids,
    );
    b.declare_output_storage(output_resource, elem_ty.clone());
    b.declare_intermediate_storage_sized(
        block_offsets_resource,
        elem_ty.clone(),
        dispatch_worker_logical_size(&elem_ty),
    );

    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(0),
            crate::types::no_buffer(),
        ],
    );
    let _output_view = b.emit_storage_view(output_resource, arr_ty.clone());
    let block_offsets_view = b.emit_storage_view(block_offsets_resource, arr_ty.clone());

    let output_len = emit_resource_len(b.graph_mut(), output_resource);
    let (tid, chunk_start, chunk_len) = emit_chunk_arithmetic(b.graph_mut(), total_threads, output_len)?;

    let off_place = b.graph_mut().intern_pure(
        super::types::PureOp::ViewIndex,
        smallvec![block_offsets_view, tid],
        elem_ty.clone(),
        None,
    );
    let off = b.emit_load(off_place, elem_ty.clone());

    let chunked_output = graph_ops::intern_chunked_resource_view(
        b.graph_mut(),
        output_resource,
        chunk_start,
        chunk_len,
        arr_ty.clone(),
        None,
    );

    b.emit_pending_map_into(
        swap_region,
        chunked_output,
        arr_ty.clone(),
        elem_ty,
        vec![off],
        chunked_output,
        arr_ty,
    );
    Ok(b.build())
}

/// Build a two-argument (`a`, `b`) helper function of type `T -> T -> T` named
/// `name`, whose body is produced by `body(graph, a_nid, b_nid)` and returned.
fn synthesize_binary_fn(
    name: String,
    elem_ty: Type<TypeName>,
    span: crate::ast::Span,
    body: impl FnOnce(&mut EGraph, NodeId, NodeId) -> NodeId,
) -> SemanticFunc {
    let mut graph = EGraph::new();
    let a_nid = graph.add_func_param(0, elem_ty.clone());
    let b_nid = graph.add_func_param(1, elem_ty.clone());
    let result = body(&mut graph, a_nid, b_nid);
    let entry_block = graph.skeleton.entry;
    graph.skeleton.blocks[entry_block].term = SkeletonTerminator::Return(Some(result));
    SemanticFunc::new(
        name,
        span,
        None,
        vec![
            (elem_ty.clone(), "a".to_string()),
            (elem_ty.clone(), "b".to_string()),
        ],
        elem_ty,
        graph,
        LookupMap::new(),
    )
}

/// A two-argument helper whose body is `inner(b, a)` — an arg-swapped wrapper
/// around a `T -> T -> T` combiner.
fn synthesize_swap_wrapper(
    wrapper_name: String,
    inner: String,
    elem_ty: Type<TypeName>,
    span: crate::ast::Span,
) -> SemanticFunc {
    let result_ty = elem_ty.clone();
    synthesize_binary_fn(wrapper_name, elem_ty, span, move |graph, a_nid, b_nid| {
        graph.intern_pure(PureOp::Call(inner), smallvec![b_nid, a_nid], result_ty, None)
    })
}

fn synthesize_u32_add_function(name: String, span: crate::ast::Span) -> SemanticFunc {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let result_ty = u32_ty.clone();
    synthesize_binary_fn(name, u32_ty, span, move |graph, a_nid, b_nid| {
        graph.intern_pure(
            PureOp::BinOp("+".into()),
            smallvec![a_nid, b_nid],
            result_ty,
            None,
        )
    })
}

#[cfg(test)]
#[path = "../parallelize_tests.rs"]
mod parallelize_tests;
