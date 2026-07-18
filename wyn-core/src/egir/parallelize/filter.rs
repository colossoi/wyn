//! Runtime-filter candidate analysis and five-phase kernel emission.

#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use super::*;
use crate::egir::soac::filter as filter_soac;

pub(super) fn lower_runtime_filters(
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
        let selections = analyze_filter_candidates(entry)?;
        let [(semantic_id, selection)] = selections.as_slice() else {
            continue;
        };
        let semantic_id = *semantic_id;
        if matches!(candidates.filter(semantic_id)?, RecipeSelection::Serial(_)) {
            continue;
        }
        let candidate = match selection.clone() {
            RecipeSelection::Parallel(candidate) => candidate,
            RecipeSelection::Serial(reason) => {
                return Err(error::ParallelizeError::Invalid(format!(
                    "preflight filter candidate {semantic_id:?} changed before emission: {reason:?}"
                )));
            }
        };
        let analysis = bind_filter_candidate(candidate, resources)?;
        let projected = crate::egir::program::PlannedEntry::project(entry)?;
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
                    group.semantic_slots.iter().copied().map(crate::egir::program::OutputSlotId).collect(),
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
            filter_slots.iter().copied().map(crate::egir::program::OutputSlotId).collect(),
        )?;
        lowered.insert(source);
    }
    Ok(lowered)
}

fn lower_filter_kernel(
    filter_entry: crate::egir::program::PlannedEntry,
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
    work: filter_soac::WorkBuffers,
    flags: crate::egir::program::PlannedEntry,
    scan: crate::egir::program::PlannedEntry,
    combine: crate::egir::program::PlannedEntry,
    apply_offsets: crate::egir::program::PlannedEntry,
    scatter: crate::egir::program::PlannedEntry,
    scan_workgroup_width: u32,
    scan_groups: u32,
}

fn build_filter_kernel_family(
    filter_entry: crate::egir::program::PlannedEntry,
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
        work.block_sums.0,
        work.block_offsets.0,
        Some(len_out.0),
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
        work.offsets.0,
        work.block_offsets.0,
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
        filter_soac::Plan::Scatter(filter_soac::ParallelConfig {
            buffers: work,
            scan_workgroup_width,
        }),
    )?;
    schedule.add_filter_phase_before(
        kernel,
        flags,
        schedule::DomainSelection::Explicit(domain.clone()),
        schedule::KernelKind::FilterFlags,
        filter_soac::Plan::Flags(filter_soac::ParallelConfig {
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
        filter_soac::Plan::Scan(filter_soac::ParallelConfig {
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

pub(super) fn lower_materialized_filters(
    inner: &AllocatedProgram,
    schedule: &mut schedule::KernelPlan,
    resources: &planning::ResourceIndex<'_>,
    candidates: &planning::CandidateIndex,
    policy: ParallelPolicy,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> error::Result<()> {
    for requirement in &inner.materializations {
        if requirement.kind != crate::egir::program::MaterializationKind::RuntimeArray {
            continue;
        }
        let endpoint = crate::egir::program::CompilerFlowEndpoint::Materialization(requirement.id);
        let kernel = schedule.kernel_for_flow_source(endpoint).ok_or_else(|| {
            format!(
                "runtime-array materialization {:?} was not scheduled",
                requirement.id
            )
        })?;
        let selections = analyze_filter_candidates(&requirement.entry)?;
        let [(semantic_id, selection)] = selections.as_slice() else {
            continue;
        };
        let semantic_id = *semantic_id;
        if matches!(candidates.filter(semantic_id)?, RecipeSelection::Serial(_)) {
            continue;
        }
        let candidate = match selection.clone() {
            RecipeSelection::Parallel(candidate) => candidate,
            RecipeSelection::Serial(reason) => {
                return Err(error::ParallelizeError::Invalid(format!(
                    "preflight materialized filter candidate {semantic_id:?} changed before emission: {reason:?}"
                )));
            }
        };
        let analysis = bind_filter_candidate(candidate, resources)?;
        let body = crate::egir::program::PlannedEntry::project(&requirement.entry)?;
        lower_filter_kernel(body, kernel, analysis, schedule, resources, policy, effect_ids)?;
    }
    Ok(())
}

#[derive(Clone)]
/// Complete graph-local runtime-filter recipe, consumed before entry mutation.
pub(super) struct FilterCandidate {
    pub semantic_id: SemanticOpId,
    pub space: SegSpace,
    pub len_out: SemanticResourceRef,
}

struct FilterAnalysis {
    candidate: FilterCandidate,
    work: filter_soac::WorkBuffers,
}

pub(super) fn analyze_filter_candidates(
    entry: &SemanticEntry,
) -> error::Result<Vec<(SemanticOpId, RecipeSelection<FilterCandidate>)>> {
    let mut analysis = Vec::new();
    for effect in entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects) {
        let SideEffectKind::Soac(SoacEffect(
            semantic_id,
            Soac::Filter(filter_soac::Op {
                state:
                    filter_soac::SemanticState {
                        space,
                        storage: filter_soac::Output::Runtime { length, .. },
                    },
                ..
            }),
        )) = &effect.kind
        else {
            continue;
        };
        let semantic_id = *semantic_id;
        let selection = match length {
            filter_soac::RuntimeLength::Stored(len_out) => RecipeSelection::Parallel(FilterCandidate {
                semantic_id,
                space: space.clone(),
                len_out: *len_out,
            }),
            filter_soac::RuntimeLength::ViewOnly => {
                RecipeSelection::Serial(FallbackReason::UnsupportedDestination)
            }
        };
        analysis.push((semantic_id, selection));
    }
    if analysis.len() > 1 {
        for (_, selection) in &mut analysis {
            *selection = RecipeSelection::Serial(FallbackReason::UnsupportedOperationShape);
        }
    }
    Ok(analysis)
}

fn bind_filter_candidate(
    candidate: FilterCandidate,
    resources: &planning::ResourceIndex<'_>,
) -> error::Result<FilterAnalysis> {
    let work = filter_work_buffers(candidate.semantic_id, resources)?;
    Ok(FilterAnalysis { candidate, work })
}

fn filter_work_buffers(
    owner: SemanticOpId,
    resources: &planning::ResourceIndex<'_>,
) -> error::Result<filter_soac::WorkBuffers> {
    let resource_id = |kind, slot| {
        resources.exactly_one_at(owner, kind, slot).map(|resource| SemanticResourceRef(resource.id))
    };
    Ok(filter_soac::WorkBuffers {
        flags: resource_id(CompilerResourceKind::FilterFlags, 0)?,
        offsets: resource_id(CompilerResourceKind::FilterOffsets, 1)?,
        block_sums: resource_id(CompilerResourceKind::FilterScanBlockSums, 2)?,
        block_offsets: resource_id(CompilerResourceKind::FilterScanBlockOffsets, 3)?,
    })
}
