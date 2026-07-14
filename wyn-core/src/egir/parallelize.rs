//! EGIR-side SOAC parallelization in two explicit stages:
//!
//! 1. `reify` turns every reachable Screma into a semantic `SegMap`,
//!    `SegRed`, or `SegScan` operations.
//! 2. `lower` schedules those operations: maps remain segmented for
//!    `soac_expand`, reductions become two-phase trees, and scans become
//!    three-phase block scans.
pub(crate) mod prepare;
pub mod schedule;

use std::collections::BTreeMap;

use crate::LookupMap;

use polytype::Type;
use smallvec::smallvec;

use super::graph_ops;
use super::program::{
    CompilerResource, CompilerResourceKind, LogicalResource, OutputWriter, ResourceId, ResourceOrigin,
    SemanticEntry, SemanticFunc, SemanticOpId, SemanticProgram, SemanticResourceDecl, SemanticResourceRef,
};
use super::soac::{filter, screma};
use super::types::{
    EGraph, ENode, NodeId, PureOp, RegionId, SegBody, SegSpace, SideEffect, SideEffectKind, SideEffectSite,
    SkeletonTerminator, Soac, SoacDestination,
};
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ControlHeader, InstKind};

/// Per-workgroup width of a synthesized phase-2 tree reduce.
pub const PHASE2_WIDTH: u32 = 256;
/// Per-workgroup width used to chunk a phase-1 partial reduce.
pub(crate) const REDUCE_PHASE1_WIDTH: u32 = 64;
/// Fixed workgroup count for the runtime-filter prefix scan's phase-1. The
/// scan runs `FILTER_SCAN_GROUPS * REDUCE_PHASE1_WIDTH` workers, each scanning
/// a `ceil(len / workers)` chunk, so the serial phase-2 over the per-worker
/// `block_sums` stays bounded by this constant instead of the input length.
pub(crate) const FILTER_SCAN_GROUPS: u32 = 4;

struct UnionFind {
    parents: Vec<usize>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parents: (0..len).collect(),
        }
    }

    fn find(&mut self, index: usize) -> usize {
        if self.parents[index] != index {
            self.parents[index] = self.find(self.parents[index]);
        }
        self.parents[index]
    }

    fn union(&mut self, left: usize, right: usize) {
        let (left, right) = (self.find(left), self.find(right));
        if left != right {
            self.parents[right] = left;
        }
    }
}

/// Lower semantic segmented operations into executable kernel entries.
/// Pointwise `SegMap`s remain for `soac_expand`; `SegRed`s become a chunked
/// phase 1 plus a synthesized tree reduction; `SegScan`s become chunk scans,
/// an exclusive scan of block sums, and offset-application phases.
pub fn lower(inner: &SemanticProgram) -> schedule::KernelPlan {
    use schedule::KernelPlan;

    let mut schedule = KernelPlan::seed(
        &inner.pipeline,
        &inner.entry_points,
        &inner.resources,
        &inner.region_interner,
    );
    attach_materializations(inner, &mut schedule);
    lower_runtime_filters(inner, &mut schedule);
    for entry in &inner.entry_points {
        let body =
            super::program::PlannedEntry::project(entry).expect("segmented semantic entry projection");
        plan_segmented_kernel_body(body, &mut schedule, &inner.resources);
    }
    schedule.coalesce_resource_flows(&inner.resources);
    schedule
}

pub fn lower_sequential(inner: &SemanticProgram) -> schedule::KernelPlan {
    let mut plan = schedule::KernelPlan::seed(
        &inner.pipeline,
        &inner.entry_points,
        &inner.resources,
        &inner.region_interner,
    );
    attach_materializations(inner, &mut plan);
    plan.select_sequential_recipes();
    plan.coalesce_resource_flows(&inner.resources);
    plan
}

fn plan_segmented_kernel_body(
    mut body: super::program::PlannedEntry,
    schedule: &mut schedule::KernelPlan,
    resources: &[LogicalResource],
) {
    use schedule::KernelDomain;
    let Some((block, index, _)) = kernel_effect(&body.graph) else {
        return;
    };
    let SideEffectKind::Soac(Soac::Screma(op)) =
        &body.graph.skeleton.blocks[block].side_effects[index].kind
    else {
        unreachable!()
    };
    match &op.body.kind {
        screma::Kind::Map => {
            let parent = body.name.clone();
            if let Some(split) = split_multidomain_seg_maps(&body) {
                let primary_slots = split.primary_slots;
                if !split.entries.is_empty() {
                    schedule.commit_kernel(split.primary, schedule::KernelKind::OutputDomainProjection);
                }
                schedule.set_output_projection(
                    &parent,
                    primary_slots.iter().copied().map(super::program::OutputSlotId).collect(),
                );
                for projected in split.entries {
                    let projected_name = projected.entry.name.clone();
                    schedule.add_sibling(
                        &parent,
                        projected.entry,
                        schedule::DomainSelection::Inferred(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                        schedule::KernelKind::OutputDomainProjection,
                    );
                    schedule.set_output_projection(
                        &projected_name,
                        projected
                            .semantic_slots
                            .iter()
                            .copied()
                            .map(super::program::OutputSlotId)
                            .collect(),
                    );
                }
            } else {
                schedule.commit_kernel(body, schedule::KernelKind::SerialCompute);
            }
        }
        screma::Kind::Reduce(_) => {
            if let Some(plan) = analyze_reduce_entry(&body, resources) {
                let phases = emit_reduce_entry(&mut body, plan, schedule, resources);
                let mut predecessor = body.name.clone();
                schedule.commit_kernel(body, schedule::KernelKind::ReducePhase1);
                for phase in phases {
                    let phase_name = phase.name.clone();
                    schedule.add_phase_after(
                        &predecessor,
                        phase,
                        schedule::DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
                        schedule::KernelKind::ReduceCombine,
                    );
                    predecessor = phase_name;
                }
            } else {
                commit_serial_kernel(body, schedule);
            }
        }
        screma::Kind::Scan(_) => {
            if let Some(plan) = analyze_scan_entry(&body, resources) {
                let phases = emit_scan_entry(&mut body, plan, schedule, resources);
                let mut predecessor = body.name.clone();
                let phase1_domain =
                    schedule.domain_of(&body.name).unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
                schedule.commit_kernel(body, schedule::KernelKind::ScanPhase1);
                for (phase_index, phase) in phases.into_iter().enumerate() {
                    let phase_name = phase.name.clone();
                    schedule.add_phase_after(
                        &predecessor,
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
                    predecessor = phase_name;
                }
            } else {
                commit_serial_kernel(body, schedule);
            }
        }
        screma::Kind::Composite(_) => {
            commit_serial_kernel(body, schedule);
        }
    }
}

fn commit_serial_kernel(mut body: super::program::PlannedEntry, schedule: &mut schedule::KernelPlan) {
    let (block, effect, _) =
        kernel_effect(&body.graph).expect("serial recipe requires one pending kernel SegOp");
    make_screma_serial(&mut body.graph, block, effect);
    schedule.commit_kernel(body, schedule::KernelKind::SerialCompute);
}

fn lower_runtime_filters(inner: &SemanticProgram, schedule: &mut schedule::KernelPlan) {
    use crate::interface::StorageRole;
    use schedule::KernelDomain;
    for entry in &inner.entry_points {
        let Some(FilterAnalysis { space, work, len_out }) = analyze_filter_entry(entry, &inner.resources)
        else {
            continue;
        };
        let domain =
            schedule::domain_from_space(&space).unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
        let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let declaration = |reference: SemanticResourceRef, role| {
            let resource = reference.0;
            let logical = &inner.resources[resource.0 as usize];
            SemanticResourceDecl {
                resource: SemanticResourceRef(resource),
                role,
                elem_ty: u32_ty.clone(),
                size: logical.size.clone(),
            }
        };

        let flags = project_kernel_body(
            entry,
            format!("{}_filter_flags", entry.name),
            entry.execution_model.clone(),
            Vec::new(),
            Vec::new(),
            vec![declaration(work.flags, StorageRole::Output)],
            Type::Constructed(TypeName::Unit, vec![]),
        );

        let scan_storage = [
            (work.flags, StorageRole::Input),
            (work.offsets, StorageRole::Output),
            (work.block_sums, StorageRole::Output),
        ]
        .into_iter()
        .map(|(resource, role)| declaration(resource, role))
        .collect();
        let mut scan = project_kernel_body(
            entry,
            format!("{}_filter_scan", entry.name),
            crate::ssa::types::ExecutionModel::Compute {
                local_size: (REDUCE_PHASE1_WIDTH, 1, 1),
            },
            Vec::new(),
            Vec::new(),
            scan_storage,
            Type::Constructed(TypeName::Unit, vec![]),
        );
        let zero = graph_ops::intern_u32(&mut scan.graph, 0, None);

        let add_name = format!("{}_filter_scan_add", entry.name);
        let add_fn = synthesize_u32_add_function(add_name.clone(), entry.span);
        schedule.define_callable(add_fn);

        let mut phase2 = synthesize_phase2_scan(
            &scan.name,
            add_name.clone(),
            u32_ty.clone(),
            &scan.graph,
            zero,
            required_resource(work.block_sums),
            required_resource(work.block_offsets),
            Some(required_resource(len_out)),
        )
        .expect("filter scan phase-2 synthesis");
        apply_manifest_resource_sizes(&mut phase2, &inner.resources);
        let swap_wrapper_name = format!("{}_filter_scan_add_offsets", entry.name);
        let swap_wrapper =
            synthesize_swap_wrapper(swap_wrapper_name.clone(), add_name, u32_ty.clone(), entry.span);
        let swap_region = schedule.define_callable(swap_wrapper);
        let mut phase3 = synthesize_phase3_scan(
            &scan.name,
            swap_region,
            u32_ty.clone(),
            required_resource(work.offsets),
            required_resource(work.block_offsets),
            REDUCE_PHASE1_WIDTH,
        );
        apply_manifest_resource_sizes(&mut phase3, &inner.resources);

        let mut scatter_resources = entry.resource_declarations.clone();
        scatter_resources.push(declaration(work.flags, StorageRole::Input));
        scatter_resources.push(declaration(work.offsets, StorageRole::Input));
        scatter_resources.push(declaration(work.block_offsets, StorageRole::Input));
        let scatter = project_kernel_body(
            entry,
            entry.name.clone(),
            entry.execution_model.clone(),
            entry.outputs.clone(),
            entry.output_routes.clone(),
            scatter_resources,
            entry.return_ty.clone(),
        );
        schedule.commit_filter_kernel(
            scatter,
            schedule::KernelKind::FilterScatter,
            filter::Plan::Scatter(work),
        );
        schedule.add_filter_phase_before(
            &entry.name,
            flags,
            schedule::DomainSelection::Explicit(domain.clone()),
            schedule::KernelKind::FilterFlags,
            filter::Plan::Flags(work),
        );
        // The scan runs a fixed worker grid so each worker scans a large
        // chunk and `block_sums` stays `FILTER_SCAN_GROUPS * width`-sized,
        // keeping the serial phase-2 bounded by that constant rather than by
        // `len`. The flags and scatter phases stay per-element (`domain`).
        schedule.add_filter_phase_before(
            &entry.name,
            scan,
            schedule::DomainSelection::Explicit(KernelDomain::Fixed {
                x: FILTER_SCAN_GROUPS,
                y: 1,
                z: 1,
            }),
            schedule::KernelKind::FilterScan,
            filter::Plan::Scan(work),
        );
        schedule.add_phase_before(
            &entry.name,
            phase2,
            schedule::DomainSelection::Explicit(KernelDomain::Fixed { x: 1, y: 1, z: 1 }),
            schedule::KernelKind::FilterCombine,
        );
        schedule.add_phase_before(
            &entry.name,
            phase3,
            schedule::DomainSelection::Explicit(domain),
            schedule::KernelKind::FilterScan,
        );
    }
}

struct FilterAnalysis {
    space: SegSpace,
    work: filter::WorkBuffers,
    len_out: SemanticResourceRef,
}

fn analyze_filter_entry(entry: &SemanticEntry, resources: &[LogicalResource]) -> Option<FilterAnalysis> {
    let mut analysis = None;
    for effect in entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects) {
        let SideEffectKind::Soac(Soac::Filter(filter::Op {
            state:
                filter::SemanticState {
                    space,
                    storage: filter::Output::Runtime { length, .. },
                },
            ..
        })) = &effect.kind
        else {
            continue;
        };
        let filter::RuntimeLength::EntryOutput(len_out) = length else {
            return None;
        };
        let work = filter_work_buffers(effect.semantic_id?, resources)?;
        if analysis
            .replace(FilterAnalysis {
                space: space.clone(),
                work,
                len_out: *len_out,
            })
            .is_some()
        {
            return None;
        }
    }
    analysis
}

#[allow(clippy::too_many_arguments)]
fn project_kernel_body(
    source: &SemanticEntry,
    name: String,
    execution_model: crate::ssa::types::ExecutionModel,
    outputs: Vec<crate::ssa::types::EntryOutput>,
    output_routes: Vec<super::program::OutputRoute>,
    resource_declarations: Vec<SemanticResourceDecl>,
    return_ty: Type<TypeName>,
) -> super::program::PlannedEntry {
    let route_values = output_routes.iter().map(|route| route.source.value).collect();
    let projection = super::graph_projector::GraphProjector::new(&source.graph, &source.control_headers)
        .all_with_values(route_values)
        .expect("complete entry projection must be internally valid");
    super::program::PlannedEntry::from_projection(
        projection,
        name,
        source.span,
        execution_model,
        source.inputs.clone(),
        outputs,
        resource_declarations,
        source.params.clone(),
        return_ty,
        &source.aliases,
        output_routes,
    )
    .expect("complete entry projection must be internally valid")
}

#[allow(clippy::too_many_arguments)]
fn project_kernel_body_effects(
    source: &super::program::PlannedEntry,
    selected: std::collections::HashSet<SideEffectSite>,
    name: String,
    execution_model: crate::ssa::types::ExecutionModel,
    outputs: Vec<crate::ssa::types::EntryOutput>,
    output_routes: Vec<super::program::OutputRoute>,
    resource_declarations: Vec<SemanticResourceDecl>,
    return_ty: Type<TypeName>,
) -> super::program::PlannedEntry {
    let route_values = output_routes.iter().map(|route| route.source.value).collect();
    let projection = super::graph_projector::GraphProjector::new(&source.graph, &source.control_headers)
        .selected_with_values(selected, route_values)
        .expect("selected entry projection must be internally valid");
    super::program::PlannedEntry::from_projection(
        projection,
        name,
        source.span,
        execution_model,
        source.inputs.clone(),
        outputs,
        resource_declarations,
        source.params.clone(),
        return_ty,
        &source.aliases,
        output_routes,
    )
    .expect("selected entry projection must be internally valid")
}

fn filter_work_buffers(owner: SemanticOpId, resources: &[LogicalResource]) -> Option<filter::WorkBuffers> {
    let owner_matches = |compiler: &CompilerResource| compiler.owner == Some(owner);
    let resource_id = |kind| {
        resources.iter().find_map(|resource| {
            let ResourceOrigin::Compiler(compiler) = &resource.origin else {
                return None;
            };
            (compiler.kind == kind && owner_matches(compiler)).then_some(SemanticResourceRef(resource.id))
        })
    };
    Some(filter::WorkBuffers {
        flags: resource_id(CompilerResourceKind::FilterFlags)?,
        offsets: resource_id(CompilerResourceKind::FilterOffsets)?,
        block_sums: resource_id(CompilerResourceKind::FilterScanBlockSums)?,
        block_offsets: resource_id(CompilerResourceKind::FilterScanBlockOffsets)?,
    })
}

fn required_resource(reference: SemanticResourceRef) -> ResourceId {
    reference.0
}

fn apply_manifest_resource_sizes(entry: &mut super::program::PlannedEntry, resources: &[LogicalResource]) {
    for declaration in &mut entry.resource_declarations {
        let resource = declaration.resource.0;
        declaration.size = resources[resource.0 as usize].size.clone();
    }
}

fn owned_resource_ids(
    resources: &[LogicalResource],
    owner: SemanticOpId,
    kind: CompilerResourceKind,
) -> Vec<ResourceId> {
    let mut ids: Vec<_> = resources
        .iter()
        .filter_map(|resource| {
            let ResourceOrigin::Compiler(compiler) = &resource.origin else {
                return None;
            };
            (compiler.kind == kind && compiler.owner == Some(owner)).then_some((compiler.slot, resource.id))
        })
        .collect();
    ids.sort_by_key(|(slot, _)| *slot);
    ids.into_iter().map(|(_, resource)| resource).collect()
}

/// Attach allocation-created materialization entries to their consumer's
/// source pipeline before target recipes are selected.
pub(crate) fn attach_materializations(inner: &SemanticProgram, schedule: &mut schedule::KernelPlan) {
    let mut materializations = BTreeMap::new();
    for resource in &inner.resources {
        let ResourceOrigin::Compiler(compiler) = &resource.origin else {
            continue;
        };
        let Some(flow) = &compiler.flow else {
            continue;
        };
        match compiler.kind {
            CompilerResourceKind::ScalarHandoff
            | CompilerResourceKind::GatherHandoff
            | CompilerResourceKind::MultiConsumerArray => {}
            _ => continue,
        }
        let consumers = materializations.entry(flow.producer).or_insert_with(Vec::new);
        consumers.extend(flow.consumers.iter().copied());
        consumers.sort_unstable();
        consumers.dedup();
    }

    // Insert one ready producer at a time. Repeating is important for chained
    // materializations: scheduling a consumer can make its producer ready.
    while let Some((producer_id, consumer_id)) =
        materializations.iter().find_map(|(producer, consumers)| {
            if schedule.contains_flow_source(*producer) {
                return None;
            }
            consumers
                .iter()
                .find(|consumer| schedule.contains_flow_source(**consumer))
                .map(|consumer| (*producer, *consumer))
        })
    {
        let consumer = schedule
            .entry_point_for_flow_source(consumer_id)
            .expect("scheduled flow consumer has no entry point")
            .to_string();
        let super::program::CompilerFlowEndpoint::Materialization(id) = producer_id else {
            panic!("typed entry/prepass producer was omitted while seeding the kernel plan")
        };
        let requirement = inner
            .materializations
            .get(id.0 as usize)
            .filter(|requirement| requirement.id == id)
            .expect("materialization flow references a missing requirement");
        schedule.add_materialization_before(&consumer, requirement);
        let body = super::program::PlannedEntry::project(&requirement.entry)
            .expect("materialization entry projection");
        if kernel_effect(&body.graph).is_some_and(|(block, index, _)| {
            matches!(
                &body.graph.skeleton.blocks[block].side_effects[index].kind,
                SideEffectKind::Soac(Soac::Screma(screma::Op {
                    body: screma::Body {
                        kind: screma::Kind::Reduce(_) | screma::Kind::Scan(_),
                        ..
                    },
                    ..
                }))
            )
        }) {
            plan_segmented_kernel_body(body, schedule, &inner.resources);
        }
    }
}

/// The output slots a side effect fulfils, taken from the explicit routes
/// established by output realization. A pointwise map writes one slot; a fused
/// equal-domain multi-lane map writes several (one per lane). Empty means the
/// effect has no declared output ownership.
fn side_effect_output_slots_body(entry: &super::program::PlannedEntry, se: &SideEffect) -> Vec<usize> {
    side_effect_output_slots_from_routes(&entry.output_routes, se)
}

fn side_effect_output_slots_from_routes(
    routes: &[super::program::OutputRoute],
    se: &SideEffect,
) -> Vec<usize> {
    if let SideEffectKind::Soac(Soac::Screma(screma::Op {
        state: screma::SemanticState::Segmented { output_slots, .. },
        ..
    })) = &se.kind
    {
        return output_slots.iter().map(|slot| slot.0).collect();
    }
    let value_writer = se.result.map(OutputWriter::Value);
    let effect_writer = se.effects.map(|(_, output)| OutputWriter::Effect(output));
    let mut slots: Vec<usize> = routes
        .iter()
        .filter(|route| {
            route
                .writers
                .iter()
                .any(|writer| Some(*writer) == value_writer || Some(*writer) == effect_writer)
        })
        .map(|route| route.slot.0)
        .collect();
    slots.sort_unstable();
    slots.dedup();
    slots
}

/// True for a pointwise `SegMap` side-effect (a parallel output domain).
fn is_seg_map(se: &SideEffect) -> bool {
    matches!(
        &se.kind,
        SideEffectKind::Soac(Soac::Screma(screma::Op {
            body: screma::Body {
                kind: screma::Kind::Map,
                ..
            },
            state: screma::SemanticState::Segmented { .. },
        }))
    )
}

/// True when the side-effect writes externally observable state: a `scatter`, a
/// `filter` (compacted output + length cell), a SOAC writing a non-`Fresh`
/// buffer (an `OutputView` output or an in-place `InputBuffer`), or a `Store`.
/// Such an effect must run in exactly one kernel when an entry splits across
/// output domains, and it is never dead-code-pruned. Pure producers (`Fresh`
/// SOACs, `Load`, `ViewIndex`) are safe to duplicate because each split kernel
/// recomputes what it consumes.
fn is_write_effectful(se: &SideEffect) -> bool {
    match &se.kind {
        SideEffectKind::Soac(Soac::Hist(_) | Soac::Filter(_)) => true,
        SideEffectKind::Soac(Soac::Screma(op)) => op
            .body
            .maps
            .iter()
            .map(|map| map.destination)
            .chain(op.body.kind.operators().into_iter().map(|operator| operator.destination))
            .any(|d| matches!(d, SoacDestination::OutputView | SoacDestination::InputBuffer)),
        SideEffectKind::Inst(InstKind::Store { .. }) => true,
        _ => false,
    }
}

/// Output-domain splitting is construction-only: every group gets a fresh
/// graph projection containing its selected effects and producer closure.
/// that remains — no other side-effect's operands, no terminator. Iterated to a
/// fixpoint so a chain of producers (a map feeding a map feeding a dropped
/// scatter) collapses fully; write-effects and producers still feeding a kept
/// effect or an output are retained.
/// Split a multi-output entry into one kernel per iteration domain — Futhark's
/// one-SegOp-per-SegSpace structure. Side-effects are grouped by the set of
/// output slots they write (a fused equal-domain multi-lane map is one group
/// writing several slots; sibling maps over distinct domains are separate
/// groups; fixed non-SOAC outputs each form their own group). Each group
/// becomes its own entry and pipeline stage: the group owning the lowest slot
/// keeps the entry's name, the rest become `{name}_dispatch_{slot}`. Only fires
/// when at least two groups contain a parallel `SegMap`, so a single map
/// alongside fixed slots stays one sharding kernel. Returns the new clones.
struct SplitEntry {
    entry: super::program::PlannedEntry,
    semantic_slots: Vec<usize>,
}

struct EntrySplit {
    primary: super::program::PlannedEntry,
    primary_slots: Vec<usize>,
    entries: Vec<SplitEntry>,
}

fn split_multidomain_seg_maps(entry: &super::program::PlannedEntry) -> Option<EntrySplit> {
    let n_out = entry.outputs.len();
    if n_out <= 1 {
        return None;
    }

    // Union-find over output slots: union the slots co-written by any single
    // side-effect, so a fused multi-lane map keeps its slots in one group.
    let mut groups = UnionFind::new(n_out);
    let mut producers: Vec<((BlockId, usize), Vec<usize>, bool)> = Vec::new();
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            let slots = side_effect_output_slots_body(entry, se);
            if slots.is_empty() {
                continue;
            }
            for &s in &slots[1..] {
                groups.union(slots[0], s);
            }
            producers.push(((bid, i), slots, is_seg_map(se)));
        }
    }

    // Assign each producing side-effect to its group root; collect each group's
    // slots and whether it contains a parallel map.
    let mut group_of: LookupMap<(BlockId, usize), usize> = LookupMap::new();
    let mut group_slots: LookupMap<usize, Vec<usize>> = LookupMap::new();
    let mut parallel_groups: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (loc, slots, is_map) in &producers {
        let root = groups.find(slots[0]);
        group_of.insert(*loc, root);
        let entry_slots = group_slots.entry(root).or_default();
        for &s in slots {
            if !entry_slots.contains(&s) {
                entry_slots.push(s);
            }
        }
        if *is_map {
            parallel_groups.insert(root);
        }
    }
    if parallel_groups.len() < 2 {
        return None;
    }

    // Order groups by their lowest slot; the first keeps the entry's name.
    let mut groups: Vec<(usize, Vec<usize>)> = group_slots.into_iter().collect();
    for (_, slots) in groups.iter_mut() {
        slots.sort_unstable();
    }
    groups.sort_by_key(|(_, slots)| slots[0]);

    // A shared write-effectful side-effect (a scatter / filter / in-place store
    // that writes external state but is tied to no output slot) must run in
    // exactly one kernel. Assign each to the lowest-slot group that dispatches a
    // map, so it keeps a real iteration domain — mirroring the unsplit entry —
    // and `restrict_to_group` then drops it from every other clone.
    let host_root = groups
        .iter()
        .map(|(root, _)| *root)
        .find(|root| parallel_groups.contains(root))
        .expect("split fires only with at least two parallel groups");
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            if side_effect_output_slots_body(entry, se).is_empty() && is_write_effectful(se) {
                group_of.insert((bid, i), host_root);
            }
        }
    }

    let base_name = entry.name.clone();
    let project_group = |root: usize, slots: &[usize], name: String| {
        let selected = group_of
            .iter()
            .filter_map(|(&(block, index), &group)| {
                (group == root).then_some(SideEffectSite { block, index })
            })
            .collect();
        let outputs = slots.iter().map(|&slot| entry.outputs[slot].clone()).collect();
        let mut routes = entry
            .output_routes
            .iter()
            .filter(|route| slots.contains(&route.slot.0))
            .cloned()
            .collect::<Vec<_>>();
        for route in &mut routes {
            route.slot = super::program::OutputSlotId(
                slots.iter().position(|&slot| slot == route.slot.0).expect("selected route slot"),
            );
        }
        project_kernel_body_effects(
            entry,
            selected,
            name,
            entry.execution_model.clone(),
            outputs,
            routes,
            entry.resource_declarations.clone(),
            entry.return_ty.clone(),
        )
    };

    let mut clones = Vec::new();
    for (root, slots) in &groups[1..] {
        let projected = project_group(*root, slots, format!("{base_name}_dispatch_{}", slots[0]));
        clones.push(SplitEntry {
            entry: projected,
            semantic_slots: slots.clone(),
        });
    }

    // The original retains the group owning the lowest slot under its own name.
    let (root0, slots0) = &groups[0];
    let primary = project_group(*root0, slots0, entry.name.clone());
    Some(EntrySplit {
        primary,
        primary_slots: slots0.clone(),
        entries: clones,
    })
}

#[derive(Clone, Copy)]
enum ChunkInputKind {
    StorageOnly,
    StorageOrRange,
}

fn can_chunk_view(graph: &EGraph, view: NodeId, kind: ChunkInputKind) -> bool {
    graph_ops::extract_storage_view_source(graph, view).is_some()
        || matches!(kind, ChunkInputKind::StorageOrRange)
            && graph_ops::extract_array_range_operands(graph, view).is_some_and(|(_, len, _)| {
                matches!(
                    graph.types.get(&len),
                    Some(Type::Constructed(TypeName::UInt(32) | TypeName::Int(32), _))
                )
            })
}

fn can_clone_pure_subgraph(graph: &EGraph, root: NodeId, substitutions: &[NodeId]) -> bool {
    fn visit(
        graph: &EGraph,
        node: NodeId,
        substitutions: &[NodeId],
        seen: &mut std::collections::HashSet<NodeId>,
    ) -> bool {
        if substitutions.contains(&node) || !seen.insert(node) {
            return true;
        }
        match &graph.nodes[node] {
            ENode::Constant(_) => true,
            ENode::Pure { operands, .. } => {
                operands.iter().all(|operand| visit(graph, *operand, substitutions, seen))
            }
            _ => false,
        }
    }

    visit(graph, root, substitutions, &mut std::collections::HashSet::new())
}

struct ChunkedSoacInputs {
    tid: NodeId,
    chunk_start: NodeId,
    chunk_len: NodeId,
    views: Vec<NodeId>,
}

fn chunk_soac_inputs(
    graph: &mut EGraph,
    inputs: &[(NodeId, Type<TypeName>)],
    total_threads: u32,
    kind: ChunkInputKind,
    context: &str,
) -> Result<ChunkedSoacInputs, String> {
    let (first_view, _) = inputs.first().ok_or_else(|| format!("phase1 {context}: no SOAC inputs"))?;
    let input_len = input_length_for_chunking(graph, *first_view, kind, context)?;
    let (tid, chunk_start, chunk_len) = emit_chunk_arithmetic(graph, total_threads, input_len)?;

    let mut views = Vec::with_capacity(inputs.len());
    for &(view_nid, ref view_ty) in inputs {
        views.push(chunk_view_like(
            graph,
            view_nid,
            view_ty.clone(),
            chunk_start,
            chunk_len,
            kind,
            context,
        )?);
    }

    Ok(ChunkedSoacInputs {
        tid,
        chunk_start,
        chunk_len,
        views,
    })
}

fn input_length_for_chunking(
    graph: &mut EGraph,
    view_nid: NodeId,
    kind: ChunkInputKind,
    context: &str,
) -> Result<NodeId, String> {
    if let Some(br) = graph_ops::extract_storage_view_source(graph, view_nid) {
        return Ok(emit_semantic_resource_len(graph, br));
    }
    if matches!(kind, ChunkInputKind::StorageOrRange) {
        if let Some((_, len_nid, _)) = graph_ops::extract_array_range_operands(graph, view_nid) {
            return Ok(len_nid);
        }
    }
    Err(format!("phase1 {context}: input is not a chunkable view"))
}

fn chunk_view_like(
    graph: &mut EGraph,
    view_nid: NodeId,
    view_ty: Type<TypeName>,
    chunk_start: NodeId,
    chunk_len: NodeId,
    kind: ChunkInputKind,
    context: &str,
) -> Result<NodeId, String> {
    if let Some(br) = graph_ops::extract_storage_view_source(graph, view_nid) {
        return Ok(graph_ops::intern_chunked_resource_view(
            graph,
            br.0,
            chunk_start,
            chunk_len,
            view_ty,
            None,
        ));
    }
    if matches!(kind, ChunkInputKind::StorageOrRange) {
        if let Some((orig_start, _, step)) = graph_ops::extract_array_range_operands(graph, view_nid) {
            let has_step = step.is_some();
            let start_ty = graph.types[&orig_start].clone();
            let start_delta = if let Some(step) = step {
                graph_ops::intern_binop(graph, "*", chunk_start, step, start_ty.clone(), None)
            } else {
                chunk_start
            };
            let new_start = graph_ops::intern_binop(graph, "+", orig_start, start_delta, start_ty, None);
            let mut ops: smallvec::SmallVec<[NodeId; 4]> = smallvec![new_start, chunk_len];
            if let Some(s) = step {
                ops.push(s);
            }
            return Ok(graph.intern_pure(PureOp::ArrayRange { has_step }, ops, view_ty));
        }
    }
    Err(format!("phase1 {context}: input is not a chunkable view"))
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
) -> Result<(), String> {
    let w = PHASE2_WIDTH;
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
    let graph = b.graph_mut();
    let mut eff = graph_ops::next_effect_token(graph);

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
    let u32_min = catalog().lookup_by_any_name("u32.min").expect("catalog has u32.min");
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
    let acc_in = graph.add_block_param(grid_header, 0, elem_ty.clone());
    graph.skeleton.blocks[grid_header].params.push(acc_in);
    let i_in = graph.add_block_param(grid_header, 1, u32_ty.clone());
    graph.skeleton.blocks[grid_header].params.push(i_in);

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
    b.control_headers_mut().insert(
        grid_header,
        ControlHeader::Loop {
            merge: grid_after,
            continue_block: grid_cont,
        },
    );

    // grid_body: acc' = op(acc, partials[i]); → grid_cont(acc')
    let graph = b.graph_mut();
    let elem_i = graph_ops::emit_view_load(
        graph,
        grid_body,
        partials_view,
        i_in,
        elem_ty.clone(),
        &mut eff,
        None,
    );
    let acc_next = graph.intern_pure(
        PureOp::Call(op_func.clone()),
        smallvec![acc_in, elem_i],
        elem_ty.clone(),
    );
    graph.skeleton.blocks[grid_body].term = SkeletonTerminator::Branch {
        target: grid_cont,
        args: vec![acc_next],
    };

    // grid_cont(acc_c): i_next = i + W; → grid_header(acc_c, i_next)
    let acc_c = graph.add_block_param(grid_cont, 0, elem_ty.clone());
    graph.skeleton.blocks[grid_cont].params.push(acc_c);
    let one_u32 = graph_ops::intern_u32(graph, 1, None);
    let i_next = graph_ops::intern_binop(graph, "+", i_in, one_u32, u32_ty.clone(), None);
    graph.skeleton.blocks[grid_cont].term = SkeletonTerminator::Branch {
        target: grid_header,
        args: vec![acc_c, i_next],
    };

    // grid_after(acc_final): shared[lid] = acc_final; barrier; → tree_header(1)
    let acc_final = graph.add_block_param(grid_after, 0, elem_ty.clone());
    graph.skeleton.blocks[grid_after].params.push(acc_final);
    graph_ops::emit_storage_store(
        graph,
        grid_after,
        shared_view,
        lid,
        acc_final,
        elem_ty.clone(),
        &mut eff,
        None,
    );
    graph_ops::emit_workgroup_barrier(graph, grid_after, &mut eff);
    graph.skeleton.blocks[grid_after].term = SkeletonTerminator::Branch {
        target: tree_header,
        args: vec![one_u32],
    };

    // Grow an adjacent-pair tree from stride 1. This preserves source order
    // for associative, non-commutative operators.
    let stride_in = graph.add_block_param(tree_header, 0, u32_ty.clone());
    graph.skeleton.blocks[tree_header].params.push(stride_in);
    let stride_cond = graph_ops::intern_binop(graph, "<", stride_in, w_nid, bool_ty.clone(), None);
    graph.skeleton.blocks[tree_header].term = SkeletonTerminator::CondBranch {
        cond: stride_cond,
        then_target: tree_body,
        then_args: vec![],
        else_target: tree_after,
        else_args: vec![],
    };
    b.control_headers_mut().insert(
        tree_header,
        ControlHeader::Loop {
            merge: tree_after,
            continue_block: tree_cont,
        },
    );

    // Only the first lane in each adjacent pair combines the two runs.
    let graph = b.graph_mut();
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
    b.control_headers_mut().insert(
        tree_body,
        ControlHeader::Selection {
            merge: tree_sel_merge,
        },
    );

    // tree_then: shared[lid] = op(shared[lid], shared[lid+stride]); → tree_sel_merge
    let graph = b.graph_mut();
    let a = graph_ops::emit_view_load(
        graph,
        tree_then,
        shared_view,
        lid,
        elem_ty.clone(),
        &mut eff,
        None,
    );
    let lid_plus = graph_ops::intern_binop(graph, "+", lid, stride_in, u32_ty.clone(), None);
    let bb = graph_ops::emit_view_load(
        graph,
        tree_then,
        shared_view,
        lid_plus,
        elem_ty.clone(),
        &mut eff,
        None,
    );
    let combined = graph.intern_pure(PureOp::Call(op_func.clone()), smallvec![a, bb], elem_ty.clone());
    graph_ops::emit_storage_store(
        graph,
        tree_then,
        shared_view,
        lid,
        combined,
        elem_ty.clone(),
        &mut eff,
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
    graph_ops::emit_workgroup_barrier(graph, tree_cont, &mut eff);
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
    b.control_headers_mut().insert(tree_after, ControlHeader::Selection { merge: end_blk });

    // write_blk: combined = shared[0]; replay each captured output store reading
    // `combined` in place of the per-thread accumulator value. A scalar reduce
    // has one store (`out[0] = combined`); a tuple-element reduce decomposes
    // across one store per field.
    let graph = b.graph_mut();
    let s0 = graph_ops::emit_view_load(
        graph,
        write_blk,
        shared_view,
        zero_u32,
        elem_ty.clone(),
        &mut eff,
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
        graph_ops::emit_store(graph, write_blk, cloned_place, cloned_value, &mut eff, None);
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
    let index_ty = graph.types[&input_len].clone();
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
    graph.intern_pure_with_span(op, smallvec![], index_ty.clone(), None)
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
pub fn synthesize_phase2_reduce_cloning_ne_named(
    full_name: String,
    op_func: String,
    elem_ty: Type<TypeName>,
    phase1_graph: &super::types::EGraph,
    phase1_ne_nid: NodeId,
    partials_resource: ResourceId,
    accumulator_value: NodeId,
    output_stores: &[(NodeId, NodeId)],
    output_decls: &[(ResourceId, Type<TypeName>, super::program::LogicalSize)],
) -> Result<super::program::PlannedEntry, String> {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(full_name, (PHASE2_WIDTH, 1, 1));
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
    )?;
    Ok(b.build())
}

pub(crate) fn kernel_effect(graph: &EGraph) -> Option<(BlockId, usize, &SideEffect)> {
    graph.skeleton.blocks.iter().find_map(|(block, contents)| {
        contents.side_effects.iter().enumerate().find_map(|(index, effect)| {
            matches!(
                &effect.kind,
                SideEffectKind::Soac(Soac::Screma(screma::Op {
                    state: screma::SemanticState::Segmented {
                        placement: screma::Placement::Kernel,
                        ..
                    },
                    ..
                }))
            )
            .then_some((block, index, effect))
        })
    })
}

fn make_screma_serial(graph: &mut EGraph, block_id: BlockId, index: usize) {
    let SideEffectKind::Soac(Soac::Screma(op)) =
        &mut graph.skeleton.blocks[block_id].side_effects[index].kind
    else {
        unreachable!()
    };
    op.state = screma::SemanticState::Serial;
}

fn project_root_index(graph: &super::types::EGraph, value: NodeId, root: NodeId) -> Option<u32> {
    let mut cur = value;
    let mut last_index = None;
    loop {
        if cur == root {
            return last_index;
        }
        match &graph.nodes[cur] {
            super::types::ENode::Pure {
                op: super::types::PureOp::Project { index },
                operands,
            } => {
                last_index = Some(*index);
                cur = operands[0];
            }
            _ => return None,
        }
    }
}

fn storage_resource_under(graph: &super::types::EGraph, root: NodeId) -> Option<SemanticResourceRef> {
    wyn_graph::find_map_reachable(
        [root],
        wyn_graph::WalkOrder::DepthFirst,
        |node, out| out.extend(graph.nodes[node].children()),
        |node| graph_ops::extract_storage_view_source(graph, node),
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SegScratchFamily {
    Reduce,
    Scan,
}

struct SegScratchSpec {
    family: SegScratchFamily,
    resources: Vec<(Type<TypeName>, CompilerResourceKind)>,
}

/// Parse the shared eligibility gates and scratch owned by a parallel Seg op.
/// Allocation and lowering consume this same result.
fn seg_scratch_specs(graph: &EGraph, se: &SideEffect) -> Option<SegScratchSpec> {
    let SideEffectKind::Soac(Soac::Screma(screma::Op {
        body,
        state:
            screma::SemanticState::Segmented {
                placement: screma::Placement::Kernel,
                ..
            },
    })) = &se.kind
    else {
        return None;
    };
    let elem_of = |neutral: NodeId| graph.types.get(&neutral).cloned();
    let maps_are_output_views = body.maps.iter().all(|map| map.destination == SoacDestination::OutputView);
    let operators = body.kind.operators();
    match &body.kind {
        screma::Kind::Reduce(_) => {
            if operators.iter().any(|op| !op.combine.captures.is_empty())
                || body.inputs.is_empty()
                || !maps_are_output_views
                || !operators.iter().all(|op| op.destination == SoacDestination::Fresh)
            {
                return None;
            }
            let resources = operators
                .iter()
                .map(|op| Some((elem_of(op.neutral)?, CompilerResourceKind::ReducePartial)))
                .collect::<Option<_>>()?;
            Some(SegScratchSpec {
                family: SegScratchFamily::Reduce,
                resources,
            })
        }
        screma::Kind::Scan(_) => {
            if operators.len() != 1
                || !operators[0].combine.captures.is_empty()
                || body.inputs.len() != 1
                || !maps_are_output_views
                || !operators.iter().all(|op| op.destination == SoacDestination::OutputView)
            {
                return None;
            }
            let elem = elem_of(operators[0].neutral)?;
            Some(SegScratchSpec {
                family: SegScratchFamily::Scan,
                resources: vec![
                    (elem.clone(), CompilerResourceKind::ScanBlockSums),
                    (elem, CompilerResourceKind::ScanBlockOffsets),
                ],
            })
        }
        screma::Kind::Map | screma::Kind::Composite(_) => None,
    }
}

/// Reserve one binding-free logical resource for every parallel
/// SegRed/SegScan scratch lane. Physical bindings are chosen only after plan
/// validation.
pub fn enumerate_seg_scratch(inner: &SemanticProgram, first_id: u32) -> Vec<LogicalResource> {
    let mut resources = Vec::new();
    let mut next = first_id;
    for (_, entry) in inner.entries_with_endpoints() {
        // Collect scratch needs before drawing (immutable graph read), then
        // apply the assignment (mutable) — avoids aliasing graph.types.
        let mut plans: Vec<(BlockId, usize, Vec<(Type<TypeName>, CompilerResourceKind)>)> = Vec::new();
        for (block_id, block) in &entry.graph.skeleton.blocks {
            for (se_idx, se) in block.side_effects.iter().enumerate() {
                if let Some(spec) = seg_scratch_specs(&entry.graph, se) {
                    plans.push((block_id, se_idx, spec.resources));
                }
            }
        }
        for (block_id, se_idx, specs) in plans {
            let owner = entry.graph.skeleton.blocks[block_id].side_effects[se_idx].semantic_id;
            for (slot, (elem_ty, kind)) in specs.into_iter().enumerate() {
                let elem_bytes = crate::ssa::layout::type_byte_size(&elem_ty)
                    .expect("seg scratch element has no static size")
                    as u32;
                let size = super::program::LogicalSize::SameAsDispatch { elem_bytes };
                let id = ResourceId(next);
                next += 1;
                resources.push(LogicalResource {
                    id,
                    origin: ResourceOrigin::Compiler(CompilerResource::new(kind, owner, slot)),
                    elem_ty,
                    size,
                });
            }
        }
    }
    resources
}

struct ReduceAnalysis {
    block: BlockId,
    effect: usize,
    partials: Vec<ResourceId>,
    stores: Vec<Vec<ReduceOutputStore>>,
    outputs: Vec<Vec<(ResourceId, Type<TypeName>, super::program::LogicalSize)>>,
}

struct ReduceOutputStore {
    location: (BlockId, usize),
    place: NodeId,
    value: NodeId,
    writer: Option<super::types::EffectToken>,
}

fn analyze_reduce_entry(
    entry: &super::program::PlannedEntry,
    resources: &[LogicalResource],
) -> Option<ReduceAnalysis> {
    let (block, effect, _) = kernel_effect(&entry.graph)?;
    let side_effect = &entry.graph.skeleton.blocks[block].side_effects[effect];
    if seg_scratch_specs(&entry.graph, side_effect)?.family != SegScratchFamily::Reduce {
        return None;
    }
    let SideEffectKind::Soac(Soac::Screma(screma::Op { body, .. })) = &side_effect.kind else {
        return None;
    };
    let screma::Kind::Reduce(_) = &body.kind else {
        return None;
    };
    let operators = body.kind.operators();
    let n_inputs = body.inputs.len();
    let n_accs = operators.len();
    let n_maps = body.maps.len();
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
    let owner = side_effect.semantic_id?;
    let partials = owned_resource_ids(resources, owner, CompilerResourceKind::ReducePartial);
    if partials.len() != n_accs
        || operators.iter().any(|operator| !can_clone_pure_subgraph(&entry.graph, operator.neutral, &[]))
    {
        return None;
    }
    let mut stores = (0..n_accs).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut outputs: Vec<Vec<(ResourceId, Type<TypeName>, super::program::LogicalSize)>> =
        vec![Vec::new(); n_accs];
    for (block_id, block) in &entry.graph.skeleton.blocks {
        for (effect_index, effect) in block.side_effects.iter().enumerate() {
            if !matches!(effect.kind, SideEffectKind::Inst(InstKind::Store { .. })) {
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
                let logical = resources.get(resource.0 as usize)?;
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
        ReduceAnalysis {
            block,
            effect,
            partials,
            stores,
            outputs,
        },
    )
}

fn emit_reduce_entry(
    entry: &mut super::program::PlannedEntry,
    analysis: ReduceAnalysis,
    schedule: &schedule::KernelPlan,
    resources: &[LogicalResource],
) -> Vec<super::program::PlannedEntry> {
    let ReduceAnalysis {
        block: block_id,
        effect: idx,
        partials: partial_resources,
        stores,
        outputs: acc_output_decls,
    } = analysis;
    debug_assert_eq!(
        kernel_effect(&entry.graph).map(|(block, effect, _)| (block, effect)),
        Some((block_id, idx))
    );
    let total_threads = REDUCE_PHASE1_WIDTH;
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
    let (
        reduce_funcs,
        n_maps,
        map_output_view_ops,
        input_view_data,
        init_nids,
        elem_tys,
        screma_result_nid,
    ) = {
        let se = &entry.graph.skeleton.blocks[block_id].side_effects[idx];
        let SideEffectKind::Soac(Soac::Screma(screma::Op { body, .. })) = &se.kind else {
            unreachable!("reduce analysis admitted a non-reduce recipe")
        };
        let screma::Kind::Reduce(_) = &body.kind else {
            unreachable!("reduce analysis admitted a non-reduce recipe")
        };
        let operators = body.kind.operators();
        let n_inputs = body.inputs.len();
        let n_maps = body.maps.len();
        // Map outputs that have been retargeted to OutputView so the
        // chunked writes inside the Screma loop body land in the right
        // buffer. Fresh destinations would require building an immutable
        // array per loop iteration — incompatible with chunked parallel
        // writes.
        // Reduce accumulators expect Fresh destination (scalar result
        // routed via a Project-based Store outside the Screma loop).
        debug_assert!(body.maps.iter().all(|map| map.destination == SoacDestination::OutputView));
        debug_assert!(operators.iter().all(|operator| operator.destination == SoacDestination::Fresh));
        // Operand layout (gate enforces zero captures everywhere):
        //   [inputs(n_inputs), init_accs(n_accs), map_output_views(n_maps),
        //    acc_output_views(0 — all Fresh)]
        let input_view_data: Vec<(NodeId, Type<TypeName>)> = (0..n_inputs)
            .map(|k| {
                let v = se.operand_nodes[k];
                (v, entry.graph.types[&v].clone())
            })
            .collect();
        let init_nids: Vec<NodeId> = operators.iter().map(|op| op.neutral).collect();
        // `[inputs.., init_accs.., map_output_views..]`: views follow the
        // init accumulators (captures live on the SegBodies).
        let base = n_inputs;
        let map_view_ops: Vec<usize> = (0..n_maps).map(|m| base + m).collect();
        let result = se.result.expect("reduce analysis requires a result");
        let elem_tys: Vec<Type<TypeName>> =
            init_nids.iter().map(|n| entry.graph.types[n].clone()).collect();
        let reduce_funcs = schedule.callable_names(operators.iter().map(|a| a.combine.region));
        (
            reduce_funcs,
            n_maps,
            map_view_ops,
            input_view_data,
            init_nids,
            elem_tys,
            result,
        )
    };

    debug_assert_eq!(n_accs, elem_tys.len());
    // 3. Chunk all input views and every map output view; swap them back
    // into the Screma operand list.
    let chunked = chunk_soac_inputs(
        &mut entry.graph,
        &input_view_data,
        total_threads,
        ChunkInputKind::StorageOrRange,
        "SegRed",
    )
    .expect("reduce analysis admitted an unchunkable input");
    let chunk_start = chunked.chunk_start;
    let chunk_len = chunked.chunk_len;
    {
        let se = &mut entry.graph.skeleton.blocks[block_id].side_effects[idx];
        for (i, &new_view) in chunked.views.iter().enumerate() {
            se.operand_nodes[i] = new_view;
        }
    }
    for (m_idx, op_idx) in map_output_view_ops.iter().enumerate() {
        let orig_view = entry.graph.skeleton.blocks[block_id].side_effects[idx].operand_nodes[*op_idx];
        let view_ty = entry.graph.types[&orig_view].clone();
        let chunked_view = chunk_view_like(
            &mut entry.graph,
            orig_view,
            view_ty,
            chunk_start,
            chunk_len,
            ChunkInputKind::StorageOnly,
            &format!("SegRed map output {m_idx}"),
        )
        .expect("reduce analysis admitted an unchunkable map output");
        let se = &mut entry.graph.skeleton.blocks[block_id].side_effects[idx];
        se.operand_nodes[*op_idx] = chunked_view;
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
            )
        })
        .collect();
    // Drop the decomposed output stores (highest index first per block).
    drop_locations.sort_by(|a, b| b.1.cmp(&a.1));
    for (bid, sx) in drop_locations {
        entry.graph.skeleton.blocks[bid].side_effects.remove(sx);
    }
    for route in &mut entry.output_routes {
        route.writers.retain(
            |writer| !matches!(writer, OutputWriter::Effect(effect) if dropped_writers.contains(effect)),
        );
    }
    let mut next_effect = graph_ops::next_effect_token(&entry.graph);
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
            &mut next_effect,
            None,
        );
        // Clear the moved output bindings from phase 1; register partials.
        for (resource, _, _) in &acc_output_decls[acc_i] {
            if let Some(binding) = resources[resource.0 as usize].host_binding() {
                for output in &mut entry.outputs {
                    if output.storage_binding == Some(binding) {
                        output.storage_binding = None;
                    }
                }
            }
        }
        entry.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(partial_resources[acc_i]),
            role: crate::interface::StorageRole::Intermediate,
            elem_ty: elem_tys[acc_i].clone(),
            size: resources[partial_resources[acc_i].0 as usize].size.clone(),
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
        )
        .expect("reduce analysis admitted an unprojectable output");
        phase2s.push(phase2);
    }
    // Scheduling consumed the semantic SegRed. Phase 1 is now an ordinary
    // per-invocation Screma over the thread's chunk; `soac_expand` lowers that
    // local loop while the synthesized phase-2 entries combine its partials.
    make_screma_serial(&mut entry.graph, block_id, idx);
    phase2s
}

struct ScanAnalysis {
    block: BlockId,
    effect: usize,
    block_sums: ResourceId,
    block_offsets: ResourceId,
}

fn analyze_scan_entry(
    entry: &super::program::PlannedEntry,
    resources: &[LogicalResource],
) -> Option<ScanAnalysis> {
    let (block, effect, _) = kernel_effect(&entry.graph)?;
    let side_effect = &entry.graph.skeleton.blocks[block].side_effects[effect];
    if seg_scratch_specs(&entry.graph, side_effect)?.family != SegScratchFamily::Scan {
        return None;
    }
    let SideEffectKind::Soac(Soac::Screma(screma::Op { body, .. })) = &side_effect.kind else {
        return None;
    };
    let screma::Kind::Scan(_) = &body.kind else {
        return None;
    };
    let operators = body.kind.operators();
    let operator = operators[0];
    if !can_clone_pure_subgraph(&entry.graph, operator.neutral, &[]) {
        return None;
    }
    let input = *side_effect.operand_nodes.first()?;
    if !can_chunk_view(&entry.graph, input, ChunkInputKind::StorageOrRange) {
        return None;
    }
    let output_base = body.inputs.len();
    if !(0..body.maps.len()).all(|index| {
        side_effect
            .operand_nodes
            .get(output_base + index)
            .is_some_and(|view| can_chunk_view(&entry.graph, *view, ChunkInputKind::StorageOnly))
    }) {
        return None;
    }
    let scan_output = *side_effect.operand_nodes.get(output_base + body.maps.len())?;
    graph_ops::extract_storage_view_source(&entry.graph, scan_output)?;
    let owner = side_effect.semantic_id?;
    let block_sums = *owned_resource_ids(resources, owner, CompilerResourceKind::ScanBlockSums).first()?;
    let block_offsets =
        *owned_resource_ids(resources, owner, CompilerResourceKind::ScanBlockOffsets).first()?;
    Some(ScanAnalysis {
        block,
        effect,
        block_sums,
        block_offsets,
    })
}

fn emit_scan_entry(
    entry: &mut super::program::PlannedEntry,
    analysis: ScanAnalysis,
    schedule: &mut schedule::KernelPlan,
    resources: &[LogicalResource],
) -> Vec<super::program::PlannedEntry> {
    debug_assert_eq!(
        kernel_effect(&entry.graph).map(|(block, effect, _)| (block, effect)),
        Some((analysis.block, analysis.effect))
    );
    let total_threads = REDUCE_PHASE1_WIDTH;
    let (block_id, idx) = (analysis.block, analysis.effect);
    let (block_sums_resource, block_offsets_resource) = (analysis.block_sums, analysis.block_offsets);
    let (
        op_func,
        reduce_func,
        map_output_view_ops,
        scan_output_view_op,
        input_view_nid,
        input_view_ty,
        input_elem_ty,
        step_capture_nodes,
        init_nid,
        elem_ty,
    ) = {
        let se = &entry.graph.skeleton.blocks[block_id].side_effects[idx];
        let SideEffectKind::Soac(Soac::Screma(screma::Op { body, .. })) = &se.kind else {
            unreachable!("scan analysis admitted a non-scan recipe")
        };
        let screma::Kind::Scan(_) = &body.kind else {
            unreachable!("scan analysis admitted a non-scan recipe")
        };
        let operators = body.kind.operators();
        debug_assert_eq!(operators.len(), 1);
        let op = operators[0];
        let n_inputs = body.inputs.len();
        debug_assert_eq!(n_inputs, 1);
        debug_assert!(op.combine.captures.is_empty());
        debug_assert!(body.maps.iter().all(|map| map.destination == SoacDestination::OutputView));
        // `realize_outputs` retargets the scan accumulator to OutputView (its
        // prefixes feed the entry output) and appends the scan output buffer.
        debug_assert_eq!(op.destination, SoacDestination::OutputView);
        let n_maps = body.maps.len();
        let input_nid = se.operand_nodes[0];
        let init_nid = op.neutral;
        let step_capture_nodes = op.step.captures.clone();
        // `[inputs.., init_accs.., map_output_views.., scan_output_view]`:
        // output views follow the init accumulators (captures live on SegBodies).
        let output_view_base = n_inputs;
        let scan_output_view_op = output_view_base + n_maps;
        let map_output_view_ops: Vec<usize> = (0..n_maps).map(|map| output_view_base + map).collect();
        let input_ty = entry.graph.types[&input_nid].clone();
        let input_elem = body.inputs[0].element.clone();
        let elem = entry.graph.types[&init_nid].clone();
        (
            schedule.callable_name(op.step.region).to_string(),
            schedule.callable_name(op.combine.region).to_string(),
            map_output_view_ops,
            scan_output_view_op,
            input_nid,
            input_ty,
            input_elem,
            step_capture_nodes,
            init_nid,
            elem,
        )
    };

    // Chunk the input and the scan output view; swap them into the operand list.
    let chunked = chunk_soac_inputs(
        &mut entry.graph,
        &[(input_view_nid, input_view_ty.clone())],
        total_threads,
        ChunkInputKind::StorageOrRange,
        "SegScan",
    )
    .expect("scan analysis admitted an unchunkable input");
    let chunk_start = chunked.chunk_start;
    let chunk_len = chunked.chunk_len;
    let chunked_input_nid = chunked.views[0];
    {
        let se = &mut entry.graph.skeleton.blocks[block_id].side_effects[idx];
        se.operand_nodes[0] = chunked_input_nid;
    }
    for (map_index, operand_index) in map_output_view_ops.iter().enumerate() {
        let original =
            entry.graph.skeleton.blocks[block_id].side_effects[idx].operand_nodes[*operand_index];
        let view_ty = entry.graph.types[&original].clone();
        let chunked_view = chunk_view_like(
            &mut entry.graph,
            original,
            view_ty,
            chunk_start,
            chunk_len,
            ChunkInputKind::StorageOnly,
            &format!("SegScan map output {map_index}"),
        )
        .expect("scan analysis admitted an unchunkable map output");
        entry.graph.skeleton.blocks[block_id].side_effects[idx].operand_nodes[*operand_index] =
            chunked_view;
    }
    let (scan_output_storage, orig_scan_output_view_ty) = {
        let se = &entry.graph.skeleton.blocks[block_id].side_effects[idx];
        let v = se.operand_nodes[scan_output_view_op];
        let ty = entry.graph.types[&v].clone();
        let storage = graph_ops::extract_storage_view_source(&entry.graph, v)
            .expect("scan analysis requires a storage output");
        (storage, ty)
    };
    let chunked_scan_output = graph_ops::intern_chunked_resource_view(
        &mut entry.graph,
        scan_output_storage.0,
        chunk_start,
        chunk_len,
        orig_scan_output_view_ty,
        None,
    );
    {
        let se = &mut entry.graph.skeleton.blocks[block_id].side_effects[idx];
        se.operand_nodes[scan_output_view_op] = chunked_scan_output;
    }

    // Append a chunked reduce over the same input that stores each thread's
    // final accumulator to `block_sums[tid]`.
    {
        let mut next_effect = graph_ops::next_effect_token(&entry.graph);
        // `[chunked_input, init]` — the step captures live on the SegBody below.
        let reduce_operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![chunked_input_nid, init_nid];
        let tuple_ty = Type::Constructed(TypeName::Tuple(1), vec![elem_ty.clone()]);
        let screma_nid = graph_ops::emit_pending_soac(
            &mut entry.graph,
            block_id,
            Soac::Screma(screma::Op {
                body: screma::Body {
                    inputs: vec![super::types::SoacInputType {
                        array: input_view_ty,
                        element: input_elem_ty,
                    }],
                    maps: vec![],
                    kind: screma::Kind::Reduce(
                        screma::NonEmpty::from_vec(vec![screma::Operator {
                            step: SegBody {
                                region: schedule.intern_callable(&op_func),
                                captures: step_capture_nodes,
                            },
                            combine: SegBody {
                                region: schedule.intern_callable(&op_func),
                                captures: vec![],
                            },
                            input_indices: vec![0],
                            neutral: init_nid,
                            shape: Vec::new(),
                            commutative: false,
                            destination: SoacDestination::Fresh,
                            result_type: elem_ty.clone(),
                        }])
                        .expect("one reduce operator"),
                    ),
                },
                state: screma::SemanticState::Serial,
            }),
            reduce_operands,
            tuple_ty,
            &mut next_effect,
            None,
        );
        let result_nid = entry.graph.intern_pure(
            super::types::PureOp::Project { index: 0 },
            smallvec![screma_nid],
            elem_ty.clone(),
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
            &mut next_effect,
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
            size: resources[resource.0 as usize].size.clone(),
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
    )
    .expect("scan analysis admitted an unprojectable neutral");
    apply_manifest_resource_sizes(&mut phase2, resources);
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
    );
    apply_manifest_resource_sizes(&mut phase3, resources);

    // Phase 1 is now a per-invocation Screma scan over the thread's chunk plus
    // the appended block-sum reduce; `soac_expand` lowers both.
    make_screma_serial(&mut entry.graph, block_id, idx);
    vec![phase2, phase3]
}

/// Synthesize phase 2 of a parallel scan: a single-invocation sequential
/// exclusive scan over `block_sums`. `block_offsets[i]` is the prefix of
/// blocks strictly before `i`, which phase 3 can safely prepend to chunk `i`.
pub fn synthesize_phase2_scan(
    entry_name: &str,
    op_func: String,
    elem_ty: Type<TypeName>,
    phase1_graph: &super::types::EGraph,
    phase1_ne_nid: NodeId,
    block_sums_resource: ResourceId,
    block_offsets_resource: ResourceId,
    len_out: Option<ResourceId>,
) -> Result<super::program::PlannedEntry, String> {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(format!("{}_phase2_scan_sums", entry_name), (1, 1, 1));
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
        let mut effect = phase2.effect;
        let graph = b.graph_mut();
        let len_view = graph_ops::intern_resource_view(graph, len_out, elem_ty.clone(), None);
        graph_ops::emit_storage_store(
            graph,
            phase2.after,
            len_view,
            phase2.zero,
            total,
            elem_ty,
            &mut effect,
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
    /// Effect-chain cursor after the loop, so an appended store sequences after
    /// the loop's own effects with the same token the loop would have used.
    effect: u32,
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
    let graph = b.graph_mut();
    let sums = graph_ops::intern_resource_view(graph, block_sums_resource, arr_ty.clone(), None);
    let offsets = graph_ops::intern_resource_view(graph, block_offsets_resource, arr_ty, None);
    let len = emit_resource_len(graph, block_sums_resource);
    let zero = graph_ops::intern_u32(graph, 0, None);
    let one = graph_ops::intern_u32(graph, 1, None);
    let mut effect = graph_ops::next_effect_token(graph);

    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let cont = graph.skeleton.create_block();
    let after = graph.skeleton.create_block();
    let acc = graph.add_block_param(header, 0, elem_ty.clone());
    graph.skeleton.blocks[header].params.push(acc);
    let index = graph.add_block_param(header, 1, u32_ty.clone());
    graph.skeleton.blocks[header].params.push(index);
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
    b.control_headers_mut().insert(
        header,
        ControlHeader::Loop {
            merge: after,
            continue_block: cont,
        },
    );

    let graph = b.graph_mut();
    graph_ops::emit_storage_store(
        graph,
        body,
        offsets,
        index,
        acc,
        elem_ty.clone(),
        &mut effect,
        None,
    );
    let value = graph_ops::emit_view_load(graph, body, sums, index, elem_ty.clone(), &mut effect, None);
    let next_acc = graph.intern_pure(PureOp::Call(op_func), smallvec![acc, value], elem_ty.clone());
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: cont,
        args: vec![next_acc],
    };
    let continued_acc = graph.add_block_param(cont, 0, graph.types[&acc].clone());
    graph.skeleton.blocks[cont].params.push(continued_acc);
    let next_index = graph_ops::intern_binop(graph, "+", index, one, u32_ty, None);
    graph.skeleton.blocks[cont].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![continued_acc, next_index],
    };
    // Expose the grand total as an `after` block param when requested; the
    // caller (a filter compaction) appends its own store. The generic scan
    // never touches a length cell.
    let total = if want_total {
        let total = graph.add_block_param(after, 0, elem_ty.clone());
        graph.skeleton.blocks[after].params.push(total);
        Some(total)
    } else {
        None
    };
    b.set_current_block(after);
    ExclusiveScanPhase2 {
        total,
        after,
        effect,
        zero,
    }
}

/// Synthesize phase 3 of a parallel scan: a chunked compute entry where each
/// thread reads `off = block_offsets[tid]` and applies `op(off, output[i])` to
/// every element of its chunk of `output`. Map's call convention is
/// `func(elem, ...captures)`, so phase 3 routes through `swap_wrapper_name`
/// (`\(elem, off) -> op(off, elem)`) to keep `off` in the accumulator slot for
/// non-commutative ops.
pub fn synthesize_phase3_scan(
    entry_name: &str,
    swap_region: RegionId,
    elem_ty: Type<TypeName>,
    output_resource: ResourceId,
    block_offsets_resource: ResourceId,
    total_threads: u32,
) -> super::program::PlannedEntry {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(
        format!("{}_phase3_add_offsets", entry_name),
        (total_threads, 1, 1),
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
    let (tid, chunk_start, chunk_len) = emit_chunk_arithmetic(b.graph_mut(), total_threads, output_len)
        .expect("phase3: chunk arithmetic must succeed (u32.min is in the prelude)");

    let off_place = b.graph_mut().intern_pure(
        super::types::PureOp::ViewIndex,
        smallvec![block_offsets_view, tid],
        elem_ty.clone(),
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
        elem_ty.clone(),
        elem_ty,
        vec![off],
        chunked_output,
        arr_ty,
    );
    b.build()
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
        graph.intern_pure(PureOp::Call(inner), smallvec![b_nid, a_nid], result_ty)
    })
}

fn synthesize_u32_add_function(name: String, span: crate::ast::Span) -> SemanticFunc {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let result_ty = u32_ty.clone();
    synthesize_binary_fn(name, u32_ty, span, move |graph, a_nid, b_nid| {
        graph.intern_pure(PureOp::BinOp("+".into()), smallvec![a_nid, b_nid], result_ty)
    })
}

#[cfg(test)]
#[path = "parallelize_tests.rs"]
mod parallelize_tests;
