//! EGIR-side SOAC parallelization in two explicit stages:
//!
//! 1. `reify` turns every reachable Screma into a semantic `SegMap`,
//!    `SegRed`, or `SegScan` operations.
//! 2. `lower` schedules those operations: maps remain segmented for
//!    `soac_expand`, reductions become two-phase trees, and scans become
//!    three-phase block scans.
pub mod schedule;

use crate::LookupMap;

use polytype::Type;
use smallvec::smallvec;

use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ControlHeader, InstKind};
use crate::types::TypeExt;
use crate::BindingRef;

use super::graph_ops;
use super::program::{
    CompilerResource, CompilerResourceKind, EgirEntry, EgirFunc, EgirInner, EgirRegion, LogicalResource,
    RegionInterner, ResourceId, ResourceOrigin, SemanticDependency, SemanticDependencyKind, SemanticOpId,
};
use super::types::{
    EGraph, ENode, EgirSoac, FilterPhase, NodeId, PureOp, RegionId, ScremaOperator, SegBinOp, SegBody,
    SegExtent, SegLevel, SegOpKind, SegPlacement, SegResourceAccess, SegResourceAccessKind, SegSpace,
    SideEffect, SideEffectKind, SkeletonTerminator, SoacDestination,
};

/// Per-workgroup width of a synthesized phase-2 tree reduce.
pub const PHASE2_WIDTH: u32 = 256;
/// Per-workgroup width used to chunk a phase-1 partial reduce.
const REDUCE_PHASE1_WIDTH: u32 = 64;

/// Reify every reachable Screma as a semantic segmented op.
/// This pass performs no scheduling and allocates no bindings.
pub fn reify(inner: &mut EgirInner) {
    for entry in inner.entry_points.iter_mut() {
        reify_tail_soac(entry);
    }
    for function in inner.functions.iter_mut() {
        reify_function_soacs(function);
        reify_function_filter_spaces(function);
    }
    rebuild_semantic_dependencies(inner);
}

pub(crate) fn rebuild_semantic_dependencies(inner: &mut EgirInner) {
    inner.semantic_dependencies = semantic_dependencies(inner);
}

fn semantic_dependencies(inner: &EgirInner) -> Vec<SemanticDependency> {
    let mut dependencies = Vec::new();
    for entry in &inner.entry_points {
        collect_graph_dependencies(&entry.name, &entry.graph, &mut dependencies);
    }
    for function in &inner.functions {
        collect_graph_dependencies(&function.name, &function.graph, &mut dependencies);
    }
    dependencies
}

fn collect_graph_dependencies(scope: &str, graph: &EGraph, output: &mut Vec<SemanticDependency>) {
    struct Record<'a> {
        id: SemanticOpId,
        effect: &'a SideEffect,
        resources: Vec<SegResourceAccess>,
    }

    let mut records = Vec::new();
    for (_, block) in &graph.skeleton.blocks {
        for effect in &block.side_effects {
            let SideEffectKind::Soac(soac) = &effect.kind else {
                continue;
            };
            if let Some(result) = effect.result {
                let resources = match soac {
                    EgirSoac::Seg { resources, .. } => resources.clone(),
                    EgirSoac::Filter {
                        scratch_out,
                        len_out,
                        work_buffers,
                        ..
                    } => {
                        let mut resources = semantic_read_resources(graph, effect);
                        for binding in scratch_out.iter().chain(len_out.iter()) {
                            resources.push(SegResourceAccess {
                                binding: *binding,
                                access: SegResourceAccessKind::Write,
                            });
                        }
                        if let Some(work) = work_buffers {
                            for binding in [work.flags, work.offsets] {
                                resources.push(SegResourceAccess {
                                    binding,
                                    access: SegResourceAccessKind::ReadWrite,
                                });
                            }
                        }
                        resources
                    }
                    EgirSoac::Hist { .. } => {
                        let mut resources = semantic_read_resources(graph, effect);
                        if let Some(destination) = effect
                            .operand_nodes
                            .first()
                            .and_then(|node| graph_ops::extract_storage_view_source(graph, *node))
                        {
                            if let Some(resource) =
                                resources.iter_mut().find(|resource| resource.binding == destination)
                            {
                                resource.access = SegResourceAccessKind::ReadWrite;
                            }
                        }
                        resources
                    }
                    EgirSoac::Screma { .. } => Vec::new(),
                };
                records.push(Record {
                    id: SemanticOpId {
                        scope: scope.to_string(),
                        result,
                    },
                    effect,
                    resources,
                });
            }
        }
    }

    for consumer_index in 0..records.len() {
        let consumer = &records[consumer_index];
        let mut reachable = std::collections::HashSet::new();
        let mut stack = consumer.effect.referenced_nodes().collect::<Vec<_>>();
        while let Some(node) = stack.pop() {
            if reachable.insert(node) {
                stack.extend(graph.nodes[node].children());
            }
        }
        for producer in &records[..consumer_index] {
            if reachable.contains(&producer.id.result) {
                push_dependency(output, &producer.id, &consumer.id, SemanticDependencyKind::Value);
            }
            if matches!((producer.effect.effects, consumer.effect.effects),
                (Some((_, out)), Some((input, _))) if out == input)
            {
                push_dependency(output, &producer.id, &consumer.id, SemanticDependencyKind::Effect);
            }
            if producer.resources.iter().any(|left| {
                consumer.resources.iter().any(|right| {
                    left.binding == right.binding
                        && (left.access != SegResourceAccessKind::Read
                            || right.access != SegResourceAccessKind::Read)
                })
            }) {
                push_dependency(
                    output,
                    &producer.id,
                    &consumer.id,
                    SemanticDependencyKind::Resource,
                );
            }
        }
    }
}

fn push_dependency(
    output: &mut Vec<SemanticDependency>,
    producer: &SemanticOpId,
    consumer: &SemanticOpId,
    kind: SemanticDependencyKind,
) {
    let edge = SemanticDependency {
        producer: producer.clone(),
        consumer: consumer.clone(),
        kind,
    };
    if !output.contains(&edge) {
        output.push(edge);
    }
}

/// Validate the semantic boundary before any target-aware scheduling occurs.
pub fn verify_semantic(inner: &EgirInner) -> Result<(), String> {
    let verify_effect = |scope: &str, effect: &SideEffect| -> Result<(), String> {
        if matches!(effect.kind, SideEffectKind::Soac(EgirSoac::Screma { .. })) {
            return Err(format!("{scope}: raw Screma survived semantic segmentation"));
        }
        if let SideEffectKind::Soac(EgirSoac::Hist { space, body, .. }) = &effect.kind {
            if space.as_ref().is_none_or(|space| space.dims.is_empty()) {
                return Err(format!("{scope}: semantic SegHist has no concrete dimensions"));
            }
            if !inner.regions.contains_key(&body.region) {
                return Err(format!(
                    "{scope}: histogram region `{}` is absent",
                    body.region.index()
                ));
            }
            return Ok(());
        }
        if let SideEffectKind::Soac(EgirSoac::Filter {
            space,
            map_body,
            pred_body,
            ..
        }) = &effect.kind
        {
            if space.as_ref().is_none_or(|space| space.dims.is_empty()) {
                return Err(format!("{scope}: semantic SegFilter has no concrete dimensions"));
            }
            for body in map_body.iter().chain(std::iter::once(pred_body)) {
                if !inner.regions.contains_key(&body.region) {
                    return Err(format!(
                        "{scope}: filter region `{}` is absent",
                        body.region.index()
                    ));
                }
            }
            return Ok(());
        }
        let SideEffectKind::Soac(EgirSoac::Seg {
            space,
            kind,
            map_bodies,
            ..
        }) = &effect.kind
        else {
            return Ok(());
        };
        if space.dims.is_empty() {
            return Err(format!("{scope}: semantic SegOp has no concrete dimensions"));
        }
        for body in map_bodies {
            if !inner.regions.contains_key(&body.region) {
                return Err(format!(
                    "{scope}: map region `{}` is absent from the EGIR region arena",
                    body.region.index()
                ));
            }
        }
        let operators = match kind {
            SegOpKind::SegMap => &[][..],
            SegOpKind::SegRed { operators }
            | SegOpKind::SegScan { operators }
            | SegOpKind::SegComposite { operators } => operators.as_slice(),
        };
        for operator in operators {
            for body in [&operator.step, &operator.combine] {
                if !inner.regions.contains_key(&body.region) {
                    return Err(format!(
                        "{scope}: operator region `{}` is absent from the EGIR region arena",
                        body.region.index()
                    ));
                }
            }
        }
        Ok(())
    };
    for entry in &inner.entry_points {
        for (_, block) in &entry.graph.skeleton.blocks {
            for effect in &block.side_effects {
                verify_effect(&format!("entry `{}`", entry.name), effect)?;
            }
        }
    }
    for function in &inner.functions {
        for (_, block) in &function.graph.skeleton.blocks {
            for effect in &block.side_effects {
                verify_effect(&format!("function `{}`", function.name), effect)?;
            }
        }
    }
    Ok(())
}

pub fn semantic_summary(inner: &EgirInner) -> String {
    let mut output = String::new();
    let mut print_graph = |scope: &str, graph: &EGraph| {
        for (_, block) in &graph.skeleton.blocks {
            for effect in &block.side_effects {
                match &effect.kind {
                    SideEffectKind::Soac(EgirSoac::Seg {
                        space,
                        placement,
                        kind,
                        map_bodies,
                        output_slots,
                        resources,
                        ..
                    }) => {
                        use std::fmt::Write;
                        let _ = writeln!(
                            output,
                            "{scope}: {kind:?} {placement:?} space={space:?} maps={map_bodies:?} outputs={output_slots:?} resources={resources:?}"
                        );
                    }
                    SideEffectKind::Soac(EgirSoac::Filter {
                        space,
                        map_body,
                        pred_body,
                        ..
                    }) => {
                        use std::fmt::Write;
                        let _ = writeln!(
                            output,
                            "{scope}: SegFilter space={space:?} map={map_body:?} predicate={pred_body:?}"
                        );
                    }
                    SideEffectKind::Soac(EgirSoac::Hist {
                        space,
                        body,
                        update_policy,
                        ..
                    }) => {
                        use std::fmt::Write;
                        let _ = writeln!(
                            output,
                            "{scope}: SegHist space={space:?} body={body:?} update={update_policy:?}"
                        );
                    }
                    _ => {}
                }
            }
        }
    };
    for entry in &inner.entry_points {
        print_graph(&format!("entry {}", entry.name), &entry.graph);
    }
    for function in &inner.functions {
        print_graph(&format!("function {}", function.name), &function.graph);
    }
    output
}

fn reify_function_soacs(function: &mut EgirFunc) {
    let locations: Vec<(BlockId, usize)> = function
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(block_id, block)| {
            block.side_effects.iter().enumerate().filter_map(move |(index, effect)| {
                matches!(effect.kind, SideEffectKind::Soac(EgirSoac::Screma { .. }))
                    .then_some((block_id, index))
            })
        })
        .collect();
    for (block_id, index) in locations {
        reify_function_screma(function, block_id, index);
    }
}

fn reify_function_screma(function: &mut EgirFunc, block_id: BlockId, index: usize) {
    let effect = function.graph.skeleton.blocks[block_id].side_effects[index].clone();
    let SideEffectKind::Soac(EgirSoac::Screma {
        map_bodies,
        accumulators,
        input_array_types,
        input_elem_types,
        map_output_elem_types,
        map_input_indices,
        map_destinations,
        acc_destinations,
    }) = &effect.kind
    else {
        return;
    };
    let n_inputs = input_array_types.len();
    let neutrals: Vec<NodeId> =
        (0..accumulators.len()).map(|i| effect.operand_nodes[n_inputs + i]).collect();
    let map_bodies = map_bodies.clone();
    let kind = reify_seg_kind(accumulators, &neutrals, n_inputs);
    let result_types = effect
        .result
        .and_then(|result| function.graph.types.get(&result))
        .map(|ty| match ty {
            Type::Constructed(TypeName::Tuple(_), fields) => fields.clone(),
            other => vec![other.clone()],
        })
        .unwrap_or_default();
    let space = semantic_space_for_graph(&function.graph, &effect, input_array_types, input_elem_types);
    let resources = semantic_read_resources(&function.graph, &effect);
    function.graph.skeleton.blocks[block_id].side_effects[index].kind =
        SideEffectKind::Soac(EgirSoac::Seg {
            space,
            placement: SegPlacement::LaneLocal,
            kind,
            map_bodies,
            input_array_types: input_array_types.clone(),
            input_elem_types: input_elem_types.clone(),
            map_output_elem_types: map_output_elem_types.clone(),
            map_input_indices: map_input_indices.clone(),
            map_destinations: map_destinations.clone(),
            acc_destinations: acc_destinations.clone(),
            result_types,
            output_slots: Vec::new(),
            resources,
            scratch_resources: Vec::new(),
        });
}

/// Lower semantic segmented operations into executable kernel entries.
/// Pointwise `SegMap`s remain for `soac_expand`; `SegRed`s become a chunked
/// phase 1 plus a synthesized tree reduction; `SegScan`s become chunk scans,
/// an exclusive scan of block sums, and offset-application phases.
pub fn lower(inner: &mut EgirInner) {
    use schedule::{KernelDomain, KernelSchedule};

    // Borrow the region interner independently of the other `inner` fields so
    // synthesized phase kernels can intern their helper regions while the entry
    // points are mutated in place. Restored before returning.
    let mut regions = std::mem::take(&mut inner.region_interner);
    let mut schedule = KernelSchedule::seed(&inner.pipeline, &inner.entry_points);
    attach_materialization_prepasses(inner, &mut schedule);
    let filter_phases = lower_runtime_filters(inner, &mut schedule);
    inner.entry_points.extend(filter_phases);
    let mut new_entries: Vec<EgirEntry> = Vec::new();
    let mut new_functions: Vec<EgirFunc> = Vec::new();
    let mut new_regions: Vec<(RegionId, EgirRegion)> = Vec::new();
    // Snapshot ResourceId → physical binding (the manifest was built at
    // `allocate()`); the reduce/scan lowerings resolve their reserved scratch
    // through this instead of drawing fresh bindings.
    let resource_bindings: Vec<BindingRef> =
        inner.resources.iter().map(|resource| resource.legacy_binding).collect();
    for entry in inner.entry_points.iter_mut() {
        let Some(kind) = find_pending_kernel_seg(entry).map(|(bid, idx)| {
            let se = &entry.graph.skeleton.blocks[bid].side_effects[idx];
            let SideEffectKind::Soac(EgirSoac::Seg { kind, .. }) = &se.kind else {
                unreachable!()
            };
            kind.clone()
        }) else {
            continue;
        };
        match kind {
            SegOpKind::SegMap => {}
            SegOpKind::SegRed { .. } => {
                let original = entry.clone();
                if let Some(phases) = lower_reduce_entry(entry, &mut regions, &resource_bindings) {
                    let mut predecessor = entry.name.clone();
                    for ph in &phases {
                        schedule.add_phase_after(
                            &predecessor,
                            ph,
                            KernelDomain::Fixed { x: 1, y: 1, z: 1 },
                        );
                        predecessor = ph.name.clone();
                    }
                    new_entries.extend(phases);
                } else {
                    *entry = original;
                    restore_serial_seg(entry);
                }
            }
            SegOpKind::SegScan { .. } => {
                let original = entry.clone();
                if let Some((phases, swap_wrapper)) =
                    lower_scan_entry(entry, &mut regions, &resource_bindings)
                {
                    let mut predecessor = entry.name.clone();
                    let phase1_domain =
                        schedule.domain_of(&entry.name).unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
                    for (phase_index, ph) in phases.iter().enumerate() {
                        let domain = if phase_index == 0 {
                            KernelDomain::Fixed { x: 1, y: 1, z: 1 }
                        } else {
                            phase1_domain.clone()
                        };
                        schedule.add_phase_after(&predecessor, ph, domain);
                        predecessor = ph.name.clone();
                    }
                    new_entries.extend(phases);
                    let region =
                        regions.get(&swap_wrapper.name).expect("synthesized scan wrapper was interned");
                    new_regions.push((region, EgirRegion::from_function(&swap_wrapper)));
                    new_functions.push(swap_wrapper);
                } else {
                    *entry = original;
                    restore_serial_seg(entry);
                }
            }
            SegOpKind::SegComposite { .. } => {
                // The semantic form survives to this target boundary. The
                // initial portable scheduler preserves legacy behavior by
                // lowering a mixed accumulator region locally.
                restore_serial_seg(entry);
            }
        }
    }
    inner.entry_points.extend(new_entries);
    inner.functions.extend(new_functions);
    inner.regions.extend(new_regions);

    restore_lane_local(inner);

    // Each distinct iteration space is its own kernel (Futhark: one SegOp per
    // SegSpace). An entry holding several pointwise SegMaps over different
    // domains is split into one entry — and one pipeline stage — per domain.
    let EgirInner { entry_points, .. } = inner;
    let mut split_clones: Vec<EgirEntry> = Vec::new();
    for entry in entry_points.iter_mut() {
        let parent = entry.name.clone();
        let clones = split_multidomain_seg_maps(entry);
        for clone in &clones {
            schedule.add_sibling(&parent, clone, KernelDomain::Fixed { x: 1, y: 1, z: 1 });
        }
        split_clones.extend(clones);
    }
    entry_points.extend(split_clones);
    schedule.reconcile_entries(entry_points);
    schedule.coalesce_compiler_dependencies(entry_points);
    inner.kernel_schedule = schedule;
    inner.region_interner = regions;
}

fn lower_runtime_filters(inner: &mut EgirInner, schedule: &mut schedule::KernelSchedule) -> Vec<EgirEntry> {
    use crate::interface::{StorageBindingDecl, StorageRole};
    use schedule::KernelDomain;
    let mut phases = Vec::new();
    for entry in &mut inner.entry_points {
        let runtime_filter_count = entry
            .graph
            .skeleton
            .blocks
            .iter()
            .flat_map(|(_, block)| &block.side_effects)
            .filter(|effect| {
                matches!(
                    effect.kind,
                    SideEffectKind::Soac(EgirSoac::Filter {
                        scratch_out: Some(_),
                        ..
                    })
                )
            })
            .count();
        if runtime_filter_count != 1 {
            continue;
        }
        let candidates: Vec<_> = entry
            .graph
            .skeleton
            .blocks
            .iter()
            .flat_map(|(_, block)| {
                block.side_effects.iter().filter_map(|effect| match &effect.kind {
                    SideEffectKind::Soac(EgirSoac::Filter {
                        space: Some(space),
                        scratch_out: Some(_),
                        len_out: Some(len_out),
                        work_buffers: Some(work),
                        phase: FilterPhase::Semantic,
                        ..
                    }) => Some((space.clone(), *work, *len_out)),
                    _ => None,
                })
            })
            .collect();
        let [(space, work, len_out)] = candidates.as_slice() else {
            continue;
        };
        let (space, work, len_out) = (space.clone(), *work, *len_out);
        let domain =
            schedule::domain_from_space(&space).unwrap_or(KernelDomain::Fixed { x: 1, y: 1, z: 1 });
        let mut flags = entry.clone();
        flags.name = format!("{}_filter_flags", entry.name);
        flags.outputs.clear();
        flags.slot_sources.clear();
        flags.return_ty = Type::Constructed(TypeName::Unit, vec![]);
        flags.storage_bindings.clear();
        flags.storage_bindings.push(StorageBindingDecl {
            binding: work.flags,
            role: StorageRole::Output,
            elem_ty: Type::Constructed(TypeName::UInt(32), vec![]),
            length: inner
                .resources
                .iter()
                .find(|resource| resource.legacy_binding == work.flags)
                .and_then(|resource| super::program::buffer_len(&resource.size)),
        });
        set_filter_phase(&mut flags, work, FilterPhase::Flags);

        let mut scan = entry.clone();
        scan.name = format!("{}_filter_scan", entry.name);
        scan.execution_model = crate::ssa::types::ExecutionModel::Compute {
            local_size: (1, 1, 1),
        };
        scan.outputs.clear();
        scan.slot_sources.clear();
        scan.return_ty = Type::Constructed(TypeName::Unit, vec![]);
        scan.storage_bindings.clear();
        for (binding, role) in [
            (work.flags, StorageRole::Input),
            (work.offsets, StorageRole::Output),
            (len_out, StorageRole::Output),
        ] {
            let elem_ty = Type::Constructed(TypeName::UInt(32), vec![]);
            let length = if binding == len_out {
                Some(crate::pipeline_descriptor::BufferLen::Fixed { bytes: 4 })
            } else {
                inner
                    .resources
                    .iter()
                    .find(|resource| resource.legacy_binding == binding)
                    .and_then(|resource| super::program::buffer_len(&resource.size))
            };
            scan.storage_bindings.push(StorageBindingDecl {
                binding,
                role,
                elem_ty,
                length,
            });
        }
        set_filter_phase(&mut scan, work, FilterPhase::Scan);
        entry.storage_bindings.push(StorageBindingDecl {
            binding: work.flags,
            role: StorageRole::Input,
            elem_ty: Type::Constructed(TypeName::UInt(32), vec![]),
            length: inner
                .resources
                .iter()
                .find(|resource| resource.legacy_binding == work.flags)
                .and_then(|resource| super::program::buffer_len(&resource.size)),
        });
        entry.storage_bindings.push(StorageBindingDecl {
            binding: work.offsets,
            role: StorageRole::Input,
            elem_ty: Type::Constructed(TypeName::UInt(32), vec![]),
            length: inner
                .resources
                .iter()
                .find(|resource| resource.legacy_binding == work.offsets)
                .and_then(|resource| super::program::buffer_len(&resource.size)),
        });
        set_filter_phase(entry, work, FilterPhase::Scatter);
        schedule.add_phase_before(&entry.name, &flags, domain);
        schedule.add_phase_before(&entry.name, &scan, KernelDomain::Fixed { x: 1, y: 1, z: 1 });
        phases.push(flags);
        phases.push(scan);
    }
    phases
}

fn set_filter_phase(entry: &mut EgirEntry, work: super::types::FilterWorkBuffers, phase: FilterPhase) {
    for (_, block) in entry.graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            if let SideEffectKind::Soac(EgirSoac::Filter {
                work_buffers: Some(current_work),
                phase: current,
                ..
            }) = &mut effect.kind
            {
                if *current_work == work {
                    *current = phase;
                }
            }
        }
    }
}

/// Attach allocation-created materialization entries to their consumer's
/// source pipeline. Used by single-stage lowering after SegOps have been
/// restored to serial Scremas; the parallel lowering performs the equivalent
/// attachment before selecting reduce/scan phases.
pub(crate) fn attach_materialization_prepasses(inner: &EgirInner, schedule: &mut schedule::KernelSchedule) {
    use schedule::KernelDomain;
    // Insert one ready producer at a time. Repeating is important when a
    // materialized producer itself depends on another compiler prepass: once
    // its consumer is scheduled, the preceding producer becomes ready too.
    while let Some((consumer, producer_index)) = inner
        .entry_points
        .iter()
        .enumerate()
        .filter(|(_, producer)| !schedule.contains_entry(&producer.name))
        .find_map(|(producer_index, producer)| {
            let outputs: Vec<_> = producer
                .storage_bindings
                .iter()
                .filter(|declaration| declaration.role == crate::interface::StorageRole::Output)
                .map(|declaration| declaration.binding)
                .collect();
            inner
                .entry_points
                .iter()
                .find(|consumer| {
                    schedule.contains_entry(&consumer.name)
                        && consumer.storage_bindings.iter().any(|declaration| {
                            declaration.role == crate::interface::StorageRole::Input
                                && outputs.contains(&declaration.binding)
                        })
                })
                .map(|consumer| (consumer.name.clone(), producer_index))
        })
    {
        schedule.add_phase_before(
            &consumer,
            &inner.entry_points[producer_index],
            KernelDomain::Fixed { x: 1, y: 1, z: 1 },
        );
    }
}

/// The output slots a side-effect contributes to, found by scanning its operand
/// subgraph for storage views that name the entry's output bindings. A pointwise
/// map writes one slot; a fused equal-domain multi-lane map writes several (one
/// per lane). Empty for a side-effect that touches no output — shared prefix
/// computation, kept by every split stage.
fn side_effect_output_slots(entry: &EgirEntry, se: &SideEffect) -> Vec<usize> {
    if let SideEffectKind::Soac(EgirSoac::Seg { output_slots, .. }) = &se.kind {
        return output_slots.clone();
    }
    use std::collections::HashSet;
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = se.referenced_nodes().collect::<Vec<_>>();
    let mut slots: Vec<usize> = Vec::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(br) = graph_ops::extract_storage_view_source(&entry.graph, n) {
            if let Some(s) = entry.outputs.iter().position(|o| o.storage_binding == Some(br)) {
                if !slots.contains(&s) {
                    slots.push(s);
                }
            }
        }
        if let Some(node) = entry.graph.nodes.get(n) {
            stack.extend(node.children());
        }
    }
    // Scalar/tuple reductions commonly feed a later Store instead of carrying
    // an OutputView operand themselves. Follow those value edges once, at
    // semantic construction time, so output routing remains attached to the
    // SegOp and placement does not mistake an entry-root reduction for a
    // lane-local helper.
    let routes_scalar_result = matches!(
        &se.kind,
        SideEffectKind::Soac(EgirSoac::Screma { accumulators, .. }) if !accumulators.is_empty()
    ) || matches!(
        &se.kind,
        SideEffectKind::Soac(EgirSoac::Seg {
            kind: SegOpKind::SegRed { .. } | SegOpKind::SegScan { .. } | SegOpKind::SegComposite { .. },
            ..
        })
    );
    if let Some(result) = se.result.filter(|_| routes_scalar_result) {
        for (_, block) in &entry.graph.skeleton.blocks {
            for consumer in &block.side_effects {
                let SideEffectKind::Inst(InstKind::Store { .. }) = &consumer.kind else {
                    continue;
                };
                let (Some(&place), Some(&value)) =
                    (consumer.operand_nodes.first(), consumer.operand_nodes.get(1))
                else {
                    continue;
                };
                if !node_reaches(&entry.graph, value, result) {
                    continue;
                }
                let Some(binding) = storage_binding_reachable(&entry.graph, place) else {
                    continue;
                };
                if let Some(slot) =
                    entry.outputs.iter().position(|output| output.storage_binding == Some(binding))
                {
                    if !slots.contains(&slot) {
                        slots.push(slot);
                    }
                }
            }
        }
    }
    slots.sort_unstable();
    slots
}

fn storage_binding_reachable(graph: &EGraph, start: NodeId) -> Option<BindingRef> {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![start];
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        if let Some(binding) = graph_ops::extract_storage_view_source(graph, node) {
            return Some(binding);
        }
        stack.extend(graph.nodes[node].children());
    }
    None
}

fn node_reaches(graph: &EGraph, start: NodeId, target: NodeId) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![start];
    while let Some(node) = stack.pop() {
        if node == target {
            return true;
        }
        if seen.insert(node) {
            stack.extend(graph.nodes[node].children());
        }
    }
    false
}

/// True for a pointwise `SegMap` side-effect (a parallel output domain).
fn is_seg_map(se: &SideEffect) -> bool {
    matches!(
        &se.kind,
        SideEffectKind::Soac(EgirSoac::Seg {
            kind: SegOpKind::SegMap,
            ..
        })
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
        SideEffectKind::Soac(EgirSoac::Hist { .. } | EgirSoac::Filter { .. }) => true,
        SideEffectKind::Soac(
            EgirSoac::Screma {
                map_destinations,
                acc_destinations,
                ..
            }
            | EgirSoac::Seg {
                map_destinations,
                acc_destinations,
                ..
            },
        ) => map_destinations
            .iter()
            .chain(acc_destinations)
            .any(|d| matches!(d, SoacDestination::OutputView | SoacDestination::InputBuffer)),
        SideEffectKind::Inst(InstKind::Store { .. }) => true,
        _ => false,
    }
}

/// Restrict `entry` to one domain group: keep the side-effects whose group is
/// `keep_group` (plus the shared ones, group `None`), drop the rest, and narrow
/// `outputs` to that group's slots. Dropped kernels are no longer emitted; their
/// now-dead pure nodes are pruned downstream.
fn restrict_to_group(
    entry: &mut EgirEntry,
    keep_group: usize,
    keep_slots: &[usize],
    group_of: &LookupMap<(BlockId, usize), usize>,
) {
    for (bid, block) in entry.graph.skeleton.blocks.iter_mut() {
        let mut drops: Vec<usize> = block
            .side_effects
            .iter()
            .enumerate()
            .filter(|(i, _)| matches!(group_of.get(&(bid, *i)), Some(g) if *g != keep_group))
            .map(|(i, _)| i)
            .collect();
        drops.sort_unstable_by(|a, b| b.cmp(a));
        for i in drops {
            block.side_effects.remove(i);
        }
    }
    entry.outputs = keep_slots.iter().map(|&s| entry.outputs[s].clone()).collect();
}

/// Remove pure producer side-effects left dead by a split. A side-effect is
/// dead when it isn't `is_write_effectful` and its result is read by nothing
/// that remains — no other side-effect's operands, no terminator. Iterated to a
/// fixpoint so a chain of producers (a map feeding a map feeding a dropped
/// scatter) collapses fully; write-effects and producers still feeding a kept
/// effect or an output are retained.
fn prune_dead_side_effects(entry: &mut EgirEntry) {
    use std::collections::HashSet;
    loop {
        // Live = every node reachable from a side-effect operand or a terminator.
        let mut live: HashSet<NodeId> = HashSet::new();
        let mut stack: Vec<NodeId> = Vec::new();
        for (_, block) in &entry.graph.skeleton.blocks {
            for se in &block.side_effects {
                stack.extend(se.referenced_nodes());
            }
            match &block.term {
                SkeletonTerminator::Return(Some(v)) => stack.push(*v),
                SkeletonTerminator::Branch { args, .. } => stack.extend(args.iter().copied()),
                SkeletonTerminator::CondBranch {
                    cond,
                    then_args,
                    else_args,
                    ..
                } => {
                    stack.push(*cond);
                    stack.extend(then_args.iter().copied());
                    stack.extend(else_args.iter().copied());
                }
                _ => {}
            }
        }
        while let Some(n) = stack.pop() {
            if !live.insert(n) {
                continue;
            }
            if let Some(node) = entry.graph.nodes.get(n) {
                stack.extend(node.children());
            }
        }

        let dead = entry.graph.skeleton.blocks.iter().find_map(|(bid, block)| {
            block.side_effects.iter().enumerate().find_map(|(i, se)| {
                let result = se.result?;
                (!is_write_effectful(se) && !live.contains(&result)).then_some((bid, i))
            })
        });
        match dead {
            Some((bid, i)) => {
                entry.graph.skeleton.blocks[bid].side_effects.remove(i);
            }
            None => break,
        }
    }
}

/// Split a multi-output entry into one kernel per iteration domain — Futhark's
/// one-SegOp-per-SegSpace structure. Side-effects are grouped by the set of
/// output slots they write (a fused equal-domain multi-lane map is one group
/// writing several slots; sibling maps over distinct domains are separate
/// groups; fixed non-SOAC outputs each form their own group). Each group
/// becomes its own entry and pipeline stage: the group owning the lowest slot
/// keeps the entry's name, the rest become `{name}_dispatch_{slot}`. Only fires
/// when at least two groups contain a parallel `SegMap`, so a single map
/// alongside fixed slots stays one sharding kernel. Returns the new clones.
fn split_multidomain_seg_maps(entry: &mut EgirEntry) -> Vec<EgirEntry> {
    let n_out = entry.outputs.len();
    if n_out <= 1 {
        return Vec::new();
    }

    // Union-find over output slots: union the slots co-written by any single
    // side-effect, so a fused multi-lane map keeps its slots in one group.
    let mut parent: Vec<usize> = (0..n_out).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        let mut cur = x;
        while parent[cur] != r {
            let next = parent[cur];
            parent[cur] = r;
            cur = next;
        }
        r
    }
    let mut producers: Vec<((BlockId, usize), Vec<usize>, bool)> = Vec::new();
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            let slots = side_effect_output_slots(entry, se);
            if slots.is_empty() {
                continue;
            }
            for &s in &slots[1..] {
                let (a, b) = (find(&mut parent, slots[0]), find(&mut parent, s));
                parent[a] = b;
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
        let root = find(&mut parent, slots[0]);
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
        return Vec::new();
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
            if side_effect_output_slots(entry, se).is_empty() && is_write_effectful(se) {
                group_of.insert((bid, i), host_root);
            }
        }
    }

    let base_name = entry.name.clone();
    let mut clones = Vec::new();
    for (root, slots) in &groups[1..] {
        let mut clone = entry.clone();
        clone.name = format!("{base_name}_dispatch_{}", slots[0]);
        restrict_to_group(&mut clone, *root, slots, &group_of);
        prune_dead_side_effects(&mut clone);
        clones.push(clone);
    }

    // The original retains the group owning the lowest slot under its own name.
    let (root0, slots0) = &groups[0];
    restrict_to_group(entry, *root0, slots0, &group_of);
    prune_dead_side_effects(entry);
    clones
}

fn reify_tail_soac(entry: &mut EgirEntry) {
    // A multi-output entry returning a tuple of sibling maps over distinct
    // domains carries one Screma side-effect per output (TLC fusion already
    // merged equal-domain siblings into a single multi-lane Screma). Reify each
    // output-rooted one — it becomes its own SegOp, matching Futhark's
    // one-SegOp-per-SegSpace structure; the schedule pass then splits distinct
    // spaces into separate kernels. Intermediate Scremas (a map feeding a
    // downstream scatter/filter/map) are left serial so `soac_expand` keeps
    // them wired into that consumer's pipeline.
    let soac_consumed = soac_consumed_nodes(entry);
    let consumed = &soac_consumed;
    let kernel_scope = matches!(
        entry.execution_model,
        crate::ssa::types::ExecutionModel::Compute { .. }
    );
    let locs: Vec<(BlockId, usize, SegPlacement)> = entry
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(bid, block)| {
            block.side_effects.iter().enumerate().filter_map(move |(i, se)| {
                matches!(&se.kind, SideEffectKind::Soac(EgirSoac::Screma { .. })).then_some((
                    bid,
                    i,
                    if !kernel_scope || se.result.is_some_and(|r| consumed.contains(&r)) {
                        SegPlacement::LaneLocal
                    } else {
                        SegPlacement::Kernel
                    },
                ))
            })
        })
        .collect();
    if locs.is_empty() {
        reify_parallel_scatter(entry);
        reify_entry_filter_spaces(entry);
        return;
    }
    for (block_id, idx, placement) in locs {
        reify_one_screma(entry, block_id, idx, placement);
    }
    // Several kernel reductions/scans in one entry write fields of a shared
    // aggregate output; independently parallel-lowering each would need its own
    // materialized scalar plus a final assembly phase (not yet available). Place
    // them lane-locally so they lower serially — horizontal fusion in `optimize`
    // then merges same-space lane-local siblings into one serial loop.
    let kernel_accumulators = entry
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| &block.side_effects)
        .filter(|effect| {
            matches!(
                effect.kind,
                SideEffectKind::Soac(EgirSoac::Seg {
                    placement: SegPlacement::Kernel,
                    kind: SegOpKind::SegRed { .. } | SegOpKind::SegScan { .. },
                    ..
                })
            )
        })
        .count();
    if kernel_accumulators > 1 {
        for (_, block) in entry.graph.skeleton.blocks.iter_mut() {
            for effect in &mut block.side_effects {
                if let SideEffectKind::Soac(EgirSoac::Seg {
                    placement,
                    kind: SegOpKind::SegRed { .. } | SegOpKind::SegScan { .. },
                    ..
                }) = &mut effect.kind
                {
                    *placement = SegPlacement::LaneLocal;
                }
            }
        }
    }
    reify_parallel_scatter(entry);
    reify_entry_filter_spaces(entry);
}

fn reify_entry_filter_spaces(entry: &mut EgirEntry) {
    let locations: Vec<(BlockId, usize)> = entry
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(block_id, block)| {
            block.side_effects.iter().enumerate().filter_map(move |(index, effect)| {
                matches!(effect.kind, SideEffectKind::Soac(EgirSoac::Filter { .. }))
                    .then_some((block_id, index))
            })
        })
        .collect();
    for (block_id, index) in locations {
        let effect = entry.graph.skeleton.blocks[block_id].side_effects[index].clone();
        let SideEffectKind::Soac(EgirSoac::Filter {
            input_array_type,
            input_elem_type,
            ..
        }) = &effect.kind
        else {
            continue;
        };
        let new_space = semantic_space(
            entry,
            &effect,
            std::slice::from_ref(input_array_type),
            std::slice::from_ref(input_elem_type),
        );
        if let SideEffectKind::Soac(EgirSoac::Filter { space, .. }) =
            &mut entry.graph.skeleton.blocks[block_id].side_effects[index].kind
        {
            *space = Some(new_space);
        }
    }
}

fn reify_function_filter_spaces(function: &mut EgirFunc) {
    let locations: Vec<(BlockId, usize)> = function
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(block_id, block)| {
            block.side_effects.iter().enumerate().filter_map(move |(index, effect)| {
                matches!(effect.kind, SideEffectKind::Soac(EgirSoac::Filter { .. }))
                    .then_some((block_id, index))
            })
        })
        .collect();
    for (block_id, index) in locations {
        let effect = function.graph.skeleton.blocks[block_id].side_effects[index].clone();
        let SideEffectKind::Soac(EgirSoac::Filter {
            input_array_type,
            input_elem_type,
            ..
        }) = &effect.kind
        else {
            continue;
        };
        let new_space = semantic_space_for_graph(
            &function.graph,
            &effect,
            std::slice::from_ref(input_array_type),
            std::slice::from_ref(input_elem_type),
        );
        if let SideEffectKind::Soac(EgirSoac::Filter { space, .. }) =
            &mut function.graph.skeleton.blocks[block_id].side_effects[index].kind
        {
            *space = Some(new_space);
        }
    }
}

/// Every node read as input by a semantic SOAC in the entry — the union of all
/// SOAC operand subgraphs. A Screma whose result lands here is an internal
/// producer feeding a downstream scatter/filter/map; reifying it would detach
/// it from that consumer's ordered pipeline, so it stays serial. A Screma whose
/// result is not consumed by any SOAC (it routes to an entry output via the
/// terminator or an output `Store`) is an output root and may reify into its
/// own kernel.
fn soac_consumed_nodes(entry: &EgirEntry) -> std::collections::HashSet<NodeId> {
    let mut set: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    let mut stack: Vec<NodeId> = Vec::new();
    for (_, block) in &entry.graph.skeleton.blocks {
        for se in &block.side_effects {
            if matches!(se.kind, SideEffectKind::Soac(_)) {
                stack.extend(se.referenced_nodes());
            }
        }
    }
    while let Some(n) = stack.pop() {
        if !set.insert(n) {
            continue;
        }
        if let Some(node) = entry.graph.nodes.get(n) {
            stack.extend(node.children());
        }
    }
    set
}

fn reify_one_screma(entry: &mut EgirEntry, block_id: BlockId, idx: usize, placement: SegPlacement) {
    let se = entry.graph.skeleton.blocks[block_id].side_effects[idx].clone();
    let SideEffectKind::Soac(EgirSoac::Screma {
        map_bodies,
        accumulators,
        input_array_types,
        input_elem_types,
        map_output_elem_types,
        map_input_indices,
        map_destinations,
        acc_destinations,
    }) = &se.kind
    else {
        unreachable!()
    };

    let n_inputs = input_array_types.len();
    let neutrals: Vec<NodeId> = (0..accumulators.len()).map(|i| se.operand_nodes[n_inputs + i]).collect();
    let map_bodies = map_bodies.clone();
    let kind = reify_seg_kind(accumulators, &neutrals, n_inputs);
    let placement = if matches!(kind, SegOpKind::SegMap)
        && placement == SegPlacement::Kernel
        && (map_bodies.is_empty()
            || !map_destinations
                .iter()
                .all(|dest| matches!(dest, SoacDestination::OutputView | SoacDestination::InputBuffer)))
    {
        SegPlacement::LaneLocal
    } else {
        placement
    };

    let result_types = se
        .result
        .and_then(|result| entry.graph.types.get(&result))
        .map(|ty| match ty {
            Type::Constructed(TypeName::Tuple(_), fields) => fields.clone(),
            other => vec![other.clone()],
        })
        .unwrap_or_default();
    let output_slots = side_effect_output_slots(entry, &se);
    let placement = if placement == SegPlacement::Kernel && output_slots.is_empty() {
        SegPlacement::LaneLocal
    } else {
        placement
    };
    let resources = semantic_resources(entry, &se, &output_slots);
    let seg = EgirSoac::Seg {
        space: semantic_space(entry, &se, input_array_types, input_elem_types),
        placement,
        kind,
        map_bodies,
        input_array_types: input_array_types.clone(),
        input_elem_types: input_elem_types.clone(),
        map_output_elem_types: map_output_elem_types.clone(),
        map_input_indices: map_input_indices.clone(),
        map_destinations: map_destinations.clone(),
        acc_destinations: acc_destinations.clone(),
        result_types,
        output_slots,
        resources,
        scratch_resources: Vec::new(),
    };
    entry.graph.skeleton.blocks[block_id].side_effects[idx].kind = SideEffectKind::Soac(seg);
}

/// Classify a Screma's accumulator list into the matching `SegOpKind`. The
/// step/combine regions and their captures are already explicit on each
/// `ScremaOperator`, so the operators are carried across verbatim.
fn reify_seg_kind(accumulators: &[ScremaOperator], neutrals: &[NodeId], input_count: usize) -> SegOpKind {
    debug_assert_eq!(accumulators.len(), neutrals.len());
    let operators: Vec<SegBinOp> = accumulators
        .iter()
        .zip(neutrals)
        .map(|(acc, &neutral)| SegBinOp {
            kind: acc.kind,
            step: acc.step.clone(),
            combine: acc.combine.clone(),
            input_indices: if acc.input_indices.is_empty() {
                (0..input_count).collect()
            } else {
                acc.input_indices.clone()
            },
            neutral,
            shape: Vec::new(),
            // Wyn's source reduction contract is associative but currently has
            // no commutativity annotation, so preserve left-to-right order.
            commutative: false,
        })
        .collect();
    if operators.is_empty() {
        SegOpKind::SegMap
    } else if operators.iter().all(|op| matches!(op.kind, crate::tlc::ScremaAccumulator::Reduce)) {
        SegOpKind::SegRed { operators }
    } else if operators.iter().all(|op| matches!(op.kind, crate::tlc::ScremaAccumulator::Scan)) {
        SegOpKind::SegScan { operators }
    } else {
        SegOpKind::SegComposite { operators }
    }
}

fn semantic_space(
    entry: &EgirEntry,
    se: &SideEffect,
    input_array_types: &[Type<TypeName>],
    input_elem_types: &[Type<TypeName>],
) -> SegSpace {
    let primary = se.operand_nodes.first().copied();
    let extent = primary.map(|node| {
        if let Some(binding) = graph_ops::extract_storage_view_source(&entry.graph, node) {
            let elem_bytes =
                input_elem_types.first().and_then(crate::ssa::layout::type_byte_size).unwrap_or(1) as u32;
            return SegExtent::ResourceLength {
                node,
                binding,
                elem_bytes,
            };
        }
        if let Some((_, len, _)) = graph_ops::extract_array_range_operands(&entry.graph, node) {
            return extent_from_node(entry, len);
        }
        if let Some(Type::Constructed(TypeName::Size(n), _)) =
            input_array_types.first().and_then(Type::array_size)
        {
            return SegExtent::Fixed(*n as u32);
        }
        SegExtent::Value(node)
    });
    SegSpace {
        level: SegLevel::Thread,
        dims: extent.into_iter().collect(),
    }
}

fn semantic_space_for_graph(
    graph: &EGraph,
    se: &SideEffect,
    input_array_types: &[Type<TypeName>],
    input_elem_types: &[Type<TypeName>],
) -> SegSpace {
    let extent = se.operand_nodes.first().copied().map(|node| {
        if let Some(binding) = graph_ops::extract_storage_view_source(graph, node) {
            let elem_bytes =
                input_elem_types.first().and_then(crate::ssa::layout::type_byte_size).unwrap_or(1) as u32;
            return SegExtent::ResourceLength {
                node,
                binding,
                elem_bytes,
            };
        }
        if let Some((_, len, _)) = graph_ops::extract_array_range_operands(graph, node) {
            return match &graph.nodes[len] {
                ENode::Pure {
                    op: PureOp::Int(value) | PureOp::Uint(value),
                    ..
                } => value.parse().map(SegExtent::Fixed).unwrap_or(SegExtent::Value(len)),
                _ => SegExtent::Value(len),
            };
        }
        if let Some(Type::Constructed(TypeName::Size(n), _)) =
            input_array_types.first().and_then(TypeExt::array_size)
        {
            return SegExtent::Fixed(*n as u32);
        }
        SegExtent::Value(node)
    });
    SegSpace {
        level: SegLevel::Thread,
        dims: extent.into_iter().collect(),
    }
}

fn semantic_read_resources(graph: &EGraph, se: &SideEffect) -> Vec<SegResourceAccess> {
    let mut bindings = std::collections::HashSet::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack = se.referenced_nodes().collect::<Vec<_>>();
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        if let Some(binding) = graph_ops::extract_storage_view_source(graph, node) {
            bindings.insert(binding);
        }
        stack.extend(graph.nodes[node].children());
    }
    let mut result: Vec<_> = bindings
        .into_iter()
        .map(|binding| SegResourceAccess {
            binding,
            access: SegResourceAccessKind::Read,
        })
        .collect();
    result.sort_by_key(|resource| (resource.binding.set, resource.binding.binding));
    result
}

fn extent_from_node(entry: &EgirEntry, node: NodeId) -> SegExtent {
    match &entry.graph.nodes[node] {
        ENode::Pure {
            op: PureOp::Int(value) | PureOp::Uint(value),
            ..
        } => value.parse().map(SegExtent::Fixed).unwrap_or(SegExtent::Value(node)),
        ENode::FuncParam { index } => entry
            .inputs
            .get(*index)
            .and_then(|input| input.push_constant)
            .map(|slot| SegExtent::PushConstant {
                node,
                offset: slot.offset,
            })
            .unwrap_or(SegExtent::Value(node)),
        _ => SegExtent::Value(node),
    }
}

fn semantic_resources(
    entry: &EgirEntry,
    se: &SideEffect,
    output_slots: &[usize],
) -> Vec<SegResourceAccess> {
    use std::collections::HashMap;
    let mut accesses: HashMap<BindingRef, SegResourceAccessKind> = HashMap::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack = se.referenced_nodes().collect::<Vec<_>>();
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        if let Some(binding) = graph_ops::extract_storage_view_source(&entry.graph, node) {
            accesses.entry(binding).or_insert(SegResourceAccessKind::Read);
        }
        stack.extend(entry.graph.nodes[node].children());
    }
    for &slot in output_slots {
        if let Some(binding) = entry.outputs.get(slot).and_then(|output| output.storage_binding) {
            accesses
                .entry(binding)
                .and_modify(|access| *access = SegResourceAccessKind::ReadWrite)
                .or_insert(SegResourceAccessKind::Write);
        }
    }
    let mut result: Vec<_> =
        accesses.into_iter().map(|(binding, access)| SegResourceAccess { binding, access }).collect();
    result.sort_by_key(|resource| (resource.binding.set, resource.binding.binding));
    result
}

fn reify_parallel_scatter(entry: &mut EgirEntry) {
    let location = entry.graph.skeleton.blocks.iter().find_map(|(block_id, block)| {
        block
            .side_effects
            .iter()
            .position(|effect| matches!(effect.kind, SideEffectKind::Soac(EgirSoac::Hist { .. })))
            .map(|index| (block_id, index))
    });
    let Some((block_id, index)) = location else {
        return;
    };
    let mut effect = entry.graph.skeleton.blocks[block_id].side_effects[index].clone();
    let SideEffectKind::Soac(EgirSoac::Hist {
        input_array_types,
        input_elem_types,
        ..
    }) = &effect.kind
    else {
        return;
    };
    effect.operand_nodes.remove(0);
    let space = semantic_space(entry, &effect, input_array_types, input_elem_types);
    if let SideEffectKind::Soac(EgirSoac::Hist { space: target, .. }) =
        &mut entry.graph.skeleton.blocks[block_id].side_effects[index].kind
    {
        *target = Some(space);
    }
}

#[derive(Clone, Copy)]
enum ChunkInputKind {
    StorageOnly,
    StorageOrRange,
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
        return Ok(emit_storage_len(graph, br));
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
        return Ok(graph_ops::intern_chunked_storage_view(
            graph,
            br,
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
/// `ComputeStage` in `tlc::parallelize` must dispatch this same width.

#[allow(clippy::too_many_arguments)]
fn build_tree_reduce_phase2(
    b: &mut super::builder::EntryBuilder,
    op_func: String,
    elem_ty: Type<TypeName>,
    init_nid: NodeId,
    partials_binding: BindingRef,
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
            // region stamped from the binding by intern_storage_view.
            crate::types::no_region(),
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
    let partials_view = graph_ops::intern_storage_view(graph, partials_binding, view_arr_ty.clone(), None);
    let len = emit_storage_len(graph, partials_binding);
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

/// Emit a `_w_intrinsic_storage_len(set, binding)` node returning u32.
fn emit_storage_len(graph: &mut super::types::EGraph, br: BindingRef) -> NodeId {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let set_nid = graph_ops::intern_u32(graph, br.set, None);
    let binding_nid = graph_ops::intern_u32(graph, br.binding, None);
    graph_ops::intern_intrinsic(
        graph,
        catalog().known().storage_len,
        smallvec![set_nid, binding_nid],
        u32_ty,
        None,
    )
}

fn dispatch_worker_buffer_len(elem_ty: &Type<TypeName>) -> Option<crate::pipeline_descriptor::BufferLen> {
    crate::ssa::layout::type_byte_size(elem_ty).map(|bytes| {
        crate::pipeline_descriptor::BufferLen::SameAsDispatch {
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
    partials_binding: BindingRef,
    accumulator_value: NodeId,
    output_stores: &[(NodeId, NodeId)],
    output_decls: &[(
        BindingRef,
        Type<TypeName>,
        Option<crate::pipeline_descriptor::BufferLen>,
    )],
) -> Result<EgirEntry, String> {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(full_name, (PHASE2_WIDTH, 1, 1));
    b.declare_intermediate_storage_sized(
        partials_binding,
        elem_ty.clone(),
        dispatch_worker_buffer_len(&elem_ty),
    );
    for (binding, ty, length) in output_decls {
        b.declare_output_storage_sized(*binding, ty.clone(), length.clone());
    }

    let init_nid = graph_ops::clone_pure_subgraph(phase1_graph, b.graph_mut(), phase1_ne_nid)?;
    build_tree_reduce_phase2(
        &mut b,
        op_func,
        elem_ty,
        init_nid,
        partials_binding,
        phase1_graph,
        accumulator_value,
        output_stores,
    )?;
    Ok(b.build())
}

fn find_pending_seg(entry: &EgirEntry) -> Option<(BlockId, usize)> {
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            if matches!(&se.kind, SideEffectKind::Soac(EgirSoac::Seg { .. })) {
                return Some((bid, i));
            }
        }
    }
    None
}

fn find_pending_kernel_seg(entry: &EgirEntry) -> Option<(BlockId, usize)> {
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            if matches!(
                &se.kind,
                SideEffectKind::Soac(EgirSoac::Seg {
                    placement: SegPlacement::Kernel,
                    ..
                })
            ) {
                return Some((bid, i));
            }
        }
    }
    None
}

fn restore_serial_seg(entry: &mut EgirEntry) {
    let Some((block_id, idx)) = find_pending_kernel_seg(entry) else {
        return;
    };
    restore_serial_seg_at(entry, block_id, idx);
}

fn restore_serial_seg_at(entry: &mut EgirEntry, block_id: BlockId, idx: usize) {
    restore_serial_seg_in_graph(&mut entry.graph, block_id, idx);
}

fn restore_serial_seg_in_graph(graph: &mut EGraph, block_id: BlockId, idx: usize) {
    let kind = graph.skeleton.blocks[block_id].side_effects[idx].kind.clone();
    let SideEffectKind::Soac(EgirSoac::Seg {
        kind,
        map_bodies,
        input_array_types,
        input_elem_types,
        map_output_elem_types,
        map_input_indices,
        map_destinations,
        acc_destinations,
        ..
    }) = kind
    else {
        unreachable!()
    };
    let operators = match kind {
        SegOpKind::SegMap => Vec::new(),
        SegOpKind::SegRed { operators }
        | SegOpKind::SegScan { operators }
        | SegOpKind::SegComposite { operators } => operators,
    };
    let accumulators = operators
        .into_iter()
        .map(|op| ScremaOperator {
            kind: op.kind,
            step: op.step,
            combine: op.combine,
            input_indices: op.input_indices,
        })
        .collect();
    graph.skeleton.blocks[block_id].side_effects[idx].kind = SideEffectKind::Soac(EgirSoac::Screma {
        map_bodies,
        accumulators,
        input_array_types,
        input_elem_types,
        map_output_elem_types,
        map_input_indices,
        map_destinations,
        acc_destinations,
    });
}

/// Convert every semantic SegOp back to its local serial SOAC form. This is
/// used only by the terminal single-stage lowering policy; segmentation itself
/// is target-independent and therefore still happens in single-stage builds.
pub fn restore_all_serial(inner: &mut EgirInner) {
    for entry in &mut inner.entry_points {
        while let Some((block_id, idx)) = find_pending_seg(entry) {
            restore_serial_seg_at(entry, block_id, idx);
        }
    }
    for function in &mut inner.functions {
        loop {
            let location = function.graph.skeleton.blocks.iter().find_map(|(block_id, block)| {
                block.side_effects.iter().enumerate().find_map(|(index, effect)| {
                    matches!(effect.kind, SideEffectKind::Soac(EgirSoac::Seg { .. }))
                        .then_some((block_id, index))
                })
            });
            let Some((block_id, index)) = location else {
                break;
            };
            restore_serial_seg_in_graph(&mut function.graph, block_id, index);
        }
    }
}

fn restore_lane_local(inner: &mut EgirInner) {
    for entry in &mut inner.entry_points {
        loop {
            let location = entry.graph.skeleton.blocks.iter().find_map(|(block_id, block)| {
                block.side_effects.iter().enumerate().find_map(|(index, effect)| {
                    matches!(
                        effect.kind,
                        SideEffectKind::Soac(EgirSoac::Seg {
                            placement: SegPlacement::LaneLocal,
                            ..
                        })
                    )
                    .then_some((block_id, index))
                })
            });
            let Some((block_id, index)) = location else {
                break;
            };
            restore_serial_seg_in_graph(&mut entry.graph, block_id, index);
        }
    }
    for function in &mut inner.functions {
        loop {
            let location = function.graph.skeleton.blocks.iter().find_map(|(block_id, block)| {
                block.side_effects.iter().enumerate().find_map(|(index, effect)| {
                    matches!(
                        effect.kind,
                        SideEffectKind::Soac(EgirSoac::Seg {
                            placement: SegPlacement::LaneLocal,
                            ..
                        })
                    )
                    .then_some((block_id, index))
                })
            });
            let Some((block_id, index)) = location else {
                break;
            };
            restore_serial_seg_in_graph(&mut function.graph, block_id, index);
        }
    }
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

fn storage_binding_under(graph: &super::types::EGraph, root: NodeId) -> Option<BindingRef> {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        if let Some(binding) = graph_ops::extract_storage_view_source(graph, node) {
            return Some(binding);
        }
        stack.extend(graph.nodes[node].children());
    }
    None
}

/// The scratch a Seg op owns if it will parallel-lower — reduce partials or
/// scan block scratch — each paired with its accumulator element type and kind.
/// `None` for `SegMap`, lane-local ops, or shapes that fall back to serial
/// lowering. Kept in lockstep with the gates in `lower_reduce_entry` /
/// `lower_scan_entry`; a mismatch would orphan or miss a scratch binding.
fn seg_scratch_specs(
    graph: &EGraph,
    se: &SideEffect,
) -> Option<Vec<(Type<TypeName>, CompilerResourceKind)>> {
    let SideEffectKind::Soac(EgirSoac::Seg {
        placement: SegPlacement::Kernel,
        kind,
        input_array_types,
        map_destinations,
        acc_destinations,
        ..
    }) = &se.kind
    else {
        return None;
    };
    let elem_of = |neutral: NodeId| graph.types.get(&neutral).cloned();
    let maps_are_output_views = map_destinations.iter().all(|d| matches!(d, SoacDestination::OutputView));
    match kind {
        SegOpKind::SegRed { operators } => {
            if operators.iter().any(|op| !op.combine.captures.is_empty())
                || input_array_types.is_empty()
                || !maps_are_output_views
                || !acc_destinations.iter().all(|d| matches!(d, SoacDestination::Fresh))
            {
                return None;
            }
            operators
                .iter()
                .map(|op| Some((elem_of(op.neutral)?, CompilerResourceKind::ReducePartial)))
                .collect()
        }
        SegOpKind::SegScan { operators } => {
            if operators.len() != 1
                || !operators[0].combine.captures.is_empty()
                || input_array_types.len() != 1
                || !maps_are_output_views
                || !acc_destinations.iter().all(|d| matches!(d, SoacDestination::OutputView))
            {
                return None;
            }
            let elem = elem_of(operators[0].neutral)?;
            Some(vec![
                (elem.clone(), CompilerResourceKind::ScanBlockSums),
                (elem, CompilerResourceKind::ScanBlockOffsets),
            ])
        }
        SegOpKind::SegMap | SegOpKind::SegComposite { .. } => None,
    }
}

/// Reserve a logical resource for every parallel SegRed/SegScan's scratch,
/// drawing fresh bindings and recording each `ResourceId` on the owning Seg op.
/// `first_id` is the next free `ResourceId` after the host resources. Terminal
/// lowering later resolves these to the same bindings via `EgirInner::binding_of`.
pub fn enumerate_seg_scratch(
    inner: &mut EgirInner,
    binding_ids: &mut crate::IdSource<u32>,
    first_id: u32,
) -> Vec<LogicalResource> {
    let mut resources = Vec::new();
    let mut next = first_id;
    let mut used_bindings: std::collections::HashSet<BindingRef> = inner
        .entry_points
        .iter()
        .flat_map(|entry| {
            entry
                .inputs
                .iter()
                .filter_map(|input| input.storage_binding)
                .chain(entry.outputs.iter().filter_map(|output| output.storage_binding))
                .chain(entry.storage_bindings.iter().map(|declaration| declaration.binding))
        })
        .collect();
    for entry in inner.entry_points.iter_mut() {
        // Collect scratch needs before drawing (immutable graph read), then
        // apply the assignment (mutable) — avoids aliasing graph.types.
        let mut plans: Vec<(BlockId, usize, Vec<(Type<TypeName>, CompilerResourceKind)>)> = Vec::new();
        for (block_id, block) in &entry.graph.skeleton.blocks {
            for (se_idx, se) in block.side_effects.iter().enumerate() {
                if let Some(specs) = seg_scratch_specs(&entry.graph, se) {
                    plans.push((block_id, se_idx, specs));
                }
            }
        }
        for (block_id, se_idx, specs) in plans {
            let mut ids = Vec::with_capacity(specs.len());
            let mut scratch_bindings = Vec::with_capacity(specs.len());
            let owner = entry.graph.skeleton.blocks[block_id].side_effects[se_idx].result.map(|result| {
                SemanticOpId {
                    scope: entry.name.clone(),
                    result,
                }
            });
            for (slot, (elem_ty, kind)) in specs.into_iter().enumerate() {
                let binding = loop {
                    let candidate =
                        BindingRef::new(crate::egir::from_tlc::AUTO_STORAGE_SET, binding_ids.next_id());
                    if used_bindings.insert(candidate) {
                        break candidate;
                    }
                };
                let size = super::program::logical_size(dispatch_worker_buffer_len(&elem_ty).as_ref());
                let id = ResourceId(next);
                next += 1;
                resources.push(LogicalResource {
                    id,
                    origin: ResourceOrigin::Compiler(CompilerResource::new(kind, owner.clone(), slot)),
                    legacy_binding: binding,
                    elem_ty,
                    size,
                });
                ids.push(id);
                scratch_bindings.push(binding);
            }
            if let SideEffectKind::Soac(EgirSoac::Seg {
                scratch_resources,
                resources,
                ..
            }) = &mut entry.graph.skeleton.blocks[block_id].side_effects[se_idx].kind
            {
                *scratch_resources = ids;
                for binding in scratch_bindings {
                    if !resources.iter().any(|resource| resource.binding == binding) {
                        resources.push(SegResourceAccess {
                            binding,
                            access: SegResourceAccessKind::ReadWrite,
                        });
                    }
                }
                resources.sort_by_key(|resource| (resource.binding.set, resource.binding.binding));
            }
        }
    }
    resources
}

fn lower_reduce_entry(
    entry: &mut EgirEntry,
    regions: &mut RegionInterner,
    resource_bindings: &[BindingRef],
) -> Option<Vec<EgirEntry>> {
    let total_threads = REDUCE_PHASE1_WIDTH;

    // 2. Locate the semantic SegRed and pull its metadata.
    let Some((block_id, idx)) = find_pending_kernel_seg(entry) else {
        return None;
    };
    if seg_scratch_specs(
        &entry.graph,
        &entry.graph.skeleton.blocks[block_id].side_effects[idx],
    )
    .is_none()
    {
        return None;
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
        let SideEffectKind::Soac(EgirSoac::Seg {
            kind: SegOpKind::SegRed { operators },
            map_bodies,
            input_array_types,
            map_destinations,
            acc_destinations,
            ..
        }) = &se.kind
        else {
            return None;
        };
        let n_accs = operators.len();
        // Map and step captures remain in the original phase-1 envelope.
        // Combine captures require an explicit environment on phase 2.
        if operators.iter().any(|operator| !operator.combine.captures.is_empty()) {
            return None;
        }
        let n_inputs = input_array_types.len();
        if n_inputs == 0 {
            return None;
        }
        let n_maps = map_bodies.len();
        // Map outputs that have been retargeted to OutputView so the
        // chunked writes inside the Screma loop body land in the right
        // buffer. Fresh destinations would require building an immutable
        // array per loop iteration — incompatible with chunked parallel
        // writes.
        if !map_destinations.iter().all(|d| matches!(d, SoacDestination::OutputView)) {
            return None;
        }
        // Reduce accumulators expect Fresh destination (scalar result
        // routed via a Project-based Store outside the Screma loop).
        if !acc_destinations.iter().all(|d| matches!(d, SoacDestination::Fresh)) {
            return None;
        }
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
        let base = n_inputs + n_accs;
        let map_view_ops: Vec<usize> = (0..n_maps).map(|m| base + m).collect();
        let result = se.result?;
        let elem_tys: Vec<Type<TypeName>> =
            init_nids.iter().map(|n| entry.graph.types[n].clone()).collect();
        let reduce_funcs = regions.names(operators.iter().map(|a| a.combine.region));
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

    let n_accs = elem_tys.len();
    // Partial buffers were reserved as logical resources at the allocation
    // boundary; resolve the Seg op's `scratch_resources` to their bindings.
    let partial_bindings: Vec<BindingRef> = {
        let SideEffectKind::Soac(EgirSoac::Seg {
            scratch_resources, ..
        }) = &entry.graph.skeleton.blocks[block_id].side_effects[idx].kind
        else {
            return None;
        };
        // The manifest is authoritative: if it did not reserve one partial per
        // accumulator this op is not the parallel shape, so fall back to serial.
        if scratch_resources.len() != n_accs {
            return None;
        }
        scratch_resources.iter().map(|id| resource_bindings[id.0 as usize]).collect()
    };

    // The caller snapshots the entry before invoking this lowering, so any
    // late routing failure restores the complete pre-lowering graph.

    // 3. Chunk all input views and every map output view; swap them back
    // into the Screma operand list.
    let chunked = chunk_soac_inputs(
        &mut entry.graph,
        &input_view_data,
        total_threads,
        ChunkInputKind::StorageOrRange,
        "SegRed",
    )
    .ok()?;
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
        .ok()?;
        let se = &mut entry.graph.skeleton.blocks[block_id].side_effects[idx];
        se.operand_nodes[*op_idx] = chunked_view;
    }

    // 4. Capture every output store of each accumulator. `realize_outputs`
    // decomposes a tuple-element accumulator across one store per scalar field
    // (`Project{field}(…Project{acc_pos}(screma_result))`), so an accumulator may
    // own several stores. Bucket them by `acc_i = project_root_index − n_maps`
    // (the index adjacent to the Screma result), recording each store's
    // `(place, value)` nodes plus the distinct output bindings it targets and
    // their entry-output sizing.
    let mut acc_stores: Vec<Vec<(NodeId, NodeId)>> = vec![Vec::new(); n_accs];
    let mut acc_output_decls: Vec<
        Vec<(
            BindingRef,
            Type<TypeName>,
            Option<crate::pipeline_descriptor::BufferLen>,
        )>,
    > = vec![Vec::new(); n_accs];
    let mut drop_locations: Vec<(BlockId, usize)> = Vec::new();
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            if !matches!(&se.kind, SideEffectKind::Inst(InstKind::Store { .. })) {
                continue;
            }
            let (Some(&place), Some(&value)) = (se.operand_nodes.first(), se.operand_nodes.get(1)) else {
                continue;
            };
            let Some(root_idx) = project_root_index(&entry.graph, value, screma_result_nid) else {
                continue;
            };
            if (root_idx as usize) < n_maps {
                continue; // map output Stores aren't generated for OutputView destinations
            }
            let acc_i = (root_idx as usize) - n_maps;
            if acc_i >= n_accs {
                continue;
            }
            acc_stores[acc_i].push((place, value));
            if let Some(binding) = storage_binding_under(&entry.graph, place) {
                if !acc_output_decls[acc_i].iter().any(|(b, _, _)| *b == binding) {
                    let Some(out) = entry.outputs.iter().find(|o| o.storage_binding == Some(binding))
                    else {
                        return None;
                    };
                    acc_output_decls[acc_i].push((binding, out.ty.clone(), out.length.clone()));
                }
            }
            drop_locations.push((bid, i));
        }
    }
    if (0..n_accs).any(|acc_i| acc_stores[acc_i].is_empty() || acc_output_decls[acc_i].is_empty()) {
        return None;
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
    let mut next_effect = graph_ops::next_effect_token(&entry.graph);
    for acc_i in 0..n_accs {
        let elem_ty = elem_tys[acc_i].clone();
        let arr_ty = Type::Constructed(
            TypeName::Array,
            vec![
                elem_ty.clone(),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
                Type::Variable(0),
                crate::types::no_region(),
            ],
        );
        let partials_view =
            graph_ops::intern_storage_view(&mut entry.graph, partial_bindings[acc_i], arr_ty, None);
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
        for (binding, _, _) in &acc_output_decls[acc_i] {
            for o in entry.outputs.iter_mut() {
                if o.storage_binding == Some(*binding) {
                    o.storage_binding = None;
                }
            }
        }
        entry.storage_bindings.push(crate::interface::StorageBindingDecl {
            binding: partial_bindings[acc_i],
            role: crate::interface::StorageRole::Intermediate,
            elem_ty: elem_tys[acc_i].clone(),
            length: dispatch_worker_buffer_len(&elem_tys[acc_i]),
        });
    }
    // A moved output binding may also carry an Output storage declaration (e.g. a
    // hoisted prepass result). Phase 1 no longer writes it; phase 2 owns it.
    let moved: std::collections::HashSet<BindingRef> =
        acc_output_decls.iter().flatten().map(|(b, _, _)| *b).collect();
    entry.storage_bindings.retain(|declaration| {
        declaration.role != crate::interface::StorageRole::Output || !moved.contains(&declaration.binding)
    });

    // 6. Synthesize one phase 2 entry per accumulator. The phase-1 snapshot still
    // holds the captured store nodes (dropping the side-effects left the pure
    // value/place subgraphs intact) and `accumulator_values`.
    let phase1_snapshot = entry.graph.clone();
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
            &phase1_snapshot,
            init_nids[acc_i],
            partial_bindings[acc_i],
            accumulator_values[acc_i],
            &acc_stores[acc_i],
            &acc_output_decls[acc_i],
        )
        .ok()?;
        phase2s.push(phase2);
    }
    // Scheduling consumed the semantic SegRed. Phase 1 is now an ordinary
    // per-invocation Screma over the thread's chunk; `soac_expand` lowers that
    // local loop while the synthesized phase-2 entries combine its partials.
    restore_serial_seg(entry);
    Some(phase2s)
}

/// Lower a `SegScan` into a three-phase block scan. Phase 1 (the original
/// entry) scans each thread's chunk into the output buffer and reduces the
/// chunk to `block_sums[tid]`; phase 2 sequentially scans `block_sums` into
/// `block_offsets`; phase 3 adds each chunk's offset back over the output
/// buffer through an arg-swapped wrapper around the combiner (so a
/// non-commutative op keeps the running prefix in its `acc` slot). Returns the
/// phase-2 and phase-3 entries plus the swap-wrapper function, or `None` for an
/// unsupported shape (captures, multiple inputs/accumulators).
fn lower_scan_entry(
    entry: &mut EgirEntry,
    regions: &mut RegionInterner,
    resource_bindings: &[BindingRef],
) -> Option<(Vec<EgirEntry>, EgirFunc)> {
    let total_threads = REDUCE_PHASE1_WIDTH;

    let (block_id, idx) = find_pending_kernel_seg(entry)?;
    seg_scratch_specs(
        &entry.graph,
        &entry.graph.skeleton.blocks[block_id].side_effects[idx],
    )?;
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
        let SideEffectKind::Soac(EgirSoac::Seg {
            kind: SegOpKind::SegScan { operators },
            map_bodies,
            input_array_types,
            input_elem_types,
            map_destinations,
            acc_destinations,
            ..
        }) = &se.kind
        else {
            return None;
        };
        if operators.len() != 1 {
            return None;
        }
        let op = &operators[0];
        if !op.combine.captures.is_empty() {
            return None;
        }
        let n_inputs = input_array_types.len();
        if n_inputs != 1 {
            return None;
        }
        if !map_destinations.iter().all(|d| matches!(d, SoacDestination::OutputView)) {
            return None;
        }
        // `realize_outputs` retargets the scan accumulator to OutputView (its
        // prefixes feed the entry output) and appends the scan output buffer.
        if !acc_destinations.iter().all(|d| matches!(d, SoacDestination::OutputView)) {
            return None;
        }
        let n_accs = operators.len();
        let n_maps = map_bodies.len();
        let input_nid = se.operand_nodes[0];
        let init_nid = op.neutral;
        let step_capture_nodes = op.step.captures.clone();
        // `[inputs.., init_accs.., map_output_views.., scan_output_view]`:
        // output views follow the init accumulators (captures live on SegBodies).
        let output_view_base = n_inputs + n_accs;
        let scan_output_view_op = output_view_base + n_maps;
        let map_output_view_ops: Vec<usize> = (0..n_maps).map(|map| output_view_base + map).collect();
        let input_ty = entry.graph.types[&input_nid].clone();
        let input_elem = input_elem_types[0].clone();
        let elem = entry.graph.types[&init_nid].clone();
        (
            regions.name(op.step.region).to_string(),
            regions.name(op.combine.region).to_string(),
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

    // Block scratch was reserved as `[ScanBlockSums, ScanBlockOffsets]` logical
    // resources at the allocation boundary; resolve them to their bindings.
    let (block_sums_binding, block_offsets_binding) = {
        let SideEffectKind::Soac(EgirSoac::Seg {
            scratch_resources, ..
        }) = &entry.graph.skeleton.blocks[block_id].side_effects[idx].kind
        else {
            return None;
        };
        // The manifest is authoritative (see the reduce path); a scan that did
        // not reserve both block buffers is not the parallel shape.
        if scratch_resources.len() != 2 {
            return None;
        }
        (
            resource_bindings[scratch_resources[0].0 as usize],
            resource_bindings[scratch_resources[1].0 as usize],
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
    .ok()?;
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
        .ok()?;
        entry.graph.skeleton.blocks[block_id].side_effects[idx].operand_nodes[*operand_index] =
            chunked_view;
    }
    let (scan_output_storage, orig_scan_output_view_ty) = {
        let se = &entry.graph.skeleton.blocks[block_id].side_effects[idx];
        let v = se.operand_nodes[scan_output_view_op];
        let ty = entry.graph.types[&v].clone();
        let storage = graph_ops::extract_storage_view_source(&entry.graph, v)?;
        (storage, ty)
    };
    let chunked_scan_output = graph_ops::intern_chunked_storage_view(
        &mut entry.graph,
        scan_output_storage,
        chunk_start,
        chunk_len,
        orig_scan_output_view_ty,
        None,
    );
    {
        let se = &mut entry.graph.skeleton.blocks[block_id].side_effects[idx];
        se.operand_nodes[scan_output_view_op] = chunked_scan_output;
    }

    // Snapshot phase 1 before appending the chunked reduce so phase 2's
    // neutral-element clone sees a clean graph.
    let phase1_snapshot = entry.graph.clone();

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
            EgirSoac::Screma {
                map_bodies: vec![],
                accumulators: vec![ScremaOperator {
                    kind: crate::tlc::ScremaAccumulator::Reduce,
                    step: SegBody {
                        region: regions.intern(&op_func),
                        captures: step_capture_nodes,
                    },
                    combine: SegBody {
                        region: regions.intern(&op_func),
                        captures: vec![],
                    },
                    input_indices: vec![0],
                }],
                input_array_types: vec![input_view_ty],
                input_elem_types: vec![input_elem_ty],
                map_output_elem_types: vec![],
                map_input_indices: vec![],
                map_destinations: vec![],
                acc_destinations: vec![SoacDestination::Fresh],
            },
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
                crate::types::no_region(),
            ],
        );
        let block_sums_view =
            graph_ops::intern_storage_view(&mut entry.graph, block_sums_binding, arr_ty, None);
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
    for binding in [block_sums_binding, block_offsets_binding] {
        entry.storage_bindings.push(crate::interface::StorageBindingDecl {
            binding,
            role: crate::interface::StorageRole::Intermediate,
            elem_ty: elem_ty.clone(),
            length: dispatch_worker_buffer_len(&elem_ty),
        });
    }

    let phase2 = synthesize_phase2_scan(
        &entry.name,
        reduce_func.clone(),
        elem_ty.clone(),
        &phase1_snapshot,
        init_nid,
        block_sums_binding,
        block_offsets_binding,
    )
    .ok()?;
    let swap_wrapper_name = format!("{}_scan_op_swap", entry.name);
    let swap_wrapper = synthesize_swap_wrapper(
        swap_wrapper_name.clone(),
        reduce_func,
        elem_ty.clone(),
        entry.span,
    );
    // Intern the synthesized wrapper so `soac_expand` recovers its `Call` name;
    // the caller also publishes its complete body into the region arena.
    let swap_region = regions.intern(&swap_wrapper_name);
    let phase3 = synthesize_phase3_scan(
        &entry.name,
        swap_region,
        elem_ty,
        scan_output_storage,
        block_offsets_binding,
        total_threads,
    );

    // Phase 1 is now a per-invocation Screma scan over the thread's chunk plus
    // the appended block-sum reduce; `soac_expand` lowers both.
    restore_serial_seg(entry);
    Some((vec![phase2, phase3], swap_wrapper))
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
    block_sums_binding: BindingRef,
    block_offsets_binding: BindingRef,
) -> Result<EgirEntry, String> {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(format!("{}_phase2_scan_sums", entry_name), (1, 1, 1));
    let scratch_len = dispatch_worker_buffer_len(&elem_ty);
    b.declare_intermediate_storage_sized(block_sums_binding, elem_ty.clone(), scratch_len.clone());
    b.declare_intermediate_storage_sized(block_offsets_binding, elem_ty.clone(), scratch_len);

    let init_nid = graph_ops::clone_pure_subgraph(phase1_graph, b.graph_mut(), phase1_ne_nid)?;
    build_exclusive_scan_phase2(
        &mut b,
        op_func,
        elem_ty,
        init_nid,
        block_sums_binding,
        block_offsets_binding,
    );
    Ok(b.build())
}

fn build_exclusive_scan_phase2(
    b: &mut super::builder::EntryBuilder,
    op_func: String,
    elem_ty: Type<TypeName>,
    init_nid: NodeId,
    block_sums_binding: BindingRef,
    block_offsets_binding: BindingRef,
) {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(0),
            crate::types::no_region(),
        ],
    );
    let entry_block = b.graph_mut().skeleton.entry;
    let graph = b.graph_mut();
    let sums = graph_ops::intern_storage_view(graph, block_sums_binding, arr_ty.clone(), None);
    let offsets = graph_ops::intern_storage_view(graph, block_offsets_binding, arr_ty, None);
    let len = emit_storage_len(graph, block_sums_binding);
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
        else_args: vec![],
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
    let next_acc = graph.intern_pure(PureOp::Call(op_func), smallvec![acc, value], elem_ty);
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
    b.set_current_block(after);
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
    output_binding: BindingRef,
    block_offsets_binding: BindingRef,
    total_threads: u32,
) -> EgirEntry {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(
        format!("{}_phase3_add_offsets", entry_name),
        (total_threads, 1, 1),
    );
    b.declare_output_storage(output_binding, elem_ty.clone());
    b.declare_intermediate_storage_sized(
        block_offsets_binding,
        elem_ty.clone(),
        dispatch_worker_buffer_len(&elem_ty),
    );

    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(0),
            crate::types::no_region(),
        ],
    );
    let _output_view = b.emit_storage_view(output_binding, arr_ty.clone());
    let block_offsets_view = b.emit_storage_view(block_offsets_binding, arr_ty.clone());

    let output_len = {
        let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let set_nid = graph_ops::intern_u32(b.graph_mut(), output_binding.set, None);
        let binding_nid = graph_ops::intern_u32(b.graph_mut(), output_binding.binding, None);
        graph_ops::intern_intrinsic(
            b.graph_mut(),
            catalog().known().storage_len,
            smallvec![set_nid, binding_nid],
            u32_ty,
            None,
        )
    };
    let (tid, chunk_start, chunk_len) = emit_chunk_arithmetic(b.graph_mut(), total_threads, output_len)
        .expect("phase3: chunk arithmetic must succeed (u32.min is in the prelude)");

    let off_place = b.graph_mut().intern_pure(
        super::types::PureOp::ViewIndex,
        smallvec![block_offsets_view, tid],
        elem_ty.clone(),
    );
    let off = b.emit_load(off_place, elem_ty.clone());

    let chunked_output = graph_ops::intern_chunked_storage_view(
        b.graph_mut(),
        output_binding,
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

/// Build a two-argument helper function named `wrapper_name` whose body is
/// `inner(b, a)` — an arg-swapped wrapper around a `T -> T -> T` combiner.
fn synthesize_swap_wrapper(
    wrapper_name: String,
    inner: String,
    elem_ty: Type<TypeName>,
    span: crate::ast::Span,
) -> EgirFunc {
    let mut graph = EGraph::new();
    let a_nid = graph.add_func_param(0, elem_ty.clone());
    let b_nid = graph.add_func_param(1, elem_ty.clone());
    let call_nid = graph.intern_pure(PureOp::Call(inner), smallvec![b_nid, a_nid], elem_ty.clone());
    let entry_block = graph.skeleton.entry;
    graph.skeleton.blocks[entry_block].term = SkeletonTerminator::Return(Some(call_nid));
    EgirFunc::new(
        wrapper_name,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tlc::ScremaAccumulator;

    /// Region indices used by the operator fixtures below: step is region 0,
    /// combine is region 1. Opaque indices stand in for the named regions a
    /// real conversion would intern.
    const STEP_REGION: RegionId = RegionId::from_index(0);
    const COMBINE_REGION: RegionId = RegionId::from_index(1);

    fn accumulator(
        kind: ScremaAccumulator,
        step_captures: Vec<NodeId>,
        combine_captures: Vec<NodeId>,
    ) -> ScremaOperator {
        ScremaOperator {
            kind,
            step: SegBody {
                region: STEP_REGION,
                captures: step_captures,
            },
            combine: SegBody {
                region: COMBINE_REGION,
                captures: combine_captures,
            },
            input_indices: vec![],
        }
    }

    fn neutral(graph: &mut EGraph, index: usize) -> NodeId {
        graph.add_func_param(index, Type::Constructed(TypeName::Unit, vec![]))
    }

    #[test]
    fn reduction_accumulator_reifies_as_seg_red_operator() {
        let mut graph = EGraph::new();
        let ne = neutral(&mut graph, 0);
        let kind = reify_seg_kind(
            &[accumulator(ScremaAccumulator::Reduce, vec![ne], vec![ne, ne])],
            &[ne],
            1,
        );
        let SegOpKind::SegRed { operators } = kind else {
            panic!("reduction must reify as SegRed")
        };
        assert_eq!(operators.len(), 1);
        assert_eq!(operators[0].step.region, STEP_REGION);
        assert_eq!(operators[0].combine.region, COMBINE_REGION);
        assert_eq!(operators[0].neutral, ne);
        assert!(operators[0].shape.is_empty());
        assert_eq!(operators[0].step.captures, vec![ne]);
        assert_eq!(operators[0].combine.captures, vec![ne, ne]);
        assert!(
            !operators[0].commutative,
            "Wyn does not yet declare commutativity"
        );
    }

    #[test]
    fn scan_accumulator_reifies_as_seg_scan_operator() {
        let mut graph = EGraph::new();
        let ne = neutral(&mut graph, 0);
        let kind = reify_seg_kind(&[accumulator(ScremaAccumulator::Scan, vec![], vec![])], &[ne], 1);
        assert!(matches!(kind, SegOpKind::SegScan { operators } if operators.len() == 1));
    }

    #[test]
    fn mixed_reduce_and_scan_stays_serial_until_joint_scheduler_exists() {
        let accumulators = [
            accumulator(ScremaAccumulator::Reduce, vec![], vec![]),
            accumulator(ScremaAccumulator::Scan, vec![], vec![]),
        ];
        let mut graph = EGraph::new();
        let neutrals = [neutral(&mut graph, 0), neutral(&mut graph, 1)];
        assert!(matches!(
            reify_seg_kind(&accumulators, &neutrals, 1),
            SegOpKind::SegComposite { operators } if operators.len() == 2
        ));
    }

    #[test]
    fn idle_chunk_start_is_clamped_before_remaining_subtraction() {
        let mut graph = EGraph::new();
        let len = graph.add_func_param(0, Type::Constructed(TypeName::UInt(32), vec![]));
        let (_, start, _) =
            emit_chunk_arithmetic(&mut graph, REDUCE_PHASE1_WIDTH, len).expect("u32 chunk arithmetic");
        assert!(matches!(
            &graph.nodes[start],
            super::super::types::ENode::Pure {
                op: PureOp::Intrinsic { .. },
                operands,
            } if operands.as_slice().last() == Some(&len)
        ));
    }

    #[test]
    fn scan_phase2_writes_exclusive_prefix_before_combining_current_block() {
        let elem_ty = Type::Constructed(TypeName::Int(32), vec![]);
        let mut phase1 = EGraph::new();
        let neutral = phase1.intern_pure(PureOp::Int("0".into()), smallvec![], elem_ty.clone());
        let sums = BindingRef::new(0, 40);
        let offsets = BindingRef::new(0, 41);
        let phase2 = synthesize_phase2_scan(
            "prefix",
            "combine".into(),
            elem_ty,
            &phase1,
            neutral,
            sums,
            offsets,
        )
        .expect("phase2 synthesis");

        let stored_value = phase2
            .graph
            .skeleton
            .blocks
            .iter()
            .flat_map(|(_, block)| &block.side_effects)
            .find_map(|effect| {
                if !matches!(effect.kind, SideEffectKind::Inst(InstKind::Store { .. })) {
                    return None;
                }
                let place = *effect.operand_nodes.first()?;
                (storage_binding_under(&phase2.graph, place) == Some(offsets))
                    .then(|| effect.operand_nodes[1])
            })
            .expect("block-offset store");
        assert!(matches!(
            phase2.graph.nodes[stored_value],
            super::super::types::ENode::BlockParam { .. }
        ));
    }
}
