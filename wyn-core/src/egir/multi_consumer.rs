//! Shared materialization of semantic array producers with multiple consumers.
//!
//! A pure `SegMap` with two or more value consumers cannot be vertically
//! fused without duplicating work.  This pass turns it into one compiler-owned
//! storage-producing prepass and rewires every consumer to the shared view.
//! The logical resource is allocated here; target lowering merely schedules
//! the already-explicit producer before its consumers.

use std::collections::{HashMap, HashSet};

use polytype::Type;

use super::graph_ops;
use super::program::{
    buffer_len, CompilerResource, CompilerResourceKind, EgirInner, LogicalSize, SemanticDependencyKind,
    SemanticOpId,
};
use super::types::{
    EGraph, EgirSoac, NodeId, PureOp, SegExtent, SegOpKind, SegPlacement, SegResourceAccess,
    SegResourceAccessKind, SideEffectKind, SkeletonTerminator, SoacDestination,
};
use crate::ast::TypeName;
use crate::interface::{StorageBindingDecl, StorageRole};
use crate::{BindingRef, IdSource};

#[derive(Clone)]
struct Candidate {
    entry: usize,
    result: NodeId,
}

struct InputReplacement {
    project: NodeId,
    view: NodeId,
    view_ty: Type<TypeName>,
    elem_ty: Type<TypeName>,
    binding: BindingRef,
}

pub fn run(inner: &mut EgirInner, binding_ids: &mut IdSource<u32>) {
    loop {
        super::parallelize::rebuild_semantic_dependencies(inner);
        let Some(candidate) = find_candidate(inner) else {
            break;
        };
        materialize_candidate(inner, binding_ids, candidate);
    }
    super::parallelize::rebuild_semantic_dependencies(inner);
    if cfg!(debug_assertions) {
        verify_no_unallocated_multi_consumer_maps(inner);
    }
}

/// The allocation boundary promises that a pure internal SegMap is either
/// consumed once (and therefore eligible for vertical fusion) or backed by a
/// shared logical resource.  Keep that promise executable: otherwise a newly
/// introduced source shape can silently regress to one materialization per
/// consumer.
fn verify_no_unallocated_multi_consumer_maps(inner: &EgirInner) {
    let mut consumers: HashMap<SemanticOpId, HashSet<SemanticOpId>> = HashMap::new();
    for dependency in &inner.semantic_dependencies {
        if dependency.kind == SemanticDependencyKind::Value {
            consumers.entry(dependency.producer.clone()).or_default().insert(dependency.consumer.clone());
        }
    }

    for entry in &inner.entry_points {
        for (block_id, block) in &entry.graph.skeleton.blocks {
            for (effect_index, effect) in block.side_effects.iter().enumerate() {
                let Some(result) = effect.result else {
                    continue;
                };
                let id = SemanticOpId {
                    scope: entry.name.clone(),
                    result,
                };
                if consumers.get(&id).map_or(0, HashSet::len) < 2 {
                    continue;
                }
                let SideEffectKind::Soac(EgirSoac::Seg {
                    kind: SegOpKind::SegMap,
                    map_destinations,
                    output_slots,
                    resources,
                    ..
                }) = &effect.kind
                else {
                    continue;
                };
                let internal_pure_map = output_slots.is_empty()
                    && map_destinations.iter().all(|destination| *destination == SoacDestination::Fresh)
                    && resources.iter().all(|resource| resource.access == SegResourceAccessKind::Read);
                let materializable = entry.graph.skeleton.blocks.len() == 1
                    && dependencies_are_cloneable(
                        &entry.graph,
                        block_id,
                        &dependency_effects(&entry.graph, block_id, effect_index),
                    );
                assert!(
                    !internal_pure_map || !materializable,
                    "pure multi-consumer SegMap {id:?} survived logical allocation"
                );
            }
        }
    }
}

fn find_candidate(inner: &EgirInner) -> Option<Candidate> {
    let mut consumers: HashMap<SemanticOpId, HashSet<SemanticOpId>> = HashMap::new();
    for dependency in &inner.semantic_dependencies {
        if dependency.kind == SemanticDependencyKind::Value {
            consumers.entry(dependency.producer.clone()).or_default().insert(dependency.consumer.clone());
        }
    }
    for (entry_index, entry) in inner.entry_points.iter().enumerate() {
        if entry.graph.skeleton.blocks.len() != 1 {
            continue;
        }
        for (block_id, block) in &entry.graph.skeleton.blocks {
            for (effect_index, effect) in block.side_effects.iter().enumerate() {
                let Some(result) = effect.result else {
                    continue;
                };
                let id = SemanticOpId {
                    scope: entry.name.clone(),
                    result,
                };
                if consumers.get(&id).map_or(0, HashSet::len) < 2 {
                    continue;
                }
                let SideEffectKind::Soac(EgirSoac::Seg {
                    kind: SegOpKind::SegMap,
                    map_bodies,
                    map_destinations,
                    output_slots,
                    resources,
                    ..
                }) = &effect.kind
                else {
                    continue;
                };
                if !map_bodies.is_empty()
                    && output_slots.is_empty()
                    && map_destinations.iter().all(|destination| *destination == SoacDestination::Fresh)
                    && resources.iter().all(|resource| resource.access == SegResourceAccessKind::Read)
                    && dependencies_are_cloneable(
                        &entry.graph,
                        block_id,
                        &dependency_effects(&entry.graph, block_id, effect_index),
                    )
                {
                    return Some(Candidate {
                        entry: entry_index,
                        result,
                    });
                }
            }
        }
    }
    None
}

fn dependencies_are_cloneable(
    graph: &EGraph,
    block_id: crate::ssa::framework::BlockId,
    effects: &HashSet<usize>,
) -> bool {
    let block = &graph.skeleton.blocks[block_id];
    effects.iter().all(|&index| {
        matches!(
            &block.side_effects[index].kind,
            SideEffectKind::Soac(EgirSoac::Seg {
                map_destinations,
                acc_destinations,
                output_slots,
                resources,
                ..
            }) if output_slots.is_empty()
                && map_destinations.iter().all(|destination| *destination == SoacDestination::Fresh)
                && acc_destinations.iter().all(|destination| *destination == SoacDestination::Fresh)
                && resources.iter().all(|resource| resource.access == SegResourceAccessKind::Read)
        )
    })
}

fn materialize_candidate(inner: &mut EgirInner, binding_ids: &mut IdSource<u32>, candidate: Candidate) {
    let entry = &inner.entry_points[candidate.entry];
    let (block_id, effect_index) = entry
        .graph
        .skeleton
        .blocks
        .iter()
        .find_map(|(block_id, block)| {
            block
                .side_effects
                .iter()
                .position(|effect| effect.result == Some(candidate.result))
                .map(|index| (block_id, index))
        })
        .expect("multi-consumer producer disappeared");
    let producer_effect = entry.graph.skeleton.blocks[block_id].side_effects[effect_index].clone();
    let producer_dependencies = dependency_effects(&entry.graph, block_id, effect_index);
    let dependency_bindings = dependency_bindings(&entry.graph, block_id, &producer_dependencies);
    let SideEffectKind::Soac(EgirSoac::Seg {
        space,
        map_bodies,
        map_output_elem_types,
        result_types,
        ..
    }) = &producer_effect.kind
    else {
        unreachable!();
    };
    let used_bindings = all_bindings(inner);
    let mut bindings = Vec::with_capacity(map_bodies.len());
    let mut sizes = Vec::with_capacity(map_bodies.len());
    let mut used = used_bindings;
    for elem_ty in map_output_elem_types {
        let binding = next_free_binding(binding_ids, &mut used);
        bindings.push(binding);
        sizes.push(size_for_space(space, elem_ty));
    }

    let mut producer = entry.clone();
    producer.name = fresh_entry_name(inner, &format!("{}_materialize_shared", entry.name));
    producer.outputs.clear();
    producer.slot_sources.clear();
    producer.return_ty = Type::Constructed(TypeName::Unit, vec![]);
    producer.storage_bindings.retain(|declaration| {
        declaration.role == StorageRole::Input || dependency_bindings.contains(&declaration.binding)
    });
    let producer_owner = SemanticOpId {
        scope: producer.name.clone(),
        result: candidate.result,
    };
    let producer_graph = &mut producer.graph;
    let mut output_views = Vec::new();
    for ((&binding, elem_ty), size) in bindings.iter().zip(map_output_elem_types).zip(&sizes) {
        let view = graph_ops::intern_storage_view(producer_graph, binding, elem_ty.clone(), None);
        output_views.push(view);
        producer.storage_bindings.push(StorageBindingDecl {
            binding,
            role: StorageRole::Output,
            elem_ty: elem_ty.clone(),
            length: buffer_len(size),
        });
    }
    for (producer_block_id, block) in producer_graph.skeleton.blocks.iter_mut() {
        if producer_block_id == block_id {
            let mut index = 0usize;
            block.side_effects.retain(|_| {
                let retain = producer_dependencies.contains(&index);
                index += 1;
                retain
            });
        } else {
            block.side_effects.clear();
        }
        block.term = SkeletonTerminator::Return(None);
    }
    let producer_effect = producer_graph.skeleton.blocks[block_id]
        .side_effects
        .iter_mut()
        .find(|effect| effect.result == Some(candidate.result))
        .expect("producer clone retained its SegMap");
    if let SideEffectKind::Soac(EgirSoac::Seg {
        placement,
        map_destinations,
        resources,
        ..
    }) = &mut producer_effect.kind
    {
        *placement = SegPlacement::Kernel;
        map_destinations.fill(SoacDestination::OutputView);
        producer_effect.operand_nodes.extend(output_views.iter().copied());
        for &binding in &bindings {
            resources.push(SegResourceAccess {
                binding,
                access: SegResourceAccessKind::Write,
            });
        }
        resources.sort_by_key(|resource| (resource.binding.set, resource.binding.binding));
    }

    // Rewire the source entry to read the shared storage views and remove the
    // now-separate producer operation.
    let entry = &mut inner.entry_points[candidate.entry];
    let mut project_views = Vec::new();
    for (lane, ((&binding, elem_ty), result_ty)) in
        bindings.iter().zip(map_output_elem_types).zip(result_types).enumerate()
    {
        let view = graph_ops::intern_storage_view(&mut entry.graph, binding, elem_ty.clone(), None);
        let project = entry
            .graph
            .nodes
            .iter()
            .find_map(|(node, definition)| match definition {
                super::types::ENode::Pure {
                    op: PureOp::Project { index },
                    operands,
                } if *index as usize == lane && operands.first() == Some(&candidate.result) => Some(node),
                _ => None,
            })
            .unwrap_or_else(|| {
                entry.graph.intern_pure(
                    PureOp::Project { index: lane as u32 },
                    smallvec::smallvec![candidate.result],
                    result_ty.clone(),
                )
            });
        project_views.push(InputReplacement {
            project,
            view,
            view_ty: entry.graph.types[&view].clone(),
            elem_ty: elem_ty.clone(),
            binding,
        });
        entry.storage_bindings.push(StorageBindingDecl {
            binding,
            role: StorageRole::Input,
            elem_ty: elem_ty.clone(),
            length: buffer_len(&sizes[lane]),
        });
    }
    retarget_input_metadata(&mut entry.graph, &project_views);
    for replacement in &project_views {
        graph_ops::replace_all_references(&mut entry.graph, replacement.project, replacement.view);
    }
    // Preserve whole-result users as well as projected lanes. The Seg result
    // is a tuple even for one output, so rebuild that exact tuple from the
    // shared storage views before retiring the producer.
    let result_ty = entry.graph.types[&candidate.result].clone();
    let replacement_tuple = entry.graph.intern_pure(
        PureOp::Tuple(project_views.len()),
        project_views.iter().map(|replacement| replacement.view).collect(),
        result_ty,
    );
    graph_ops::replace_all_references(&mut entry.graph, candidate.result, replacement_tuple);
    for source in entry.slot_sources.iter_mut().flatten() {
        if let Some(replacement) =
            project_views.iter().find(|replacement| replacement.project == source.value)
        {
            source.value = replacement.view;
        }
    }
    entry.graph.skeleton.blocks[block_id].side_effects.remove(effect_index);
    super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph);

    for (slot, &binding) in bindings.iter().enumerate() {
        inner.resource_hints.insert(
            binding,
            CompilerResource::new(
                CompilerResourceKind::MultiConsumerArray,
                Some(producer_owner.clone()),
                slot,
            ),
        );
    }
    inner.entry_points.push(producer);
}

/// Return the transitive semantic producer closure needed to compute one
/// effect.  Materialization is an entry prepass, so an internal producer chain
/// must move with the multi-consumer map instead of leaving dangling Project
/// nodes in the cloned graph.
fn dependency_effects(
    graph: &EGraph,
    block_id: crate::ssa::framework::BlockId,
    root: usize,
) -> HashSet<usize> {
    let block = &graph.skeleton.blocks[block_id];
    let producers: HashMap<NodeId, usize> = block
        .side_effects
        .iter()
        .enumerate()
        .filter_map(|(index, effect)| effect.result.map(|result| (result, index)))
        .collect();
    let mut retained = HashSet::from([root]);
    let mut effects = vec![root];
    while let Some(effect_index) = effects.pop() {
        let mut nodes: Vec<_> = block.side_effects[effect_index].referenced_nodes().collect();
        let mut visited = HashSet::new();
        while let Some(node) = nodes.pop() {
            if !visited.insert(node) {
                continue;
            }
            if let Some(&producer) = producers.get(&node) {
                if retained.insert(producer) {
                    effects.push(producer);
                }
                continue;
            }
            if let Some(definition) = graph.nodes.get(node) {
                nodes.extend(definition.children());
            }
        }
    }
    retained
}

fn dependency_bindings(
    graph: &EGraph,
    block_id: crate::ssa::framework::BlockId,
    effects: &HashSet<usize>,
) -> HashSet<BindingRef> {
    let block = &graph.skeleton.blocks[block_id];
    let mut bindings = HashSet::new();
    for &effect_index in effects {
        let effect = &block.side_effects[effect_index];
        let mut nodes: Vec<_> = effect.referenced_nodes().collect();
        let mut visited = HashSet::new();
        while let Some(node) = nodes.pop() {
            if !visited.insert(node) {
                continue;
            }
            if let Some(binding) = graph_ops::extract_storage_view_source(graph, node) {
                bindings.insert(binding);
            }
            if let Some(definition) = graph.nodes.get(node) {
                nodes.extend(definition.children());
            }
        }
        match &effect.kind {
            SideEffectKind::Soac(EgirSoac::Seg { resources, .. }) => {
                bindings.extend(resources.iter().map(|resource| resource.binding));
            }
            SideEffectKind::Soac(EgirSoac::Filter {
                scratch_out, len_out, ..
            }) => {
                bindings.extend(scratch_out.iter().chain(len_out).copied());
            }
            _ => {}
        }
    }
    bindings
}

fn retarget_input_metadata(graph: &mut EGraph, replacements: &[InputReplacement]) {
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            match &mut effect.kind {
                SideEffectKind::Soac(EgirSoac::Seg {
                    space,
                    input_array_types,
                    input_elem_types,
                    resources,
                    ..
                }) => {
                    for input in 0..input_array_types.len() {
                        if let Some(replacement) = replacements
                            .iter()
                            .find(|replacement| effect.operand_nodes[input] == replacement.project)
                        {
                            input_array_types[input] = replacement.view_ty.clone();
                            input_elem_types[input] = replacement.elem_ty.clone();
                            if !resources.iter().any(|resource| resource.binding == replacement.binding) {
                                resources.push(SegResourceAccess {
                                    binding: replacement.binding,
                                    access: SegResourceAccessKind::Read,
                                });
                            }
                        }
                    }
                    replace_space_nodes(space, replacements);
                }
                SideEffectKind::Soac(EgirSoac::Screma {
                    input_array_types,
                    input_elem_types,
                    ..
                }) => {
                    for input in 0..input_array_types.len() {
                        if let Some(replacement) = replacements
                            .iter()
                            .find(|replacement| effect.operand_nodes[input] == replacement.project)
                        {
                            input_array_types[input] = replacement.view_ty.clone();
                            input_elem_types[input] = replacement.elem_ty.clone();
                        }
                    }
                }
                SideEffectKind::Soac(EgirSoac::Filter {
                    space,
                    input_array_type,
                    input_elem_type,
                    ..
                }) => {
                    if let Some(replacement) = replacements
                        .iter()
                        .find(|replacement| effect.operand_nodes[0] == replacement.project)
                    {
                        *input_array_type = replacement.view_ty.clone();
                        *input_elem_type = replacement.elem_ty.clone();
                    }
                    if let Some(space) = space {
                        replace_space_nodes(space, replacements);
                    }
                }
                _ => {}
            }
        }
    }
}

fn replace_space_nodes(space: &mut super::types::SegSpace, replacements: &[InputReplacement]) {
    for extent in &mut space.dims {
        let node = match extent {
            SegExtent::PushConstant { node, .. }
            | SegExtent::ResourceLength { node, .. }
            | SegExtent::Value(node) => node,
            SegExtent::Fixed(_) => continue,
        };
        if let Some(replacement) = replacements.iter().find(|replacement| *node == replacement.project) {
            *node = replacement.view;
            if let SegExtent::ResourceLength {
                binding: extent_binding,
                ..
            } = extent
            {
                *extent_binding = replacement.binding;
            }
        }
    }
}

fn all_bindings(inner: &EgirInner) -> HashSet<BindingRef> {
    inner
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
        .collect()
}

fn next_free_binding(binding_ids: &mut IdSource<u32>, used: &mut HashSet<BindingRef>) -> BindingRef {
    loop {
        let binding = BindingRef::new(super::from_tlc::AUTO_STORAGE_SET, binding_ids.next_id());
        if used.insert(binding) {
            return binding;
        }
    }
}

fn size_for_space(space: &super::types::SegSpace, elem_ty: &Type<TypeName>) -> LogicalSize {
    let elem_bytes = crate::ssa::layout::type_byte_size(elem_ty).unwrap_or(1);
    if let Some(count) = space.dims.iter().try_fold(1u64, |count, extent| match extent {
        SegExtent::Fixed(length) => count.checked_mul(*length as u64),
        _ => None,
    }) {
        return LogicalSize::FixedBytes(count.saturating_mul(elem_bytes as u64));
    }
    match space.dims.as_slice() {
        [SegExtent::ResourceLength {
            binding,
            elem_bytes: source_elem_bytes,
            ..
        }] => LogicalSize::LikeBinding {
            binding: *binding,
            elem_bytes,
            src_elem_bytes: *source_elem_bytes,
        },
        _ => LogicalSize::SameAsDispatch { elem_bytes },
    }
}

fn fresh_entry_name(inner: &EgirInner, base: &str) -> String {
    if inner.entry_points.iter().all(|entry| entry.name != base) {
        return base.to_string();
    }
    for suffix in 1.. {
        let candidate = format!("{base}_{suffix}");
        if inner.entry_points.iter().all(|entry| entry.name != candidate) {
            return candidate;
        }
    }
    unreachable!()
}
