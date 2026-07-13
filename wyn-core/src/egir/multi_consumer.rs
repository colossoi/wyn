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
    CompilerResource, CompilerResourceKind, LogicalSize, MaterializationId, MaterializationRequirement,
    MaterializationSubstitution, ResourceId, ResourceOrigin, SemanticDependencyKind, SemanticOpId,
    SemanticProgram, SemanticResourceDecl, SemanticResourceRef,
};
use super::types::{
    EGraph, EgirSoac, NodeId, PureOp, SegExtent, SegOpKind, SegPlacement, SegResourceAccess,
    SegResourceAccessKind, SideEffectKind, SoacDestination,
};
use crate::ast::TypeName;
use crate::interface::StorageRole;

#[derive(Clone)]
struct Candidate {
    entry: usize,
    result: NodeId,
    id: SemanticOpId,
}

struct InputReplacement {
    project: NodeId,
    view: NodeId,
    view_ty: Type<TypeName>,
    elem_ty: Type<TypeName>,
    resource: ResourceId,
}

pub fn run(inner: &mut SemanticProgram) {
    loop {
        super::semantic_graph::rebuild_dependencies(inner);
        let Some(candidate) = find_candidate(inner) else {
            break;
        };
        materialize_candidate(inner, candidate);
    }
    super::semantic_graph::rebuild_dependencies(inner);
    if cfg!(debug_assertions) {
        verify_no_unallocated_multi_consumer_maps(inner);
    }
}

/// The allocation boundary promises that a pure internal SegMap is either
/// consumed once (and therefore eligible for vertical fusion) or backed by a
/// shared logical resource.  Keep that promise executable: otherwise a newly
/// introduced source shape can silently regress to one materialization per
/// consumer.
fn verify_no_unallocated_multi_consumer_maps(inner: &SemanticProgram) {
    let mut consumers: HashMap<SemanticOpId, HashSet<SemanticOpId>> = HashMap::new();
    for dependency in &inner.semantic_dependencies {
        if dependency.kind == SemanticDependencyKind::Value {
            consumers.entry(dependency.producer.clone()).or_default().insert(dependency.consumer.clone());
        }
    }

    for entry in &inner.entry_points {
        for (block_id, block) in &entry.graph.skeleton.blocks {
            for (effect_index, effect) in block.side_effects.iter().enumerate() {
                let Some(_) = effect.result else {
                    continue;
                };
                let id = effect.semantic_id.expect("semantic operation id assigned after segmentation");
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

fn find_candidate(inner: &SemanticProgram) -> Option<Candidate> {
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
                let id = effect.semantic_id.expect("semantic operation id assigned after segmentation");
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
                        id,
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

fn materialize_candidate(inner: &mut SemanticProgram, candidate: Candidate) {
    let entry = &inner.entry_points[candidate.entry];
    let producer_index = entry.graph.side_effect_index();
    let producer_site = producer_index.site(candidate.result).expect("multi-consumer producer disappeared");
    let (block_id, effect_index) = (producer_site.block, producer_site.index);
    let producer_effect = entry.graph.skeleton.blocks[block_id].side_effects[effect_index].clone();
    let producer_dependencies = dependency_effects(&entry.graph, block_id, effect_index);
    let dependency_resources = dependency_resources(&entry.graph, block_id, &producer_dependencies);
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
    let first_resource = inner.resources.len() as u32;
    let output_resources =
        (0..map_bodies.len()).map(|slot| ResourceId(first_resource + slot as u32)).collect::<Vec<_>>();
    let mut sizes = Vec::with_capacity(map_bodies.len());
    for elem_ty in map_output_elem_types {
        sizes.push(size_for_space(space, elem_ty));
    }

    let producer_storage = entry
        .resource_declarations
        .iter()
        .filter(|declaration| {
            declaration.role == StorageRole::Input || dependency_resources.contains(&declaration.resource.0)
        })
        .cloned()
        .collect();
    let projection = super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
        .selected(HashSet::from([super::graph_projector::EffectSite {
            block: block_id,
            index: effect_index,
        }]))
        .expect("multi-consumer producer projection must be valid");
    let projected_result =
        projection.node(candidate.result).expect("multi-consumer producer result must be projected");
    let producer_aliases = projection.remap_aliases(&entry.aliases);
    let mut producer = MaterializationRequirement {
        id: MaterializationId(inner.materializations.len() as u32),
        producer: candidate.id,
        name: fresh_entry_name(inner, &format!("{}_materialize_shared", entry.name)),
        span: entry.span,
        execution_model: entry.execution_model.clone(),
        inputs: entry.inputs.clone(),
        resource_declarations: producer_storage,
        params: entry.params.clone(),
        return_ty: Type::Constructed(TypeName::Unit, vec![]),
        graph: projection.graph,
        control_headers: projection.control_headers,
        aliases: producer_aliases,
        substitutions: Vec::new(),
    };
    let producer_owner = candidate.id;
    let producer_graph = &mut producer.graph;
    let mut output_views = Vec::new();
    for ((&resource, elem_ty), size) in output_resources.iter().zip(map_output_elem_types).zip(&sizes) {
        let view = graph_ops::intern_resource_view(producer_graph, resource, elem_ty.clone(), None);
        output_views.push(view);
        producer.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(resource),
            role: StorageRole::Output,
            elem_ty: elem_ty.clone(),
            size: size.clone(),
        });
        producer.substitutions.push(MaterializationSubstitution {
            resource: SemanticResourceRef(resource),
            consumers: Vec::new(),
        });
    }
    let retained_index = producer_graph.side_effect_index();
    let producer_effect = retained_index
        .effect_mut(producer_graph, projected_result)
        .expect("producer projection retained its SegMap");
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
        for &resource in &output_resources {
            resources.push(SegResourceAccess {
                resource: SemanticResourceRef(resource),
                access: SegResourceAccessKind::Write,
            });
        }
        resources.sort_by_key(|access| access.resource.0 .0);
    }

    // Rewire the source entry to read the shared storage views and remove the
    // now-separate producer operation.
    let entry = &mut inner.entry_points[candidate.entry];
    let mut project_views = Vec::new();
    for (lane, ((&resource, elem_ty), result_ty)) in
        output_resources.iter().zip(map_output_elem_types).zip(result_types).enumerate()
    {
        let view = graph_ops::intern_resource_view(&mut entry.graph, resource, elem_ty.clone(), None);
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
            resource,
        });
        entry.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(resource),
            role: StorageRole::Input,
            elem_ty: elem_ty.clone(),
            size: sizes[lane].clone(),
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
    for route in &mut entry.output_routes {
        if let Some(replacement) =
            project_views.iter().find(|replacement| replacement.project == route.source.value)
        {
            route.source.value = replacement.view;
        }
    }
    entry.graph.skeleton.blocks[block_id].side_effects.remove(effect_index);
    super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph);

    for (slot, ((&resource, elem_ty), size)) in
        output_resources.iter().zip(map_output_elem_types).zip(&sizes).enumerate()
    {
        debug_assert_eq!(resource.0 as usize, inner.resources.len());
        inner.resources.push(super::program::LogicalResource {
            id: resource,
            origin: ResourceOrigin::Compiler(CompilerResource::new(
                CompilerResourceKind::MultiConsumerArray,
                Some(producer_owner),
                slot,
            )),
            elem_ty: elem_ty.clone(),
            size: size.clone(),
        });
    }
    inner.materializations.push(producer);
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
            // Stop at a node that is another effect's result: that producer
            // is retained whole, and its own operands belong to it.
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

fn dependency_resources(
    graph: &EGraph,
    block_id: crate::ssa::framework::BlockId,
    effects: &HashSet<usize>,
) -> HashSet<ResourceId> {
    let block = &graph.skeleton.blocks[block_id];
    let mut dependencies = HashSet::new();
    for &effect_index in effects {
        let effect = &block.side_effects[effect_index];
        wyn_graph::for_each_reachable(
            effect.referenced_nodes(),
            wyn_graph::WalkOrder::DepthFirst,
            |node, nodes| {
                if let Some(definition) = graph.nodes.get(node) {
                    nodes.extend(definition.children());
                }
            },
            |node| {
                if let Some(resource) = graph_ops::extract_storage_view_source(graph, node) {
                    dependencies.insert(resource.0);
                }
            },
        );
        match &effect.kind {
            SideEffectKind::Soac(EgirSoac::Seg { resources, .. }) => {
                dependencies.extend(resources.iter().map(|access| access.resource.0));
            }
            SideEffectKind::Soac(EgirSoac::Filter { output, .. }) => {
                if let super::types::FilterOutput::Runtime { scratch, length } = output {
                    dependencies.insert(scratch.0);
                    if let super::types::RuntimeFilterLength::EntryOutput(length) = length {
                        dependencies.insert(length.0);
                    }
                }
            }
            _ => {}
        }
    }
    dependencies
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
                            if !resources.iter().any(|access| access.resource.0 == replacement.resource) {
                                resources.push(SegResourceAccess {
                                    resource: SemanticResourceRef(replacement.resource),
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
                    state,
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
                    match state {
                        super::types::FilterState::Semantic { space }
                        | super::types::FilterState::Scheduled { space, .. } => {
                            replace_space_nodes(space, replacements)
                        }
                        super::types::FilterState::Raw => {}
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
            if let SegExtent::ResourceLength { resource, .. } = extent {
                *resource = SemanticResourceRef(replacement.resource);
            }
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
            resource,
            elem_bytes: source_elem_bytes,
            ..
        }] => LogicalSize::LikeResource {
            resource: resource.0,
            elem_bytes,
            src_elem_bytes: *source_elem_bytes,
        },
        _ => LogicalSize::SameAsDispatch { elem_bytes },
    }
}

fn fresh_entry_name(inner: &SemanticProgram, base: &str) -> String {
    let available = |name: &str| {
        inner.entry_points.iter().all(|entry| entry.name != name)
            && inner.materializations.iter().all(|requirement| requirement.name != name)
    };
    if available(base) {
        return base.to_string();
    }
    for suffix in 1.. {
        let candidate = format!("{base}_{suffix}");
        if available(&candidate) {
            return candidate;
        }
    }
    unreachable!()
}
