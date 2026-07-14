//! Semantic residency planning for arrays and cross-dispatch scalars.
//!
//! The pass recognizes shared producers, runtime gathers, and invariant scalar
//! reductions after output realization and semantic fusion. It records each as
//! a typed materialization requirement, allocates its logical handoff resource,
//! and rewires consumers to explicit storage views or loads. Target lowering
//! only chooses and schedules the physical kernel recipe.

use std::collections::{HashMap, HashSet};

use polytype::Type;

use super::graph_ops;
use super::program::{
    CompilerResource, CompilerResourceKind, LogicalSize, MaterializationId, MaterializationKind,
    MaterializationRequirement, MaterializationSubstitution, ResourceId, SemanticDependencyKind,
    SemanticOpId, SemanticProgram, SemanticResourceDecl, SemanticResourceRef,
};
use super::types::{
    EGraph, ENode, EgirSoac, NodeId, PureOp, SegExtent, SegOpKind, SegPlacement, SegResourceAccess,
    SegResourceAccessKind, SideEffectKind, SideEffectSite, SoacDestination,
};
use crate::ast::TypeName;
use crate::interface::StorageRole;
use crate::ssa::types::ExecutionModel;

#[derive(Clone)]
struct Candidate {
    entry: usize,
    result: NodeId,
    id: SemanticOpId,
    kind: MaterializationKind,
}

#[derive(Clone)]
struct OutputSpec {
    elem_ty: Type<TypeName>,
    result_ty: Type<TypeName>,
    size: LogicalSize,
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
    super::realize_outputs::reconcile::run(inner)
        .expect("materialized storage views must reconcile with captured region parameters");
    if cfg!(debug_assertions) {
        verify_residency_requirements_satisfied(inner);
        super::realize_outputs::verify::check(inner)
            .expect("runtime-sized Composite survived EGIR residency planning");
    }
}

/// The allocation boundary promises that a pure internal SegMap is either
/// consumed once (and therefore eligible for vertical fusion) or backed by a
/// shared logical resource.  Keep that promise executable: otherwise a newly
/// introduced source shape can silently regress to one materialization per
/// consumer.
fn verify_residency_requirements_satisfied(inner: &SemanticProgram) {
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
                let id = effect.required_semantic_id();
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
        for (block_id, block) in &entry.graph.skeleton.blocks {
            if block_id != entry.graph.skeleton.entry {
                continue;
            }
            for (effect_index, effect) in block.side_effects.iter().enumerate() {
                let Some(result) = effect.result else {
                    continue;
                };
                let id = effect.required_semantic_id();
                let SideEffectKind::Soac(EgirSoac::Seg {
                    kind,
                    map_bodies,
                    map_destinations,
                    acc_destinations,
                    resources,
                    ..
                }) = &effect.kind
                else {
                    continue;
                };
                let pure = map_destinations
                    .iter()
                    .all(|destination| *destination == SoacDestination::Fresh)
                    && acc_destinations.iter().all(|destination| *destination == SoacDestination::Fresh)
                    && resources.iter().all(|resource| {
                        resource.access == SegResourceAccessKind::Read
                            || entry
                                .resource_abi
                                .outputs
                                .iter()
                                .flatten()
                                .any(|output| *output == resource.resource.0)
                    });
                let dependencies = dependency_effects(&entry.graph, block_id, effect_index);
                let upstream = dependencies
                    .iter()
                    .copied()
                    .filter(|index| *index != effect_index)
                    .collect::<HashSet<_>>();
                if !pure || !dependencies_are_cloneable(&entry.graph, block_id, &upstream) {
                    continue;
                }
                let semantic_consumers = consumers.get(&id);
                let consumer_count = semantic_consumers.map_or(0, HashSet::len);
                let materialization = match kind {
                    SegOpKind::SegMap if !map_bodies.is_empty() => {
                        if consumer_count >= 2 {
                            Some(MaterializationKind::SharedArray)
                        } else if runtime_composite(&entry.graph.types[&result])
                            && requires_array_residency(&entry.graph, result, block_id, effect_index)
                        {
                            Some(MaterializationKind::Gather)
                        } else {
                            None
                        }
                    }
                    SegOpKind::SegScan { operators } if !operators.is_empty() => {
                        if consumer_count >= 2 {
                            Some(MaterializationKind::SharedArray)
                        } else if runtime_composite(&entry.graph.types[&result])
                            && requires_array_residency(&entry.graph, result, block_id, effect_index)
                        {
                            Some(MaterializationKind::Gather)
                        } else {
                            None
                        }
                    }
                    SegOpKind::SegRed { operators }
                        if operators.len() == 1
                            && (has_segmented_consumer(entry, semantic_consumers)
                                || !matches!(entry.execution_model, ExecutionModel::Compute { .. }))
                            && scalar_result_is_used(&entry.graph, result, block_id, effect_index)
                            && invocation_invariant(entry, block_id, &dependencies) =>
                    {
                        Some(MaterializationKind::Scalar)
                    }
                    _ => None,
                };
                if let Some(kind) = materialization {
                    return Some(Candidate {
                        entry: entry_index,
                        result,
                        id,
                        kind,
                    });
                }
            }
        }
    }
    None
}

fn has_segmented_consumer(
    entry: &super::program::SemanticEntry,
    consumers: Option<&HashSet<SemanticOpId>>,
) -> bool {
    let Some(consumers) = consumers else {
        return false;
    };
    entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects).any(|effect| {
        matches!(effect.kind, SideEffectKind::Soac(EgirSoac::Seg { .. }))
            && consumers.contains(&effect.required_semantic_id())
    })
}

fn runtime_composite(ty: &Type<TypeName>) -> bool {
    match ty {
        Type::Constructed(TypeName::Array, args) if args.len() >= 3 => {
            matches!(args[1], Type::Constructed(TypeName::ArrayVariantComposite, _))
                && !matches!(args[2], Type::Constructed(TypeName::Size(_), _))
        }
        Type::Constructed(_, args) => args.iter().any(runtime_composite),
        Type::Variable(_) => false,
    }
}

fn depends_on(graph: &EGraph, root: NodeId, target: NodeId) -> bool {
    wyn_graph::reachable_set([root], wyn_graph::WalkOrder::DepthFirst, |node, pending| {
        if let Some(definition) = graph.nodes.get(node) {
            pending.extend(definition.children());
        }
    })
    .contains(&target)
}

fn requires_array_residency(
    graph: &EGraph,
    result: NodeId,
    producer_block: crate::ssa::framework::BlockId,
    producer_index: usize,
) -> bool {
    if graph.nodes.iter().any(|(_, node)| {
        matches!(node, ENode::Pure { op: PureOp::Index, operands } if operands.first().is_some_and(|base| depends_on(graph, *base, result)))
    }) {
        return true;
    }
    graph.skeleton.blocks.iter().any(|(block_id, block)| {
        block.side_effects.iter().enumerate().any(|(index, effect)| {
            if block_id == producer_block && index == producer_index {
                return false;
            }
            let SideEffectKind::Soac(soac) = &effect.kind else {
                return false;
            };
            soac.capture_nodes().any(|capture| depends_on(graph, capture, result))
        })
    })
}

fn scalar_result_is_used(
    graph: &EGraph,
    result: NodeId,
    producer_block: crate::ssa::framework::BlockId,
    producer_index: usize,
) -> bool {
    graph.skeleton.blocks.iter().any(|(block_id, block)| {
        block.side_effects.iter().enumerate().any(|(index, effect)| {
            (block_id != producer_block || index != producer_index)
                && effect.referenced_nodes().any(|node| depends_on(graph, node, result))
        }) || block
            .term
            .clone()
            .try_map::<()>(
                |node| if depends_on(graph, node, result) { Err(()) } else { Ok(node) },
                Ok,
            )
            .is_err()
    })
}

fn invocation_invariant(
    entry: &super::program::SemanticEntry,
    block_id: crate::ssa::framework::BlockId,
    effects: &HashSet<usize>,
) -> bool {
    if matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
        return true;
    }
    if entry.params.len() != entry.inputs.len() {
        return false;
    }
    let block = &entry.graph.skeleton.blocks[block_id];
    let mut roots = Vec::new();
    for &index in effects {
        roots.extend(block.side_effects[index].referenced_nodes());
    }
    let reachable = wyn_graph::reachable_set(roots, wyn_graph::WalkOrder::DepthFirst, |node, pending| {
        if let Some(definition) = entry.graph.nodes.get(node) {
            pending.extend(definition.children());
        }
    });
    reachable.into_iter().all(|node| {
        let Some(ENode::FuncParam { index }) = entry.graph.nodes.get(node) else {
            return true;
        };
        entry.inputs.get(*index).is_some_and(|input| {
            input.storage_binding.is_some()
                || input.uniform_binding.is_some()
                || input.push_constant.is_some()
        })
    })
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
    let source_output_resources =
        entry.resource_abi.outputs.iter().flatten().copied().collect::<HashSet<_>>();
    let producer_index = entry.graph.side_effect_index();
    let producer_site = producer_index.site(candidate.result).expect("multi-consumer producer disappeared");
    let (block_id, effect_index) = (producer_site.block, producer_site.index);
    let producer_effect = entry.graph.skeleton.blocks[block_id].side_effects[effect_index].clone();
    let producer_dependencies = dependency_effects(&entry.graph, block_id, effect_index);
    let dependency_resources = dependency_resources(&entry.graph, block_id, &producer_dependencies);
    let SideEffectKind::Soac(EgirSoac::Seg {
        space,
        kind,
        map_output_elem_types,
        result_types,
        ..
    }) = &producer_effect.kind
    else {
        unreachable!();
    };
    let output_specs = output_specs(
        &entry.graph,
        candidate.kind,
        space,
        kind,
        map_output_elem_types,
        result_types,
    )
    .expect("materialization candidate must expose concrete outputs");

    let producer_storage = entry
        .resource_declarations
        .iter()
        .filter(|declaration| dependency_resources.contains(&declaration.resource.0))
        .cloned()
        .collect();
    let projection = super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
        .selected_entry_recipe(HashSet::from([SideEffectSite {
            block: block_id,
            index: effect_index,
        }]))
        .expect("multi-consumer producer projection must be valid");
    let projected_result =
        projection.node(candidate.result).expect("multi-consumer producer result must be projected");
    let producer_aliases = projection.remap_aliases(&entry.aliases);
    let execution_model = match entry.execution_model {
        ExecutionModel::Compute { local_size } => ExecutionModel::Compute { local_size },
        ExecutionModel::Vertex | ExecutionModel::Fragment => ExecutionModel::Compute {
            local_size: (64, 1, 1),
        },
    };
    let name_suffix = match candidate.kind {
        MaterializationKind::SharedArray => "materialize_shared",
        MaterializationKind::Gather => "gather_materialize",
        MaterializationKind::Scalar => "prepass_scalar",
    };
    let mut producer = MaterializationRequirement {
        id: MaterializationId(inner.materializations.len() as u32),
        kind: candidate.kind,
        producer: candidate.id,
        entry: super::program::SemanticEntry {
            name: fresh_entry_name(inner, &format!("{}_{name_suffix}", entry.name)),
            span: entry.span,
            execution_model,
            inputs: entry.inputs.clone(),
            outputs: Vec::new(),
            resource_abi: super::program::EntryResourceAbi {
                inputs: entry.resource_abi.inputs.clone(),
                outputs: Vec::new(),
            },
            resource_declarations: producer_storage,
            params: entry.params.clone(),
            return_ty: Type::Constructed(TypeName::Unit, vec![]),
            graph: projection.graph,
            control_headers: projection.control_headers,
            aliases: producer_aliases,
            output_routes: Vec::new(),
        },
        substitutions: Vec::new(),
    };
    if !matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
        compact_graphics_materialization(&mut producer.entry);
    }
    let producer_owner = candidate.id;
    let resource_kind = match candidate.kind {
        MaterializationKind::SharedArray => CompilerResourceKind::MultiConsumerArray,
        MaterializationKind::Gather => CompilerResourceKind::GatherHandoff,
        MaterializationKind::Scalar => CompilerResourceKind::ScalarHandoff,
    };
    let output_resources = output_specs
        .iter()
        .enumerate()
        .map(|(slot, output)| {
            inner.alloc_compiler_resource(
                CompilerResource::new(resource_kind, Some(producer_owner), slot),
                output.elem_ty.clone(),
                output.size.clone(),
            )
        })
        .collect::<Vec<_>>();
    let producer_graph = &mut producer.entry.graph;
    let mut output_views = Vec::new();
    for (&resource, output) in output_resources.iter().zip(&output_specs) {
        let view = graph_ops::intern_resource_view(producer_graph, resource, output.elem_ty.clone(), None);
        output_views.push(view);
        producer.entry.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(resource),
            role: StorageRole::Output,
            elem_ty: output.elem_ty.clone(),
            size: output.size.clone(),
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
        acc_destinations,
        output_slots,
        resources,
        ..
    }) = &mut producer_effect.kind
    {
        *placement = SegPlacement::Kernel;
        *output_slots = Vec::new();
        resources.retain(|access| {
            access.access == SegResourceAccessKind::Read
                || !source_output_resources.contains(&access.resource.0)
        });
        if candidate.kind != MaterializationKind::Scalar {
            map_destinations.fill(SoacDestination::OutputView);
            acc_destinations.fill(SoacDestination::OutputView);
            producer_effect.operand_nodes.extend(output_views.iter().copied());
            for &resource in &output_resources {
                resources.push(SegResourceAccess {
                    resource: SemanticResourceRef(resource),
                    access: SegResourceAccessKind::Write,
                });
            }
            resources.sort_by_key(|access| access.resource);
        }
    }
    if candidate.kind != MaterializationKind::Scalar {
        let original_ty = producer_graph.types[&projected_result].clone();
        let view_types =
            output_views.iter().map(|view| producer_graph.types[view].clone()).collect::<Vec<_>>();
        let materialized_ty = if view_types.len() == 1 && original_ty == output_specs[0].result_ty {
            view_types[0].clone()
        } else {
            Type::Constructed(TypeName::Tuple(view_types.len()), view_types)
        };
        producer_graph.retype_node(projected_result, materialized_ty);
    }
    if candidate.kind == MaterializationKind::Scalar {
        let result_ty = producer_graph.types[&projected_result].clone();
        let value = if result_ty == output_specs[0].result_ty {
            projected_result
        } else {
            producer_graph.intern_pure(
                PureOp::Project { index: 0 },
                smallvec::smallvec![projected_result],
                output_specs[0].result_ty.clone(),
            )
        };
        let zero = graph_ops::intern_u32(producer_graph, 0, None);
        let mut next_effect = graph_ops::next_effect_token(producer_graph);
        graph_ops::emit_storage_store(
            producer_graph,
            producer_graph.skeleton.entry,
            output_views[0],
            zero,
            value,
            output_specs[0].elem_ty.clone(),
            &mut next_effect,
            None,
        );
    }

    // Rewire the source entry to read the materialized resources and remove
    // the now-separate producer operation.
    let entry = &mut inner.entry_points[candidate.entry];
    let mut array_replacements = Vec::new();
    let mut replacements = Vec::new();
    let mut scalar_effects = Vec::new();
    for (lane, (&resource, output)) in output_resources.iter().zip(&output_specs).enumerate() {
        let view =
            graph_ops::intern_resource_view(&mut entry.graph, resource, output.elem_ty.clone(), None);
        let project = entry
            .graph
            .nodes
            .iter()
            .find_map(|(node, definition)| match definition {
                ENode::Pure {
                    op: PureOp::Project { index },
                    operands,
                } if *index as usize == lane && operands.first() == Some(&candidate.result) => Some(node),
                _ => None,
            })
            .unwrap_or_else(|| {
                entry.graph.intern_pure(
                    PureOp::Project { index: lane as u32 },
                    smallvec::smallvec![candidate.result],
                    output.result_ty.clone(),
                )
            });
        let value = if candidate.kind == MaterializationKind::Scalar {
            let zero = graph_ops::intern_u32(&mut entry.graph, 0, None);
            let place = entry.graph.intern_pure(
                PureOp::ViewIndex,
                smallvec::smallvec![view, zero],
                output.elem_ty.clone(),
            );
            let mut next_effect = graph_ops::next_effect_token(&entry.graph);
            let loaded = graph_ops::emit_load(
                &mut entry.graph,
                block_id,
                place,
                output.elem_ty.clone(),
                &mut next_effect,
                None,
            );
            scalar_effects.push(
                entry.graph.skeleton.blocks[block_id]
                    .side_effects
                    .pop()
                    .expect("new scalar load must be last"),
            );
            loaded
        } else {
            array_replacements.push(InputReplacement {
                project,
                view,
                view_ty: entry.graph.types[&view].clone(),
                elem_ty: output.elem_ty.clone(),
                resource,
            });
            view
        };
        replacements.push((project, value, resource));
        entry.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(resource),
            role: StorageRole::Input,
            elem_ty: output.elem_ty.clone(),
            size: output.size.clone(),
        });
    }
    retarget_input_metadata(&mut entry.graph, &array_replacements);
    for &(project, value, _) in &replacements {
        graph_ops::replace_all_references(&mut entry.graph, project, value);
        let value_ty = entry.graph.types[&value].clone();
        entry.graph.retype_node(project, value_ty);
    }
    let result_ty = entry.graph.types[&candidate.result].clone();
    let replacement_values = replacements.iter().map(|(_, value, _)| *value).collect::<Vec<_>>();
    let replacement_result = if replacement_values.len() == 1 && result_ty == output_specs[0].result_ty {
        replacement_values[0]
    } else {
        let replacement_ty = Type::Constructed(
            TypeName::Tuple(replacement_values.len()),
            replacement_values.iter().map(|value| entry.graph.types[value].clone()).collect(),
        );
        entry.graph.intern_pure(
            PureOp::Tuple(replacement_values.len()),
            replacement_values.into(),
            replacement_ty,
        )
    };
    graph_ops::replace_all_references(&mut entry.graph, candidate.result, replacement_result);
    let replacement_result_ty = entry.graph.types[&replacement_result].clone();
    entry.graph.retype_node(candidate.result, replacement_result_ty);
    for route in &mut entry.output_routes {
        if let Some((_, value, _)) =
            replacements.iter().find(|(project, _, _)| *project == route.source.value)
        {
            route.source.value = *value;
        }
    }
    entry.graph.skeleton.blocks[block_id].side_effects.remove(effect_index);
    for (offset, effect) in scalar_effects.into_iter().enumerate() {
        entry.graph.skeleton.blocks[block_id].side_effects.insert(effect_index + offset, effect);
    }
    for &(_, value, resource) in &replacements {
        if candidate.kind == MaterializationKind::Scalar {
            add_resource_read_for_value(&mut entry.graph, value, resource);
        }
    }
    super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph);

    inner.materializations.push(producer);
}

fn output_specs(
    graph: &EGraph,
    materialization: MaterializationKind,
    space: &super::types::SegSpace,
    kind: &SegOpKind,
    map_output_elem_types: &[Type<TypeName>],
    result_types: &[Type<TypeName>],
) -> Option<Vec<OutputSpec>> {
    let output_elem_types = match (materialization, kind) {
        (MaterializationKind::Scalar, SegOpKind::SegRed { .. }) => result_types.to_vec(),
        (_, SegOpKind::SegMap) => map_output_elem_types.to_vec(),
        (_, SegOpKind::SegScan { operators }) => map_output_elem_types
            .iter()
            .cloned()
            .chain(operators.iter().map(|operator| graph.types[&operator.neutral].clone()))
            .collect(),
        _ => return None,
    };
    (output_elem_types.len() == result_types.len()).then(|| {
        output_elem_types
            .into_iter()
            .zip(result_types)
            .map(|(elem_ty, result_ty)| OutputSpec {
                size: if materialization == MaterializationKind::Scalar {
                    LogicalSize::FixedBytes(crate::ssa::layout::type_byte_size(&elem_ty).unwrap_or(1) as u64)
                } else {
                    size_for_space(space, &elem_ty)
                },
                elem_ty,
                result_ty: result_ty.clone(),
            })
            .collect()
    })
}

fn compact_graphics_materialization(entry: &mut super::program::SemanticEntry) {
    let roots = entry
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| block.side_effects.iter().flat_map(|effect| effect.referenced_nodes()));
    let reachable = wyn_graph::reachable_set(roots, wyn_graph::WalkOrder::DepthFirst, |node, pending| {
        if let Some(definition) = entry.graph.nodes.get(node) {
            pending.extend(definition.children());
        }
    });
    let mut kept = reachable
        .into_iter()
        .filter_map(|node| match entry.graph.nodes.get(node) {
            Some(ENode::FuncParam { index }) => Some((*index, node)),
            _ => None,
        })
        .collect::<Vec<_>>();
    kept.sort_by_key(|(index, _)| *index);
    kept.dedup_by_key(|(index, _)| *index);
    let inputs = kept.iter().map(|(index, _)| entry.inputs[*index].clone()).collect();
    let params = kept.iter().map(|(index, _)| entry.params[*index].clone()).collect();
    let resource_inputs = kept.iter().map(|(index, _)| entry.resource_abi.inputs[*index]).collect();
    for (new_index, (_, node)) in kept.into_iter().enumerate() {
        if let Some(ENode::FuncParam { index }) = entry.graph.nodes.get_mut(node) {
            *index = new_index;
        }
    }
    entry.inputs = inputs;
    entry.params = params;
    entry.resource_abi.inputs = resource_inputs;
}

fn add_resource_read_for_value(graph: &mut EGraph, value: NodeId, resource: ResourceId) {
    let mut sites = Vec::new();
    for (block_id, block) in &graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            if matches!(effect.kind, SideEffectKind::Soac(EgirSoac::Seg { .. }))
                && effect.referenced_nodes().any(|node| depends_on(graph, node, value))
            {
                sites.push((block_id, index));
            }
        }
    }
    for (block, index) in sites {
        let SideEffectKind::Soac(EgirSoac::Seg { resources, .. }) =
            &mut graph.skeleton.blocks[block].side_effects[index].kind
        else {
            continue;
        };
        if !resources.iter().any(|access| access.resource.0 == resource) {
            resources.push(SegResourceAccess {
                resource: SemanticResourceRef(resource),
                access: SegResourceAccessKind::Read,
            });
            resources.sort_by_key(|access| access.resource);
        }
    }
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
    let mut result_retypes = Vec::new();
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            match &mut effect.kind {
                SideEffectKind::Soac(EgirSoac::Seg {
                    space,
                    kind,
                    input_array_types,
                    input_elem_types,
                    map_input_indices,
                    map_destinations,
                    acc_destinations,
                    result_types,
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
                    for (lane, destination) in map_destinations.iter().enumerate() {
                        if *destination == SoacDestination::InputBuffer {
                            if let Some(input) =
                                map_input_indices.get(lane).and_then(|inputs| inputs.first())
                            {
                                result_types[lane] = input_array_types[*input].clone();
                            }
                        }
                    }
                    for (lane, destination) in acc_destinations.iter().enumerate() {
                        if *destination == SoacDestination::InputBuffer {
                            if let Some(input) = kind
                                .operators()
                                .get(lane)
                                .and_then(|operator| operator.input_indices.first())
                            {
                                result_types[map_destinations.len() + lane] =
                                    input_array_types[*input].clone();
                            }
                        }
                    }
                    if let Some(result) = effect.result {
                        result_retypes.push((result, result_types.clone()));
                    }
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
    for (result, result_types) in result_retypes {
        let current = graph.types[&result].clone();
        let retyped = match current {
            Type::Constructed(TypeName::Tuple(_), _) => {
                Type::Constructed(TypeName::Tuple(result_types.len()), result_types)
            }
            _ if result_types.len() == 1 => result_types[0].clone(),
            _ => Type::Constructed(TypeName::Tuple(result_types.len()), result_types),
        };
        graph.retype_node(result, retyped);
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
            && inner.materializations.iter().all(|requirement| requirement.entry.name != name)
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
