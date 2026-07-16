//! Semantic residency planning for arrays and cross-dispatch scalars.
//!
//! The pass recognizes shared producers, runtime gathers, invariant scalar
//! reductions, and cost-eligible preludes of parallel operations after output
//! realization and semantic fusion. It records each decision as a typed
//! materialization plan, allocates its logical handoff resource, and rewires
//! consumers to explicit storage views or loads. Target lowering only chooses
//! and schedules the physical kernel recipe.

use std::collections::{HashMap, HashSet};

use polytype::Type;

use super::graph_ops;
use super::program::{
    CompilerResource, CompilerResourceKind, LogicalSize, MaterializationId, MaterializationKind,
    MaterializationRequirement, MaterializationSubstitution, OutputWriter, ResourceId,
    SemanticDependencyKind, SemanticOpId, SemanticProgram, SemanticResourceDecl, SemanticResourceRef,
};
use super::soac::{filter, screma};
use super::types::{
    EGraph, ENode, NodeId, PureOp, SegExtent, SegResourceAccess, SegResourceAccessKind, SideEffectKind,
    SideEffectSite, Soac, SoacDestination,
};
use crate::ast::TypeName;
use crate::interface::StorageRole;
use crate::ssa::types::ExecutionModel;
use crate::types::TypeExt;

struct MaterializationPlan {
    entry: usize,
    kind: MaterializationKind,
    source: MaterializationSource,
}

enum MaterializationSource {
    OperationResult(OperationResultMaterialization),
    ParallelPrelude(ParallelPreludeMaterialization),
}

struct OperationResultMaterialization {
    result: NodeId,
    producer: SemanticOpId,
    source_site: SideEffectSite,
    projected_site: SideEffectSite,
    projected_result: NodeId,
    projection: super::graph_projector::GraphProjection,
    layout: MaterializedValueLayout,
}

enum MaterializedValueLayout {
    /// One independently storage-layoutable resource per result component.
    /// Scalar, gather, and shared-array policies all use this layout.
    FixedArity(Vec<OutputSpec>),
    /// Variable-cardinality array represented by capacity storage plus a
    /// separately stored logical length.
    RuntimeArray {
        scratch: ResourceId,
        elem_ty: Type<TypeName>,
        result_ty: Type<TypeName>,
        size: LogicalSize,
    },
}

struct RuntimeArrayHandoff {
    data: ResourceId,
    length: ResourceId,
    elem_ty: Type<TypeName>,
    result_ty: Type<TypeName>,
    size: LogicalSize,
}

struct ParallelPrelude {
    root: NodeId,
    consumers: Vec<SemanticOpId>,
}

struct ParallelPreludeMaterialization {
    prelude: ParallelPrelude,
    insertion_site: SideEffectSite,
    recipe: super::graph_projector::ProjectedValueRecipe,
    elem_ty: Type<TypeName>,
    size: LogicalSize,
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
    select_in_place_destinations(inner);
    loop {
        super::semantic_graph::rebuild_dependencies(inner);
        let Some(plan) = next_materialization_plan(inner) else {
            break;
        };
        apply_materialization(inner, plan);
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

/// Resolve TLC's uniqueness-only candidates from the final semantic use graph.
/// Output realization and fusion have already run, so `InputBuffer` here is a
/// physical choice rather than a source-level prediction.
fn select_in_place_destinations(inner: &mut SemanticProgram) {
    for entry in &mut inner.entry_points {
        select_in_place_in_graph(&mut entry.graph);
    }
    for function in &mut inner.functions {
        select_in_place_in_graph(&mut function.graph);
    }
}

fn select_in_place_in_graph(graph: &mut EGraph) {
    // Multi-block liveness needs block-parameter substitution. Stay sound and
    // conservative until that representation is needed by a reuse candidate.
    if graph.skeleton.blocks.len() != 1 {
        clear_unique_input_candidates(graph);
        return;
    }
    let block_id = graph.skeleton.entry;
    let effect_count = graph.skeleton.blocks[block_id].side_effects.len();
    for effect_index in 0..effect_count {
        let (operands, map_inputs, operator_inputs, map_destinations, acc_destinations, filter_candidate) = {
            let effect = &graph.skeleton.blocks[block_id].side_effects[effect_index];
            match &effect.kind {
                SideEffectKind::Soac(Soac::Screma(op)) => (
                    effect.operand_nodes.to_vec(),
                    op.lanes().maps.iter().map(|map| map.input_indices.clone()).collect(),
                    op.operators()
                        .into_iter()
                        .map(|operator| operator.input_indices.clone())
                        .collect::<Vec<_>>(),
                    op.lanes().maps.iter().map(|map| map.destination).collect(),
                    op.operators().into_iter().map(|operator| operator.destination).collect(),
                    false,
                ),
                SideEffectKind::Soac(Soac::Filter(filter::Op {
                    state:
                        filter::SemanticState {
                            storage: filter::Output::Local { destination, .. },
                            ..
                        },
                    ..
                })) => (
                    effect.operand_nodes.to_vec(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    *destination == SoacDestination::UniqueInput,
                ),
                _ => continue,
            }
        };

        let mut input_use_counts = HashMap::<usize, usize>::new();
        for &input in map_inputs.iter().flatten().chain(operator_inputs.iter().flatten()) {
            *input_use_counts.entry(input.index()).or_default() += 1;
        }
        let mut claimed = HashSet::<usize>::new();
        let resolve = |input: usize, claimed: &mut HashSet<usize>| {
            input_use_counts.get(&input) == Some(&1)
                && claimed.insert(input)
                && operands.get(input).is_some_and(|&node| {
                    reusable_input_type(&graph.types[&node])
                        && input_is_dead_after(graph, block_id, effect_index, node)
                })
        };

        let new_maps: Vec<_> = map_destinations
            .iter()
            .enumerate()
            .map(|(lane, destination)| {
                if *destination != SoacDestination::UniqueInput {
                    return *destination;
                }
                map_inputs
                    .get(lane)
                    .and_then(|inputs| inputs.first())
                    .copied()
                    .filter(|input| resolve(input.index(), &mut claimed))
                    .map_or(SoacDestination::Fresh, |_| SoacDestination::InputBuffer)
            })
            .collect();
        let new_accs: Vec<_> = acc_destinations
            .iter()
            .enumerate()
            .map(|(operator, destination)| {
                if *destination != SoacDestination::UniqueInput {
                    return *destination;
                }
                operator_inputs
                    .get(operator)
                    .and_then(|inputs| inputs.first())
                    .copied()
                    .filter(|input| resolve(input.index(), &mut claimed))
                    .map_or(SoacDestination::Fresh, |_| SoacDestination::InputBuffer)
            })
            .collect();

        let filter_destination = filter_candidate.then(|| {
            if reusable_input_type(&graph.types[&operands[0]])
                && input_is_dead_after(graph, block_id, effect_index, operands[0])
            {
                SoacDestination::InputBuffer
            } else {
                SoacDestination::Fresh
            }
        });
        let effect = &mut graph.skeleton.blocks[block_id].side_effects[effect_index];
        match &mut effect.kind {
            SideEffectKind::Soac(Soac::Screma(op)) => {
                for (map, destination) in op.lanes_mut().maps.iter_mut().zip(new_maps) {
                    map.destination = destination;
                }
                for (operator, destination) in op.operators_mut().into_iter().zip(new_accs) {
                    operator.destination = destination;
                }
            }
            SideEffectKind::Soac(Soac::Filter(filter::Op {
                state:
                    filter::SemanticState {
                        storage: filter::Output::Local { destination, .. },
                        ..
                    },
                ..
            })) => {
                if let Some(resolved) = filter_destination {
                    *destination = resolved;
                }
            }
            _ => {}
        }
        retype_input_buffer_results(graph, block_id, effect_index);
    }
}

fn retype_input_buffer_results(
    graph: &mut EGraph,
    block: crate::ssa::framework::BlockId,
    effect_index: usize,
) {
    let effect = &graph.skeleton.blocks[block].side_effects[effect_index];
    let Some(result) = effect.result else {
        return;
    };
    let projections: Vec<_> = graph
        .nodes
        .iter()
        .filter_map(|(node, definition)| match definition {
            ENode::Pure {
                op: PureOp::Project { index },
                operands,
            } if operands.as_slice() == [result] => Some((node, *index as usize)),
            _ => None,
        })
        .collect();
    let (result_types, changed) = {
        let SideEffectKind::Soac(Soac::Screma(op)) = &effect.kind else {
            return;
        };
        let mut retyped = op.result_types();
        // Output realization may already have changed a projected field to a
        // storage view. Preserve those later EGIR decisions while changing
        // only fields whose uniqueness candidate became a physical reuse.
        for (projection, field) in &projections {
            retyped[*field] = graph.types[projection].clone();
        }
        let mut changed = false;
        for (lane, map) in op.lanes().maps.iter().enumerate() {
            if map.destination == SoacDestination::InputBuffer {
                if let Some(input) = map.input_indices.first() {
                    retyped[lane] = op.lanes().inputs[input.index()].array.clone();
                    changed = true;
                }
            }
        }
        for (operator_index, operator) in op.operators().into_iter().enumerate() {
            if operator.destination == SoacDestination::InputBuffer {
                if let Some(input) = operator.input_indices.first() {
                    retyped[op.lanes().maps.len() + operator_index] =
                        op.lanes().inputs[input.index()].array.clone();
                    changed = true;
                }
            }
        }
        (retyped, changed)
    };
    if !changed {
        return;
    }
    if let SideEffectKind::Soac(Soac::Screma(op)) =
        &mut graph.skeleton.blocks[block].side_effects[effect_index].kind
    {
        op.set_result_types(&result_types);
    }
    graph.retype_node(
        result,
        Type::Constructed(TypeName::Tuple(result_types.len()), result_types.clone()),
    );
    for (projection, field) in projections {
        if let Some(ty) = result_types.get(field) {
            graph.retype_node(projection, ty.clone());
        }
    }
}

fn clear_unique_input_candidates(graph: &mut EGraph) {
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            match &mut effect.kind {
                SideEffectKind::Soac(Soac::Screma(op)) => {
                    for map in &mut op.lanes_mut().maps {
                        if map.destination == SoacDestination::UniqueInput {
                            map.destination = SoacDestination::Fresh;
                        }
                    }
                    for operator in op.operators_mut() {
                        if operator.destination == SoacDestination::UniqueInput {
                            operator.destination = SoacDestination::Fresh;
                        }
                    }
                }
                SideEffectKind::Soac(Soac::Filter(filter::Op {
                    state:
                        filter::SemanticState {
                            storage: filter::Output::Local { destination, .. },
                            ..
                        },
                    ..
                })) if *destination == SoacDestination::UniqueInput => {
                    *destination = SoacDestination::Fresh;
                }
                _ => {}
            }
        }
    }
}

fn input_is_dead_after(
    graph: &EGraph,
    block: crate::ssa::framework::BlockId,
    index: usize,
    input: NodeId,
) -> bool {
    let body = &graph.skeleton.blocks[block];
    body.side_effects[index + 1..]
        .iter()
        .flat_map(|effect| effect.referenced_nodes())
        .chain(body.term.referenced_nodes())
        .all(|root| !node_reaches(graph, root, input))
}

fn node_reaches(graph: &EGraph, root: NodeId, target: NodeId) -> bool {
    wyn_graph::reaches_ordered(root, target, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        out.extend(graph.nodes[node].children());
    })
}

fn reusable_input_type(ty: &Type<TypeName>) -> bool {
    match ty.array_variant() {
        Some(Type::Constructed(TypeName::ArrayVariantVirtual, _)) => return false,
        Some(Type::Constructed(TypeName::ArrayVariantView, _)) => return true,
        _ => {}
    }
    let runtime_sized =
        ty.array_size().is_some_and(|size| !matches!(size, Type::Constructed(TypeName::Size(_), _)));
    !runtime_sized || crate::types::array_view_buffer(ty).is_some()
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
                if matches!(
                    &effect.kind,
                    SideEffectKind::Soac(Soac::Filter(filter::Op {
                        state: filter::SemanticState {
                            storage: filter::Output::Runtime {
                                length: filter::RuntimeLength::ViewOnly,
                                ..
                            },
                            ..
                        },
                        ..
                    }))
                ) {
                    assert!(
                        !has_parallel_consumer(entry, consumers.get(&id)),
                        "runtime-array producer {id:?} survived inside a parallel consumer"
                    );
                }
                if consumers.get(&id).map_or(0, HashSet::len) < 2 {
                    continue;
                }
                let SideEffectKind::Soac(Soac::Screma(screma::Op::Map {
                    lanes: screma::Lanes { maps, .. },
                    state:
                        screma::SemanticState::Segmented {
                            output_slots,
                            resources,
                            ..
                        },
                })) = &effect.kind
                else {
                    continue;
                };
                let internal_pure_map = output_slots.is_empty()
                    && maps.iter().all(|map| map.destination == SoacDestination::Fresh)
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

fn next_materialization_plan(inner: &SemanticProgram) -> Option<MaterializationPlan> {
    plan_operation_result(inner).or_else(|| plan_parallel_prelude(inner))
}

fn apply_materialization(inner: &mut SemanticProgram, plan: MaterializationPlan) {
    match plan.source {
        MaterializationSource::OperationResult(operation) => {
            materialize_operation_result(inner, plan.entry, plan.kind, operation);
        }
        MaterializationSource::ParallelPrelude(prelude) => {
            materialize_parallel_prelude(inner, plan.entry, prelude);
        }
    }
}

fn plan_operation_result(inner: &SemanticProgram) -> Option<MaterializationPlan> {
    let consumers = semantic_value_consumers(inner);
    for (entry_index, entry) in inner.entry_points.iter().enumerate() {
        for (block_id, block) in &entry.graph.skeleton.blocks {
            for (effect_index, effect) in block.side_effects.iter().enumerate() {
                let Some(result) = effect.result else {
                    continue;
                };
                let id = effect.required_semantic_id();
                let semantic_consumers = consumers.get(&id);
                let source_site = SideEffectSite {
                    block: block_id,
                    index: effect_index,
                };
                match &effect.kind {
                    SideEffectKind::Soac(Soac::Screma(op)) => {
                        let Some(kind) =
                            operation_result_residency(entry, op, result, source_site, semantic_consumers)
                        else {
                            continue;
                        };
                        if let Some(plan) =
                            operation_result_plan(entry_index, entry, op, result, id, source_site, kind)
                        {
                            return Some(plan);
                        }
                    }
                    SideEffectKind::Soac(Soac::Filter(op)) => {
                        if let Some(plan) = filter_runtime_array_plan(
                            entry_index,
                            entry,
                            op,
                            result,
                            id,
                            source_site,
                            semantic_consumers,
                        ) {
                            return Some(plan);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    None
}

fn filter_runtime_array_plan(
    entry_index: usize,
    entry: &super::program::SemanticEntry,
    op: &filter::Op<super::types::Semantic>,
    result: NodeId,
    producer: SemanticOpId,
    source_site: SideEffectSite,
    consumers: Option<&HashSet<SemanticOpId>>,
) -> Option<MaterializationPlan> {
    let filter::SemanticState {
        space,
        storage:
            filter::Output::Runtime {
                scratch,
                length: filter::RuntimeLength::ViewOnly,
            },
    } = &op.state
    else {
        return None;
    };
    if !has_parallel_consumer(entry, consumers) {
        return None;
    }
    let elem_ty = op.body.output_element_type().clone();
    crate::ssa::layout::storage_elem_stride(&elem_ty)?;
    let result_ty = entry.graph.types[&result].clone();
    let projection = super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
        .selected_operation_recipe(HashSet::from([source_site]))
        .ok()?;
    let projected_result = projection.node(result)?;
    let projected_site = projection.graph.side_effect_index().site(projected_result)?;
    Some(MaterializationPlan {
        entry: entry_index,
        kind: MaterializationKind::RuntimeArray,
        source: MaterializationSource::OperationResult(OperationResultMaterialization {
            result,
            producer,
            source_site,
            projected_site,
            projected_result,
            projection,
            layout: MaterializedValueLayout::RuntimeArray {
                scratch: scratch.0,
                size: size_for_space(space, &elem_ty),
                elem_ty,
                result_ty,
            },
        }),
    })
}

fn semantic_value_consumers(inner: &SemanticProgram) -> HashMap<SemanticOpId, HashSet<SemanticOpId>> {
    let mut consumers = HashMap::new();
    for dependency in &inner.semantic_dependencies {
        if dependency.kind == SemanticDependencyKind::Value {
            consumers.entry(dependency.producer).or_insert_with(HashSet::new).insert(dependency.consumer);
        }
    }
    consumers
}

fn operation_result_residency(
    entry: &super::program::SemanticEntry,
    op: &screma::Op<super::types::Semantic>,
    result: NodeId,
    site: SideEffectSite,
    consumers: Option<&HashSet<SemanticOpId>>,
) -> Option<MaterializationKind> {
    let screma::SemanticState::Segmented { resources, .. } = op.semantic_state() else {
        return None;
    };
    let cloneable = op.lanes().maps.iter().all(|map| map.destination == SoacDestination::Fresh)
        && op.operators().into_iter().all(|operator| operator.destination == SoacDestination::Fresh)
        && resources.iter().all(|resource| {
            resource.access == SegResourceAccessKind::Read
                || entry.resource_abi.outputs.iter().flatten().any(|output| *output == resource.resource.0)
        });
    let dependencies = dependency_effects(&entry.graph, site.block, site.index);
    let upstream =
        dependencies.iter().copied().filter(|index| *index != site.index).collect::<HashSet<_>>();
    if !cloneable || !dependencies_are_cloneable(&entry.graph, site.block, &upstream) {
        return None;
    }

    match op {
        screma::Op::Map { lanes, .. } if !lanes.maps.is_empty() => {
            array_result_residency(entry, result, site, consumers)
        }
        screma::Op::Scan { .. } => array_result_residency(entry, result, site, consumers),
        screma::Op::Reduce { operators, .. }
            if operators.rest.is_empty()
                && (has_segmented_screma_consumer(entry, consumers)
                    || !matches!(entry.execution_model, ExecutionModel::Compute { .. }))
                && scalar_result_is_used(&entry.graph, result, site.block, site.index)
                && invocation_invariant(entry, site.block, &dependencies) =>
        {
            Some(MaterializationKind::Scalar)
        }
        _ => None,
    }
}

fn array_result_residency(
    entry: &super::program::SemanticEntry,
    result: NodeId,
    site: SideEffectSite,
    consumers: Option<&HashSet<SemanticOpId>>,
) -> Option<MaterializationKind> {
    if consumers.map_or(0, HashSet::len) >= 2 {
        Some(MaterializationKind::SharedArray)
    } else if runtime_composite(&entry.graph.types[&result])
        && requires_array_residency(&entry.graph, result, site.block, site.index)
    {
        Some(MaterializationKind::Gather)
    } else {
        None
    }
}

fn operation_result_plan(
    entry_index: usize,
    entry: &super::program::SemanticEntry,
    op: &screma::Op<super::types::Semantic>,
    result: NodeId,
    producer: SemanticOpId,
    source_site: SideEffectSite,
    kind: MaterializationKind,
) -> Option<MaterializationPlan> {
    let screma::SemanticState::Segmented { space, .. } = op.semantic_state() else {
        return None;
    };
    let output_specs = output_specs(&entry.graph, kind, space, op)?;
    let projection = super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
        .selected_operation_recipe(HashSet::from([source_site]))
        .ok()?;
    let projected_result = projection.node(result)?;
    let projected_site = projection.graph.side_effect_index().site(projected_result)?;
    Some(MaterializationPlan {
        entry: entry_index,
        kind,
        source: MaterializationSource::OperationResult(OperationResultMaterialization {
            result,
            producer,
            source_site,
            projected_site,
            projected_result,
            projection,
            layout: MaterializedValueLayout::FixedArity(output_specs),
        }),
    })
}

fn plan_parallel_prelude(inner: &SemanticProgram) -> Option<MaterializationPlan> {
    for (entry_index, entry) in inner.entry_points.iter().enumerate() {
        let dependencies = super::semantic_graph::SemanticGraph::with_operation_captures(
            &inner.semantic_dependencies,
            &entry.graph,
        );
        for prelude in parallel_preludes(entry, &dependencies) {
            let ty = &entry.graph.types[&prelude.root];
            let Some(stride) = crate::ssa::layout::storage_elem_stride(ty) else {
                continue;
            };
            if ty.is_array() {
                continue;
            }
            let Some(consumer_sites) = operation_sites(&dependencies, &prelude.consumers) else {
                continue;
            };
            let Some(consumer_block) = consumer_sites.first().map(|site| site.block) else {
                continue;
            };
            if consumer_sites.iter().any(|site| site.block != consumer_block)
                || !consumer_sites.iter().all(|site| supports_parallel_prefix_consumer(entry, *site))
            {
                continue;
            }
            let consumer_site_set = consumer_sites.iter().copied().collect::<HashSet<_>>();
            if !source_is_observed_only_by(entry, prelude.root, &consumer_site_set) {
                continue;
            }
            let Some(insertion_site) = consumer_sites.iter().min_by_key(|site| site.index).copied() else {
                continue;
            };
            let Ok(recipe) =
                super::graph_projector::GraphProjector::new(&entry.graph, &entry.control_headers)
                    .captured_value_recipe(prelude.root, insertion_site)
            else {
                continue;
            };
            let Some(analysis) = super::residency_cost::analyze_prelude(inner, entry, &recipe) else {
                continue;
            };
            let invocations = launched_consumer_invocations(entry, &dependencies, &prelude.consumers);
            if !super::residency_cost::materialization_is_profitable(analysis.cost, invocations) {
                continue;
            }
            return Some(MaterializationPlan {
                entry: entry_index,
                kind: MaterializationKind::Scalar,
                source: MaterializationSource::ParallelPrelude(ParallelPreludeMaterialization {
                    prelude,
                    insertion_site,
                    recipe,
                    elem_ty: ty.clone(),
                    size: LogicalSize::FixedBytes(u64::from(stride)),
                }),
            });
        }
    }
    None
}

fn parallel_preludes(
    entry: &super::program::SemanticEntry,
    dependencies: &super::semantic_graph::SemanticGraph,
) -> Vec<ParallelPrelude> {
    let mut preludes = Vec::<ParallelPrelude>::new();
    let mut by_root = HashMap::<NodeId, usize>::new();
    for operation in dependencies.operations() {
        let Some(site) = dependencies.operation_site(&operation) else {
            continue;
        };
        let SideEffectKind::Soac(soac) =
            &entry.graph.skeleton.blocks[site.block].side_effects[site.index].kind
        else {
            continue;
        };
        if soac.scheduling_space().is_none() {
            continue;
        }
        for root in dependencies.operation_captures(&operation) {
            if let Some(index) = by_root.get(&root).copied() {
                preludes[index].consumers.push(operation);
            } else {
                by_root.insert(root, preludes.len());
                preludes.push(ParallelPrelude {
                    root,
                    consumers: vec![operation],
                });
            }
        }
    }
    preludes
}

fn operation_sites(
    dependencies: &super::semantic_graph::SemanticGraph,
    operations: &[SemanticOpId],
) -> Option<Vec<SideEffectSite>> {
    operations.iter().map(|operation| dependencies.operation_site(operation)).collect()
}

fn supports_parallel_prefix_consumer(entry: &super::program::SemanticEntry, site: SideEffectSite) -> bool {
    matches!(
        &entry.graph.skeleton.blocks[site.block].side_effects[site.index].kind,
        SideEffectKind::Soac(Soac::Screma(screma::Op::Map {
            lanes,
            state: screma::SemanticState::Segmented { .. },
        })) if !lanes.maps.is_empty()
    )
}

fn source_is_observed_only_by(
    entry: &super::program::SemanticEntry,
    root: NodeId,
    consumers: &HashSet<SideEffectSite>,
) -> bool {
    for (block_id, block) in &entry.graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            if effect.referenced_nodes().any(|reference| pure_depends_on(&entry.graph, reference, root))
                && !consumers.contains(&SideEffectSite {
                    block: block_id,
                    index,
                })
            {
                return false;
            }
        }
        if block
            .term
            .referenced_nodes()
            .into_iter()
            .any(|reference| pure_depends_on(&entry.graph, reference, root))
        {
            return false;
        }
    }
    true
}

fn pure_depends_on(graph: &EGraph, root: NodeId, target: NodeId) -> bool {
    wyn_graph::reaches_ordered(
        root,
        target,
        wyn_graph::WalkOrder::DepthFirst,
        |node, out| match &graph.nodes[node] {
            ENode::Pure { operands, .. } => out.extend(operands.iter().copied()),
            ENode::Union { left, right } => out.extend([*left, *right]),
            ENode::FuncParam { .. }
            | ENode::BlockParam { .. }
            | ENode::Constant(_)
            | ENode::SideEffectResult => {}
        },
    )
}

fn launched_consumer_invocations(
    entry: &super::program::SemanticEntry,
    dependencies: &super::semantic_graph::SemanticGraph,
    consumers: &[SemanticOpId],
) -> u64 {
    let workgroup = match entry.execution_model {
        ExecutionModel::Compute { local_size } => u64::from(local_size.0)
            .saturating_mul(u64::from(local_size.1))
            .saturating_mul(u64::from(local_size.2))
            .max(1),
        ExecutionModel::Vertex | ExecutionModel::Fragment => 1,
    };
    consumers.iter().fold(0u64, |total, consumer| {
        let Some(site) = dependencies.operation_site(consumer) else {
            return total;
        };
        let SideEffectKind::Soac(soac) =
            &entry.graph.skeleton.blocks[site.block].side_effects[site.index].kind
        else {
            return total;
        };
        let Some(space) = soac.scheduling_space() else {
            return total;
        };
        let logical = space.dims.iter().try_fold(1u64, |count, extent| match extent {
            SegExtent::Fixed(length) => count.checked_mul(u64::from(*length)),
            _ => None,
        });
        let launched = logical.map_or(workgroup, |count| count.div_ceil(workgroup) * workgroup);
        total.saturating_add(launched)
    })
}

fn has_parallel_consumer(
    entry: &super::program::SemanticEntry,
    consumers: Option<&HashSet<SemanticOpId>>,
) -> bool {
    has_matching_consumer(entry, consumers, |soac| soac.scheduling_space().is_some())
}

fn has_segmented_screma_consumer(
    entry: &super::program::SemanticEntry,
    consumers: Option<&HashSet<SemanticOpId>>,
) -> bool {
    has_matching_consumer(entry, consumers, |soac| {
        matches!(
            soac,
            Soac::Screma(op)
                if matches!(op.semantic_state(), screma::SemanticState::Segmented { .. })
        )
    })
}

fn has_matching_consumer(
    entry: &super::program::SemanticEntry,
    consumers: Option<&HashSet<SemanticOpId>>,
    mut supports: impl FnMut(&Soac<super::types::Semantic>) -> bool,
) -> bool {
    let Some(consumers) = consumers else {
        return false;
    };
    entry.graph.skeleton.blocks.iter().flat_map(|(_, block)| &block.side_effects).any(|effect| {
        matches!(&effect.kind, SideEffectKind::Soac(soac) if supports(soac))
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
    graph_ops::value_producer_closure(graph, [root]).nodes.contains(&target)
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
    let reachable = graph_ops::execution_value_producer_closure(&entry.graph, roots).nodes;
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
            SideEffectKind::Soac(Soac::Screma(op))
                if matches!(op.semantic_state(), screma::SemanticState::Segmented { output_slots, resources, .. }
                    if output_slots.is_empty()
                        && op.lanes().maps.iter().all(|map| map.destination == SoacDestination::Fresh)
                        && op.operators().into_iter().all(|operator| operator.destination == SoacDestination::Fresh)
                        && resources.iter().all(|resource| resource.access == SegResourceAccessKind::Read))
        )
    })
}

fn materialize_operation_result(
    inner: &mut SemanticProgram,
    entry_index: usize,
    kind: MaterializationKind,
    operation: OperationResultMaterialization,
) {
    if matches!(&operation.layout, MaterializedValueLayout::RuntimeArray { .. }) {
        materialize_runtime_array_result(inner, entry_index, operation);
        return;
    }
    let OperationResultMaterialization {
        result,
        producer: producer_id,
        source_site,
        projected_site,
        projected_result,
        projection,
        layout,
    } = operation;
    let MaterializedValueLayout::FixedArity(output_specs) = layout else {
        unreachable!("runtime-array layouts are handled before fixed-arity materialization")
    };
    let entry = &inner.entry_points[entry_index];
    let source_output_resources =
        entry.resource_abi.outputs.iter().flatten().copied().collect::<HashSet<_>>();
    let (block_id, effect_index) = (source_site.block, source_site.index);
    let producer_dependencies = dependency_effects(&entry.graph, block_id, effect_index);
    let dependency_resources = dependency_resources(entry, block_id, &producer_dependencies);

    let producer_storage = resource_declarations_for(entry, &dependency_resources);
    let execution_model = match entry.execution_model {
        ExecutionModel::Compute { local_size } => ExecutionModel::Compute { local_size },
        ExecutionModel::Vertex | ExecutionModel::Fragment => ExecutionModel::Compute {
            local_size: (64, 1, 1),
        },
    };
    let name_suffix = match kind {
        MaterializationKind::SharedArray => "materialize_shared",
        MaterializationKind::Gather => "gather_materialize",
        MaterializationKind::Scalar => "prepass_scalar",
        MaterializationKind::RuntimeArray => {
            unreachable!("runtime arrays use their layout-specific materialization path")
        }
    };
    let compact_inputs = !matches!(entry.execution_model, ExecutionModel::Compute { .. });
    let producer_entry = projected_materialization_entry(
        inner,
        entry,
        name_suffix,
        execution_model,
        producer_storage,
        projection,
    );
    let mut producer = MaterializationRequirement {
        id: MaterializationId(inner.materializations.len() as u32),
        kind,
        producer: Some(producer_id),
        entry: producer_entry,
        substitutions: Vec::new(),
    };
    if compact_inputs {
        compact_entry_interface(&mut producer.entry);
    }
    let producer_owner = producer_id;
    let resource_kind = match kind {
        MaterializationKind::SharedArray => CompilerResourceKind::MultiConsumerArray,
        MaterializationKind::Gather => CompilerResourceKind::GatherHandoff,
        MaterializationKind::Scalar => CompilerResourceKind::ScalarHandoff,
        MaterializationKind::RuntimeArray => {
            unreachable!("runtime arrays reuse their existing data resource")
        }
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
    configure_operation_materialization(
        &mut producer,
        kind,
        projected_site,
        projected_result,
        &output_resources,
        &output_specs,
        &source_output_resources,
    );

    rewrite_materialized_operation_source(
        &mut inner.entry_points[entry_index],
        kind,
        result,
        source_site,
        &output_resources,
        &output_specs,
    );
    inner.materializations.push(producer);
}

fn materialize_runtime_array_result(
    inner: &mut SemanticProgram,
    entry_index: usize,
    operation: OperationResultMaterialization,
) {
    let OperationResultMaterialization {
        result,
        producer: producer_id,
        source_site,
        projected_site,
        projection,
        layout,
        ..
    } = operation;
    let MaterializedValueLayout::RuntimeArray {
        scratch,
        elem_ty,
        result_ty,
        size,
    } = layout
    else {
        unreachable!("runtime-array materializer received a direct layout")
    };
    let entry = &inner.entry_points[entry_index];
    let dependencies = dependency_effects(&entry.graph, source_site.block, source_site.index);
    let dependency_resources = dependency_resources(entry, source_site.block, &dependencies);
    let producer_storage = resource_declarations_for(entry, &dependency_resources);
    let execution_model = match entry.execution_model {
        ExecutionModel::Compute { local_size } => ExecutionModel::Compute { local_size },
        ExecutionModel::Vertex | ExecutionModel::Fragment => ExecutionModel::Compute {
            local_size: (64, 1, 1),
        },
    };
    let producer_entry = projected_materialization_entry(
        inner,
        entry,
        "materialize_filter",
        execution_model,
        producer_storage,
        projection,
    );
    let id = MaterializationId(inner.materializations.len() as u32);
    let length = inner.alloc_compiler_resource(
        CompilerResource::new(CompilerResourceKind::FilterLenCell, Some(producer_id), 1),
        Type::Constructed(TypeName::UInt(32), vec![]),
        LogicalSize::FixedBytes(4),
    );
    let handoff = RuntimeArrayHandoff {
        data: scratch,
        length,
        elem_ty,
        result_ty,
        size,
    };
    let mut producer = MaterializationRequirement {
        id,
        kind: MaterializationKind::RuntimeArray,
        producer: Some(producer_id),
        entry: producer_entry,
        substitutions: [handoff.data, handoff.length]
            .into_iter()
            .map(|resource| MaterializationSubstitution {
                resource: SemanticResourceRef(resource),
                consumers: Vec::new(),
            })
            .collect(),
    };
    set_resource_declaration(
        &mut producer.entry,
        handoff.data,
        StorageRole::Output,
        &handoff.elem_ty,
        &handoff.size,
    );
    set_resource_declaration(
        &mut producer.entry,
        handoff.length,
        StorageRole::Output,
        &Type::Constructed(TypeName::UInt(32), vec![]),
        &LogicalSize::FixedBytes(4),
    );
    let effect =
        &mut producer.entry.graph.skeleton.blocks[projected_site.block].side_effects[projected_site.index];
    let SideEffectKind::Soac(Soac::Filter(filter::Op {
        state: filter::SemanticState { storage, .. },
        ..
    })) = &mut effect.kind
    else {
        return;
    };
    *storage = filter::Output::Runtime {
        scratch: SemanticResourceRef(handoff.data),
        length: filter::RuntimeLength::Stored(SemanticResourceRef(handoff.length)),
    };
    compact_entry_interface(&mut producer.entry);

    rewrite_runtime_array_source(
        &mut inner.entry_points[entry_index],
        result,
        source_site,
        &handoff,
    );
    inner.materializations.push(producer);
}

fn rewrite_runtime_array_source(
    entry: &mut super::program::SemanticEntry,
    result: NodeId,
    source_site: SideEffectSite,
    handoff: &RuntimeArrayHandoff,
) {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    set_resource_declaration(
        entry,
        handoff.data,
        StorageRole::Input,
        &handoff.elem_ty,
        &handoff.size,
    );
    set_resource_declaration(
        entry,
        handoff.length,
        StorageRole::Input,
        &u32_ty,
        &LogicalSize::FixedBytes(4),
    );
    let length_view =
        graph_ops::intern_resource_view(&mut entry.graph, handoff.length, u32_ty.clone(), None);
    let (survivor_count, load_effect) =
        detached_scalar_handoff_load(&mut entry.graph, length_view, &u32_ty);
    let zero = graph_ops::intern_u32(&mut entry.graph, 0, None);
    let view = graph_ops::intern_chunked_resource_view(
        &mut entry.graph,
        handoff.data,
        zero,
        survivor_count,
        handoff.elem_ty.clone(),
        None,
    );
    let view_ty = entry.graph.types[&view].clone();
    retarget_input_metadata(
        &mut entry.graph,
        &[InputReplacement {
            project: result,
            view,
            view_ty,
            elem_ty: handoff.elem_ty.clone(),
            resource: handoff.data,
        }],
    );
    graph_ops::replace_all_references(&mut entry.graph, result, view);
    entry.graph.retype_node(result, handoff.result_ty.clone());
    for route in &mut entry.output_routes {
        if route.source.value == result {
            route.source.value = view;
        }
    }
    let block = &mut entry.graph.skeleton.blocks[source_site.block];
    block.side_effects.remove(source_site.index);
    block.side_effects.insert(source_site.index, load_effect);
    refresh_resource_reads_for_values(&mut entry.graph, &[survivor_count, view]);
    super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph);
    compact_entry_interface(entry);
}

fn configure_operation_materialization(
    producer: &mut MaterializationRequirement,
    kind: MaterializationKind,
    producer_site: SideEffectSite,
    producer_result: NodeId,
    output_resources: &[ResourceId],
    output_specs: &[OutputSpec],
    source_output_resources: &HashSet<ResourceId>,
) {
    let mut output_views = Vec::new();
    for (&resource, output) in output_resources.iter().zip(output_specs) {
        output_views.push(declare_resource_view(
            &mut producer.entry,
            resource,
            StorageRole::Output,
            &output.elem_ty,
            &output.size,
        ));
        producer.substitutions.push(MaterializationSubstitution {
            resource: SemanticResourceRef(resource),
            consumers: Vec::new(),
        });
    }

    configure_materialized_soac(
        &mut producer.entry.graph,
        kind,
        producer_site,
        &output_views,
        output_resources,
        source_output_resources,
    );
    configure_materialized_result(
        &mut producer.entry.graph,
        kind,
        producer_result,
        &output_views,
        output_specs,
    );
}

fn configure_materialized_soac(
    graph: &mut EGraph,
    kind: MaterializationKind,
    producer_site: SideEffectSite,
    output_views: &[NodeId],
    output_resources: &[ResourceId],
    source_output_resources: &HashSet<ResourceId>,
) {
    let producer_effect = &mut graph.skeleton.blocks[producer_site.block].side_effects[producer_site.index];
    let SideEffectKind::Soac(Soac::Screma(op)) = &mut producer_effect.kind else {
        return;
    };
    let screma::SemanticState::Segmented {
        placement,
        output_slots,
        resources,
        ..
    } = op.semantic_state_mut()
    else {
        return;
    };
    *placement = screma::Placement::Kernel;
    *output_slots = Vec::new();
    resources.retain(|access| {
        access.access == SegResourceAccessKind::Read
            || !source_output_resources.contains(&access.resource.0)
    });
    if kind == MaterializationKind::Scalar {
        return;
    }
    for map in &mut op.lanes_mut().maps {
        map.destination = SoacDestination::OutputView;
    }
    for operator in op.operators_mut() {
        operator.destination = SoacDestination::OutputView;
    }
    producer_effect.operand_nodes.extend(output_views.iter().copied());
    let screma::SemanticState::Segmented { resources, .. } = op.semantic_state_mut() else {
        return;
    };
    resources.extend(output_resources.iter().map(|resource| SegResourceAccess {
        resource: SemanticResourceRef(*resource),
        access: SegResourceAccessKind::Write,
    }));
    resources.sort_by_key(|access| access.resource);
}

fn configure_materialized_result(
    graph: &mut EGraph,
    kind: MaterializationKind,
    result: NodeId,
    output_views: &[NodeId],
    output_specs: &[OutputSpec],
) {
    if kind == MaterializationKind::Scalar {
        if let Some((&output_view, output)) = output_views.first().zip(output_specs.first()) {
            let result_ty = graph.types[&result].clone();
            let value = if result_ty == output.result_ty {
                result
            } else {
                graph.intern_pure(
                    PureOp::Project { index: 0 },
                    smallvec::smallvec![result],
                    output.result_ty.clone(),
                    None,
                )
            };
            emit_scalar_handoff_store(graph, graph.skeleton.entry, output_view, value, &output.elem_ty);
        }
        return;
    }

    let original_ty = graph.types[&result].clone();
    let view_types = output_views.iter().map(|view| graph.types[view].clone()).collect::<Vec<_>>();
    let preserve_single =
        output_specs.first().is_some_and(|output| view_types.len() == 1 && original_ty == output.result_ty);
    let materialized_ty = if preserve_single {
        view_types[0].clone()
    } else {
        Type::Constructed(TypeName::Tuple(view_types.len()), view_types)
    };
    graph.retype_node(result, materialized_ty);
}

fn rewrite_materialized_operation_source(
    entry: &mut super::program::SemanticEntry,
    kind: MaterializationKind,
    result: NodeId,
    producer_site: SideEffectSite,
    output_resources: &[ResourceId],
    output_specs: &[OutputSpec],
) {
    let (block_id, effect_index) = (producer_site.block, producer_site.index);
    let mut array_replacements = Vec::new();
    let mut replacements = Vec::new();
    let mut scalar_effects = Vec::new();
    for (lane, (&resource, output)) in output_resources.iter().zip(output_specs).enumerate() {
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
                } if *index as usize == lane && operands.first() == Some(&result) => Some(node),
                _ => None,
            })
            .unwrap_or_else(|| {
                entry.graph.intern_pure(
                    PureOp::Project { index: lane as u32 },
                    smallvec::smallvec![result],
                    output.result_ty.clone(),
                    None,
                )
            });
        let value = if kind == MaterializationKind::Scalar {
            let (loaded, load_effect) =
                detached_scalar_handoff_load(&mut entry.graph, view, &output.elem_ty);
            scalar_effects.push(load_effect);
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
    let result_ty = entry.graph.types[&result].clone();
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
            None,
        )
    };
    graph_ops::replace_all_references(&mut entry.graph, result, replacement_result);
    let replacement_result_ty = entry.graph.types[&replacement_result].clone();
    entry.graph.retype_node(result, replacement_result_ty);
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
    if kind == MaterializationKind::Scalar {
        let loaded_values = replacements.iter().map(|(_, value, _)| *value).collect::<Vec<_>>();
        refresh_resource_reads_for_values(&mut entry.graph, &loaded_values);
    }
    super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph);
}

fn materialize_parallel_prelude(
    inner: &mut SemanticProgram,
    entry_index: usize,
    prelude: ParallelPreludeMaterialization,
) {
    let ParallelPreludeMaterialization {
        prelude,
        insertion_site,
        recipe,
        elem_ty,
        size,
    } = prelude;
    let root = prelude.root;
    let super::graph_projector::ProjectedValueRecipe {
        projection,
        value: projected_root,
        result_block,
        source,
    } = recipe;
    let producer_effects = projection.source_effects().clone();
    let producer_entry = {
        let entry = &inner.entry_points[entry_index];
        projected_materialization_entry(
            inner,
            entry,
            "prepass_scalar",
            ExecutionModel::Compute {
                local_size: (1, 1, 1),
            },
            producer_resources_for_graph(entry, &projection.graph, projected_root),
            projection,
        )
    };
    let id = MaterializationId(inner.materializations.len() as u32);
    let resource = inner.alloc_compiler_resource(
        CompilerResource::new(CompilerResourceKind::ScalarHandoff, None, 0),
        elem_ty.clone(),
        size.clone(),
    );
    let mut producer = MaterializationRequirement {
        id,
        kind: MaterializationKind::Scalar,
        producer: None,
        entry: producer_entry,
        substitutions: vec![MaterializationSubstitution {
            resource: SemanticResourceRef(resource),
            consumers: Vec::new(),
        }],
    };
    let output_view = declare_resource_view(
        &mut producer.entry,
        resource,
        StorageRole::Output,
        &elem_ty,
        &size,
    );
    emit_scalar_handoff_store(
        &mut producer.entry.graph,
        result_block,
        output_view,
        projected_root,
        &elem_ty,
    );
    compact_entry_interface(&mut producer.entry);

    let entry = &mut inner.entry_points[entry_index];
    let view = declare_resource_view(entry, resource, StorageRole::Input, &elem_ty, &size);
    let (loaded, load_effect) = detached_scalar_handoff_load(&mut entry.graph, view, &elem_ty);
    graph_ops::replace_all_references(&mut entry.graph, root, loaded);
    match source {
        super::graph_projector::ValueRecipeSource::EntryBlock => {
            replace_prelude_effects_with_load(entry, &producer_effects, insertion_site, load_effect);
        }
        super::graph_projector::ValueRecipeSource::StructuredPrefix { continuation } => {
            replace_structured_prefix_with_load(entry, &producer_effects, continuation, loaded, load_effect)
        }
    }
    refresh_resource_reads_for_values(&mut entry.graph, &[loaded]);
    super::semantic_opt::eliminate_dead_seg_ops_in_graph(&mut entry.graph);

    inner.materializations.push(producer);
}

fn projected_materialization_entry(
    program: &SemanticProgram,
    source: &super::program::SemanticEntry,
    name_suffix: &str,
    execution_model: ExecutionModel,
    resource_declarations: Vec<SemanticResourceDecl>,
    projection: super::graph_projector::GraphProjection,
) -> super::program::SemanticEntry {
    let aliases = projection.remap_aliases(&source.aliases);
    super::program::SemanticEntry {
        name: fresh_entry_name(program, &format!("{}_{name_suffix}", source.name)),
        span: source.span,
        execution_model,
        inputs: source.inputs.clone(),
        outputs: Vec::new(),
        resource_abi: super::program::EntryResourceAbi {
            inputs: source.resource_abi.inputs.clone(),
            outputs: Vec::new(),
        },
        resource_declarations,
        params: source.params.clone(),
        return_ty: Type::Constructed(TypeName::Unit, vec![]),
        graph: projection.graph,
        control_headers: projection.control_headers,
        aliases,
        output_routes: Vec::new(),
    }
}

fn producer_resources_for_graph(
    entry: &super::program::SemanticEntry,
    graph: &EGraph,
    result: NodeId,
) -> Vec<SemanticResourceDecl> {
    let reachable = graph_ops::execution_value_producer_closure(graph, [result]).nodes;
    let resources = resources_referenced_by_nodes(entry, graph, reachable);
    resource_declarations_for(entry, &resources)
}

fn resources_referenced_by_nodes(
    entry: &super::program::SemanticEntry,
    graph: &EGraph,
    nodes: impl IntoIterator<Item = NodeId>,
) -> HashSet<ResourceId> {
    let mut resources = HashSet::new();
    for node in nodes {
        if let Some(resource) = graph_ops::extract_storage_view_source(graph, node) {
            resources.insert(resource.0);
        }
        if let Some(ENode::FuncParam { index }) = graph.nodes.get(node) {
            resources.extend(entry.resource_abi.inputs.get(*index).copied().flatten());
        }
    }
    resources
}

fn resource_declarations_for(
    entry: &super::program::SemanticEntry,
    resources: &HashSet<ResourceId>,
) -> Vec<SemanticResourceDecl> {
    entry
        .resource_declarations
        .iter()
        .filter(|declaration| resources.contains(&declaration.resource.0))
        .cloned()
        .collect()
}

fn declare_resource_view(
    entry: &mut super::program::SemanticEntry,
    resource: ResourceId,
    role: StorageRole,
    elem_ty: &Type<TypeName>,
    size: &LogicalSize,
) -> NodeId {
    let view = graph_ops::intern_resource_view(&mut entry.graph, resource, elem_ty.clone(), None);
    set_resource_declaration(entry, resource, role, elem_ty, size);
    view
}

fn set_resource_declaration(
    entry: &mut super::program::SemanticEntry,
    resource: ResourceId,
    role: StorageRole,
    elem_ty: &Type<TypeName>,
    size: &LogicalSize,
) {
    if let Some(declaration) =
        entry.resource_declarations.iter_mut().find(|declaration| declaration.resource.0 == resource)
    {
        declaration.role = role;
        declaration.elem_ty = elem_ty.clone();
        declaration.size = size.clone();
    } else {
        entry.resource_declarations.push(SemanticResourceDecl {
            resource: SemanticResourceRef(resource),
            role,
            elem_ty: elem_ty.clone(),
            size: size.clone(),
        });
    }
}

fn emit_scalar_handoff_store(
    graph: &mut EGraph,
    block: crate::ssa::framework::BlockId,
    output_view: NodeId,
    value: NodeId,
    elem_ty: &Type<TypeName>,
) {
    let zero = graph_ops::intern_u32(graph, 0, None);
    let mut next_effect = graph_ops::next_effect_token(graph);
    graph_ops::emit_storage_store(
        graph,
        block,
        output_view,
        zero,
        value,
        elem_ty.clone(),
        &mut next_effect,
        None,
    );
}

fn detached_scalar_handoff_load(
    graph: &mut EGraph,
    view: NodeId,
    elem_ty: &Type<TypeName>,
) -> (NodeId, super::types::SideEffect) {
    let zero = graph_ops::intern_u32(graph, 0, None);
    let place = graph.intern_pure(
        PureOp::ViewIndex,
        smallvec::smallvec![view, zero],
        elem_ty.clone(),
        None,
    );
    let mut next_effect = graph_ops::next_effect_token(graph);
    graph_ops::detached_load(graph, place, elem_ty.clone(), &mut next_effect, None)
}

fn replace_prelude_effects_with_load(
    entry: &mut super::program::SemanticEntry,
    producer_effects: &HashSet<SideEffectSite>,
    insertion_site: SideEffectSite,
    load_effect: super::types::SideEffect,
) {
    let mut removed = producer_effects.iter().map(|site| site.index).collect::<Vec<_>>();
    removed.sort_unstable();
    removed.dedup();
    let removed_before_consumer = removed.iter().filter(|index| **index < insertion_site.index).count();
    for index in removed.iter().rev() {
        entry.graph.skeleton.blocks[insertion_site.block].side_effects.remove(*index);
    }
    entry.graph.skeleton.blocks[insertion_site.block]
        .side_effects
        .insert(insertion_site.index - removed_before_consumer, load_effect);
}

fn replace_structured_prefix_with_load(
    entry: &mut super::program::SemanticEntry,
    producer_effects: &HashSet<SideEffectSite>,
    continuation: crate::ssa::framework::BlockId,
    loaded: NodeId,
    load_effect: super::types::SideEffect,
) {
    remove_effect_sites(&mut entry.graph, producer_effects);
    let source_entry = entry.graph.skeleton.entry;
    entry.graph.skeleton.blocks[source_entry].side_effects.push(load_effect);
    entry.graph.skeleton.blocks[source_entry].term = super::types::SkeletonTerminator::Branch {
        target: continuation,
        args: vec![loaded],
    };
    let aliases = super::skel_opt::run_one_body(&mut entry.graph);
    entry.aliases.extend(aliases);
    retain_live_control_headers(entry);
}

fn remove_effect_sites(graph: &mut EGraph, effects: &HashSet<SideEffectSite>) {
    let mut by_block = HashMap::<crate::ssa::framework::BlockId, Vec<usize>>::new();
    for site in effects {
        by_block.entry(site.block).or_default().push(site.index);
    }
    for (block, mut indices) in by_block {
        indices.sort_unstable();
        indices.dedup();
        for index in indices.into_iter().rev() {
            graph.skeleton.blocks[block].side_effects.remove(index);
        }
    }
}

fn retain_live_control_headers(entry: &mut super::program::SemanticEntry) {
    let blocks = &entry.graph.skeleton.blocks;
    entry.control_headers.retain(|header, control| {
        if !blocks.contains_key(*header)
            || !matches!(
                blocks[*header].term,
                super::types::SkeletonTerminator::CondBranch { .. }
            )
        {
            return false;
        }
        match control {
            crate::ssa::types::ControlHeader::Loop {
                merge,
                continue_block,
            } => blocks.contains_key(*merge) && blocks.contains_key(*continue_block),
            crate::ssa::types::ControlHeader::Selection { merge } => blocks.contains_key(*merge),
        }
    });
}

fn output_specs(
    graph: &EGraph,
    materialization: MaterializationKind,
    space: &super::types::SegSpace,
    op: &screma::Op<super::types::Semantic>,
) -> Option<Vec<OutputSpec>> {
    let result_types = op.result_types();
    let output_elem_types = match (materialization, op) {
        (MaterializationKind::Scalar, screma::Op::Reduce { .. }) => result_types.clone(),
        (_, screma::Op::Map { lanes, .. }) => {
            lanes.maps.iter().map(|map| map.output_element_type.clone()).collect()
        }
        (_, screma::Op::Scan { lanes, operators, .. }) => lanes
            .maps
            .iter()
            .map(|map| map.output_element_type.clone())
            .chain(
                std::iter::once(&operators.first)
                    .chain(&operators.rest)
                    .map(|operator| graph.types[&operator.neutral].clone()),
            )
            .collect(),
        _ => return None,
    };
    (output_elem_types.len() == result_types.len()).then(|| {
        output_elem_types
            .into_iter()
            .zip(&result_types)
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

fn compact_entry_interface(entry: &mut super::program::SemanticEntry) {
    compact_entry_inputs(entry);
    let mut used_resources = entry
        .resource_abi
        .inputs
        .iter()
        .chain(&entry.resource_abi.outputs)
        .copied()
        .flatten()
        .collect::<HashSet<_>>();
    for (_, block) in &entry.graph.skeleton.blocks {
        for effect in &block.side_effects {
            used_resources.extend(
                super::semantic_graph::read_resources(&entry.graph, effect)
                    .into_iter()
                    .map(|access| access.resource.0),
            );
            if let SideEffectKind::Soac(Soac::Screma(op)) = &effect.kind {
                if let screma::SemanticState::Segmented { resources, .. } = op.semantic_state() {
                    used_resources.extend(resources.iter().map(|access| access.resource.0));
                }
            }
        }
    }
    entry.resource_declarations.retain(|declaration| {
        declaration.role != StorageRole::Input || used_resources.contains(&declaration.resource.0)
    });
}

fn compact_entry_inputs(entry: &mut super::program::SemanticEntry) {
    let mut roots = entry
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(_, block)| {
            block
                .side_effects
                .iter()
                .flat_map(|effect| effect.referenced_nodes())
                .chain(block.term.referenced_nodes())
        })
        .collect::<Vec<_>>();
    for route in &entry.output_routes {
        roots.push(route.source.value);
        roots.extend(route.writers.iter().filter_map(|writer| match writer {
            OutputWriter::Value(value) => Some(*value),
            OutputWriter::Effect(_) => None,
        }));
    }
    let reachable = graph_ops::execution_value_producer_closure(&entry.graph, roots).nodes;
    let reachable_resources = resources_referenced_by_nodes(entry, &entry.graph, reachable.iter().copied());
    let mut kept_indices = reachable
        .iter()
        .filter_map(|node| match entry.graph.nodes.get(*node) {
            Some(ENode::FuncParam { index }) => Some(*index),
            _ => None,
        })
        .collect::<HashSet<_>>();
    for (index, resource) in entry.resource_abi.inputs.iter().enumerate() {
        if resource.is_some_and(|resource| reachable_resources.contains(&resource)) {
            kept_indices.insert(index);
        }
    }
    let mut kept = entry
        .graph
        .nodes
        .iter()
        .filter_map(|(node, definition)| match definition {
            ENode::FuncParam { index } if kept_indices.contains(index) => Some((*index, node)),
            _ => None,
        })
        .collect::<Vec<_>>();
    kept.sort_by_key(|(index, _)| *index);
    kept.dedup_by_key(|(index, _)| *index);
    let inputs = kept.iter().map(|(index, _)| entry.inputs[*index].clone()).collect();
    let params = kept.iter().map(|(index, _)| entry.params[*index].clone()).collect();
    let resource_inputs = kept.iter().map(|(index, _)| entry.resource_abi.inputs[*index]).collect();
    let retained = kept.iter().map(|(_, node)| *node).collect::<HashSet<_>>();
    graph_ops::remove_unretained_func_params(&mut entry.graph, &retained);
    for (new_index, (_, node)) in kept.into_iter().enumerate() {
        if let Some(ENode::FuncParam { index }) = entry.graph.nodes.get_mut(node) {
            *index = new_index;
        }
    }
    entry.inputs = inputs;
    entry.params = params;
    entry.resource_abi.inputs = resource_inputs;
}

fn refresh_resource_reads_for_values(graph: &mut EGraph, values: &[NodeId]) {
    let mut sites = Vec::<SideEffectSite>::new();
    for (block_id, block) in &graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            if matches!(&effect.kind, SideEffectKind::Soac(Soac::Screma(op)) if matches!(op.semantic_state(), screma::SemanticState::Segmented { .. }))
                && effect
                    .referenced_nodes()
                    .any(|node| values.iter().any(|value| depends_on(graph, node, *value)))
            {
                sites.push(SideEffectSite {
                    block: block_id,
                    index,
                });
            }
        }
    }
    for site in sites {
        let reads = {
            let effect = &graph.skeleton.blocks[site.block].side_effects[site.index];
            super::semantic_graph::read_resources(graph, effect)
        };
        let SideEffectKind::Soac(Soac::Screma(op)) =
            &mut graph.skeleton.blocks[site.block].side_effects[site.index].kind
        else {
            continue;
        };
        let screma::SemanticState::Segmented { resources, .. } = op.semantic_state_mut() else {
            continue;
        };
        resources.retain(|access| access.access != SegResourceAccessKind::Read);
        for read in reads {
            if let Some(existing) = resources.iter_mut().find(|access| access.resource == read.resource) {
                if existing.access == SegResourceAccessKind::Write {
                    existing.access = SegResourceAccessKind::ReadWrite;
                }
            } else {
                resources.push(read);
            }
        }
        resources.sort_by_key(|access| access.resource);
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
    entry: &super::program::SemanticEntry,
    block_id: crate::ssa::framework::BlockId,
    effects: &HashSet<usize>,
) -> HashSet<ResourceId> {
    let graph = &entry.graph;
    let block = &graph.skeleton.blocks[block_id];
    let mut dependencies = HashSet::new();
    for &effect_index in effects {
        let effect = &block.side_effects[effect_index];
        let producer_nodes = graph_ops::value_producer_closure(graph, effect.referenced_nodes()).nodes;
        dependencies.extend(resources_referenced_by_nodes(entry, graph, producer_nodes));
        match &effect.kind {
            SideEffectKind::Soac(Soac::Screma(op)) => {
                if let screma::SemanticState::Segmented { resources, .. } = op.semantic_state() {
                    dependencies.extend(resources.iter().map(|access| access.resource.0));
                }
            }
            SideEffectKind::Soac(Soac::Filter(filter::Op {
                state: filter::SemanticState { storage, .. },
                ..
            })) => {
                if let filter::Output::Runtime { scratch, length } = storage {
                    dependencies.insert(scratch.0);
                    if let filter::RuntimeLength::Stored(length) = length {
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
                SideEffectKind::Soac(Soac::Screma(op)) => {
                    let mut new_resources = Vec::new();
                    for (input, input_type) in op.lanes_mut().inputs.iter_mut().enumerate() {
                        if let Some(replacement) = replacements
                            .iter()
                            .find(|replacement| effect.operand_nodes[input] == replacement.project)
                        {
                            input_type.array = replacement.view_ty.clone();
                            input_type.element = replacement.elem_ty.clone();
                            new_resources.push(replacement.resource);
                        }
                    }
                    {
                        let screma::SemanticState::Segmented { space, resources, .. } =
                            op.semantic_state_mut()
                        else {
                            continue;
                        };
                        replace_space_nodes(space, replacements);
                        for resource in new_resources {
                            if !resources.iter().any(|access| access.resource.0 == resource) {
                                resources.push(SegResourceAccess {
                                    resource: SemanticResourceRef(resource),
                                    access: SegResourceAccessKind::Read,
                                });
                            }
                        }
                    }
                    let input_arrays =
                        op.lanes().inputs.iter().map(|input| input.array.clone()).collect::<Vec<_>>();
                    for map in &mut op.lanes_mut().maps {
                        if map.destination == SoacDestination::InputBuffer {
                            if let Some(input) = map.input_indices.first() {
                                map.result_type = input_arrays[input.index()].clone();
                            }
                        }
                    }
                    for operator in op.operators_mut() {
                        if operator.destination == SoacDestination::InputBuffer {
                            if let Some(input) = operator.input_indices.first() {
                                operator.result_type = input_arrays[input.index()].clone();
                            }
                        }
                    }
                    if let Some(result) = effect.result {
                        result_retypes.push((result, op.result_types()));
                    }
                }
                SideEffectKind::Soac(Soac::Filter(filter::Op { body, state })) => {
                    if let Some(replacement) = replacements
                        .iter()
                        .find(|replacement| effect.operand_nodes[0] == replacement.project)
                    {
                        let input = match &mut body.input {
                            filter::Input::Plain(input) | filter::Input::Mapped { input, .. } => input,
                        };
                        input.array = replacement.view_ty.clone();
                        input.element = replacement.elem_ty.clone();
                    }
                    replace_space_nodes(&mut state.space, replacements);
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
