//! Same-space producer/consumer fusion.
//!
//! A pure `SegMap` whose result has exactly one semantic consumer is folded
//! into that consumer by composing callable regions.  The producer and
//! consumer must be in the same block, and every intervening operation must be
//! conflict-free according to the semantic dependency graph. This keeps
//! effect-token splicing explicit while permitting independent bindings.
//! Multi-consumer producers are left to logical allocation.

use polytype::Type;
use smallvec::smallvec;

use crate::ast::{Span, TypeName};
use crate::egir::program::{SemanticFunc, SemanticProgram};
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::screma;
use crate::egir::types::{
    EGraph, ENode, NodeId, PureOp, SegBody, SegResourceAccess, SegResourceAccessKind, SegSpace,
    SideEffectKind, SkeletonTerminator, Soac, SoacDestination, SoacInputType,
};
use crate::flow::BlockId;
use crate::LookupMap;

#[derive(Clone, Copy)]
pub(super) enum FusionSite {
    Entry(usize),
    Function(usize),
}

#[derive(Clone)]
struct Candidate {
    site: FusionSite,
    block: BlockId,
    producer: usize,
    consumer: usize,
    consumer_inputs: Vec<usize>,
    producer_output: usize,
}

pub fn fuse_producer_into_consumer(inner: &mut SemanticProgram, oracle: &SemanticGraph) -> bool {
    let candidate = find_candidate(inner, oracle);
    let Some(candidate) = candidate else {
        return false;
    };
    apply_fusion(inner, candidate);
    true
}

fn find_candidate(inner: &SemanticProgram, oracle: &SemanticGraph) -> Option<Candidate> {
    for (index, entry) in inner.entry_points.iter().enumerate() {
        if let Some(candidate) = find_in_graph(&entry.graph, &entry.name, FusionSite::Entry(index), oracle)
        {
            return Some(candidate);
        }
    }
    for (index, function) in inner.functions.iter().enumerate() {
        if let Some(candidate) = find_in_graph(
            &function.graph,
            &function.name,
            FusionSite::Function(index),
            oracle,
        ) {
            return Some(candidate);
        }
    }
    None
}

fn find_in_graph(
    graph: &EGraph,
    _scope: &str,
    site: FusionSite,
    oracle: &SemanticGraph,
) -> Option<Candidate> {
    for (block_id, block) in &graph.skeleton.blocks {
        for producer_index in 0..block.side_effects.len().saturating_sub(1) {
            let producer = &block.side_effects[producer_index];
            let SideEffectKind::Soac(
                producer_id,
                Soac::Screma(screma::Op::Map {
                    lanes: screma::Lanes { maps, .. },
                    state:
                        screma::SemanticState::Segmented {
                            placement: producer_placement,
                            output_slots,
                            resources,
                            ..
                        },
                }),
            ) = &producer.kind
            else {
                continue;
            };
            if maps.is_empty()
                || !output_slots.is_empty()
                || !maps.iter().all(|map| {
                    matches!(
                        map.destination,
                        SoacDestination::Fresh | SoacDestination::UniqueInput
                    )
                })
                || resources.iter().any(|resource| resource.access != SegResourceAccessKind::Read)
            {
                continue;
            }
            let Some(producer_result) = producer.result else {
                continue;
            };
            if oracle.value_consumer_count(&producer_id) != 1 {
                continue;
            }

            for consumer_index in (producer_index + 1)..block.side_effects.len() {
                let consumer = &block.side_effects[consumer_index];
                let SideEffectKind::Soac(consumer_id, Soac::Screma(consumer_op)) = &consumer.kind else {
                    continue;
                };
                let screma::SemanticState::Segmented {
                    resources: consumer_resources,
                    ..
                } = consumer_op.semantic_state()
                else {
                    continue;
                };
                if *producer_placement != screma::Placement::LaneLocal {
                    continue;
                }
                let Some(_) = consumer.result else {
                    continue;
                };
                if oracle.conflicts(&producer_id, &consumer_id)
                    && !matches!((producer.effects, consumer.effects),
                    (Some((_, producer_out)), Some((consumer_in, _))) if producer_out == consumer_in)
                {
                    continue;
                }
                if resources.iter().any(|producer_resource| {
                    consumer_resources.iter().any(|consumer_resource| {
                        producer_resource.resource == consumer_resource.resource
                            && (producer_resource.access != SegResourceAccessKind::Read
                                || consumer_resource.access != SegResourceAccessKind::Read)
                    })
                }) {
                    continue;
                }
                // Folding P into C moves P's pure computation down to C's
                // position.  Every intervening semantic op must therefore be free
                // of ordering conflicts with P; opaque effects remain a
                // conservative barrier.
                if !((producer_index + 1)..consumer_index).all(|index| {
                    let effect = &block.side_effects[index];
                    match (&effect.kind, effect.result) {
                        (SideEffectKind::Soac(intervening, Soac::Screma(_)), Some(_)) => {
                            !oracle.conflicts(&producer_id, &intervening)
                        }
                        _ => effect.effects.is_none(),
                    }
                }) {
                    continue;
                }
                let n_inputs = consumer_op.lanes().inputs.len();
                let projected: Vec<(usize, usize)> = consumer.operand_nodes[..n_inputs]
                    .iter()
                    .enumerate()
                    .filter_map(|(input, &operand)| {
                        projection_of(graph, operand, producer_result).map(|output| (input, output))
                    })
                    .collect();
                let Some(&(_, producer_output)) = projected.first() else {
                    continue;
                };
                let projected_roots: std::collections::HashSet<_> =
                    projected.iter().map(|(input, _)| consumer.operand_nodes[*input]).collect();
                if producer_output >= maps.len()
                    || projected.iter().any(|&(_, output)| output != producer_output)
                    || consumer.referenced_nodes().any(|root| {
                        reaches(graph, root, producer_result) && !projected_roots.contains(&root)
                    })
                    || !producer_is_used_only_by(
                        graph,
                        block_id,
                        producer_index,
                        consumer_index,
                        producer_result,
                    )
                {
                    continue;
                }
                return Some(Candidate {
                    site,
                    block: block_id,
                    producer: producer_index,
                    consumer: consumer_index,
                    consumer_inputs: projected.into_iter().map(|(input, _)| input).collect(),
                    producer_output,
                });
            }
        }
    }
    None
}

pub(super) fn projection_of(graph: &EGraph, node: NodeId, root: NodeId) -> Option<usize> {
    match &graph.nodes[node] {
        ENode::Pure {
            op: PureOp::Project { index },
            operands,
        } if operands.first() == Some(&root) => Some(*index as usize),
        _ => None,
    }
}

pub(super) fn reaches(graph: &EGraph, start: NodeId, target: NodeId) -> bool {
    wyn_graph::reaches_ordered(start, target, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        out.extend(graph.nodes[node].children());
    })
}

pub(super) fn producer_is_used_only_by(
    graph: &EGraph,
    producer_block: BlockId,
    producer_index: usize,
    consumer_index: usize,
    producer_result: NodeId,
) -> bool {
    for (block_id, block) in &graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            if block_id == producer_block && (index == producer_index || index == consumer_index) {
                continue;
            }
            if effect.referenced_nodes().any(|node| reaches(graph, node, producer_result)) {
                return false;
            }
        }
        if block.term.referenced_nodes().into_iter().any(|root| reaches(graph, root, producer_result)) {
            return false;
        }
    }
    true
}

#[derive(Clone)]
struct ProducerParts {
    space: SegSpace,
    body: SegBody,
    source_indices: Vec<screma::InputId>,
    input_nodes: Vec<NodeId>,
    inputs: Vec<SoacInputType>,
    resources: Vec<SegResourceAccess>,
}

fn apply_fusion(inner: &mut SemanticProgram, candidate: Candidate) {
    let (graph, span, scope) = graph_and_span(inner, candidate.site);
    let outer_types = graph.types.clone();
    let producer_effect = graph.skeleton.blocks[candidate.block].side_effects[candidate.producer].clone();
    let consumer_effect = graph.skeleton.blocks[candidate.block].side_effects[candidate.consumer].clone();

    let SideEffectKind::Soac(
        _,
        Soac::Screma(screma::Op::Map {
            lanes: producer_lanes,
            state: screma::SemanticState::Segmented { resources, space, .. },
        }),
    ) = producer_effect.kind
    else {
        unreachable!();
    };
    let producer_input_count = producer_lanes.inputs.len();
    let producer_map = &producer_lanes.maps[candidate.producer_output];
    let source_indices = producer_map.input_indices.clone();
    let producer = ProducerParts {
        space,
        body: producer_map.body.clone(),
        source_indices: source_indices.clone(),
        input_nodes: source_indices
            .iter()
            .map(|index| producer_effect.operand_nodes[index.index()])
            .collect(),
        inputs: source_indices.iter().map(|index| producer_lanes.inputs[index.index()].clone()).collect(),
        resources,
    };
    debug_assert!(producer_input_count >= producer.source_indices.len());

    let SideEffectKind::Soac(_, Soac::Screma(mut consumer_op)) = consumer_effect.kind else {
        unreachable!();
    };
    let old_input_count = consumer_op.lanes().inputs.len();
    let mut new_inputs = consumer_op.lanes().inputs.clone();
    for &input in candidate.consumer_inputs.iter().rev() {
        new_inputs.remove(input);
    }
    new_inputs.extend(producer.inputs.iter().cloned());
    let new_elem_types = new_inputs.iter().map(|input| input.element.clone()).collect::<Vec<_>>();
    let appended_base = old_input_count - candidate.consumer_inputs.len();

    let mut new_maps = consumer_op.lanes().maps.clone();
    let mut synthesized = Vec::new();
    for (lane, map) in new_maps.iter_mut().enumerate() {
        let body = map.body.clone();
        let indices = map.input_indices.clone();
        let mut rebased = Vec::new();
        for &index in &indices {
            if candidate.consumer_inputs.contains(&index.index()) {
                rebased.extend(
                    (0..producer.inputs.len()).map(|offset| screma::InputId(appended_base + offset)),
                );
            } else {
                rebased.push(rebase_after_removals(index, &candidate.consumer_inputs));
            }
        }
        if indices.iter().any(|index| candidate.consumer_inputs.contains(&index.index())) {
            let (new_body, function) = compose_map_region(
                inner,
                &scope,
                span,
                lane,
                &producer,
                &body,
                &indices,
                &rebased,
                &candidate.consumer_inputs,
                &new_elem_types,
                &outer_types,
            );
            map.body = new_body;
            synthesized.push(function);
            if map.destination == SoacDestination::UniqueInput
                && indices.first().is_some_and(|index| candidate.consumer_inputs.contains(&index.index()))
            {
                map.destination = SoacDestination::Fresh;
            }
        }
        map.input_indices = rebased;
    }

    for (operator_index, operator) in consumer_op.operators_mut().into_iter().enumerate() {
        let old_indices = operator.input_indices.clone();
        let mut rebased = Vec::new();
        for &index in &old_indices {
            if candidate.consumer_inputs.contains(&index.index()) {
                rebased.extend(
                    (0..producer.inputs.len()).map(|offset| screma::InputId(appended_base + offset)),
                );
            } else {
                rebased.push(rebase_after_removals(index, &candidate.consumer_inputs));
            }
        }
        if old_indices.iter().any(|index| candidate.consumer_inputs.contains(&index.index())) {
            let (step, function) = compose_step_region(
                inner,
                &scope,
                span,
                operator_index,
                &producer,
                operator,
                &old_indices,
                &rebased,
                &candidate.consumer_inputs,
                &new_elem_types,
                &outer_types,
            );
            operator.step = step;
            synthesized.push(function);
            if operator.destination == SoacDestination::UniqueInput
                && old_indices
                    .first()
                    .is_some_and(|index| candidate.consumer_inputs.contains(&index.index()))
            {
                operator.destination = SoacDestination::Fresh;
            }
        }
        operator.input_indices = rebased;
    }

    // Publish synthesized functions and their complete region bodies before
    // installing references to their RegionIds in the consumer.
    for function in synthesized {
        inner.define_region(function);
    }

    let graph = graph_mut(inner, candidate.site);
    let block = &mut graph.skeleton.blocks[candidate.block];
    let producer_effects = block.side_effects[candidate.producer].effects;
    let consumer_effects = block.side_effects[candidate.consumer].effects;
    // Legality above ensures that every intervening effect is token-free, so
    // the producer/consumer token endpoints can be spliced exactly as for an
    // adjacent pair.
    let fused_effects = match (producer_effects, consumer_effects) {
        (Some((input, _)), Some((_, output))) => Some((input, output)),
        (Some(effects), None) | (None, Some(effects)) => Some(effects),
        (None, None) => None,
    };
    let consumer = &mut block.side_effects[candidate.consumer];
    let n_old_inputs = old_input_count;
    let tail = consumer.operand_nodes[n_old_inputs..].to_vec();
    let mut operands = consumer.operand_nodes[..n_old_inputs].to_vec();
    for &input in candidate.consumer_inputs.iter().rev() {
        operands.remove(input);
    }
    operands.extend(producer.input_nodes.iter().copied());
    operands.extend(tail);
    consumer.operand_nodes = operands.into();
    if let SideEffectKind::Soac(_, Soac::Screma(op)) = &mut consumer.kind {
        consumer_op.lanes_mut().maps = new_maps;
        consumer_op.lanes_mut().inputs = new_inputs;
        let screma::SemanticState::Segmented { space, resources, .. } = consumer_op.semantic_state_mut()
        else {
            unreachable!();
        };
        *space = producer.space;
        *resources = SegResourceAccess::merge(resources, &producer.resources);
        *op = consumer_op;
    }
    consumer.effects = fused_effects;
    block.side_effects.remove(candidate.producer);
}

pub(super) fn graph_and_span(inner: &SemanticProgram, site: FusionSite) -> (&EGraph, Span, String) {
    match site {
        FusionSite::Entry(index) => {
            let entry = &inner.entry_points[index];
            (&entry.graph, entry.span, entry.name.clone())
        }
        FusionSite::Function(index) => {
            let function = &inner.functions[index];
            (&function.graph, function.span, function.name.clone())
        }
    }
}

pub(super) fn graph_mut(inner: &mut SemanticProgram, site: FusionSite) -> &mut EGraph {
    match site {
        FusionSite::Entry(index) => &mut inner.entry_points[index].graph,
        FusionSite::Function(index) => &mut inner.functions[index].graph,
    }
}

#[allow(clippy::too_many_arguments)]
fn compose_map_region(
    inner: &mut SemanticProgram,
    scope: &str,
    span: Span,
    lane: usize,
    producer: &ProducerParts,
    consumer: &SegBody,
    old_indices: &[screma::InputId],
    new_indices: &[screma::InputId],
    replaced_inputs: &[usize],
    new_elem_types: &[Type<TypeName>],
    outer_types: &LookupMap<NodeId, Type<TypeName>>,
) -> (SegBody, SemanticFunc) {
    let element_types: Vec<_> =
        new_indices.iter().map(|index| new_elem_types[index.index()].clone()).collect();
    let capture_types = capture_types(
        outer_types,
        producer.body.captures.iter().chain(&consumer.captures),
    );
    let mut params: Vec<(Type<TypeName>, String)> = element_types
        .iter()
        .enumerate()
        .map(|(index, ty)| (ty.clone(), format!("element_{index}")))
        .collect();
    params.extend(
        capture_types.iter().enumerate().map(|(index, ty)| (ty.clone(), format!("capture_{index}"))),
    );
    let producer_region = &inner.ir.regions[&producer.body.region];
    let consumer_region = &inner.ir.regions[&consumer.region];
    let mut graph = EGraph::new();
    let args: Vec<_> =
        params.iter().enumerate().map(|(index, (ty, _))| graph.add_func_param(index, ty.clone())).collect();
    let element_count = element_types.len();
    let producer_capture_start = element_count;
    let consumer_capture_start = producer_capture_start + producer.body.captures.len();
    let mut cursor = 0;
    let mut producer_elements = None;
    let mut consumer_elements = Vec::with_capacity(old_indices.len());
    for &index in old_indices {
        if replaced_inputs.contains(&index.index()) {
            let end = cursor + producer.inputs.len();
            if producer_elements.is_none() {
                producer_elements = Some(args[cursor..end].to_vec());
            }
            cursor = end;
            consumer_elements.push(NodeId::default());
        } else {
            consumer_elements.push(args[cursor]);
            cursor += 1;
        }
    }
    let mut producer_args: smallvec::SmallVec<[NodeId; 4]> =
        producer_elements.expect("composed map body consumes producer output").into_iter().collect();
    producer_args.extend(args[producer_capture_start..consumer_capture_start].iter().copied());
    let produced = graph.intern_pure(
        PureOp::Call(producer_region.name.clone()),
        producer_args,
        producer_region.return_ty.clone(),
        None,
    );
    for (position, &index) in old_indices.iter().enumerate() {
        if replaced_inputs.contains(&index.index()) {
            consumer_elements[position] = produced;
        }
    }
    let mut consumer_args: smallvec::SmallVec<[NodeId; 4]> = consumer_elements.into_iter().collect();
    consumer_args.extend(args[consumer_capture_start..].iter().copied());
    let result = graph.intern_pure(
        PureOp::Call(consumer_region.name.clone()),
        consumer_args,
        consumer_region.return_ty.clone(),
        None,
    );
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
    let name = fresh_region_name(inner, &format!("{scope}_vertical_map_{lane}"));
    let region = inner.ir.region_interner.intern(&name);
    let function = SemanticFunc::new(
        name,
        span,
        None,
        params,
        consumer_region.return_ty.clone(),
        graph,
        LookupMap::new(),
    );
    (
        SegBody {
            region,
            captures: producer.body.captures.iter().chain(&consumer.captures).copied().collect(),
        },
        function,
    )
}

#[allow(clippy::too_many_arguments)]
fn compose_step_region(
    inner: &mut SemanticProgram,
    scope: &str,
    span: Span,
    operator_index: usize,
    producer: &ProducerParts,
    operator: &screma::Operator,
    old_indices: &[screma::InputId],
    new_indices: &[screma::InputId],
    replaced_inputs: &[usize],
    new_elem_types: &[Type<TypeName>],
    outer_types: &LookupMap<NodeId, Type<TypeName>>,
) -> (SegBody, SemanticFunc) {
    let producer_region = &inner.ir.regions[&producer.body.region];
    let consumer_region = &inner.ir.regions[&operator.step.region];
    let accumulator_ty = consumer_region.params[0].0.clone();
    let capture_types = capture_types(
        outer_types,
        producer.body.captures.iter().chain(&operator.step.captures),
    );
    let mut params = vec![(accumulator_ty, "accumulator".to_string())];
    let element_types: Vec<_> =
        new_indices.iter().map(|index| new_elem_types[index.index()].clone()).collect();
    params.extend(
        element_types.iter().enumerate().map(|(index, ty)| (ty.clone(), format!("element_{index}"))),
    );
    params.extend(
        capture_types.iter().enumerate().map(|(index, ty)| (ty.clone(), format!("capture_{index}"))),
    );
    let mut graph = EGraph::new();
    let args: Vec<_> =
        params.iter().enumerate().map(|(index, (ty, _))| graph.add_func_param(index, ty.clone())).collect();
    let producer_capture_start = 1 + element_types.len();
    let consumer_capture_start = producer_capture_start + producer.body.captures.len();
    let mut cursor = 1;
    let mut producer_elements = None;
    let mut consumer_elements = Vec::with_capacity(old_indices.len());
    for &index in old_indices {
        if replaced_inputs.contains(&index.index()) {
            let end = cursor + producer.inputs.len();
            if producer_elements.is_none() {
                producer_elements = Some(args[cursor..end].to_vec());
            }
            cursor = end;
            consumer_elements.push(NodeId::default());
        } else {
            consumer_elements.push(args[cursor]);
            cursor += 1;
        }
    }
    let mut producer_args: smallvec::SmallVec<[NodeId; 4]> =
        producer_elements.expect("composed accumulator consumes producer output").into_iter().collect();
    producer_args.extend(args[producer_capture_start..consumer_capture_start].iter().copied());
    let produced = graph.intern_pure(
        PureOp::Call(producer_region.name.clone()),
        producer_args,
        producer_region.return_ty.clone(),
        None,
    );
    let mut consumer_args = smallvec![args[0]];
    for (position, &index) in old_indices.iter().enumerate() {
        if replaced_inputs.contains(&index.index()) {
            consumer_elements[position] = produced;
        }
    }
    consumer_args.extend(consumer_elements);
    consumer_args.extend(args[consumer_capture_start..].iter().copied());
    let result = graph.intern_pure(
        PureOp::Call(consumer_region.name.clone()),
        consumer_args,
        consumer_region.return_ty.clone(),
        None,
    );
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
    let name = fresh_region_name(inner, &format!("{scope}_vertical_step_{operator_index}"));
    let region = inner.ir.region_interner.intern(&name);
    let function = SemanticFunc::new(
        name,
        span,
        None,
        params,
        consumer_region.return_ty.clone(),
        graph,
        LookupMap::new(),
    );
    (
        SegBody {
            region,
            captures: producer.body.captures.iter().chain(&operator.step.captures).copied().collect(),
        },
        function,
    )
}

fn rebase_after_removals(index: screma::InputId, removed: &[usize]) -> screma::InputId {
    screma::InputId(
        index.index() - removed.iter().filter(|&&removed_index| removed_index < index.index()).count(),
    )
}

pub(super) fn capture_types<'a>(
    types: &LookupMap<NodeId, Type<TypeName>>,
    captures: impl Iterator<Item = &'a NodeId>,
) -> Vec<Type<TypeName>> {
    captures
        .map(|capture| types.get(capture).expect("capture node is absent from its owning graph").clone())
        .collect()
}

pub(super) fn fresh_region_name(inner: &SemanticProgram, base: &str) -> String {
    if inner.region_interner.get(base).is_none() {
        return base.to_string();
    }
    for suffix in 1.. {
        let candidate = format!("{base}_{suffix}");
        if inner.region_interner.get(&candidate).is_none() {
            return candidate;
        }
    }
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egir::program::{RegionInterner, SemanticEntry, SemanticOpId};
    use crate::egir::semantic_exec::{RegionExecutor, Value};
    use crate::egir::types::{EffectToken, SegExtent, SegLevel, SideEffect};
    use crate::flow::ExecutionModel;

    fn captured_binary_function(name: &str, op: &str) -> SemanticFunc {
        let int = Type::Constructed(TypeName::Int(32), vec![]);
        let mut graph = EGraph::new();
        let left = graph.add_func_param(0, int.clone());
        let right = graph.add_func_param(1, int.clone());
        let result = graph.intern_pure(
            PureOp::BinOp(op.into()),
            smallvec![left, right],
            int.clone(),
            None,
        );
        graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
        SemanticFunc::new(
            name.into(),
            Span::new(0, 0, 0, 0),
            None,
            vec![(int.clone(), "x".into()), (int.clone(), "capture".into())],
            int,
            graph,
            LookupMap::new(),
        )
    }

    #[test]
    fn map_producer_and_map_consumer_compose_regions_value_exactly() {
        let int = Type::Constructed(TypeName::Int(32), vec![]);
        let array = Type::Constructed(
            TypeName::Array,
            vec![
                int.clone(),
                Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
                Type::Constructed(TypeName::Size(8), vec![]),
                Type::Constructed(TypeName::NoBuffer, vec![]),
            ],
        );
        let tuple = Type::Constructed(TypeName::Tuple(1), vec![array.clone()]);
        let producer_fn = captured_binary_function("producer", "+");
        let consumer_fn = captured_binary_function("consumer", "*");
        let mut interner = RegionInterner::default();
        let producer_region = interner.intern("producer");
        let consumer_region = interner.intern("consumer");
        let mut graph = EGraph::new();
        let input = graph.add_func_param(0, array.clone());
        let producer_capture = graph.intern_pure(PureOp::Int("1".into()), smallvec![], int.clone(), None);
        let consumer_capture = graph.intern_pure(PureOp::Int("2".into()), smallvec![], int.clone(), None);
        let producer_result = graph.alloc_side_effect_result(tuple.clone());
        graph.skeleton.blocks[graph.skeleton.entry].side_effects.push(SideEffect {
            kind: SideEffectKind::Soac(
                SemanticOpId(0),
                Soac::Screma(screma::Op::Map {
                    lanes: screma::Lanes {
                        inputs: vec![SoacInputType {
                            array: array.clone(),
                            element: int.clone(),
                        }],
                        maps: vec![screma::Map {
                            body: SegBody {
                                region: producer_region,
                                captures: vec![producer_capture],
                            },
                            input_indices: vec![screma::InputId(0)],
                            output_element_type: int.clone(),
                            destination: SoacDestination::Fresh,
                            result_type: array.clone(),
                        }],
                    },
                    state: screma::SemanticState::Segmented {
                        space: SegSpace {
                            level: SegLevel::Thread,
                            dims: vec![SegExtent::Fixed(8)],
                        },
                        placement: screma::Placement::LaneLocal,
                        output_slots: vec![],
                        resources: vec![],
                    },
                }),
            ),
            operand_nodes: smallvec![input],
            result: Some(producer_result),
            effects: Some((EffectToken::from(0), EffectToken::from(1))),
            span: None,
        });
        let projected = graph.intern_pure(
            PureOp::Project { index: 0 },
            smallvec![producer_result],
            array.clone(),
            None,
        );
        // A token-free independent operation may sit between producer and
        // consumer. F3 must use the dependency oracle rather than requiring
        // textual adjacency.
        let unrelated_result = graph.alloc_side_effect_result(tuple.clone());
        let mut unrelated = graph.skeleton.blocks[graph.skeleton.entry].side_effects[0].clone();
        let SideEffectKind::Soac(id, _) = &mut unrelated.kind else {
            unreachable!();
        };
        *id = SemanticOpId(1);
        unrelated.result = Some(unrelated_result);
        unrelated.effects = None;
        graph.skeleton.blocks[graph.skeleton.entry].side_effects.push(unrelated);
        let consumer_result = graph.alloc_side_effect_result(tuple.clone());
        graph.skeleton.blocks[graph.skeleton.entry].side_effects.push(SideEffect {
            kind: SideEffectKind::Soac(
                SemanticOpId(2),
                Soac::Screma(screma::Op::Map {
                    lanes: screma::Lanes {
                        inputs: vec![SoacInputType {
                            array: array.clone(),
                            element: int.clone(),
                        }],
                        maps: vec![screma::Map {
                            body: SegBody {
                                region: consumer_region,
                                captures: vec![consumer_capture],
                            },
                            input_indices: vec![screma::InputId(0)],
                            output_element_type: int,
                            destination: SoacDestination::Fresh,
                            result_type: array.clone(),
                        }],
                    },
                    state: screma::SemanticState::Segmented {
                        space: SegSpace {
                            level: SegLevel::Thread,
                            dims: vec![SegExtent::Fixed(8)],
                        },
                        placement: screma::Placement::LaneLocal,
                        output_slots: vec![],
                        resources: vec![],
                    },
                }),
            ),
            operand_nodes: smallvec![projected],
            result: Some(consumer_result),
            effects: Some((EffectToken::from(1), EffectToken::from(2))),
            span: None,
        });
        let output = graph.intern_pure(
            PureOp::Project { index: 0 },
            smallvec![consumer_result],
            array.clone(),
            None,
        );
        graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(output));
        let entry = SemanticEntry::new_with_resources(
            "entry".into(),
            Span::new(0, 0, 0, 0),
            ExecutionModel::Compute {
                local_size: (64, 1, 1),
            },
            vec![],
            vec![],
            vec![],
            vec![(array, "xs".into())],
            tuple,
            graph,
            LookupMap::new(),
        );
        let mut inner = SemanticProgram::new(
            vec![producer_fn, consumer_fn],
            vec![],
            vec![entry],
            vec![],
            Default::default(),
            interner,
        );
        crate::egir::semantic_graph::rebuild_dependencies(&mut inner);
        let oracle = SemanticGraph::new(&inner.semantic_dependencies);
        assert!(fuse_producer_into_consumer(&mut inner, &oracle));
        let block =
            &inner.entry_points[0].graph.skeleton.blocks[inner.entry_points[0].graph.skeleton.entry];
        assert_eq!(block.side_effects.len(), 2);
        assert_eq!(block.side_effects[0].result, Some(unrelated_result));
        assert_eq!(
            block.side_effects[1].effects,
            Some((EffectToken::from(0), EffectToken::from(2)))
        );
        let effect = &block.side_effects[1];
        let SideEffectKind::Soac(id, Soac::Screma(op)) = &effect.kind else {
            panic!("one fused SegMap")
        };
        assert_eq!(
            *id,
            SemanticOpId(2),
            "fusion keeps the consumer's semantic identity"
        );
        let executor = RegionExecutor::new(&inner.regions);
        assert_eq!(
            executor
                .call(
                    &op.lanes().maps[0].body.region,
                    &[Value::Int(3), Value::Int(1), Value::Int(2)],
                )
                .unwrap(),
            Value::Int(8)
        );
        assert_eq!(
            op.lanes().maps[0].body.captures,
            [producer_capture, consumer_capture]
        );
    }
}
