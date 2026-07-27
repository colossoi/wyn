//! Fuse pointwise producers into shape-changing SOAC envelopes.
//!
//! `SegFilter` can carry one pre-predicate map body directly. `SegHist` has a
//! general callable envelope, so composition replaces the consumed map result
//! with the producer's source elements. Keeping these rewrites in semantic
//! EGIR makes space, resource, and effect legality available at the decision.

use polytype::Type;
use smallvec::SmallVec;

use super::{capture_types, graph_and_span, producer_is_used_only_by};
use crate::ast::TypeName;
use crate::egir::graph_ops;
use crate::egir::ir::{splice_effect_tokens, Body, BodySite};
use crate::egir::program::{fresh_region_name, CoreProgramData, RegionInterner, SemanticFunc};
use crate::egir::reify::Segmented;
use crate::egir::semantic_graph::SemanticGraph;
use crate::egir::soac::{filter, hist, screma};
use crate::egir::types::{
    EGraph, NodeId, PureOp, ResourceAccess, SegBody, SegSpace, SideEffect, SideEffectKind,
    SkeletonTerminator, Soac, SoacEffect, SoacInputType,
};
use crate::flow::BlockId;
use crate::LookupMap;

#[derive(Clone, Copy)]
enum EnvelopeKind {
    Filter,
    Hist,
}

#[derive(Clone)]
pub(super) struct Candidate {
    site: BodySite,
    block: BlockId,
    producer: usize,
    consumer: usize,
    output: usize,
    consumer_inputs: Vec<usize>,
    kind: EnvelopeKind,
}

#[derive(Clone)]
struct ProducerParts {
    space: SegSpace,
    body: SegBody,
    input_nodes: Vec<NodeId>,
    inputs: Vec<SoacInputType>,
    output_elem_type: Type<TypeName>,
}

pub(super) fn analyze(inner: &Segmented, oracle: &SemanticGraph) -> Option<Candidate> {
    find_candidate(inner, oracle)
}

fn find_candidate(inner: &Segmented, oracle: &SemanticGraph) -> Option<Candidate> {
    for (index, entry) in inner.entry_points.iter().enumerate() {
        if let Some(candidate) = find_in_graph(&entry.graph, BodySite::Entry(index), oracle) {
            return Some(candidate);
        }
    }
    for function in &inner.functions {
        if let Some(candidate) = find_in_graph(&function.graph, BodySite::Function(function.region), oracle)
        {
            return Some(candidate);
        }
    }
    None
}

fn find_in_graph(graph: &EGraph, site: BodySite, oracle: &SemanticGraph) -> Option<Candidate> {
    for (block_id, block) in &graph.skeleton.blocks {
        for producer_index in 0..block.side_effects.len().saturating_sub(1) {
            let producer = &block.side_effects[producer_index];
            let SideEffectKind::Soac(SoacEffect(
                producer_id,
                Soac::Screma(screma::Op {
                    lanes: screma::Lanes { maps, .. },
                    operators,
                    post_maps: _,
                    hidden_scan_outputs: _,
                    state:
                        screma::SemanticState::Segmented {
                            output_slots,
                            resources,
                            ..
                        },
                }),
            )) = &producer.kind
            else {
                continue;
            };
            if !operators.is_empty()
                || maps.is_empty()
                || !output_slots.is_empty()
                || !maps.iter().all(|map| map.destination.is_unplaced())
                || resources.iter().any(|resource| resource.access != ResourceAccess::Read)
            {
                continue;
            }
            let Some(producer_result) = producer.result else {
                continue;
            };
            if oracle.value_consumer_count(producer_id) != 1 {
                continue;
            }

            for consumer_index in (producer_index + 1)..block.side_effects.len() {
                let consumer = &block.side_effects[consumer_index];
                let Some(consumer_id) = consumer.kind.soac_id() else {
                    continue;
                };
                if !intervening_ops_are_safe(block, producer_index, consumer_index, producer_id, oracle) {
                    continue;
                }
                match &consumer.kind {
                    SideEffectKind::Soac(SoacEffect(
                        _,
                        Soac::Filter(filter::Op {
                            body:
                                filter::Body {
                                    input: filter::Input::Plain(_),
                                    ..
                                },
                            ..
                        }),
                    )) => {
                        let Some(output) = consumer.operand_nodes.first().and_then(|&operand| {
                            graph_ops::projection_index(graph, operand, producer_result)
                        }) else {
                            continue;
                        };
                        if output >= maps.len()
                            || maps[output].input_indices.len() != 1
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
                        // The direct pair stays in source order. An effect-token
                        // edge is therefore harmless; a resource edge is not
                        // expected because the filter writes fresh storage.
                        if oracle.conflicts(producer_id, consumer_id) && !token_adjacent(producer, consumer)
                        {
                            continue;
                        }
                        return Some(Candidate {
                            site,
                            block: block_id,
                            producer: producer_index,
                            consumer: consumer_index,
                            output,
                            consumer_inputs: vec![0],
                            kind: EnvelopeKind::Filter,
                        });
                    }
                    SideEffectKind::Soac(SoacEffect(
                        _,
                        Soac::Hist(hist::Op {
                            body,
                            state: hist::State::Segmented(_),
                        }),
                    )) => {
                        if producer_reads_hist_destination(graph, producer, consumer) {
                            continue;
                        }
                        let projected: Vec<(usize, usize)> = consumer.operand_nodes
                            [1..1 + body.inputs.len()]
                            .iter()
                            .enumerate()
                            .filter_map(|(input, &operand)| {
                                graph_ops::projection_index(graph, operand, producer_result)
                                    .map(|output| (input, output))
                            })
                            .collect();
                        let Some(&(_, output)) = projected.first() else {
                            continue;
                        };
                        if output >= maps.len()
                            || projected.iter().any(|&(_, candidate_output)| candidate_output != output)
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
                            output,
                            consumer_inputs: projected.into_iter().map(|(input, _)| input).collect(),
                            kind: EnvelopeKind::Hist,
                        });
                    }
                    _ => {}
                }
            }
        }
    }
    None
}

fn intervening_ops_are_safe(
    block: &crate::egir::types::SkeletonBlock,
    producer: usize,
    consumer: usize,
    producer_id: &crate::egir::program::SemanticOpId,
    oracle: &SemanticGraph,
) -> bool {
    ((producer + 1)..consumer).all(|index| {
        let effect = &block.side_effects[index];
        match &effect.kind {
            SideEffectKind::Soac(SoacEffect(intervening, _)) => !oracle.conflicts(producer_id, intervening),
            _ => effect.effects.is_none(),
        }
    })
}

fn token_adjacent(producer: &SideEffect, consumer: &SideEffect) -> bool {
    matches!(
        (producer.effects, consumer.effects),
        (Some((_, producer_out)), Some((consumer_in, _))) if producer_out == consumer_in
    )
}

fn producer_reads_hist_destination(graph: &EGraph, producer: &SideEffect, hist: &SideEffect) -> bool {
    let Some(destination) =
        hist.operand_nodes.first().and_then(|&node| graph_ops::extract_storage_view_source(graph, node))
    else {
        return false;
    };
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &producer.kind else {
        return false;
    };
    let screma::SemanticState::Segmented { resources, .. } = op.semantic_state() else {
        return false;
    };
    resources.iter().any(|resource| resource.resource == destination)
}

fn producer_parts(graph: &EGraph, candidate: &Candidate) -> ProducerParts {
    let effect = &graph.skeleton.blocks[candidate.block].side_effects[candidate.producer];
    let SideEffectKind::Soac(SoacEffect(
        _,
        Soac::Screma(screma::Op {
            lanes,
            operators: _,
            post_maps: _,
            hidden_scan_outputs: _,
            state: screma::SemanticState::Segmented { space, .. },
        }),
    )) = &effect.kind
    else {
        unreachable!();
    };
    let map = &lanes.maps[candidate.output];
    let source_indices = &map.input_indices;
    ProducerParts {
        space: space.clone(),
        body: map.body.clone(),
        input_nodes: source_indices.iter().map(|index| effect.operand_nodes[index.index()]).collect(),
        inputs: source_indices.iter().map(|index| lanes.inputs[index.index()].clone()).collect(),
        output_elem_type: map.output_element_type.clone(),
    }
}

pub(super) fn apply(inner: Segmented, candidate: Candidate) -> Segmented {
    match candidate.kind {
        EnvelopeKind::Filter => apply_filter(inner, candidate),
        EnvelopeKind::Hist => apply_hist(inner, candidate),
    }
}

fn apply_filter(inner: Segmented, candidate: Candidate) -> Segmented {
    let producer = producer_parts(graph_and_span(&inner, candidate.site).0, &candidate);
    debug_assert_eq!(producer.input_nodes.len(), 1);
    inner.rewrite_body(candidate.site, |body| {
        let rewrite_graph = |graph: &mut EGraph| {
            let block = &mut graph.skeleton.blocks[candidate.block];
            let fused_effects = splice_effect_tokens(
                block.side_effects[candidate.producer].effects,
                block.side_effects[candidate.consumer].effects,
            );
            let consumer = &mut block.side_effects[candidate.consumer];
            consumer.operand_nodes[0] = producer.input_nodes[0];
            if let SideEffectKind::Soac(SoacEffect(_, Soac::Filter(filter::Op { body, state }))) =
                &mut consumer.kind
            {
                body.input = filter::Input::Mapped {
                    input: producer.inputs[0].clone(),
                    body: producer.body.clone(),
                    output_element_type: producer.output_elem_type.clone(),
                };
                state.space = producer.space.clone();
                if let filter::Output::Local { destination, .. } = &mut state.storage {
                    if destination.is_unplaced_unique_input() {
                        destination.make_fresh();
                    }
                }
            }
            consumer.effects = fused_effects;
            block.side_effects.remove(candidate.producer);
        };
        match body {
            Body::Entry(mut entry) => {
                rewrite_graph(&mut entry.graph);
                Body::Entry(entry)
            }
            Body::Function(mut function) => {
                rewrite_graph(&mut function.graph);
                Body::Function(function)
            }
            Body::Constant(_) => unreachable!("envelope fusion never targets constants"),
        }
    })
}

fn apply_hist(inner: Segmented, candidate: Candidate) -> Segmented {
    let (producer, hist_effect, outer_types, span, scope) = {
        let (graph, span, scope) = graph_and_span(&inner, candidate.site);
        (
            producer_parts(graph, &candidate),
            graph.skeleton.blocks[candidate.block].side_effects[candidate.consumer].clone(),
            graph.nodes.iter().map(|(node, data)| (node, data.ty.clone())).collect(),
            span,
            scope,
        )
    };
    let SideEffectKind::Soac(SoacEffect(_, Soac::Hist(hist::Op { body: hist_body, .. }))) =
        &hist_effect.kind
    else {
        unreachable!();
    };

    let mut new_array_types = Vec::new();
    let mut new_elem_types = Vec::new();
    let mut new_input_nodes = Vec::new();
    let mut old_to_new = vec![None; hist_body.inputs.len()];
    let insert_at = candidate.consumer_inputs[0];
    for input in 0..hist_body.inputs.len() {
        if input == insert_at {
            new_array_types.extend(producer.inputs.iter().map(|input| input.array.clone()));
            new_elem_types.extend(producer.inputs.iter().map(SoacInputType::element));
            new_input_nodes.extend(producer.input_nodes.iter().copied());
        }
        if candidate.consumer_inputs.contains(&input) {
            continue;
        }
        old_to_new[input] = Some(new_array_types.len());
        new_array_types.push(hist_body.inputs[input].array.clone());
        new_elem_types.push(hist_body.inputs[input].element());
        new_input_nodes.push(hist_effect.operand_nodes[1 + input]);
    }
    let producer_base =
        insert_at - candidate.consumer_inputs.iter().filter(|&&input| input < insert_at).count();
    let (new_input_nodes, new_array_types, new_elem_types, input_remap) =
        super::deduplicate_array_inputs(new_input_nodes, new_array_types, new_elem_types);
    for mapped in old_to_new.iter_mut().flatten() {
        *mapped = input_remap[*mapped];
    }
    let producer_inputs: Vec<usize> = (producer_base..producer_base + producer.input_nodes.len())
        .map(|input| input_remap[input])
        .collect();
    let mut region_interner = inner.data.region_interner.clone();
    let (body, function) = compose_hist_region(
        &inner,
        &mut region_interner,
        &scope,
        span,
        &producer,
        &hist_body.body,
        &new_elem_types,
        &old_to_new,
        &candidate.consumer_inputs,
        &producer_inputs,
        &outer_types,
    );
    let rebuilt = inner.rewrite_body(candidate.site, |program_body| {
        let rewrite_graph = |graph: &mut EGraph| {
            let block = &mut graph.skeleton.blocks[candidate.block];
            let fused_effects = splice_effect_tokens(
                block.side_effects[candidate.producer].effects,
                block.side_effects[candidate.consumer].effects,
            );
            let destination = block.side_effects[candidate.consumer].operand_nodes[0];
            let consumer = &mut block.side_effects[candidate.consumer];
            consumer.operand_nodes = std::iter::once(destination)
                .chain(new_input_nodes.iter().copied())
                .collect::<SmallVec<[NodeId; 4]>>();
            if let SideEffectKind::Soac(SoacEffect(
                _,
                Soac::Hist(hist::Op {
                    body: consumer_body,
                    state,
                }),
            )) = &mut consumer.kind
            {
                consumer_body.body = body.clone();
                consumer_body.inputs =
                    new_array_types.iter().cloned().map(|array| SoacInputType { array }).collect();
                *state = hist::State::Segmented(producer.space.clone());
            }
            consumer.effects = fused_effects;
            block.side_effects.remove(candidate.producer);
        };
        match program_body {
            Body::Entry(mut entry) => {
                rewrite_graph(&mut entry.graph);
                Body::Entry(entry)
            }
            Body::Function(mut function) => {
                rewrite_graph(&mut function.graph);
                Body::Function(function)
            }
            Body::Constant(_) => unreachable!("envelope fusion never targets constants"),
        }
    });
    rebuilt.extend_functions([function]).map_data(|data| CoreProgramData {
        region_interner,
        ..data
    })
}

#[allow(clippy::too_many_arguments)]
fn compose_hist_region(
    inner: &Segmented,
    region_interner: &mut RegionInterner,
    scope: &str,
    span: crate::ast::Span,
    producer: &ProducerParts,
    hist: &SegBody,
    element_types: &[Type<TypeName>],
    old_to_new: &[Option<usize>],
    replaced: &[usize],
    producer_inputs: &[usize],
    outer_types: &LookupMap<NodeId, Type<TypeName>>,
) -> (SegBody, SemanticFunc) {
    let capture_types = capture_types(outer_types, producer.body.captures.iter().chain(&hist.captures));
    let mut params: Vec<_> = element_types
        .iter()
        .enumerate()
        .map(|(index, ty)| (ty.clone(), format!("element_{index}")))
        .collect();
    params.extend(
        capture_types.iter().enumerate().map(|(index, ty)| (ty.clone(), format!("capture_{index}"))),
    );
    let (producer_name, producer_return_ty) = {
        let region = inner.region(producer.body.region).expect("producer region");
        (region.name.clone(), region.return_ty.clone())
    };
    let (hist_name, hist_return_ty) = {
        let region = inner.region(hist.region).expect("histogram region");
        (region.name.clone(), region.return_ty.clone())
    };
    let mut graph = EGraph::new();
    let args: Vec<_> =
        params.iter().enumerate().map(|(index, (ty, _))| graph.add_func_param(index, ty.clone())).collect();
    let producer_capture_start = element_types.len();
    let hist_capture_start = producer_capture_start + producer.body.captures.len();
    let mut producer_args: SmallVec<[NodeId; 4]> =
        producer_inputs.iter().map(|&input| args[input]).collect();
    producer_args.extend(args[producer_capture_start..hist_capture_start].iter().copied());
    let produced = graph.intern_pure(
        PureOp::Call(producer_name),
        producer_args,
        producer_return_ty,
        None,
    );
    let mut hist_args = SmallVec::<[NodeId; 4]>::new();
    for (input, mapped) in old_to_new.iter().enumerate() {
        hist_args.push(if replaced.contains(&input) {
            produced
        } else {
            args[mapped.expect("unreplaced histogram input has no new position")]
        });
    }
    hist_args.extend(args[hist_capture_start..].iter().copied());
    let result = graph.intern_pure(PureOp::Call(hist_name), hist_args, hist_return_ty.clone(), None);
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
    let name = fresh_region_name(region_interner, &format!("{scope}_map_hist"));
    let region = region_interner.intern(&name);
    let function = SemanticFunc::new(region, name, span, None, params, hist_return_ty, graph);
    (
        SegBody {
            region,
            captures: producer.body.captures.iter().chain(&hist.captures).copied().collect(),
        },
        function,
    )
}
