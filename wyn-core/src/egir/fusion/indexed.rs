//! Scalar demand fusion for statically indexed pointwise producers.
//!
//! `Index(SegMap(f, inputs), k)` computes one element by indexing each source
//! and calling `f`. EGIR has explicit projections, use edges, and callable
//! regions, so this is a small graph rewrite rather than a TLC tree pass.

use smallvec::smallvec;

use crate::egir::graph_ops;
use crate::egir::ir::{Body, BodySite, RealizedOutputRoute};
use crate::egir::program::{OutputWriter, Program, SemanticResourceRef};
use crate::egir::reify::Segmented;
use crate::egir::soac::screma;
use crate::egir::types::{EGraph, ENode, NodeId, PureOp, ResourceAccess, SideEffectKind, Soac, SoacEffect};
use crate::flow::BlockId;
use crate::ssa::types::ConstantValue;

#[derive(Clone)]
pub(super) struct Candidate {
    site: BodySite,
    block: BlockId,
    effect: usize,
    index: NodeId,
    index_value: NodeId,
    output: usize,
}

pub(super) fn analyze(inner: &Program<Segmented>) -> Option<Candidate> {
    find_candidate(inner)
}

fn find_candidate(inner: &Program<Segmented>) -> Option<Candidate> {
    for (index, entry) in inner.entry_points.iter().enumerate() {
        let output_resources = entry.outputs.iter().map(|output| output.resource).collect::<Vec<_>>();
        let output_routes = entry.routes().cloned().collect::<Vec<_>>();
        if let Some(candidate) = find_in_graph(
            &entry.graph,
            BodySite::Entry(index),
            &output_resources,
            &output_routes,
        ) {
            return Some(candidate);
        }
    }
    for function in &inner.functions {
        if let Some(candidate) =
            find_in_graph(&function.graph, BodySite::Function(function.region), &[], &[])
        {
            return Some(candidate);
        }
    }
    None
}

fn find_in_graph(
    graph: &EGraph,
    site: BodySite,
    output_resources: &[Option<SemanticResourceRef>],
    output_routes: &[RealizedOutputRoute],
) -> Option<Candidate> {
    for (block_id, block) in &graph.skeleton.blocks {
        for (effect_index, effect) in block.side_effects.iter().enumerate() {
            let SideEffectKind::Soac(SoacEffect(
                _,
                Soac::Screma(screma::Op::Map {
                    lanes: screma::Lanes { maps, .. },
                    state:
                        screma::SemanticState::Segmented {
                            output_slots,
                            resources,
                            ..
                        },
                }),
            )) = &effect.kind
            else {
                continue;
            };
            let indirect_output_resources: std::collections::HashSet<_> = output_slots
                .iter()
                .filter_map(|slot| output_resources.get(slot.0).copied().flatten())
                .collect();
            if maps.is_empty()
                || !maps.iter().all(|map| map.destination.is_unplaced())
                || resources.iter().any(|resource| {
                    resource.access != ResourceAccess::Read
                        && !indirect_output_resources.contains(&resource.resource)
                })
            {
                continue;
            }
            let Some(result) = effect.result else {
                continue;
            };
            let demands = graph
                .nodes
                .iter()
                .filter_map(|(node, definition)| {
                    let ENode::Pure {
                        op: PureOp::Index,
                        operands,
                    } = &definition.kind
                    else {
                        return None;
                    };
                    let [base, index] = operands.as_slice() else {
                        return None;
                    };
                    if !is_static_index(graph, *index) {
                        return None;
                    }
                    graph_ops::projection_index(graph, *base, result).map(|output| (node, *index, output))
                })
                .collect::<Vec<_>>();
            let [(index, index_value, output)] = demands.as_slice() else {
                continue;
            };
            if *output >= maps.len()
                || !used_only_through(graph, block_id, effect_index, result, *index, output_routes)
            {
                continue;
            }
            return Some(Candidate {
                site,
                block: block_id,
                effect: effect_index,
                index: *index,
                index_value: *index_value,
                output: *output,
            });
        }
    }
    None
}

fn is_static_index(graph: &EGraph, node: NodeId) -> bool {
    match &graph.nodes[node].kind {
        ENode::Constant(ConstantValue::I32(_) | ConstantValue::U32(_)) => true,
        ENode::Pure {
            op: PureOp::Int(_) | PureOp::Uint(_),
            operands,
        } => operands.is_empty(),
        _ => false,
    }
}

fn used_only_through(
    graph: &EGraph,
    producer_block: BlockId,
    producer_effect: usize,
    result: NodeId,
    demand: NodeId,
    output_routes: &[RealizedOutputRoute],
) -> bool {
    for (block_id, block) in &graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            if block_id == producer_block && index == producer_effect {
                continue;
            }
            for root in effect.referenced_nodes() {
                if graph_ops::pure_depends_on(graph, root, result)
                    && !graph_ops::pure_depends_on(graph, root, demand)
                {
                    return false;
                }
            }
        }
        for root in block.term.referenced_nodes() {
            if graph_ops::pure_depends_on(graph, root, result)
                && !graph_ops::pure_depends_on(graph, root, demand)
            {
                return false;
            }
        }
    }
    for route in output_routes {
        let root = route.source.value;
        if graph_ops::pure_depends_on(graph, root, result)
            && !graph_ops::pure_depends_on(graph, root, demand)
        {
            return false;
        }
    }
    true
}

pub(super) fn apply(inner: Program<Segmented>, candidate: Candidate) -> Program<Segmented> {
    let (region_name, input_nodes, input_elem_types, captures, producer_result) = {
        let graph = inner.body_graph(candidate.site).expect("indexed fusion body");
        let effect = &graph.skeleton.blocks[candidate.block].side_effects[candidate.effect];
        let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
            unreachable!();
        };
        let map = &op.lanes().maps[candidate.output];
        let indices = &map.input_indices;
        (
            inner.region(map.body.region).expect("map region").name.clone(),
            indices.iter().map(|index| effect.operand_nodes[index.index()]).collect::<Vec<_>>(),
            indices.iter().map(|index| op.lanes().inputs[index.index()].element()).collect::<Vec<_>>(),
            map.body.captures.clone(),
            effect.result.expect("indexed SegMap has no result"),
        )
    };

    let site = candidate.site;
    inner.rewrite_body(site, |body| {
        let rewrite_graph = |graph: &mut EGraph| {
            let mut args = smallvec::SmallVec::<[NodeId; 4]>::new();
            for (input, elem_ty) in input_nodes.iter().zip(&input_elem_types) {
                args.push(graph.intern_pure(
                    PureOp::Index,
                    smallvec![*input, candidate.index_value],
                    elem_ty.clone(),
                    None,
                ));
            }
            args.extend(captures.iter().copied());
            let result_ty = graph.nodes[candidate.index].ty.clone();
            let scalar = graph.intern_pure(PureOp::Call(region_name.clone()), args, result_ty, None);
            crate::egir::graph_ops::replace_all_references(graph, candidate.index, scalar);

            let block = &mut graph.skeleton.blocks[candidate.block];
            let removed_effects = block.side_effects[candidate.effect].effects;
            block.side_effects.remove(candidate.effect);
            if let Some((input, output)) = removed_effects {
                for effect in &mut block.side_effects[candidate.effect..] {
                    if let Some((effect_input, _)) = &mut effect.effects {
                        if *effect_input == output {
                            *effect_input = input;
                            break;
                        }
                    }
                }
            }
            scalar
        };
        match body {
            Body::Entry(mut entry) => {
                let scalar = rewrite_graph(&mut entry.graph);
                for route in entry.routes_mut() {
                    if route.source.value == candidate.index {
                        route.source.value = scalar;
                    }
                    route.writers.retain(|writer| *writer != OutputWriter::Value(producer_result));
                }
                Body::Entry(entry)
            }
            Body::Function(mut function) => {
                rewrite_graph(&mut function.graph);
                Body::Function(function)
            }
            Body::Constant(_) => unreachable!("indexed fusion never targets constants"),
        }
    })
}
