//! Scalar demand fusion for statically indexed pointwise producers.
//!
//! `Index(SegMap(f, inputs), k)` computes one element by indexing each source
//! and calling `f`. EGIR has explicit projections, use edges, and callable
//! regions, so this is a small graph rewrite rather than a TLC tree pass.

use smallvec::smallvec;

use crate::egir::program::{OutputRoute, OutputWriter, SemanticProgram, SemanticResourceRef};
use crate::egir::soac::screma;
use crate::egir::types::{
    EGraph, ENode, NodeId, PureOp, SegResourceAccessKind, SideEffectKind, Soac, SoacDestination,
};
use crate::flow::BlockId;
use crate::ssa::types::ConstantValue;

#[derive(Clone, Copy)]
enum Site {
    Entry(usize),
    Function(usize),
}

#[derive(Clone)]
struct Candidate {
    site: Site,
    block: BlockId,
    effect: usize,
    index: NodeId,
    index_value: NodeId,
    output: usize,
}

pub fn scalarize_indexed_segmap(inner: &mut SemanticProgram) -> bool {
    let Some(candidate) = find_candidate(inner) else {
        return false;
    };
    apply(inner, candidate);
    true
}

fn find_candidate(inner: &SemanticProgram) -> Option<Candidate> {
    for (index, entry) in inner.entry_points.iter().enumerate() {
        let output_resources = entry.outputs.iter().map(|output| output.resource).collect::<Vec<_>>();
        if let Some(candidate) = find_in_graph(
            &entry.graph,
            Site::Entry(index),
            &output_resources,
            &entry.output_routes,
        ) {
            return Some(candidate);
        }
    }
    for (index, function) in inner.functions.iter().enumerate() {
        if let Some(candidate) = find_in_graph(&function.graph, Site::Function(index), &[], &[]) {
            return Some(candidate);
        }
    }
    None
}

fn find_in_graph(
    graph: &EGraph,
    site: Site,
    output_resources: &[Option<SemanticResourceRef>],
    output_routes: &[OutputRoute],
) -> Option<Candidate> {
    for (block_id, block) in &graph.skeleton.blocks {
        for (effect_index, effect) in block.side_effects.iter().enumerate() {
            let SideEffectKind::Soac(
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
            ) = &effect.kind
            else {
                continue;
            };
            let indirect_output_resources: std::collections::HashSet<_> = output_slots
                .iter()
                .filter_map(|slot| output_resources.get(slot.0).copied().flatten())
                .collect();
            if maps.is_empty()
                || !maps.iter().all(|map| {
                    matches!(
                        map.destination,
                        SoacDestination::Fresh | SoacDestination::UniqueInput
                    )
                })
                || resources.iter().any(|resource| {
                    resource.access != SegResourceAccessKind::Read
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
                    } = definition
                    else {
                        return None;
                    };
                    let [base, index] = operands.as_slice() else {
                        return None;
                    };
                    if !is_static_index(graph, *index) {
                        return None;
                    }
                    projection_of(graph, *base, result).map(|output| (node, *index, output))
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

fn projection_of(graph: &EGraph, node: NodeId, root: NodeId) -> Option<usize> {
    match &graph.nodes[node] {
        ENode::Pure {
            op: PureOp::Project { index },
            operands,
        } if operands.first() == Some(&root) => Some(*index as usize),
        _ => None,
    }
}

fn is_static_index(graph: &EGraph, node: NodeId) -> bool {
    match &graph.nodes[node] {
        ENode::Constant(ConstantValue::I32(_) | ConstantValue::U32(_)) => true,
        ENode::Pure {
            op: PureOp::Int(_) | PureOp::Uint(_),
            operands,
        } => operands.is_empty(),
        _ => false,
    }
}

fn reaches(graph: &EGraph, start: NodeId, target: NodeId) -> bool {
    wyn_graph::reaches_ordered(start, target, wyn_graph::WalkOrder::DepthFirst, |node, out| {
        out.extend(graph.nodes[node].children());
    })
}

fn used_only_through(
    graph: &EGraph,
    producer_block: BlockId,
    producer_effect: usize,
    result: NodeId,
    demand: NodeId,
    output_routes: &[OutputRoute],
) -> bool {
    for (block_id, block) in &graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            if block_id == producer_block && index == producer_effect {
                continue;
            }
            for root in effect.referenced_nodes() {
                if reaches(graph, root, result) && !reaches(graph, root, demand) {
                    return false;
                }
            }
        }
        for root in block.term.referenced_nodes() {
            if reaches(graph, root, result) && !reaches(graph, root, demand) {
                return false;
            }
        }
    }
    for route in output_routes {
        let root = route.source.value;
        if reaches(graph, root, result) && !reaches(graph, root, demand) {
            return false;
        }
    }
    true
}

fn apply(inner: &mut SemanticProgram, candidate: Candidate) {
    let (region_name, input_nodes, input_elem_types, captures, producer_result) = {
        let graph = graph(inner, candidate.site);
        let effect = &graph.skeleton.blocks[candidate.block].side_effects[candidate.effect];
        let SideEffectKind::Soac(_, Soac::Screma(op)) = &effect.kind else {
            unreachable!();
        };
        let map = &op.lanes().maps[candidate.output];
        let indices = &map.input_indices;
        (
            inner.regions[&map.body.region].name.clone(),
            indices.iter().map(|index| effect.operand_nodes[index.index()]).collect::<Vec<_>>(),
            indices
                .iter()
                .map(|index| op.lanes().inputs[index.index()].element.clone())
                .collect::<Vec<_>>(),
            map.body.captures.clone(),
            effect.result.expect("indexed SegMap has no result"),
        )
    };

    let graph = graph_mut(inner, candidate.site);
    let mut args = smallvec::SmallVec::<[NodeId; 4]>::new();
    for (input, elem_ty) in input_nodes.iter().zip(&input_elem_types) {
        args.push(graph.intern_pure(
            PureOp::Index,
            smallvec![*input, candidate.index_value],
            elem_ty.clone(),
            None,
        ));
    }
    args.extend(captures);
    let result_ty = graph.types[&candidate.index].clone();
    let scalar = graph.intern_pure(PureOp::Call(region_name), args, result_ty, None);
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

    if let Site::Entry(index) = candidate.site {
        for route in &mut inner.entry_points[index].output_routes {
            if route.source.value == candidate.index {
                route.source.value = scalar;
            }
            route.writers.retain(|writer| *writer != OutputWriter::Value(producer_result));
        }
    }
}

fn graph(inner: &SemanticProgram, site: Site) -> &EGraph {
    match site {
        Site::Entry(index) => &inner.entry_points[index].graph,
        Site::Function(index) => &inner.functions[index].graph,
    }
}

fn graph_mut(inner: &mut SemanticProgram, site: Site) -> &mut EGraph {
    match site {
        Site::Entry(index) => &mut inner.entry_points[index].graph,
        Site::Function(index) => &mut inner.functions[index].graph,
    }
}
