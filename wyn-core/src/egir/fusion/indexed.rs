//! Scalarize producer-independent point demands of a canonical pointwise Screma.
//!
//! For `map(f, inputs)[i]`, index every source at `i` and invoke the complete
//! pre-lambda.  When all observable uses of the produced arrays are such
//! demands, the full-domain map and its storage disappear.  Multiple demands
//! are handled uniformly; profitability policy is separate from legality.

use std::collections::HashSet;

use crate::egir::graph_ops;
use crate::egir::ir::{BodySite, RealizedOutputRoute};
use crate::egir::program::SemanticResourceRef;
use crate::egir::reify::Segmented;
use crate::egir::soac::{lambda as lambda_ops, screma};
use crate::egir::types::{
    EGraph, ENode, NodeId, PureOp, ResourceAccess, SegExtent, SegSpace, SideEffectKind, Soac, SoacEffect,
};
use crate::flow::BlockId;
use smallvec::smallvec;

use super::support;

#[derive(Clone, Copy)]
struct Demand {
    index: NodeId,
    index_value: NodeId,
    output: usize,
}

#[derive(Clone)]
pub(super) struct Candidate {
    site: BodySite,
    block: BlockId,
    effect: usize,
    demands: Vec<Demand>,
}

pub(super) fn analyze(inner: &Segmented) -> Option<Candidate> {
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
    let live = graph_ops::reachable_execution_values(graph).into_iter().collect::<HashSet<_>>();
    for (block_id, block) in &graph.skeleton.blocks {
        for (effect_index, effect) in block.side_effects.iter().enumerate() {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                continue;
            };
            let screma::SemanticState::Segmented {
                space,
                output_slots,
                resources,
                ..
            } = op.semantic_state()
            else {
                continue;
            };
            let indirect_output_resources = output_slots
                .iter()
                .filter_map(|slot| output_resources.get(slot.0).copied().flatten())
                .collect::<HashSet<_>>();
            if !op.is_map()
                || !op.form.post.is_identity()
                || op.form.pre.result_types.is_empty()
                || op.result_state.iter().any(|result| !result.destination.is_unplaced())
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
                    if !live.contains(&node) {
                        return None;
                    }
                    let ENode::Pure {
                        op: PureOp::Index,
                        operands,
                    } = &definition.kind
                    else {
                        return None;
                    };
                    let [base, index_value] = operands.as_slice() else {
                        return None;
                    };
                    if graph_ops::value_depends_on(graph, *index_value, result) {
                        return None;
                    }
                    let field = graph_ops::projection_index(graph, *base, result)?;
                    let screma::ResultId::Post(output) = op.form.result_id(field)? else {
                        return None;
                    };
                    (output < op.form.pre.result_types.len()).then_some(Demand {
                        index: node,
                        index_value: *index_value,
                        output,
                    })
                })
                .collect::<Vec<_>>();
            if demands.is_empty()
                || !point_scalarization_is_profitable(space, demands.len())
                || !used_only_through(graph, block_id, effect_index, result, &demands, output_routes)
            {
                continue;
            }
            return Some(Candidate {
                site,
                block: block_id,
                effect: effect_index,
                demands,
            });
        }
    }
    None
}

/// Scalarization evaluates the complete map body once per point demand. A
/// fixed domain supplies its own break-even point. Without a static extent, cap
/// the number of copies so an unknown-size producer cannot cause unbounded IR
/// and runtime-work duplication.
fn point_scalarization_is_profitable(space: &SegSpace, demands: usize) -> bool {
    const UNKNOWN_DOMAIN_DEMAND_LIMIT: usize = 2;

    match space.dims() {
        [SegExtent::Fixed(elements)] => demands as u64 <= u64::from(*elements),
        _ => demands <= UNKNOWN_DOMAIN_DEMAND_LIMIT,
    }
}
fn used_only_through(
    graph: &EGraph,
    producer_block: BlockId,
    producer_effect: usize,
    result: NodeId,
    demands: &[Demand],
    output_routes: &[RealizedOutputRoute],
) -> bool {
    let demand_nodes = demands.iter().map(|demand| demand.index).collect::<HashSet<_>>();
    let has_unindexed_path = |root| support::pure_depends_on_avoiding(graph, root, result, &demand_nodes);
    for (block_id, block) in &graph.skeleton.blocks {
        for (index, effect) in block.side_effects.iter().enumerate() {
            if block_id == producer_block && index == producer_effect {
                continue;
            }
            if effect.referenced_nodes().any(has_unindexed_path) {
                return false;
            }
        }
        if block.term.referenced_nodes().into_iter().any(has_unindexed_path) {
            return false;
        }
    }
    if output_routes.iter().map(|route| route.source.value).any(has_unindexed_path) {
        return false;
    }
    true
}

pub(super) fn apply(inner: Segmented, candidate: Candidate) -> Segmented {
    let (pre, input_nodes, producer_result) = {
        let graph = inner.body_graph(candidate.site).expect("indexed fusion body");
        let effect = &graph.skeleton.blocks[candidate.block].side_effects[candidate.effect];
        let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
            unreachable!();
        };
        (
            op.form.pre.clone(),
            effect.operand_nodes[..op.inputs.len()].to_vec(),
            effect.result.expect("indexed map has no result"),
        )
    };

    inner.rewrite_body(candidate.site, |body| {
        let rewrite_graph = |graph: &mut EGraph| {
            let mut replacements = Vec::with_capacity(candidate.demands.len());
            for demand in &candidate.demands {
                let mut arguments = input_nodes
                    .iter()
                    .zip(&pre.parameter_types)
                    .map(|(&input, elem_ty)| {
                        graph.intern_pure(
                            PureOp::Index,
                            smallvec![input, demand.index_value],
                            elem_ty.clone(),
                            None,
                        )
                    })
                    .collect::<Vec<_>>();
                arguments.extend_from_slice(pre.captures());
                let results = lambda_ops::emit_call(graph, &pre, arguments);
                let scalar = results[demand.output];
                graph_ops::replace_all_references(graph, demand.index, scalar);
                replacements.push((demand.index, scalar));
            }

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
            replacements
        };
        support::rewrite_body_graph_with_entry(body, rewrite_graph, |entry, replacements| {
            support::replace_route_sources(entry, &replacements);
            support::remove_value_writer(entry, producer_result);
        })
    })
}
