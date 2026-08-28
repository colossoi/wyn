//! Scalarize producer-independent point demands of a canonical pointwise Screma.
//!
//! For `map(f, inputs)[i]`, index every source at `i` and invoke the complete
//! pre-lambda.  When all observable uses of the produced arrays are such
//! demands, the full-domain map and its storage disappear.  Multiple demands
//! are handled uniformly; profitability policy is separate from legality.

use crate::egir;
use std::collections::HashSet;

use crate::egir::graph_ops;
use crate::egir::ir::RealizedOutputRoute;
use crate::egir::reify::Segmented;
use crate::egir::soac::{lambda as lambda_ops, screma};
use crate::egir::types::{
    EGraph, PureOp, ResourceAccess, SegExtent, SegSpace, SideEffectKind, Soac, SoacEffect, ValueId,
    ValueKind,
};
use crate::flow::BlockId;
use crate::BindingRef;
use smallvec::smallvec;

use super::{support, FusionEffect};

#[derive(Clone, Copy)]
struct Demand {
    index: ValueId,
    index_value: ValueId,
    output: usize,
}

#[derive(Clone)]
pub(super) struct Candidate {
    producer: FusionEffect,
    demands: Vec<Demand>,
}

pub(super) fn analyze(inner: &Segmented) -> Option<Candidate> {
    super::bodies(inner).find_map(|(_, graph, entry)| {
        let output_resources = entry
            .map(|entry| entry.outputs.iter().map(|output| output.resource).collect::<Vec<_>>())
            .unwrap_or_default();
        let output_routes =
            entry.map(|entry| entry.routes().cloned().collect::<Vec<_>>()).unwrap_or_default();
        find_in_graph(graph, &output_resources, &output_routes)
    })
}

fn find_in_graph(
    graph: &EGraph,
    output_resources: &[Option<BindingRef>],
    output_routes: &[RealizedOutputRoute],
) -> Option<Candidate> {
    let live = graph_ops::reachable_execution_values_with_roots(
        graph,
        output_routes.iter().flat_map(RealizedOutputRoute::referenced_values),
    )
    .into_iter()
    .collect::<HashSet<_>>();
    for (block_id, block) in &graph.skeleton.blocks {
        for (effect_index, effect) in block.side_effects.iter().enumerate() {
            let SideEffectKind::Soac(SoacEffect(id, Soac::Screma(op))) = &effect.kind else {
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
                || resources.iter().any(|resource| {
                    resource.access != ResourceAccess::Read
                        && !indirect_output_resources.contains(&resource.resource)
                })
            {
                continue;
            }
            let Some(result) = effect.value_result() else {
                continue;
            };
            let demands = graph
                .nodes
                .iter()
                .filter_map(|(node, definition)| {
                    if !live.contains(&node) {
                        return None;
                    }
                    let ValueKind::Pure {
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
                producer: FusionEffect(*id),
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
    result: ValueId,
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
            if graph_ops::effect_value_inputs(graph, effect).into_iter().any(has_unindexed_path) {
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

pub(super) fn apply(inner: Segmented, candidate: Candidate) -> super::FusionResult<Segmented> {
    let location = candidate.producer.resolve(&inner)?;
    let (pre, input_nodes, producer_result) = {
        let Some(graph) = inner.body_graph(location.body) else {
            return Err(super::FusionError::MissingEffect(candidate.producer.0));
        };
        let Some(block) = graph.skeleton.blocks.get(location.block) else {
            return Err(super::FusionError::MissingEffect(candidate.producer.0));
        };
        let Some(effect) = block.side_effects.get(location.index) else {
            return Err(super::FusionError::MissingEffect(candidate.producer.0));
        };
        let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
            return Err(super::FusionError::InvalidCandidate(
                "indexed producer changed kind after candidate analysis".to_owned(),
            ));
        };
        let Some(input_operands) = effect.operands.get(..op.inputs.len()) else {
            return Err(super::FusionError::InvalidCandidate(
                "indexed producer has fewer operands than input types".to_owned(),
            ));
        };
        let input_nodes =
            input_operands.iter().map(|operand| operand.value()).collect::<Option<Vec<ValueId>>>();
        let Some(input_nodes) = input_nodes else {
            return Err(super::FusionError::InvalidCandidate(
                "indexed producer input uses the place channel".to_owned(),
            ));
        };
        let Some(producer_result) = effect.value_result() else {
            return Err(super::FusionError::InvalidCandidate(
                "indexed producer has no by-value result".to_owned(),
            ));
        };
        (op.form.pre.clone(), input_nodes, producer_result)
    };
    let callee = if let Some(body) = pre.seg_body() {
        let Some(region) = inner.region(body.region) else {
            return Err(super::FusionError::InvalidCandidate(
                "indexed producer lambda region is absent".to_owned(),
            ));
        };
        Some(region.clone())
    } else {
        None
    };

    inner.try_rewrite_body(location.body, |body| {
        support::try_rewrite_body_graph_with_entry(
            body,
            |graph| {
                let mut replacements = Vec::with_capacity(candidate.demands.len());
                for demand in &candidate.demands {
                    let arguments = input_nodes
                        .iter()
                        .zip(&pre.parameter_types)
                        .map(|(&input, elem_ty)| {
                            let value = graph.intern_pure(
                                PureOp::Index,
                                smallvec![input, demand.index_value],
                                elem_ty.clone(),
                                None,
                            );
                            graph.operand_ref(value)
                        })
                        .collect::<Vec<_>>();
                    let mut operands = arguments;
                    operands.extend_from_slice(pre.captures());
                    let results =
                        lambda_ops::emit_call(graph, location.block, &pre, callee.as_ref(), operands);
                    let Some(result) = results.get(demand.output) else {
                        return Err(super::FusionError::InvalidCandidate(
                            "indexed demand names an absent lambda result".to_owned(),
                        ));
                    };
                    let Ok(scalar) = egir::graph_ops::pack_result_values(graph, result) else {
                        return Err(super::FusionError::InvalidCandidate(
                            "indexed demand result is not available by value".to_owned(),
                        ));
                    };
                    graph.replace_value_references(demand.index, scalar);
                    replacements.push((demand.index, scalar));
                }

                let Some(block) = graph.skeleton.blocks.get_mut(location.block) else {
                    return Err(super::FusionError::MissingEffect(candidate.producer.0));
                };
                let Some(effect) = block.side_effects.get(location.index) else {
                    return Err(super::FusionError::MissingEffect(candidate.producer.0));
                };
                let removed_effects = effect.effects;
                block.side_effects.remove(location.index);
                if let Some((input, output)) = removed_effects {
                    for effect in &mut block.side_effects[location.index..] {
                        if let Some((effect_input, _)) = &mut effect.effects {
                            if *effect_input == output {
                                *effect_input = input;
                                break;
                            }
                        }
                    }
                }
                Ok(replacements)
            },
            |entry, replacements| {
                support::replace_route_sources(entry, &replacements);
                support::remove_value_writer(entry, producer_result);
                Ok(())
            },
        )
    })
}
