//! Scalarize statically indexed demands of a canonical pointwise Screma.
//!
//! For `map(f, inputs)[i]`, index every source at `i` and invoke the complete
//! pre-lambda.  When all observable uses of the produced arrays are such
//! demands, the full-domain map and its storage disappear.  Multiple demands
//! are handled uniformly; profitability policy is separate from legality.

use std::collections::HashSet;

use polytype::Type;
use smallvec::{smallvec, SmallVec};

use crate::ast::TypeName;
use crate::egir::graph_ops;
use crate::egir::ir::{Body, BodySite, RealizedOutputRoute};
use crate::egir::program::{OutputWriter, SemanticResourceRef};
use crate::egir::reify::Segmented;
use crate::egir::soac::screma;
use crate::egir::types::{EGraph, ENode, NodeId, PureOp, ResourceAccess, SideEffectKind, Soac, SoacEffect};
use crate::flow::BlockId;
use crate::ssa::types::ConstantValue;

#[derive(Clone, Copy)]
struct Demand {
    index: NodeId,
    index_value: NodeId,
    output: usize,
}

#[derive(Clone)]
pub(crate) struct Candidate {
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
                    if !is_static_index(graph, *index_value) {
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
    demands: &[Demand],
    output_routes: &[RealizedOutputRoute],
) -> bool {
    let demand_nodes = demands.iter().map(|demand| demand.index).collect::<HashSet<_>>();
    let has_unindexed_path = |root| pure_depends_on_avoiding(graph, root, result, &demand_nodes);
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

/// Does `root` reach `target` without crossing a recognized indexed demand?
/// Treating demands as cut vertices rejects roots that also retain a separate
/// path to the full producer result.
fn pure_depends_on_avoiding(graph: &EGraph, root: NodeId, target: NodeId, cut: &HashSet<NodeId>) -> bool {
    let mut pending = vec![root];
    let mut visited = HashSet::new();
    while let Some(node) = pending.pop() {
        if cut.contains(&node) || !visited.insert(node) {
            continue;
        }
        if node == target {
            return true;
        }
        if let Some(definition) = graph.nodes.get(node) {
            pending.extend(definition.kind.children());
        }
    }
    false
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
    let region_name =
        pre.seg_body().map(|body| inner.region(body.region).expect("map pre-lambda region").name.clone());

    inner.rewrite_body(candidate.site, |body| {
        let rewrite_graph = |graph: &mut EGraph| {
            let mut replacements = Vec::with_capacity(candidate.demands.len());
            for demand in &candidate.demands {
                let arguments = input_nodes
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
                let results = if let Some(name) = &region_name {
                    let body = pre.seg_body().expect("region name without body");
                    let mut operands = SmallVec::<[NodeId; 4]>::from_vec(arguments);
                    operands.extend(body.captures.iter().copied());
                    let result = graph.intern_pure(
                        PureOp::Call(name.clone()),
                        operands,
                        lambda_return_type(&pre.result_types),
                        None,
                    );
                    unpack_result(graph, result, &pre.result_types)
                } else {
                    arguments
                };
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
        match body {
            Body::Entry(mut entry) => {
                let replacements = rewrite_graph(&mut entry.graph);
                for route in entry.routes_mut() {
                    if let Some((_, scalar)) =
                        replacements.iter().find(|(demand, _)| route.source.value == *demand)
                    {
                        route.source.value = *scalar;
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

fn lambda_return_type(types: &[Type<TypeName>]) -> Type<TypeName> {
    match types {
        [ty] => ty.clone(),
        _ => Type::Constructed(TypeName::Tuple(types.len()), types.to_vec()),
    }
}

fn unpack_result(graph: &mut EGraph, result: NodeId, types: &[Type<TypeName>]) -> Vec<NodeId> {
    match types {
        [_] => vec![result],
        _ => types
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                graph.intern_pure(
                    PureOp::Project { index: index as u32 },
                    smallvec![result],
                    ty.clone(),
                    None,
                )
            })
            .collect(),
    }
}
