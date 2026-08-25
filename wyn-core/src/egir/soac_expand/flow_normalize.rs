use crate::ast::TypeName;
use crate::egir::graph_ops::{
    bind_by_value_result, detached_alloca, emit_result_to_place, fold_exposed_projections,
    rebind_result_projection_references, rewrite_result_store_consumers,
};
use crate::egir::ir::Language;
use crate::egir::types::{
    by_value_function_result, EGraph, EffectToken, FlowValueId, Physical, PlaceDestination, PlaceId,
    ResultBinding, ResultDestination, SkeletonTerminator, ValueId, ValueKind, WynLanguage,
};
use crate::flow::BlockId;
use crate::types::{self, TypeExt};
use crate::SortedSet;
use polytype::Type;
use wyn_base::IdSource;

#[derive(Clone, Copy)]
enum EdgeArm {
    Branch,
    Then,
    Else,
}

#[derive(Clone)]
struct IncomingEdge {
    predecessor: BlockId,
    arm: EdgeArm,
    argument: ValueId,
    arguments: Vec<FlowValueId>,
}

struct ArrayLeaf {
    ty: Type<TypeName>,
    place: PlaceId,
    view: ValueId,
}

pub(super) fn normalize_place_backed_flow(
    graph: &mut EGraph<Physical>,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    while let Some((block, slot, ty)) = next_materialized_flow_parameter(graph) {
        normalize_parameter(graph, block, slot, ty, effect_ids)?;
    }
    Ok(())
}

fn next_materialized_flow_parameter(graph: &EGraph<Physical>) -> Option<(BlockId, usize, Type<TypeName>)> {
    graph.skeleton.blocks.iter().find_map(|(block, body)| {
        body.params.iter().enumerate().find_map(|(slot, parameter)| {
            let ty = graph.nodes[parameter.value()].ty.clone();
            type_contains_addressable_array(&ty).then_some((block, slot, ty))
        })
    })
}

fn type_contains_addressable_array(ty: &Type<TypeName>) -> bool {
    if ty.array_variant().is_some() {
        return WynLanguage::is_materialized_aggregate(ty);
    }
    WynLanguage::product_fields(ty).is_some_and(|fields| fields.iter().any(type_contains_addressable_array))
}

fn normalize_parameter(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    slot: usize,
    ty: Type<TypeName>,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    let incoming = incoming_edges(graph, block, slot)?;
    if incoming.is_empty() {
        return Err(format!(
            "materialized block parameter {slot} in {block:?} has no incoming edge"
        ));
    }

    let abi = by_value_function_result::<WynLanguage>(ty.clone());
    let leaf_types = abi.destination_leaves().into_iter().map(|leaf| leaf.ty().clone()).collect::<Vec<_>>();
    let incoming_leaves = incoming
        .iter()
        .map(|edge| {
            let binding = bind_by_value_result(graph, &abi, edge.argument);
            let values = binding.values();
            if values.len() != leaf_types.len() {
                return Err("flow argument does not match its parameter result tree".to_owned());
            }
            Ok(values)
        })
        .collect::<Result<Vec<_>, String>>()?;
    fold_exposed_projections(graph);
    let old_parameter = graph.skeleton.blocks[block].params[slot].value();

    let mut replacements = Vec::with_capacity(leaf_types.len());
    let mut scalar_parameters = Vec::new();
    let mut array_leaves = Vec::with_capacity(leaf_types.len());
    for (leaf, leaf_ty) in leaf_types.iter().enumerate() {
        if type_contains_addressable_array(leaf_ty) {
            let values = incoming_leaves
                .iter()
                .map(|values| graph.canonical_value(values[leaf]))
                .collect::<Vec<_>>();
            if values.windows(2).all(|pair| pair[0] == pair[1])
                && matches!(graph.nodes[values[0]].kind(), ValueKind::PlaceView { .. })
            {
                replacements.push(values[0]);
                let ValueKind::PlaceView { place } = graph.nodes[values[0]].kind() else {
                    unreachable!()
                };
                array_leaves.push(Some(ArrayLeaf {
                    ty: leaf_ty.clone(),
                    place: *place,
                    view: values[0],
                }));
                continue;
            }

            let (place, effect) = detached_alloca(graph, leaf_ty.clone(), effect_ids, None);
            graph.skeleton.blocks[graph.skeleton.entry].side_effects.insert(0, effect);
            let view_ty = types::view_array_of(leaf_ty, types::no_buffer());
            let view = graph.add_place_view(place, view_ty, None).value();
            replacements.push(view);
            array_leaves.push(Some(ArrayLeaf {
                ty: leaf_ty.clone(),
                place,
                view,
            }));
        } else {
            let parameter = graph.add_block_param(block, leaf_ty.clone());
            replacements.push(parameter);
            scalar_parameters.push(leaf);
            array_leaves.push(None);
        }
    }

    let rebuilt = abi.bind(
        |return_slot, _| replacements[return_slot.index()],
        |_| unreachable!("by-value flow ABI has no destination parameter"),
    );
    let mut leaf = 0usize;
    let rebuilt = rebuilt.map_destinations(|_, destination| {
        let destination = array_leaves[leaf]
            .as_ref()
            .map(|array| ResultDestination::Place(PlaceDestination::Fixed(array.place)))
            .unwrap_or_else(|| destination.clone());
        leaf += 1;
        destination
    });
    rewrite_result_store_consumers(graph, old_parameter, &rebuilt, effect_ids)?;
    rebind_result_projection_references(graph, old_parameter, &rebuilt)?;
    fold_exposed_projections(graph);

    for (edge_index, edge) in incoming.iter().enumerate() {
        let mut arguments = edge.arguments.clone();
        arguments.extend(
            scalar_parameters.iter().map(|leaf| graph.admit_flow_value(incoming_leaves[edge_index][*leaf])),
        );

        let transfers = array_leaves
            .iter()
            .enumerate()
            .filter_map(|(leaf, destination)| {
                destination.as_ref().map(|destination| (incoming_leaves[edge_index][leaf], destination))
            })
            .filter(|(source, destination)| graph.canonical_value(*source) != destination.view)
            .collect::<Vec<_>>();

        if transfers.is_empty() {
            set_edge_arguments(graph, edge, arguments);
            continue;
        }

        let edge_block = graph.skeleton.create_block();
        redirect_edge(graph, edge, edge_block);
        let mut tail = edge_block;
        for (source, destination) in transfers {
            tail = emit_array_transfer(graph, tail, source, destination, effect_ids)?;
        }
        graph.skeleton.blocks[tail].term = SkeletonTerminator::Branch {
            target: block,
            args: arguments,
        };
    }

    graph.remove_block_param_slots(block, &SortedSet::from([slot]));
    fold_exposed_projections(graph);
    Ok(())
}

fn incoming_edges(
    graph: &EGraph<Physical>,
    target: BlockId,
    slot: usize,
) -> Result<Vec<IncomingEdge>, String> {
    let mut incoming = Vec::new();
    for (predecessor, body) in &graph.skeleton.blocks {
        match &body.term {
            SkeletonTerminator::Branch {
                target: branch_target,
                args,
            } if *branch_target == target => {
                incoming.push(edge(predecessor, EdgeArm::Branch, args, slot)?);
            }
            SkeletonTerminator::CondBranch {
                then_target,
                then_args,
                else_target,
                else_args,
                ..
            } => {
                if *then_target == target {
                    incoming.push(edge(predecessor, EdgeArm::Then, then_args, slot)?);
                }
                if *else_target == target {
                    incoming.push(edge(predecessor, EdgeArm::Else, else_args, slot)?);
                }
            }
            _ => {}
        }
    }
    Ok(incoming)
}

fn edge(
    predecessor: BlockId,
    arm: EdgeArm,
    args: &[FlowValueId],
    slot: usize,
) -> Result<IncomingEdge, String> {
    let argument = args
        .get(slot)
        .ok_or_else(|| format!("incoming edge to block parameter {slot} has no argument"))?
        .value();
    Ok(IncomingEdge {
        predecessor,
        arm,
        argument,
        arguments: args.to_vec(),
    })
}

fn set_edge_arguments(graph: &mut EGraph<Physical>, edge: &IncomingEdge, arguments: Vec<FlowValueId>) {
    match (&mut graph.skeleton.blocks[edge.predecessor].term, edge.arm) {
        (SkeletonTerminator::Branch { args, .. }, EdgeArm::Branch) => *args = arguments,
        (SkeletonTerminator::CondBranch { then_args, .. }, EdgeArm::Then) => *then_args = arguments,
        (SkeletonTerminator::CondBranch { else_args, .. }, EdgeArm::Else) => *else_args = arguments,
        _ => panic!("incoming CFG edge changed during place-backed flow normalization"),
    }
}

fn redirect_edge(graph: &mut EGraph<Physical>, edge: &IncomingEdge, target: BlockId) {
    match (&mut graph.skeleton.blocks[edge.predecessor].term, edge.arm) {
        (
            SkeletonTerminator::Branch {
                target: branch_target,
                args,
            },
            EdgeArm::Branch,
        ) => {
            *branch_target = target;
            args.clear();
        }
        (
            SkeletonTerminator::CondBranch {
                then_target,
                then_args,
                ..
            },
            EdgeArm::Then,
        ) => {
            *then_target = target;
            then_args.clear();
        }
        (
            SkeletonTerminator::CondBranch {
                else_target,
                else_args,
                ..
            },
            EdgeArm::Else,
        ) => {
            *else_target = target;
            else_args.clear();
        }
        _ => panic!("incoming CFG edge changed during place-backed flow normalization"),
    }
}

fn emit_array_transfer(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    source: ValueId,
    destination: &ArrayLeaf,
    effect_ids: &mut IdSource<EffectToken>,
) -> Result<BlockId, String> {
    let source = graph.canonical_value(source);
    let result = ResultBinding::destination(destination.ty.clone(), ResultDestination::ReturnValue(source));
    emit_result_to_place(graph, block, &result, destination.place, effect_ids, None)
}
