//! Resolution of uniquely owned SOAC inputs into concrete in-place storage.
//!
//! TLC's ownership pass can mark a SOAC result as `UniqueInput`: overwriting
//! the operation's consumed input would be semantically legal because no
//! caller-visible alias must retain its old contents. That marker is a
//! capability, not yet a storage decision. Output realization and semantic
//! fusion can redirect a result to output storage, combine several consumers
//! of one input, or otherwise change which values are live at the operation.
//!
//! This pass runs over the final semantic use graph. It resolves each remaining
//! `UniqueInput` candidate to `InputBuffer` only when the input has physical
//! backing storage, is used by exactly one result of the operation, and has no
//! observers after that operation. Every other candidate becomes `Fresh`.
//! Already-resolved destinations such as `OutputView` are preserved. Thus TLC
//! grants permission to consume an input; this module decides whether doing so
//! is a valid and useful physical allocation.
//!
//! For example, in this Wyn function:
//!
//! ```text
//! def increment(a: *[8]i32) [8]i32 =
//!   map(|x: i32| x + 1, a)
//! ```
//!
//! `*` lets TLC grant the map a `UniqueInput` capability. Since `a` has no later
//! observer, this pass can resolve that capability to `InputBuffer`, allowing
//! the map to return `a` after overwriting it.
//!
//! A later read of the original array instead requires fresh result storage:
//!
//! ```text
//! def increment_and_read(j: i32) ([8]i32, i32) =
//!   let a = [1, 2, 3, 4, 5, 6, 7, 8] in
//!   let incremented = map(|x: i32| x + 1, a) in
//!   (incremented, a[j])
//! ```
//!
//! Even a uniquely used input cannot be reused when it has no physical buffer:
//!
//! ```text
//! def increment_range(n: i32) [n]i32 =
//!   map(|x: i32| x + 1, 0 ..< n)
//! ```
//!
//! The range is virtual, so its result must be `Fresh`. Unique ownership
//! permits reuse; it does not promise mutation or manufacture backing storage.

use polytype::Type;

use super::super::graph_ops;
use super::super::soac::{filter, screma};
use super::super::types::{
    EGraph, ENode, NodeId, PureOp, SideEffectKind, Soac, SoacDestination, SoacEffect, SoacPlacement,
};
use super::ResourcesAllocated;
use crate::ast::TypeName;
use crate::flow::BlockId;
use crate::types::TypeExt;

/// Resolve every outstanding unique-input capability to a physical destination.
pub(super) fn resolve_destinations(program: ResourcesAllocated) -> ResourcesAllocated {
    program.map_graphs(|_, mut graph| {
        resolve_graph_destinations(&mut graph);
        graph
    })
}

fn resolve_graph_destinations(graph: &mut EGraph) {
    // Multi-block liveness needs block-parameter substitution. Stay sound and
    // conservative until that representation is needed by a reuse candidate.
    if graph.skeleton.blocks.len() != 1 {
        discard_unique_input_candidates(graph);
        return;
    }
    let block_id = graph.skeleton.entry;
    let uses = graph_ops::ValueUseIndex::build(graph);
    let effect_count = graph.skeleton.blocks[block_id].side_effects.len();
    for effect_index in 0..effect_count {
        let screma_resolution = match &graph.skeleton.blocks[block_id].side_effects[effect_index].kind {
            SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                let effect = &graph.skeleton.blocks[block_id].side_effects[effect_index];
                let input = (op.inputs.len() == 1).then(|| effect.operand_nodes[0]);
                let reusable_input = input.filter(|&node| {
                    input_has_reusable_storage(&graph.nodes[node].ty)
                        && input_has_no_later_observers(&uses, block_id, effect_index, node)
                });
                let single_array_result = op.form.post.result_types.len() == 1;
                Some(
                    op.result_state
                        .iter()
                        .enumerate()
                        .map(|(field, result)| {
                            if !result.destination.is_unplaced_unique_input() {
                                return result.destination;
                            }
                            if single_array_result
                                && matches!(op.form.result_id(field), Some(screma::ResultId::Post(0)))
                                && reusable_input.is_some()
                            {
                                result.destination.placed(SoacPlacement::InputBuffer)
                            } else {
                                SoacDestination::fresh()
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            }
            _ => None,
        };

        let filter_resolution = match &graph.skeleton.blocks[block_id].side_effects[effect_index].kind {
            SideEffectKind::Soac(SoacEffect(
                _,
                Soac::Filter(filter::Op {
                    state:
                        filter::SemanticState {
                            storage: filter::Output::Local { destination, .. },
                            ..
                        },
                    ..
                }),
            )) => {
                let effect = &graph.skeleton.blocks[block_id].side_effects[effect_index];
                destination.is_unplaced_unique_input().then(|| {
                    if effect.operand_nodes.first().is_some_and(|&input| {
                        input_has_reusable_storage(&graph.nodes[input].ty)
                            && input_has_no_later_observers(&uses, block_id, effect_index, input)
                    }) {
                        SoacDestination::unique_input().placed(SoacPlacement::InputBuffer)
                    } else {
                        SoacDestination::fresh()
                    }
                })
            }
            _ => None,
        };

        let effect = &mut graph.skeleton.blocks[block_id].side_effects[effect_index];
        match &mut effect.kind {
            SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                if let Some(destinations) = screma_resolution {
                    for (result, destination) in op.result_state.iter_mut().zip(destinations) {
                        result.destination = destination;
                    }
                }
            }
            SideEffectKind::Soac(SoacEffect(
                _,
                Soac::Filter(filter::Op {
                    state:
                        filter::SemanticState {
                            storage: filter::Output::Local { destination, .. },
                            ..
                        },
                    ..
                }),
            )) => {
                if let Some(resolved) = filter_resolution {
                    *destination = resolved;
                }
            }
            _ => {}
        }
        retype_reused_results(graph, block_id, effect_index);
    }
}
fn retype_reused_results(graph: &mut EGraph, block: BlockId, effect_index: usize) {
    let effect = &graph.skeleton.blocks[block].side_effects[effect_index];
    let Some(result) = effect.result else {
        return;
    };
    let projections: Vec<_> = graph
        .nodes
        .iter()
        .filter_map(|(node, definition)| match &definition.kind {
            ENode::Pure {
                op: PureOp::Project { index },
                operands,
            } if operands.as_slice() == [result] => Some((node, *index as usize)),
            _ => None,
        })
        .collect();
    let Type::Constructed(TypeName::Tuple(_), mut result_types) = graph.nodes[result].ty.clone() else {
        return;
    };
    for (projection, field) in &projections {
        if let Some(ty) = result_types.get_mut(*field) {
            *ty = graph.nodes[*projection].ty.clone();
        }
    }

    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
        return;
    };
    if op.inputs.len() != 1 || op.form.post.result_types.len() != 1 {
        return;
    }
    let Some(field) = (0..op.result_count()).find(|&field| {
        matches!(op.form.result_id(field), Some(screma::ResultId::Post(0)))
            && op.destination(field).is_some_and(SoacDestination::is_input_buffer)
    }) else {
        return;
    };
    result_types[field] = op.inputs[0].array.clone();

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
fn discard_unique_input_candidates(graph: &mut EGraph) {
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for effect in &mut block.side_effects {
            match &mut effect.kind {
                SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                    for result in &mut op.result_state {
                        if result.destination.is_unplaced_unique_input() {
                            result.destination.make_fresh();
                        }
                    }
                }
                SideEffectKind::Soac(SoacEffect(
                    _,
                    Soac::Filter(filter::Op {
                        state:
                            filter::SemanticState {
                                storage: filter::Output::Local { destination, .. },
                                ..
                            },
                        ..
                    }),
                )) if destination.is_unplaced_unique_input() => {
                    destination.make_fresh();
                }
                _ => {}
            }
        }
    }
}
fn input_has_no_later_observers(
    uses: &graph_ops::ValueUseIndex,
    block: BlockId,
    index: usize,
    input: NodeId,
) -> bool {
    let observers = uses.pure_observers(input);
    !observers.effect_sites().any(|site| site.block == block && site.index > index)
        && !observers.terminator_blocks().any(|observer| observer == block)
}

fn input_has_reusable_storage(ty: &Type<TypeName>) -> bool {
    match ty.array_variant() {
        Some(Type::Constructed(TypeName::ArrayVariantVirtual, _)) => return false,
        Some(Type::Constructed(TypeName::ArrayVariantView, _)) => return true,
        _ => {}
    }
    let runtime_sized =
        ty.array_size().is_some_and(|size| !matches!(size, Type::Constructed(TypeName::Size(_), _)));
    !runtime_sized || crate::types::array_view_buffer(ty).is_some()
}
