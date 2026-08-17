//! Resolution of uniquely owned SOAC inputs into concrete in-place storage.
//!
//! TLC's ownership pass can mark a SOAC result as `UniqueInput`: overwriting
//! the operation's consumed input would be semantically legal because no
//! caller-visible alias must retain its old contents. That marker is a
//! capability, not yet a storage decision. Output realization and semantic
//! fusion can redirect a result to output storage or combine result routes.
//!
//! This pass runs over the final semantic use graph. It resolves each remaining
//! `UniqueInput` candidate to `InputBuffer` only when the input has physical
//! backing storage and is used by exactly one result of the operation. Every
//! other candidate becomes `Fresh`.
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
//! `*` lets TLC grant the map a `UniqueInput` capability after proving `a` is
//! dead after the map. This pass can resolve that capability to `InputBuffer`, allowing
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

use super::super::soac::{filter, screma};
use super::super::types::{
    EGraph, SideEffectKind, Soac, SoacDestination, SoacEffect, SoacPlacement, ValueId,
};
use super::ResourcesAllocated;

/// Resolve every outstanding unique-input capability to a physical destination.
pub(super) fn resolve_destinations(program: ResourcesAllocated) -> ResourcesAllocated {
    program.map_graphs(|_, mut graph| {
        resolve_graph_destinations(&mut graph);
        graph
    })
}

fn resolve_graph_destinations(graph: &mut EGraph) {
    let sites = graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(block, body)| (0..body.side_effects.len()).map(move |index| (block, index)))
        .collect::<Vec<_>>();
    for (block_id, effect_index) in sites {
        let screma_resolution = match &graph.skeleton.blocks[block_id].side_effects[effect_index].kind {
            SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                let effect = &graph.skeleton.blocks[block_id].side_effects[effect_index];
                let input = (op.inputs.len() == 1).then(|| effect.operands[0].value()).flatten();
                let reusable_input = input.filter(|&node| input_has_reusable_storage(graph, node));
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
                    if effect
                        .operands
                        .first()
                        .and_then(|input| input.value())
                        .is_some_and(|input| input_has_reusable_storage(graph, input))
                    {
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
    }
}
fn input_has_reusable_storage(graph: &EGraph, input: ValueId) -> bool {
    matches!(graph.operand_ref(input), super::super::types::OperandRef::View(_))
}
