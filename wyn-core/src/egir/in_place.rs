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

use std::collections::{HashMap, HashSet};

use polytype::Type;

use super::graph_ops;
use super::program::AllocatedProgram;
use super::soac::filter;
use super::types::{
    EGraph, ENode, NodeId, PureOp, SideEffectKind, Soac, SoacDestination, SoacEffect, SoacPlacement,
};
use crate::ast::TypeName;
use crate::flow::BlockId;
use crate::types::TypeExt;

/// Resolve every outstanding unique-input capability to a physical destination.
pub(super) fn resolve_destinations(program: &mut AllocatedProgram) {
    for entry in &mut program.entry_points {
        resolve_graph_destinations(&mut entry.graph);
    }
    for function in &mut program.functions {
        resolve_graph_destinations(&mut function.graph);
    }
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
        let (
            operands,
            map_inputs,
            operator_inputs,
            map_destinations,
            accumulator_destinations,
            filter_has_candidate,
        ) = {
            let effect = &graph.skeleton.blocks[block_id].side_effects[effect_index];
            match &effect.kind {
                SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => (
                    effect.operand_nodes.to_vec(),
                    op.lanes().maps.iter().map(|map| map.input_indices.clone()).collect(),
                    op.operators()
                        .into_iter()
                        .map(|operator| operator.input_indices.clone())
                        .collect::<Vec<_>>(),
                    op.lanes().maps.iter().map(|map| map.destination).collect(),
                    op.operators().into_iter().map(|operator| operator.destination).collect(),
                    false,
                ),
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
                )) => (
                    effect.operand_nodes.to_vec(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    destination.is_unplaced_unique_input(),
                ),
                _ => continue,
            }
        };

        let mut input_use_counts = HashMap::<usize, usize>::new();
        for &input in map_inputs.iter().flatten().chain(operator_inputs.iter().flatten()) {
            *input_use_counts.entry(input.index()).or_default() += 1;
        }
        let mut claimed_inputs = HashSet::<usize>::new();
        let can_reuse = |input: usize, claimed_inputs: &mut HashSet<usize>| {
            input_use_counts.get(&input) == Some(&1)
                && claimed_inputs.insert(input)
                && operands.get(input).is_some_and(|&node| {
                    input_has_reusable_storage(&graph.types[&node])
                        && input_has_no_later_observers(&uses, block_id, effect_index, node)
                })
        };

        let resolved_maps: Vec<_> = map_destinations
            .iter()
            .enumerate()
            .map(|(lane, destination)| {
                if !destination.is_unplaced_unique_input() {
                    return *destination;
                }
                map_inputs
                    .get(lane)
                    .and_then(|inputs| inputs.first())
                    .copied()
                    .filter(|input| can_reuse(input.index(), &mut claimed_inputs))
                    .map_or_else(SoacDestination::fresh, |_| {
                        destination.placed(SoacPlacement::InputBuffer)
                    })
            })
            .collect();
        let resolved_accumulators: Vec<_> = accumulator_destinations
            .iter()
            .enumerate()
            .map(|(operator, destination)| {
                if !destination.is_unplaced_unique_input() {
                    return *destination;
                }
                operator_inputs
                    .get(operator)
                    .and_then(|inputs| inputs.first())
                    .copied()
                    .filter(|input| can_reuse(input.index(), &mut claimed_inputs))
                    .map_or_else(SoacDestination::fresh, |_| {
                        destination.placed(SoacPlacement::InputBuffer)
                    })
            })
            .collect();

        let resolved_filter = filter_has_candidate.then(|| {
            if operands.first().is_some_and(|&input| {
                input_has_reusable_storage(&graph.types[&input])
                    && input_has_no_later_observers(&uses, block_id, effect_index, input)
            }) {
                SoacDestination::unique_input().placed(SoacPlacement::InputBuffer)
            } else {
                SoacDestination::fresh()
            }
        });
        let effect = &mut graph.skeleton.blocks[block_id].side_effects[effect_index];
        match &mut effect.kind {
            SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                for (map, destination) in op.lanes_mut().maps.iter_mut().zip(resolved_maps) {
                    map.destination = destination;
                }
                for (operator, destination) in op.operators_mut().into_iter().zip(resolved_accumulators) {
                    operator.destination = destination;
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
                if let Some(resolved) = resolved_filter {
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
        .filter_map(|(node, definition)| match definition {
            ENode::Pure {
                op: PureOp::Project { index },
                operands,
            } if operands.as_slice() == [result] => Some((node, *index as usize)),
            _ => None,
        })
        .collect();
    let (result_types, changed) = {
        let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
            return;
        };
        let mut retyped = op.result_types();
        // Output realization may already have changed a projected field to a
        // storage view. Preserve those later EGIR decisions while changing
        // only fields whose uniqueness candidate became a physical reuse.
        for (projection, field) in &projections {
            retyped[*field] = graph.types[projection].clone();
        }
        let mut changed = false;
        for (lane, map) in op.lanes().maps.iter().enumerate() {
            if map.destination.is_input_buffer() {
                if let Some(input) = map.input_indices.first() {
                    retyped[lane] = op.lanes().inputs[input.index()].array.clone();
                    changed = true;
                }
            }
        }
        for (operator_index, operator) in op.operators().into_iter().enumerate() {
            if operator.destination.is_input_buffer() {
                if let Some(input) = operator.input_indices.first() {
                    retyped[op.lanes().maps.len() + operator_index] =
                        op.lanes().inputs[input.index()].array.clone();
                    changed = true;
                }
            }
        }
        (retyped, changed)
    };
    if !changed {
        return;
    }
    if let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) =
        &mut graph.skeleton.blocks[block].side_effects[effect_index].kind
    {
        op.set_result_types(&result_types);
    }
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
                    for map in &mut op.lanes_mut().maps {
                        if map.destination.is_unplaced_unique_input() {
                            map.destination.make_fresh();
                        }
                    }
                    for operator in op.operators_mut() {
                        if operator.destination.is_unplaced_unique_input() {
                            operator.destination.make_fresh();
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
