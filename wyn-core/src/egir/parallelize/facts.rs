//! Checked entry facts and semantic-operation eligibility classification.
//!
//! Facts in this module are graph-local and short-lived: analysis rebuilds
//! them for a projected entry and emission consumes them before mutating that
//! graph. Persistent parallel/serial decisions belong to `planning` instead.

#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use super::*;
fn segmented_screma_effect(graph: &EGraph) -> Option<(BlockId, usize, &SideEffect)> {
    graph.skeleton.blocks.iter().find_map(|(block, contents)| {
        contents.side_effects.iter().enumerate().find_map(|(index, effect)| {
            matches!(
                &effect.kind,
                SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op)))
                    if matches!(
                        op.semantic_state(),
                        screma::SemanticState::Segmented {
                            placement: screma::Placement::Kernel,
                            ..
                        }
                    )
            )
            .then_some((block, index, effect))
        })
    })
}

pub(super) fn segmented_recipe_effect(
    entry: &crate::egir::program::PlannedEntry,
) -> Option<(BlockId, usize, &SideEffect)> {
    if let Some(effect) = segmented_screma_effect(&entry.graph) {
        return Some(effect);
    }
    if !matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
        return None;
    }
    let mut promoted = None;
    for (block, contents) in &entry.graph.skeleton.blocks {
        for (index, effect) in contents.side_effects.iter().enumerate() {
            let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
                continue;
            };
            if matches!(
                op.semantic_state(),
                screma::SemanticState::Segmented {
                    placement: screma::Placement::LaneLocal,
                    output_slots,
                    ..
                } if !output_slots.is_empty()
            ) && matches!(op, screma::Op::Reduce { .. } | screma::Op::Scan { .. })
            {
                if promoted.is_some() {
                    return None;
                }
                promoted = Some((block, index, effect));
            }
        }
    }
    promoted
}

pub(crate) fn parallel_recipe_effect(
    entry: &crate::egir::program::PlannedEntry,
) -> Option<(BlockId, usize, &SideEffect)> {
    segmented_recipe_effect(entry).or_else(|| {
        entry.graph.skeleton.blocks.iter().find_map(|(block, contents)| {
            contents.side_effects.iter().enumerate().find_map(|(index, effect)| {
                matches!(&effect.kind, SideEffectKind::Soac(SoacEffect(_, Soac::Filter(_))))
                    .then_some((block, index, effect))
            })
        })
    })
}

pub(super) fn make_screma_serial(graph: &mut EGraph, block_id: BlockId, index: usize) -> error::Result<()> {
    let effect = graph
        .skeleton
        .get_effect_mut(SideEffectSite {
            block: block_id,
            index,
        })
        .ok_or_else(|| {
            error::ParallelizeError::Invalid(format!("stale segmented effect site {block_id:?}:{index}"))
        })?;
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &mut effect.kind else {
        return Err("segmented effect site no longer contains a Screma operation".into());
    };
    *op.semantic_state_mut() = screma::SemanticState::Serial;
    Ok(())
}

pub(super) fn semantic_effect(graph: &EGraph, block: BlockId, index: usize) -> error::Result<&SideEffect> {
    graph.skeleton.get_effect(SideEffectSite { block, index }).ok_or_else(|| {
        error::ParallelizeError::Invalid(format!("stale semantic effect site {block:?}:{index}"))
    })
}

pub(super) fn semantic_effect_mut(
    graph: &mut EGraph,
    block: BlockId,
    index: usize,
) -> error::Result<&mut SideEffect> {
    graph.skeleton.get_effect_mut(SideEffectSite { block, index }).ok_or_else(|| {
        error::ParallelizeError::Invalid(format!("stale semantic effect site {block:?}:{index}"))
    })
}

pub(super) fn semantic_node_type(graph: &EGraph, node: NodeId) -> error::Result<Type<TypeName>> {
    graph
        .types
        .get(&node)
        .cloned()
        .ok_or_else(|| error::ParallelizeError::Invalid(format!("semantic node {node:?} has no type")))
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SegScratchFamily {
    Reduce,
    Scan,
}

/// Classify the eligibility gates shared by candidate selection and lowering.
pub(super) fn seg_recipe_family(se: &SideEffect) -> std::result::Result<SegScratchFamily, FallbackReason> {
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &se.kind else {
        return Err(FallbackReason::UnsupportedOperationShape);
    };
    let valid_placement = match op.semantic_state() {
        screma::SemanticState::Segmented {
            placement: screma::Placement::Kernel,
            ..
        } => true,
        screma::SemanticState::Segmented {
            placement: screma::Placement::LaneLocal,
            output_slots,
            ..
        } => !output_slots.is_empty() && matches!(op, screma::Op::Reduce { .. } | screma::Op::Scan { .. }),
        screma::SemanticState::Serial => false,
    };
    if !valid_placement {
        return Err(FallbackReason::UnsupportedPlacement);
    }
    let lanes = op.lanes();
    let operators = op.operators();
    let maps_are_output_views = lanes.maps.iter().all(|map| map.destination.is_output_view());
    match op {
        screma::Op::Reduce { .. } => {
            if operators.iter().any(|op| !op.combine.captures.is_empty()) {
                return Err(FallbackReason::UnsupportedCaptures);
            }
            if lanes.inputs.is_empty() {
                return Err(FallbackReason::UnsupportedOperationShape);
            }
            if !maps_are_output_views || !operators.iter().all(|op| op.destination.is_unplaced_fresh()) {
                return Err(FallbackReason::UnsupportedDestination);
            }
            Ok(SegScratchFamily::Reduce)
        }
        screma::Op::Scan { .. } => {
            if operators.len() != 1 || lanes.inputs.len() != 1 {
                return Err(FallbackReason::UnsupportedOperationShape);
            }
            if !operators[0].combine.captures.is_empty() {
                return Err(FallbackReason::UnsupportedCaptures);
            }
            if !maps_are_output_views || !operators.iter().all(|op| op.destination.is_output_view()) {
                return Err(FallbackReason::UnsupportedDestination);
            }
            Ok(SegScratchFamily::Scan)
        }
        screma::Op::Map { .. } | screma::Op::Composite { .. } => {
            Err(FallbackReason::UnsupportedOperationShape)
        }
    }
}
