//! Checked entry facts and semantic-operation eligibility classification.
//!
//! Facts in this module are graph-local and short-lived: analysis rebuilds
//! them for a projected entry and emission consumes them before mutating that
//! graph. Persistent parallel/serial decisions belong to `planning` instead.

use super::*;

pub(super) struct LocatedScrema<'a> {
    pub site: SideEffectSite,
    pub effect: &'a SideEffect,
    pub owner: SemanticOpId,
    pub op: &'a screma::Op<crate::egir::types::Semantic>,
}

pub(super) struct LocatedReduce<'a> {
    pub site: SideEffectSite,
    pub effect: &'a SideEffect,
    pub owner: SemanticOpId,
    pub lanes: &'a screma::Lanes,
    pub operators: &'a screma::NonEmpty<screma::Operator>,
    op: &'a screma::Op<crate::egir::types::Semantic>,
}

pub(super) struct LocatedScan<'a> {
    pub site: SideEffectSite,
    pub effect: &'a SideEffect,
    pub owner: SemanticOpId,
    pub lanes: &'a screma::Lanes,
    pub operators: &'a screma::NonEmpty<screma::Operator>,
    op: &'a screma::Op<crate::egir::types::Semantic>,
}

pub(super) enum SegmentedRecipe<'a> {
    Reduce(LocatedReduce<'a>),
    Scan(LocatedScan<'a>),
    Map,
    Composite(SideEffectSite),
}

fn segmented_screma_effect(graph: &EGraph) -> Option<LocatedScrema<'_>> {
    graph.skeleton.blocks.iter().find_map(|(block, contents)| {
        contents.side_effects.iter().enumerate().find_map(|(index, effect)| {
            let SideEffectKind::Soac(SoacEffect(owner, Soac::Screma(op))) = &effect.kind else {
                return None;
            };
            matches!(
                op.semantic_state(),
                screma::SemanticState::Segmented {
                    placement: screma::Placement::Kernel,
                    ..
                }
            )
            .then_some(LocatedScrema {
                site: SideEffectSite { block, index },
                effect,
                owner: *owner,
                op,
            })
        })
    })
}

pub(super) fn segmented_recipe_effect(
    entry: &crate::egir::program::PlannedEntry,
) -> Option<LocatedScrema<'_>> {
    if let Some(effect) = segmented_screma_effect(&entry.graph) {
        return Some(effect);
    }
    if !matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
        return None;
    }
    let mut promoted = None;
    for (block, contents) in &entry.graph.skeleton.blocks {
        for (index, effect) in contents.side_effects.iter().enumerate() {
            let SideEffectKind::Soac(SoacEffect(owner, Soac::Screma(op))) = &effect.kind else {
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
                promoted = Some(LocatedScrema {
                    site: SideEffectSite { block, index },
                    effect,
                    owner: *owner,
                    op,
                });
            }
        }
    }
    promoted
}

pub(super) fn segmented_recipe(entry: &crate::egir::program::PlannedEntry) -> Option<SegmentedRecipe<'_>> {
    let located = segmented_recipe_effect(entry)?;
    Some(match located.op {
        screma::Op::Reduce { lanes, operators, .. } => SegmentedRecipe::Reduce(LocatedReduce {
            site: located.site,
            effect: located.effect,
            owner: located.owner,
            lanes,
            operators,
            op: located.op,
        }),
        screma::Op::Scan { lanes, operators, .. } => SegmentedRecipe::Scan(LocatedScan {
            site: located.site,
            effect: located.effect,
            owner: located.owner,
            lanes,
            operators,
            op: located.op,
        }),
        screma::Op::Map { .. } => SegmentedRecipe::Map,
        screma::Op::Composite { .. } => SegmentedRecipe::Composite(located.site),
    })
}

pub(super) fn parallel_recipe_effect(entry: &crate::egir::program::PlannedEntry) -> Option<&SideEffect> {
    segmented_recipe_effect(entry).map(|located| located.effect).or_else(|| {
        entry.graph.skeleton.blocks.values().find_map(|contents| {
            contents
                .side_effects
                .iter()
                .find(|effect| matches!(&effect.kind, SideEffectKind::Soac(SoacEffect(_, Soac::Filter(_)))))
        })
    })
}

pub(super) fn make_screma_serial(graph: &mut EGraph, site: SideEffectSite) -> error::Result<()> {
    let effect = graph.skeleton.get_effect_mut(site).ok_or_else(|| {
        error::ParallelizeError::Invalid(format!(
            "stale segmented effect site {:?}:{}",
            site.block, site.index
        ))
    })?;
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &mut effect.kind else {
        return Err("segmented effect site no longer contains a Screma operation".into());
    };
    *op.semantic_state_mut() = screma::SemanticState::Serial;
    Ok(())
}

pub(super) fn semantic_effect(graph: &EGraph, site: SideEffectSite) -> error::Result<&SideEffect> {
    graph.skeleton.get_effect(site).ok_or_else(|| {
        error::ParallelizeError::Invalid(format!(
            "stale semantic effect site {:?}:{}",
            site.block, site.index
        ))
    })
}

pub(super) fn semantic_effect_mut(
    graph: &mut EGraph,
    site: SideEffectSite,
) -> error::Result<&mut SideEffect> {
    graph.skeleton.get_effect_mut(site).ok_or_else(|| {
        error::ParallelizeError::Invalid(format!(
            "stale semantic effect site {:?}:{}",
            site.block, site.index
        ))
    })
}

pub(super) fn semantic_node_type(graph: &EGraph, node: NodeId) -> error::Result<Type<TypeName>> {
    graph
        .types
        .get(&node)
        .cloned()
        .ok_or_else(|| error::ParallelizeError::Invalid(format!("semantic node {node:?} has no type")))
}

fn valid_recipe_placement(op: &screma::Op<crate::egir::types::Semantic>) -> bool {
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
    valid_placement
}

pub(super) fn reduce_recipe_eligibility(
    recipe: &LocatedReduce<'_>,
) -> std::result::Result<(), FallbackReason> {
    if !valid_recipe_placement(recipe.op) {
        return Err(FallbackReason::UnsupportedPlacement);
    }
    if recipe.operators.iter().any(|op| !op.combine.captures.is_empty()) {
        return Err(FallbackReason::UnsupportedCaptures);
    }
    if recipe.lanes.inputs.is_empty() {
        return Err(FallbackReason::UnsupportedOperationShape);
    }
    if !recipe.lanes.maps.iter().all(|map| map.destination.is_output_view())
        || !recipe.operators.iter().all(|op| op.destination.is_unplaced_fresh())
    {
        return Err(FallbackReason::UnsupportedDestination);
    }
    Ok(())
}

pub(super) fn scan_recipe_eligibility(recipe: &LocatedScan<'_>) -> std::result::Result<(), FallbackReason> {
    if !valid_recipe_placement(recipe.op) {
        return Err(FallbackReason::UnsupportedPlacement);
    }
    if !recipe.operators.rest.is_empty() || recipe.lanes.inputs.len() != 1 {
        return Err(FallbackReason::UnsupportedOperationShape);
    }
    if !recipe.operators.first.combine.captures.is_empty() {
        return Err(FallbackReason::UnsupportedCaptures);
    }
    if !recipe.lanes.maps.iter().all(|map| map.destination.is_output_view())
        || !recipe.operators.iter().all(|op| op.destination.is_output_view())
    {
        return Err(FallbackReason::UnsupportedDestination);
    }
    Ok(())
}
