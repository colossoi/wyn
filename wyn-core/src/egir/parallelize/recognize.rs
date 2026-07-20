//! One-pass recognition of target-supported recipes in a projected entry.
//!
//! Facts in this module are graph-local and short-lived: analysis rebuilds
//! them for a projected entry and emission consumes them before mutating that
//! graph. Persistent parallel/serial decisions belong to `planning` instead.

use super::*;

#[derive(Clone, Copy)]
pub(super) struct LocatedScrema<'a> {
    pub site: SideEffectSite,
    pub effect: &'a SideEffect,
    pub owner: SemanticOpId,
    pub op: &'a screma::Op<crate::egir::types::Semantic>,
}

pub(super) struct ParallelReduce<'a> {
    located: LocatedScrema<'a>,
    lanes: &'a screma::Lanes,
    operators: &'a screma::NonEmpty<screma::Operator>,
}

pub(super) struct ParallelScan<'a> {
    located: LocatedScrema<'a>,
    lanes: &'a screma::Lanes,
    operators: &'a screma::NonEmpty<screma::Operator>,
}

#[derive(Clone)]
pub(super) struct SerialScremaRecipe {
    site: SideEffectSite,
    owner: SemanticOpId,
    op: screma::Op<crate::egir::types::Semantic>,
}

impl LocatedScrema<'_> {
    pub(super) fn serial_recipe(&self) -> SerialScremaRecipe {
        SerialScremaRecipe {
            site: self.site,
            owner: self.owner,
            op: self.op.clone(),
        }
    }
}

impl<'a> ParallelReduce<'a> {
    fn recognize(
        located: LocatedScrema<'a>,
        lanes: &'a screma::Lanes,
        operators: &'a screma::NonEmpty<screma::Operator>,
    ) -> RecipeSelection<Self> {
        if operators.iter().any(|operator| !operator.combine.captures.is_empty()) {
            return RecipeSelection::Serial(FallbackReason::UnsupportedCaptures);
        }
        if lanes.inputs.is_empty() {
            return RecipeSelection::Serial(FallbackReason::UnsupportedOperationShape);
        }
        if !lanes.maps.iter().all(|map| map.destination.is_output_view())
            || !operators.iter().all(|operator| operator.destination.is_unplaced_fresh())
        {
            return RecipeSelection::Serial(FallbackReason::UnsupportedDestination);
        }
        RecipeSelection::Parallel(Self {
            located,
            lanes,
            operators,
        })
    }

    pub(super) fn serial_recipe(&self) -> SerialScremaRecipe {
        self.located.serial_recipe()
    }

    pub(super) fn into_parts(
        self,
    ) -> (
        LocatedScrema<'a>,
        &'a screma::Lanes,
        &'a screma::NonEmpty<screma::Operator>,
    ) {
        (self.located, self.lanes, self.operators)
    }
}

impl<'a> ParallelScan<'a> {
    fn recognize(
        located: LocatedScrema<'a>,
        lanes: &'a screma::Lanes,
        operators: &'a screma::NonEmpty<screma::Operator>,
    ) -> RecipeSelection<Self> {
        if !operators.rest.is_empty() || lanes.inputs.len() != 1 {
            return RecipeSelection::Serial(FallbackReason::UnsupportedOperationShape);
        }
        if !operators.first.combine.captures.is_empty() {
            return RecipeSelection::Serial(FallbackReason::UnsupportedCaptures);
        }
        if !lanes.maps.iter().all(|map| map.destination.is_output_view())
            || !operators.iter().all(|operator| operator.destination.is_output_view())
        {
            return RecipeSelection::Serial(FallbackReason::UnsupportedDestination);
        }
        RecipeSelection::Parallel(Self {
            located,
            lanes,
            operators,
        })
    }

    pub(super) fn serial_recipe(&self) -> SerialScremaRecipe {
        self.located.serial_recipe()
    }

    pub(super) fn into_parts(
        self,
    ) -> (
        LocatedScrema<'a>,
        &'a screma::Lanes,
        &'a screma::NonEmpty<screma::Operator>,
    ) {
        (self.located, self.lanes, self.operators)
    }
}

pub(super) enum SegmentedRecipe<'a> {
    Reduce(ParallelReduce<'a>),
    Scan(ParallelScan<'a>),
    Serial(SerialScremaRecipe, FallbackReason),
    Map,
    Composite(LocatedScrema<'a>),
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
        screma::Op::Reduce { lanes, operators, .. } => {
            match ParallelReduce::recognize(located, lanes, operators) {
                RecipeSelection::Parallel(recipe) => SegmentedRecipe::Reduce(recipe),
                RecipeSelection::Serial(reason) => SegmentedRecipe::Serial(located.serial_recipe(), reason),
            }
        }
        screma::Op::Scan { lanes, operators, .. } => {
            match ParallelScan::recognize(located, lanes, operators) {
                RecipeSelection::Parallel(recipe) => SegmentedRecipe::Scan(recipe),
                RecipeSelection::Serial(reason) => SegmentedRecipe::Serial(located.serial_recipe(), reason),
            }
        }
        screma::Op::Map { .. } => SegmentedRecipe::Map,
        screma::Op::Composite { .. } => SegmentedRecipe::Composite(located),
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

pub(super) fn make_screma_serial(graph: &mut EGraph, recipe: SerialScremaRecipe) {
    let mut op = recipe.op;
    *op.semantic_state_mut() = screma::SemanticState::Serial;
    graph.skeleton.effect_mut(recipe.site).kind =
        SideEffectKind::Soac(SoacEffect(recipe.owner, Soac::Screma(op)));
}

pub(super) fn semantic_effect(graph: &EGraph, site: SideEffectSite) -> &SideEffect {
    graph.skeleton.effect(site)
}

pub(super) fn semantic_effect_mut(graph: &mut EGraph, site: SideEffectSite) -> &mut SideEffect {
    graph.skeleton.effect_mut(site)
}

pub(super) fn semantic_node_type(graph: &EGraph, node: NodeId) -> Type<TypeName> {
    graph.types[&node].clone()
}
