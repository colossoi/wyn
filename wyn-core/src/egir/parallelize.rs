//! EGIR-side SOAC parallelization.
//!
//! Consumes the `ParallelizationPlan`s produced by `tlc::parallelize` and
//! tags planned entries' tail SOACs with `PendingSoac::Parallel`, leaving
//! `soac_expand` to actually emit the lane-indexed kernel.
//!
//! Today only the Map strategy is implemented; reduce / scan / redomap
//! still flow through TLC parallelize. See the architecture plan at
//! `~/.claude/plans/greedy-baking-elephant.md`.
use std::collections::HashMap;

use super::program::EgirInner;
use super::types::{PendingSoac, SideEffectKind, SoacDestination};
use crate::tlc::parallelize::{ParallelStrategy, ParallelizationPlan};

/// Walk every entry; for each entry that has a plan, rewrite the tail
/// SOAC in place to `PendingSoac::Parallel { serial: <existing> }`.
///
/// Non-Map strategies are skipped silently — they keep flowing through
/// TLC parallelize's existing path until their EGIR migration lands.
pub fn run(inner: &mut EgirInner, plans: &HashMap<String, ParallelizationPlan>) {
    for entry in inner.entry_points.iter_mut() {
        let Some(plan) = plans.get(&entry.name) else {
            continue;
        };
        if plan.strategy != ParallelStrategy::Map {
            continue;
        }
        rewrite_tail_map(entry);
    }
}

fn rewrite_tail_map(entry: &mut super::program::EgirEntry) {
    // Walk every block and locate the (unique) tail Map. analyze_entry
    // already proved the entry's tail expression is the Map; from_tlc
    // turns it into a `PendingSoac::Map` side-effect. Wrap it.
    for (_, block) in entry.graph.skeleton.blocks.iter_mut() {
        for se in block.side_effects.iter_mut() {
            let SideEffectKind::Pending(ref pending) = se.kind else {
                continue;
            };
            // Only the entry's tail Map (which targets the auto-bound
            // OutputView) is parallelizable. Intermediate Maps that
            // produce fresh arrays for downstream consumers (e.g. a
            // Map → Reduce fusion that hasn't fused) must stay serial.
            match pending {
                PendingSoac::Map {
                    destination: SoacDestination::OutputView,
                    ..
                } => {}
                _ => continue,
            }
            // PendingSoac derives Clone, so the clean version is just to
            // clone the inner SOAC out, then overwrite. The wrapper is
            // dominantly a thin marker — clone cost is one Vec<Type> per
            // entry per compilation.
            let SideEffectKind::Pending(inner) = se.kind.clone() else {
                unreachable!()
            };
            se.kind = SideEffectKind::Pending(PendingSoac::Parallel {
                serial: Box::new(inner),
            });
            return;
        }
    }
}
