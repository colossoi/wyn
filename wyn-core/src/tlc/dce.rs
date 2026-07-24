//! Shared TLC definition-level dead-code elimination.
//!
//! Reachability is computed from entry points and extern definitions through
//! top-level symbol references. The definition vector is retained in place, so
//! live definitions and all of their tree allocations move through unchanged.

use super::{Def, DefMeta, Family, Payload, SoacOp, Term, TermKind, VarRef, WalkDecision};
use crate::{LookupMap, SymbolId};

/// Remove definitions that are not reachable from an entry point or extern.
///
/// This is shared by the final reachability phase and by transformations such
/// as inlining that make definitions dead as a local consequence.
pub(super) fn eliminate_unreachable_defs<F: Family>(defs: &mut Vec<Def<F>>) {
    let reachable = {
        let def_map: LookupMap<SymbolId, &Def<F>> = defs.iter().map(|def| (def.name, def)).collect();
        let roots = defs
            .iter()
            .filter(|def| {
                matches!(def.meta, DefMeta::EntryPoint(_)) || matches!(def.body.kind, TermKind::Extern(_))
            })
            .map(|def| def.name);

        wyn_graph::reachable_set(roots, wyn_graph::WalkOrder::DepthFirst, |symbol, references| {
            if let Some(def) = def_map.get(&symbol) {
                references.extend(
                    collect_definition_references(&def.body)
                        .into_iter()
                        .filter(|reference| def_map.contains_key(reference)),
                );
            }
        })
    };

    defs.retain(|def| reachable.contains(&def.name));
}

/// Collect raw top-level-symbol candidates from one definition body.
///
/// Scope tracking is unnecessary because `SymbolId`s are globally unique.
/// The DCE graph later discards references that do not name a definition.
fn collect_definition_references<C: Payload, S: Payload>(term: &Term<C, S>) -> Vec<SymbolId> {
    let mut references = Vec::new();
    term.walk(&mut |term: &Term<C, S>| {
        if let TermKind::Var(VarRef::Symbol(symbol)) = &term.kind {
            references.push(*symbol);
        }

        // Destinations are stored outside child Terms, so the generic walker
        // cannot observe these symbol references.
        if let TermKind::Soac(soac) = &term.kind {
            let destination = match soac {
                SoacOp::Scatter { dest, .. } | SoacOp::ReduceByIndex { dest, .. } => Some(dest),
                _ => None,
            };
            if let Some(destination) = destination {
                references.push(destination.id);
            }
        }
        WalkDecision::Recurse
    });
    references
}
