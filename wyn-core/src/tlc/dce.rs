//! Shared TLC definition-level dead-code elimination.
//!
//! Reachability is computed from entry points through top-level symbol
//! references. The definition vector is retained in place, so
//! live definitions and all of their tree allocations move through unchanged.

use super::{Def, DefMeta, Family};
use crate::egglog::dataflow::reference_paths;
use crate::{LookupMap, SymbolId};

#[cfg(test)]
#[path = "dce_tests.rs"]
mod dce_tests;

/// Remove definitions that are not reachable from an entry point.
///
/// Inlining passes use this to clean up definitions they make dead.
pub(super) fn eliminate_unreachable_defs<F: Family>(defs: &mut Vec<Def<F>>) {
    let reachable = {
        let def_map: LookupMap<SymbolId, &Def<F>> = defs.iter().map(|def| (def.name, def)).collect();
        let roots =
            defs.iter().filter(|def| matches!(def.meta, DefMeta::EntryPoint(_))).map(|def| def.name);

        wyn_graph::reachable_set(roots, wyn_graph::WalkOrder::DepthFirst, |symbol, references| {
            if let Some(def) = def_map.get(&symbol) {
                references.extend(
                    reference_paths(&def.body)
                        .into_iter()
                        .map(|(symbol, _)| symbol)
                        .filter(|reference| def_map.contains_key(reference)),
                );
            }
        })
    };

    defs.retain(|def| reachable.contains(&def.name));
}
