//! Placement lookup and dominance pruning, indexed by site and expression.
use super::bounds::Bounds;
use super::uses::Uses;
use crate::egglog::data::{
    ExprId, Ir, OperationId, OperationKind, PlacementData, PlacementId, PlacementSite,
};
use std::collections::{BTreeMap, BTreeSet};
use wyn_base::IdArena;

#[derive(Default)]
pub(super) struct Placements {
    pub sites: BTreeMap<PlacementSite, BTreeSet<ExprId>>,
}
impl Placements {
    pub fn insert(&mut self, site: PlacementSite, expression: ExprId) {
        self.sites.entry(site).or_default().insert(expression);
    }

    pub fn finish(self, data: &Ir, uses: &Uses, control: &Bounds) -> IdArena<PlacementId, PlacementData> {
        let mut expressions: BTreeMap<ExprId, Vec<(usize, OperationId)>> = BTreeMap::new();
        let mut kept = Self::default();
        for (site, values) in self.sites {
            for e in values {
                if !uses.dag.descendants.contains_key(&e) {
                    continue;
                }
                match site {
                    PlacementSite::Operation(op) if uses.operations.contains(&op) => {
                        expressions.entry(e).or_default().push((control.operations[&op].0, op));
                    }
                    PlacementSite::Expression(choice) if uses.dag.descendants.contains_key(&choice) => {
                        kept.insert(site, e)
                    }
                    _ => {}
                }
            }
        }
        // Sweep each expression's sites through sorted coverage intervals. No
        // placement pairs or walks up lexical-parent chains are required.
        for (e, mut sites) in expressions {
            sites.sort_unstable();
            let mut pending: BTreeMap<usize, usize> = BTreeMap::new();
            let mut end = 0;
            for (point, op) in sites {
                while pending.first_key_value().is_some_and(|(&start, _)| start <= point) {
                    if let Some((_, stop)) = pending.pop_first() {
                        end = end.max(stop);
                    }
                }
                if point < end {
                    continue;
                }
                kept.insert(PlacementSite::Operation(op), e);
                if matches!(data.operations[op].kind, OperationKind::Call { .. }) {
                    let after = control.operations[&op].1;
                    pending
                        .entry(after.start)
                        .and_modify(|end| *end = (*end).max(after.end))
                        .or_insert(after.end);
                }
                for r in data.operations[op].kind.structured_regions() {
                    let b = control.regions[&r];
                    pending.entry(b.start).and_modify(|end| *end = (*end).max(b.end)).or_insert(b.end);
                }
            }
        }
        let mut result = IdArena::default();
        for (before, values) in kept.sites {
            for expression in values {
                result.alloc(PlacementData { before, expression });
            }
        }
        result
    }
}
