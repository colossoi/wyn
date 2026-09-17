//! Topological binding propagation and placement with shared scope sets.
use super::uses::Uses;
use super::{bounds, total_node, uses, OptimizeError, Placements};
use crate::egglog::data::{ExprId, ExprKind, Ir, OperationKind, PlacementSite};
use crate::egglog::dependencies::analyze;
use crate::egglog::timing::{span, time};
use bounds::Bounds;
use std::collections::BTreeMap;
use wyn_base::persistent_sets::{Set, EMPTY};
use wyn_graph::DfsInterval;

pub(super) struct Analysis {
    pub uses: Uses,
    pub control: Bounds,
    lexical: BTreeMap<ExprId, Option<DfsInterval>>,
}
impl Analysis {
    pub(super) fn new(data: &Ir) -> Result<Self, OptimizeError> {
        let _timing = span("analyze placements");
        let summary = time("analyze dependencies", || analyze(data));
        let schedules = time("validate dependency order", || summary.schedules(data))?;
        let control = time("derive binding bounds", || bounds::analyze(data, &schedules))?;
        let uses = time("collect expression uses", || uses::analyze(data, &summary.live));
        let lexical = time("propagate binding bounds", || {
            let mut values: BTreeMap<ExprId, Option<DfsInterval>> = BTreeMap::new();
            for &e in &uses.dag.order {
                let bound = match data.expressions[e].kind {
                    ExprKind::Parameter(p) => control.regions.get(&data.parameters[p].region).copied(),
                    ExprKind::OperationResult(op) => control.operations.get(&op).map(|&(_, b)| b),
                    _ if total_node(data, e) => data.expressions[e]
                        .kind
                        .children()
                        .into_iter()
                        .try_fold(DfsInterval::ANY, |a, x| values[&x].and_then(|b| a.intersect(b))),
                    _ => None,
                };
                values.insert(e, bound);
            }
            values
        });
        Ok(Self {
            uses,
            control,
            lexical,
        })
    }

    pub(super) fn place(&mut self, data: &Ir, placements: &mut Placements) {
        let dag = &mut self.uses.dag;
        let mut available_events: BTreeMap<usize, Vec<(ExprId, bool)>> = BTreeMap::new();
        let mut safe = EMPTY;
        for (&e, bound) in &self.lexical {
            if let Some(b) = bound.filter(|_| computation(data, e)) {
                safe = dag.sets.insert(safe, e.as_u32());
                available_events.entry(b.start).or_default().push((e, true));
                available_events.entry(b.end).or_default().push((e, false));
            }
        }
        let mut operations: Vec<_> =
            self.uses.operations.iter().map(|&op| (self.control.operations[&op].0, op)).collect();
        operations.sort_unstable();
        let mut available = EMPTY;
        let mut coverage: BTreeMap<usize, Vec<(usize, Set)>> = BTreeMap::new();
        let mut active: Vec<(usize, Set)> = vec![];
        for (point, op) in operations {
            while available_events.first_key_value().is_some_and(|(&p, _)| p <= point) {
                if let Some((_, events)) = available_events.pop_first() {
                    for (e, add) in events {
                        let singleton = dag.sets.singleton(e.as_u32());
                        available = if add {
                            dag.sets.union(available, singleton)
                        } else {
                            dag.sets.difference(available, singleton)
                        };
                    }
                }
            }
            while active.last().is_some_and(|&(end, _)| end <= point) {
                active.pop();
            }
            while coverage.first_key_value().is_some_and(|(&p, _)| p <= point) {
                if let Some((_, events)) = coverage.pop_first() {
                    for (end, set) in events {
                        if end <= point {
                            continue;
                        }
                        let inherited = active.last().map_or(EMPTY, |&(_, s)| s);
                        let covered = dag.sets.union(inherited, set);
                        active.push((end, covered));
                    }
                }
            }
            let candidates = match data.operations[op].kind {
                OperationKind::Loop { header, body, .. } => {
                    dag.sets.union(self.uses.scopes[&header], self.uses.scopes[&body])
                }
                OperationKind::If {
                    then_region,
                    else_region,
                    ..
                } => dag.sets.intersection(self.uses.scopes[&then_region], self.uses.scopes[&else_region]),
                _ => continue,
            };
            let possible = dag.sets.intersection(candidates, available);
            let selected = dag.sets.difference(possible, active.last().map_or(EMPTY, |&(_, s)| s));
            for e in dag.sets.iter(selected).map(ExprId::from) {
                placements.insert(PlacementSite::Operation(op), e);
            }
            if selected == EMPTY {
                continue;
            }
            let regions = data.operations[op].kind.structured_regions();
            for &r in &regions {
                let b = self.control.regions[&r];
                if regions.iter().any(|other| *other != r && self.control.regions[other].contains(b.start))
                {
                    continue;
                }
                coverage.entry(b.start).or_default().push((b.end, selected));
            }
        }
        // Propagate guaranteed cached values down the expression-use DAG. A
        // shared child inherits only coverage present along EVERY incoming use.
        // This avoids retraversing arms and redundant placements in nested ifs.
        let mut incoming: BTreeMap<ExprId, Set> = dag.roots.iter().map(|&e| (e, EMPTY)).collect();
        for &e in dag.order.iter().rev() {
            let mut covered = incoming.get(&e).copied().unwrap_or(EMPTY);
            if let ExprKind::If {
                then_value,
                else_value,
                ..
            } = data.expressions[e].kind
            {
                let common =
                    dag.sets.intersection(dag.descendants[&then_value], dag.descendants[&else_value]);
                let common = dag.sets.intersection(common, safe);
                let selected = dag.sets.difference(common, covered);
                for value in dag.sets.iter(selected).map(ExprId::from) {
                    placements.insert(PlacementSite::Expression(e), value);
                }
                covered = dag.sets.union(covered, selected);
            }
            for child in data.expressions[e].kind.children() {
                let next = match incoming.get(&child) {
                    Some(&old) => dag.sets.intersection(old, covered),
                    None => covered,
                };
                incoming.insert(child, next);
            }
        }
    }
}

pub(super) fn computation(data: &Ir, e: ExprId) -> bool {
    matches!(
        data.expressions[e].kind,
        ExprKind::PureApp { .. } | ExprKind::Coerce(_) | ExprKind::Project { .. } | ExprKind::Vector(_)
    )
}
