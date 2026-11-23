//! Reorder siblings using executable region orders and current dependency proofs.
use super::{analysis, effects_conflict, timing, Effects, Query};
use crate::egglog::dataflow::edit;
use crate::egglog::graph::{Graph, Origin};
use crate::egglog::OptimizeError;
use crate::{LookupMap, LookupSet};
use egglog_engine::sort::Z;
use egglog_engine::Value;

pub(in crate::egglog::fusion) fn run(graph: &mut Graph) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust sibling motion");
    let query = Query(graph);
    let mut motion = Motion::read(query)?;
    let mut regions = Vec::new();
    let mut selected = LookupSet::new();
    for (&region, operations) in &graph.region_orders {
        for (before, index) in motion.candidates(operations, 0) {
            if !analysis::uses_output_of(query, operations[index], operations[before])? {
                regions.push(region);
                selected.extend(operations.iter().copied());
                break;
            }
        }
    }
    if regions.is_empty() {
        return Ok(false);
    }
    Effects::for_each(query, |op, effect| {
        if selected.contains(&op) {
            motion.effects.entry(op).or_default().add(effect);
        }
    })?;
    query.for_each("OperationUse", |row| {
        let op = row[0];
        if (motion.preparation.contains(&op) || motion.total_index.contains(&op))
            && !motion.boundary.contains(&op)
            && selected.contains(&op)
        {
            motion.uses.entry(op).or_default().push(row[1]);
        }
        Ok(())
    })?;
    let mut updated = Vec::new();
    for region in regions {
        let mut order = graph.region_order(region)?.to_vec();
        if motion.reorder(graph, &mut order)? {
            updated.push((region, order));
        }
    }
    if updated.is_empty() {
        return Ok(false);
    }
    let _timing = timing::span("Rust sibling order publication");
    edit::reorder_regions(graph, updated)?;
    Ok(true)
}

#[derive(Default)]
struct Motion {
    preparation: LookupSet<Value>,
    total_index: LookupSet<Value>,
    invocation_dependent: LookupSet<Value>,
    boundary: LookupSet<Value>,
    branch: LookupSet<Value>,
    parallel: LookupSet<Value>,
    effects: LookupMap<Value, Effects>,
    uses: LookupMap<Value, Vec<Value>>,
}

impl Motion {
    fn read(query: Query<'_>) -> Result<Self, OptimizeError> {
        let mut motion = Self::default();
        for (tables, set) in [
            (&["FreshOutput"][..], &mut motion.preparation),
            (&["TotalIndex"][..], &mut motion.total_index),
            (&["InvocationDependent"][..], &mut motion.invocation_dependent),
            (&["ParallelOperation"][..], &mut motion.parallel),
            (&["BranchOperation"][..], &mut motion.branch),
            (&["AccumulatorOperation"][..], &mut motion.boundary),
        ] {
            for &table in tables {
                query.for_each(table, |row| {
                    set.insert(row[0]);
                    Ok(())
                })?;
            }
        }
        for (operation, code) in query.0.codes() {
            if code.has_expression_identity() {
                motion.preparation.insert(operation);
            }
        }
        motion.boundary.extend(&motion.parallel);
        motion.boundary.extend(&motion.branch);
        Ok(motion)
    }

    fn may_move(&self, first: Value, next: Value, follower: Option<Value>) -> bool {
        !self.invocation_dependent.contains(&next)
            && !self.boundary.contains(&next)
            && ((self.boundary.contains(&first) && self.preparation.contains(&next))
                || (self.parallel.contains(&first)
                    && self.total_index.contains(&next)
                    && follower.is_some_and(|op| self.parallel.contains(&op))))
    }

    fn candidates<'a>(
        &'a self,
        operations: &'a [Value],
        start: usize,
    ) -> impl Iterator<Item = (usize, usize)> + 'a {
        operations.iter().enumerate().skip(start).flat_map(move |(before, &first)| {
            let tail = &operations[before + 1..];
            tail.iter()
                .enumerate()
                .take_while(move |&(i, &moving)| {
                    self.boundary.contains(&first) && (i == 0 || self.preparation.contains(&moving))
                })
                .filter_map(move |(i, &moving)| {
                    let follower = tail.get(i + 1).copied();
                    self.may_move(first, moving, follower).then_some((before, before + 1 + i))
                })
        })
    }

    fn reorder(&self, graph: &mut Graph, operations: &mut [Value]) -> Result<bool, OptimizeError> {
        let mut start = 0;
        let mut changed = false;
        while let Some((before, index)) = self.next_move(graph, operations, start)? {
            operations[before..=index].rotate_right(1);
            changed = true;
            // An earlier anchor's scan stops at the preceding non-preparation,
            // unless that operation is the anchor's first candidate. The last
            // candidate can also inspect its immediate follower.
            start = operations[..before]
                .iter()
                .rposition(|op| !self.preparation.contains(op))
                .map_or(0, |position| position.saturating_sub(1));
        }
        Ok(changed)
    }

    fn next_move(
        &self,
        graph: &mut Graph,
        operations: &[Value],
        start: usize,
    ) -> Result<Option<(usize, usize)>, OptimizeError> {
        for (before, index) in self.candidates(operations, start) {
            let (first, next) = (operations[before], operations[index]);
            if !self.preparation.contains(&next) {
                let Some(&follower) = operations.get(index + 1) else {
                    continue;
                };
                let result = graph.output(next, 0)?;
                if !Query(graph).contains("OperationUse", (follower, result))? {
                    continue;
                }
            }
            if self.branch.contains(&first) {
                let query = Query(graph);
                let cheap = if let Some(work) = query.lookup("InstructionWork", next)? {
                    graph.value_to_base::<i64>(work) <= 4
                } else if let Some(work) = query.lookup("OperationWork", next)? {
                    u64::try_from(&*graph.value_to_base::<Z>(work)).is_ok_and(|work| work <= 4)
                } else {
                    false
                };
                if !cheap {
                    continue;
                }
            }
            // Keep the source dependencies of a blocked preparation even
            // when an independent preparation can move across it.
            let crossed = &operations[before..index];
            let mut blocked = false;
            for &name in self.uses.get(&next).into_iter().flatten() {
                // A capability can be defined before both occurrences and
                // consumed between them. Keep its occurrence order intact.
                if Query(graph).enode("AccNameRef", name)?.is_some()
                    || matches!(graph.origin(name), Some(Origin::Result(producer, _)) if crossed.contains(&producer))
                {
                    blocked = true;
                    break;
                }
            }
            if blocked {
                continue;
            }
            let no_effects = Effects::default();
            let effect = self.effects.get(&next).unwrap_or(&no_effects);
            if effect.traps {
                continue;
            }
            let mut conflict = false;
            for op in crossed {
                let prior = self.effects.get(op).unwrap_or(&no_effects);
                if effects_conflict(graph, prior, effect, true)? {
                    conflict = true;
                    break;
                }
            }
            if !conflict {
                return Ok(Some((before, index)));
            }
        }
        Ok(None)
    }
}
