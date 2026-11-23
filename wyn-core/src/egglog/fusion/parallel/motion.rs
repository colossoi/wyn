//! Move siblings on the live graph without publishing each move.
use super::body::{Body, Effects};
use super::{effects_conflict, Query};
use crate::egglog::graph::Graph;
use crate::egglog::OptimizeError;
use crate::memory_ir::{RegionId, ValueRef};
use crate::LookupSet;
use egglog_engine::sort::Z;
use egglog_engine::Value;

#[derive(Default)]
pub(super) struct Motion {
    preparation: LookupSet<Value>,
    total_index: LookupSet<Value>,
    invocation_dependent: LookupSet<Value>,
    boundary: LookupSet<Value>,
    branch: LookupSet<Value>,
    parallel: LookupSet<Value>,
}

impl Motion {
    pub fn read(query: Query<'_>) -> Result<Self, OptimizeError> {
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

    pub fn candidates<'a, K: Copy + 'a>(
        &'a self,
        operations: &'a [K],
        start: usize,
        key: impl Fn(K) -> Value + Copy + 'a,
    ) -> impl Iterator<Item = (usize, usize)> + 'a {
        operations.iter().enumerate().skip(start).flat_map(move |(before, &anchor)| {
            let first = key(anchor);
            let tail = &operations[before + 1..];
            tail.iter()
                .enumerate()
                .take_while(move |&(i, &moving)| {
                    self.boundary.contains(&first) && (i == 0 || self.preparation.contains(&key(moving)))
                })
                .filter_map(move |(i, &moving)| {
                    let follower = tail.get(i + 1).copied().map(key);
                    self.may_move(first, key(moving), follower).then_some((before, before + 1 + i))
                })
        })
    }

    pub fn run(&self, graph: &mut Graph, body: &mut Body, region: RegionId) -> Result<bool, OptimizeError> {
        let mut start = 0;
        let mut changed = false;
        while let Some(before) = self.step(graph, body, region, start)? {
            changed = true;
            // An earlier anchor's scan stops at the preceding non-preparation,
            // unless that operation is the anchor's first candidate. The last
            // candidate can also inspect its immediate follower.
            start = body.graph.ir.regions[region].operations[..before]
                .iter()
                .rposition(|&op| !self.preparation.contains(&body.graph.operations[op]))
                .map_or(0, |position| position.saturating_sub(1));
        }
        Ok(changed)
    }

    fn step(
        &self,
        graph: &mut Graph,
        body: &mut Body,
        region: RegionId,
        start: usize,
    ) -> Result<Option<usize>, OptimizeError> {
        let operations = &body.graph.ir.regions[region].operations;
        let mut selected = None;
        for (before, index) in self.candidates(operations, start, |op| body.graph.operations[op]) {
            let (anchor, moving) = (operations[before], operations[index]);
            let first = body.graph.operations[anchor];
            let next = body.graph.operations[moving];
            let preparation = self.preparation.contains(&next);
            let follower = operations.get(index + 1).copied();
            if !preparation {
                let Some(follower) = follower else { continue };
                let result = body.graph.names[&ValueRef::Result(moving, 0)];
                if !Query(graph).contains("OperationUse", (body.graph.operations[follower], result))? {
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
            let start = body.positions[anchor].1;
            let end = body.positions[moving].1;
            let mut blocked = false;
            for (_, name) in body.tree_uses(moving) {
                // A capability can be defined before both occurrences and
                // consumed between them. Keep its occurrence order intact.
                if Query(graph).enode("AccNameRef", name)?.is_some() {
                    blocked = true;
                    break;
                }
                if let ValueRef::Result(producer, _) = body.graph.refs[&name] {
                    let (owner, order) = body.positions[producer];
                    if owner == region && (start..end).contains(&order) {
                        blocked = true;
                        break;
                    }
                }
            }
            if blocked {
                continue;
            }
            let no_effects = Effects::default();
            let effect = body.effects.get(moving).unwrap_or(&no_effects);
            if effect.traps {
                continue;
            }
            let mut conflict = false;
            for &crossed in &operations[before..index] {
                let prior = body.effects.get(crossed).unwrap_or(&no_effects);
                if effects_conflict(graph, prior, effect, true)? {
                    conflict = true;
                    break;
                }
            }
            if !conflict {
                selected = Some((before, index));
                break;
            }
        }
        let Some((before, index)) = selected else {
            return Ok(None);
        };
        body.move_within(region, before, index);
        Ok(Some(before))
    }
}
