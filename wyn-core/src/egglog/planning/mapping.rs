//! Rank admitted traversals; costs cannot introduce a domain or execution scope.
use crate::egglog::{cost::CostModel, output_error as error, query::Query, OptimizeError};
use crate::LookupMap;
use egglog_engine::{sort::Z, EGraph, RawValues, Value, Write};
use std::collections::HashSet;

const WIDTHS: [u32; 5] = [1, 32, 64, 128, 256];

#[derive(Clone, Copy)]
struct Candidate {
    shape: Value,
    traversal: Value,
    rank: i64,
    cost: f64,
}

#[derive(Default)]
struct Work {
    serial: f64,
    parallel: f64,
    outputs: f64,
}

struct Estimates<'a> {
    query: Query<'a>,
}

impl Estimates<'_> {
    fn number(&self, name: &str, key: Value) -> Result<Option<f64>, OptimizeError> {
        Ok(self.query.lookup(name, (key,))?.map(|n| self.query.0.value_to_base::<i64>(n).max(0) as f64))
    }

    fn cost(&self, table: &str, key: Value) -> Result<Option<f64>, OptimizeError> {
        Ok(self
            .query
            .lookup(table, (key,))?
            .map(|n| u64::try_from(&*self.query.0.value_to_base::<Z>(n)).unwrap_or(u64::MAX) as f64))
    }

    fn region(&self, region: Value) -> Result<f64, OptimizeError> {
        Ok(self.number("MapPrefixWork", region)?.unwrap_or(65.0))
    }

    fn traversal(&self, traversal: Value) -> Result<Work, OptimizeError> {
        for (constructor, region) in [("TraversalFor", 1), ("TraversalCall", 2)] {
            if let Some(fields) = self.query.enode(constructor, traversal)? {
                let mut work = self.traversal(fields[4])?;
                work.serial += self.region(fields[region])?;
                return Ok(work);
            }
        }
        if let Some(fields) = self.query.enode("TraversalChoice", traversal)? {
            let a = self.traversal(fields[2])?;
            let b = self.traversal(fields[4])?;
            return Ok(Work {
                serial: 1.0 + a.serial.max(b.serial),
                parallel: a.parallel.max(b.parallel),
                outputs: a.outputs.max(b.outputs),
            });
        }
        if let Some(fields) = self.query.enode("TraversalReindex", traversal)? {
            return self.traversal(fields[0]);
        }
        if let Some(fields) = self.query.enode("TraversalResult", traversal)? {
            let mut regions = Vec::new();
            self.query.for_each("PlanMember", |row| {
                if row[0] == fields[2] {
                    if let Some(region) = self.query.lookup("SourceOperatorBody", (row[1],))? {
                        regions.push(region);
                    }
                }
                Ok(())
            })?;
            let mut ops = 0.0;
            for region in regions {
                let cost = self.query.constructor("RegionCost", (region,))?;
                let full = match cost {
                    Some(cost) => self.cost("CostValue", cost)?,
                    None => None,
                };
                ops += full.unwrap_or(self.region(region)?);
            }
            // Unknown lengths are estimates, never capacity or legality proofs.
            let n = self.cost("CostExtent", fields[4])?.unwrap_or(256.0);
            return Ok(Work {
                serial: 0.0,
                parallel: n * ops.max(1.0),
                outputs: n,
            });
        }
        if self.query.enode("TraversalValue", traversal)?.is_some()
            || self.query.enode("TraversalIndex", traversal)?.is_some()
        {
            return Ok(Work {
                serial: 1.0,
                parallel: 0.0,
                outputs: 1.0,
            });
        }
        Err(error("unrecognized map candidate traversal"))
    }
}

pub(super) fn select(graph: &mut EGraph, costs: CostModel) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut hosts = HashSet::new();
    let estimates = Estimates { query };
    let mut candidates = LookupMap::<Value, LookupMap<i64, [Option<Candidate>; 5]>>::default();
    query.for_each("MapPlan", |row| {
        candidates.entry(row[1]).or_default();
        if query.lookup("MapHost", (row[0],))?.is_some_and(|host| graph.value_to_base::<bool>(host)) {
            hosts.insert(row[1]);
        }
        Ok(())
    })?;
    query.for_each("OutputTraversal", |row| {
        let [owner, slot, shape, traversal] = [row[0], row[1], row[2], row[3]];
        let rank = graph.value_to_base::<i64>(query.required("ShapeRank", (shape,))?);
        let extent = query.required("ShapeExtent", (shape,))?;
        let coordinates =
            estimates.cost("CostExtent", extent)?.unwrap_or(f64::from(costs.resident_workgroups) * 32.0);
        let work = estimates.traversal(traversal)?;
        let group_safe = hosts.contains(&owner)
            && query.contains("GroupSafeTraversal", (traversal,))?
            && query.contains("ParallelTraversalWork", (traversal,))?;
        let slots = candidates.entry(owner).or_default();
        let choices = slots.entry(graph.value_to_base::<i64>(slot)).or_default();
        for (scope, choice) in choices.iter_mut().enumerate() {
            if scope > 0 && !group_safe {
                continue;
            }
            let width = WIDTHS[scope];
            let candidate = Candidate {
                shape,
                traversal,
                rank,
                cost: costs.mapping(coordinates, work.serial, work.parallel, work.outputs, width),
            };
            if choice.is_none_or(|old| {
                candidate.cost < old.cost || (candidate.cost == old.cost && candidate.rank > old.rank)
            }) {
                *choice = Some(candidate);
            }
        }
        Ok(())
    })?;
    let mut selected = Vec::new();
    for (owner, slots) in candidates {
        let scope = (0..WIDTHS.len())
            .filter_map(|scope| {
                let total: Option<f64> = slots.values().map(|choices| choices[scope].map(|c| c.cost)).sum();
                total.map(|total| (scope, total))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let Some((scope, _)) = scope else {
            return Err(error("map has no legal execution candidate"));
        };
        let mut outputs = Vec::new();
        for (slot, choices) in slots {
            let Some(chosen) = choices[scope] else {
                return Err(error("map output has no legal execution candidate"));
            };
            outputs.push((slot, chosen));
        }
        selected.push((owner, WIDTHS[scope], outputs));
    }
    graph.update(|mut sink| {
        for (owner, width, outputs) in selected {
            let group = width > 1;
            if group {
                sink.set("MapWorkgroupWidth", owner, i64::from(width))?;
            }
            let scope = sink.add(
                if group { "WorkgroupScope" } else { "InvocationScope" },
                RawValues(vec![]),
            )?;
            sink.set("MapExecution", owner, scope)?;
            for (slot, chosen) in outputs {
                sink.set("MapOutputDomain", (owner, slot), chosen.shape)?;
                sink.set("MapOutputTraversal", (owner, slot), chosen.traversal)?;
            }
        }
        Ok(())
    })?;
    Ok(())
}
