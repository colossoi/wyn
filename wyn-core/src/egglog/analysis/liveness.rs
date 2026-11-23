//! Lexical liveness for host scheduling and storage proofs.
use super::{ensure, Query};
use crate::egglog::graph::Graph;
use crate::egglog::{timing, OptimizeError};
use crate::{LookupMap, LookupSet};
use egglog_engine::{Value, Write};
use std::ops::Range;

#[derive(Default)]
pub(in crate::egglog) struct Liveness {
    regions: LookupMap<Value, Vec<(Value, Range<usize>)>>,
}

impl Liveness {
    pub fn after(&self, region: Value, position: usize) -> LookupSet<Value> {
        self.regions
            .get(&region)
            .into_iter()
            .flatten()
            .filter_map(|(name, live)| live.contains(&position).then_some(*name))
            .collect()
    }

    #[cfg(test)]
    pub fn contains_region(&self, region: Value) -> bool {
        self.regions.contains_key(&region)
    }
}

pub(in crate::egglog) fn liveness(graph: &mut Graph, roots: &[Value]) -> Result<Liveness, OptimizeError> {
    let _timing = timing::span("Rust execution liveness");
    ensure(graph)?;
    let query = Query(graph);
    let mut uses = LookupMap::<_, Vec<_>>::new();
    let mut yields = LookupMap::<_, Vec<_>>::new();
    for (table, names) in [("OperationUse", &mut uses), ("EndDemand", &mut yields)] {
        query.for_each(table, |row| {
            names.entry(row[0]).or_default().push(row[1]);
            Ok(())
        })?;
    }
    let mut live = Liveness::default();
    let mut demanded = LookupSet::new();
    let mut pending = Vec::new();
    for &root in roots {
        if query.contains("LiveRegion", root)? {
            pending.push(root);
        }
    }
    while let Some(region) = pending.pop() {
        if live.regions.contains_key(&region) {
            continue;
        }
        let operations = graph.region_order(region)?;
        // A value is live after its definition and before its last use. Captures
        // start at region entry, and yielded values are used at region exit.
        let mut ranges = LookupMap::<Value, Range<usize>>::new();
        for (index, &operation) in operations.iter().enumerate() {
            for name in graph.outputs(operation) {
                ranges.entry(name).or_default().start = index;
            }
            for &name in uses.get(&operation).into_iter().flatten() {
                ranges.entry(name).or_default().end = index;
            }
            pending.extend(graph.attached_regions(operation));
        }
        for &name in yields.get(&region).into_iter().flatten() {
            ranges.entry(name).or_default().end = operations.len();
        }
        let mut values = Vec::new();
        for (binding, range) in ranges {
            if range.is_empty() {
                continue;
            }
            if let Some(name) = query.value_name(binding)? {
                if query.contains("ArrayValue", name)? {
                    demanded.insert(binding);
                }
                values.push((name, range));
            } else if query.enode("AccNameRef", binding)?.is_some() {
                demanded.insert(binding);
            }
        }
        live.regions.insert(region, values);
    }
    if !demanded.is_empty() {
        graph.update(|mut sink| {
            for name in demanded {
                sink.add("ResourceDemand", name)?;
            }
            Ok(())
        })?;
        graph.parse_and_run_program(None, "(run-schedule (saturate dataflow-resources))")?;
    }
    Ok(live)
}
