//! Select stable operation orders once per region, retaining only the final facts.
use super::super::query::Query;
use super::OptimizeError;
use crate::LookupMap;
use egglog_engine::{EGraph, Write};
use wyn_graph::topo_sort_by_dependencies;

pub(super) fn apply(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut regions = LookupMap::<_, Vec<_>>::default();
    query.for_each("PlannedSourcePosition", |row| {
        regions.entry(row[1]).or_default().push((graph.value_to_base::<i64>(row[2]), row[0]));
        Ok(())
    })?;
    let mut dependencies = LookupMap::<_, Vec<_>>::default();
    query.for_each("OperationDependency", |row| {
        dependencies.entry(row[0]).or_default().push(row[1]);
        Ok(())
    })?;
    let mut effects = LookupMap::<_, Vec<_>>::default();
    query.for_each("EffectInput", |row| {
        effects.entry(row[0]).or_default().push(row[1]);
        Ok(())
    })?;
    query.for_each("EffectWait", |row| {
        dependencies.entry(row[0]).or_default().extend(effects.get(&row[1]).into_iter().flatten());
        Ok(())
    })?;
    let mut orders = Vec::new();
    for (region, mut positions) in regions {
        positions.sort();
        // The supplied universe confines dependency/effect edges to this region.
        let order = topo_sort_by_dependencies(positions.into_iter().map(|(_, op)| op), |op, out| {
            out.extend(dependencies.get(&op).into_iter().flatten().copied());
        })
        .map_err(|_| OptimizeError::Extraction("cycle in the fused execution graph".into()))?;
        orders.push((region, order));
    }
    graph.update(|mut sink| {
        for (region, order) in &orders {
            for (position, &op) in order.iter().enumerate() {
                sink.add("OperationOrderPosition", (op, *region, position as i64))?;
            }
            for pair in order.windows(2) {
                sink.add("Consecutive", (pair[0], pair[1]))?;
            }
        }
        Ok::<_, egglog_engine::Error>(())
    })?;
    Ok(())
}

#[cfg(test)]
#[path = "order_tests.rs"]
mod tests;
