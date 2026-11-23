//! Splice proven branch arms into their enclosing Rust regions.
use crate::egglog::dataflow::edit;
use crate::egglog::graph::Graph;
use crate::egglog::{output_error, query::Query, timing, OptimizeError};
use crate::LookupMap;
use egglog_engine::Value;
use wyn_graph::ReplacementForest;

pub(super) fn apply(graph: &mut Graph) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let mut selected = LookupMap::<Value, Value>::new();
    query.for_each("BranchCandidate", |row| {
        selected.insert(row[0], row[1]);
        Ok(())
    })?;
    if selected.is_empty() {
        return Ok(false);
    }
    let _timing = timing::span("Rust constant branch splicing");
    let mut pending = query.live_roots()?;
    let mut replacements = ReplacementForest::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();
    while let Some(region) = pending.pop() {
        let mut operations = query.0.region_order(region)?.to_vec();
        operations.reverse();
        let mut remaining = Vec::with_capacity(operations.len());
        let mut spliced = false;
        while let Some(operation) = operations.pop() {
            let Some(&chosen) = selected.get(&operation) else {
                pending.extend(graph.attached_regions(operation));
                remaining.push(operation);
                continue;
            };
            let Some((edge, _)) = graph.controls(operation).find(|&(_, arm)| arm == chosen) else {
                return Err(output_error("constant branch proof names a foreign arm"));
            };
            let arguments = graph.arguments(operation, edge);
            let params = graph.parameters(chosen);
            let outputs = graph.outputs(operation);
            let yields = graph.yield_values(chosen);
            if arguments.len() != params.len() || outputs.len() != yields.len() {
                return Err(output_error(
                    "constant branch has an inconsistent region signature",
                ));
            }
            for (old, new) in params.into_iter().zip(arguments).chain(outputs.into_iter().zip(yields)) {
                replacements.replace(old, new).map_err(|error| output_error(error.to_string()))?;
            }
            // A nested branch in the selected arm now occupies this block.
            operations.extend(query.0.region_order(chosen)?.iter().rev().copied());
            removed.push(operation);
            spliced = true;
        }
        if spliced {
            changed.push((region, remaining));
        }
    }
    if removed.is_empty() {
        return Ok(false);
    }
    let replacements: Vec<_> = replacements.into_map().into_iter().collect();
    edit::replace(graph, &replacements)?;
    graph.update(|mut sink| {
        for (region, operations) in changed {
            for (order, operation) in operations.into_iter().enumerate() {
                sink.set_region(operation, region);
                sink.set_order(operation, order as i64);
            }
        }
        Ok(())
    })?;
    edit::remove(graph, &removed)?;
    Ok(true)
}
