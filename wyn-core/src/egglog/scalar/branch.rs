//! Splice proven branch arms on the memory graph before publishing its changed edges.
use crate::egglog::dataflow::{body::BodyGraph, edit};
use crate::egglog::{output_error, query::Query, timing, OptimizeError};
use crate::memory_ir::{Function, ValueRef};
use crate::LookupMap;
use egglog_engine::{EGraph, Value, Write};
use std::mem::take;
use wyn_graph::ReplacementForest;

pub(super) fn apply(graph: &mut EGraph, body: Option<BodyGraph>) -> Result<bool, OptimizeError> {
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
    let mut body = match body {
        Some(body) => body,
        None => BodyGraph::read(query)?,
    };
    let mut pending = body
        .ir
        .functions
        .values()
        .filter_map(|function| match function {
            Function::Defined(root) => Some(*root),
            Function::External => None,
        })
        .collect::<Vec<_>>();
    let mut replacements = ReplacementForest::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();
    while let Some(region) = pending.pop() {
        let mut operations = take(&mut body.ir.regions[region].operations);
        operations.reverse();
        let mut remaining = Vec::with_capacity(operations.len());
        let mut spliced = false;
        while let Some(op) = operations.pop() {
            let key = body.operations[op];
            let operation = &body.ir.operations[op];
            let Some(&chosen) = selected.get(&key) else {
                pending.extend(operation.bodies.iter().map(|edge| edge.region));
                remaining.push(op);
                continue;
            };
            let Some(arm) = operation.bodies.iter().find(|edge| body.regions[edge.region] == chosen) else {
                return Err(output_error("constant branch proof names a foreign arm"));
            };
            let arm_region = arm.region;
            let chosen = &mut body.ir.regions[arm_region];
            if operation.result_types.len() != chosen.yields.len()
                || arm.arguments.len() != chosen.parameters.len()
            {
                return Err(output_error(
                    "constant branch has an inconsistent region signature",
                ));
            }
            for (slot, &argument) in arm.arguments.iter().enumerate() {
                replacements
                    .replace(ValueRef::Parameter(arm_region, slot), argument)
                    .map_err(|error| output_error(error.to_string()))?;
            }
            for (slot, &value) in chosen.yields.iter().enumerate() {
                replacements
                    .replace(ValueRef::Result(op, slot), value)
                    .map_err(|error| output_error(error.to_string()))?;
            }
            // A nested branch in the selected arm now occupies this block.
            // The other arm is unreachable and contributes no edits.
            operations.extend(take(&mut chosen.operations).into_iter().rev());
            removed.push(key);
            body.ir.operations.remove(op);
            spliced = true;
        }
        body.ir.regions[region].operations = remaining;
        if spliced {
            changed.push(region);
        }
    }
    if removed.is_empty() {
        return Ok(false);
    }
    let replacements = replacements.into_map();
    let uses = body.ir.redirect_uses(&replacements);
    body.publish_substitutions(graph, &replacements, uses)?;
    graph.update(|mut sink| {
        for region in changed {
            for (order, &op) in body.ir.regions[region].operations.iter().enumerate() {
                let key = body.operations[op];
                sink.remove("OperationRegion", key)?;
                sink.set("OperationRegion", key, body.regions[region])?;
                sink.remove("OperationOrder", key)?;
                sink.set("OperationOrder", key, order as i64)?;
            }
        }
        Ok(())
    })?;
    edit::remove(graph, &removed)?;
    Ok(true)
}
