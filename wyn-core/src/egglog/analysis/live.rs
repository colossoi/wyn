//! Seed the shared backward-flow solver from explicit execution points.
use super::Query;
use crate::egglog::OptimizeError;
use egglog_engine::{EGraph, Value, Write};
use wyn_base::Interner;

pub(super) fn publish(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut regions = Vec::new();
    query.for_each("LiveRegion", |row| {
        regions.push((row[0], query.operations(row[0])?));
        Ok(())
    })?;
    let mut uses = Vec::new();
    query.for_each("OperationUse", |row| {
        if query.contains("LiveOperation", row[0])? {
            uses.push((row[0], row[1]));
        }
        Ok(())
    })?;
    let mut definitions = Vec::new();
    query.for_function("Output", |keys, name| {
        if query.contains("LiveOperation", keys[0])? {
            definitions.push((keys[0], name));
        }
        Ok(())
    })?;
    let mut yields = Vec::new();
    query.for_each("EndDemand", |row| {
        if query.contains("LiveRegion", row[0])? {
            yields.push((row[0], row[1]));
        }
        Ok(())
    })?;
    let mut names = Interner::<i64, Value>::new();
    graph.update(|mut sink| {
        for (region, operations) in regions {
            let mut before = sink.add("FlowRegion", region)?;
            for operation in operations {
                let point = sink.add("OperationEvaluation", operation)?;
                sink.add("FlowNext", (before, point))?;
                sink.add("FlowExecution", (point, point, point))?;
                before = point;
            }
            let exit = sink.add("FlowRegionEnd", region)?;
            sink.add("FlowNext", (before, exit))?;
        }
        for (operation, name) in uses {
            let point = sink.add("OperationEvaluation", operation)?;
            let ordinal = names.intern(&name);
            sink.set("NameOrdinal", name, ordinal)?;
            sink.add("FlowLive", (point, ordinal))?;
        }
        for (operation, name) in definitions {
            let point = sink.add("OperationEvaluation", operation)?;
            let ordinal = names.intern(&name);
            sink.set("NameOrdinal", name, ordinal)?;
            sink.set("FlowKilled", (point, ordinal), true)?;
        }
        for (region, name) in yields {
            let exit = sink.add("FlowRegionEnd", region)?;
            let ordinal = names.intern(&name);
            sink.set("NameOrdinal", name, ordinal)?;
            sink.add("FlowLive", (exit, ordinal))?;
        }
        Ok(())
    })?;
    Ok(())
}
