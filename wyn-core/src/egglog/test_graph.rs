//! Flat graph fixtures for phase tests; no executable syntax is reconstructed.
use super::dataflow::ir::{self, Code};
use super::query::Query;
use super::{from_tlc, memory, OptimizeError};
use crate::egglog::graph::Graph;
use egglog_engine::{Value, Write};

pub(super) fn fixture(codes: &[(i64, Code)], facts: &str) -> Result<Graph, OptimizeError> {
    let mut graph = Graph::default();
    memory::load(&mut graph)?;
    graph.update(|mut sink| {
        Ok((|| -> Result<(), OptimizeError> {
            for (id, code) in codes {
                let operation = sink.add("OperationId", *id)?;
                sink.add("Operation", operation)?;
                ir::properties(&mut sink, operation, code.clone())?;
            }
            Ok(())
        })())
    })??;
    facts_with_positions(&mut graph, facts)?;
    Ok(graph)
}

pub(super) fn import(source: &str) -> Result<Graph, OptimizeError> {
    let source = crate::compile_thru_tlc(source).unwrap();
    Ok(from_tlc::import(&source)?.0)
}

pub(super) fn entry(graph: &Graph) -> Value {
    Query(graph).first("EntryOrder").unwrap().unwrap()[0]
}

pub(super) fn root(graph: &Graph) -> Value {
    Query(graph).0.function_root(entry(graph)).unwrap()
}

pub(super) fn facts_with_positions(graph: &mut Graph, facts: &str) -> Result<(), OptimizeError> {
    let fields = [
        "FixtureRegion",
        "FixtureOrder",
        "FixtureFunctionRoot",
        "FixtureCallTarget",
        "FixtureControlRegion",
    ];
    if !fields.iter().any(|field| facts.contains(field)) {
        graph.parse_and_run_program(None, facts)?;
        return Ok(());
    }
    if !graph.get_function_names().iter().any(|name| name == "FixtureRegion") {
        graph.parse_and_run_program(
            None,
            "
            (function FixtureRegion (OperationKey) RegionKey :no-merge)
            (function FixtureOrder (OperationKey) i64 :no-merge)
            (function FixtureFunctionRoot (FunctionName) RegionKey :no-merge)
            (function FixtureCallTarget (OperationKey) FunctionName :no-merge)
            (function FixtureControlRegion (OperationKey i64) RegionKey :no-merge)
        ",
        )?;
    }
    let (setup, checks) = facts.split_once("(run-schedule").unwrap_or((facts, ""));
    graph.parse_and_run_program(None, setup)?;
    let mut regions = Vec::new();
    let mut orders = Vec::new();
    let query = Query(graph);
    query.for_function("FixtureRegion", |key, region| {
        regions.push((key[0], region));
        Ok(())
    })?;
    query.for_function("FixtureOrder", |key, order| {
        orders.push((key[0], graph.value_to_base::<i64>(order)));
        Ok(())
    })?;
    let mut roots = Vec::new();
    let mut targets = Vec::new();
    query.for_function("FixtureFunctionRoot", |key, region| {
        roots.push((key[0], region));
        Ok(())
    })?;
    query.for_function("FixtureCallTarget", |key, function| {
        targets.push((key[0], function));
        Ok(())
    })?;
    let mut controls = Vec::new();
    query.for_function("FixtureControlRegion", |keys, child| {
        controls.push((keys[0], graph.value_to_base::<i64>(keys[1]), child));
        Ok(())
    })?;
    graph.update(|mut sink| {
        for (operation, slot, child) in controls {
            sink.set_control(operation, slot, child);
            sink.set("RegionOwnedBy", child, operation)?;
        }
        for (function, region) in roots {
            sink.set_function_root(function, region);
            sink.add("RootFunction", (region, function))?;
        }
        for (operation, function) in targets {
            sink.set_call_target(operation, function);
        }
        for (operation, region) in regions {
            sink.set_region(operation, region);
        }
        for (operation, order) in orders {
            sink.set_order(operation, order);
        }
        Ok(())
    })?;
    for field in fields {
        graph.clear_function(field)?;
    }
    if !checks.is_empty() {
        graph.parse_and_run_program(None, &format!("(run-schedule{checks}"))?;
    }
    Ok(())
}
