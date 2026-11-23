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
    Query(graph).required("FunctionRoot", entry(graph)).unwrap()
}

pub(super) fn facts_with_positions(graph: &mut Graph, facts: &str) -> Result<(), OptimizeError> {
    if !facts.contains("FixtureRegion") && !facts.contains("FixtureOrder") {
        graph.parse_and_run_program(None, facts)?;
        return Ok(());
    }
    if !graph.get_function_names().iter().any(|name| name == "FixtureRegion") {
        graph.parse_and_run_program(
            None,
            "
            (function FixtureRegion (OperationKey) RegionKey :no-merge)
            (function FixtureOrder (OperationKey) i64 :no-merge)
        ",
        )?;
    }
    graph.parse_and_run_program(None, facts)?;
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
    graph.update(|mut sink| {
        for (operation, region) in regions {
            sink.set_region(operation, region);
        }
        for (operation, order) in orders {
            sink.set_order(operation, order);
        }
        Ok(())
    })?;
    graph.clear_function("FixtureRegion")?;
    graph.clear_function("FixtureOrder")?;
    Ok(())
}
