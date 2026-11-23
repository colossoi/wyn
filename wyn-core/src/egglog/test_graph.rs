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
    graph.parse_and_run_program(None, facts)?;
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
