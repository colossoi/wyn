//! Flat graph fixtures for phase tests; no executable syntax is reconstructed.
use super::dataflow::ir::{self, Code};
use super::query::Query;
use super::{from_tlc, memory, OptimizeError};
use egglog_engine::{EGraph, Value, Write};

pub(super) fn fixture(codes: &[(i64, Code)], facts: &str) -> Result<EGraph, OptimizeError> {
    let mut graph = EGraph::default();
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

pub(super) fn import(source: &str) -> Result<EGraph, OptimizeError> {
    let source = crate::compile_thru_tlc(source).unwrap();
    Ok(from_tlc::import(&source)?.0)
}

pub(super) fn entry(graph: &EGraph) -> Value {
    Query(graph).first("EntryOrder").unwrap().unwrap()[0]
}

pub(super) fn root(graph: &EGraph) -> Value {
    Query(graph).required("FunctionRoot", entry(graph)).unwrap()
}
