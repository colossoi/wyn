//! Pass orchestration over the authoritative graph.
use super::dataflow::{
    edit,
    ir::{self, Code},
};
use super::query::Query;
use super::{analysis, scalar, OptimizeError};
use crate::LookupMap;
use egglog_engine::sort::VecContainer;
use egglog_engine::{Core, EGraph, Write};

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    super::dataflow::ir::load(graph)?;
    scalar::load(graph)?;
    super::planning::load(graph)
}

pub(super) fn inline(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = super::timing::span("egglog helper inlining");
    loop {
        edit::clear(graph)?;
        analysis::run(graph)?;
        graph.parse_and_run_program(
            None,
            "(run-schedule (saturate graph-candidates) (saturate graph-inline))",
        )?;
        let Some(candidate) = Query(graph).first("InlineCandidate")? else {
            return Ok(());
        };
        let call = candidate[0];
        let arguments = Query(graph).sequence("Input", call)?;
        let outputs = Query(graph).sequence("Output", call)?;
        let region = edit::clone_region(graph, call, candidate[1], &arguments)?;
        let results = Query(graph).sequence("RegionYield", region)?;
        if outputs.len() != results.len() {
            return Err(super::output_error("inlined result signature differs from call"));
        }
        edit::replace(graph, &outputs.into_iter().zip(results).collect::<Vec<_>>())?;
        let operations = Query(graph).operations(region)?;
        edit::splice(graph, call, &operations)?;
    }
}

pub(super) fn simplify(graph: &mut EGraph, hoist: bool) -> Result<(), OptimizeError> {
    let _timing = super::timing::span(if hoist {
        "egglog hoisting simplification"
    } else {
        "egglog reducing simplification"
    });
    loop {
        edit::clear(graph)?;
        analysis::run(graph)?;
        super::planning::analyze(graph)?;
        graph.parse_and_run_program(
            None,
            "(run-schedule (saturate graph-candidates scalar-analysis dataflow-scopes))",
        )?;
        if scalar::apply(graph)? || scratch_shapes(graph)? {
            continue;
        }
        let query = Query(graph);
        let mut replacements = LookupMap::default();
        query.for_each("ForwardCandidate", |row| {
            replacements.entry(row[0]).or_insert(row[1]);
            Ok(())
        })?;
        if !replacements.is_empty() {
            let mut copies = Vec::new();
            for (&old, &new) in &replacements {
                Query(graph).for_each("Defines", |row| {
                    if row[0] == old {
                        if let Some(site) = Query(graph).enode("OperationSite", row[1])? {
                            if analysis::count(Query(graph), "OutputCount", site[0])? == 1 {
                                copies.push((site[0], new));
                            }
                        }
                    }
                    Ok(())
                })?;
            }
            // A proven redundant partial evaluation becomes a pure copy before DCE.
            for (operation, name) in copies {
                edit::set_code(graph, operation, Code::Use)?;
                edit::inputs(graph, operation, &[name])?;
            }
            edit::replace(graph, &replacements.into_iter().collect::<Vec<_>>())?;
            // Forwarded definitions become dead once all consumers have been redirected.
            analysis::run(graph)?;
            edit::clear(graph)?;
        }
        graph.parse_and_run_program(
            None,
            "(run-schedule (saturate graph-candidates) (saturate graph-needed) (saturate graph-result-select))",
        )?;
        let query = Query(graph);
        let mut dead = Vec::new();
        query.for_each("LiveOperation", |row| {
            if !query.contains("NeededOperation", row[0])? {
                dead.push(row[0]);
            }
            Ok(())
        })?;
        if !dead.is_empty() {
            edit::remove(graph, &dead)?;
            continue;
        }
        if let Some(candidate) = Query(graph).first("DeadResultClause")? {
            let clause = graph.value_to_base::<i64>(candidate[1]);
            edit::remove_clause(graph, candidate[0], clause)?;
            continue;
        }
        if hoist && (super::planning::factor_conditionals(graph)? || super::fusion::motion(graph, true)?) {
            continue;
        }
        return Ok(());
    }
}

fn scratch_shapes(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let mut replacements = Vec::new();
    query.for_each("LiveOperation", |row| {
        let operation = row[0];
        if ir::code(graph, query.required("OperationPayload", operation)?) != Code::Scratch {
            return Ok(());
        }
        let source = query.name(query.required("Input", (operation, 0_i64))?)?;
        let Some(info) = query.enode("LogicalValue", query.required("ValueType", source)?)? else {
            return Ok(());
        };
        let Some(array) = query.enode("ArrayType", info[0])? else {
            return Ok(());
        };
        if query.vector(array[1])?.len() == 1 && query.enode("ElementValue", array[2])?.is_some() {
            if let Some(count) = query.lookup("LogicalOuterCount", source)? {
                replacements.push((operation, count));
            }
        }
        Ok(())
    })?;
    for &(operation, count) in &replacements {
        edit::set_code(graph, operation, Code::Uninitialized)?;
        edit::inputs(graph, operation, &[])?;
        graph.update(|mut sink| {
            let shape = sink.container_to_value(VecContainer {
                data: vec![count],
                do_rebuild: true,
            });
            sink.set("OperationShape", operation, shape)
        })?;
    }
    Ok(!replacements.is_empty())
}

pub(super) fn validate(graph: &mut EGraph) -> Result<(), OptimizeError> {
    analysis::run(graph)
}

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;
