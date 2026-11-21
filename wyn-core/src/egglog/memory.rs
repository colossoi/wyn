//! Pass orchestration over the authoritative graph.
use super::dataflow::{
    body, edit,
    ir::{self, Code},
};
use super::query::Query;
use super::{analysis, scalar, timing, OptimizeError};
use crate::LookupMap;
use egglog_engine::sort::VecContainer;
use egglog_engine::{Core, EGraph, Value, Write};
use wyn_graph::dfs_postorder;

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    super::dataflow::ir::load(graph)?;
    scalar::load(graph)?;
    super::planning::load(graph)
}

pub(super) fn inline(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = timing::span("egglog helper inlining");
    let selection = timing::span("egglog inline selection");
    edit::clear(graph)?;
    analysis::structure(graph)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate graph-inline-required) (saturate graph-inline))",
    )?;
    let mut roots = Vec::new();
    let mut calls = LookupMap::<Value, Vec<(Value, Value)>>::default();
    Query(graph).for_each("InlineCandidate", |row| {
        let [caller, call, callee] = [row[0], row[1], row[2]];
        calls
            .entry(caller)
            .or_insert_with(|| {
                roots.push(caller);
                Vec::new()
            })
            .push((call, callee));
        Ok(())
    })?;
    for table in [
        "InlineCall",
        "InlineParallel",
        "InlineRequired",
        "InlineCandidate",
    ] {
        graph.clear_function(table)?;
    }
    if calls.is_empty() {
        return Ok(());
    }
    // Selected targets are nonrecursive. Preparing their bodies first ensures
    // that copying a callee cannot introduce more selected calls.
    let order = dfs_postorder(
        roots,
        |_| false,
        |region, dependencies| {
            if let Some(sites) = calls.get(&region) {
                dependencies.extend(sites.iter().map(|&(_, callee)| callee));
            }
        },
    );
    drop(selection);
    for caller in order {
        if let Some(sites) = calls.remove(&caller) {
            for (call, callee) in sites {
                inline_call(graph, call, callee)?;
            }
        }
    }
    Ok(())
}

fn inline_call(graph: &mut EGraph, call: Value, callee: Value) -> Result<(), OptimizeError> {
    let cloning = timing::span("egglog inline cloning");
    let arguments = Query(graph).sequence("Input", call)?;
    let outputs = Query(graph).sequence("Output", call)?;
    let region = edit::clone_region(graph, call, callee, &arguments)?;
    drop(cloning);
    let substitution = timing::span("egglog inline substitution");
    let results = Query(graph).sequence("RegionYield", region)?;
    if outputs.len() != results.len() {
        return Err(super::output_error("inlined result signature differs from call"));
    }
    edit::replace(graph, &outputs.into_iter().zip(results).collect::<Vec<_>>())?;
    drop(substitution);
    let _splicing = timing::span("egglog inline splicing");
    let operations = Query(graph).operations(region)?;
    edit::splice(graph, call, &operations)
}

pub(super) fn simplify(graph: &mut EGraph, hoist: bool) -> Result<(), OptimizeError> {
    let _timing = timing::span(if hoist {
        "egglog hoisting simplification"
    } else {
        "egglog reducing simplification"
    });
    loop {
        edit::clear(graph)?;
        analysis::ensure(graph)?;
        if body::forward_copies(graph)? {
            continue;
        }
        super::planning::analyze(graph)?;
        graph.parse_and_run_program(
            None,
            "(run-schedule (saturate graph-candidate-index graph-candidates scalar-analysis dataflow-scopes))",
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
            query.for_each("Defines", |row| {
                if let Some(&new) = replacements.get(&row[0]) {
                    if let Some(site) = query.enode("OperationSite", row[1])? {
                        if analysis::count(query, "OutputCount", site[0])? == 1 {
                            copies.push((site[0], row[0], new));
                        }
                    }
                }
                Ok(())
            })?;
            // A proven redundant partial evaluation becomes a pure copy before DCE.
            for (operation, old, name) in copies {
                edit::set_code(graph, operation, Code::Use)?;
                edit::inputs(graph, operation, &[name])?;
                replacements.remove(&old);
            }
            if !replacements.is_empty() {
                edit::replace(graph, &replacements.into_iter().collect::<Vec<_>>())?;
            }
            if body::forward_copies(graph)? {
                continue;
            }
            analysis::ensure(graph)?;
            edit::clear(graph)?;
        }
        graph.parse_and_run_program(
            None,
            "(run-schedule (saturate graph-candidate-index graph-needed) (saturate graph-result-select))",
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

#[cfg(test)]
pub(super) fn validate(graph: &mut EGraph) -> Result<(), OptimizeError> {
    analysis::run(graph)
}

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;
