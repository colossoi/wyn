//! Pass orchestration over the authoritative graph.
use super::dataflow::{
    body, cse, edit,
    ir::{self, Code},
};
use super::query::Query;
use super::{analysis, scalar, timing, OptimizeError};
use crate::LookupMap;
use egglog_engine::{EGraph, Value};
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
    edit::clear(graph)?;
    loop {
        analysis::ensure(graph)?;
        let body = {
            let _timing = timing::span("Rust expression reuse");
            cse::publish(graph)?
        };
        {
            let _timing = timing::span("egglog scalar candidates");
            graph.parse_and_run_program(
                None,
                "(run-schedule (saturate graph-candidates scalar-analysis))",
            )?;
        }
        if reduce_values(graph, Some(body))? {
            continue;
        }

        if remove_dead_values(graph)? {
            continue;
        }

        // Only shape-dependent rewrites need planning facts.
        let before_planning = edit::revision(graph)?;
        super::planning::analyze(graph)?;
        if scalar::forward_lengths(graph, "ForwardLength")? {
            graph.parse_and_run_program(None, "(run-schedule (saturate scalar-analysis))")?;
        }
        if reduce_values(graph, None)? {
            continue;
        }
        scratch_shapes(graph)?;
        if edit::revision(graph)? != before_planning {
            edit::clear_demand(graph)?;
            continue;
        }
        if hoist && (super::planning::factor_conditionals(graph)? || super::fusion::hoist(graph)?) {
            continue;
        }
        return Ok(());
    }
}

fn remove_dead_values(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    let _timing = timing::span("egglog dead operation selection");
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate graph-needed) (saturate graph-result-select))",
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
        let _timing = timing::span("egglog dead operation removal");
        edit::remove(graph, &dead)?;
        return Ok(true);
    }
    if let Some(candidate) = query.first("DeadResultClause")? {
        let _timing = timing::span("egglog dead result removal");
        let clause = graph.value_to_base::<i64>(candidate[1]);
        edit::remove_clause(graph, candidate[0], clause)?;
        return Ok(true);
    }
    Ok(false)
}

fn reduce_values(graph: &mut EGraph, mut body: Option<body::BodyGraph>) -> Result<bool, OptimizeError> {
    let revision = edit::revision(graph)?;
    if scalar::apply(graph, &mut body)? {
        return Ok(true);
    }
    if edit::revision(graph)? != revision {
        body = None;
    }
    body::forward_equivalents(graph, body)
}

fn scratch_shapes(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut replacements = Vec::new();
    query.for_each("ScratchAllocation", |row| {
        let operation = row[0];
        if !query.contains("LiveOperation", operation)?
            || ir::code(graph, query.required("OperationPayload", operation)?) != Code::Scratch
        {
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
    scalar::materialize_scratch(graph, replacements)
}

#[cfg(test)]
pub(super) fn validate(graph: &mut EGraph) -> Result<(), OptimizeError> {
    analysis::run(graph)
}

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;
