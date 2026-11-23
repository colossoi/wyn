//! Pass orchestration over the authoritative graph.
use super::dataflow::{body, cse, edit, ir::Code};
use super::query::Query;
use super::{analysis, scalar, timing, OptimizeError};
use crate::egglog::graph::Graph;
use crate::LookupMap;
use egglog_engine::Value;
use wyn_graph::dfs_postorder;

mod demand;
mod remove_leaves;

pub(super) fn load(graph: &mut Graph) -> Result<(), OptimizeError> {
    super::dataflow::ir::load(graph)?;
    scalar::load(graph)?;
    super::planning::load(graph)
}

pub(super) fn inline(graph: &mut Graph) -> Result<(), OptimizeError> {
    let _timing = timing::span("egglog helper inlining");
    let selection = timing::span("egglog inline selection");
    edit::clear(graph)?;
    analysis::structure(graph, true)?;
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

fn inline_call(graph: &mut Graph, call: Value, callee: Value) -> Result<(), OptimizeError> {
    let cloning = timing::span("egglog inline cloning");
    let arguments = Query(graph).0.inputs(call);
    let outputs = Query(graph).sequence("Output", call)?;
    let region = edit::clone_region(graph, call, callee, &arguments)?;
    drop(cloning);
    let substitution = timing::span("egglog inline substitution");
    let results = Query(graph).0.yield_values(region);
    if outputs.len() != results.len() {
        return Err(super::output_error("inlined result signature differs from call"));
    }
    edit::replace(graph, &outputs.into_iter().zip(results).collect::<Vec<_>>())?;
    drop(substitution);
    let _splicing = timing::span("egglog inline splicing");
    let operations = Query(graph).operations(region)?;
    edit::splice(graph, call, &operations)
}

pub(super) fn simplify(graph: &mut Graph, hoist: bool) -> Result<(), OptimizeError> {
    let _timing = timing::span(if hoist {
        "egglog hoisting simplification"
    } else {
        "egglog reducing simplification"
    });
    edit::clear(graph)?;
    let mut body = None;
    loop {
        analysis::ensure(graph)?;
        {
            let _timing = timing::span("Rust expression reuse");
            let current = body::BodyGraph::get_or_read(&mut body, Query(graph))?;
            cse::publish(graph, current)?;
        }
        {
            let _timing = timing::span("egglog scalar candidates");
            graph.parse_and_run_program(
                None,
                "(run-schedule (saturate graph-candidates scalar-analysis))",
            )?;
        }
        if reduce_values(graph, &mut body)? {
            continue;
        }

        if remove_dead_values(graph, &mut body)? {
            continue;
        }

        // Only shape-dependent rewrites need planning facts.
        let before_planning = edit::revision(graph)?;
        super::planning::analyze(graph)?;
        if scalar::forward_lengths(graph, "ForwardLength")? {
            body = None;
            graph.parse_and_run_program(None, "(run-schedule (saturate scalar-analysis))")?;
        }
        if reduce_values(graph, &mut body)? {
            continue;
        }
        scratch_shapes(graph)?;
        if edit::revision(graph)? != before_planning {
            body = None;
            continue;
        }
        if hoist && (super::planning::factor_conditionals(graph)? || super::fusion::hoist(graph)?) {
            body = None;
            continue;
        }
        return Ok(());
    }
}

fn remove_dead_values(
    graph: &mut Graph,
    body: &mut Option<body::BodyGraph>,
) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust dead operation selection");
    let query = Query(graph);
    let current = body::BodyGraph::get_or_read(body, query)?;
    let needed = demand::Needed::read(query, current)?;
    let dead =
        current.ir.operations.keys().filter(|op| !needed.operations.contains(op)).collect::<Vec<_>>();
    if !dead.is_empty() {
        let _timing = timing::span("Rust dead operation removal");
        if !remove_leaves::apply(graph, current, &needed, &dead)? {
            let keys = dead.iter().map(|&op| current.operations[op]).collect::<Vec<_>>();
            edit::remove(graph, &keys)?;
            *body = None;
        }
        return Ok(true);
    }
    if let Some((operation, clause)) = demand::dead_clause(query, |name| needed.uses(current, name))? {
        let _timing = timing::span("egglog dead result removal");
        edit::remove_clause(graph, operation, clause)?;
        *body = None;
        return Ok(true);
    }
    Ok(false)
}

fn reduce_values(graph: &mut Graph, body: &mut Option<body::BodyGraph>) -> Result<bool, OptimizeError> {
    let revision = edit::revision(graph)?;
    if scalar::apply(graph, body)? {
        *body = None;
        return Ok(true);
    }
    if edit::revision(graph)? != revision {
        *body = None;
    }
    body::forward_equivalents(graph, body)
}

fn scratch_shapes(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut replacements = Vec::new();
    query.for_each("ScratchAllocation", |row| {
        let operation = row[0];
        if !query.contains("LiveOperation", operation)? || query.0.code(operation)? != Code::Scratch {
            return Ok(());
        }
        let source = query.name(query.0.input(operation, 0_i64)?)?;
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
pub(super) fn validate(graph: &mut Graph) -> Result<(), OptimizeError> {
    analysis::run(graph)
}

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;
