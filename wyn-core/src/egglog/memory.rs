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
    if graph.structure().read().targets.is_empty() {
        return Ok(());
    }
    // Full scalar optimization already has the live call sites and SCCs from
    // fusion. Selecting inlines does not invalidate those proofs.
    let inline_all = Query(graph).flag("InlineAll", egglog_engine::RawValues(Vec::new()))?;
    if !inline_all || !analysis::current(graph)? {
        analysis::structure(graph, true)?;
    }
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
    let outputs = Query(graph).0.outputs(call);
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
    loop {
        analysis::ensure(graph)?;
        {
            let _timing = timing::span("Rust expression reuse");
            cse::publish(graph)?;
        }
        {
            let _timing = timing::span("egglog scalar candidates");
            graph.parse_and_run_program(
                None,
                "(run-schedule (saturate graph-candidates scalar-analysis))",
            )?;
        }
        if reduce_values(graph)? {
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
        if reduce_values(graph)? {
            continue;
        }
        scratch_shapes(graph)?;
        if edit::revision(graph)? != before_planning {
            continue;
        }
        if hoist && (super::planning::factor_conditionals(graph)? || super::fusion::hoist(graph)?) {
            continue;
        }
        return Ok(());
    }
}

fn remove_dead_values(graph: &mut Graph) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust dead operation selection");
    let query = Query(graph);
    let needed = demand::Needed::read(query)?;
    let mut dead = Vec::new();
    query.for_each("LiveOperation", |row| {
        if !needed.operations.contains(&row[0]) {
            dead.push(row[0]);
        }
        Ok(())
    })?;
    if !dead.is_empty() {
        let _timing = timing::span("Rust dead operation removal");
        if !remove_leaves::apply(graph, &needed, &dead)? {
            edit::remove(graph, &dead)?;
        }
        return Ok(true);
    }
    if let Some((operation, clause)) = demand::dead_clause(query, |name| needed.names.contains(&name))? {
        let _timing = timing::span("egglog dead result removal");
        edit::remove_clause(graph, operation, clause)?;
        return Ok(true);
    }
    Ok(false)
}

fn reduce_values(graph: &mut Graph) -> Result<bool, OptimizeError> {
    if scalar::apply(graph)? {
        return Ok(true);
    }
    body::forward_equivalents(graph)
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
