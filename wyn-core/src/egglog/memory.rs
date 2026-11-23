//! Pass orchestration over the authoritative graph.
use super::dataflow::{cse, edit, ir::Code};
use super::query::Query;
use super::{analysis, scalar, timing, OptimizeError};
use crate::egglog::graph::Graph;
use egglog_engine::Value;

mod demand;
mod forward;
mod remove_leaves;

pub(super) fn load(graph: &mut Graph) -> Result<(), OptimizeError> {
    super::dataflow::ir::load(graph)?;
    scalar::load(graph)?;
    super::planning::load(graph)
}

pub(super) fn inline(graph: &mut Graph, all: bool) -> Result<(), OptimizeError> {
    let _timing = timing::span("egglog helper inlining");
    let selection = timing::span("Rust inline selection");
    if graph.structure().read().targets.is_empty() {
        return Ok(());
    }
    let calls = analysis::calls::Reachability::read(graph)?.inlines(graph, all)?;
    drop(selection);
    if !calls.is_empty() {
        analysis::seed_substitution(graph)?;
        for (call, callee) in calls {
            inline_call(graph, call, callee)?;
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
        if !remove_leaves::apply(graph, &dead, |_, operation| {
            Ok(needed.use_sites(operation).collect())
        })? {
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
    forward::run(graph)
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
