//! Simplify the expression DAG, then place shared and invariant computations.
use super::timing;
use super::{data::*, expressions, extract, from_tlc::Converted, fusion::expr, rewrite, OptimizeError};
use egglog_engine::{ast::Command, EGraph};
use std::collections::{BTreeMap, BTreeSet};

mod fold;
mod hoist;
mod read;
pub(super) use hoist::index as placement_index;

/// Run after expression insertion and before scheduling. Arithmetic uses equality
/// saturation and cost-based extraction; hoisting traverses the extracted DAG and
/// structured regions to place computations separately from expression identity.
/// Memory reads and opaque calls are never speculated.
pub fn optimize_expressions(mut converted: Converted) -> Result<Converted, OptimizeError> {
    if converted.expression_program.is_none() || !converted.data.blocks.is_empty() {
        return Err(error("insert expressions first, and optimize before scheduling"));
    }
    simplify(&mut converted.data)?;
    hoist::run(&mut converted.data)?;
    let _output = timing::span("export optimized expressions");
    let (mut program, _) = expressions::program(&converted.data, &[])?;
    program.extend(expressions::parse(include_str!("arithmetic.egg"))?);
    program.extend(expressions::parse(&hoist::output(&converted.data))?);
    converted.program = expressions::parse(include_str!("ids.egg"))?;
    converted.program.extend(program.iter().cloned());
    converted.expression_program = Some(program);
    Ok(converted)
}

fn error(message: &str) -> OptimizeError {
    OptimizeError::Output(format!("scalar optimization: {message}"))
}

fn graph(program: Vec<Command>, rules: &str, facts: &str) -> Result<EGraph, OptimizeError> {
    let mut graph = EGraph::default();
    timing::time("load and run expression graph", || {
        graph.parse_and_run_program(None, include_str!("ids.egg"))?;
        graph.run_program(program)
    })?;
    timing::time("load rules", || graph.parse_and_run_program(None, rules))?;
    timing::time("load analysis facts and run rules", || {
        graph.parse_and_run_program(None, facts)
    })?;
    Ok(graph)
}

fn name(id: ExprId) -> String {
    format!("$expr-{}", id.as_u32())
}

fn simplify(data: &mut AssociatedData) -> Result<(), OptimizeError> {
    let _timing = timing::span("arithmetic EqSat");
    let (program, live) = expressions::program(data, &[])?;
    let mut facts = String::new();
    for (&id, t) in &data.types {
        if matches!(
            t.ty,
            crate::types::Type::Constructed(
                crate::types::TypeName::Int(_) | crate::types::TypeName::UInt(_),
                _
            )
        ) {
            facts.push_str(&format!("(IntegerType (TypeId {}))\n", id.as_u32()));
        }
    }
    for &id in &live {
        fold::facts(data, id, &mut facts);
    }
    // Keep alternatives in one e-graph throughout analysis and extraction.
    // Bound reassociation to avoid exponential exploration of long sums.
    facts.push_str("(run-schedule (repeat 4 (seq (run arithmetic) (run algebra))))\n");
    let mut graph = graph(program, include_str!("arithmetic.egg"), &facts)?;
    let mut evaluated = BTreeSet::new();
    for round in 0..32 {
        if !read::constants(&mut graph, data, round, &mut evaluated)? {
            break;
        }
        timing::time("arithmetic rules", || {
            graph.parse_and_run_program(None, "(run-schedule (repeat 2 (run arithmetic)))")
        })?;
    }
    let replacements = read::extract(&mut graph, data, &live)?;
    timing::time("apply extracted expressions", || {
        rewrite::all(data, &replacements)
    });
    Ok(())
}

#[cfg(test)]
#[path = "scalar_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "scalar_scaling_tests.rs"]
mod scaling_tests;
