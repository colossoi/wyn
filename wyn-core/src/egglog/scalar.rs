//! Simplify the expression DAG, then place shared and invariant computations.
use super::timing;
use super::{data::*, expressions, extract, from_tlc::Converted, fusion::expr, rewrite, OptimizeError};
use egglog_engine::{ast::Command, EGraph};
use std::collections::{BTreeMap, BTreeSet};

mod fold;
mod hoist;
mod read;
pub(super) use hoist::at as placements;

/// Run after expression insertion and before scheduling. Arithmetic uses equality
/// saturation and cost-based extraction; hoisting records evaluation sites separately
/// from expression identity. Memory reads and opaque calls are never speculated.
pub fn optimize_expressions(mut converted: Converted) -> Result<Converted, OptimizeError> {
    if converted.expression_program.is_none() || !converted.data.blocks.is_empty() {
        return Err(error("insert expressions first, and optimize before scheduling"));
    }
    simplify(&mut converted.data)?;
    hoist::run(&mut converted.data)?;
    let _output = timing::span("export optimized expressions");
    let (mut program, live) = expressions::program(&converted.data, &[])?;
    program.extend(expressions::parse(include_str!("arithmetic.egg"))?);
    program.extend(expressions::parse(include_str!("hoist.egg"))?);
    program.extend(expressions::parse(&hoist::facts(&converted.data, &live)?)?);
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

/// Direct children only: executions and lexical lambda bodies are separate graphs.
fn children(data: &AssociatedData, id: ExprId) -> Vec<ExprId> {
    match &data.expressions[id].kind {
        ExprKind::PureApp { function, args } => {
            std::iter::once(*function).chain(args.iter().copied()).collect()
        }
        ExprKind::Tuple(xs) | ExprKind::Vector(xs) | ExprKind::Closure { captures: xs, .. } => xs.clone(),
        ExprKind::Coerce(x) | ExprKind::Project { tuple: x, .. } => vec![*x],
        ExprKind::If {
            condition,
            then_value,
            else_value,
        } => vec![*condition, *then_value, *else_value],
        ExprKind::Array(a) => {
            fn array(a: &Array, out: &mut Vec<ExprId>) {
                match a {
                    Array::Value(x) => out.push(*x),
                    Array::Literal(xs) => out.extend(xs),
                    Array::Zip(xs) => xs.iter().for_each(|a| array(a, out)),
                    Array::Range { start, len, step } => {
                        out.extend([Some(*start), Some(*len), *step].into_iter().flatten())
                    }
                }
            }
            let mut out = vec![];
            array(a, &mut out);
            out
        }
        _ => vec![],
    }
}

#[cfg(test)]
#[path = "scalar_tests.rs"]
mod tests;
