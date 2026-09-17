//! Simplify the expression DAG, then place shared and invariant computations.
use super::data::{intern_expr, ExprId, Ir};
use super::{term, Expressions, OptimizeError, Placed, Program, Simplified};
use crate::egglog::dependencies::analyze;
use crate::egglog::expressions::{emit_facts, RUN};
use crate::egglog::parse_program;
use crate::egglog::rewrite::all;
use crate::egglog::timing::{span, time};
use crate::types::{Type, TypeName};
use egglog_engine::ast::Command;
use egglog_engine::EGraph;
use std::collections::BTreeSet;

mod fold;
mod hoist;
mod read;
pub(super) use hoist::index as placement_index;

/// Simplify typed scalar expressions with equality saturation and cost-based extraction.
pub fn simplify(mut program: Program<Expressions>) -> Result<Program<Simplified>, OptimizeError> {
    simplify_ir(&mut program.ir, program.state)?;
    Ok(Program {
        ir: program.ir,
        state: Simplified,
    })
}

/// Place shared and invariant computations in structured regions.
/// Memory reads and opaque calls are never speculated.
pub fn place(mut program: Program<Simplified>) -> Result<Program<Placed>, OptimizeError> {
    let placements = hoist::run(&mut program.ir)?;
    let _output = span("export optimized expressions");
    let mut facts = expression_facts(&program.ir)?;
    facts.extend(parse_program("wyn-placements.egg", &hoist::output(&placements))?);
    Ok(Program {
        ir: program.ir,
        state: Placed { facts, placements },
    })
}

fn expression_facts(data: &Ir) -> Result<Vec<Command>, OptimizeError> {
    let dependencies = time("analyze dependencies", || analyze(data));
    time("validate dependency order", || dependencies.schedules(data))?;
    let (source, _) = time("emit expression facts", || emit_facts(data, &dependencies, &[]))?;
    let mut facts = parse_program("ids.egg", include_str!("ids.egg"))?;
    facts.extend(parse_program("wyn-expressions.egg", &source)?);
    facts.extend(parse_program("expressions-run.egg", RUN)?);
    facts.extend(parse_program("arithmetic.egg", include_str!("arithmetic.egg"))?);
    Ok(facts)
}
fn error(message: &str) -> OptimizeError {
    OptimizeError::Output(format!("scalar optimization: {message}"))
}

fn name(id: ExprId) -> String {
    format!("$expr-{}", id.as_u32())
}

fn simplify_ir(data: &mut Ir, expressions: Expressions) -> Result<(), OptimizeError> {
    let _timing = span("arithmetic EqSat");
    let dependencies = time("analyze dependencies", || analyze(data));
    time("validate dependency order", || dependencies.schedules(data))?;
    let Expressions { facts: program, live } = expressions;
    let mut facts = String::new();
    for (&id, t) in &data.types {
        if matches!(t.ty, Type::Constructed(TypeName::Int(_) | TypeName::UInt(_), _)) {
            facts.push_str(&format!("(IntegerType (TypeId {}))\n", id.as_u32()));
        }
    }
    for &id in &live {
        fold::facts(data, id, &mut facts);
    }
    // Keep alternatives in one e-graph throughout analysis and extraction.
    // Bound reassociation to avoid exponential exploration of long sums.
    let mut graph = EGraph::default();
    time("load expression graph", || {
        graph.parse_and_run_program(Some("ids.egg".into()), include_str!("ids.egg"))?;
        graph.run_program(program)
    })?;
    time("expression rules", || graph.parse_and_run_program(None, RUN))?;
    time("load arithmetic rules and facts", || {
        graph.parse_and_run_program(Some("arithmetic.egg".into()), include_str!("arithmetic.egg"))?;
        graph.parse_and_run_program(Some("wyn-constants.egg".into()), &facts)
    })?;
    time("arithmetic and algebra rules", || {
        graph.parse_and_run_program(
            None,
            "(run-schedule (repeat 4 (seq (run arithmetic) (run algebra))))",
        )
    })?;
    let mut evaluated = BTreeSet::new();
    for round in 0..32 {
        if !read::constants(&mut graph, data, &dependencies.live, round, &mut evaluated)? {
            break;
        }
        time("arithmetic rules", || {
            graph.parse_and_run_program(None, "(run-schedule (repeat 2 (run arithmetic)))")
        })?;
    }
    let replacements = read::extract(&mut graph, data, &live)?;
    time("apply extracted expressions", || all(data, &replacements));
    Ok(())
}

#[cfg(test)]
#[path = "scalar_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "scalar_scaling_tests.rs"]
mod scaling_tests;
