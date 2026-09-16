//! Simplify the expression DAG, then place shared and invariant computations.
use super::{
    data::intern_expr as expr, data::*, expressions, from_tlc::Converted, rewrite, term, OptimizeError,
};
use super::{dependencies, timing};
use egglog_engine::EGraph;
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
    let dependencies = timing::time("analyze dependencies", || dependencies::analyze(&converted.data));
    timing::time("validate dependency order", || {
        dependencies.schedules(&converted.data)
    })?;
    let (source, _) = timing::time("emit expression facts", || {
        expressions::emit_facts(&converted.data, &dependencies, &[])
    })?;
    let mut program = term::parse("wyn-expressions.egg", &source)?;
    program.extend(term::parse("expressions-run.egg", expressions::RUN)?);
    program.extend(term::parse("arithmetic.egg", include_str!("arithmetic.egg"))?);
    program.extend(term::parse(
        "wyn-placements.egg",
        &hoist::output(&converted.data),
    )?);
    converted.program = term::parse("ids.egg", include_str!("ids.egg"))?;
    converted.program.extend(program.iter().cloned());
    converted.expression_program = Some(program);
    Ok(converted)
}

fn error(message: &str) -> OptimizeError {
    OptimizeError::Output(format!("scalar optimization: {message}"))
}

fn name(id: ExprId) -> String {
    format!("$expr-{}", id.as_u32())
}

fn simplify(data: &mut AssociatedData) -> Result<(), OptimizeError> {
    let _timing = timing::span("arithmetic EqSat");
    let dependencies = timing::time("analyze dependencies", || dependencies::analyze(data));
    timing::time("validate dependency order", || dependencies.schedules(data))?;
    let (source, live) = timing::time("emit expression facts", || {
        expressions::emit_facts(data, &dependencies, &[])
    })?;
    let program = timing::time("parse expression facts", || {
        term::parse("wyn-expressions.egg", &source)
    })?;
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
    let mut graph = EGraph::default();
    timing::time("load expression graph", || {
        graph.parse_and_run_program(Some("ids.egg".into()), include_str!("ids.egg"))?;
        graph.run_program(program)
    })?;
    timing::time("expression rules", || {
        graph.parse_and_run_program(None, expressions::RUN)
    })?;
    timing::time("load arithmetic rules and facts", || {
        graph.parse_and_run_program(Some("arithmetic.egg".into()), include_str!("arithmetic.egg"))?;
        graph.parse_and_run_program(Some("wyn-constants.egg".into()), &facts)
    })?;
    timing::time("arithmetic and algebra rules", || {
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
