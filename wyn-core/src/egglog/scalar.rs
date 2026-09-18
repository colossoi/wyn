//! Simplify the expression DAG, then place shared and invariant computations.
use super::data::intern_expr;
use super::{term, Expressions, OptimizeError, Placed, Program, Simplified};
use crate::egglog::rewrite::all;
use crate::egglog::timing::span;

mod fold;
mod hoist;
mod read;
pub(super) use hoist::index as placement_index;

/// Fold constants and simplify scalar expressions, optionally exploring algebraic alternatives.
pub fn simplify(
    mut program: Program<Expressions>,
    algebra: bool,
) -> Result<Program<Simplified>, OptimizeError> {
    let data = &mut program.ir;
    let _timing = span("egglog arithmetic EqSat");
    let Expressions { mut graph } = program.state;
    // Keep alternatives in one e-graph throughout analysis and extraction.
    // Bound reassociation to avoid exponential exploration of long sums.
    fold::register(&mut graph);
    graph.parse_and_run_program(Some("arithmetic.egg".into()), include_str!("arithmetic.egg"))?;
    graph.update(|sink| fold::facts(data, sink))?;
    graph.parse_and_run_program(
        None,
        if algebra {
            "(run-schedule (seq (saturate (run arithmetic)) (repeat 4 (seq (run algebra) (saturate (run arithmetic))))))"
        } else {
            "(run-schedule (saturate (run arithmetic)))"
        },
    )?;
    let replacements = read::extract(&graph, data)?;
    all(data, &replacements);
    Ok(Program {
        ir: program.ir,
        state: Simplified,
    })
}

/// Place shared and invariant computations in structured regions.
/// Memory reads and opaque calls are never speculated.
pub fn place(mut program: Program<Simplified>) -> Result<Program<Placed>, OptimizeError> {
    let placements = hoist::run(&mut program.ir)?;
    Ok(Program {
        ir: program.ir,
        state: Placed { placements },
    })
}

fn error(message: &str) -> OptimizeError {
    OptimizeError::Output(format!("scalar optimization: {message}"))
}

#[cfg(test)]
#[path = "scalar_tests.rs"]
mod tests;
