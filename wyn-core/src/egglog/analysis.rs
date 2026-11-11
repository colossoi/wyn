//! Validate and derive effects from the authoritative memory bodies.
use super::{memory, OptimizeError};
use egglog_engine::EGraph;

#[cfg(test)]
#[path = "analysis_tests.rs"]
mod tests;

pub(super) fn run(graph: &mut EGraph) -> Result<(), OptimizeError> {
    memory::validate(graph)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate schedule-cost memory-origin) (saturate memory-effects))",
    )?;
    Ok(())
}
