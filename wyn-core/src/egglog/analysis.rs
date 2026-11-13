//! Validate and derive effects from the authoritative memory bodies.
use super::{memory, OptimizeError};
use egglog_engine::EGraph;

#[cfg(test)]
#[path = "analysis_tests.rs"]
mod tests;

pub(super) fn run(graph: &mut EGraph) -> Result<(), OptimizeError> {
    memory::validate(graph)?;
    super::timing::time("logical cost analysis", || {
        super::timing::evaluate(
            graph,
            None,
            "(run-schedule (saturate schedule-cost memory-origin))",
        )
    })?;
    super::timing::time("logical effect analysis", || {
        super::timing::evaluate(graph, None, "(run-schedule (saturate memory-effects))")
    })?;
    super::timing::evaluate(
        graph,
        None,
        "(run-schedule
        (saturate schedule-cost simplify-analysis)
        (saturate reuse-reads)
        (saturate reuse-legality)
        (saturate reuse-select))",
    )?;
    Ok(())
}
