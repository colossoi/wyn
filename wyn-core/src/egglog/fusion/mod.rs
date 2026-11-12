//! Fusion replaces unified statements and their memory bindings in egglog.
#[cfg(test)]
use super::timing::span;
use super::{memory, OptimizeError};
use egglog_engine::EGraph;

#[cfg(test)]
mod tests;

pub(super) fn new_graph() -> Result<EGraph, OptimizeError> {
    let mut graph = EGraph::default();
    memory::load(&mut graph)?;
    Ok(graph)
}

pub(super) const SCHEDULE: &str = "(seq
    (saturate fusion-index memory-names)
    (saturate memory-scopes)
    (saturate schedule-cost memory-origin)
    (saturate fusion-domains memory-rename memory-names)
    (saturate fusion-forward memory-names)
    (saturate fusion-select memory-edits memory-rename memory-names)
    (saturate schedule-cost memory-origin)
    (saturate memory-effects simplify-analysis)
    (saturate fusion-scopes memory-rename memory-names)
    (saturate fusion-candidates)
    (saturate memory-rewrite-fusion memory-rename))";

#[cfg(test)]
pub(super) fn run(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = span("egglog fusion");
    // Finish analysis before consulting negative facts about dependencies.
    // Candidate keys include the program revision, so stale proofs cannot
    // authorize contractions after surrounding statements have changed.
    graph.parse_and_run_program(None, &format!("(run-schedule (saturate {SCHEDULE}))"))?;
    Ok(())
}

#[cfg(test)]
#[path = "reachability_tests.rs"]
mod reachability_tests;
