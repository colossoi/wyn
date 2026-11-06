//! Structural fusion on the persistent egglog graph.
//! Plans remain in egglog for the subsequent placement and scheduling passes.
use super::timing::{span, time};
use super::{parse_program, OptimizeError};
use egglog_engine::EGraph;

#[cfg(test)]
mod tests;

pub(super) fn new_graph() -> Result<EGraph, OptimizeError> {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(
        Some("fusion schema".into()),
        concat!(include_str!("../flow.egg"), "\n", include_str!("schema.egg")),
    )?;
    Ok(graph)
}

pub(super) fn run(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = span("egglog fusion");
    let rules = time("egglog fusion / parse rules", || {
        parse_program("fusion.egg", include_str!("fusion.egg"))
    })?;
    time("egglog fusion / load rules", || graph.run_program(rules))?;
    graph.parse_and_run_program(Some("fusion composition".into()), include_str!("composition.egg"))?;
    graph.run_program(parse_program(
        "fusion import",
        "(run-schedule (saturate source-fusion))",
    )?)?;
    time("egglog fusion / run schedule", || {
        graph.parse_and_run_program(None, "(PlanningRound 0) (run-schedule (saturate (seq (saturate fusion-dependencies) (saturate fusion-paths) (saturate fusion-counts) (saturate fusion-candidates) (saturate fusion-select) fusion-choose (saturate fusion-record) (saturate fusion-commit) fusion-clear-candidates fusion-clear-paths fusion-union fusion-clear)))")
    })?;
    graph.run_program(parse_program(
        "fusion signatures",
        "(run-schedule (saturate fusion-signatures))",
    )?)?;
    Ok(())
}

#[cfg(test)]
#[path = "reachability_tests.rs"]
mod reachability_tests;
