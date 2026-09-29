//! Structural fusion on the persistent egglog graph.
//! Plans remain in egglog for the subsequent placement and scheduling passes.
use super::timing::{span, time};
use super::{parse_program, OptimizeError};
use egglog_engine::EGraph;

mod reachability;

pub(super) fn new_graph() -> Result<EGraph, OptimizeError> {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(
        Some("fusion schema".into()),
        concat!(include_str!("../ids.egg"), "\n", include_str!("schema.egg")),
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
    time("egglog fusion / run schedule", || reachability::run(graph))?;
    graph.run_program(parse_program(
        "fusion signatures",
        "(run-schedule (saturate fusion-signatures))",
    )?)?;
    Ok(())
}
