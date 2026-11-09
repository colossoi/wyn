//! Structural fusion on the persistent egglog graph.
//! Plans remain in egglog for the subsequent placement and scheduling passes.
use super::query::Query;
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
    graph.parse_and_run_program(
        Some("parallel schema".into()),
        include_str!("../parallel/schema.egg"),
    )?;
    let rules = time("egglog fusion / parse rules", || {
        parse_program("fusion.egg", include_str!("fusion.egg"))
    })?;
    time("egglog fusion / load rules", || graph.run_program(rules))?;
    graph.parse_and_run_program(Some("parallel IR".into()), include_str!("../parallel.egg"))?;
    graph.run_program(parse_program(
        "fusion import",
        "(run-schedule (saturate source-fusion) (saturate traversal-depths))",
    )?)?;
    Query(graph).for_each("Current", |row| {
        Query(graph).required("PlanBody", (row[1],))?;
        Ok(())
    })?;
    graph.parse_and_run_program(None, "(PlanningRound 0)")?;
    let choose = parse_program("fusion round", "(run-schedule (seq (saturate fusion-counts) (saturate fusion-dependencies) (saturate fusion-paths) (saturate fusion-counts) (saturate fusion-candidates) (saturate fusion-replacements) (saturate fusion-select) fusion-choose (saturate fusion-commit)))")?;
    let commit = parse_program(
        "fusion contraction",
        "(run-schedule (seq fusion-union fusion-clear))",
    )?;
    time("egglog fusion / run schedule", || -> Result<(), OptimizeError> {
        loop {
            graph.run_program(choose.clone())?;
            let chosen = Query(graph).row("Chosen", |_| true)?.is_some();
            // Proofs belong to this graph revision. Retain only the frozen
            // replacement across contraction, including when there is no candidate.
            for table in [
                "Pair",
                "PlanCandidate",
                "JoinCandidate",
                "JoinedCandidate",
                "StreamRepresentationBlocked",
                "MaskCandidate",
                "MaskBindings",
                "MaskTotals",
                "Retained",
                "ExclusiveStream",
                "ClearStream",
                "ClearOutputStream",
                "FilterPostMapCompatible",
                "BarrierAllowed",
                "MaskablePlan",
                "GroupReach",
                "OrderWalk",
                "GroupPath",
                "OrderIntermediate",
                "BestPriority",
                "BestRegion",
                "BestProducer",
                "BestConsumer",
            ] {
                graph.clear_function(table)?;
            }
            if !chosen {
                break;
            }
            graph.run_program(commit.clone())?;
        }
        Ok(())
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
