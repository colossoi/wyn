//! Run execution placement and structural dispatch planning on native facts.
use super::{query::Query, timing, OptimizeError};
use crate::PipelineTopologyPolicy;
use egglog_engine::{EGraph, RawValues, Write};

#[cfg(test)]
#[path = "planning_tests.rs"]
mod tests;

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(Some("planner schema".into()), include_str!("planning/schema.egg"))?;
    Ok(())
}

pub(super) fn place(
    graph: &mut EGraph,
    topology: PipelineTopologyPolicy,
    costs: super::cost::CostModel,
) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(
        Some("planning bridge".into()),
        concat!(
            include_str!("planning/import.egg"),
            "\n",
            include_str!("planning/abi.egg"),
            "\n",
            include_str!("planning/representation.egg"),
            "\n",
            include_str!("planning/boundaries.egg"),
            "\n",
            include_str!("planning/views.egg"),
            "\n",
            include_str!("planning/repeatability.egg"),
            "\n",
            include_str!("planning/shapes.egg"),
            "\n",
            include_str!("planning/maps.egg"),
            "\n",
            include_str!("planning/segments.egg"),
            include_str!("planning/execution.egg"),
            "\n",
            include_str!("planning/bindings.egg"),
            include_str!("planning/output-layouts.egg"),
            "\n",
            include_str!("planning/capacity.egg"),
            "\n",
            include_str!("planning/loops.egg"),
            "\n",
            include_str!("planning/dispatch.egg"),
            "\n",
            include_str!("planning/schedule.egg"),
            "\n",
            include_str!("planning/residency.egg"),
            "\n",
            include_str!("planning/allocation.egg"),
            "\n",
            include_str!("planning/epilogues.egg")
        ),
    )?;
    let policy = match topology {
        PipelineTopologyPolicy::AllowGenerated => "AllowGeneratedTopology",
        PipelineTopologyPolicy::AuthoredOnly => "AuthoredTopology",
    };
    graph.update(|mut sink| sink.add(policy, RawValues(Vec::new())))?;
    timing::time("egglog placement / import", || {
        graph.parse_and_run_program(
            None,
            "(run-schedule (saturate planning-constants) (saturate planning-import))",
        )
    })?;
    timing::time("egglog placement / order", || order(graph))?;
    super::cost::register(graph, costs, false);
    graph.parse_and_run_program(Some("native loop costs".into()), include_str!("cost.egg"))?;
    graph.parse_and_run_program(
        Some("execution placement".into()),
        include_str!("planning/place.egg"),
    )?;
    Ok(())
}

pub(super) fn schedule(graph: &mut EGraph, costs: super::cost::CostModel) -> Result<(), OptimizeError> {
    super::cost::register(graph, costs, true);
    graph.parse_and_run_program(
        Some("schedule inputs".into()),
        include_str!("planning/prepare-schedule.egg"),
    )?;
    Query(graph).for_each("ParallelCollective", |row| {
        if Query(graph).lookup("OperationGrid", (row[0],))?.is_some() {
            Query(graph).required("SelectedCollectiveGroups", (row[0],))?;
        }
        Ok(())
    })?;
    graph.parse_and_run_program(
        Some("structural scheduling".into()),
        include_str!("planning/run.egg"),
    )?;
    if std::env::var_os("WYN_EGGLOG_PROFILE").is_some() {
        for output in graph.parse_and_run_program(None, "(print-stats) (print-size)")? {
            eprintln!("{output}");
        }
    }
    Ok(())
}

fn order(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate order-import) (saturate (saturate order-select) order-emit))",
    )?;
    Query(graph).for_each("OrderPending", |_| {
        Err(OptimizeError::Extraction(
            "cycle in the fused execution graph".into(),
        ))
    })
}

#[cfg(test)]
#[path = "planning/order_tests.rs"]
mod order_tests;
