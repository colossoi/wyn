//! Settle source summaries and execution policies in the structural egraph.
use super::query::Query;
use super::{parse_program, timing, OptimizeError};
use egglog_engine::EGraph;

mod work;

#[cfg(test)]
#[path = "analysis_tests.rs"]
mod tests;

pub(super) fn run(graph: &mut EGraph) -> Result<(), OptimizeError> {
    timing::time("egglog analysis / summaries", || {
        graph
            .run_program(parse_program(
                "source summary",
                "(run-schedule (saturate source-summary))",
            )?)
            .map_err(OptimizeError::from)
    })?;
    timing::time("egglog analysis / dependencies", || {
        graph
            .run_program(parse_program(
                "source dependencies",
                "(run-schedule (saturate source-dependencies))",
            )?)
            .map_err(OptimizeError::from)
    })?;
    let query = Query(graph);
    query.for_function("SourceOperationValue", |_, value| {
        query.required("SourceWork", (value,))?;
        query.required("SourceReadOnly", (value,))?;
        query.required("SourceDuplicable", (value,))?;
        Ok(())
    })?;
    for table in ["SourceSummaryEnters", "SourceRegionEnters"] {
        query.for_each(table, |row| {
            query.required("SourceRegionWork", (row[1],))?;
            query.required("SourceRegionDuplicable", (row[1],))?;
            Ok(())
        })?;
    }
    work::register(graph)?;
    timing::time("egglog analysis / execution facts", || {
        graph.parse_and_run_program(Some("execution summaries".into()), concat!(
            include_str!("analysis/work.egg"), "\n", include_str!("analysis/effects.egg"),
            "\n(run-schedule (saturate work-summaries) (saturate work-costs) (saturate work-select) (saturate effect-positions) effect-edges)"
        ))
    })?;
    graph.parse_and_run_program(Some("fusion import".into()), include_str!("fusion/import.egg"))?;
    Ok(())
}
