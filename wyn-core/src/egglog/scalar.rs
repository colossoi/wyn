//! Native scalar admission and optimization over the shared source graph.
//! Selected-expression evaluation stays in this graph through emission.
use super::query::Query;
use super::{parse_program, source, timing, OptimizeError, ScalarOptimization};
use egglog_engine::{EGraph, RawValues, Write};

pub(super) mod evaluation;
pub(super) mod extract;
mod fold;
mod import;

#[cfg(test)]
#[path = "scalar/analysis_tests.rs"]
mod analysis_tests;

mod selected;
pub(super) use selected::Selected;

const RULES: &str = concat!(
    include_str!("scalar/schema.egg"),
    "\n",
    include_str!("scalar/extents.egg"),
    "\n",
    include_str!("scalar/analysis.egg"),
    "\n",
    include_str!("scalar/inline.egg"),
    "\n",
    include_str!("scalar/micro.egg"),
);

pub(super) fn admit(graph: &mut EGraph) -> Result<(), OptimizeError> {
    fold::register(graph);
    graph.parse_and_run_program(Some("scalar rules".into()), RULES)?;
    graph.parse_and_run_program(
        Some("scalar admission".into()),
        include_str!("scalar/admission.egg"),
    )?;
    graph.parse_and_run_program(Some("host preferences".into()), include_str!("host.egg"))?;
    graph.parse_and_run_program(None, "(run-schedule (saturate scalar-admission) (saturate scalar-contexts) (saturate scalar-demands) (saturate host-targets) (saturate scalar-import) (run readout))")?;
    Ok(())
}

pub(super) fn run(
    graph: &mut EGraph,
    identities: &source::Identities<'_>,
    policy: ScalarOptimization,
) -> Result<Selected, OptimizeError> {
    let _timing = timing::span("egglog scalar optimization");
    let schedule = parse_program(
        "scalar fixed point",
        match policy {
            ScalarOptimization::Basic => include_str!("scalar/basic.egg"),
            ScalarOptimization::Full => include_str!("scalar/schedule.egg"),
        },
    )?;
    let facts = Query(graph);
    for &region in identities.scopes.keys() {
        facts.required("ExprContextForScope", (region,))?;
    }
    timing::time("egglog scalar / admission", || {
        graph.parse_and_run_program(
            Some("native scalar import".into()),
            include_str!("scalar/import.egg"),
        )?;
        graph.update(|sink| Ok(import::syntax(sink, identities)))??;
        if policy == ScalarOptimization::Full {
            graph.update(|mut sink| sink.add("ExprImportInlining", RawValues(vec![])))?;
        }
        graph.parse_and_run_program(None, "(run-schedule (saturate scalar-source-flags) (saturate scalar-source-forms) scalar-source-select (saturate (seq (saturate scalar-request) (saturate scalar-build))))")?;
        for table in ["ExprImportComplete", "ExprTemplateComplete"] {
            Query(graph).for_function(table, |_, complete| {
                if graph.value_to_base::<bool>(complete) {
                    Ok(())
                } else {
                    Err(OptimizeError::Output(
                        "incomplete or cyclic scalar admission".into(),
                    ))
                }
            })?;
        }
        Ok::<_, OptimizeError>(())
    })?;
    graph.parse_and_run_program(None, "(run-schedule (saturate scalar-extent-sources) (saturate scalar-extents) (saturate (seq scalar-substitute scalar-arguments)))")?;
    timing::time("egglog scalar / fixed point", || graph.run_program(schedule))?;
    timing::time("egglog scalar / effect summaries", || {
        graph.parse_and_run_program(None, "(run-schedule (saturate scalar-effects))")
    })?;
    graph.parse_and_run_program(
        Some("scalar host preferences".into()),
        include_str!("scalar/host.egg"),
    )?;
    graph.clear_function("ExprContextActive")?;
    let selected = extract::select_roots(graph, policy)?;
    evaluation::prepare(graph)?;
    Ok(selected)
}
