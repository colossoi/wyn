//! Analyze immutable source facts, plan in egglog, then construct sidecar bodies.
use crate::egglog::{from_tlc::Converted, term, timing, OptimizeError};
use egglog_engine::EGraph;

pub(super) mod analysis;
mod build;
#[cfg(test)]
mod parity_tests;
mod plan;
#[cfg(test)]
mod tests;

/// Analyze once, plan on the persistent egglog graph, then materialize once.
/// No sidecar mutation or source reanalysis occurs inside the planning loop.
pub fn fuse(mut converted: Converted) -> Result<Converted, OptimizeError> {
    let _timing = timing::span("fusion");
    if converted.expression_program.is_some() || !converted.data.blocks.is_empty() {
        return Err(OptimizeError::Output(
            "fusion must precede expression insertion and scheduling".into(),
        ));
    }

    let mut sink = analysis::Egglog::new();
    timing::time("derive fusion facts", || {
        analysis::emit(&converted.data, &mut sink)
    })?;
    let mut program = term::parse("wyn-fusion-facts.egg", &sink.text)?;
    program.extend(term::parse("fusion.egg", include_str!("fusion.egg"))?);
    let mut graph = EGraph::default();
    timing::time("load fusion graph", || graph.run_program(program.clone()))?;
    let schedule = term::parse("fusion-schedule.egg", include_str!("schedule.egg"))?;
    timing::time("plan fusion", || graph.run_program(schedule.clone()))?;
    program.extend(schedule);
    let steps = timing::time("read fusion plan", || plan::read(&graph))?;
    timing::time("construct fused bodies", || {
        for step in steps {
            build::apply_step(&mut converted.data, step)?;
        }
        Ok::<_, OptimizeError>(())
    })?;
    converted.program = program;
    Ok(converted)
}
