//! Analyze immutable source facts, plan in egglog, then construct sidecar bodies.
use crate::egglog::timing::{span, time};
use crate::egglog::{parse_program, Fused, Imported, OptimizeError, Program};
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
pub fn fuse(mut wyn_program: Program<Imported>) -> Result<Program<Fused>, OptimizeError> {
    let _timing = span("fusion");
    let mut facts = wyn_program.state.facts;
    facts.extend(parse_program("fusion.egg", include_str!("fusion.egg"))?);
    let mut graph = EGraph::default();
    time("load fusion graph", || graph.run_program(facts))?;
    let schedule = parse_program("fusion-schedule.egg", include_str!("schedule.egg"))?;
    time("plan fusion", || graph.run_program(schedule))?;
    let steps = time("read fusion plan", || plan::read(&graph))?;
    time("construct fused bodies", || {
        for step in steps {
            build::apply_step(&mut wyn_program.ir, step)?;
        }
        Ok::<_, OptimizeError>(())
    })?;
    Ok(Program {
        ir: wyn_program.ir,
        state: Fused,
    })
}
