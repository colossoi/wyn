//! Analyze immutable source facts, plan in egglog, then construct sidecar bodies.
use crate::egglog::timing::{span, time};
use crate::egglog::{parse_program, Fused, Imported, OptimizeError, Program};

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
    let _timing = span("egglog fusion");
    let mut graph = wyn_program.state.graph;
    let rules = time("egglog fusion / parse rules", || {
        parse_program("fusion.egg", include_str!("fusion.egg"))
    })?;
    time("egglog fusion / load rules", || graph.run_program(rules))?;
    let schedule = parse_program("fusion-schedule.egg", include_str!("schedule.egg"))?;
    time("egglog fusion / run schedule", || graph.run_program(schedule))?;
    let steps = time("egglog fusion / read plan", || plan::read(&graph))?;
    let _build = span("egglog fusion / build bodies");
    for step in steps {
        build::apply_step(&mut wyn_program.ir, step)?;
    }
    drop(_build);
    Ok(Program {
        ir: wyn_program.ir,
        state: Fused,
    })
}
