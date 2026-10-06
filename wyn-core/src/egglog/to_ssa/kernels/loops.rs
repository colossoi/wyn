//! Emit scheduled loop entry and exit phases.
use super::super::{error, Body, OptimizeError};
use super::invocation;
use egglog_engine::Value;

pub(super) fn emit(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    phase_owner: Value,
    phase: &str,
    phase_width: u32,
) -> Result<(), OptimizeError> {
    let Some(source) = body.compiler.facts.source(phase_owner) else {
        return Err(error("loop source missing"));
    };
    let Some((header, _)) = body.compiler.facts.loops(source) else {
        return Err(error("loop regions missing"));
    };
    let Some(state) = body.compiler.facts.loop_state(header) else {
        return Err(error("loop state missing"));
    };
    let Some(resource) = body.compiler.facts.value_ref(state) else {
        return Err(error("loop carry storage missing"));
    };
    if phase == "loop_enter" {
        let output = body.resource(scope, resource, 2)?;
        let initial = body.compiler.facts.loop_initial(header)?;
        let initial = body.value(scope, initial)?;
        let n = body.length(initial.clone())?;
        let (start, step) = invocation(body, phase_width)?;
        body.counted(start, n, step, vec![], |body, index, _| {
            let value = body.index(initial, index.clone())?;
            body.store(output, index, value)?;
            Ok(vec![])
        })?;
    }
    Ok(())
}
