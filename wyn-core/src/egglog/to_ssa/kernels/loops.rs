//! Transfers and predicates for the shared dispatch-loop representation.
use super::super::{error, Body, OptimizeError};
use super::{element, invocation};
use crate::egglog::query::Query;
use crate::LookupMap;
use egglog_engine::Value;

pub(super) fn emit(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operation: Value,
    phase: &str,
    width: u32,
) -> Result<(), OptimizeError> {
    if phase == "loop_exit" {
        return Ok(());
    }
    let source = body.compiler.facts.source(operation).ok_or_else(|| error("loop source missing"))?;
    let (header, iteration) =
        body.compiler.facts.loops(source).ok_or_else(|| error("loop regions missing"))?;
    if phase == "loop_test" {
        body.context =
            body.compiler.facts.context(header).ok_or_else(|| error("loop test context missing"))?;
        let condition = body
            .compiler
            .facts
            .loop_form(source)
            .and_then(|form| body.compiler.facts.enode("WhileCondition", form))
            .ok_or_else(|| error("loop predicate missing"))?[0];
        let value = body.value(header, condition)?;
        let resource = body
            .compiler
            .facts
            .constructor("Temporary", (operation, "loop_condition", 0i64))
            .ok_or_else(|| error("loop predicate allocation missing"))?;
        let output = body.resource(scope, resource, 2)?;
        return body.store(output, Body::number(0), value);
    }
    if phase == "loop_step" {
        body.context =
            body.compiler.facts.context(iteration).ok_or_else(|| error("loop step context missing"))?;
    }
    let Some(plan) = body.compiler.facts.group(operation) else {
        return Err(error("loop transfer has no selected body"));
    };
    let carries = body.compiler.facts.loop_carries(operation)?;
    let (start, stride) = invocation(body, width)?;
    for carry in carries {
        let resource = if phase == "loop_enter" {
            body.compiler.facts.value_ref(carry.state).ok_or_else(|| error("loop state storage missing"))?
        } else {
            Query(&body.compiler.program.graph).required("LoopNext", (operation, carry.slot))?
        };
        let output = body.resource(scope, resource, 2)?;
        let value = if phase == "loop_enter" { carry.initial } else { carry.next };
        let value_scope = if phase == "loop_enter" { scope } else { iteration };
        let length = if carry.array {
            let extent =
                Query(&body.compiler.program.graph).required("LoopCarryExtent", (operation, carry.slot))?;
            body.extent(scope, extent)?
        } else {
            Body::number(1)
        };
        body.counted(start.clone(), length, stride.clone(), vec![], |body, index, _| {
            let item = if carry.array {
                element(
                    body,
                    value_scope,
                    plan,
                    value,
                    std::slice::from_ref(&index),
                    &mut LookupMap::default(),
                )?
            } else {
                body.value(value_scope, value)?
            };
            body.store(output.clone(), index, item)?;
            Ok(vec![])
        })?;
    }
    Ok(())
}
