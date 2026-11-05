//! Transfers and predicates for the shared dispatch-loop representation.
use super::super::{error, Body, OptimizeError, Typed};
use super::invocation;
use crate::egglog::query::Query;
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
            .record("SourceLoopForm", "WhileCondition", (source,))
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
                element(body, value_scope, value, index.clone())?
            } else {
                body.value(value_scope, value)?
            };
            body.store(output.clone(), index, item)?;
            Ok(vec![])
        })?;
    }
    Ok(())
}

// Transfer an array choice elementwise. A carried branch must not first copy
// its entire fixed array into each invocation's private storage.
fn element(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    source: Value,
    index: Typed,
) -> Result<Typed, OptimizeError> {
    if body.compiler.facts.contains("ArrayStreamProjection", (source,)) {
        let (parent, field) = body
            .compiler
            .facts
            .projection(source)
            .ok_or_else(|| error("array stream projection has no parent"))?;
        let value = element(body, scope, parent, index)?;
        return body.field(value, field);
    }
    if body.compiler.facts.access(source).is_some() {
        let value = body.value(scope, source)?;
        return body.index(value, index);
    }
    if let Some(actual) = body.compiler.facts.alias(source) {
        return element(body, scope, actual, index);
    }
    if let Some((parent, field)) = body.compiler.facts.projection(source) {
        if let Some(actual) = body.compiler.facts.source_field(parent, field)? {
            if actual != source {
                return element(body, scope, actual, index);
            }
        }
    }
    if let Some((yes, no)) = body.compiler.facts.branches(source) {
        let condition = body
            .compiler
            .facts
            .branch_condition(source)
            .ok_or_else(|| error("array choice condition missing"))?;
        let condition = body.value(scope, condition)?;
        let a = body.compiler.facts.result(yes).ok_or_else(|| error("array choice result missing"))?;
        let b = body.compiler.facts.result(no).ok_or_else(|| error("array choice result missing"))?;
        return body.branch(
            scope,
            condition,
            |body| element(body, yes, a, index.clone()),
            |body| element(body, no, b, index.clone()),
            Some((yes, no)),
        );
    }
    let value = body.value(scope, source)?;
    body.index(value, index)
}
