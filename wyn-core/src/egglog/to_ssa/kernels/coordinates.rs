//! Row-major domains and input indexing; the last axis varies fastest.

use super::super::{error, Body, OptimizeError, Typed};
use crate::egglog::query::Query;
use crate::op::BinaryOperator::{Add, Divide, Less, LessEqual, Remainder, Subtract};
use egglog_engine::Value;

/// Invert the selected logical shape. Upper-bound search skips empty segments.
pub(super) fn domain(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    shape: Value,
    index: Typed,
) -> Result<Vec<Typed>, OptimizeError> {
    if body.compiler.facts.enode("OneAxis", shape).is_some() {
        return Ok(vec![index]);
    }
    if let Some(fields) = body.compiler.facts.enode("AppendAxis", shape) {
        let length = body.extent(scope, fields[1])?;
        let parent = body.binary(Divide, index.clone(), length.clone())?;
        let child = body.binary(Remainder, index, length)?;
        let mut coordinates = domain(body, scope, fields[0], parent)?;
        coordinates.push(child);
        return Ok(coordinates);
    }
    let Some(fields) = body.compiler.facts.enode("Segments", shape) else {
        return Err(error("unknown iteration shape"));
    };
    let extent = Query(&body.compiler.program.graph).required("ShapeExtent", (fields[0],))?;
    let parents = body.extent(scope, extent)?;
    let parents = body.cast(parents, &Body::number(0).ty)?;
    let offsets = body.resource(scope, fields[1], 1)?;
    let bounds = body.while_loop(
        vec![Body::number(0), parents],
        |body, state| body.binary(Less, state[0].clone(), state[1].clone()),
        |body, state| {
            let sum = body.binary(Add, state[0].clone(), state[1].clone())?;
            let middle = body.binary(Divide, sum, Body::number(2))?;
            let offset = body.index(offsets.clone(), middle.clone())?;
            let before = body.binary(LessEqual, offset, index.clone())?;
            let next = body.binary(Add, middle.clone(), Body::number(1))?;
            Ok(vec![
                body.select(before.clone(), next, state[0].clone())?,
                body.select(before, state[1].clone(), middle)?,
            ])
        },
    )?;
    let parent = body.binary(Subtract, bounds[0].clone(), Body::number(1))?;
    let offset = body.index(offsets, parent.clone())?;
    let child = body.binary(Subtract, index, offset)?;
    let mut coordinates = domain(body, scope, fields[0], parent)?;
    coordinates.push(child);
    Ok(coordinates)
}
