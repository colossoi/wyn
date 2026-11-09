//! Materialize the offset invariant before allocating the next array level.
use super::super::{error, Body, OptimizeError};
use super::{coordinates, emit_traversal};
use crate::egglog::query::Query;
use crate::op::BinaryOperator::Add;
use crate::LookupMap;
use egglog_engine::Value;

pub(super) fn emit(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    shape: &[Value],
) -> Result<(), OptimizeError> {
    let [source, parents, offsets] = *shape else {
        return Err(error("invalid shape phase"));
    };
    let query = Query(&body.compiler.program.graph);
    let rank = query.required("ShapeRank", (parents,))?;
    let rank = body.compiler.facts.integer(rank);
    let owner = query.required("SourceValueOperation", (source,))?;
    let plan = query.required("SelectedPlan", (owner,))?;
    let traversal = query.required("ElementTraversal", (source, rank))?;
    let extent = query.required("ShapeExtent", (parents,))?;
    let Some(count) = query.constructor("SegmentCount", (offsets,))? else {
        return Err(error("segment count resource missing"));
    };
    let n = body.extent(scope, extent)?;
    let output = body.resource(scope, offsets, 2)?;
    let count = body.resource(scope, count, 2)?;
    let total = body.counted(
        Body::number(0),
        n.clone(),
        Body::number(1),
        vec![Body::number(0)],
        |body, index, state| {
            body.store(output.clone(), index.clone(), state[0].clone())?;
            let coordinates = coordinates::domain(body, scope, parents, index)?;
            let length = emit_traversal(
                body,
                scope,
                Some(plan),
                traversal,
                &coordinates,
                &mut LookupMap::default(),
                &|body, scope, leaf| leaf.length(body, scope),
            )?;
            Ok(vec![body.binary(Add, state[0].clone(), length)?])
        },
    )?;
    body.store(output, n, total[0].clone())?;
    body.store(count, Body::number(0), total[0].clone())
}
