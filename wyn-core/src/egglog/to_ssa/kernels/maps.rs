//! Lower the selected map traversal, retaining nested output storage.
use super::super::{error, Body, OptimizeError};
use super::{coordinates, invocation, write_arrays};
use crate::LookupMap;
use egglog_engine::Value;

pub(super) fn emit(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    owner: Value,
    extent: Value,
    width: u32,
    plan: Value,
    results: &[(String, i64, Value)],
) -> Result<(), OptimizeError> {
    let Some(rank) = body.compiler.facts.lookup("MapLaunchRank", (owner,)) else {
        return Err(error("map launch has no selected rank"));
    };
    let rank = body.compiler.facts.unsigned(rank, "map launch rank")?;
    let mut dimensions = Vec::new();
    if rank > 1 {
        for axis in 0..rank {
            let Some(extent) = body.compiler.facts.lookup("MapLaunchAxis", (owner, i64::from(axis))) else {
                return Err(error("map launch axis is missing"));
            };
            dimensions.push(body.extent(scope, extent)?);
        }
    }
    let n = body.extent(scope, extent)?;
    let (start, step) = invocation(body, width)?;
    body.counted(start, n, step, vec![], |body, index, _| {
        let mut cache = LookupMap::default();
        let indices = if dimensions.is_empty() {
            vec![index]
        } else {
            coordinates::decode(body, &dimensions, index)?
        };
        write_arrays(body, scope, owner, plan, results, &indices, &mut cache)?;
        Ok(vec![])
    })?;
    Ok(())
}
