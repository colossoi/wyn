//! Find and order a producer and the pure preparations that depend on it.
use super::{Body, Query};
use crate::egglog::OptimizeError;
use crate::memory_ir::{OperationId, RegionId, ValueRef};
use crate::LookupSet;
use egglog_engine::Value;
use std::iter::once;

pub(super) fn pinned_operations(query: Query<'_>) -> Result<LookupSet<Value>, OptimizeError> {
    let mut pinned = LookupSet::new();
    for table in ["ConsumedToken", "InvocationDependent", "OperationWrite"] {
        query.for_each(table, |row| {
            pinned.insert(row[0]);
            Ok(())
        })?;
    }
    Ok(pinned)
}

pub(super) fn movable_group(
    body: &Body,
    producer: OperationId,
    scope: OperationId,
    members: &LookupSet<OperationId>,
) -> Option<Vec<OperationId>> {
    let region = body.positions[producer].0;
    let (_, start) = body.positions[producer];
    let (_, end) = body.positions[scope];
    if start >= end {
        return None;
    }
    let operations = &body.graph.ir.regions[region].operations;
    let range = &operations[operations.partition_point(|&op| body.positions[op].1 <= start)
        ..operations.partition_point(|&op| body.positions[op].1 < end)];
    let traps = members.iter().any(|&op| body.effects.get(op).is_some_and(|effect| effect.traps));
    if range.iter().any(|&op| {
        !members.contains(&op)
            && (!read_only(body, op) || (traps && body.effects.get(op).is_some_and(|effect| effect.traps)))
    }) {
        return None;
    }
    if members.iter().any(|&op| {
        let children = body.subregions(op);
        !read_only(body, op)
            || body.positions[op].1 >= end
            || !body.tree_uses(op).into_iter().all(|(_, name)| {
                let Some(&value) = body.graph.refs.get(&name) else {
                    return false;
                };
                if matches!(value, ValueRef::Result(op, _) if members.contains(&op)) {
                    return true;
                }
                body.available_before(value, scope)
                    || children.iter().any(|&child| body.inside(body.definition_region(value), child))
            })
    }) {
        return None;
    }
    Some(once(producer).chain(range.iter().copied().filter(|op| members.contains(op))).collect())
}

pub(super) fn read_only(body: &Body, operation: OperationId) -> bool {
    !body.pinned.contains(&operation)
        && body.effects.get(operation).is_none_or(|effect| effect.writes.is_empty())
}

pub(super) fn demands(
    query: Query<'_>,
    body: &Body,
    producer: OperationId,
) -> Result<(LookupSet<OperationId>, LookupSet<RegionId>), OptimizeError> {
    let position = body.positions[producer];
    let mut pending = vec![producer];
    let mut members = LookupSet::new();
    members.insert(producer);
    let mut uses = Vec::new();
    while let Some(operation) = pending.pop() {
        for slot in 0..body.graph.ir.operations[operation].result_types.len() {
            let name = body.graph.names[&ValueRef::Result(operation, slot)];
            for site in body.users(name) {
                if let Some(user) = body.site_operation(site) {
                    let (region, order) = body.positions[user];
                    if region == position.0
                        && order > position.1
                        && (members.contains(&user) || is_preparation(query, body.graph.operations[user])?)
                    {
                        if members.insert(user) {
                            pending.push(user);
                        }
                        continue;
                    }
                }
                uses.push(site);
            }
        }
    }
    let mut demands = LookupSet::new();
    for site in uses {
        if body.site_operation(site).is_none_or(|op| !members.contains(&op)) {
            demands.insert(body.demand_region(site)?);
        }
    }
    Ok((members, demands))
}

pub(super) fn child_on_path(
    body: &Body,
    mut region: RegionId,
    parent: RegionId,
) -> Option<(OperationId, RegionId)> {
    while let Some(owner) = body.parent(region) {
        let enclosing = body.positions[owner].0;
        if enclosing == parent {
            return Some((owner, region));
        }
        region = enclosing;
    }
    None
}

fn is_preparation(query: Query<'_>, operation: Value) -> Result<bool, OptimizeError> {
    Ok(query.contains("DiscardableOperation", operation)?
        && query.contains("RepeatableOperation", operation)?
        && query
            .lookup("PureOperation", operation)?
            .is_some_and(|flag| query.0.value_to_base::<bool>(flag)))
}
