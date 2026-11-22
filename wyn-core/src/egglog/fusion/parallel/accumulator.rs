//! Join adjacent accumulator scopes and place producers where their results are used.
use super::placement::{child_on_path, demands, movable_group, pinned_operations, read_only};
use super::{barriers, domains, effects_conflict, fuse_region, Body, Code, Effects, Query};
use crate::egglog::{analysis::count, timing, OptimizeError};
use crate::memory_ir::{RegionId, ValueRef};
use crate::{LookupSet, SortedSet};
use egglog_engine::{EGraph, Value};

pub(in crate::egglog::fusion) fn run(graph: &mut EGraph, sinking: bool) -> Result<bool, OptimizeError> {
    let _timing =
        timing::span(if sinking { "Rust accumulator producer sinking" } else { "Rust accumulator fusion" });
    let query = Query(graph);
    let mut accumulators = LookupSet::new();
    query.for_each("AccumulatorOperation", |row| {
        accumulators.insert(row[0]);
        Ok(())
    })?;
    if accumulators.is_empty() {
        return Ok(false);
    }
    let blocked = if sinking { pinned_operations(query)? } else { LookupSet::new() };
    let mut regions = Vec::new();
    query.for_function("RegionOperations", |key, order| {
        let operations = query.vector(order)?;
        let relevant = if sinking {
            let mut producer = false;
            let mut relevant = false;
            for operation in operations {
                if producer && accumulators.contains(&operation) {
                    relevant = true;
                    break;
                }
                if blocked.contains(&operation) {
                    producer = false;
                } else if !accumulators.contains(&operation) {
                    producer |= query.lookup("ControlRegion", (operation, 0_i64))?.is_some();
                }
            }
            relevant
        } else {
            operations.windows(2).any(|pair| pair.iter().all(|op| accumulators.contains(op)))
        };
        if relevant {
            regions.push(key[0]);
        }
        Ok(())
    })?;
    if regions.is_empty() {
        return Ok(false);
    }
    let mut groups = LookupSet::new();
    for op in accumulators {
        let region = query.required("ControlRegion", (op, 0_i64))?;
        let count_groups = count(query, "AccumulatorGroupCount", op)?;
        if count(query, "RegionParameterCount", region)? == count_groups
            && count(query, "RegionYieldCount", region)? == count_groups
        {
            groups.insert(op);
        }
    }
    let mut body = Body::read(query, domains(query)?, barriers(query)?)?;
    body.pinned.extend(blocked.iter().filter_map(|op| body.graph.operation_ids.get(op).copied()));
    let mut pending: SortedSet<_> = regions.iter().map(|region| body.regions[region]).collect();
    let mut changed = false;
    while let Some(region) = pending.pop_first() {
        let joined = join_one(graph, &mut body, region, &groups)?;
        let edited = if let Some(destination) = joined {
            Some(destination)
        } else if sinking {
            sink_one(Query(graph), &mut body, region)?
        } else {
            None
        };
        let Some(destination) = edited else { continue };
        changed = true;
        // Moving a read-only producer or joining scopes can put their traversals together.
        // Recover those fusions before another accumulator edit changes the order.
        for region in [destination, region] {
            fuse_region(graph, &mut body, region)?;
        }
        pending.extend(body.take_dirty());
    }
    if changed {
        body.publish(graph)?;
    }
    Ok(changed)
}

fn join_one(
    graph: &mut EGraph,
    body: &mut Body,
    region: RegionId,
    groups: &LookupSet<Value>,
) -> Result<Option<RegionId>, OptimizeError> {
    let mut selected = None;
    let empty = Effects::default();
    for pair in body.graph.ir.regions[region].operations.windows(2) {
        let [first, second] = [pair[0], pair[1]];
        if !groups.contains(&body.graph.operations[first])
            || !groups.contains(&body.graph.operations[second])
            || body.tree_uses(second).iter().any(|&(_, name)| {
                matches!(body.graph.refs.get(&name), Some(ValueRef::Result(op, _)) if *op == first)
            })
        {
            continue;
        }
        let a = body.effects.get(first).unwrap_or(&empty);
        let b = body.effects.get(second).unwrap_or(&empty);
        if (body.unknown_aliases && (!a.writes.is_empty() || !b.writes.is_empty()))
            || effects_conflict(graph, a, b, true)?
        {
            continue;
        }
        selected = Some((first, second));
        break;
    }
    selected.map(|(first, second)| body.join_accumulators(first, second)).transpose()
}

fn sink_one(
    query: Query<'_>,
    body: &mut Body,
    region: RegionId,
) -> Result<Option<RegionId>, OptimizeError> {
    let mut selected = None;
    for &producer in &body.graph.ir.regions[region].operations {
        if body.graph.ir.operations[producer].bodies.is_empty()
            || body.is_code(producer, &Code::WithAcc)
            || !read_only(body, producer)
        {
            continue;
        }
        let (members, demands) = demands(query, body, producer)?;
        let mut targets = demands.iter().filter_map(|&demand| child_on_path(body, demand, region));
        let Some((scope, destination)) = targets.next() else {
            continue;
        };
        if !body.is_code(scope, &Code::WithAcc)
            || body.lane(scope)? != destination
            || demands.iter().any(|&demand| !body.inside(demand, destination))
        {
            continue;
        }
        let Some(moved) = movable_group(body, producer, scope, &members) else {
            continue;
        };
        selected = Some((scope, destination, moved));
        break;
    }
    let Some((scope, destination, moved)) = selected else {
        return Ok(None);
    };
    body.sink_producers(scope, destination, &moved);
    Ok(Some(destination))
}
