//! Join adjacent accumulator scopes and place producers where their results are used.
use super::placement::{child_on_path, demands, movable_group, pinned_operations, read_only};
use super::{barriers, domains, effects_conflict, fuse_region, Body, Code, Effects, Query};
use crate::egglog::graph::Graph;
use crate::egglog::{timing, OptimizeError};
use crate::memory_ir::{OperationId, RegionId, ValueRef};
use crate::{LookupSet, SortedSet};
use egglog_engine::Value;

pub(in crate::egglog::fusion) fn run(graph: &mut Graph, sinking: bool) -> Result<bool, OptimizeError> {
    let _timing =
        timing::span(if sinking { "Rust accumulator producer sinking" } else { "Rust accumulator fusion" });
    let query = Query(graph);
    let mut accumulators = LookupSet::new();
    query.for_each("AccumulatorOperation", |row| {
        accumulators.insert(row[0]);
        Ok(())
    })?;
    let mut destinations = accumulators.clone();
    if sinking {
        destinations.extend(
            query.0.codes().filter_map(|(op, code)| (*code == Code::WithReservedSpan).then_some(op)),
        );
    }
    if destinations.is_empty() {
        return Ok(false);
    }
    let blocked = if sinking { pinned_operations(query)? } else { LookupSet::new() };
    let mut regions = Vec::new();
    for (&region, operations) in &graph.region_orders {
        let relevant = if sinking {
            let mut producer = false;
            let mut relevant = false;
            for &operation in operations {
                if producer && destinations.contains(&operation) {
                    relevant = true;
                    break;
                }
                if blocked.contains(&operation) {
                    producer = matches!(query.0.code(operation)?, Code::Parallel);
                } else if !destinations.contains(&operation) {
                    producer |= query.0.control_at(operation, 0_i64).is_some();
                }
            }
            relevant
        } else {
            operations.windows(2).any(|pair| pair.iter().all(|op| accumulators.contains(op)))
        };
        if relevant {
            regions.push(region);
        }
    }
    if regions.is_empty() {
        return Ok(false);
    }
    let mut groups = LookupSet::new();
    for op in accumulators {
        let region = query.0.control_region(op, 0_i64)?;
        let count_groups = query.0.accumulator_group_count(op);
        if query.0.parameter_count(region) == count_groups && query.0.yield_count(region) == count_groups {
            groups.insert(op);
        }
    }
    let mut body = Body::read(query, regions.iter().copied(), domains(query)?, barriers(query)?)?;
    body.pinned.extend(blocked.iter().filter_map(|op| body.graph.operation_ids.get(op).copied()));
    let pending = regions.iter().map(|region| body.regions[region]).collect();
    let changed = apply(graph, &mut body, pending, &groups, sinking)?;
    if changed {
        body.publish(graph)?;
    }
    Ok(changed)
}

pub(super) fn join(graph: &mut Graph, body: &mut Body) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust accumulator fusion");
    let is_scope = |op: OperationId| body.is_code(op, &Code::WithAcc);
    let pending: SortedSet<_> = body
        .graph
        .ir
        .regions
        .iter()
        .filter(|(_, block)| block.operations.windows(2).any(|pair| pair.iter().copied().all(is_scope)))
        .map(|(region, _)| region)
        .collect();
    if pending.is_empty() {
        return Ok(false);
    }

    let mut groups = LookupSet::new();
    let query = Query(graph);
    for op in body
        .graph
        .ir
        .regions
        .values()
        .flat_map(|block| block.operations.iter().copied())
        .filter(|&op| is_scope(op))
    {
        let key = body.graph.operations[op];
        let lane = &body.graph.ir.regions[body.lane(op)?];
        let count = query.0.accumulator_group_count(key) as usize;
        if lane.parameters.len() == count && lane.yields.len() == count {
            groups.insert(key);
        }
    }
    apply(graph, body, pending, &groups, false)
}

fn apply(
    graph: &mut Graph,
    body: &mut Body,
    mut pending: SortedSet<RegionId>,
    groups: &LookupSet<Value>,
    sinking: bool,
) -> Result<bool, OptimizeError> {
    let mut changed = false;
    while let Some(region) = pending.pop_first() {
        let joined = join_one(graph, body, region, groups)?;
        let edited = if let Some(destination) = joined {
            Some(destination)
        } else if sinking {
            sink_one(graph, body, region)?
        } else {
            None
        };
        let Some(destination) = edited else { continue };
        changed = true;
        // Moving a read-only producer or joining scopes can put their traversals together.
        // Recover those fusions before another accumulator edit changes the order.
        for region in [destination, region] {
            fuse_region(graph, body, region)?;
        }
        pending.extend(body.take_dirty());
    }
    Ok(changed)
}

fn join_one(
    graph: &mut Graph,
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
    graph: &mut Graph,
    body: &mut Body,
    region: RegionId,
) -> Result<Option<RegionId>, OptimizeError> {
    let mut selected = None;
    for producer in body.graph.ir.regions[region].operations.clone() {
        if body.graph.ir.operations[producer].bodies.is_empty()
            || body.is_code(producer, &Code::WithAcc)
            || (!read_only(body, producer) && !body.is_code(producer, &Code::Parallel))
        {
            continue;
        }
        let (members, demands) = demands(Query(graph), body, producer)?;
        let mut targets = demands.iter().filter_map(|&demand| child_on_path(body, demand, region));
        let Some((scope, destination)) = targets.next() else {
            continue;
        };
        let span = body.is_code(scope, &Code::WithReservedSpan);
        // Reservation only mutates a compiler-private cursor. Reads in the producer
        // cannot observe it, and the writer executes once even for a failed claim.
        if !(body.is_code(scope, &Code::WithAcc) || span && members.iter().all(|&op| read_only(body, op)))
            || body.lane(scope)? != destination
            || demands.iter().any(|&demand| !body.inside(demand, destination))
        {
            continue;
        }
        if span {
            let Some(&consumer) = body.graph.ir.regions[destination].operations.first() else {
                continue;
            };
            let (Some(a), Some(b)) = (body.domains.get(producer), body.domains.get(consumer)) else {
                continue;
            };
            if members.len() != 1
                || a.group != b.group
                || !super::legal_at(graph, body, producer, consumer, destination, 0)?
                    .is_some_and(|loads| !loads.is_empty())
            {
                continue;
            }
        }
        // The accumulator body runs exactly once. Moving the immediately preceding
        // traversal to its beginning preserves execution order, including effects.
        if members.len() == 1 && body.positions[producer].1 + 1 == body.positions[scope].1 {
            selected = Some((scope, destination, vec![producer]));
            break;
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
