//! Stream pointwise consumers over packed outputs on the current region graph.
use super::body::{ClauseId, ClauseKind};
use super::{
    barriers, domains, effects_conflict, fuse_region, pointwise_index, Body, Code, Effects, Query, Site,
};
use crate::egglog::graph::Graph;
use crate::egglog::{output_error, timing, OptimizeError};
use crate::memory_ir::{OperationId, RegionId, UseSite, ValueRef};
use crate::{LookupMap, LookupSet};
use egglog_engine::sort::Z;
use egglog_engine::{RawValues, Value};
use std::collections::VecDeque;

pub(in crate::egglog::fusion) fn compact(graph: &mut Graph) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust guarded compaction");
    let query = Query(graph);
    let Some(compacted) = query.constructor("Compacted", RawValues(vec![]))? else {
        return Ok(false);
    };
    let mut producers = LookupMap::new();
    {
        let structure = graph.structure().read();
        for (&op, results) in &structure.parallels {
            for (&output, binding) in
                structure.outputs.get(&op).into_iter().flatten().zip(&results.bindings)
            {
                let Some(clause) = binding.clause.and_then(|slot| usize::try_from(slot).ok()) else {
                    continue;
                };
                if results.kinds.get(clause).copied().flatten() == Some(compacted) {
                    producers.insert(output, op);
                }
            }
        }
    }
    if producers.is_empty() {
        return Ok(false);
    }
    let mut lengths = LookupMap::<Value, Vec<Value>>::new();
    query.for_each("CountLength", |row| {
        if let Some(name) = query.constructor("ValueNameRef", row[1])? {
            lengths.entry(row[0]).or_default().push(name);
        }
        Ok(())
    })?;
    if !lengths.values().flatten().any(|name| producers.contains_key(name)) {
        return Ok(false);
    }
    let domains = domains(query)?;
    let mut regions = Vec::new();
    for (&region, operations) in &graph.region_orders {
        let mut preceding = LookupSet::new();
        for &operation in operations {
            let Some(domain) = domains.get(&operation) else {
                continue;
            };
            if let [(extent, _)] = domain.extents[..] {
                if lengths.get(&extent).is_some_and(|sources| {
                    sources.iter().any(|name| producers.get(name).is_some_and(|op| preceding.contains(op)))
                }) {
                    regions.push(region);
                    break;
                }
            }
            preceding.insert(operation);
        }
    }
    if regions.is_empty() {
        return Ok(false);
    }
    let mut body = Body::read(query, domains, barriers(query)?)?;
    let mut pending: VecDeque<_> = regions.into_iter().map(|region| body.regions[&region]).collect();
    let mut queued: LookupSet<_> = pending.iter().copied().collect();
    let mut changed = false;
    while let Some(region) = pending.pop_front() {
        queued.remove(&region);
        let Some(block) = body.graph.ir.regions.get(region) else {
            continue;
        };
        for second in block.operations.clone() {
            let Some(domain) = body.domains.get(second) else {
                continue;
            };
            let [(extent, _)] = domain.extents[..] else {
                continue;
            };
            let Some(sources) = lengths.get(&extent) else {
                continue;
            };
            let mut clauses = body.clauses_of(second);
            let Some(consumer) = clauses.next() else { continue };
            if clauses.next().is_some() || body.clause(consumer).kind == ClauseKind::Compacted {
                continue;
            }
            drop(clauses);
            for &source in sources {
                let Some(&ValueRef::Result(first, slot)) = body.graph.refs.get(&source) else {
                    continue;
                };
                if !follows(Query(graph), &body, first, second, region)? {
                    continue;
                }
                let Some((producer, _)) = body.output_value(first, slot) else {
                    continue;
                };
                let filter = body.clause(producer);
                if filter.kind != ClauseKind::Compacted {
                    continue;
                }
                let Some(predicate) = filter.predicate else {
                    return Err(output_error("compacted clause has no predicate"));
                };
                let Some(loads) = legal(graph, &mut body, first, second, producer, consumer)? else {
                    continue;
                };
                let predicate = body.resolve(predicate);
                if body.clause(consumer).kind != ClauseKind::Collected {
                    body.guard(graph, second, consumer, predicate)?;
                }
                body.compact_result(second, consumer, predicate);
                body.splice(Query(graph), first, second, &loads)?;
                // Removing a consumer can unblock ordinary same-domain producers.
                // Fuse them before a second guard turns their callbacks into barriers.
                fuse_region(graph, &mut body, region)?;
                changed = true;
                break;
            }
        }
        for dirty in body.take_dirty() {
            if queued.insert(dirty) {
                pending.push_back(dirty);
            }
        }
    }
    if changed {
        body.publish(graph)?;
    }
    Ok(changed)
}

fn follows(
    query: Query<'_>,
    body: &Body,
    first: OperationId,
    second: OperationId,
    region: RegionId,
) -> Result<bool, OptimizeError> {
    let Some(domain) = body.domains.get(first) else {
        return Ok(false);
    };
    let (source, before) = body.positions[first];
    let after = body.positions[second].1;
    if source != region
        || before >= after
        || domain.extents.len() != 1
        || domain.execution != body.domains[second].execution
    {
        return Ok(false);
    }
    let operations = &body.graph.ir.regions[region].operations;
    for &middle in &operations[operations.partition_point(|&op| body.positions[op].1 <= before)
        ..operations.partition_point(|&op| body.positions[op].1 < after)]
    {
        let key = body.graph.operations[middle];
        if body.barriers.contains(&middle)
            || !query.contains("DiscardableOperation", key)?
            || !query.contains("RepeatableOperation", key)?
            || !pure(query, key)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn legal(
    graph: &mut Graph,
    body: &mut Body,
    first: OperationId,
    second: OperationId,
    producer: ClauseId,
    consumer: ClauseId,
) -> Result<Option<Vec<(OperationId, Value)>>, OptimizeError> {
    let query = Query(graph);
    let lane = body.lane(second)?;
    let children = body.subregions(second);
    let [axis] = body.domains[second].axes[..] else {
        return Ok(None);
    };
    let axis = body.resolve(axis);
    let domain = query.required("OperationDomain", body.graph.operations[second])?;
    let ordinal = query
        .lookup("DomainOrdinal", domain)?
        .map(|name| query.constructor("ValueNameRef", name))
        .transpose()?
        .flatten();
    let ordinal = ordinal.map(|name| body.resolve(name));
    let mut loads = Vec::new();
    let mut index_ops = LookupSet::new();
    for (site, name) in body.tree_uses(second) {
        let Some(&value) = body.graph.refs.get(&name) else {
            return Ok(None);
        };
        if let ValueRef::Result(op, slot) = value {
            if op == first {
                let Site::Operand(UseSite::Input(load, 0)) = site else {
                    return Ok(None);
                };
                let Some((clause, forwarded)) = body.output_value(first, slot) else {
                    return Ok(None);
                };
                let inputs = &body.graph.ir.operations[load].inputs;
                let result = body.clause(clause);
                // A preceding map may have forwarded the extent from its input.
                // Co-packed clauses in the same traversal have the same element order.
                let same_stream = clause == producer
                    || (result.kind == ClauseKind::Compacted
                        && result.predicate == body.clause(producer).predicate);
                if !same_stream
                    || !body.is_code(load, &Code::Index)
                    || !body.inside(body.positions[load].0, lane)
                {
                    return Ok(None);
                }
                let [_, index] = inputs[..] else {
                    return Ok(None);
                };
                let Some(proof) = pointwise_index(query, body, index, axis)? else {
                    return Ok(None);
                };
                index_ops.extend(proof);
                loads.push((load, forwarded));
                continue;
            }
        }
        if name != axis
            && ordinal != Some(name)
            && site != Site::Attribute(second)
            && !children.iter().any(|&region| body.inside(body.definition_region(value), region))
            && !body.available_before(value, first)
        {
            return Ok(None);
        }
    }
    let pointwise: LookupSet<_> = loads.iter().map(|&(load, _)| load).collect();
    let indices = std::iter::once(axis)
        .chain(ordinal)
        .chain(index_ops.iter().map(|&op| body.graph.names[&ValueRef::Result(op, 0)]));
    if loads.is_empty()
        || indices.flat_map(|name| body.users(name)).any(|site| {
            !matches!(site, Site::Operand(UseSite::Input(op, slot))
            if index_ops.contains(&op) || (slot == 1 && pointwise.contains(&op)))
        })
    {
        return Ok(None);
    }
    match body.clause(consumer).kind {
        ClauseKind::Collected if !safe_region(query, body, lane, &pointwise)? => return Ok(None),
        ClauseKind::Folded if !maskable(query, body, second, consumer)? => return Ok(None),
        _ => {}
    }
    let empty = Effects::default();
    let a = body.effects.get(first).unwrap_or(&empty);
    let b = body.effects.get(second).unwrap_or(&empty);
    if (body.unknown_aliases && (!a.writes.is_empty() || !b.writes.is_empty()))
        || effects_conflict(graph, a, b, false)?
    {
        return Ok(None);
    }
    Ok(Some(loads))
}

fn pure(query: Query<'_>, key: Value) -> Result<bool, OptimizeError> {
    Ok(query.lookup("PureOperation", key)?.is_some_and(|flag| query.0.value_to_base::<bool>(flag)))
}

fn safe_region(
    query: Query<'_>,
    body: &Body,
    region: RegionId,
    loads: &LookupSet<OperationId>,
) -> Result<bool, OptimizeError> {
    let mut pending = vec![region];
    while let Some(region) = pending.pop() {
        for &op in &body.graph.ir.regions[region].operations {
            if loads.contains(&op) || pure(query, body.graph.operations[op])? {
                continue;
            }
            if body.is_code(op, &Code::Branch) || body.is_code(op, &Code::CountedLoop) {
                pending.extend(body.subregions(op));
            } else {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn maskable(
    query: Query<'_>,
    body: &Body,
    operation: OperationId,
    clause: ClauseId,
) -> Result<bool, OptimizeError> {
    let result = body.clause(clause);
    if result.values.len() != result.neutral.len() {
        return Ok(false);
    }
    let Some(reducer) = body.reducer(operation, clause) else {
        return Ok(false);
    };
    for (slot, &name) in result.values.iter().enumerate() {
        let Some(state) = body.graph.ir.regions[reducer].parameters.get(slot) else {
            return Ok(false);
        };
        let Some(value) = query.value_name(name)? else {
            return Ok(false);
        };
        let ty = query.required("ValueType", value)?;
        if ty != state.ty {
            return Ok(false);
        }
        let Some(element) = query.enode("ScalarInfo", ty)? else {
            return Ok(false);
        };
        if !query.lookup("ElementBytes", element[0])?.is_some_and(|bytes| {
            u64::try_from(&*query.0.value_to_base::<Z>(bytes)).is_ok_and(|bytes| bytes > 0)
        }) {
            return Ok(false);
        }
    }
    Ok(true)
}
