//! Keep parallel candidate discovery, splices and cleanup on the same Rust graph.
use super::super::dataflow::ir::Code;
use super::super::payload::Payload;
use super::super::query::Query;
use super::super::{timing, OptimizeError};
use crate::memory_ir::{OperationId, UseSite, ValueRef};
use crate::op::BinaryOperator;
use crate::{LookupMap, LookupSet, StableMap};
use egglog_engine::{EGraph, RawValues, Value, Write};
use std::collections::VecDeque;

mod body;

use body::{Body, Site};

#[derive(Clone)]
struct Domain {
    execution: Value,
    extents: Vec<(Value, Option<Value>)>,
    axes: Vec<Value>,
}

impl Domain {
    fn compatible(&self, other: &Self) -> bool {
        self.execution == other.execution
            && self.extents.len() == other.extents.len()
            && self
                .extents
                .iter()
                .zip(&other.extents)
                .all(|((a, n), (b, m))| a == b || (n.is_some() && n == m))
    }
}

fn domains(query: Query<'_>) -> Result<LookupMap<Value, Domain>, OptimizeError> {
    let mut bounds = LookupMap::new();
    query.for_each("ConstantBound", |row| {
        bounds.insert(row[0], row[1]);
        Ok(())
    })?;
    let mut domains = LookupMap::new();
    query.for_each("ParallelOperation", |row| {
        let op = row[0];
        if let Some(execution) = query.lookup("OperationExecution", op)? {
            let domain = query.required("OperationDomain", op)?;
            let rank = query.0.value_to_base(query.required("DomainRank", domain)?);
            let extents = query
                .indexed("DomainExtent", &[domain], rank)?
                .into_iter()
                .map(|extent| (extent, bounds.get(&extent).copied()))
                .collect();
            let mut axes = Vec::new();
            for axis in query.indexed("DomainAxis", &[domain], rank)? {
                if let Some(name) = query.constructor("ValueNameRef", axis)? {
                    axes.push(name);
                }
            }
            domains.insert(
                op,
                Domain {
                    execution,
                    extents,
                    axes,
                },
            );
        }
        Ok(())
    })?;
    Ok(domains)
}

fn barriers(query: Query<'_>) -> Result<LookupSet<Value>, OptimizeError> {
    let mut barriers = LookupSet::new();
    for table in ["PureOperation", "MovableOperation"] {
        query.for_function(table, |key, flag| {
            if !query.0.value_to_base::<bool>(flag) {
                barriers.insert(key[0]);
            }
            Ok(())
        })?;
    }
    Ok(barriers)
}

fn pairs<'a, K: Copy>(
    operations: &[K],
    domain: impl Fn(K) -> Option<&'a Domain>,
    barrier: impl Fn(K) -> bool,
) -> impl Iterator<Item = (K, K)> {
    let mut preceding = Vec::<(K, &Domain)>::new();
    let mut candidates = Vec::new();
    for &operation in operations {
        let domain = domain(operation);
        if let Some(domain) = domain {
            candidates.extend(
                preceding
                    .iter()
                    .filter_map(|(first, a)| a.compatible(domain).then_some((*first, operation))),
            );
        }
        if barrier(operation) {
            preceding.clear();
        }
        if let Some(domain) = domain {
            preceding.push((operation, domain));
        }
    }
    candidates.into_iter()
}

pub(super) fn run(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust parallel fusion");
    let query = Query(graph);
    let domains = domains(query)?;
    if domains.len() < 2 {
        return Ok(false);
    }
    let barriers = barriers(query)?;
    let mut candidates = StableMap::new();
    query.for_function("RegionOperations", |keys, ops| {
        let operations = query.vector(ops)?;
        let pairs: Vec<_> =
            pairs(&operations, |op| domains.get(&op), |op| barriers.contains(&op)).collect();
        if !pairs.is_empty() {
            candidates.insert(keys[0], pairs);
        }
        Ok(())
    })?;
    if candidates.is_empty() {
        return Ok(false);
    }
    let mut body = Body::read(query, &[], domains, barriers)?;
    let mut pending = VecDeque::new();
    let mut queued = LookupSet::new();
    for (region, pairs) in candidates {
        let region = body.regions[&region];
        pending.push_back((
            region,
            Some(
                pairs
                    .into_iter()
                    .map(|(a, b)| (body.graph.operation_ids[&a], body.graph.operation_ids[&b]))
                    .collect::<Vec<_>>(),
            ),
        ));
        queued.insert(region);
    }
    let mut revised = LookupSet::new();
    let mut changed = false;
    while let Some((region, initial)) = pending.pop_front() {
        queued.remove(&region);
        let Some(current) = body.graph.ir.regions.get(region) else {
            continue;
        };
        let candidates = if let Some(initial) = initial.filter(|_| !revised.contains(&region)) {
            initial
        } else {
            pairs(
                &current.operations,
                |op| body.domains.get(op),
                |op| body.barriers.contains(&op),
            )
            .collect()
        };
        for (first, second) in candidates {
            let Some(loads) = legal(graph, &mut body, first, second)? else {
                continue;
            };
            body.splice(Query(graph), first, second, &loads)?;
            for region in body.take_dirty() {
                revised.insert(region);
                if queued.insert(region) {
                    pending.push_back((region, None));
                }
            }
            changed = true;
            break;
        }
    }
    if changed {
        body.publish(graph)?;
    }
    Ok(changed)
}

pub(super) fn apply(
    graph: &mut EGraph,
    first: Value,
    second: Value,
    loads: &[(Value, Value)],
    new_regions: &[(Value, Vec<Value>)],
) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut body = Body::read(query, new_regions, domains(query)?, barriers(query)?)?;
    let first = body.graph.operation_ids[&first];
    let second = body.graph.operation_ids[&second];
    let loads = loads.iter().map(|&(op, value)| (body.graph.operation_ids[&op], value)).collect::<Vec<_>>();
    body.splice(query, first, second, &loads)?;
    body.publish(graph)
}

fn legal(
    graph: &mut EGraph,
    body: &mut Body,
    first: OperationId,
    second: OperationId,
) -> Result<Option<Vec<(OperationId, Value)>>, OptimizeError> {
    let first_lane = body.lane(first)?;
    let second_lane = body.lane(second)?;
    let domain = body.domains[second].clone();
    let region = body.positions[first].0;
    let second_regions = body.subregions(second);
    let mut loads = Vec::new();
    for (site, name) in body.tree_uses(second) {
        let ValueRef::Result(producer, slot) = body.graph.refs[&name] else {
            continue;
        };
        if body.positions[producer].0 != region || body.positions[producer].1 < body.positions[first].1 {
            continue;
        }
        if producer != first {
            return Ok(None);
        }
        let Site::Operand(UseSite::Input(load, 0)) = site else {
            return Ok(None);
        };
        if !body.is_code(load, &Code::Index) || !body.inside(body.positions[load].0, second_lane) {
            return Ok(None);
        }
        let Some(value) = body.forwarded_value(first, slot) else {
            return Ok(None);
        };
        let indices = &body.graph.ir.operations[load].inputs[1..];
        if indices.len() != domain.extents.len() || indices.len() != domain.axes.len() {
            return Ok(None);
        }
        for (&index, &axis) in indices.iter().zip(&domain.axes) {
            if !index_equal(graph, body, index, axis)? {
                return Ok(None);
            }
        }
        loads.push((load, value));
    }
    if loads.is_empty()
        && ((body.has_parallel_in(first_lane) && body.folded(second))
            || (second_regions.iter().any(|&child| body.has_parallel_in(child)) && body.folded(first)))
    {
        return Ok(None);
    }
    let a = body.effects.get(first).cloned().unwrap_or_default();
    let b = body.effects.get(second).cloned().unwrap_or_default();
    if body.unknown_aliases && (!a.writes.is_empty() || !b.writes.is_empty()) {
        return Ok(None);
    }
    if a.writes.is_empty() && b.writes.is_empty() && !(a.traps && b.traps) {
        return Ok(Some(loads));
    }
    if !loads.is_empty()
        && body.contributes(second)
        && a.writes.is_empty()
        && !body.external.is_some_and(|external| b.writes.contains(&external))
        && b.writes.is_disjoint(&a.reads)
    {
        return Ok(Some(loads));
    }
    for table in [
        "ParallelRead",
        "ParallelWrite",
        "ParallelTrap",
        "ParallelEffectConflict",
    ] {
        graph.clear_function(table)?;
    }
    graph.update(|mut sink| {
        for (side, summary) in [(0_i64, a), (1, b)] {
            for root in summary.reads {
                sink.add("ParallelRead", (side, root))?;
            }
            for root in summary.writes {
                sink.add("ParallelWrite", (side, root))?;
            }
            if summary.traps {
                sink.add("ParallelTrap", side)?;
            }
        }
        Ok(())
    })?;
    graph.parse_and_run_program(None, "(run-schedule (run parallel-effects))")?;
    Ok((!Query(graph).contains("ParallelEffectConflict", RawValues(vec![]))?).then_some(loads))
}

fn index_equal(
    graph: &mut EGraph,
    body: &Body,
    index: ValueRef,
    axis: Value,
) -> Result<bool, OptimizeError> {
    let name = body.graph.names[&index];
    if name == axis {
        return Ok(true);
    }
    let ValueRef::Result(sub, 0) = index else {
        return Ok(false);
    };
    let [ValueRef::Result(add, 0), delta] = &body.graph.ir.operations[sub].inputs[..] else {
        return Ok(false);
    };
    if !body.is_code(sub, &Code::Scalar(Payload::Binary(BinaryOperator::Subtract)))
        || !body.is_code(*add, &Code::Scalar(Payload::Binary(BinaryOperator::Add)))
    {
        return Ok(false);
    }
    let [a, b] = &body.graph.ir.operations[*add].inputs[..] else {
        return Ok(false);
    };
    let query = Query(graph);
    let Some(width) = query.lookup("ValuePrimitive", name)? else {
        return Ok(false);
    };
    if !query.contains("IntegerScalar", width)? {
        return Ok(false);
    }
    let sum = body.graph.names[&ValueRef::Result(*add, 0)];
    let (a, b, delta) = (body.graph.names[a], body.graph.names[b], body.graph.names[delta]);
    for operand in [axis, a, b, delta, sum] {
        if query.lookup("ValuePrimitive", operand)? != Some(width) {
            return Ok(false);
        }
    }
    graph.update(|mut sink| sink.add("ParallelCancellation", (name, axis, a, b, delta)))?;
    graph.parse_and_run_program(None, "(run-schedule (run parallel-index-equality))")?;
    Query(graph).contains("ParallelIndexEqual", (name, axis))
}
