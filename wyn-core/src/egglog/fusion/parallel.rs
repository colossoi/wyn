//! Keep parallel candidate discovery, splices and cleanup on the same Rust graph.
use super::super::dataflow::ir::Code;
use super::super::payload::Payload;
use super::super::query::Query;
use super::super::{analysis, timing, OptimizeError};
use crate::memory_ir::{OperationId, RegionId, UseSite, ValueRef};
use crate::op::BinaryOperator;
use crate::{LookupMap, LookupSet};
use egglog_engine::{EGraph, RawValues, Value, Write};
use std::collections::VecDeque;

mod accumulator;
mod body;
mod branch;
mod compact;
mod hoist;
mod motion;
mod placement;

pub(super) use accumulator::run as accumulators;
use body::{Body, Effects, Site};
pub(super) use branch::run as branches;
pub(super) use compact::compact;
pub(super) use hoist::hoist;
use motion::Motion;

#[derive(Clone)]
struct Domain {
    execution: Value,
    extents: Vec<(Value, Option<Value>)>,
    axes: Vec<Value>,
    group: usize,
}

fn domains(query: Query<'_>) -> Result<LookupMap<Value, Domain>, OptimizeError> {
    let mut bounds = LookupMap::new();
    query.for_each("ConstantBound", |row| {
        bounds.insert(row[0], row[1]);
        Ok(())
    })?;
    let mut domains = LookupMap::new();
    let mut groups = LookupMap::new();
    query.for_each("ParallelOperation", |row| {
        let op = row[0];
        if let Some(execution) = query.lookup("OperationExecution", op)? {
            let domain = query.required("OperationDomain", op)?;
            let rank = query.0.value_to_base(query.required("DomainRank", domain)?);
            let extents = query
                .indexed("DomainExtent", &[domain], rank)?
                .into_iter()
                .map(|extent| (extent, bounds.get(&extent).copied()))
                .collect::<Vec<_>>();
            let key = extents
                .iter()
                .map(|&(extent, bound)| (bound.is_some(), bound.unwrap_or(extent)))
                .collect::<Vec<_>>();
            let next_group = groups.len();
            let group = *groups.entry((execution, key)).or_insert(next_group);
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
                    group,
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

fn compatible_pair(body: &Body, region: RegionId, first: OperationId, second: OperationId) -> bool {
    let (Some(first_domain), Some(second_domain)) = (body.domains.get(first), body.domains.get(second))
    else {
        return false;
    };
    let (a, start) = body.positions[first];
    let (b, end) = body.positions[second];
    if a != region || b != region || start >= end || first_domain.group != second_domain.group {
        return false;
    }
    let operations = &body.graph.ir.regions[region].operations;
    let middle = &operations[operations.partition_point(|&op| body.positions[op].1 <= start)
        ..operations.partition_point(|&op| body.positions[op].1 < end)];
    middle.iter().all(|op| !body.barriers.contains(op))
}

fn parallel_users(
    body: &Body,
    producer: OperationId,
    region: RegionId,
) -> Result<LookupSet<OperationId>, OptimizeError> {
    let mut users = LookupSet::new();
    for slot in 0..body.graph.ir.operations[producer].result_types.len() {
        let name = body.graph.names[&ValueRef::Result(producer, slot)];
        for site in body.users(name) {
            let demand = body.demand_region(site)?;
            if let Some((consumer, _)) = placement::child_on_path(body, demand, region) {
                if body.domains.contains_key(consumer) {
                    users.insert(consumer);
                }
            }
        }
    }
    Ok(users)
}

type Pair = (OperationId, OperationId);

#[derive(Default)]
struct Vertical {
    pending: VecDeque<Pair>,
    queued: LookupSet<Pair>,
}

impl Vertical {
    fn enqueue(&mut self, body: &Body, mut pairs: Vec<Pair>) -> usize {
        pairs.sort_unstable_by_key(|&(first, second)| (body.positions[second].1, body.positions[first].1));
        let before = self.pending.len();
        self.pending.extend(pairs.into_iter().filter(|&pair| self.queued.insert(pair)));
        self.pending.len() - before
    }

    fn seed(&mut self, body: &Body, region: RegionId) -> Result<usize, OptimizeError> {
        let mut pairs = Vec::new();
        for &producer in &body.graph.ir.regions[region].operations {
            if body.domains.contains_key(producer) {
                pairs.extend(
                    parallel_users(body, producer, region)?
                        .into_iter()
                        .map(|consumer| (producer, consumer)),
                );
            }
        }
        Ok(self.enqueue(body, pairs))
    }

    fn reconsider(&mut self, body: &Body, region: RegionId, consumers: LookupSet<OperationId>) -> usize {
        let mut pairs = Vec::new();
        for consumer in consumers {
            if !body.domains.contains_key(consumer) {
                continue;
            }
            // A consumer can now depend on a result defined earlier in this region.
            for (_, name) in body.tree_uses(consumer) {
                if let Some(&ValueRef::Result(producer, _)) = body.graph.refs.get(&name) {
                    if body.domains.contains_key(producer) && body.positions[producer].0 == region {
                        pairs.push((producer, consumer));
                    }
                }
            }
        }
        self.enqueue(body, pairs)
    }

    fn pop(&mut self) -> Option<Pair> {
        let pair = self.pending.pop_front()?;
        self.queued.remove(&pair);
        Some(pair)
    }
}

fn horizontal_pair(
    graph: &mut EGraph,
    body: &mut Body,
    region: RegionId,
    examined: &mut usize,
) -> Result<Option<Pair>, OptimizeError> {
    let mut preceding = LookupMap::<usize, Vec<OperationId>>::new();
    for next in body.graph.ir.regions[region].operations.clone() {
        let group = body.domains.get(next).map(|domain| domain.group);
        if let Some(earlier) = group.and_then(|group| preceding.get(&group)) {
            for &first in earlier {
                *examined += 1;
                if legal(graph, body, first, next)?.is_some_and(|loads| loads.is_empty()) {
                    return Ok(Some((first, next)));
                }
            }
        }
        if body.barriers.contains(&next) {
            preceding.clear();
        }
        if let Some(group) = group {
            preceding.entry(group).or_default().push(next);
        }
    }
    Ok(None)
}

fn fuse_region(graph: &mut EGraph, body: &mut Body, region: RegionId) -> Result<bool, OptimizeError> {
    let mut vertical = Vertical::default();
    let mut generated = vertical.seed(body, region)?;
    let mut vertical_checks = 0;
    let mut horizontal_checks = 0;
    let mut changed = false;
    loop {
        let mut selected = None;
        while let Some((first, second)) = vertical.pop() {
            if !compatible_pair(body, region, first, second) {
                continue;
            }
            vertical_checks += 1;
            if let Some(loads) = legal(graph, body, first, second)? {
                if !loads.is_empty() {
                    selected = Some((first, second, loads));
                    break;
                }
            }
        }
        let (first, second, loads) = if let Some(selected) = selected {
            selected
        } else if let Some((first, second)) = horizontal_pair(graph, body, region, &mut horizontal_checks)?
        {
            (first, second, Vec::new())
        } else {
            break;
        };
        let barrier_moved = body.barriers.contains(&second);
        let consumers = parallel_users(body, second, region)?;
        body.splice(Query(graph), first, second, &loads)?;
        generated += if barrier_moved {
            vertical.seed(body, region)?
        } else {
            vertical.reconsider(body, region, consumers)
        };
        changed = true;
    }
    if std::env::var_os("WYN_FUSION_COUNTS").is_some() {
        eprintln!("Rust fusion candidates\t{generated}\t{vertical_checks}\t{horizontal_checks}");
    }
    Ok(changed)
}

pub(super) fn run(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust parallel fusion");
    let query = Query(graph);
    let domains = domains(query)?;
    if domains.len() < 2 {
        return Ok(false);
    }
    let barriers = barriers(query)?;
    let mut regions = Vec::new();
    query.for_function("RegionOperations", |key, order| {
        let mut seen = LookupSet::new();
        for op in query.vector(order)? {
            let group = domains.get(&op).map(|domain| domain.group);
            if group.is_some_and(|group| seen.contains(&group)) {
                regions.push(key[0]);
                break;
            }
            if barriers.contains(&op) {
                seen.clear();
            }
            seen.extend(group);
        }
        Ok(())
    })?;
    if regions.is_empty() {
        return Ok(false);
    }
    let mut body = Body::read(query, domains, barriers)?;
    let mut pending: VecDeque<_> = regions.into_iter().map(|region| body.regions[&region]).collect();
    let mut queued: LookupSet<_> = pending.iter().copied().collect();
    let mut changed = false;
    while let Some(region) = pending.pop_front() {
        queued.remove(&region);
        if body.graph.ir.regions.contains_key(region) && fuse_region(graph, &mut body, region)? {
            for dirty in body.take_dirty() {
                if dirty != region && queued.insert(dirty) {
                    pending.push_back(dirty);
                }
            }
            changed = true;
        }
    }
    if changed {
        let _handoff = timing::span("parallel fusion handoff");
        body.simplify(Query(graph))?;
        if accumulator::join(graph, &mut body)? {
            body.simplify(Query(graph))?;
        }
        body.publish(graph)?;
    }
    Ok(changed)
}

pub(super) fn move_siblings(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust sibling motion");
    let query = Query(graph);
    let motion = Motion::read(query)?;
    let mut regions = Vec::new();
    query.for_function("RegionOperations", |keys, ops| {
        let operations = query.vector(ops)?;
        for (before, index) in motion.candidates(&operations, |op| op) {
            if !analysis::uses_output_of(query, operations[index], operations[before])? {
                regions.push(keys[0]);
                break;
            }
        }
        Ok(())
    })?;
    if regions.is_empty() {
        return Ok(false);
    }
    let mut body = Body::read(query, LookupMap::new(), LookupSet::new())?;
    let mut changed = false;
    for region in regions {
        let id = body.regions[&region];
        while motion.step(graph, &mut body, id)? {
            changed = true;
        }
    }
    if changed {
        body.publish_sibling_order(graph)?;
    }
    Ok(changed)
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
            if pointwise_index(Query(graph), body, index, axis)?.is_none() {
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
    let no_effects = Effects::default();
    let a = body.effects.get(first).unwrap_or(&no_effects);
    let b = body.effects.get(second).unwrap_or(&no_effects);
    if body.unknown_aliases && (!a.writes.is_empty() || !b.writes.is_empty()) {
        return Ok(None);
    }
    if !loads.is_empty()
        && body.contributes(second)
        && a.writes.is_empty()
        && !body.external.is_some_and(|external| b.writes.contains(&external))
        && b.writes.is_disjoint(&a.reads)
    {
        return Ok(Some(loads));
    }
    Ok((!effects_conflict(graph, a, b, true)?).then_some(loads))
}

fn effects_conflict(
    graph: &mut EGraph,
    a: &Effects,
    b: &Effects,
    compare_traps: bool,
) -> Result<bool, OptimizeError> {
    if a.writes.is_empty() && b.writes.is_empty() && !(compare_traps && a.traps && b.traps) {
        return Ok(false);
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
            for &root in &summary.reads {
                sink.add("ParallelRead", (side, root))?;
            }
            for &root in &summary.writes {
                sink.add("ParallelWrite", (side, root))?;
            }
            if compare_traps && summary.traps {
                sink.add("ParallelTrap", side)?;
            }
        }
        Ok(())
    })?;
    graph.parse_and_run_program(None, "(run-schedule (run parallel-effects))")?;
    Query(graph).contains("ParallelEffectConflict", RawValues(vec![]))
}

fn pointwise_index(
    query: Query<'_>,
    body: &Body,
    index: ValueRef,
    axis: Value,
) -> Result<Option<Vec<OperationId>>, OptimizeError> {
    let name = body.graph.names[&index];
    if name == axis {
        return Ok(Some(Vec::new()));
    }
    let ValueRef::Result(sub, 0) = index else {
        return Ok(None);
    };
    let [ValueRef::Result(add, 0), delta] = &body.graph.ir.operations[sub].inputs[..] else {
        return Ok(None);
    };
    if !body.is_code(sub, &Code::Scalar(Payload::Binary(BinaryOperator::Subtract)))
        || !body.is_code(*add, &Code::Scalar(Payload::Binary(BinaryOperator::Add)))
    {
        return Ok(None);
    }
    let [a, b] = &body.graph.ir.operations[*add].inputs[..] else {
        return Ok(None);
    };
    let Some(width) = query.lookup("ValuePrimitive", name)? else {
        return Ok(None);
    };
    if !query.contains("IntegerScalar", width)? {
        return Ok(None);
    }
    let sum = body.graph.names[&ValueRef::Result(*add, 0)];
    let (a, b, delta) = (body.graph.names[a], body.graph.names[b], body.graph.names[delta]);
    if (a, b) != (axis, delta) && (a, b) != (delta, axis) {
        return Ok(None);
    }
    for operand in [axis, a, b, delta, sum] {
        if query.lookup("ValuePrimitive", operand)? != Some(width) {
            return Ok(None);
        }
    }
    // Addition and subtraction cancel in one wrapping integer type.
    Ok(Some(vec![*add, sub]))
}
