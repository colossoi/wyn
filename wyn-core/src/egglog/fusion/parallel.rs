//! Keep parallel candidate discovery, splices and cleanup on the same Rust graph.
use super::super::dataflow::ir::Code;
use super::super::query::Query;
use super::super::{analysis, timing, OptimizeError};
use crate::egglog::graph::{Formula, Graph, Origin};
use crate::memory_ir::{OperationId, RegionId, UseSite, ValueRef};
use crate::{LookupMap, LookupSet};
use egglog_engine::{RawValues, Value, Write};
use std::collections::VecDeque;

mod accumulator;
mod body;
mod branch;
mod compact;
mod hoist;
mod motion;
mod placement;

pub(super) use accumulator::run as accumulators;
use body::{Body, Site};
pub(super) use branch::run as branches;
pub(super) use compact::compact;
pub(super) use hoist::hoist;
pub(super) use motion::run as move_siblings;

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
            let domain = query.0.operation_domain(op)?;
            let extents = query
                .0
                .domain_extents(domain)?
                .into_iter()
                .map(|extent| (extent, bounds.get(&extent).copied()))
                .collect::<Vec<_>>();
            let mut key = Vec::with_capacity(extents.len());
            for &(extent, bound) in &extents {
                let formula = extent_formula(query, extent)?;
                let bound = bound.or_else(|| bounds.get(&formula).copied());
                key.push((bound.is_some(), bound.unwrap_or(formula)));
            }
            let next_group = groups.len();
            let group = *groups.entry((execution, key)).or_insert(next_group);
            let mut axes = Vec::new();
            for axis in query.0.domain_axes(domain)? {
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

fn extent_formula(query: Query<'_>, mut extent: Value) -> Result<Value, OptimizeError> {
    while let Some(value) = query.enode("IndexValue", extent)? {
        let reference = query.constructor("ValueNameRef", value[0])?;
        let Some(Origin::Result(op, _)) = reference.and_then(|name| query.0.origin(name)) else {
            break;
        };
        if query.0.code(op)? != Code::IndexValue {
            break;
        }
        extent = query.0.formula(op, Formula::Index)?;
    }
    Ok(extent)
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
    graph: &mut Graph,
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

fn fuse_region(graph: &mut Graph, body: &mut Body, region: RegionId) -> Result<bool, OptimizeError> {
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

pub(super) fn run(graph: &mut Graph) -> Result<bool, OptimizeError> {
    let _timing = timing::span("Rust parallel fusion");
    let query = Query(graph);
    let domains = domains(query)?;
    if domains.len() < 2 {
        return Ok(false);
    }
    let barriers = barriers(query)?;
    let mut regions = Vec::new();
    for (&region, operations) in &graph.region_orders {
        let mut seen = LookupSet::new();
        for &op in operations {
            let group = domains.get(&op).map(|domain| domain.group);
            if group.is_some_and(|group| seen.contains(&group)) {
                regions.push(region);
                break;
            }
            if barriers.contains(&op) {
                seen.clear();
            }
            seen.extend(group);
        }
    }
    if regions.is_empty() {
        return Ok(false);
    }
    let mut body = Body::read(query, regions.iter().copied(), domains, barriers)?;
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

fn legal(
    graph: &mut Graph,
    body: &mut Body,
    first: OperationId,
    second: OperationId,
) -> Result<Option<Vec<(OperationId, Value)>>, OptimizeError> {
    let (region, position) = body.positions[first];
    legal_at(graph, body, first, second, region, position)
}

fn legal_at(
    graph: &mut Graph,
    body: &mut Body,
    first: OperationId,
    second: OperationId,
    region: RegionId,
    position: usize,
) -> Result<Option<Vec<(OperationId, Value)>>, OptimizeError> {
    let first_lane = body.lane(first)?;
    let second_lane = body.lane(second)?;
    let second_regions = body.subregions(second);

    let mut loads = Vec::new();
    for (site, name) in body.tree_uses(second) {
        let ValueRef::Result(producer, slot) = body.graph.refs[&name] else {
            continue;
        };
        if producer != first {
            if body.positions[producer].0 == region && body.positions[producer].1 >= position {
                return Ok(None);
            }
            continue;
        }
        let Site::Operand(UseSite::Input(load, 0)) = site else {
            return Ok(None);
        };
        if !body.is_code(load, &Code::TraversalInput) || !body.inside(body.positions[load].0, second_lane) {
            return Ok(None);
        }
        let Some(value) = body.forwarded_value(first, slot) else {
            return Ok(None);
        };
        let domain = &body.domains[second];
        let indices = &body.graph.ir.operations[load].inputs[1..];
        if indices.len() != domain.extents.len()
            || indices.iter().map(|index| body.graph.names[index]).ne(domain.axes.iter().copied())
        {
            return Ok(None);
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

#[derive(Clone, Default)]
struct Effects {
    reads: LookupSet<Value>,
    writes: LookupSet<Value>,
    traps: bool,
}

enum Effect {
    Read(Value),
    Write(Value),
    Trap,
}

impl Effects {
    fn for_each(query: Query<'_>, mut visit: impl FnMut(Value, Effect)) -> Result<(), OptimizeError> {
        query.for_each("OperationRead", |row| {
            visit(row[0], Effect::Read(row[1]));
            Ok(())
        })?;
        query.for_each("OperationWrite", |row| {
            visit(row[0], Effect::Write(row[1]));
            Ok(())
        })?;
        query.for_each("OperationTrap", |row| {
            visit(row[0], Effect::Trap);
            Ok(())
        })
    }

    fn add(&mut self, effect: Effect) {
        match effect {
            Effect::Read(root) => {
                self.reads.insert(root);
            }
            Effect::Write(root) => {
                self.writes.insert(root);
            }
            Effect::Trap => self.traps = true,
        }
    }
}

fn effects_conflict(
    graph: &mut Graph,
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
