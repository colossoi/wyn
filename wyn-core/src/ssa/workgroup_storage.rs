//! Reuse workgroup allocations with the same liveness as private arrays.
//! Cross-lane accesses extend each lifetime to its next synchronization point.
use super::invalid;
use crate::error::Result;
use crate::op::{OpTag, PureViewSource};
use crate::ssa::dataflow::Dataflow;
use crate::ssa::storage::{MemoryRoot, MemoryRoots};
use crate::ssa::types::{ConstantValue, FuncBody, InstId, InstKind, ValueRef};
use crate::types::{Type, TypeExt};
use crate::{FunctionId, LookupMap, LookupSet, ResourceAccess};
use egglog_engine::sort::SetContainer;
use egglog_engine::{Read, Value, Write};

#[derive(Default, Clone)]
struct Access {
    reads: LookupSet<MemoryRoot>,
    writes: LookupSet<MemoryRoot>,
}

impl Access {
    fn conflicts(&self, other: &Self) -> bool {
        let overlaps = |a: &LookupSet<_>, b: &LookupSet<_>| {
            a.iter()
                .any(|x| b.iter().any(|y| x == y || *x == MemoryRoot::Unknown || *y == MemoryRoot::Unknown))
        };
        overlaps(&self.writes, &other.reads)
            || overlaps(&self.writes, &other.writes)
            || overlaps(&self.reads, &other.writes)
    }
    fn extend(&mut self, other: &Self) {
        self.reads.extend(&other.reads);
        self.writes.extend(&other.writes);
    }
}

#[derive(Default)]
struct Step {
    instruction: Option<InstId>,
    define: Option<u32>,
    transfer: Option<(u32, u32)>,
    barrier: bool,
    access: Access,
}

struct Accesses {
    flow: Dataflow,
    steps: LookupMap<(Value, Value), Step>,
}

impl Accesses {
    fn new(
        body: &FuncBody,
        roots: &MemoryRoots,
        calls: &LookupMap<FunctionId, Option<ResourceAccess>>,
    ) -> Result<Self> {
        let mut accesses = LookupMap::default();
        for (inst, node) in &body.inner.insts {
            let (reads, writes) = roots.access(&node.data, calls);
            let step = Step {
                instruction: Some(inst),
                define: match node.data {
                    InstKind::Op {
                        tag: OpTag::StorageView(PureViewSource::Workgroup { id, .. }),
                        ..
                    } => Some(id),
                    _ => None,
                },
                barrier: matches!(node.data, InstKind::ControlBarrier),
                access: Access {
                    reads: reads.into_iter().filter(shared).collect(),
                    writes: writes.into_iter().filter(shared).collect(),
                },
                ..Step::default()
            };
            if step.define.is_some()
                || step.barrier
                || !step.access.reads.is_empty()
                || !step.access.writes.is_empty()
            {
                accesses.insert(inst, step);
            }
        }
        let events = body
            .inner
            .blocks
            .iter()
            .map(|(b, data)| {
                (
                    b,
                    data.insts
                        .iter()
                        .filter(|inst| accesses.contains_key(inst))
                        .map(|&inst| vec![inst])
                        .collect(),
                )
            })
            .collect();
        let mut flow = Dataflow::new(body, &events).map_err(|e| invalid(&e.to_string()))?;
        flow.graph
            .parse_and_run_program(None, "(run-schedule (saturate flow-control-index))")
            .map_err(|e| invalid(&e.to_string()))?;
        let steps = accesses.into_iter().map(|(inst, access)| (flow.instructions[&inst], access)).collect();
        let mut result = Self { flow, steps };
        for (block, data) in &body.inner.blocks {
            let returned = Access {
                reads: roots.returned(&data.term).into_iter().filter(shared).collect(),
                writes: LookupSet::default(),
            };
            if !returned.reads.is_empty() {
                result.steps.insert(
                    result.flow.terminators[&block],
                    Step {
                        access: returned,
                        ..Step::default()
                    },
                );
            }
        }
        let points: Vec<_> = result.steps.keys().copied().collect();
        for point in points {
            let Some(inst) = result.steps[&point].instruction else {
                continue;
            };
            let InstKind::Copy {
                destination,
                source,
                length,
                collective: true,
                ..
            } = body.inner.insts[inst].data
            else {
                continue;
            };
            let Some(length) = integer(length) else {
                continue;
            };
            let (Some(source), Some(destination)) = (
                whole_view(body, source, length),
                whole_view(body, destination, length),
            ) else {
                continue;
            };
            let before = result.until_barrier(point.0, true, &LookupSet::default())?;
            if !before.reads.iter().chain(&before.writes).any(|root| {
                matches!(root, MemoryRoot::Unknown)
                    || *root == MemoryRoot::Workgroup(source)
                    || *root == MemoryRoot::Workgroup(destination)
            }) {
                if let Some(step) = result.steps.get_mut(&point) {
                    step.transfer = Some((source, destination));
                }
            }
        }
        Ok(result)
    }

    fn until_barrier(&self, start: Value, backwards: bool, removed: &LookupSet<InstId>) -> Result<Access> {
        let mut access = Access::default();
        let mut visited = LookupSet::default();
        let mut pending = vec![start];
        while let Some(point) = pending.pop() {
            if !visited.insert(point) {
                continue;
            }
            let graph = &self.flow.graph;
            let table = if backwards { "FlowPredecessors" } else { "FlowSuccessors" };
            let Some(neighbors) =
                graph.read(|r| r.lookup(table, point)).map_err(|e| invalid(&e.to_string()))?
            else {
                continue;
            };
            let Some(neighbors) = graph.value_to_container::<SetContainer>(neighbors) else {
                return Err(invalid("invalid control-flow neighbor set"));
            };
            for &next in &neighbors.data {
                let edge = if backwards { (next, point) } else { (point, next) };
                if let Some(step) = self.steps.get(&edge) {
                    if step.barrier && !step.instruction.is_some_and(|inst| removed.contains(&inst)) {
                        continue;
                    }
                    access.extend(&step.access);
                }
                pending.push(next);
            }
        }
        Ok(access)
    }
}

fn shared(root: &MemoryRoot) -> bool {
    matches!(root, MemoryRoot::Workgroup(_) | MemoryRoot::Unknown)
}

fn integer(value: ValueRef) -> Option<u32> {
    match value {
        ValueRef::Const(ConstantValue::U32(n)) => Some(n),
        ValueRef::Const(ConstantValue::I32(n)) => u32::try_from(n).ok(),
        _ => None,
    }
}

fn whole_view(body: &FuncBody, value: ValueRef, length: u32) -> Option<u32> {
    let node = &body.inner.insts[body.inner.inst_of_value(value.as_ssa()?)?];
    if let InstKind::Op {
        tag: OpTag::StorageView(PureViewSource::Workgroup { id, .. }),
        operands,
    } = &node.data
    {
        if integer(operands[0]) == Some(0) && integer(operands[1]) == Some(length) {
            return Some(*id);
        }
    }
    None
}

struct Buffer {
    id: u32,
    count: u32,
    element: Type,
}

pub(super) fn apply(program: &mut crate::ssa::stage::Reachable) -> Result<()> {
    let calls = crate::ssa::storage::workgroup_call_effects(program);
    for body in program
        .functions
        .iter_mut()
        .map(|f| &mut f.body)
        .chain(program.entry_points.iter_mut().map(|e| &mut e.body))
    {
        optimize_body(body, &calls)?;
    }
    Ok(())
}

fn optimize_body(body: &mut FuncBody, calls: &LookupMap<FunctionId, Option<ResourceAccess>>) -> Result<()> {
    let mut buffers = Vec::new();
    for node in body.inner.insts.values() {
        if let InstKind::Op {
            tag: OpTag::StorageView(PureViewSource::Workgroup { id, count }),
            ..
        } = node.data
        {
            let Some(element) = node.result.and_then(|v| body.get_value_type(v).elem_type()).cloned()
            else {
                return Err(invalid("workgroup allocation has no element type"));
            };
            buffers.push(Buffer { id, count, element });
        }
    }
    if buffers.is_empty() {
        return Ok(());
    }
    let roots = MemoryRoots::analyze(body);
    let mut flow = Accesses::new(body, &roots, calls)?;
    pool(body, &buffers, &roots, &mut flow)?;
    let selected = MemoryRoots::analyze(body);
    let mut removed = LookupSet::default();
    for step in flow.steps.values() {
        let Some(inst) = step.instruction else {
            continue;
        };
        if let InstKind::Copy {
            source, destination, ..
        } = body.inner.insts[inst].data
        {
            if source == destination
                || (step.transfer.is_some() && selected.value(source) == selected.value(destination))
            {
                removed.insert(inst);
            }
        }
    }
    for block in body.inner.blocks.values_mut() {
        block.insts.retain(|inst| !removed.contains(inst));
    }
    body.inner.insts.retain(|inst, _| !removed.contains(&inst));
    // Pooling introduces aliases; synchronization must see the selected storage.
    let roots = MemoryRoots::analyze(body);
    let flow = Accesses::new(body, &roots, calls)?;
    let mut removed = LookupSet::default();
    for inst in body.inner.blocks.values().flat_map(|b| b.insts.iter()) {
        if !matches!(body.inner.insts[*inst].data, InstKind::ControlBarrier) {
            continue;
        }
        let (before, after) = flow.flow.instructions[inst];
        let before = flow.until_barrier(before, true, &removed)?;
        let after = flow.until_barrier(after, false, &removed)?;
        if !before.conflicts(&after) {
            removed.insert(*inst);
        }
    }
    for data in body.inner.blocks.values_mut() {
        data.insts.retain(|inst| !removed.contains(inst));
    }
    body.inner.insts.retain(|inst, _| !removed.contains(&inst));
    Ok(())
}

fn pool(body: &mut FuncBody, buffers: &[Buffer], roots: &MemoryRoots, flow: &mut Accesses) -> Result<()> {
    let mut escaped = LookupSet::default();
    for node in body.inner.insts.values() {
        for (operand, value) in node.data.value_uses().into_iter().enumerate() {
            if matches!(
                node.data.storage_use(operand),
                crate::ssa::storage::StorageUse::Escape
            ) {
                escaped.extend(roots.value(value));
            }
        }
    }
    for data in body.inner.blocks.values() {
        escaped.extend(roots.returned(&data.term));
    }
    let eligible = |buffer: &Buffer| {
        !escaped.contains(&MemoryRoot::Unknown)
            && !escaped.contains(&MemoryRoot::Workgroup(buffer.id))
            && buffers.iter().filter(|other| other.id == buffer.id).count() == 1
    };
    let mut pairs = Vec::new();
    for (i, a) in buffers.iter().enumerate().filter(|(_, b)| eligible(b)) {
        for b in buffers.iter().skip(i + 1).filter(|b| eligible(b) && a.element == b.element) {
            pairs.push((a.id.min(b.id), a.id.max(b.id)));
        }
    }
    if pairs.is_empty() {
        return Ok(());
    }
    let map_error = |e: egglog_engine::Error| invalid(&e.to_string());
    let graph = &mut flow.flow.graph;
    graph.parse_and_run_program(None, include_str!("workgroup_storage.egg")).map_err(map_error)?;
    let mut edges = Vec::new();
    graph
        .constructor_enodes("FlowNext", |row| edges.push((row.children[0], row.children[1])))
        .map_err(map_error)?;
    graph
        .update(|mut sink| {
            for &(a, b) in &pairs {
                sink.add("PoolPair", (i64::from(a), i64::from(b)))?;
            }
            for &(before, after) in &edges {
                if !flow.steps.get(&(before, after)).is_some_and(|s| s.barrier || s.transfer.is_some()) {
                    sink.add("UnorderedStep", (before, after))?;
                }
            }
            for (&(before, after), step) in &flow.steps {
                if let Some((source, destination)) = step.transfer {
                    sink.set("FlowKilled", (before, i64::from(destination)), true)?;
                    sink.add("FlowUse", (before, i64::from(source)))?;
                    sink.add("FlowUse", (after, i64::from(destination)))?;
                    sink.add("TransferStep", (before, after, i64::from(source)))?;
                    continue;
                }
                if let Some(id) = step.define {
                    sink.set("FlowKilled", (before, i64::from(id)), true)?;
                }
                for root in step.access.reads.iter().chain(&step.access.writes) {
                    match root {
                        MemoryRoot::Workgroup(id) => {
                            sink.add("FlowUse", (before, i64::from(*id)))?;
                        }
                        MemoryRoot::Unknown => {
                            for buffer in buffers {
                                sink.add("FlowUse", (before, i64::from(buffer.id)))?;
                            }
                        }
                        _ => {}
                    }
                }
            }
            Ok::<_, egglog_engine::Error>(())
        })
        .map_err(map_error)?;
    graph
        .parse_and_run_program(
            None,
            "(run-schedule (saturate flow-live) (saturate storage-interference))",
        )
        .map_err(map_error)?;
    let mut conflicts = LookupSet::default();
    for (a, b) in pairs {
        if graph.read(|r| r.contains("Interference", (i64::from(a), i64::from(b)))).map_err(map_error)? {
            conflicts.insert((a, b));
        }
    }
    let mut order: Vec<_> = buffers.iter().filter(|b| eligible(b)).collect();
    order.sort_by_key(|b| (std::cmp::Reverse(b.count), b.id));
    let mut pools: Vec<Vec<&Buffer>> = Vec::new();
    for buffer in order {
        if let Some(pool) = pools.iter_mut().find(|pool| {
            pool[0].element == buffer.element
                && pool.iter().all(|b| !conflicts.contains(&(b.id.min(buffer.id), b.id.max(buffer.id))))
        }) {
            pool.push(buffer);
        } else {
            pools.push(vec![buffer]);
        }
    }
    let selected: LookupMap<_, _> =
        pools.iter().flat_map(|pool| pool.iter().map(|b| (b.id, (pool[0].id, pool[0].count)))).collect();
    for node in body.inner.insts.values_mut() {
        if let InstKind::Op {
            tag: OpTag::StorageView(PureViewSource::Workgroup { id, count }),
            ..
        } = &mut node.data
        {
            if let Some(&(physical, capacity)) = selected.get(id) {
                *id = physical;
                *count = capacity;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "workgroup_storage_tests.rs"]
mod tests;
