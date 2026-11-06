//! Reuse workgroup allocations with the same liveness as private arrays.
//! Cross-lane accesses extend each lifetime to its next synchronization point.
use super::invalid;
use crate::error::Result;
use crate::op::{OpTag, PureViewSource};
use crate::ssa::dataflow::Dataflow;
use crate::ssa::storage::{MemoryRoot, MemoryRoots};
use crate::ssa::types::{ConstantValue, FuncBody, InstKind, ValueRef};
use crate::types::{Type, TypeExt};
use crate::{FunctionId, LookupMap, LookupSet, ResourceAccess};
use egglog_engine::sort::SetContainer;
use egglog_engine::{Read, Value, Write};

#[derive(Default)]
struct Access {
    reads: LookupSet<i64>,
    writes: LookupSet<i64>,
}
impl Access {
    fn conflicts(&self, other: &Self) -> bool {
        !self.writes.is_disjoint(&other.reads)
            || !self.writes.is_disjoint(&other.writes)
            || !self.reads.is_disjoint(&other.writes)
    }
}

impl Dataflow {
    fn workgroup(
        body: &FuncBody,
        roots: &MemoryRoots,
        calls: &LookupMap<FunctionId, Option<ResourceAccess>>,
    ) -> Result<Self> {
        let mut accesses = LookupMap::default();
        for (inst, node) in &body.inner.insts {
            let (mut reads, mut writes) = roots.access(&node.data, calls);
            reads.retain(shared);
            writes.retain(shared);
            if !reads.is_empty()
                || !writes.is_empty()
                || matches!(
                    node.data,
                    InstKind::ControlBarrier
                        | InstKind::Op {
                            tag: OpTag::StorageView(PureViewSource::Workgroup { .. }),
                            ..
                        }
                )
            {
                accesses.insert(inst, (reads, writes));
            }
        }
        let events = body
            .inner
            .blocks
            .iter()
            .map(|(b, data)| {
                (
                    b,
                    data.insts.iter().filter(|i| accesses.contains_key(i)).map(|&i| vec![i]).collect(),
                )
            })
            .collect();
        let map_error = |e: egglog_engine::Error| invalid(&e.to_string());
        let mut flow = Dataflow::new(body, &events).map_err(map_error)?;
        flow.graph.parse_and_run_program(None, include_str!("workgroup_storage.egg")).map_err(map_error)?;
        let mut accesses: LookupMap<_, _> =
            accesses.into_iter().map(|(inst, access)| (flow.instructions[&inst].0, access)).collect();
        for (block, data) in &body.inner.blocks {
            accesses.insert(
                flow.terminators[&block].0,
                (
                    roots.returned(&data.term).into_iter().filter(shared).collect(),
                    LookupSet::default(),
                ),
            );
        }
        flow.graph
            .update(|mut sink| {
                for (&inst, &(before, after)) in &flow.instructions {
                    let instruction = &body.inner.insts[inst].data;
                    if matches!(instruction, InstKind::ControlBarrier) {
                        sink.add("WorkgroupBarrier", before)?;
                    }
                    if let InstKind::Op {
                        tag: OpTag::StorageView(PureViewSource::Workgroup { id, .. }),
                        ..
                    } = instruction
                    {
                        sink.add("WorkgroupDefine", (before, i64::from(*id)))?;
                    }
                    if let Some((source, destination)) = whole_transfer(body, instruction) {
                        sink.add(
                            "WorkgroupCopy",
                            (before, after, i64::from(source), i64::from(destination)),
                        )?;
                    }
                }
                for (point, (reads, writes)) in &accesses {
                    for (write, roots) in [(false, reads), (true, writes)] {
                        for root in roots {
                            match root {
                                MemoryRoot::Workgroup(id) => {
                                    sink.add("WorkgroupAccess", (*point, i64::from(*id), write))?;
                                }
                                MemoryRoot::Unknown => {
                                    sink.add("WorkgroupUnknown", (*point, write))?;
                                }
                                _ => {}
                            }
                        }
                    }
                }
                Ok::<_, egglog_engine::Error>(())
            })
            .map_err(map_error)?;
        flow.graph.parse_and_run_program(None,
            "(run-schedule (saturate flow-control-index) (saturate workgroup-access) (saturate workgroup-prior) workgroup-transfers (saturate workgroup-uses))"
        ).map_err(map_error)?;
        Ok(flow)
    }

    fn until_barrier(&self, start: Value, backwards: bool, removed: &LookupSet<Value>) -> Result<Access> {
        let mut access = Access::default();
        let mut visited = LookupSet::default();
        let mut pending = vec![start];
        let graph = &self.graph;
        let map_error = |e: egglog_engine::Error| invalid(&e.to_string());
        while let Some(point) = pending.pop() {
            if !visited.insert(point) {
                continue;
            }
            let table = if backwards { "FlowPredecessors" } else { "FlowSuccessors" };
            let Some(neighbors) = graph.read(|r| r.lookup(table, point)).map_err(map_error)? else {
                continue;
            };
            let Some(neighbors) = graph.value_to_container::<SetContainer>(neighbors) else {
                return Err(invalid("invalid control-flow neighbor set"));
            };
            for &next in &neighbors.data {
                let before = if backwards { next } else { point };
                if graph.read(|r| r.contains("WorkgroupBarrier", before)).map_err(map_error)?
                    && !removed.contains(&before)
                {
                    continue;
                }
                for (write, output) in [(false, &mut access.reads), (true, &mut access.writes)] {
                    if let Some(values) =
                        graph.read(|r| r.lookup("WorkgroupAccesses", (before, write))).map_err(map_error)?
                    {
                        let Some(values) = graph.value_to_container::<SetContainer>(values) else {
                            return Err(invalid("invalid workgroup access set"));
                        };
                        output.extend(values.data.iter().map(|&v| graph.value_to_base::<i64>(v)));
                    }
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

fn whole_transfer(body: &FuncBody, instruction: &InstKind) -> Option<(u32, u32)> {
    let InstKind::Copy {
        destination,
        source,
        length,
        collective: true,
        ..
    } = *instruction
    else {
        return None;
    };
    let length = integer(length)?;
    Some((
        whole_view(body, source, length)?,
        whole_view(body, destination, length)?,
    ))
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
    let mut flow = Dataflow::workgroup(body, &roots, calls)?;
    pool(body, &buffers, &roots, &mut flow)?;
    let selected = MemoryRoots::analyze(body);
    let mut removed = LookupSet::default();
    for (&inst, &(before, _)) in &flow.instructions {
        if let InstKind::Copy {
            source, destination, ..
        } = body.inner.insts[inst].data
        {
            if source == destination
                || (flow
                    .graph
                    .read(|r| r.contains("TransferStepAt", before))
                    .map_err(|e| invalid(&e.to_string()))?
                    && selected.value(source) == selected.value(destination))
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
    let flow = Dataflow::workgroup(body, &selected, calls)?;
    let mut removed = LookupSet::default();
    for inst in body.inner.blocks.values().flat_map(|b| b.insts.iter()) {
        if !matches!(body.inner.insts[*inst].data, InstKind::ControlBarrier) {
            continue;
        }
        let (before, after) = flow.instructions[inst];
        let point = before;
        let before = flow.until_barrier(before, true, &removed)?;
        let after = flow.until_barrier(after, false, &removed)?;
        if !before.conflicts(&after) {
            removed.insert(point);
        }
    }
    for data in body.inner.blocks.values_mut() {
        data.insts.retain(|inst| flow.instructions.get(inst).is_none_or(|p| !removed.contains(&p.0)));
    }
    body.inner
        .insts
        .retain(|inst, _| flow.instructions.get(&inst).is_none_or(|p| !removed.contains(&p.0)));
    Ok(())
}

fn pool(body: &mut FuncBody, buffers: &[Buffer], roots: &MemoryRoots, flow: &mut Dataflow) -> Result<()> {
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
    let graph = &mut flow.graph;

    graph
        .update(|mut sink| {
            for &(a, b) in &pairs {
                sink.add("PoolPair", (i64::from(a), i64::from(b)))?;
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
