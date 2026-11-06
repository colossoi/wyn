//! Reuse workgroup allocations with the same liveness as private arrays.
//! Cross-lane accesses extend each lifetime to its next synchronization point.
use super::invalid;
use crate::error::Result;
use crate::op::{OpTag, PureViewSource};
use crate::ssa::dataflow::Dataflow;
use crate::ssa::storage::memory_id;
use crate::ssa::types::{ConstantValue, FuncBody, InstKind, ValueRef};
use crate::types::{Type, TypeExt};
use crate::{FunctionId, LookupMap, LookupSet, ResourceAccess};
use egglog_engine::{Read, Write};

impl Dataflow {
    fn workgroup(body: &FuncBody, calls: &LookupMap<FunctionId, Option<ResourceAccess>>) -> Result<Self> {
        let map_error = |e: egglog_engine::Error| invalid(&e.to_string());
        let mut flow = Dataflow::memory(body).map_err(map_error)?;
        flow.memory_accesses(body, calls).map_err(map_error)?;
        let mut shared = LookupSet::default();
        flow.graph
            .read(|r| {
                r.constructor_enodes_while("MemorySharedInstruction", |row| {
                    shared.insert(flow.graph.value_to_base::<i64>(row.children[0]));
                    true
                })
            })
            .map_err(map_error)?;
        let events = body
            .inner
            .blocks
            .iter()
            .map(|(block, data)| {
                (
                    block,
                    data.insts
                        .iter()
                        .filter(|&&inst| {
                            shared.contains(&memory_id(inst))
                                || matches!(
                                    body.inner.insts[inst].data,
                                    InstKind::ControlBarrier
                                        | InstKind::Op {
                                            tag: OpTag::StorageView(PureViewSource::Workgroup { .. }),
                                            ..
                                        }
                                )
                        })
                        .map(|&inst| vec![inst])
                        .collect(),
                )
            })
            .collect();
        flow.control(body, &events).map_err(map_error)?;
        flow.graph.parse_and_run_program(None, include_str!("workgroup_storage.egg")).map_err(map_error)?;
        flow.graph
            .update(|mut sink| {
                for (&inst, &(before, after)) in &flow.instructions {
                    let instruction = &body.inner.insts[inst].data;
                    let source = sink.add("SsaInstruction", memory_id(inst))?;
                    sink.add("MemoryAt", (source, before))?;
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
                for (&block, &(before, _)) in &flow.terminators {
                    let source = sink.add("SsaReturn", memory_id(block))?;
                    sink.add("MemoryAt", (source, before))?;
                }
                Ok::<_, egglog_engine::Error>(())
            })
            .map_err(map_error)?;
        flow.graph.parse_and_run_program(None,
            "(run-schedule (saturate flow-control-index) (saturate workgroup-access) (saturate workgroup-prior) workgroup-transfers (saturate workgroup-uses))"
        ).map_err(map_error)?;
        Ok(flow)
    }
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
    let calls =
        crate::ssa::storage::workgroup_call_effects(program).map_err(|e| invalid(&e.to_string()))?;
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
    let mut flow = Dataflow::workgroup(body, calls)?;
    pool(body, &buffers, &mut flow)?;
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
                    && whole_transfer(body, &body.inner.insts[inst].data).is_some_and(|(a, b)| a == b))
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
    let mut flow = Dataflow::workgroup(body, calls)?;
    let mut removed = LookupSet::default();
    for inst in body.inner.blocks.values().flat_map(|b| b.insts.iter()) {
        if !matches!(body.inner.insts[*inst].data, InstKind::ControlBarrier) {
            continue;
        }
        flow.graph
            .parse_and_run_program(
                None,
                "(run-schedule workgroup-open (saturate workgroup-barriers))",
            )
            .map_err(|e| invalid(&e.to_string()))?;
        let point = flow.instructions[inst].0;
        if !flow
            .graph
            .read(|r| r.contains("WorkgroupBarrierConflict", point))
            .map_err(|e| invalid(&e.to_string()))?
        {
            flow.graph
                .update(|mut sink| sink.add("WorkgroupOpen", point))
                .map_err(|e| invalid(&e.to_string()))?;
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

fn pool(body: &mut FuncBody, buffers: &[Buffer], flow: &mut Dataflow) -> Result<()> {
    let map_error = |e: egglog_engine::Error| invalid(&e.to_string());
    let mut escaped = LookupSet::default();
    flow.graph
        .read(|r| {
            r.constructor_enodes_while("MemoryEscapedWorkgroup", |row| {
                escaped.insert(flow.graph.value_to_base::<i64>(row.children[0]) as u32);
                true
            })
        })
        .map_err(map_error)?;
    let eligible = |buffer: &Buffer| {
        !escaped.contains(&buffer.id) && buffers.iter().filter(|other| other.id == buffer.id).count() == 1
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
