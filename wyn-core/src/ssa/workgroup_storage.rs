//! Reuse workgroup allocations with the same liveness as private arrays.
//! Cross-lane accesses extend each lifetime to its next synchronization point.
use super::invalid;
use crate::error;
use crate::op::{OpTag, PureViewSource};
use crate::ssa::dataflow::Dataflow;
use crate::ssa::storage::memory_id;
use crate::ssa::types::{ConstantValue, FuncBody, InstId, InstKind, ValueRef};
use crate::types::{Type, TypeExt};
use crate::{FunctionId, LookupMap, LookupSet, ResourceAccess};
use egglog_engine::{Read, Write};

impl Dataflow {
    fn workgroup(
        body: &FuncBody,
        calls: &LookupMap<FunctionId, Option<ResourceAccess>>,
    ) -> Result<Self, egglog_engine::Error> {
        let mut flow = Dataflow::memory(body)?;
        flow.memory_accesses(body, calls)?;
        let mut shared = LookupSet::default();
        flow.graph.read(|r| {
            r.constructor_enodes_while("MemorySharedInstruction", |row| {
                shared.insert(flow.graph.value_to_base::<i64>(row.children[0]));
                true
            })
        })?;
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
        flow.control(body, &events)?;
        flow.graph.parse_and_run_program(None, include_str!("workgroup_storage.egg"))?;
        flow.graph.update(|mut sink| {
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
        })?;
        flow.graph.parse_and_run_program(None,
            "(run-schedule (saturate workgroup-access) (saturate workgroup-prior) workgroup-transfers (saturate workgroup-uses))"
        )?;
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
    inst: InstId,
    id: u32,
    count: u32,
    element: Type,
}

pub(super) fn apply(program: &mut crate::ssa::stage::Reachable) -> error::Result<()> {
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

fn optimize_body(
    body: &mut FuncBody,
    calls: &LookupMap<FunctionId, Option<ResourceAccess>>,
) -> error::Result<()> {
    let mut buffers = Vec::new();
    for (inst, node) in &body.inner.insts {
        if let InstKind::Op {
            tag: OpTag::StorageView(PureViewSource::Workgroup { id, count }),
            ..
        } = node.data
        {
            let Some(element) = node.result.and_then(|v| body.get_value_type(v).elem_type()).cloned()
            else {
                return Err(invalid("workgroup allocation has no element type"));
            };
            buffers.push(Buffer {
                inst,
                id,
                count,
                element,
            });
        }
    }
    if buffers.is_empty() {
        return Ok(());
    }
    optimize_storage(body, &buffers, calls).map_err(|e| invalid(&e.to_string()))
}

fn optimize_storage(
    body: &mut FuncBody,
    buffers: &[Buffer],
    calls: &LookupMap<FunctionId, Option<ResourceAccess>>,
) -> Result<(), egglog_engine::Error> {
    let mut flow = Dataflow::workgroup(body, calls)?;
    pool(body, buffers, &mut flow)?;
    let mut removed = LookupSet::default();
    for (&inst, &(before, _)) in &flow.instructions {
        let InstKind::Copy {
            source, destination, ..
        } = body.inner.insts[inst].data
        else {
            continue;
        };
        let shared = whole_transfer(body, &body.inner.insts[inst].data).is_some_and(|(a, b)| a == b);
        if source == destination || (shared && flow.graph.read(|r| r.contains("TransferStepAt", before))?) {
            removed.insert(inst);
            flow.graph.update(|mut sink| {
                let source = sink.add("SsaInstruction", memory_id(inst))?;
                sink.remove("MemoryAt", (source, before))
            })?;
        }
    }
    for inst in body.inner.blocks.values().flat_map(|b| b.insts.iter()) {
        if !matches!(body.inner.insts[*inst].data, InstKind::ControlBarrier) {
            continue;
        }
        flow.graph.parse_and_run_program(
            None,
            "(run-schedule workgroup-open (saturate workgroup-barriers))",
        )?;
        let point = flow.instructions[inst].0;
        if !flow.graph.read(|r| r.contains("WorkgroupBarrierConflict", point))? {
            flow.graph.update(|mut sink| sink.add("WorkgroupOpen", point))?;
            removed.insert(*inst);
        }
    }
    for data in body.inner.blocks.values_mut() {
        data.insts.retain(|inst| !removed.contains(inst));
    }
    body.inner.insts.retain(|inst, _| !removed.contains(&inst));
    Ok(())
}

fn pool(body: &mut FuncBody, buffers: &[Buffer], flow: &mut Dataflow) -> Result<(), egglog_engine::Error> {
    let mut eligible = LookupSet::default();
    for buffer in buffers {
        if buffers.iter().filter(|b| b.id == buffer.id).count() == 1
            && !flow.graph.read(|r| r.contains("MemoryEscapedWorkgroup", i64::from(buffer.id)))?
        {
            eligible.insert(buffer.id);
        }
    }
    let graph = &mut flow.graph;
    graph.update(|mut sink| {
        for (i, a) in buffers.iter().enumerate().filter(|(_, b)| eligible.contains(&b.id)) {
            for b in
                buffers.iter().skip(i + 1).filter(|b| eligible.contains(&b.id) && a.element == b.element)
            {
                sink.add("PoolPair", (i64::from(a.id.min(b.id)), i64::from(a.id.max(b.id))))?;
            }
        }
        Ok::<_, egglog_engine::Error>(())
    })?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate flow-live) (saturate storage-interference))",
    )?;
    let mut order: Vec<_> = buffers.iter().collect();
    order.sort_by_key(|b| (std::cmp::Reverse(b.count), b.id));
    let mut pools: Vec<&Buffer> = Vec::new();
    for buffer in order {
        let mut chosen = buffer;
        if eligible.contains(&buffer.id) {
            for &pool in pools.iter().filter(|p| p.element == buffer.element) {
                if !graph.read(|r| r.contains("PoolBlocked", (i64::from(buffer.id), i64::from(pool.id))))? {
                    chosen = pool;
                    break;
                }
            }
            if chosen.id == buffer.id {
                pools.push(buffer);
            }
        }
        graph.update(|mut sink| {
            sink.add("PoolMember", (i64::from(buffer.id), i64::from(chosen.id)))?;
            let logical = sink.add("MemoryWorkgroup", i64::from(buffer.id))?;
            let physical = sink.add("MemoryWorkgroup", i64::from(chosen.id))?;
            sink.union(logical, physical)?;
            Ok::<_, egglog_engine::Error>(())
        })?;
        graph.parse_and_run_program(None, "(run-schedule storage-pools)")?;
        if let InstKind::Op {
            tag: OpTag::StorageView(PureViewSource::Workgroup { id, count }),
            ..
        } = &mut body.inner.insts[buffer.inst].data
        {
            *id = chosen.id;
            *count = chosen.count;
        }
    }

    Ok(())
}

#[cfg(test)]
#[path = "workgroup_storage_tests.rs"]
mod tests;
