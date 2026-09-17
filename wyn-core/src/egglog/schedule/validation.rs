use super::{error, Edge, Exit, FunctionKind, Instruction, OptimizeError, Storage, Value};
use crate::egglog::data::{BlockId, BodyId, OperationKind};
use crate::egglog::{Program, Scheduled};
use crate::interface::EntryKind;
use std::collections::BTreeSet;

pub(super) fn validate(data: &Program<Scheduled>) -> Result<(), OptimizeError> {
    let mut launches = BTreeSet::new();
    for (&id, block) in &data.state.blocks {
        if let Some(exit) = block.loop_exit {
            if !data.state.blocks.get(exit).is_some_and(|b| b.function == block.function) {
                return Err(error("loop exit is outside its function"));
            }
        }
        let Some(owner) = data.state.blocks.get(block.function) else {
            return Err(error("block has a missing owner"));
        };
        let Some(function) = &owner.interface else {
            return Err(error("block owner is not a function entry"));
        };
        if owner.function != block.function || (block.interface.is_some() && block.function != id) {
            return Err(error("invalid function adornment"));
        }
        let device = match function.kind {
            FunctionKind::Device | FunctionKind::Kernel(_) => true,
            FunctionKind::Entry(entry) => data.entries[entry].declaration.entry_kind != EntryKind::Compute,
            FunctionKind::Host => false,
        };
        let Some(body) = data.state.bodies.get(block.body) else {
            return Err(error("missing block body"));
        };
        if !body.results.is_empty() {
            return Err(error("block instructions cannot implicitly return values"));
        }
        for instruction in &body.instructions {
            match instruction {
                Instruction::Evaluate(op) => {
                    let Some(op) = data.operations.get(*op) else {
                        return Err(error("unknown source operation"));
                    };
                    if !matches!(
                        op.kind,
                        OperationKind::Call { .. }
                            | OperationKind::EvalGlobal(_)
                            | OperationKind::Index { .. }
                    ) {
                        return Err(error("unlowered control flow or SOAC in executable body"));
                    }
                }
                Instruction::Call {
                    function: target,
                    arguments,
                    results,
                } => {
                    let Some(target) = data.state.blocks.get(*target) else {
                        return Err(error("missing call target"));
                    };
                    let Some(interface) = &target.interface else {
                        return Err(error("call target is not a function"));
                    };
                    if arguments.len() != target.parameters.len() || results.len() != interface.results {
                        return Err(error("call signature mismatch"));
                    }
                    let target_device = matches!(interface.kind, FunctionKind::Device);
                    if device != target_device || matches!(interface.kind, FunctionKind::Kernel(_)) {
                        return Err(error("call crosses host/device boundary"));
                    }
                    for value in arguments {
                        check_value(data, value)?;
                    }
                }
                Instruction::Dispatch(id) => {
                    if device {
                        return Err(error("device function contains a nested dispatch"));
                    }
                    if data.state.dispatches.get(*id).is_none() || !launches.insert(*id) {
                        return Err(error("missing or duplicated dispatch site"));
                    }
                }
                Instruction::Allocate(id) => {
                    let Some(buffer) = data.state.buffers.get(*id) else {
                        return Err(error("missing allocation"));
                    };
                    if matches!(buffer.storage, Storage::External(_))
                        || device != matches!(buffer.storage, Storage::Function)
                    {
                        return Err(error("buffer allocation is in the wrong address space"));
                    }
                }
                Instruction::BindParameter(_, value)
                | Instruction::BindExpression(_, value)
                | Instruction::BindResult(_, value) => check_value(data, value)?,
                Instruction::Load { buffer, index, .. } => {
                    check_value(data, buffer)?;
                    check_value(data, index)?;
                }
                Instruction::Store { buffer, index, value } => {
                    check_value(data, buffer)?;
                    check_value(data, index)?;
                    check_value(data, value)?;
                }
            }
        }
        match &block.exit {
            Exit::Return(body) => check_tuple(data, *body, function.results)?,
            Exit::Jump(edge) => check_edge(data, block.function, edge)?,
            Exit::Branch { condition, yes, no } => {
                check_tuple(data, *condition, 1)?;
                check_edge(data, block.function, yes)?;
                check_edge(data, block.function, no)?;
            }
        }
    }
    if launches.len() != data.state.dispatches.len() {
        return Err(error("orphaned dispatch"));
    }
    for dispatch in data.state.dispatches.values() {
        let Some(kernel) = data.state.blocks.get(dispatch.kernel) else {
            return Err(error("missing dispatched kernel"));
        };
        let Some(interface) = &kernel.interface else {
            return Err(error("dispatch target is not a function"));
        };
        let FunctionKind::Kernel(size) = interface.kind else {
            return Err(error("dispatch target is not a kernel"));
        };
        if size.contains(&0) || kernel.parameters.len() != dispatch.captures.len() || interface.results != 0
        {
            return Err(error("invalid kernel interface"));
        }
        let Some(grid) = data.state.grids.get(dispatch.grid) else {
            return Err(error("missing dispatch grid"));
        };
        for value in &grid.groups {
            check_value(data, value)?;
        }
        for &id in dispatch.reads.iter().chain(&dispatch.writes) {
            let Some(buffer) = data.state.buffers.get(id) else {
                return Err(error("unknown dispatch buffer"));
            };
            if matches!(buffer.storage, Storage::Function) {
                return Err(error("dispatch references local memory"));
            }
        }
        if !dispatch.dependencies.is_subset(&launches) {
            return Err(error("dispatch depends on an absent launch"));
        }
    }
    // Dependencies order static launch sites; loop repetition belongs to the
    // host CFG. Inserting loop backedges here would create an invalid DAG.
    let mut ordered = BTreeSet::new();
    loop {
        let before = ordered.len();
        for (&id, dispatch) in &data.state.dispatches {
            if dispatch.dependencies.is_subset(&ordered) {
                ordered.insert(id);
            }
        }
        if before == ordered.len() {
            break;
        }
    }
    if ordered.len() != launches.len() {
        return Err(error("cyclic dispatch dependencies"));
    }
    Ok(())
}

fn check_tuple(data: &Program<Scheduled>, id: BodyId, arity: usize) -> Result<(), OptimizeError> {
    let Some(body) = data.state.bodies.get(id) else {
        return Err(error("missing argument/result body"));
    };
    if !body.instructions.is_empty() || body.results.len() != arity {
        return Err(error("block edge/return arity mismatch"));
    }
    for value in &body.results {
        check_value(data, value)?;
    }
    Ok(())
}

fn check_edge(data: &Program<Scheduled>, owner: BlockId, edge: &Edge) -> Result<(), OptimizeError> {
    let Some(target) = data.state.blocks.get(edge.target) else {
        return Err(error("missing control-flow target"));
    };
    if target.function != owner {
        return Err(error("jump crosses a function boundary"));
    }
    check_tuple(data, edge.arguments, target.parameters.len())
}

fn check_value(data: &Program<Scheduled>, value: &Value) -> Result<(), OptimizeError> {
    match value {
        Value::Source(id) if data.expressions.get(*id).is_none() => {
            return Err(error("unknown opaque source expression"))
        }
        Value::Buffer(id) if data.state.buffers.get(*id).is_none() => return Err(error("unknown buffer")),
        Value::Field(value, _) => check_value(data, value)?,
        Value::Tuple(values) | Value::Primitive(_, values) => {
            for value in values {
                check_value(data, value)?;
            }
        }
        _ => {}
    }
    Ok(())
}
