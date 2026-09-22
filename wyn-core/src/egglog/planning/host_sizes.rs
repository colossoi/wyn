//! Publish size expressions whose leaves are host-visible interface values.
use egglog_engine::sort::S;
use egglog_engine::{EGraph, Value};
use std::collections::HashMap;

use crate::egglog::abi::error;
use crate::egglog::data::BufferId;
use crate::egglog::planning::{number, rows};
use crate::egglog::{OptimizeError, Program, Scheduled};
use crate::host::{BufferLen, HostSizeInput, HostSizeScalar, IntegerOp, SizeExpr, SizeOp};
use crate::BindingRef;

pub(super) fn publish(
    graph: &EGraph,
    data: &mut Program<Scheduled>,
    buffers: &HashMap<Value, BufferId>,
    bindings: &HashMap<Value, BindingRef>,
    inputs: &[HostSizeInput],
) -> Result<(), OptimizeError> {
    let mut known = HashMap::<Value, SizeExpr>::new();
    graph.constructor_enodes("AbiNumber", |node| {
        known.insert(
            node.eclass,
            SizeExpr::Integer(graph.value_to_base::<i64>(node.children[0])),
        );
    })?;
    rows(graph, "AbiHostInput", |row| {
        let Some(input) = inputs.get(number(graph, row[1])? as usize) else {
            return Err(error("missing host size scalar"));
        };
        let scalar = match input {
            HostSizeInput::Uniform { scalar, .. } | HostSizeInput::PushConstant { scalar, .. } => scalar,
        };
        if matches!(scalar, HostSizeScalar::I32 | HostSizeScalar::U32) {
            known.insert(row[0], SizeExpr::Scalar(input.clone()));
        }
        Ok(())
    })?;
    let mut status = Ok(());
    graph.constructor_enodes_while("AbiBufferLength", |node| {
        status = (|| {
            let Some(binding) = bindings.get(&node.children[0]) else {
                return Err(error("host length has no binding"));
            };
            let stride = number(graph, node.children[1])?;
            if stride == 0 {
                return Err(error("host length has zero stride"));
            }
            known.insert(
                node.eclass,
                SizeExpr::BufferLength {
                    set: binding.set,
                    binding: binding.binding,
                    stride,
                },
            );
            Ok(())
        })();
        status.is_ok()
    })?;
    status?;
    let mut aliases = vec![];
    rows(graph, "AbiAlias", |row| {
        aliases.push((row[0], row[1]));
        Ok(())
    })?;
    let mut operations = vec![];
    graph.constructor_enodes("AbiBinary", |node| {
        let name = graph.value_to_base::<S>(node.children[0]);
        let op = match name.as_str() {
            "i32_add" => Some(SizeOp::I32(IntegerOp::Add)),
            "i32_sub" => Some(SizeOp::I32(IntegerOp::Subtract)),
            "i32_mul" => Some(SizeOp::I32(IntegerOp::Multiply)),
            "u32_add" => Some(SizeOp::U32(IntegerOp::Add)),
            "u32_sub" => Some(SizeOp::U32(IntegerOp::Subtract)),
            "u32_mul" => Some(SizeOp::U32(IntegerOp::Multiply)),
            "add" => Some(SizeOp::Add),
            "sub" => Some(SizeOp::Subtract),
            "mul" => Some(SizeOp::Multiply),
            "ceil_div" => Some(SizeOp::Ceiling),
            "mod" => Some(SizeOp::Mod),
            "min" => Some(SizeOp::Min),
            "max" => Some(SizeOp::Max),
            _ => None,
        };
        if let Some(op) = op {
            operations.push((node.eclass, op, node.children[1], node.children[2]));
        }
    })?;
    loop {
        let before = known.len();
        for &(target, source) in &aliases {
            if known.contains_key(&target) {
                continue;
            }
            if let Some(value) = known.get(&source).cloned() {
                known.insert(target, value);
            }
        }
        for &(target, op, left, right) in &operations {
            if known.contains_key(&target) {
                continue;
            }
            if let (Some(left), Some(right)) = (known.get(&left), known.get(&right)) {
                let value = SizeExpr::Binary {
                    op,
                    left: Box::new(left.clone()),
                    right: Box::new(right.clone()),
                };
                known.insert(target, value);
            }
        }
        if known.len() == before {
            break;
        }
    }
    rows(graph, "AbiAllocationBytes", |row| {
        let Some(id) = buffers.get(&row[0]) else {
            return Ok(());
        };
        let Some(binding) = data.state.abi.bindings.get_mut(id) else {
            return Ok(());
        };
        if !matches!(
            binding.length,
            Some(BufferLen::HostProvided { .. } | BufferLen::SameAsDispatch { .. })
        ) {
            return Ok(());
        }
        let Some(bytes) = known.get(&row[1]) else {
            return Ok(());
        };
        binding.length = Some(BufferLen::Computed { bytes: bytes.clone() });
        Ok(())
    })?;
    Ok(())
}
