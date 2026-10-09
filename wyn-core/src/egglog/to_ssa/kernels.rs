//! Expand scheduled phases straight into SSA instructions and structured loops.

use super::{error, Body, OptimizeError, Typed};
use crate::builtins::catalog;
use crate::op::{BinaryOperator, OpTag};
use crate::types::{self, Type, TypeName};
use crate::LookupMap;
use egglog_engine::sort::S;
use egglog_engine::Value;

mod coordinates;
mod filter;
mod indexed;
mod loops;
mod screma;
use filter::{compact, serial_filter};
use indexed::{buckets, indexed};
use screma::screma;

pub(super) fn emit(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    phase_owner: Value,
    phase: &str,
    phase_extent: Value,
    phase_width: u32,
) -> Result<(), OptimizeError> {
    if matches!(phase, "loop_enter" | "loop_exit") {
        return loops::emit(body, scope, phase_owner, phase, phase_width);
    }
    if phase == "scalar" {
        let operations = body.compiler.facts.scalar_group(phase_owner);
        let Some(context) = body.compiler.facts.dispatch_context(phase_owner) else {
            return Err(error("scalar dispatch has no selected context"));
        };
        body.context = context;
        for operation in operations {
            let Some(source) = body.compiler.facts.source(operation) else {
                return Err(error("scalar dispatch source missing"));
            };
            let value = body.value(scope, source)?;
            if let Some(output) = body.slot(scope, operation, "scalar", 0, 2)? {
                let zero = body.literal("0", &types::i32())?;
                body.store(output, zero, value)?;
            }
        }
        return Ok(());
    }
    let Some(plan) = body.compiler.facts.group(phase_owner) else {
        return Err(error("kernel has no fusion plan"));
    };
    let results = body.compiler.facts.results(plan)?;
    match phase {
        "elements" => {
            let n = body.extent(scope, phase_extent)?;
            let (start, step) = invocation(body, phase_width)?;
            body.counted(start, n, step, vec![], |body, index, _| {
                let mut cache = LookupMap::default();
                write_arrays(body, scope, phase_owner, plan, &results, index, &mut cache)?;
                Ok(vec![])
            })?;
            Ok(())
        }
        "scatter" | "initialize" | "atomic" => {
            indexed(body, scope, phase_owner, phase, phase_extent, phase_width, plan)
        }
        "clear" | "buckets" => buckets(body, scope, phase_owner, phase, phase_width, plan),
        "ordered" => {
            let Some(recipe) = body.compiler.facts.lookup("OrderedRecipe", (phase_owner,)) else {
                return Err(error("ordered kernel has no selected algorithm"));
            };
            match body.compiler.program.graph.value_to_base::<S>(recipe).as_ref() {
                "screma" => screma(body, scope, phase_owner, phase, phase_width, plan, &results),
                "filter" => serial_filter(body, scope, phase_owner, plan, &results),
                "indexed" => indexed(body, scope, phase_owner, phase, phase_extent, phase_width, plan),
                "buckets" => buckets(body, scope, phase_owner, phase, phase_width, plan),
                _ => Err(error("unknown ordered algorithm")),
            }
        }
        "compact" => compact(body, scope, phase_owner, phase_width, plan, &results),
        "chunks" | "combine" | "offsets" => {
            screma(body, scope, phase_owner, phase, phase_width, plan, &results)
        }
        _ => Err(error(format!("scheduled {} kernel is not implemented", phase))),
    }
}

pub(super) fn invocation(body: &mut Body<'_, '_, '_>, width: u32) -> Result<(Typed, Typed), OptimizeError> {
    let uint = Type::Constructed(TypeName::UInt(32), vec![]);
    let start = body.op(
        OpTag::Intrinsic {
            id: catalog().known().thread_id,
            overload_idx: 0,
        },
        vec![],
        uint.clone(),
    )?;
    let groups_ty = uint.clone();
    let groups = body.op(
        OpTag::Intrinsic {
            id: catalog().known().num_workgroups,
            overload_idx: 0,
        },
        vec![],
        groups_ty,
    )?;
    let width = body.literal(&width.to_string(), &uint)?;
    if let Some((x, y, z)) = body.grid {
        let mut start = start;
        for (id, stride) in [
            (catalog().known().thread_id_y, x),
            (catalog().known().thread_id_z, x * y),
        ] {
            let axis = body.op(OpTag::Intrinsic { id, overload_idx: 0 }, vec![], uint.clone())?;
            let stride = body.literal(&stride.to_string(), &uint)?;
            let stride = body.binary(BinaryOperator::Multiply, stride, width.clone())?;
            let offset = body.binary(BinaryOperator::Multiply, axis, stride)?;
            start = body.binary(BinaryOperator::Add, start, offset)?;
        }
        let groups = body.literal(&(u64::from(x) * u64::from(y) * u64::from(z)).to_string(), &uint)?;
        return Ok((start, body.binary(BinaryOperator::Multiply, groups, width)?));
    }
    let step = body.binary(BinaryOperator::Multiply, groups, width)?;
    Ok((start, step))
}

pub(super) fn element(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    plan: Value,
    source: Value,
    index: Typed,
    cache: &mut LookupMap<Value, Typed>,
) -> Result<Typed, OptimizeError> {
    if let Some(value) = cache.get(&source) {
        return Ok(value.clone());
    }
    if let Some(actual) = body.compiler.facts.alias(source) {
        return element(body, scope, plan, actual, index, cache);
    }
    if let Some((parent, field)) = body.compiler.facts.projection(source) {
        let value = element(body, scope, plan, parent, index, cache)?;
        return body.field(value, field);
    }
    if let Some((array, start, _)) = body.compiler.facts.slice(source) {
        let start = body.value(scope, start)?;
        let start = body.cast(start, &index.ty)?;
        let index = body.binary(BinaryOperator::Add, index, start)?;
        // The base is evaluated at a different index from this stream's cache.
        let value = element(body, scope, plan, array, index, &mut LookupMap::default())?;
        cache.insert(source, value.clone());
        return Ok(value);
    }
    if let Some(operation) = body.compiler.facts.operation(source) {
        if body.compiler.facts.member(plan, operation) {
            if body.compiler.facts.operation_kind(operation)? == "map" {
                let owner = body.compiler.facts.operation_scope(operation)?;
                let inputs = body.compiler.facts.inputs(operation)?;
                let mut arguments = Vec::new();
                for (_, input) in inputs {
                    arguments.push(element(body, scope, plan, input, index.clone(), cache)?);
                }
                let value = body.callback(owner, operation, arguments)?;
                cache.insert(source, value.clone());
                return Ok(value);
            }
        }
    }
    let array = body.value(scope, source)?;
    let value = body.index(array, index)?;
    cache.insert(source, value.clone());
    Ok(value)
}

pub(super) fn write_arrays(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operation: Value,
    plan: Value,
    results: &[(String, i64, Value)],
    index: Typed,
    cache: &mut LookupMap<Value, Typed>,
) -> Result<(), OptimizeError> {
    let mut position = 0;
    for (role, _, source) in results {
        if role != "array" {
            continue;
        }
        if let Some(output) = body.slot(scope, operation, "output", position, 2)? {
            let value = element(body, scope, plan, *source, index.clone(), cache)?;
            body.store(output, index.clone(), value)?;
        }
        position += 1;
    }
    Ok(())
}
