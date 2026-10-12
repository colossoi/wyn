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
mod maps;
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
        "elements" => maps::emit(
            body,
            scope,
            phase_owner,
            phase_extent,
            phase_width,
            plan,
            &results,
        ),
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
    indices: &[Typed],
    cache: &mut LookupMap<Value, Typed>,
) -> Result<Typed, OptimizeError> {
    let Some((index, rest)) = indices.split_first() else {
        return body.value(scope, source);
    };
    if let Some(value) = cache.get(&source) {
        return Ok(value.clone());
    }
    if let Some(actual) = body.compiler.facts.alias(source) {
        return element(body, scope, plan, actual, indices, cache);
    }
    if let Some((parent, field)) = body.compiler.facts.projection(source) {
        let value = element(body, scope, plan, parent, indices, cache)?;
        return body.field(value, field);
    }
    if let Some((array, start, _)) = body.compiler.facts.slice(source) {
        let start = body.value(scope, start)?;
        let start = body.cast(start, &index.ty)?;
        let index = body.binary(BinaryOperator::Add, index.clone(), start)?;
        let mut indices = indices.to_vec();
        indices[0] = index;
        // The base is evaluated at a different index from this stream's cache.
        let value = element(body, scope, plan, array, &indices, &mut LookupMap::default())?;
        cache.insert(source, value.clone());
        return Ok(value);
    }
    if let Some(operation) = body.compiler.facts.operation(source) {
        if body
            .compiler
            .facts
            .lookup("PlanMapping", (plan, operation))
            .is_some_and(|mapping| body.compiler.program.graph.value_to_base::<bool>(mapping))
        {
            let Some(callback) = body.compiler.facts.callback(operation) else {
                return Err(error("mapping component has no callback"));
            };
            let owner = body.compiler.facts.operation_scope(operation)?;
            let inputs = body.compiler.facts.inputs(operation)?;
            let mut arguments = Vec::new();
            // Input elements and nested output leaves have different ranks.
            let mut input_cache = LookupMap::default();
            for (_, input) in inputs {
                let cache = if rest.is_empty() { &mut *cache } else { &mut input_cache };
                arguments.push(element(
                    body,
                    scope,
                    plan,
                    input,
                    std::slice::from_ref(index),
                    cache,
                )?);
            }
            let value = if rest.is_empty() {
                body.callback(owner, operation, arguments)?
            } else {
                body.in_region(owner, callback, arguments, |body, result| {
                    region_element(body, callback, result, rest)
                })?
            };
            cache.insert(source, value.clone());
            return Ok(value);
        }
    }
    let mut value = body.value(scope, source)?;
    for index in indices {
        value = body.index(value, index.clone())?;
    }
    cache.insert(source, value.clone());
    Ok(value)
}

fn region_element(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    source: Value,
    indices: &[Typed],
) -> Result<Typed, OptimizeError> {
    if let Some(actual) = body.compiler.facts.alias(source) {
        return region_element(body, scope, actual, indices);
    }
    if let Some(callee) = body.compiler.facts.lookup("PlanCallee", (source,)) {
        let Some(fields) = body.compiler.facts.enode("SourceInvocation", source) else {
            return Err(error("planned callback call has no invocation"));
        };
        let mut arguments = Vec::new();
        for argument in body.compiler.facts.vector(fields[2])? {
            arguments.push(body.value(scope, argument)?);
        }
        return body.in_region(scope, callee, arguments, |body, result| {
            region_element(body, callee, result, indices)
        });
    }
    let Some(plan) = body.compiler.facts.operation(source).and_then(|op| body.compiler.facts.group(op))
    else {
        return Err(error("nested domain has no selected fusion plan"));
    };
    element(body, scope, plan, source, indices, &mut LookupMap::default())
}

pub(super) fn write_arrays(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operation: Value,
    plan: Value,
    results: &[(String, i64, Value)],
    indices: &[Typed],
    cache: &mut LookupMap<Value, Typed>,
) -> Result<(), OptimizeError> {
    let mut position = 0;
    for (role, _, source) in results {
        if role != "array" {
            continue;
        }
        if let Some(output) = body.slot(scope, operation, "output", position, 2)? {
            let value = element(body, scope, plan, *source, indices, cache)?;
            body.store_at(output, indices, value)?;
        }
        position += 1;
    }
    Ok(())
}
