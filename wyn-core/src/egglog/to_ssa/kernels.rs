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
        "elements" | "segments" => maps::emit(
            body,
            scope,
            phase_owner,
            phase == "segments",
            phase_width,
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
                "screma" => screma(body, scope, phase_owner, phase, phase_width, plan, &results, None),
                "filter" => serial_filter(body, scope, phase_owner, plan, &results),
                "indexed" => indexed(body, scope, phase_owner, phase, phase_extent, phase_width, plan),
                "buckets" => buckets(body, scope, phase_owner, phase, phase_width, plan),
                _ => Err(error("unknown ordered algorithm")),
            }
        }
        "compact" => compact(body, scope, phase_owner, phase_width, plan, &results),
        "chunks" | "combine" | "offsets" => {
            screma(body, scope, phase_owner, phase, phase_width, plan, &results, None)
        }
        _ => Err(error(format!("scheduled {} kernel is not implemented", phase))),
    }
}

pub(super) fn invocation(body: &mut Body<'_, '_, '_>, width: u32) -> Result<(Typed, Typed), OptimizeError> {
    let (group, groups) = workgroup(body)?;
    let width = Body::number(width);
    let start = body.binary(BinaryOperator::Multiply, group, width.clone())?;
    let lane = body.local_id()?;
    let start = body.binary(BinaryOperator::Add, start, lane)?;
    let step = body.binary(BinaryOperator::Multiply, groups, width)?;
    Ok((start, step))
}

fn workgroup(body: &mut Body<'_, '_, '_>) -> Result<(Typed, Typed), OptimizeError> {
    let uint = Type::Constructed(TypeName::UInt(32), vec![]);
    let coordinates = body.op(
        OpTag::Intrinsic {
            id: catalog().known().workgroup_id,
            overload_idx: 0,
        },
        vec![],
        types::vec(3, uint.clone()),
    )?;
    let mut start = body.field(coordinates.clone(), 0)?;
    if let Some((x, y, z)) = body.grid {
        for (axis, stride) in [(1, u64::from(x)), (2, u64::from(x) * u64::from(y))] {
            let coordinate = body.field(coordinates.clone(), axis)?;
            let stride = body.literal(&stride.to_string(), &uint)?;
            let offset = body.binary(BinaryOperator::Multiply, coordinate, stride)?;
            start = body.binary(BinaryOperator::Add, start, offset)?;
        }
        let groups = body.literal(&(u64::from(x) * u64::from(y) * u64::from(z)).to_string(), &uint)?;
        return Ok((start, groups));
    }
    let step = body.op(
        OpTag::Intrinsic {
            id: catalog().known().num_workgroups,
            overload_idx: 0,
        },
        vec![],
        uint,
    )?;
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
    let Some(index) = indices.first() else {
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
            let Some(bindings) =
                body.compiler.facts.lookup("MapBindings", (source, 0i64, indices.len() as i64, false))
            else {
                return Err(error("map has no selected callback bindings"));
            };
            let value = emit_bindings(
                body,
                scope,
                bindings,
                indices,
                cache,
                &mut |body, scope, source| body.value(scope, source),
            )?;
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

fn emit_bindings<T>(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    bindings: Value,
    indices: &[Typed],
    cache: &mut LookupMap<Value, Typed>,
    emit: &mut impl FnMut(&mut Body<'_, '_, '_>, Value, Value) -> Result<T, OptimizeError>,
) -> Result<T, OptimizeError> {
    if let Some(fields) = body.compiler.facts.enode("MapValueBinding", bindings) {
        return emit(body, scope, fields[0]);
    }
    let (owner, region, arguments, next) =
        if let Some(fields) = body.compiler.facts.enode("MapIndexBinding", bindings) {
            let axis = body.compiler.facts.unsigned(fields[4], "callback input axis")? as usize;
            let Some(index) = indices.get(axis) else {
                return Err(error("callback input axis exceeds the selected domain"));
            };
            let mut arguments = Vec::new();
            // Cached leaves and input rows are evaluated at different coordinate ranks.
            let mut input_cache = LookupMap::default();
            let cache = if indices.len() == 1 { cache } else { &mut input_cache };
            for input in body.compiler.facts.vector(fields[3])? {
                arguments.push(element(
                    body,
                    scope,
                    fields[2],
                    input,
                    std::slice::from_ref(index),
                    cache,
                )?);
            }
            (fields[0], fields[1], arguments, fields[5])
        } else if let Some(fields) = body.compiler.facts.enode("MapCallBinding", bindings) {
            let mut arguments = Vec::new();
            for argument in body.compiler.facts.vector(fields[2])? {
                arguments.push(body.value(scope, argument)?);
            }
            (fields[0], fields[1], arguments, fields[3])
        } else {
            return Err(error("unknown selected callback binding"));
        };
    body.in_region(owner, region, arguments, |body, _| {
        emit_bindings(body, region, next, indices, &mut LookupMap::default(), emit)
    })
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
