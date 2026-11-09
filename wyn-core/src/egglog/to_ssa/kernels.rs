//! Expand scheduled phases straight into SSA instructions and structured loops.

use super::{error, Body, OptimizeError, Typed};
use crate::builtins::catalog;
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::types::{self, Type, TypeName};
use crate::LookupMap;
use egglog_engine::Value;

mod contribution;
mod coordinates;
mod loops;
mod maps;
mod screma;
mod updates;
use screma::screma;

pub(super) fn emit(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    phase_owner: Value,
    phase: &str,
    phase_width: u32,
) -> Result<(), OptimizeError> {
    if matches!(phase, "loop_enter" | "loop_test" | "loop_step" | "loop_exit") {
        return loops::emit(body, scope, phase_owner, phase, phase_width);
    }
    if phase == "view" {
        let source = body.compiler.facts.source(phase_owner).ok_or_else(|| error("view source missing"))?;
        let value = body.local(scope, source)?;
        let length = body.length(value.clone())?;
        if body.compiler.facts.loops(source).is_some() {
            // Published views start at zero even when the loop finishes with a slice.
            let storage = body
                .compiler
                .facts
                .lookup("ArrayStorage", (source,))
                .ok_or_else(|| error("loop storage missing"))?;
            let (binding, element, _) =
                body.compiler.bindings.buffer(storage)?.ok_or_else(|| error("loop allocation missing"))?;
            let element = body.compiler.facts.physical_type(element, true)?;
            let output = body.op(
                OpTag::StorageView(PureViewSource::Storage(binding)),
                vec![Body::number(0), length.clone()],
                Body::view_type(&element, crate::types::buffer_tag(binding)),
            )?;
            body.copy_array(output, value, length.clone(), &[])?;
        }
        let slot = body
            .compiler
            .facts
            .constructor("ViewLength", (source,))
            .ok_or_else(|| error("view length slot missing"))?;
        let slot = body.resource(scope, slot, 2)?;
        return body.store(slot, Body::number(0), length);
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
                let zero = body.literal(0, &types::i32())?;
                body.store(output, zero, value)?;
            }
        }
        return Ok(());
    }
    let Some(plan) = body.compiler.facts.group(phase_owner) else {
        return Err(error("kernel has no fusion plan"));
    };
    let results = body.compiler.facts.array_outputs(plan)?;
    match phase {
        "elements" => maps::emit(body, scope, phase_owner, phase, phase_width, plan, &results),
        "initialize" => updates::initialize(body, scope, phase_owner, phase, phase_width),
        "ordered" | "local" => {
            updates::initialize(body, scope, phase_owner, phase, phase_width)?;
            if body.compiler.facts.accumulators(plan)?.is_empty() {
                maps::emit(body, scope, phase_owner, phase, phase_width, plan, &results)
            } else {
                screma(body, scope, phase_owner, phase, phase_width, plan, &results)
            }
        }
        "compact" => screma(body, scope, phase_owner, phase, phase_width, plan, &results),
        "chunks" | "combine" | "offsets" => {
            screma(body, scope, phase_owner, phase, phase_width, plan, &results)
        }
        _ => Err(error(format!("scheduled {} kernel is not implemented", phase))),
    }
}

fn iteration(
    body: &mut Body<'_, '_, '_>,
    phase: &str,
    width: u32,
) -> Result<(Typed, Typed), OptimizeError> {
    match phase {
        "ordered" => Ok((Body::number(0), Body::number(1))),
        "local" => Ok((body.local_id()?, Body::number(width))),
        _ => invocation(body, width),
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
            let stride = body.literal((stride) as i64, &uint)?;
            let offset = body.binary(BinaryOperator::Multiply, coordinate, stride)?;
            start = body.binary(BinaryOperator::Add, start, offset)?;
        }
        let groups = body.literal((u64::from(x) * u64::from(y) * u64::from(z)) as i64, &uint)?;
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
    body.invocation_local(|body| element_local(body, scope, plan, source, indices, cache))
}

fn element_local(
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
    if body.compiler.facts.contains("ArrayStreamProjection", (source,)) {
        let Some((parent, field)) = body.compiler.facts.projection(source) else {
            return Err(error("array stream projection has no parent"));
        };
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
        if body.compiler.facts.contains("PlanMember", (plan, operation))
            && body
                .compiler
                .facts
                .lookup("PlanMapping", (operation,))
                .is_some_and(|mapping| body.compiler.program.graph.value_to_base::<bool>(mapping))
        {
            let Some(traversal) =
                body.compiler.facts.lookup("ElementTraversal", (source, indices.len() as i64))
            else {
                return Err(error("map has no selected callback traversal"));
            };
            let value = emit_traversal(
                body,
                scope,
                traversal,
                indices,
                cache,
                &mut |body, scope, leaf| {
                    let Some(fields) = body
                        .compiler
                        .facts
                        .enode("TraversalValue", leaf)
                        .or_else(|| body.compiler.facts.enode("TraversalResult", leaf))
                    else {
                        return Err(error("element traversal must yield a value"));
                    };
                    body.value(scope, fields[0])
                },
            )?;
            cache.insert(source, value.clone());
            return Ok(value);
        }
    }
    // A stored boundary is already evaluated; distribute indexing only through
    // unmaterialized choices and fields, preserving each arm's lexical scope.
    if body.compiler.facts.access(source).is_none() {
        if let Some((parent, field)) = body.compiler.facts.projection(source) {
            if let Some(actual) = body.compiler.facts.source_field(parent, field)? {
                if actual != source {
                    return element(body, scope, plan, actual, indices, cache);
                }
            }
        }
        if let Some((yes, no)) = body.compiler.facts.branches(source) {
            let (Some(condition), Some(a), Some(b)) = (
                body.compiler.facts.branch_condition(source),
                body.compiler.facts.result(yes),
                body.compiler.facts.result(no),
            ) else {
                return Err(error("array choice has incomplete regions"));
            };
            let condition = body.value(scope, condition)?;
            return body.branch(
                scope,
                condition,
                |body| element(body, yes, plan, a, indices, &mut cache.clone()),
                |body| element(body, no, plan, b, indices, &mut cache.clone()),
                Some((yes, no)),
            );
        }
    }
    let mut value = body.value(scope, source)?;
    for index in indices {
        value = body.index(value, index.clone())?;
    }
    cache.insert(source, value.clone());
    Ok(value)
}

fn emit_traversal<T>(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    traversal: Value,
    indices: &[Typed],
    cache: &mut LookupMap<Value, Typed>,
    emit: &mut impl FnMut(&mut Body<'_, '_, '_>, Value, Value) -> Result<T, OptimizeError>,
) -> Result<T, OptimizeError> {
    let (owner, region, arguments, next, remaining) =
        if let Some(fields) = body.compiler.facts.enode("TraversalFor", traversal) {
            let Some((index, remaining)) = indices.split_first() else {
                return Err(error("traversal rank exceeds the selected domain"));
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
            (fields[0], fields[1], arguments, fields[4], remaining)
        } else if let Some(fields) = body.compiler.facts.enode("TraversalCall", traversal) {
            let mut arguments = Vec::new();
            for argument in body.compiler.facts.vector(fields[3])? {
                arguments.push(body.value(scope, argument)?);
            }
            (fields[1], fields[2], arguments, fields[4], indices)
        } else if indices.is_empty() {
            return emit(body, scope, traversal);
        } else {
            return Err(error("traversal rank is less than the selected domain"));
        };
    body.in_region(owner, region, arguments, |body, _| {
        emit_traversal(body, region, next, remaining, &mut LookupMap::default(), emit)
    })
}
pub(super) fn write_arrays(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operation: Value,
    plan: Value,
    results: &[(i64, Value)],
    indices: &[Typed],
    cache: &mut LookupMap<Value, Typed>,
) -> Result<(), OptimizeError> {
    write_arrays_at(body, scope, operation, plan, results, indices, indices, cache)
}

fn write_arrays_at(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operation: Value,
    plan: Value,
    results: &[(i64, Value)],
    indices: &[Typed],
    destinations: &[Typed],
    cache: &mut LookupMap<Value, Typed>,
) -> Result<(), OptimizeError> {
    for (position, (slot, source)) in results.iter().enumerate() {
        let output = body.output_target(scope, operation, "output", position as i64)?;
        let mappings = body.compiler.facts.set("MappedOutputs", (operation, *slot));
        if output.is_some() || !mappings.is_empty() {
            let value = element(body, scope, plan, *source, indices, cache)?;
            for destination in mappings {
                body.store_mapped(scope, destination, destinations[0].clone(), value.clone())?;
            }
            if let Some(output) = output {
                output.store(body, destinations, value)?;
            }
        }
    }
    Ok(())
}
