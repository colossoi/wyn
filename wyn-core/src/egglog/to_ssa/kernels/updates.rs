//! Output actions share bounds checking and serial/atomic accumulation.
use super::super::{builder_error, error, Body, OptimizeError, Typed};
use super::{coordinates, element, invocation};
use crate::egglog::facts::OutputAction;
use crate::op::{BinaryOperator, OpTag};
use crate::ssa::types::{AtomicOp, InstKind};
use crate::types::{self, Type, TypeExt, TypeName};
use crate::LookupMap;
use egglog_engine::Value;

#[derive(Clone, Copy)]
enum Update {
    Store,
    Add,
    Combine(Value),
    Atomic(AtomicOp),
    AtomicCombine(Value),
}

/// Add/combine return the previous value, allowing the same primitive to reserve
/// a slot. Store returns its argument and never reads uninitialized storage.
fn apply(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    output: Typed,
    key: Typed,
    value: Typed,
    update: Update,
) -> Result<Typed, OptimizeError> {
    match update {
        Update::Store => {
            body.store(output, key, value.clone())?;
            Ok(value)
        }
        Update::Add | Update::Combine(_) => {
            let old = body.index(output.clone(), key.clone())?;
            let next = match update {
                Update::Combine(operation) => body.callback(scope, operation, vec![old.clone(), value])?,
                _ => body.binary(BinaryOperator::Add, old.clone(), value)?,
            };
            body.store(output, key, next)?;
            Ok(old)
        }
        Update::Atomic(op) => {
            let (place, ty) = body.index_place(output, key)?;
            let value = body.cast(value, &ty)?;
            let result = body
                .builder
                .push_inst(
                    InstKind::Atomic {
                        place,
                        op,
                        values: vec![value.value],
                    },
                    ty.clone(),
                )
                .map_err(builder_error)?;
            Ok(Typed {
                value: result.into(),
                ty,
            })
        }
        Update::AtomicCombine(operation) => {
            let (place, ty) = body.index_place(output, key)?;
            let value = body.cast(value, &ty)?;
            let old = body
                .builder
                .push_inst(
                    InstKind::Atomic {
                        place,
                        op: AtomicOp::Load,
                        values: vec![],
                    },
                    ty.clone(),
                )
                .map_err(builder_error)?;
            let old = Typed {
                value: old.into(),
                ty: ty.clone(),
            };
            let state_ty = types::tuple(vec![ty, types::bool_type()]);
            let done = body.op(OpTag::Bool(false), vec![], types::bool_type())?;
            let initial = body.tuple(vec![old, done])?;
            let state = body.retry(initial, |body, state| {
                let old = body.field(state, 0)?;
                let next = body.callback(scope, operation, vec![old.clone(), value.clone()])?;
                let result = body
                    .builder
                    .push_inst(
                        InstKind::Atomic {
                            place,
                            op: AtomicOp::CompareExchange,
                            values: vec![old.value, next.value],
                        },
                        state_ty.clone(),
                    )
                    .map_err(builder_error)?;
                Ok(Typed {
                    value: result.into(),
                    ty: state_ty.clone(),
                })
            })?;
            body.field(state, 0)
        }
    }
}

fn in_bounds(body: &mut Body<'_, '_, '_>, key: Typed, length: Typed) -> Result<Typed, OptimizeError> {
    let zero = body.literal(0, &key.ty)?;
    let nonnegative = body.binary(BinaryOperator::GreaterEqual, key.clone(), zero)?;
    let below = body.binary(BinaryOperator::Less, key, length)?;
    body.binary(BinaryOperator::LogicalAnd, nonnegative, below)
}

fn destination(body: &mut Body<'_, '_, '_>, scope: Value, owner: Value) -> Result<Typed, OptimizeError> {
    let Some(destination) = body.compiler.facts.lookup("SelectedDestination", (owner,)) else {
        return Err(error("update has no selected destination"));
    };
    if let Some(fields) = body.compiler.facts.enode("DestinationBuffer", destination) {
        return body.resource(scope, fields[0], 3);
    }
    let Some(fields) = body.compiler.facts.enode("DestinationValue", destination) else {
        return Err(error("invalid update destination"));
    };
    body.value(scope, fields[0])
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

pub(super) fn indexed(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    owner: Value,
    phase: &str,
    extent: Value,
    width: u32,
    plan: Value,
) -> Result<(), OptimizeError> {
    let action = body.compiler.facts.action(plan)?;
    let original = match action {
        OutputAction::Write { destination, .. } | OutputAction::Accumulate { destination, .. } => {
            destination
        }
        _ => return Err(error("indexed loop requires an update action")),
    };
    let output = destination(body, scope, owner)?;
    let extent = if phase == "ordered" {
        let Some(extent) = body.compiler.facts.domain(owner) else {
            return Err(error("update domain missing"));
        };
        extent
    } else {
        extent
    };
    let n = body.extent(scope, extent)?;
    let (start, step) = iteration(body, phase, width)?;
    body.counted(start, n, step, vec![], |body, index, _| {
        if phase == "initialize" {
            let original = body.value(scope, original)?;
            let value = body.index(original, index.clone())?;
            body.store(output.clone(), index, value)?;
        } else {
            update(
                body,
                scope,
                owner,
                phase,
                plan,
                output.clone(),
                index,
                &mut LookupMap::default(),
            )?;
        }
        Ok(vec![])
    })?;
    Ok(())
}

pub(super) fn update(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    owner: Value,
    phase: &str,
    plan: Value,
    output: Typed,
    index: Typed,
    cache: &mut LookupMap<Value, Typed>,
) -> Result<(), OptimizeError> {
    let (key, value, update) = match body.compiler.facts.action(plan)? {
        OutputAction::Write { callback, .. } => {
            let mut arguments = Vec::new();
            for (_, input) in body.compiler.facts.inputs(callback)? {
                arguments.push(element(body, scope, plan, input, &[index.clone()], cache)?);
            }
            let pair = body.callback(scope, callback, arguments)?;
            (body.field(pair.clone(), 0)?, body.field(pair, 1)?, Update::Store)
        }
        OutputAction::Accumulate {
            combine,
            keys,
            values,
            ..
        } => {
            let key = element(body, scope, plan, keys, &[index.clone()], cache)?;
            let value = element(body, scope, plan, values, &[index], cache)?;
            let update = if phase == "atomic" {
                let Some(op) = body.compiler.facts.atomic(owner) else {
                    return Err(error("atomic update missing"));
                };
                if op == AtomicOp::CompareExchange {
                    Update::AtomicCombine(combine)
                } else {
                    Update::Atomic(op)
                }
            } else {
                Update::Combine(combine)
            };
            (key, value, update)
        }
        _ => return Err(error("indexed loop requires an update action")),
    };
    let length = body.length(output.clone())?;
    let valid = in_bounds(body, key.clone(), length)?;
    body.when(valid, |body| {
        apply(body, scope, output, key, value, update)?;
        Ok(())
    })
}

pub(super) fn buckets(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    owner: Value,
    phase: &str,
    width: u32,
    plan: Value,
) -> Result<(), OptimizeError> {
    let OutputAction::Reserve { callback } = body.compiler.facts.action(plan)? else {
        return Err(error("reservation loop requires a bucket action"));
    };
    let output = destination(body, scope, owner)?;
    let Some(counts) = body.slot(scope, owner, "counts", 0, if phase == "clear" { 2 } else { 3 })? else {
        return Err(error("reservation counts missing"));
    };
    let Some(overflow) = body.slot(scope, owner, "overflow", 0, 2)? else {
        return Err(error("reservation overflow missing"));
    };
    let zero = body.literal(0, &types::i32())?;
    let one = body.literal(1, &types::i32())?;
    let serial = phase == "ordered";
    if serial || phase == "clear" {
        let count = body.length(output.clone())?;
        let (start, step) = iteration(body, phase, width)?;
        let first = body.binary(BinaryOperator::Equal, start.clone(), zero.clone())?;
        body.when(first, |body| {
            body.store(overflow.clone(), zero.clone(), zero.clone())
        })?;
        body.counted(start, count, step, vec![], |body, index, _| {
            body.store(counts.clone(), index, zero.clone())?;
            Ok(vec![])
        })?;
        if !serial {
            return Ok(());
        }
    }
    let (dimensions, input_dimensions) = body.compiler.facts.domain_shape(owner)?;
    let inputs = body.compiler.facts.inputs(callback)?;
    let dimensions =
        dimensions.iter().map(|&extent| body.extent(scope, extent)).collect::<Result<Vec<_>, _>>()?;
    let n = coordinates::length(body, &dimensions)?;
    let Some(Type::Constructed(TypeName::Size(capacity), _)) =
        output.ty.elem_type().and_then(|row| row.array_size())
    else {
        return Err(error("reservation capacity must be static"));
    };
    let capacity =
        Body::number(u32::try_from(*capacity).map_err(|_| error("reservation capacity exceeds u32"))?);
    let length = body.length(output.clone())?;
    let (start, step) = iteration(body, phase, width)?;
    body.counted(start, n, step, vec![], |body, index, _| {
        let coordinates = coordinates::decode(body, &dimensions, index)?;
        let arguments =
            coordinates::arguments(body, scope, plan, &inputs, &input_dimensions, &coordinates)?;
        let emission = body.callback(scope, callback, arguments)?;
        let active = body.field(emission.clone(), 0)?;
        body.when(active, |body| {
            let key = body.field(emission.clone(), 1)?;
            let valid = in_bounds(body, key.clone(), length.clone())?;
            let succeeded = body.branch(
                scope,
                valid,
                |body| {
                    let update = if serial { Update::Add } else { Update::Atomic(AtomicOp::Add) };
                    let slot = apply(body, scope, counts.clone(), key.clone(), one.clone(), update)?;
                    let room = body.binary(BinaryOperator::Less, slot.clone(), capacity.clone())?;
                    body.when(room.clone(), |body| {
                        let value = body.field(emission.clone(), 2)?;
                        body.store_at(output.clone(), &[key, slot], value)
                    })?;
                    Ok(room)
                },
                |body| body.op(OpTag::Bool(false), vec![], types::bool_type()),
                None,
            )?;
            let no = body.op(OpTag::Bool(false), vec![], types::bool_type())?;
            let failed = body.binary(BinaryOperator::Equal, succeeded, no)?;
            body.when(failed, |body| {
                let update = if serial { Update::Store } else { Update::Atomic(AtomicOp::Exchange) };
                apply(body, scope, overflow.clone(), zero.clone(), one.clone(), update)?;
                Ok(())
            })
        })?;
        Ok(vec![])
    })?;
    Ok(())
}
