//! Expand scheduled phases straight into SSA instructions and structured loops.

use super::super::{builder_error, error, Body, OptimizeError, Typed};
use super::{element, invocation, store};
use crate::op::{BinaryOperator, OpTag};
use crate::ssa::types::{AtomicOp, InstKind};
use crate::types::{self, Type, TypeExt, TypeName};
use crate::LookupMap;
use egglog_engine::Value;

fn destination_value(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operation: Value,
) -> Result<Typed, OptimizeError> {
    let Some(destination) = body.compiler.facts.lookup("SelectedDestination", (operation,)) else {
        return Err(error("update has no selected destination"));
    };
    if let Some(fields) = body.compiler.facts.enode("DestinationBuffer", destination) {
        return body.resource(scope, fields[0], 3);
    }
    let Some(fields) = body.compiler.facts.enode("DestinationValue", destination) else {
        return Err(error("unknown selected update destination"));
    };
    let Some(source) = body.compiler.plan.expr(fields[0]) else {
        return Err(error("selected destination has no source value"));
    };
    body.value(scope, source)
}

pub(super) fn indexed(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    phase_owner: Value,
    phase: &str,
    phase_extent: Value,
    phase_width: u32,
    plan: Value,
) -> Result<(), OptimizeError> {
    let Some(destination) = body.compiler.facts.destination(phase_owner) else {
        return Err(error("indexed destination missing"));
    };
    let output = destination_value(body, scope, phase_owner)?;
    let serial = phase == "ordered";
    let domain = if serial {
        let Some(domain) = body.compiler.plan.domain(phase_owner) else {
            return Err(error("ordered operation domain missing"));
        };
        domain
    } else {
        phase_extent
    };
    let n = body.extent(scope, domain)?;
    let (start, step) = if serial {
        (
            body.literal("0", &types::i32())?,
            body.literal("1", &types::i32())?,
        )
    } else {
        invocation(body, phase_width)?
    };
    body.counted(start, n, step, vec![], |body, index, _| {
        if phase == "initialize" {
            let original = body.value(scope, destination)?;
            let value = body.index(original, index.clone())?;
            store(body, output, index, value)?;
            return Ok(vec![]);
        }
        update(
            body,
            scope,
            phase_owner,
            phase,
            plan,
            output,
            index,
            &mut LookupMap::default(),
        )?;
        Ok(vec![])
    })?;
    Ok(())
}

pub(super) fn update(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    phase_owner: Value,
    phase: &str,
    plan: Value,
    output: Typed,
    index: Typed,
    cache: &mut LookupMap<Value, Typed>,
) -> Result<(), OptimizeError> {
    let kind = body.compiler.facts.operation_kind(phase_owner)?;
    let inputs = body.compiler.facts.inputs(phase_owner)?;
    let mut arguments = Vec::new();
    for &(_, input) in &inputs {
        arguments.push(element(body, scope, plan, input, index.clone(), cache)?);
    }
    let (key, value) = if kind == "scatter" {
        let pair = body.callback(scope, phase_owner, arguments)?;
        (body.field(pair.clone(), 0)?, body.field(pair, 1)?)
    } else {
        if arguments.len() != 2 {
            return Err(error("indexed reducer requires keys and values"));
        }
        (arguments.remove(0), arguments.remove(0))
    };
    let zero = body.literal("0", &key.ty)?;
    let length = body.length(output.clone())?;
    let nonnegative = body.binary(BinaryOperator::GreaterEqual, key.clone(), zero)?;
    let below = body.binary(BinaryOperator::Less, key.clone(), length)?;
    let valid = body.binary(BinaryOperator::LogicalAnd, nonnegative, below)?;
    body.when(valid, |body| {
        if phase == "atomic" {
            let (place, ty) = body.index_place(output, key)?;
            let value = body.cast(value, &ty)?;
            let Some(update) = body.compiler.plan.atomic(phase_owner) else {
                return Err(error("missing atomic"));
            };
            if update != AtomicOp::CompareExchange {
                body.builder
                    .push_inst(
                        InstKind::Atomic {
                            place,
                            op: update,
                            values: vec![value.value],
                        },
                        ty,
                    )
                    .map_err(builder_error)?;
                return Ok(());
            }
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
            // A compare-exchange loop implements every associative i32/u32
            // callback while preserving the chosen atomic execution domain.
            let old = Typed {
                value: old.into(),
                ty: ty.clone(),
            };
            let state_ty = types::tuple(vec![ty.clone(), types::bool_type()]);
            let done = body.op(OpTag::Bool(false), vec![], types::bool_type())?;
            let initial = body.op(OpTag::Tuple(2), vec![old, done], state_ty.clone())?;
            body.retry(initial, |body, state| {
                let old = body.field(state, 0)?;
                let next = body.callback(scope, phase_owner, vec![old.clone(), value.clone()])?;
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
        } else {
            let value = if kind == "reduce-by-index" {
                let previous = body.index(output.clone(), key.clone())?;
                body.callback(scope, phase_owner, vec![previous, value])?
            } else {
                value
            };
            store(body, output, key, value)?;
        }
        Ok(())
    })?;
    Ok(())
}

pub(super) fn buckets(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    phase_owner: Value,
    phase: &str,
    phase_width: u32,
    plan: Value,
) -> Result<(), OptimizeError> {
    let (domain_rank, input_dimensions) = body.compiler.facts.bucket_shape(phase_owner)?;
    let output = destination_value(body, scope, phase_owner)?;
    let Some(counts) = body.slot(
        scope,
        phase_owner,
        "counts",
        0,
        if phase == "clear" { 2 } else { 3 },
    )?
    else {
        return Err(error("bucket counts missing"));
    };
    let Some(overflow) = body.slot(scope, phase_owner, "overflow", 0, 2)? else {
        return Err(error("bucket overflow missing"));
    };
    let zero = body.literal("0", &types::i32())?;
    let one = body.literal("1", &types::i32())?;
    let serial = phase == "ordered";
    if serial || phase == "clear" {
        let count = body.length(output.clone())?;
        let (start, step) =
            if serial { (zero.clone(), one.clone()) } else { invocation(body, phase_width)? };
        let first = body.binary(BinaryOperator::Equal, start.clone(), zero.clone())?;
        body.when(first, |body| {
            store(body, overflow.clone(), zero.clone(), zero.clone())
        })?;
        body.counted(start, count, step, vec![], |body, index, _| {
            store(body, counts.clone(), index, zero.clone())?;
            Ok(vec![])
        })?;
        if !serial {
            return Ok(());
        }
    }
    bucket_updates(
        body,
        scope,
        phase_owner,
        plan,
        output,
        counts,
        overflow,
        &input_dimensions,
        domain_rank,
        serial,
        phase_width,
    )
}

pub(in crate::egglog::to_ssa) fn bucket_updates(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operation: Value,
    plan: Value,
    output: Typed,
    counts: Typed,
    overflow: Typed,
    input_dimensions: &[Vec<usize>],
    domain_rank: usize,
    serial: bool,
    width: u32,
) -> Result<(), OptimizeError> {
    let zero = body.literal("0", &types::i32())?;
    let one = body.literal("1", &types::i32())?;
    let inputs = body.compiler.facts.inputs(operation)?;
    let mut dimensions = vec![None; domain_rank];
    for axes in input_dimensions {
        for &axis in axes {
            let n = if let Some(extent) = body.compiler.plan.bucket_axis(operation, axis as i64) {
                body.extent(scope, extent)?
            } else {
                return Err(error("bucket axis has no planned extent"));
            };
            let Some(slot) = dimensions.get_mut(axis) else {
                return Err(error("bucket axis out of range"));
            };
            *slot = Some(n);
        }
    }
    let dimensions = dimensions
        .into_iter()
        .map(|dimension| {
            let Some(dimension) = dimension else {
                return Err(error("unbound bucket dimension"));
            };
            Ok(dimension)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut n = one.clone();
    for dimension in &dimensions {
        n = body.binary(BinaryOperator::Multiply, n, dimension.clone())?;
    }
    let capacity = match output.ty.elem_type().and_then(|row| row.array_size()) {
        Some(Type::Constructed(TypeName::Size(n), _)) => *n,
        _ => return Err(error("bucket capacity must be static")),
    };
    let capacity = body.literal(
        &capacity.to_string(),
        &Type::Constructed(TypeName::UInt(32), vec![]),
    )?;
    let bucket_count = body.length(output.clone())?;
    let (start, step) = if serial { (zero.clone(), one.clone()) } else { invocation(body, width)? };
    body.counted(start, n, step, vec![], |body, index, _| {
        let mut coordinates = vec![zero.clone(); dimensions.len()];
        let mut remainder = index;
        for i in (0..dimensions.len()).rev() {
            coordinates[i] = body.binary(
                BinaryOperator::Remainder,
                remainder.clone(),
                dimensions[i].clone(),
            )?;
            remainder = body.binary(BinaryOperator::Divide, remainder, dimensions[i].clone())?;
        }
        let mut args = Vec::new();
        let mut cache = LookupMap::default();
        for ((_, source), axes) in inputs.iter().zip(input_dimensions) {
            let Some((&first, rest)) = axes.split_first() else {
                args.push(body.value(scope, *source)?);
                continue;
            };
            let mut value = element(body, scope, plan, *source, coordinates[first].clone(), &mut cache)?;
            for &axis in rest {
                value = body.index(value, coordinates[axis].clone())?;
            }
            args.push(value);
        }
        let emission = body.callback(scope, operation, args)?;
        let active = body.field(emission.clone(), 0)?;
        body.when(active, |body| {
            let key = body.field(emission.clone(), 1)?;
            let positive = body.binary(BinaryOperator::GreaterEqual, key.clone(), zero.clone())?;
            let below = body.binary(BinaryOperator::Less, key.clone(), bucket_count.clone())?;
            let valid = body.binary(BinaryOperator::LogicalAnd, positive, below)?;
            let succeeded = body.branch(
                scope,
                valid,
                |body| {
                    let slot = if serial {
                        let slot = body.index(counts.clone(), key.clone())?;
                        let next = body.binary(BinaryOperator::Add, slot.clone(), one.clone())?;
                        store(body, counts.clone(), key.clone(), next)?;
                        slot
                    } else {
                        let (place, ty) = body.index_place(counts.clone(), key.clone())?;
                        let one = body.cast(one.clone(), &ty)?;
                        let value = body
                            .builder
                            .push_inst(
                                InstKind::Atomic {
                                    place,
                                    op: AtomicOp::Add,
                                    values: vec![one.value],
                                },
                                ty.clone(),
                            )
                            .map_err(builder_error)?;
                        Typed {
                            value: value.into(),
                            ty,
                        }
                    };
                    let room = body.binary(BinaryOperator::Less, slot.clone(), capacity.clone())?;
                    body.when(room.clone(), |body| {
                        let (row, row_ty) = body.index_place(output.clone(), key)?;
                        let Some(ty) = row_ty.elem_type().cloned() else {
                            return Err(error("bucket row has no element"));
                        };
                        let place = body.builder.new_place(ty.clone());
                        body.builder
                            .push_void_inst(InstKind::PlaceIndex {
                                place: row,
                                index: slot.value,
                                result: place,
                            })
                            .map_err(builder_error)?;
                        let value = body.field(emission.clone(), 2)?;
                        let value = body.cast(value, &ty)?;
                        body.builder
                            .push_void_inst(InstKind::Store {
                                place,
                                value: value.value,
                            })
                            .map_err(builder_error)?;
                        Ok(())
                    })?;
                    Ok(room)
                },
                |body| body.op(OpTag::Bool(false), vec![], types::bool_type()),
                None,
            )?;
            let no = body.op(OpTag::Bool(false), vec![], types::bool_type())?;
            let failed = body.binary(BinaryOperator::Equal, succeeded, no)?;
            body.when(failed, |body| {
                if serial {
                    store(body, overflow.clone(), zero.clone(), one.clone())?;
                } else {
                    let (place, ty) = body.index_place(overflow.clone(), zero.clone())?;
                    let one = body.cast(one.clone(), &ty)?;
                    body.builder
                        .push_inst(
                            InstKind::Atomic {
                                place,
                                op: AtomicOp::Exchange,
                                values: vec![one.value],
                            },
                            ty,
                        )
                        .map_err(builder_error)?;
                }
                Ok(())
            })
        })?;
        Ok(vec![])
    })?;
    Ok(())
}
