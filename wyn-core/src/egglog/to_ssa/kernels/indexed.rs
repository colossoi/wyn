//! Expand scheduled phases straight into SSA instructions and structured loops.

use super::super::plan::Stage;
use super::super::{builder_error, error, Body, OptimizeError, Typed};
use super::{element, invocation, store};
use crate::op::{BinaryOperator, OpTag};
use crate::ssa::types::{AtomicOp, InstKind};
use crate::tlc::{SoacOp, TermKind};
use crate::types::{self, Type, TypeExt, TypeName};
use crate::LookupMap;
use egglog_engine::Value;

pub(super) fn indexed(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    stage: &Stage,
    plan: Value,
) -> Result<(), OptimizeError> {
    let Some(source) = body.compiler.plan.source(stage.operation) else {
        return Err(error("indexed source missing"));
    };
    let Some(&(term, _)) = body.compiler.program.identities.origins.get(&source) else {
        return Err(error("indexed source term missing"));
    };
    let Some(destination) = body.compiler.facts.destination(stage.operation) else {
        return Err(error("indexed destination missing"));
    };
    let output = if body.compiler.plan.slot(stage.operation, "output", 0).is_some() {
        let Some(view) = body.slot(
            scope,
            stage.operation,
            "output",
            0,
            if stage.phase == "initialize" { 2 } else { 3 },
        )?
        else {
            return Err(error("indexed output is not materialized"));
        };
        view
    } else if let Some(resource) = body.compiler.plan.value_ref(destination) {
        body.resource(scope, resource, 3)?
    } else {
        body.value(scope, destination)?
    };
    let serial = stage.phase == "ordered";
    let domain = if serial {
        let Some(domain) = body.compiler.plan.domain(stage.operation) else {
            return Err(error("ordered operation domain missing"));
        };
        domain
    } else {
        stage.extent
    };
    let n = body.extent(scope, domain)?;
    let (start, step) = if serial {
        (
            body.literal("0", &types::i32())?,
            body.literal("1", &types::i32())?,
        )
    } else {
        invocation(body, stage.width)?
    };
    if serial
        && matches!(term.kind, TermKind::Soac(SoacOp::Scatter { .. }))
        && body.compiler.plan.slot(stage.operation, "output", 0).is_some()
    {
        let original = body.value(scope, destination)?;
        let count = body.length(original.clone())?;
        body.copy_array(output.clone(), original, count)?;
    }
    body.counted(start, n, step, vec![], |body, index, _| {
        if stage.phase == "initialize" {
            let original = body.value(scope, destination)?;
            let value = body.index(original, index.clone())?;
            store(body, output, index, value)?;
            return Ok(vec![]);
        }
        update(body, scope, stage, plan, output, index, &mut LookupMap::default())?;
        Ok(vec![])
    })?;
    Ok(())
}

pub(super) fn update(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    stage: &Stage,
    plan: Value,
    output: Typed,
    index: Typed,
    cache: &mut LookupMap<Value, Typed>,
) -> Result<(), OptimizeError> {
    let Some(source) = body.compiler.plan.source(stage.operation) else {
        return Err(error("indexed source missing"));
    };
    let Some(&(term, _)) = body.compiler.program.identities.origins.get(&source) else {
        return Err(error("indexed source term missing"));
    };
    let inputs = body.compiler.facts.inputs(stage.operation);
    let mut arguments = Vec::new();
    for &(_, input) in &inputs {
        arguments.push(element(body, scope, plan, input, index.clone(), cache)?);
    }
    let (key, value) = if matches!(term.kind, TermKind::Soac(SoacOp::Scatter { .. })) {
        let pair = body.callback(scope, stage.operation, arguments)?;
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
        if stage.phase == "atomic" {
            let (place, ty) = body.index_place(output, key)?;
            let value = body.cast(value, &ty)?;
            let Some(update) = body.compiler.plan.atomic(stage.operation) else {
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
                let next = body.callback(scope, stage.operation, vec![old.clone(), value.clone()])?;
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
            let value = if matches!(term.kind, TermKind::Soac(SoacOp::ReduceByIndex { .. })) {
                let previous = body.index(output.clone(), key.clone())?;
                body.callback(scope, stage.operation, vec![previous, value])?
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
    stage: &Stage,
    plan: Value,
) -> Result<(), OptimizeError> {
    let Some(source) = body.compiler.plan.source(stage.operation) else {
        return Err(error("missing source"));
    };
    let Some(&(term, _)) = body.compiler.program.identities.origins.get(&source) else {
        return Err(error("bucket source missing"));
    };
    let TermKind::Soac(SoacOp::BucketScatter {
        input_dimensions,
        domain_rank,
        ..
    }) = &term.kind
    else {
        return Err(error("invalid bucket recipe"));
    };
    let Some(destination) = body.compiler.facts.destination(stage.operation) else {
        return Err(error("missing destination"));
    };
    let output = if let Some(resource) = body.compiler.plan.value_ref(destination) {
        body.resource(scope, resource, 3)?
    } else {
        body.value(scope, destination)?
    };
    let Some(counts) = body.slot(
        scope,
        stage.operation,
        "counts",
        0,
        if stage.phase == "clear" { 2 } else { 3 },
    )?
    else {
        return Err(error("bucket counts missing"));
    };
    let Some(overflow) = body.slot(scope, stage.operation, "overflow", 0, 2)? else {
        return Err(error("bucket overflow missing"));
    };
    let zero = body.literal("0", &types::i32())?;
    let one = body.literal("1", &types::i32())?;
    let serial = stage.phase == "ordered";
    if serial || stage.phase == "clear" {
        let count = body.length(output.clone())?;
        let (start, step) =
            if serial { (zero.clone(), one.clone()) } else { invocation(body, stage.width)? };
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
        stage.operation,
        plan,
        output,
        counts,
        overflow,
        input_dimensions,
        *domain_rank,
        serial,
        stage.width,
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
    input_dimensions: &[Vec<u8>],
    domain_rank: u8,
    serial: bool,
    width: u32,
) -> Result<(), OptimizeError> {
    let zero = body.literal("0", &types::i32())?;
    let one = body.literal("1", &types::i32())?;
    let inputs = body.compiler.facts.inputs(operation);
    let mut dimensions = vec![None; usize::from(domain_rank)];
    for ((_, source), axes) in inputs.iter().zip(input_dimensions) {
        let Some(ty) = body.compiler.facts.source_type(*source) else {
            return Err(error("bucket input type missing"));
        };
        let mut ty = ty.clone();
        if !body.compiler.facts.operation(*source).is_some() {
            ty = body.value(scope, *source)?.ty;
        }
        for (i, &axis) in axes.iter().enumerate() {
            while let Type::Constructed(TypeName::Tuple(_), fields) = &ty {
                let Some(first) = fields.first() else {
                    return Err(error("empty bucket input"));
                };
                ty = first.clone();
            }
            let n = if let Some(extent) = body.compiler.plan.bucket_axis(operation, i64::from(axis)) {
                body.extent(scope, extent)?
            } else if i == 0 {
                if let Some(op) = body.compiler.facts.operation(*source) {
                    if body.compiler.plan.member(plan, op) {
                        let Some(domain) = body.compiler.plan.domain(operation) else {
                            return Err(error("bucket operation domain missing"));
                        };
                        body.extent(scope, domain)?
                    } else {
                        let array = body.value(scope, *source)?;
                        body.length(array)?
                    }
                } else {
                    let array = body.value(scope, *source)?;
                    body.length(array)?
                }
            } else if let Some(Type::Constructed(TypeName::Size(n), _)) = ty.array_size() {
                body.literal(&n.to_string(), &types::i32())?
            } else {
                return Err(error("bucket inner dimension has no capacity"));
            };
            let Some(slot) = dimensions.get_mut(usize::from(axis)) else {
                return Err(error("bucket axis out of range"));
            };
            *slot = Some(n);
            if i + 1 < axes.len() {
                let Some(element) = ty.elem_type() else {
                    return Err(error("bucket input rank mismatch"));
                };
                ty = element.clone();
            }
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
            let mut value = element(
                body,
                scope,
                plan,
                *source,
                coordinates[usize::from(first)].clone(),
                &mut cache,
            )?;
            for &axis in rest {
                value = body.index(value, coordinates[usize::from(axis)].clone())?;
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
