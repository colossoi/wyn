// Expand scheduled phases straight into SSA instructions and structured loops.
use super::super::{builder_error, error, Body, OptimizeError, Typed};
use super::{element, indexed, invocation, write_arrays};
use crate::builtins::catalog;
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::ssa::types::InstKind;
use crate::types::{self, Type, TypeName};
use crate::LookupMap;
use egglog_engine::Value;

pub(super) fn screma(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    phase_owner: Value,
    phase: &str,
    phase_width: u32,
    plan: Value,
    results: &[(String, i64, Value)],
) -> Result<(), OptimizeError> {
    let scans: Vec<_> = results.iter().filter(|r| r.0 == "scan").map(|r| r.2).collect();
    let totals: Vec<_> = results.iter().filter(|r| r.0 == "total").map(|r| r.2).collect();
    let mut sources: Vec<_> = scans.iter().chain(&totals).copied().collect();
    let mut initial = Vec::new();
    let mut operators = Vec::new();
    for &source in &sources {
        let Some(operation) = body.compiler.facts.operation(source) else {
            return Err(error("accumulator operation missing"));
        };
        let neutral = body.compiler.facts.neutral(operation)?;
        initial.push(body.value(scope, neutral)?);
        operators.push(operation);
    }
    for (index, filter) in body.compiler.facts.counts(plan)? {
        let index = scans.len() + usize::try_from(index).map_err(|_| error("negative count slot"))?;
        if index > operators.len() {
            return Err(error("noncontiguous count accumulator"));
        }
        let Some(source) = body.compiler.facts.source(filter) else {
            return Err(error("filter accumulator source missing"));
        };
        sources.insert(index, source);
        operators.insert(index, filter);
        initial.insert(index, body.literal("0", &types::i32())?);
    }
    let Some(domain) = body.compiler.facts.domain(phase_owner) else {
        return Err(error("collective domain missing"));
    };
    let n = body.extent(scope, domain)?;
    if phase == "ordered" {
        let zero = body.literal("0", &types::i32())?;
        let one = body.literal("1", &types::i32())?;
        let state = body.counted(zero.clone(), n, one, initial, |body, index, state| {
            let mut cache = LookupMap::default();
            let mut next = Vec::new();
            for (i, &operator) in operators.iter().enumerate() {
                let value = accumulate_at(
                    body,
                    scope,
                    plan,
                    operator,
                    state[i].clone(),
                    index.clone(),
                    &mut cache,
                )?;
                cache.insert(sources[i], value.clone());
                next.push(value);
            }
            write_arrays(body, scope, phase_owner, plan, results, index, &mut cache)?;
            Ok(next)
        })?;
        for (i, value) in state.into_iter().skip(scans.len()).enumerate() {
            if let Some(output) = body.slot(scope, phase_owner, "total", i as i64, 2)? {
                body.store(output, zero.clone(), value)?;
            }
        }
        return Ok(());
    }
    let uint = Type::Constructed(TypeName::UInt(32), vec![]);
    let n = body.cast(n, &uint)?;
    let zero = body.literal("0", &uint)?;
    let one = body.literal("1", &uint)?;
    let Some(key) = body.compiler.facts.constructor("Stage", (phase_owner, "chunks")) else {
        return Err(error("collective chunk schedule missing"));
    };
    let chunks_width = body.compiler.facts.phase_width(key)?;
    let Some((x, y, z)) = body.compiler.facts.dispatch_grid(key)? else {
        return Err(error("collective requires a fixed chunk grid"));
    };
    let groups = u64::from(x) * u64::from(y) * u64::from(z);
    let threads = groups * u64::from(chunks_width);
    if threads > i32::MAX as u64 {
        return Err(error("collective grid exceeds the 32-bit index range"));
    }
    let width = body.literal(&chunks_width.to_string(), &uint)?;
    let lane = body.op(
        OpTag::Intrinsic {
            id: catalog().known().local_id,
            overload_idx: 0,
        },
        vec![],
        uint.clone(),
    )?;
    // Workgroups own contiguous ranges of whole tiles, preserving operand order.
    let extra = body.literal(&(threads - 1).to_string(), &uint)?;
    let threads = body.literal(&threads.to_string(), &uint)?;
    let tiles = body.binary(BinaryOperator::Add, n.clone(), extra)?;
    let tiles = body.binary(BinaryOperator::Divide, tiles, threads)?;
    let span = body.binary(BinaryOperator::Multiply, tiles.clone(), width.clone())?;
    match phase {
        "chunks" => {
            let (thread, _) = invocation(body, phase_width)?;
            let group = body.binary(BinaryOperator::Divide, thread, width.clone())?;
            let base = body.binary(BinaryOperator::Multiply, group.clone(), span.clone())?;
            let state = body.counted(zero.clone(), tiles, one, initial.clone(), |body, tile, carry| {
                let offset = body.binary(BinaryOperator::Multiply, tile, width.clone())?;
                let offset = body.binary(BinaryOperator::Add, base.clone(), offset)?;
                let index = body.binary(BinaryOperator::Add, offset, lane.clone())?;
                let valid = body.binary(BinaryOperator::Less, index.clone(), n.clone())?;
                let tuple_ty = types::tuple(initial.iter().map(|value| value.ty.clone()).collect());
                let incoming = body.branch(
                    scope,
                    valid.clone(),
                    |body| {
                        let mut cache = LookupMap::default();
                        let mut values = Vec::new();
                        for (i, &operator) in operators.iter().enumerate() {
                            let value = accumulate_at(
                                body,
                                scope,
                                plan,
                                operator,
                                initial[i].clone(),
                                index.clone(),
                                &mut cache,
                            )?;
                            values.push(value);
                        }
                        if scans.is_empty() {
                            write_arrays(
                                body,
                                scope,
                                phase_owner,
                                plan,
                                results,
                                index.clone(),
                                &mut cache,
                            )?;
                        }
                        body.op(OpTag::Tuple(values.len()), values, tuple_ty.clone())
                    },
                    |body| body.op(OpTag::Tuple(initial.len()), initial.clone(), tuple_ty.clone()),
                    None,
                )?;
                let values = (0..operators.len())
                    .map(|i| body.field(incoming.clone(), i))
                    .collect::<Result<Vec<_>, _>>()?;
                let (prefixes, totals) = if scans.is_empty() {
                    (
                        Vec::new(),
                        workgroup_reduce(
                            body,
                            scope,
                            &operators,
                            values,
                            &initial,
                            lane.clone(),
                            phase_width,
                        )?,
                    )
                } else {
                    let (prefixes, _, totals) = workgroup_scan(
                        body,
                        scope,
                        &operators,
                        values,
                        &initial,
                        lane.clone(),
                        phase_width,
                    )?;
                    (prefixes, totals)
                };
                body.when(valid, |body| {
                    for (i, prefix) in prefixes.iter().take(scans.len()).enumerate() {
                        if let Some(output) = body.slot(scope, phase_owner, "prefix", i as i64, 2)? {
                            let value = combine_accumulator(
                                body,
                                scope,
                                operators[i],
                                carry[i].clone(),
                                prefix.clone(),
                            )?;
                            body.store(output, index.clone(), value)?;
                        }
                    }
                    Ok(())
                })?;
                operators
                    .iter()
                    .enumerate()
                    .map(|(i, &operator)| {
                        combine_accumulator(body, scope, operator, carry[i].clone(), totals[i].clone())
                    })
                    .collect()
            })?;
            let first = body.binary(BinaryOperator::Equal, lane, zero)?;
            body.when(first, |body| {
                for (i, value) in state.into_iter().enumerate() {
                    if let Some(output) = body.slot(scope, phase_owner, "partial", i as i64, 2)? {
                        body.store(output, group.clone(), value)?;
                    }
                }
                Ok(())
            })?;
        }
        "combine" => {
            let groups = body.literal(&groups.to_string(), &uint)?;
            let width = body.literal(&phase_width.to_string(), &uint)?;
            let totals = body.counted(
                zero.clone(),
                groups.clone(),
                width.clone(),
                initial.clone(),
                |body, base, carry| {
                    let index = body.binary(BinaryOperator::Add, base, lane.clone())?;
                    let valid = body.binary(BinaryOperator::Less, index.clone(), groups)?;
                    let mut values = Vec::new();
                    for (i, neutral) in initial.iter().enumerate() {
                        values.push(body.branch(
                            scope,
                            valid.clone(),
                            |body| {
                                let Some(partial) =
                                    body.slot(scope, phase_owner, "partial", i as i64, 1)?
                                else {
                                    return Err(error("collective partial is not materialized"));
                                };
                                let value = body.index(partial, index.clone())?;
                                body.cast(value, &neutral.ty)
                            },
                            |_| Ok(neutral.clone()),
                            None,
                        )?);
                    }
                    let (offsets, totals) = if scans.is_empty() {
                        (
                            Vec::new(),
                            workgroup_reduce(
                                body,
                                scope,
                                &operators,
                                values,
                                &initial,
                                lane.clone(),
                                phase_width,
                            )?,
                        )
                    } else {
                        let (_, offsets, totals) = workgroup_scan(
                            body,
                            scope,
                            &operators,
                            values,
                            &initial,
                            lane.clone(),
                            phase_width,
                        )?;
                        (offsets, totals)
                    };
                    body.when(valid, |body| {
                        for (i, offset) in offsets.into_iter().take(scans.len()).enumerate() {
                            if let Some(output) = body.slot(scope, phase_owner, "offset", i as i64, 2)? {
                                let value = combine_accumulator(
                                    body,
                                    scope,
                                    operators[i],
                                    carry[i].clone(),
                                    offset,
                                )?;
                                body.store(output, index.clone(), value)?;
                            }
                        }
                        Ok(())
                    })?;
                    operators
                        .iter()
                        .enumerate()
                        .map(|(i, &operator)| {
                            combine_accumulator(body, scope, operator, carry[i].clone(), totals[i].clone())
                        })
                        .collect()
                },
            )?;
            let first = body.binary(BinaryOperator::Equal, lane, zero.clone())?;
            body.when(first, |body| {
                for (i, value) in totals.into_iter().skip(scans.len()).enumerate() {
                    if let Some(output) = body.slot(scope, phase_owner, "total", i as i64, 2)? {
                        body.store(output, zero.clone(), value)?;
                    }
                }
                Ok(())
            })?;
        }
        "offsets" => {
            let (start, step) = invocation(body, phase_width)?;
            body.counted(start, n, step, vec![], |body, index, _| {
                let chunk = body.binary(BinaryOperator::Divide, index.clone(), span)?;
                let mut cache = LookupMap::default();
                for (i, &source) in scans.iter().enumerate() {
                    let Some(prefix) = body.slot(scope, phase_owner, "prefix", i as i64, 1)? else {
                        return Err(error("scan prefix is not materialized"));
                    };
                    let Some(offset) = body.slot(scope, phase_owner, "offset", i as i64, 1)? else {
                        return Err(error("scan offset is not materialized"));
                    };
                    let a = body.index(offset, chunk.clone())?;
                    let b = body.index(prefix, index.clone())?;
                    let value = combine_accumulator(body, scope, operators[i], a, b)?;
                    cache.insert(source, value);
                }
                if body.compiler.facts.destination(phase_owner).is_some() {
                    let Some(output) = body.slot(scope, phase_owner, "output", 0, 2)? else {
                        return Err(error("scan output action has no destination"));
                    };
                    indexed::update(body, scope, phase_owner, phase, plan, output, index, &mut cache)?;
                } else {
                    write_arrays(body, scope, phase_owner, plan, results, index, &mut cache)?;
                }
                Ok(vec![])
            })?;
        }
        _ => return Err(error("invalid collective phase")),
    }
    Ok(())
}

/// Reduce adjacent contiguous ranges in source order. A single shared bank is
/// sufficient: a round's right-hand ranges never write during that round.
fn workgroup_reduce(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operators: &[Value],
    values: Vec<Typed>,
    neutral: &[Typed],
    lane: Typed,
    width: u32,
) -> Result<Vec<Typed>, OptimizeError> {
    assert!(
        width.is_power_of_two(),
        "workgroup reduction width must be a power of two: {width}"
    );
    let zero = body.literal("0", &lane.ty)?;
    let length = body.literal(&width.to_string(), &lane.ty)?;
    let mut shared = Vec::new();
    for (i, value) in values.into_iter().enumerate() {
        let buffer = body.op(
            OpTag::StorageView(PureViewSource::Workgroup {
                id: i as u32,
                count: width,
            }),
            vec![zero.clone(), length.clone()],
            Body::view_type(&value.ty, types::no_buffer()),
        )?;
        body.store(buffer.clone(), lane.clone(), value)?;
        shared.push(buffer);
    }
    body.builder.push_void_inst(InstKind::ControlBarrier).map_err(builder_error)?;
    for bit in 0..width.trailing_zeros() {
        let distance = body.literal(&(1u32 << bit).to_string(), &lane.ty)?;
        let span = body.literal(&(2u32 << bit).to_string(), &lane.ty)?;
        let remainder = body.binary(BinaryOperator::Remainder, lane.clone(), span)?;
        let active = body.binary(BinaryOperator::Equal, remainder, zero.clone())?;
        body.when(active, |body| {
            let right = body.binary(BinaryOperator::Add, lane.clone(), distance)?;
            for (i, buffer) in shared.iter().enumerate() {
                let a = body.index(buffer.clone(), lane.clone())?;
                let b = body.index(buffer.clone(), right.clone())?;
                let value = combine_accumulator(body, scope, operators[i], a, b)?;
                body.store(buffer.clone(), lane.clone(), value)?;
            }
            Ok(())
        })?;
        body.builder.push_void_inst(InstKind::ControlBarrier).map_err(builder_error)?;
    }
    let totals = shared
        .into_iter()
        .zip(neutral)
        .map(|(buffer, neutral)| {
            let total = body.index(buffer, zero.clone())?;
            body.cast(total, &neutral.ty)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Every lane reads the total before the next tile overwrites shared storage.
    body.builder.push_void_inst(InstKind::ControlBarrier).map_err(builder_error)?;
    Ok(totals)
}

pub(super) fn workgroup_scan(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operators: &[Value],
    mut values: Vec<Typed>,
    neutral: &[Typed],
    lane: Typed,
    width: u32,
) -> Result<(Vec<Typed>, Vec<Typed>, Vec<Typed>), OptimizeError> {
    assert!(
        width.is_power_of_two(),
        "workgroup scan width must be a power of two: {width}"
    );
    let zero = body.literal("0", &lane.ty)?;
    let length = body.literal(&width.to_string(), &lane.ty)?;
    let last = body.literal(&(width - 1).to_string(), &lane.ty)?;
    let mut shared = Vec::new();
    for (i, value) in values.iter().enumerate() {
        let mut banks = Vec::new();
        for bank in 0..2 {
            banks.push(body.op(
                OpTag::StorageView(PureViewSource::Workgroup {
                    id: (i * 2 + bank) as u32,
                    count: width,
                }),
                vec![zero.clone(), length.clone()],
                Body::view_type(&value.ty, types::no_buffer()),
            )?);
        }
        body.store(banks[0].clone(), lane.clone(), value.clone())?;
        shared.push(banks);
    }
    body.builder.push_void_inst(InstKind::ControlBarrier).map_err(builder_error)?;
    let mut bank = 0;
    for bit in 0..width.trailing_zeros() {
        let distance = body.literal(&(1u32 << bit).to_string(), &lane.ty)?;
        let valid = body.binary(BinaryOperator::GreaterEqual, lane.clone(), distance.clone())?;
        for (i, value) in values.iter_mut().enumerate() {
            *value = body.branch(
                scope,
                valid.clone(),
                |body| {
                    let index = body.binary(BinaryOperator::Subtract, lane.clone(), distance.clone())?;
                    let peer = body.index(shared[i][bank].clone(), index)?;
                    combine_accumulator(body, scope, operators[i], peer, value.clone())
                },
                |_| Ok(value.clone()),
                None,
            )?;
            body.store(shared[i][1 - bank].clone(), lane.clone(), value.clone())?;
        }
        bank = 1 - bank;
        body.builder.push_void_inst(InstKind::ControlBarrier).map_err(builder_error)?;
    }
    let one = body.literal("1", &lane.ty)?;
    let after_first = body.binary(BinaryOperator::Greater, lane.clone(), zero)?;
    let mut exclusive = Vec::new();
    let mut totals = Vec::new();
    for (i, banks) in shared.iter().enumerate() {
        exclusive.push(body.branch(
            scope,
            after_first.clone(),
            |body| {
                let previous = body.binary(BinaryOperator::Subtract, lane.clone(), one.clone())?;
                let value = body.index(banks[bank].clone(), previous)?;
                body.cast(value, &neutral[i].ty)
            },
            |_| Ok(neutral[i].clone()),
            None,
        )?);
        let value = body.index(banks[bank].clone(), last.clone())?;
        totals.push(body.cast(value, &neutral[i].ty)?);
    }
    // All lanes must finish reading before the next tile reuses shared storage.
    body.builder.push_void_inst(InstKind::ControlBarrier).map_err(builder_error)?;
    Ok((values, exclusive, totals))
}

/// Accumulate from the source stream, guarding fused filter inputs in place.
/// Reading the filtered array would recursively execute this same collective.
fn accumulate_at(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    plan: Value,
    operator: Value,
    state: Typed,
    index: Typed,
    cache: &mut LookupMap<Value, Typed>,
) -> Result<Typed, OptimizeError> {
    let Some(input) = body.compiler.facts.input(operator, 0) else {
        return Err(error("accumulator input missing"));
    };
    let filtered = match body.compiler.facts.operation(input) {
        Some(filter) if body.compiler.facts.member(plan, filter) && is_count(body, filter)? => Some(filter),
        _ => None,
    };
    if let Some(filter) = filtered {
        let Some(source) = body.compiler.facts.input(filter, 0) else {
            return Err(error("filter input missing"));
        };
        let incoming = element(body, scope, plan, source, index, cache)?;
        let keep = body.callback(scope, filter, vec![incoming.clone()])?;
        body.branch(
            scope,
            keep,
            |body| accumulate_element(body, scope, operator, state.clone(), incoming),
            |_| Ok(state.clone()),
            None,
        )
    } else {
        let incoming = element(body, scope, plan, input, index, cache)?;
        accumulate_element(body, scope, operator, state, incoming)
    }
}

fn is_count(body: &Body<'_, '_, '_>, operator: Value) -> Result<bool, OptimizeError> {
    Ok(body.compiler.facts.operation_kind(operator)? == "filter")
}
fn accumulate_element(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operator: Value,
    state: Typed,
    incoming: Typed,
) -> Result<Typed, OptimizeError> {
    if is_count(body, operator)? {
        let keep = body.callback(scope, operator, vec![incoming])?;
        let increment = body.cast(keep, &state.ty)?;
        return body.binary(BinaryOperator::Add, state, increment);
    }
    body.callback(scope, operator, vec![state, incoming])
}
fn combine_accumulator(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operator: Value,
    a: Typed,
    b: Typed,
) -> Result<Typed, OptimizeError> {
    if is_count(body, operator)? {
        body.binary(BinaryOperator::Add, a, b)
    } else {
        body.callback(scope, operator, vec![a, b])
    }
}
