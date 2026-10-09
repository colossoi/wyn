// Expand scheduled phases straight into SSA instructions and structured loops.
use super::super::{error, Body, OptimizeError, Typed};
use super::{element, indexed, invocation, write_arrays};
use crate::op::BinaryOperator::{
    Add, Divide, Equal, Greater, GreaterEqual, Less, Multiply, Remainder, Subtract,
};
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
    let totals = results.iter().filter(|r| r.0 == "total").map(|r| r.2);
    let mut accumulators = Vec::new();
    for source in scans.iter().copied().chain(totals) {
        let Some(operation) = body.compiler.facts.operation(source) else {
            return Err(error("accumulator operation missing"));
        };
        let neutral = body.compiler.facts.neutral(operation)?;
        let neutral = body.value(scope, neutral)?;
        accumulators.push(Accumulator::new(body, plan, source, operation, neutral)?);
    }
    for (index, filter) in body.compiler.facts.counts(plan)? {
        let index = scans.len() + usize::try_from(index).map_err(|_| error("negative count slot"))?;
        if index > accumulators.len() {
            return Err(error("noncontiguous count accumulator"));
        }
        let Some(source) = body.compiler.facts.source(filter) else {
            return Err(error("filter accumulator source missing"));
        };
        let neutral = body.literal("0", &types::i32())?;
        accumulators.insert(index, Accumulator::new(body, plan, source, filter, neutral)?);
    }
    let initial: Vec<_> = accumulators.iter().map(|a| a.neutral.clone()).collect();
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
            for (accumulator, state) in accumulators.iter().zip(state) {
                let value = accumulator.accumulate(body, scope, plan, state, &index, &mut cache)?;
                cache.insert(accumulator.source, value.clone());
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
    let width = Body::number(chunks_width);
    let lane = body.local_id()?;
    // Workgroups own contiguous ranges of whole tiles, preserving operand order.
    let tiles = body.binary(Add, n.clone(), Body::number((threads - 1) as u32))?;
    let tiles = body.binary(Divide, tiles, Body::number(threads as u32))?;
    let span = body.binary(Multiply, tiles.clone(), width.clone())?;
    match phase {
        "chunks" => {
            let (thread, _) = invocation(body, phase_width)?;
            let group = body.binary(Divide, thread, width.clone())?;
            let base = body.binary(Multiply, group.clone(), span.clone())?;
            let state = body.counted(
                Body::number(0),
                tiles,
                Body::number(1),
                initial.clone(),
                |body, tile, carry| {
                    let offset = body.binary(Multiply, tile, width.clone())?;
                    let offset = body.binary(Add, base.clone(), offset)?;
                    let index = body.binary(Add, offset, lane.clone())?;
                    let valid = body.binary(Less, index.clone(), n.clone())?;
                    let incoming = body.branch(
                        scope,
                        valid.clone(),
                        |body| {
                            let mut cache = LookupMap::default();
                            let mut values = Vec::new();
                            for accumulator in &accumulators {
                                let value = accumulator.accumulate(
                                    body,
                                    scope,
                                    plan,
                                    accumulator.neutral.clone(),
                                    &index,
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
                            body.tuple(values)
                        },
                        |body| body.tuple(initial.clone()),
                        None,
                    )?;
                    let values = (0..accumulators.len())
                        .map(|i| body.field(incoming.clone(), i))
                        .collect::<Result<Vec<_>, _>>()?;
                    let (prefixes, totals) = if scans.is_empty() {
                        (
                            Vec::new(),
                            workgroup_reduce(
                                body,
                                scope,
                                &accumulators,
                                values,
                                lane.clone(),
                                phase_width,
                            )?,
                        )
                    } else {
                        let (prefixes, _, totals) =
                            workgroup_scan(body, scope, &accumulators, values, lane.clone(), phase_width)?;
                        (prefixes, totals)
                    };
                    body.when(valid, |body| {
                        for (i, prefix) in prefixes.into_iter().take(scans.len()).enumerate() {
                            if let Some(output) = body.slot(scope, phase_owner, "prefix", i as i64, 2)? {
                                let value =
                                    accumulators[i].combine(body, scope, carry[i].clone(), prefix)?;
                                body.store(output, index.clone(), value)?;
                            }
                        }
                        Ok(())
                    })?;
                    accumulators
                        .iter()
                        .zip(carry.into_iter().zip(totals))
                        .map(|(accumulator, (carry, total))| accumulator.combine(body, scope, carry, total))
                        .collect()
                },
            )?;
            let first = body.binary(Equal, lane, Body::number(0))?;
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
            let groups = Body::number(groups as u32);
            let totals = body.counted(
                Body::number(0),
                groups.clone(),
                Body::number(phase_width),
                initial.clone(),
                |body, base, carry| {
                    let index = body.binary(Add, base, lane.clone())?;
                    let valid = body.binary(Less, index.clone(), groups)?;
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
                                &accumulators,
                                values,
                                lane.clone(),
                                phase_width,
                            )?,
                        )
                    } else {
                        let (_, offsets, totals) =
                            workgroup_scan(body, scope, &accumulators, values, lane.clone(), phase_width)?;
                        (offsets, totals)
                    };
                    body.when(valid, |body| {
                        for (i, offset) in offsets.into_iter().take(scans.len()).enumerate() {
                            if let Some(output) = body.slot(scope, phase_owner, "offset", i as i64, 2)? {
                                let value =
                                    accumulators[i].combine(body, scope, carry[i].clone(), offset)?;
                                body.store(output, index.clone(), value)?;
                            }
                        }
                        Ok(())
                    })?;
                    accumulators
                        .iter()
                        .zip(carry.into_iter().zip(totals))
                        .map(|(accumulator, (carry, total))| accumulator.combine(body, scope, carry, total))
                        .collect()
                },
            )?;
            let first = body.binary(Equal, lane, Body::number(0))?;
            body.when(first, |body| {
                for (i, value) in totals.into_iter().skip(scans.len()).enumerate() {
                    if let Some(output) = body.slot(scope, phase_owner, "total", i as i64, 2)? {
                        body.store(output, Body::number(0), value)?;
                    }
                }
                Ok(())
            })?;
        }
        "offsets" => {
            let (start, step) = invocation(body, phase_width)?;
            let writes_destination = body.compiler.facts.destination(phase_owner).is_some();
            body.counted(start, n, step, vec![], |body, index, _| {
                let chunk = body.binary(Divide, index.clone(), span)?;
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
                    let value = accumulators[i].combine(body, scope, a, b)?;
                    cache.insert(source, value);
                }
                if writes_destination {
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
    accumulators: &[Accumulator],
    values: Vec<Typed>,
    lane: Typed,
    width: u32,
) -> Result<Vec<Typed>, OptimizeError> {
    assert!(
        width.is_power_of_two(),
        "workgroup reduction width must be a power of two: {width}"
    );
    let mut shared = Vec::new();
    for (i, value) in values.into_iter().enumerate() {
        let buffer = body.workgroup_array(i as u32, width, &value.ty)?;
        body.store(buffer.clone(), lane.clone(), value)?;
        shared.push(buffer);
    }
    body.workgroup_barrier()?;
    for bit in 0..width.trailing_zeros() {
        let remainder = body.binary(Remainder, lane.clone(), Body::number(2 << bit))?;
        let active = body.binary(Equal, remainder, Body::number(0))?;
        body.when(active, |body| {
            let right = body.binary(Add, lane.clone(), Body::number(1 << bit))?;
            for (i, buffer) in shared.iter().enumerate() {
                let a = body.index(buffer.clone(), lane.clone())?;
                let b = body.index(buffer.clone(), right.clone())?;
                let value = accumulators[i].combine(body, scope, a, b)?;
                body.store(buffer.clone(), lane.clone(), value)?;
            }
            Ok(())
        })?;
        body.workgroup_barrier()?;
    }
    let totals = shared
        .into_iter()
        .zip(accumulators)
        .map(|(buffer, accumulator)| {
            let total = body.index(buffer, Body::number(0))?;
            body.cast(total, &accumulator.neutral.ty)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Every lane reads the total before the next tile overwrites shared storage.
    body.workgroup_barrier()?;
    Ok(totals)
}

pub(super) fn workgroup_scan(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    accumulators: &[Accumulator],
    mut values: Vec<Typed>,
    lane: Typed,
    width: u32,
) -> Result<(Vec<Typed>, Vec<Typed>, Vec<Typed>), OptimizeError> {
    assert!(
        width.is_power_of_two(),
        "workgroup scan width must be a power of two: {width}"
    );
    let mut shared = Vec::new();
    for (i, value) in values.iter().enumerate() {
        let banks = [
            body.workgroup_array((i * 2) as u32, width, &value.ty)?,
            body.workgroup_array((i * 2 + 1) as u32, width, &value.ty)?,
        ];
        body.store(banks[0].clone(), lane.clone(), value.clone())?;
        shared.push(banks);
    }
    body.workgroup_barrier()?;
    let mut bank = 0;
    for bit in 0..width.trailing_zeros() {
        let distance = Body::number(1 << bit);
        let valid = body.binary(GreaterEqual, lane.clone(), distance.clone())?;
        for (i, value) in values.iter_mut().enumerate() {
            *value = body.branch(
                scope,
                valid.clone(),
                |body| {
                    let index = body.binary(Subtract, lane.clone(), distance.clone())?;
                    let peer = body.index(shared[i][bank].clone(), index)?;
                    accumulators[i].combine(body, scope, peer, value.clone())
                },
                |_| Ok(value.clone()),
                None,
            )?;
            body.store(shared[i][1 - bank].clone(), lane.clone(), value.clone())?;
        }
        bank = 1 - bank;
        body.workgroup_barrier()?;
    }
    let after_first = body.binary(Greater, lane.clone(), Body::number(0))?;
    let mut exclusive = Vec::new();
    let mut totals = Vec::new();
    for (i, banks) in shared.iter().enumerate() {
        let neutral = &accumulators[i].neutral;
        exclusive.push(body.branch(
            scope,
            after_first.clone(),
            |body| {
                let previous = body.binary(Subtract, lane.clone(), Body::number(1))?;
                let value = body.index(banks[bank].clone(), previous)?;
                body.cast(value, &neutral.ty)
            },
            |_| Ok(neutral.clone()),
            None,
        )?);
        let value = body.index(banks[bank].clone(), Body::number(width - 1))?;
        totals.push(body.cast(value, &neutral.ty)?);
    }
    // All lanes must finish reading before the next tile reuses shared storage.
    body.workgroup_barrier()?;
    Ok((values, exclusive, totals))
}

/// Resolved recipe inputs, shared by serial accumulation and cooperative phases.
pub(super) struct Accumulator {
    source: Value,
    input: Value,
    filter: Option<Value>,
    operation: Value,
    count: bool,
    neutral: Typed,
}

impl Accumulator {
    pub fn new(
        body: &Body<'_, '_, '_>,
        plan: Value,
        source: Value,
        operation: Value,
        neutral: Typed,
    ) -> Result<Self, OptimizeError> {
        let Some(mut input) = body.compiler.facts.input(operation, 0) else {
            return Err(error("accumulator input missing"));
        };
        let filter = match body.compiler.facts.operation(input) {
            Some(filter)
                if body.compiler.facts.member(plan, filter)
                    && body.compiler.facts.operation_kind(filter)? == "filter" =>
            {
                Some(filter)
            }
            _ => None,
        };
        if let Some(filter) = filter {
            let Some(source) = body.compiler.facts.input(filter, 0) else {
                return Err(error("filter input missing"));
            };
            input = source;
        }
        Ok(Self {
            source,
            input,
            filter,
            operation,
            count: body.compiler.facts.operation_kind(operation)? == "filter",
            neutral,
        })
    }

    /// Guard fused filter inputs in place; reading their array would reenter this collective.
    fn accumulate(
        &self,
        body: &mut Body<'_, '_, '_>,
        scope: Value,
        plan: Value,
        state: Typed,
        index: &Typed,
        cache: &mut LookupMap<Value, Typed>,
    ) -> Result<Typed, OptimizeError> {
        let incoming = element(body, scope, plan, self.input, index.clone(), cache)?;
        if let Some(filter) = self.filter {
            let keep = body.callback(scope, filter, vec![incoming.clone()])?;
            body.branch(
                scope,
                keep,
                |body| self.accumulate_element(body, scope, state.clone(), incoming),
                |_| Ok(state.clone()),
                None,
            )
        } else {
            self.accumulate_element(body, scope, state, incoming)
        }
    }

    fn accumulate_element(
        &self,
        body: &mut Body<'_, '_, '_>,
        scope: Value,
        state: Typed,
        incoming: Typed,
    ) -> Result<Typed, OptimizeError> {
        if self.count {
            let keep = body.callback(scope, self.operation, vec![incoming])?;
            let increment = body.cast(keep, &state.ty)?;
            return body.binary(Add, state, increment);
        }
        body.callback(scope, self.operation, vec![state, incoming])
    }

    fn combine(
        &self,
        body: &mut Body<'_, '_, '_>,
        scope: Value,
        a: Typed,
        b: Typed,
    ) -> Result<Typed, OptimizeError> {
        if self.count {
            body.binary(Add, a, b)
        } else {
            body.callback(scope, self.operation, vec![a, b])
        }
    }
}
