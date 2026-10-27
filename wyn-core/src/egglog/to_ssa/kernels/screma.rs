// Expand scheduled phases straight into SSA instructions and structured loops.
use super::super::{error, Body, OptimizeError, Typed};
use super::{coordinates, element, indexed, invocation, write_arrays};
use crate::op::BinaryOperator::{
    Add, Divide, Equal, Greater, GreaterEqual, Less, Multiply, Remainder, Subtract,
};
use crate::op::OpTag;
use crate::types::{self, Type, TypeExt, TypeName};
use crate::LookupMap;
use egglog_engine::Value;

pub(super) struct SegmentOutput<'a> {
    pub source: Value,
    pub output: Typed,
    pub indices: &'a [Typed],
    pub axes: Value,
    pub role: &'a str,
    pub slot: usize,
}

impl SegmentOutput<'_> {
    fn store(
        &self,
        body: &mut Body<'_, '_, '_>,
        inner: Option<Typed>,
        value: Typed,
    ) -> Result<(), OptimizeError> {
        let indices = coordinates::project(body, self.axes, self.indices, inner)?;
        body.store_at(self.output.clone(), &indices, value)
    }
}

pub(super) fn screma(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    phase_owner: Value,
    phase: &str,
    phase_width: u32,
    plan: Value,
    results: &[(String, i64, Value)],
    segment: Option<&SegmentOutput<'_>>,
) -> Result<(), OptimizeError> {
    // Local recipes have no separate combine dispatch. Complete captured totals
    // first; the normal traversal can then emit each prefix with the final total.
    // These plans have duplicable callbacks; replay only their original domain.
    if matches!(phase, "ordered" | "local" | "segment")
        && body.compiler.facts.contains("PlanLateTotals", (plan,))
    {
        screma(
            body,
            scope,
            phase_owner,
            if phase == "ordered" { "ordered-totals" } else { "local-totals" },
            phase_width,
            plan,
            &[],
            None,
        )?;
    }
    let totals_only = matches!(phase, "ordered-totals" | "local-totals");
    let phase = match phase {
        "ordered-totals" => "ordered",
        "local-totals" => "local",
        phase => phase,
    };
    let mut components = body.compiler.facts.accumulators(plan)?;
    if totals_only {
        components.retain(|component| component.role != "scan");
    }
    let has_scans = components.iter().any(|component| component.role == "scan");
    let mut accumulators = Vec::new();
    for component in &components {
        let source = component.source;
        let Some(operation) = body.compiler.facts.operation(source) else {
            return Err(error("accumulator operation missing"));
        };
        let neutral = if component.role == "count" {
            let ty = body.compiler.facts.ty(component.ty).ok_or_else(|| error("count type missing"))?;
            body.literal("0", ty)?
        } else {
            let neutral = body.compiler.facts.neutral(operation)?;
            let layout = body.compiler.facts.value_layout(neutral)?;
            let value = body.value(scope, neutral)?;
            body.materialize(value, layout)?
        };
        accumulators.push(Accumulator::new(body, plan, source, operation, neutral)?);
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
            if !totals_only {
                write_output(body, scope, phase_owner, phase, plan, results, index, &mut cache)?;
            }
            Ok(next)
        })?;
        for (component, value) in components.iter().zip(state).filter(|(c, _)| c.role != "scan") {
            if totals_only {
                body.values.insert(component.source, value);
                continue;
            }
            if let Some(output) = body.slot(scope, phase_owner, "total", component.output_slot, 2)? {
                body.store(output, zero.clone(), value)?;
            }
        }
        return Ok(());
    }
    let uint = Type::Constructed(TypeName::UInt(32), vec![]);
    let n = body.cast(n, &uint)?;
    let (chunks_width, groups) = if segment.is_some() || phase == "local" {
        (phase_width, 1)
    } else {
        let Some(key) = body.compiler.facts.constructor("Stage", (phase_owner, "chunks")) else {
            return Err(error("collective chunk schedule missing"));
        };
        if body.compiler.facts.dispatch_grid(key)?.is_none() {
            return Err(error("collective requires a fixed chunk grid"));
        }
        let Some(groups) = body.compiler.facts.lookup("CollectiveGroups", (phase_owner,)) else {
            return Err(error("collective group count missing"));
        };
        (
            body.compiler.phase_width(key)?,
            u64::from(body.compiler.facts.positive(groups, "collective group count")?),
        )
    };
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
        "chunks" | "segment" | "local" => {
            let group = if segment.is_some() || phase == "local" {
                Body::number(0)
            } else {
                let (thread, _) = invocation(body, phase_width)?;
                body.binary(Divide, thread, width.clone())?
            };
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
                            if !has_scans && segment.is_none() && !totals_only {
                                write_arrays(
                                    body,
                                    scope,
                                    phase_owner,
                                    plan,
                                    results,
                                    &[index.clone()],
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
                    let (prefixes, totals) = if !has_scans {
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
                        let mut cache = LookupMap::default();
                        for (i, (component, prefix)) in components
                            .iter()
                            .zip(prefixes)
                            .enumerate()
                            .filter(|(_, (c, _))| c.role == "scan")
                        {
                            let value = accumulators[i].combine(body, scope, carry[i].clone(), prefix)?;
                            if segment.is_some() || phase == "local" {
                                cache.insert(accumulators[i].source, value);
                            } else if let Some(output) =
                                body.slot(scope, phase_owner, "prefix", component.output_slot, 2)?
                            {
                                body.store(output, index.clone(), value)?;
                            }
                        }
                        if let Some(target) = segment {
                            if target.role == "array" {
                                let value = element(
                                    body,
                                    scope,
                                    plan,
                                    target.source,
                                    std::slice::from_ref(&index),
                                    &mut cache,
                                )?;
                                target.store(body, Some(index.clone()), value)?;
                            }
                        } else if phase == "local" && has_scans {
                            write_output(
                                body,
                                scope,
                                phase_owner,
                                phase,
                                plan,
                                results,
                                index.clone(),
                                &mut cache,
                            )?;
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
            if totals_only {
                for (component, value) in components.iter().zip(state) {
                    body.values.insert(component.source, value);
                }
                return Ok(());
            }
            let first = body.binary(Equal, lane, Body::number(0))?;
            body.when(first, |body| {
                for (i, value) in state.into_iter().enumerate() {
                    if let Some(target) = segment {
                        if target.role == "total" && target.slot == i {
                            target.store(body, None, value)?;
                        }
                    } else if phase == "local" {
                        let component = &components[i];
                        if component.role != "scan" {
                            if let Some(output) =
                                body.slot(scope, phase_owner, "total", component.output_slot, 2)?
                            {
                                body.store(output, Body::number(0), value)?;
                            }
                        }
                    } else if let Some(output) = body.slot(scope, phase_owner, "partial", i as i64, 2)? {
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
                    let valid = body.binary(Less, index.clone(), groups.clone())?;
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
                    let (offsets, totals) = if !has_scans {
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
                        for (i, (component, offset)) in components
                            .iter()
                            .zip(offsets)
                            .enumerate()
                            .filter(|(_, (c, _))| c.role == "scan")
                        {
                            if let Some(output) =
                                body.slot(scope, phase_owner, "offset", component.output_slot, 2)?
                            {
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
                for (component, value) in components.iter().zip(totals).filter(|(c, _)| c.role != "scan") {
                    if let Some(output) =
                        body.slot(scope, phase_owner, "total", component.output_slot, 2)?
                    {
                        body.store(output, Body::number(0), value)?;
                    }
                }
                Ok(())
            })?;
        }
        "offsets" => {
            let (start, step) = invocation(body, phase_width)?;
            body.counted(start, n, step, vec![], |body, index, _| {
                let chunk = body.binary(Divide, index.clone(), span.clone())?;
                let mut cache = LookupMap::default();
                for (i, component) in components.iter().enumerate().filter(|(_, c)| c.role == "scan") {
                    let Some(prefix) = body.slot(scope, phase_owner, "prefix", component.output_slot, 1)?
                    else {
                        return Err(error("scan prefix is not materialized"));
                    };
                    let Some(offset) = body.slot(scope, phase_owner, "offset", component.output_slot, 1)?
                    else {
                        return Err(error("scan offset is not materialized"));
                    };
                    let a = body.index(offset, chunk.clone())?;
                    let b = body.index(prefix, index.clone())?;
                    let value = accumulators[i].combine(body, scope, a, b)?;
                    cache.insert(component.source, value);
                }
                write_output(body, scope, phase_owner, phase, plan, results, index, &mut cache)?;
                Ok(vec![])
            })?;
        }
        _ => return Err(error("invalid collective phase")),
    }
    Ok(())
}

fn write_output(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    owner: Value,
    phase: &str,
    plan: Value,
    results: &[(String, i64, Value)],
    index: Typed,
    cache: &mut LookupMap<Value, Typed>,
) -> Result<(), OptimizeError> {
    if body.compiler.facts.destination(owner).is_some() {
        let Some(output) = body.slot(scope, owner, "output", 0, 2)? else {
            return Err(error("scan output action has no destination"));
        };
        indexed::update(body, scope, owner, phase, plan, output, index, cache)
    } else {
        write_arrays(body, scope, owner, plan, results, &[index], cache)
    }
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
    if accumulators.is_empty() {
        return Ok(Vec::new());
    }
    let mut shared = Vec::new();
    for value in values {
        let buffer = CollectiveStorage::new(body, width, &value.ty)?;
        buffer.store(body, lane.clone(), value)?;
        shared.push(buffer);
    }
    body.workgroup_barrier()?;
    for bit in 0..width.trailing_zeros() {
        let remainder = body.binary(Remainder, lane.clone(), Body::number(2 << bit))?;
        let active = body.binary(Equal, remainder, Body::number(0))?;
        body.when(active, |body| {
            let right = body.binary(Add, lane.clone(), Body::number(1 << bit))?;
            for (i, buffer) in shared.iter().enumerate() {
                let a = buffer.load(body, lane.clone())?;
                let b = buffer.load(body, right.clone())?;
                let value = accumulators[i].combine(body, scope, a, b)?;
                buffer.store(body, lane.clone(), value)?;
            }
            Ok(())
        })?;
        body.workgroup_barrier()?;
    }
    let totals = shared
        .into_iter()
        .zip(accumulators)
        .map(|(buffer, accumulator)| {
            let total = buffer.load(body, Body::number(0))?;
            body.cast(total, &accumulator.neutral.ty)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Every lane reads the total before the next tile overwrites shared storage.
    body.workgroup_barrier()?;
    Ok(totals)
}

enum CollectiveStorage {
    Lanes(Typed),
    Columns(Type, Vec<CollectiveStorage>),
}

impl CollectiveStorage {
    fn new(body: &mut Body<'_, '_, '_>, width: u32, ty: &Type) -> Result<Self, OptimizeError> {
        if ty.array_variant().is_some_and(types::is_array_variant_composite) {
            if let Some(Type::Constructed(TypeName::Size(n), _)) = ty.array_size() {
                if body.compiler.facts.contains("ColumnarCollectiveExtent", (*n as i64,)) {
                    let Some(element) = ty.elem_type() else {
                        return Err(error("collective array has no element type"));
                    };
                    let columns =
                        (0..*n).map(|_| Self::new(body, width, element)).collect::<Result<_, _>>()?;
                    return Ok(Self::Columns(ty.clone(), columns));
                }
            }
        }
        Ok(Self::Lanes(body.workgroup_array(width, ty)?))
    }

    fn load(&self, body: &mut Body<'_, '_, '_>, lane: Typed) -> Result<Typed, OptimizeError> {
        match self {
            Self::Lanes(buffer) => body.index(buffer.clone(), lane),
            Self::Columns(ty, columns) => {
                let values = columns
                    .iter()
                    .map(|column| column.load(body, lane.clone()))
                    .collect::<Result<_, _>>()?;
                body.op(OpTag::ArrayLit, values, ty.clone())
            }
        }
    }

    fn store(&self, body: &mut Body<'_, '_, '_>, lane: Typed, value: Typed) -> Result<(), OptimizeError> {
        match self {
            Self::Lanes(buffer) => body.store(buffer.clone(), lane, value),
            Self::Columns(_, columns) => {
                for (i, column) in columns.iter().enumerate() {
                    let item = body.index(value.clone(), Body::number(i as u32))?;
                    column.store(body, lane.clone(), item)?;
                }
                Ok(())
            }
        }
    }
}

pub(super) fn workgroup_scan(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    accumulators: &[Accumulator],
    mut values: Vec<Typed>,
    lane: Typed,
    width: u32,
) -> Result<(Vec<Typed>, Vec<Typed>, Vec<Typed>), OptimizeError> {
    let mut shared = Vec::new();
    for value in &values {
        let buffer = CollectiveStorage::new(body, width, &value.ty)?;
        buffer.store(body, lane.clone(), value.clone())?;
        shared.push(buffer);
    }
    body.workgroup_barrier()?;
    for bit in 0..width.trailing_zeros() {
        let distance = Body::number(1 << bit);
        let valid = body.binary(GreaterEqual, lane.clone(), distance.clone())?;
        for (i, value) in values.iter_mut().enumerate() {
            *value = body.branch(
                scope,
                valid.clone(),
                |body| {
                    let index = body.binary(Subtract, lane.clone(), distance.clone())?;
                    let peer = shared[i].load(body, index)?;
                    accumulators[i].combine(body, scope, peer, value.clone())
                },
                |_| Ok(value.clone()),
                None,
            )?;
        }
        // Finish all reads before reusing the same shared bank for the next step.
        body.workgroup_barrier()?;
        for (buffer, value) in shared.iter().zip(&values) {
            buffer.store(body, lane.clone(), value.clone())?;
        }
        body.workgroup_barrier()?;
    }
    let after_first = body.binary(Greater, lane.clone(), Body::number(0))?;
    let mut exclusive = Vec::new();
    let mut totals = Vec::new();
    for (i, buffer) in shared.iter().enumerate() {
        let neutral = &accumulators[i].neutral;
        exclusive.push(body.branch(
            scope,
            after_first.clone(),
            |body| {
                let previous = body.binary(Subtract, lane.clone(), Body::number(1))?;
                let value = buffer.load(body, previous)?;
                body.cast(value, &neutral.ty)
            },
            |_| Ok(neutral.clone()),
            None,
        )?);
        let value = buffer.load(body, Body::number(width - 1))?;
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
    layout: Option<Value>,
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
        let layout = if body.compiler.facts.operation_kind(operation)? == "filter" {
            None
        } else {
            Some(body.compiler.facts.value_layout(body.compiler.facts.neutral(operation)?)?)
        };
        Ok(Self {
            layout,
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
        let incoming = element(body, scope, plan, self.input, std::slice::from_ref(index), cache)?;
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
        let value = body.callback(scope, self.operation, vec![state, incoming])?;
        self.materialize(body, value)
    }

    fn materialize(&self, body: &mut Body<'_, '_, '_>, value: Typed) -> Result<Typed, OptimizeError> {
        match self.layout {
            Some(layout) => body.materialize(value, layout),
            None => Ok(value),
        }
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
            let value = body.callback(scope, self.operation, vec![a, b])?;
            self.materialize(body, value)
        }
    }
}
