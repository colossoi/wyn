// Expand scheduled phases straight into SSA instructions and structured loops.
use super::super::{error, Body, OptimizeError, Typed};
use super::contribution::Contribution;
use super::{invocation, updates, write_arrays};
use crate::builtins::catalog;
use crate::egglog::facts::OutputAction;
use crate::op::BinaryOperator::{
    Add, Divide, Equal, Greater, GreaterEqual, Less, Multiply, Remainder, Subtract,
};
use crate::op::OpTag;
use crate::types::{self, Type, TypeExt, TypeName};
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
    // Local recipes have no separate combine dispatch. Complete captured totals
    // first; the normal traversal can then emit each prefix with the final total.
    // These plans have duplicable callbacks; replay only their original domain.
    if matches!(phase, "ordered" | "local") && body.compiler.facts.contains("PlanLateTotals", (plan,)) {
        screma(
            body,
            scope,
            phase_owner,
            if phase == "ordered" { "ordered-totals" } else { "local-totals" },
            phase_width,
            plan,
            &[],
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
        accumulators.push(Accumulator::new(body, scope, plan, component.source)?);
    }
    let reductions: Vec<_> = accumulators.iter().map(|a| &a.reduction).collect();
    let initial: Vec<_> = accumulators.iter().map(|a| a.reduction.neutral.clone()).collect();
    let Some(domain) = body.compiler.facts.domain(phase_owner) else {
        return Err(error("collective domain missing"));
    };
    let n = body.extent(scope, domain)?;
    if phase == "ordered" {
        let zero = body.literal(0, &types::i32())?;
        let one = body.literal(1, &types::i32())?;
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
            if let Some(output) = body.output_target(scope, phase_owner, "total", component.output_slot)? {
                output.store(body, &[zero.clone()], value)?;
            }
        }
        return Ok(());
    }
    let uint = Type::Constructed(TypeName::UInt(32), vec![]);
    let n = body.cast(n, &uint)?;
    let (chunks_width, groups) = if phase == "local" {
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
        "chunks" | "local" => {
            let group = if phase == "local" {
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
                                    accumulator.reduction.neutral.clone(),
                                    &index,
                                    &mut cache,
                                )?;
                                values.push(value);
                            }
                            if !has_scans && !totals_only {
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
                            workgroup_reduce(body, scope, &reductions, values, lane.clone(), phase_width)?,
                        )
                    } else {
                        let (prefixes, _, totals) =
                            workgroup_scan(body, scope, &reductions, values, lane.clone(), phase_width)?;
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
                            let value =
                                accumulators[i].reduction.combine(body, scope, carry[i].clone(), prefix)?;
                            if phase == "local" {
                                cache.insert(accumulators[i].source, value);
                            } else if let Some(output) =
                                body.slot(scope, phase_owner, "prefix", component.output_slot, 2)?
                            {
                                body.store(output, index.clone(), value)?;
                            }
                        }
                        if phase == "local" && has_scans {
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
                        .map(|(accumulator, (carry, total))| {
                            accumulator.reduction.combine(body, scope, carry, total)
                        })
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
                    if phase == "local" {
                        let component = &components[i];
                        if component.role != "scan" {
                            if let Some(output) =
                                body.output_target(scope, phase_owner, "total", component.output_slot)?
                            {
                                output.store(body, &[Body::number(0)], value)?;
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
                            workgroup_reduce(body, scope, &reductions, values, lane.clone(), phase_width)?,
                        )
                    } else {
                        let (_, offsets, totals) =
                            workgroup_scan(body, scope, &reductions, values, lane.clone(), phase_width)?;
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
                                let value = accumulators[i].reduction.combine(
                                    body,
                                    scope,
                                    carry[i].clone(),
                                    offset,
                                )?;
                                body.store(output, index.clone(), value)?;
                            }
                        }
                        Ok(())
                    })?;
                    accumulators
                        .iter()
                        .zip(carry.into_iter().zip(totals))
                        .map(|(accumulator, (carry, total))| {
                            accumulator.reduction.combine(body, scope, carry, total)
                        })
                        .collect()
                },
            )?;
            let first = body.binary(Equal, lane, Body::number(0))?;
            body.when(first, |body| {
                for (component, value) in components.iter().zip(totals).filter(|(c, _)| c.role != "scan") {
                    if let Some(output) =
                        body.output_target(scope, phase_owner, "total", component.output_slot)?
                    {
                        output.store(body, &[Body::number(0)], value)?;
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
                    let value = accumulators[i].reduction.combine(body, scope, a, b)?;
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
    if matches!(body.compiler.facts.action(plan)?, OutputAction::Update { .. }) {
        let Some(output) = body.slot(scope, owner, "output", 0, 2)? else {
            return Err(error("scan output action has no destination"));
        };
        updates::update(body, scope, owner, phase, plan, output, index, cache)
    } else {
        write_arrays(body, scope, owner, plan, results, &[index], cache)
    }
}

/// Reduce adjacent contiguous ranges in source order. A single shared bank is
/// sufficient: a round's right-hand ranges never write during that round.
fn workgroup_reduce(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    accumulators: &[&Reduction],
    values: Vec<Typed>,
    lane: Typed,
    width: u32,
) -> Result<Vec<Typed>, OptimizeError> {
    if accumulators.is_empty() {
        return Ok(Vec::new());
    }
    if body.subgroup.is_some() && accumulators.iter().all(|a| a.subgroup_addition(body)) {
        let (_, _, totals) = body.subgroup_scan(values, width)?;
        return Ok(totals);
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

fn workgroup_scan(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    accumulators: &[&Reduction],
    mut values: Vec<Typed>,
    lane: Typed,
    width: u32,
) -> Result<(Vec<Typed>, Vec<Typed>, Vec<Typed>), OptimizeError> {
    if width == 1 {
        return Ok((
            values.clone(),
            accumulators.iter().map(|a| a.neutral.clone()).collect(),
            values,
        ));
    }
    if body.subgroup.is_some() && accumulators.iter().all(|a| a.subgroup_addition(body)) {
        return body.subgroup_scan(values, width);
    }
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

/// One fold has one contribution expression and one reducer.
struct Accumulator {
    source: Value,
    contribution: Value,
    reduction: Reduction,
}

struct Reduction {
    reducer: Reducer,
    neutral: Typed,
}

enum Reducer {
    Add,
    Callback {
        operation: Value,
        layout: Value,
    },
}

impl Accumulator {
    fn new(
        body: &mut Body<'_, '_, '_>,
        scope: Value,
        plan: Value,
        source: Value,
    ) -> Result<Self, OptimizeError> {
        let Some(fold) = body.compiler.facts.lookup("PlanFold", (plan, source)) else {
            return Err(error("parallel fold is missing"));
        };
        let Some(fields) = body.compiler.facts.enode("FoldWith", fold) else {
            return Err(error("invalid parallel fold"));
        };
        let (reducer, neutral) = if let Some(reducer) = body.compiler.facts.enode("CombineWith", fields[0])
        {
            let layout = body.compiler.facts.value_layout(reducer[1])?;
            let neutral = body.value(scope, reducer[1])?;
            (
                Reducer::Callback {
                    operation: reducer[0],
                    layout,
                },
                body.materialize(neutral, layout)?,
            )
        } else if let Some(reducer) = body.compiler.facts.enode("AddValues", fields[0]) {
            let Some(ty) = body.compiler.facts.ty(reducer[0]) else {
                return Err(error("additive fold type is missing"));
            };
            (Reducer::Add, body.literal(0, ty)?)
        } else {
            return Err(error("invalid parallel reducer"));
        };
        Ok(Self {
            source,
            contribution: fields[1],
            reduction: Reduction { reducer, neutral },
        })
    }

    fn accumulate(
        &self,
        body: &mut Body<'_, '_, '_>,
        scope: Value,
        plan: Value,
        state: Typed,
        index: &Typed,
        cache: &mut LookupMap<Value, Typed>,
    ) -> Result<Typed, OptimizeError> {
        Contribution {
            scope,
            plan,
            index,
            skipped: &|_, _| Ok(state.clone()),
        }
        .emit(body, self.contribution, cache, &mut |body, _, incoming| {
            self.reduction.combine(body, scope, state.clone(), incoming)
        })
    }
}

impl Reduction {
    fn subgroup_addition(&self, body: &Body<'_, '_, '_>) -> bool {
        match self.reducer {
            Reducer::Add => true,
            Reducer::Callback { operation, .. } => {
                body.compiler.facts.contains("SubgroupAddition", (operation,))
            }
        }
    }

    fn combine(
        &self,
        body: &mut Body<'_, '_, '_>,
        scope: Value,
        a: Typed,
        b: Typed,
    ) -> Result<Typed, OptimizeError> {
        match self.reducer {
            Reducer::Add => {
                let b = body.cast(b, &a.ty)?;
                body.binary(Add, a, b)
            }
            Reducer::Callback { operation, layout } => {
                let value = body.callback(scope, operation, vec![a, b])?;
                body.materialize(value, layout)
            }
        }
    }
}

fn empty_retained(body: &mut Body<'_, '_, '_>, ty: &Type) -> Result<Typed, OptimizeError> {
    let Type::Constructed(TypeName::Tuple(_), fields) = ty else {
        return Err(error("packing input pair missing"));
    };
    let unused = body.op(
        OpTag::Intrinsic {
            id: catalog().known().uninit,
            overload_idx: 0,
        },
        vec![],
        fields[1].clone(),
    )?;
    body.tuple(vec![Body::number(0), unused])
}

/// Packing is an additive prefix plus a guarded output action. Width one uses
/// exactly the same order and predicate lifetime as cooperative execution.
pub(super) fn pack(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    owner: Value,
    width: u32,
    plan: Value,
    results: &[(String, i64, Value)],
) -> Result<(), OptimizeError> {
    let OutputAction::Pack { predicate, input } = body.compiler.facts.action(plan)? else {
        return Err(error("packing requires a prefix output action"));
    };
    let Some(source) = body.compiler.facts.source(predicate) else {
        return Err(error("packing result missing"));
    };
    let Some(domain) = body.compiler.facts.domain(owner) else {
        return Err(error("packing domain missing"));
    };
    let n = body.extent(scope, domain)?;
    let uint = Type::Constructed(TypeName::UInt(32), vec![]);
    let n = body.cast(n, &uint)?;
    let zero = Body::number(0);
    let one = Body::number(1);
    let lane = if width == 1 { zero.clone() } else { body.local_id()? };
    let sum = body.binary(Add, n.clone(), Body::number(width - 1))?;
    let tiles = body.binary(Divide, sum, Body::number(width))?;
    let reduction = Reduction {
        reducer: Reducer::Add,
        neutral: zero.clone(),
    };
    let counts = body.counted(
        zero.clone(),
        tiles,
        one.clone(),
        vec![zero.clone()],
        |body, tile, carry| {
            let base = body.binary(Multiply, tile, Body::number(width))?;
            let index = body.binary(Add, base, lane.clone())?;
            let valid = body.binary(Less, index.clone(), n.clone())?;
            // Retain the input beside its flag; invalid lanes never evaluate it.
            let retained = body.branch_with_type(
                Some(scope),
                valid,
                |body| {
                    let Some(input) = body.compiler.facts.lookup("StreamInput", (plan, input)) else {
                        return Err(error("packing contribution missing"));
                    };
                    Contribution {
                        scope,
                        plan,
                        index: &index,
                        skipped: &empty_retained,
                    }
                    .emit(
                        body,
                        input,
                        &mut LookupMap::default(),
                        &mut |body, _, value| {
                            let flag = body.callback(scope, predicate, vec![value.clone()])?;
                            let flag = body.cast(flag, &uint)?;
                            body.tuple(vec![flag, value])
                        },
                    )
                },
                empty_retained,
                None,
            )?;
            let flag = body.field(retained.clone(), 0)?;
            let (prefix, _, total) = workgroup_scan(
                body,
                scope,
                &[&reduction],
                vec![flag.clone()],
                lane.clone(),
                width,
            )?;
            let selected = body.binary(Greater, flag, zero.clone())?;
            body.when(selected, |body| {
                let value = body.field(retained, 1)?;
                let mut cache = LookupMap::default();
                cache.insert(input, value.clone());
                cache.insert(source, value);
                let offset = body.binary(Subtract, prefix[0].clone(), one.clone())?;
                let destination = body.binary(Add, carry[0].clone(), offset)?;
                super::write_arrays_at(
                    body,
                    scope,
                    owner,
                    plan,
                    results,
                    &[index],
                    &[destination],
                    &mut cache,
                )
            })?;
            Ok(vec![body.binary(Add, carry[0].clone(), total[0].clone())?])
        },
    )?;
    let first = body.binary(Equal, lane, zero.clone())?;
    body.when(first, |body| {
        if let Some(output) = body.slot(scope, owner, "length", 0, 2)? {
            body.store(output, zero, counts[0].clone())?;
        }
        Ok(())
    })
}
