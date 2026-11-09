// Expand scheduled phases straight into SSA instructions and structured loops.
use super::super::{error, Body, OptimizeError, Typed};
use super::contribution::Contribution;
use super::{invocation, updates, write_arrays};
use crate::builtins::catalog;
use crate::egglog::facts::{AccumulatorComponent, FoldOutput, OutputAction};
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
    let packing = match body.compiler.facts.action(plan)? {
        OutputAction::Pack { predicate, input } => Some((predicate, input)),
        _ => None,
    };
    // Packing's count is a prefix fold. A serial recipe uses the same loop at width one.
    let phase_width = if phase == "ordered" && packing.is_some() { 1 } else { phase_width };
    let phase = if packing.is_some() { "local" } else { phase };
    let totals_only = matches!(phase, "ordered-totals" | "local-totals");
    let phase = match phase {
        "ordered-totals" => "ordered",
        "local-totals" => "local",
        phase => phase,
    };
    let mut accumulators = body
        .compiler
        .facts
        .accumulators(plan)?
        .into_iter()
        .map(|component| Accumulator::new(body, scope, plan, component))
        .collect::<Result<Vec<_>, _>>()?;
    if totals_only {
        accumulators.retain(|a| matches!(a.output, FoldOutput::Total(_)));
    }
    let packing = if let Some((predicate, input)) = packing {
        let Some(index) = accumulators.iter().position(|a| a.output == FoldOutput::PackedLength) else {
            return Err(error("packing requires a count prefix"));
        };
        Some((predicate, input, index))
    } else {
        None
    };
    let has_scans = accumulators.iter().any(|a| !matches!(a.output, FoldOutput::Total(_)));
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
        for (accumulator, value) in accumulators.iter().zip(state) {
            let FoldOutput::Total(slot) = accumulator.output else {
                continue;
            };
            if totals_only {
                body.values.insert(accumulator.source, value);
                continue;
            }
            if let Some(output) = body.output_target(scope, phase_owner, "total", slot)? {
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
    let lane = if phase_width == 1 { Body::number(0) } else { body.local_id()? };
    // Workgroups own contiguous ranges of whole tiles, preserving operand order.
    let tiles = body.binary(Add, n.clone(), Body::number((threads - 1) as u32))?;
    let tiles = body.binary(Divide, tiles, Body::number(threads as u32))?;
    let span = body.binary(Multiply, tiles.clone(), width.clone())?;
    match phase {
        "chunks" | "local" | "combine" => {
            let (n, tiles) = if phase == "combine" {
                let n = Body::number(groups as u32);
                let rounded = body.binary(Add, n.clone(), Body::number(phase_width - 1))?;
                (n, body.binary(Divide, rounded, Body::number(phase_width))?)
            } else {
                (n, tiles)
            };
            let group = if phase != "chunks" {
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
                    let offset = body.binary(Multiply, tile, Body::number(phase_width))?;
                    let offset = body.binary(Add, base.clone(), offset)?;
                    let index = body.binary(Add, offset, lane.clone())?;
                    let valid = body.binary(Less, index.clone(), n.clone())?;
                    let incoming = body.branch_with_type(
                        Some(scope),
                        valid.clone(),
                        |body| {
                            let mut cache = LookupMap::default();
                            let mut values = Vec::new();
                            for (i, accumulator) in accumulators.iter().enumerate() {
                                let value = if phase == "combine" {
                                    let Some(partial) =
                                        body.slot(scope, phase_owner, "partial", i as i64, 1)?
                                    else {
                                        return Err(error("collective partial is not materialized"));
                                    };
                                    let value = body.index(partial, index.clone())?;
                                    body.cast(value, &initial[i].ty)?
                                } else if accumulator.output == FoldOutput::PackedLength {
                                    packing_input(body, scope, plan, &index, &mut cache)?
                                } else {
                                    accumulator.accumulate(
                                        body,
                                        scope,
                                        plan,
                                        accumulator.reduction.neutral.clone(),
                                        &index,
                                        &mut cache,
                                    )?
                                };
                                values.push(value);
                            }
                            if !has_scans && !totals_only && phase != "combine" {
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
                        |body, ty| {
                            let mut values = initial.clone();
                            if let Some((_, _, i)) = packing {
                                let Type::Constructed(TypeName::Tuple(_), fields) = ty else {
                                    return Err(error("collective input tuple missing"));
                                };
                                values[i] = empty_retained(body, &fields[i])?;
                            }
                            body.tuple(values)
                        },
                        None,
                    )?;
                    let mut values = (0..accumulators.len())
                        .map(|i| body.field(incoming.clone(), i))
                        .collect::<Result<Vec<_>, _>>()?;
                    let retained = if let Some((_, _, i)) = packing {
                        let pair = values[i].clone();
                        values[i] = body.field(pair.clone(), 0)?;
                        Some(pair)
                    } else {
                        None
                    };
                    let (prefixes, totals) = if !has_scans {
                        (
                            Vec::new(),
                            workgroup_reduce(body, scope, &reductions, values, lane.clone(), phase_width)?,
                        )
                    } else {
                        let (prefixes, offsets, totals) =
                            workgroup_scan(body, scope, &reductions, values, lane.clone(), phase_width)?;
                        (if phase == "combine" { offsets } else { prefixes }, totals)
                    };
                    body.when(valid, |body| {
                        let mut cache = LookupMap::default();
                        for (i, (accumulator, prefix)) in
                            accumulators.iter().zip(prefixes.iter().cloned()).enumerate()
                        {
                            let FoldOutput::Prefix(slot) = accumulator.output else {
                                continue;
                            };
                            let value =
                                accumulators[i].reduction.combine(body, scope, carry[i].clone(), prefix)?;
                            if phase == "local" {
                                cache.insert(accumulators[i].source, value);
                            } else {
                                let role = if phase == "combine" { "offset" } else { "prefix" };
                                if let Some(output) = body.slot(scope, phase_owner, role, slot, 2)? {
                                    body.store(output, index.clone(), value)?;
                                }
                            }
                        }
                        if let (Some((predicate, input, i)), Some(retained)) = (packing, retained) {
                            let flag = body.field(retained.clone(), 0)?;
                            let selected = body.binary(Greater, flag, Body::number(0))?;
                            return body.when(selected, |body| {
                                let value = body.field(retained, 1)?;
                                let Some(source) = body.compiler.facts.source(predicate) else {
                                    return Err(error("packing result missing"));
                                };
                                cache.insert(input, value.clone());
                                cache.insert(source, value);
                                let prefix = body.binary(Add, carry[i].clone(), prefixes[i].clone())?;
                                let destination = body.binary(Subtract, prefix, Body::number(1))?;
                                super::write_arrays_at(
                                    body,
                                    scope,
                                    phase_owner,
                                    plan,
                                    results,
                                    &[index.clone()],
                                    &[destination],
                                    &mut cache,
                                )
                            });
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
                for (accumulator, value) in accumulators.iter().zip(state) {
                    body.values.insert(accumulator.source, value);
                }
                return Ok(());
            }
            let first = body.binary(Equal, lane, Body::number(0))?;
            body.when(first, |body| {
                for (i, value) in state.into_iter().enumerate() {
                    if phase != "chunks" {
                        match accumulators[i].output {
                            FoldOutput::PackedLength => {
                                if let Some(output) = body.slot(scope, phase_owner, "length", 0, 2)? {
                                    body.store(output, Body::number(0), value)?;
                                }
                            }
                            FoldOutput::Total(slot) => {
                                if let Some(output) =
                                    body.output_target(scope, phase_owner, "total", slot)?
                                {
                                    output.store(body, &[Body::number(0)], value)?;
                                }
                            }
                            FoldOutput::Prefix(_) => {}
                        }
                    } else if let Some(output) = body.slot(scope, phase_owner, "partial", i as i64, 2)? {
                        body.store(output, group.clone(), value)?;
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
                for (i, accumulator) in accumulators.iter().enumerate() {
                    let FoldOutput::Prefix(slot) = accumulator.output else {
                        continue;
                    };
                    let Some(prefix) = body.slot(scope, phase_owner, "prefix", slot, 1)? else {
                        return Err(error("scan prefix is not materialized"));
                    };
                    let Some(offset) = body.slot(scope, phase_owner, "offset", slot, 1)? else {
                        return Err(error("scan offset is not materialized"));
                    };
                    let a = body.index(offset, chunk.clone())?;
                    let b = body.index(prefix, index.clone())?;
                    let value = accumulators[i].reduction.combine(body, scope, a, b)?;
                    cache.insert(accumulator.source, value);
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
    output: FoldOutput,
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
        region: Value,
        layout: Value,
    },
}

impl Accumulator {
    fn new(
        body: &mut Body<'_, '_, '_>,
        scope: Value,
        plan: Value,
        component: AccumulatorComponent,
    ) -> Result<Self, OptimizeError> {
        let source = component.source;
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
                    region: reducer[0],
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
            output: component.output,
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
            Reducer::Callback { region, .. } => body.compiler.facts.contains("SubgroupAddition", (region,)),
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
            Reducer::Callback { region, layout } => {
                let value = body.apply_region(scope, region, vec![a, b])?;
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

fn packing_input(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    plan: Value,
    index: &Typed,
    cache: &mut LookupMap<Value, Typed>,
) -> Result<Typed, OptimizeError> {
    let Some(input) = body.compiler.facts.lookup("PackingContribution", (plan,)) else {
        return Err(error("packing contribution missing"));
    };
    Contribution {
        scope,
        plan,
        index,
        skipped: &empty_retained,
    }
    .emit(body, input, cache, &mut |body, _, pair| {
        let flag = body.field(pair.clone(), 0)?;
        let flag = body.cast(flag, &Body::number(0).ty)?;
        let value = body.field(pair, 1)?;
        body.tuple(vec![flag, value])
    })
}
