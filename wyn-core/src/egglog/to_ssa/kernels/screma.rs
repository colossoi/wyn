// Expand scheduled phases straight into SSA instructions and structured loops.
use super::super::{error, Body, OptimizeError, Typed};
use super::contribution::Contribution;
use super::{element, invocation, updates, write_arrays};
use crate::builtins::catalog;
use crate::egglog::facts::{AccumulatorComponent, FoldOutput, OutputAction};
use crate::egglog::query::Query;
use crate::op::BinaryOperator::{Add, Divide, Equal, Greater, Less, Multiply, Remainder, Subtract};
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
    results: &[(i64, Value)],
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
    let final_outputs = Query(&body.compiler.program.graph).flag("FinalOutputPass", (plan,))?;
    let measure = phase == "measure";
    let phase_width =
        if matches!(phase, "ordered" | "ordered-totals" | "measure") { 1 } else { phase_width };
    let totals_only = measure || matches!(phase, "ordered-totals" | "local-totals");
    let phase = match phase {
        "ordered" | "ordered-totals" | "measure" => "local",
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
    if totals_only && !measure {
        accumulators.retain(|a| matches!(a.output, FoldOutput::Total(_)));
    }
    let route = body.compiler.facts.record("PlanOutputRoute", "OutputAt", (plan,));
    let mut routed = Vec::new();
    let mut ordinary = Vec::new();
    for &(slot, source) in results {
        if Query(&body.compiler.program.graph).flag("RoutedArray", (plan, source))? {
            routed.push((slot, source));
        } else {
            ordinary.push((slot, source));
        }
    }
    let has_scans = accumulators.iter().any(|a| !matches!(a.output, FoldOutput::Total(_)));
    let initial: Vec<_> = accumulators.iter().map(|a| a.reduction.neutral.clone()).collect();
    let Some(domain) = body.compiler.facts.domain(phase_owner) else {
        return Err(error("collective domain missing"));
    };
    let n = body.extent(scope, domain)?;
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
                    let mut all_inputs = Vec::new();
                    let mut prefixes = Vec::new();
                    let mut totals = Vec::new();
                    let mut cache = LookupMap::default();
                    for group in accumulators.chunk_by(|a, b| a.stage == b.stage) {
                        let start = all_inputs.len();
                        let carry = &carry[start..start + group.len()];
                        let reductions: Vec<_> = group.iter().map(|a| &a.reduction).collect();
                        let incoming = body.branch_with_type(
                            Some(scope),
                            valid.clone(),
                            |body| {
                                let mut cache = cache.clone();
                                let mut values = Vec::new();
                                for accumulator in group {
                                    let value = if phase == "combine" {
                                        let Some(partial) =
                                            body.slot(scope, phase_owner, "partial", accumulator.slot, 1)?
                                        else {
                                            return Err(error("collective partial is not materialized"));
                                        };
                                        let value = body.index(partial, index.clone())?;
                                        body.cast(value, &accumulator.reduction.neutral.ty)?
                                    } else {
                                        accumulator.input(body, scope, plan, &index, &mut cache)?
                                    };
                                    let value = if phase == "combine" {
                                        body.tuple(vec![value, Body::number(1)])?
                                    } else {
                                        value
                                    };
                                    values.push(value);
                                }
                                body.tuple(values)
                            },
                            |body, ty| {
                                let Type::Constructed(TypeName::Tuple(_), fields) = ty else {
                                    return Err(error("collective input tuple missing"));
                                };
                                let values = group
                                    .iter()
                                    .zip(fields)
                                    .map(|(a, ty)| a.empty_input(body, ty))
                                    .collect::<Result<Vec<_>, _>>()?;
                                body.tuple(values)
                            },
                            None,
                        )?;
                        let inputs = (0..group.len())
                            .map(|i| body.field(incoming.clone(), i))
                            .collect::<Result<Vec<_>, _>>()?;
                        let values = inputs
                            .iter()
                            .cloned()
                            .map(|v| body.field(v, 0))
                            .collect::<Result<Vec<_>, _>>()?;
                        let mode = if !has_scans {
                            CollectiveOutput::Totals
                        } else if phase == "combine" {
                            CollectiveOutput::Exclusive
                        } else {
                            CollectiveOutput::Inclusive
                        };
                        let (group_prefixes, group_totals) = if phase_width == 1 {
                            let mut next = Vec::new();
                            for ((accumulator, input), state) in group.iter().zip(&inputs).zip(carry) {
                                let active = body.field(input.clone(), 1)?;
                                let active = body.binary(Greater, active, Body::number(0))?;
                                next.push(body.branch(
                                    scope,
                                    active,
                                    |body| {
                                        let value = body.field(input.clone(), 0)?;
                                        accumulator.reduction.combine(body, scope, state.clone(), value)
                                    },
                                    |_| Ok(state.clone()),
                                    None,
                                )?);
                            }
                            let prefixes = if matches!(mode, CollectiveOutput::Exclusive) {
                                carry.to_vec()
                            } else {
                                next.clone()
                            };
                            (prefixes, next)
                        } else {
                            let (prefixes, totals) = workgroup_fold(
                                body,
                                scope,
                                &reductions,
                                values,
                                lane.clone(),
                                phase_width,
                                mode,
                            )?;
                            let prefixes = reductions
                                .iter()
                                .zip(carry.iter().zip(prefixes))
                                .map(|(r, (a, b))| r.combine(body, scope, a.clone(), b))
                                .collect::<Result<Vec<_>, _>>()?;
                            let totals = reductions
                                .iter()
                                .zip(carry.iter().zip(totals))
                                .map(|(r, (a, b))| r.combine(body, scope, a.clone(), b))
                                .collect::<Result<Vec<_>, _>>()?;
                            (prefixes, totals)
                        };

                        if phase == "local" {
                            for (accumulator, prefix) in group.iter().zip(&group_prefixes) {
                                if matches!(accumulator.output, FoldOutput::Prefix(_)) {
                                    cache.insert(accumulator.source, prefix.clone());
                                }
                            }
                        }
                        all_inputs.extend(inputs);
                        prefixes.extend(group_prefixes);
                        totals.extend(group_totals);
                    }
                    let inputs = all_inputs;
                    if totals_only {
                        return Ok(totals);
                    }
                    body.when(valid, |body| {
                        for (i, accumulator) in accumulators.iter().enumerate() {
                            if phase != "combine" {
                                for (j, &(formal, _)) in accumulator.retained.iter().enumerate() {
                                    cache.insert(formal, body.field(inputs[i].clone(), j + 2)?);
                                }
                            }
                            let FoldOutput::Prefix(slot) = accumulator.output else {
                                continue;
                            };
                            if phase == "local" {
                                cache.insert(accumulator.source, prefixes[i].clone());
                            } else {
                                let role = if phase == "combine" { "offset" } else { "prefix" };
                                if let Some(output) = body.slot(scope, phase_owner, role, slot, 2)? {
                                    body.store(output, index.clone(), prefixes[i].clone())?;
                                }
                            }
                        }
                        if let Some(route) = &route {
                            let Some(i) = accumulators.iter().position(|a| a.source == route[0]) else {
                                return Err(error("output route refers to an unavailable prefix"));
                            };
                            let incoming = body.field(inputs[i].clone(), 0)?;
                            let arguments = body.tuple(vec![incoming, prefixes[i].clone()])?;
                            let selected = body.formula(scope, route[1], &arguments, &[])?;
                            body.when(selected, |body| {
                                let destination = body.formula(scope, route[2], &arguments, &[])?;
                                super::write_arrays(
                                    body,
                                    scope,
                                    phase_owner,
                                    plan,
                                    &routed,
                                    &[index.clone()],
                                    &[destination],
                                    &mut cache,
                                )
                            })?;
                        }
                        if phase == "local" || (phase == "chunks" && !has_scans && !final_outputs) {
                            write_output(
                                body,
                                scope,
                                phase_owner,
                                phase,
                                plan,
                                &ordinary,
                                index.clone(),
                                &mut cache,
                            )?;
                        }
                        Ok(())
                    })?;
                    Ok(totals)
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
                            FoldOutput::LocalPrefix(resource) => {
                                if let Some(output) = body.optional_resource(scope, resource, 2)? {
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
                    } else if let Some(output) =
                        body.slot(scope, phase_owner, "partial", accumulators[i].slot, 2)?
                    {
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
    results: &[(i64, Value)],
    index: Typed,
    cache: &mut LookupMap<Value, Typed>,
) -> Result<(), OptimizeError> {
    if matches!(body.compiler.facts.action(plan)?, OutputAction::Updates) {
        updates::update(
            body,
            scope,
            owner,
            phase,
            plan,
            std::slice::from_ref(&index),
            cache,
        )?;
    }
    write_arrays(
        body,
        scope,
        owner,
        plan,
        results,
        std::slice::from_ref(&index),
        std::slice::from_ref(&index),
        cache,
    )
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

#[derive(Clone, Copy)]
enum CollectiveOutput {
    Totals,
    Exclusive,
    Inclusive,
}

/// Ordered tree reduction, followed by a Blelloch downsweep only for prefixes.
/// Every round writes disjoint nodes, so both passes share one storage bank.
fn workgroup_fold(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    accumulators: &[&Reduction],
    values: Vec<Typed>,
    lane: Typed,
    width: u32,
    mode: CollectiveOutput,
) -> Result<(Vec<Typed>, Vec<Typed>), OptimizeError> {
    if accumulators.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    if width == 1 || (body.subgroup.is_some() && accumulators.iter().all(|a| a.subgroup_addition(body))) {
        let (inclusive, exclusive, totals) = if width == 1 {
            (
                values.clone(),
                accumulators.iter().map(|a| a.neutral.clone()).collect(),
                values,
            )
        } else {
            body.subgroup_scan(values, width)?
        };
        let prefixes = match mode {
            CollectiveOutput::Totals => Vec::new(),
            CollectiveOutput::Exclusive => exclusive,
            CollectiveOutput::Inclusive => inclusive,
        };
        return Ok((prefixes, totals));
    }
    if !width.is_power_of_two() {
        return Err(error("collective width must be a positive power of two"));
    }
    let mut shared = Vec::new();
    for value in &values {
        let buffer = CollectiveStorage::new(body, width, &value.ty)?;
        buffer.store(body, lane.clone(), value.clone())?;
        shared.push(buffer);
    }
    body.workgroup_barrier()?;
    let end = body.binary(Add, lane.clone(), Body::number(1))?;
    for bit in 0..width.trailing_zeros() {
        let remainder = body.binary(Remainder, end.clone(), Body::number(2 << bit))?;
        let active = body.binary(Equal, remainder, Body::number(0))?;
        body.when(active, |body| {
            let left = body.binary(Subtract, lane.clone(), Body::number(1 << bit))?;
            for (buffer, accumulator) in shared.iter().zip(accumulators) {
                let a = buffer.load(body, left.clone())?;
                let b = buffer.load(body, lane.clone())?;
                let value = accumulator.combine(body, scope, a, b)?;
                buffer.store(body, lane.clone(), value)?;
            }
            Ok(())
        })?;
        body.workgroup_barrier()?;
    }
    let mut totals = Vec::new();
    for (buffer, accumulator) in shared.iter().zip(accumulators) {
        let value = buffer.load(body, Body::number(width - 1))?;
        totals.push(body.cast(value, &accumulator.neutral.ty)?);
    }
    // Capture totals in every lane before resetting the root or starting a new tile.
    body.workgroup_barrier()?;
    if matches!(mode, CollectiveOutput::Totals) {
        return Ok((Vec::new(), totals));
    }
    let last = body.binary(Equal, end.clone(), Body::number(width))?;
    body.when(last, |body| {
        for (buffer, accumulator) in shared.iter().zip(accumulators) {
            buffer.store(body, lane.clone(), accumulator.neutral.clone())?;
        }
        Ok(())
    })?;
    body.workgroup_barrier()?;
    for bit in (0..width.trailing_zeros()).rev() {
        let remainder = body.binary(Remainder, end.clone(), Body::number(2 << bit))?;
        let active = body.binary(Equal, remainder, Body::number(0))?;
        body.when(active, |body| {
            let left = body.binary(Subtract, lane.clone(), Body::number(1 << bit))?;
            for (buffer, accumulator) in shared.iter().zip(accumulators) {
                let left_total = buffer.load(body, left.clone())?;
                let prefix = buffer.load(body, lane.clone())?;
                let right_prefix = accumulator.combine(body, scope, prefix.clone(), left_total)?;
                buffer.store(body, left.clone(), prefix)?;
                buffer.store(body, lane.clone(), right_prefix)?;
            }
            Ok(())
        })?;
        body.workgroup_barrier()?;
    }
    let mut prefixes = Vec::new();
    for ((buffer, accumulator), value) in shared.iter().zip(accumulators).zip(values) {
        let prefix = buffer.load(body, lane.clone())?;
        prefixes.push(match mode {
            CollectiveOutput::Inclusive => accumulator.combine(body, scope, prefix, value)?,
            CollectiveOutput::Exclusive => body.cast(prefix, &accumulator.neutral.ty)?,
            CollectiveOutput::Totals => unreachable!("totals returned before the downsweep"),
        });
    }
    body.workgroup_barrier()?;
    Ok((prefixes, totals))
}

/// One fold has one contribution expression and one reducer.
struct Accumulator {
    slot: i64,
    stage: i64,
    output: FoldOutput,
    source: Value,
    contribution: Value,
    retained: Vec<(Value, Value)>,
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
            slot: component.slot,
            stage: component.stage,
            source,
            output: component.output,
            contribution: fields[1],
            retained: body
                .compiler
                .facts
                .vector(fields[2])?
                .into_iter()
                .map(|binding| {
                    let Some(fields) = body.compiler.facts.enode("CaptureBinding", binding) else {
                        return Err(error("invalid fold retained binding"));
                    };
                    Ok((fields[0], fields[1]))
                })
                .collect::<Result<_, _>>()?,
            reduction: Reduction { reducer, neutral },
        })
    }

    // The activation field lets serial execution skip the reducer on rejected lanes.
    fn input(
        &self,
        body: &mut Body<'_, '_, '_>,
        scope: Value,
        plan: Value,
        index: &Typed,
        cache: &mut LookupMap<Value, Typed>,
    ) -> Result<Typed, OptimizeError> {
        Contribution {
            scope,
            plan,
            indices: std::slice::from_ref(index),
            skipped: &|body, ty| self.empty_input(body, ty),
        }
        .emit(body, self.contribution, cache, &mut |body, cache, incoming| {
            let value = body.cast(incoming, &self.reduction.neutral.ty)?;
            let retained = self
                .retained
                .iter()
                .map(|&(_, actual)| {
                    element(
                        body,
                        scope,
                        Some(plan),
                        actual,
                        std::slice::from_ref(index),
                        cache,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut values = vec![value, Body::number(1)];
            values.extend(retained);
            body.tuple(values)
        })
    }

    fn empty_input(&self, body: &mut Body<'_, '_, '_>, ty: &Type) -> Result<Typed, OptimizeError> {
        let Type::Constructed(TypeName::Tuple(_), fields) = ty else {
            return Err(error("collective input tuple missing"));
        };
        let mut values = vec![self.reduction.neutral.clone(), Body::number(0)];
        for ty in &fields[2..] {
            values.push(body.op(
                OpTag::Intrinsic {
                    id: catalog().known().uninit,
                    overload_idx: 0,
                },
                vec![],
                ty.clone(),
            )?);
        }
        body.tuple(values)
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
