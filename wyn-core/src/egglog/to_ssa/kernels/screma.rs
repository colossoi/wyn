use super::super::interface;
// Expand scheduled phases straight into SSA instructions and structured loops.
use super::super::plan::Stage;
use super::super::{builder_error, error, Body, OptimizeError, Typed};
use super::{element, invocation, store};
use crate::builtins::catalog;
use crate::op::{BinaryOperator, OpTag, PureViewSource};
use crate::ssa::types::InstKind;
use crate::tlc::{SoacOp, TermKind};
use crate::types::{self, Type, TypeName};
use crate::LookupMap;
use egglog_engine::Value;

pub(super) fn screma(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    stage: &Stage,
    plan: Value,
    results: &[(String, i64, Value)],
) -> Result<(), OptimizeError> {
    let scans: Vec<_> = results.iter().filter(|r| r.0 == "scan").map(|r| r.2).collect();
    let totals: Vec<_> = results.iter().filter(|r| r.0 == "total").map(|r| r.2).collect();
    let mut sources: Vec<_> = scans.iter().chain(&totals).copied().collect();
    let mut initial = Vec::new();
    let mut operators = Vec::new();
    for &source in &sources {
        let Some(&(term, owner)) = body.compiler.program.identities.origins.get(&source) else {
            return Err(error("accumulator source missing"));
        };
        let Some(operation) = body.compiler.facts.operation(source) else {
            return Err(error("accumulator operation missing"));
        };
        let TermKind::Soac(SoacOp::Reduce { ne, .. } | SoacOp::Scan { ne, .. }) = &term.kind else {
            return Err(error("invalid accumulator source"));
        };
        initial.push(body.source(owner, ne)?);
        operators.push(operation);
    }
    for (index, filter) in body.compiler.plan.counts(plan) {
        let index = scans.len() + usize::try_from(index).map_err(|_| error("negative count slot"))?;
        if index > operators.len() {
            return Err(error("noncontiguous count accumulator"));
        }
        let Some(source) = body.compiler.plan.source(filter) else {
            return Err(error("filter accumulator source missing"));
        };
        sources.insert(index, source);
        operators.insert(index, filter);
        initial.insert(index, body.literal("0", &types::i32())?);
    }
    let Some(domain) = body.compiler.plan.domain(stage.operation) else {
        return Err(error("collective domain missing"));
    };
    let n = body.extent(scope, domain)?;
    let extra = body.literal("63", &types::i32())?;
    let chunks = body.binary(BinaryOperator::Add, n.clone(), extra)?;
    let width = body.literal("64", &types::i32())?;
    let chunks = body.binary(BinaryOperator::Divide, chunks, width.clone())?;
    match stage.phase.as_str() {
        "ordered" => {
            let zero = body.literal("0", &types::i32())?;
            let one = body.literal("1", &types::i32())?;
            let state = body.counted(zero.clone(), n, one, initial, |body, index, state| {
                let mut cache = LookupMap::default();
                let mut next = Vec::new();
                for (i, &operator) in operators.iter().enumerate() {
                    let Some(input) = body.compiler.facts.input(operator, 0) else {
                        return Err(error("operation input missing"));
                    };
                    let value = element(body, scope, plan, input, index.clone(), &mut cache)?;
                    let value = accumulate_element(body, scope, operator, state[i].clone(), value)?;
                    cache.insert(sources[i], value.clone());
                    next.push(value);
                }
                write_arrays(body, scope, stage.operation, plan, results, index, &mut cache)?;
                Ok(next)
            })?;
            for (i, value) in state.into_iter().skip(scans.len()).enumerate() {
                if let Some(output) = body.slot(scope, stage.operation, "total", i as i64, 2)? {
                    store(body, output, zero.clone(), value)?;
                }
            }
        }
        "chunks" => {
            let (start, step) = invocation(body, stage.width)?;
            body.counted(start, chunks, step, vec![], |body, chunk, _| {
                let start = body.binary(BinaryOperator::Multiply, chunk.clone(), width.clone())?;
                let bound = body.binary(BinaryOperator::Add, start.clone(), width.clone())?;
                let cond = body.binary(BinaryOperator::Less, bound.clone(), n.clone())?;
                let n = body.cast(n.clone(), &bound.ty)?;
                let bound = body.select(cond, bound, n)?;
                let one = body.literal("1", &start.ty)?;
                let state = body.counted(start, bound, one, initial.clone(), |body, index, state| {
                    let mut cache = LookupMap::default();
                    let mut next = Vec::new();
                    for (i, &operator) in operators.iter().enumerate() {
                        let Some(input) = body.compiler.facts.input(operator, 0) else {
                            return Err(error("accumulator input missing"));
                        };
                        let filtered = body
                            .compiler
                            .facts
                            .operation(input)
                            .filter(|filter| body.compiler.plan.member(plan, *filter))
                            .filter(|_| {
                                body.compiler.program.identities.origins.get(&input).is_some_and(
                                    |(term, _)| matches!(term.kind, TermKind::Soac(SoacOp::Filter { .. })),
                                )
                            });
                        let value = if let Some(filter) = filtered {
                            let Some(source) = body.compiler.facts.input(filter, 0) else {
                                return Err(error("filtered reduction input missing"));
                            };
                            let incoming = element(body, scope, plan, source, index.clone(), &mut cache)?;
                            let keep = body.callback(scope, filter, vec![incoming.clone()])?;
                            body.branch(
                                scope,
                                keep,
                                |body| body.callback(scope, operator, vec![state[i].clone(), incoming]),
                                |_| Ok(state[i].clone()),
                                None,
                            )?
                        } else {
                            let incoming = element(body, scope, plan, input, index.clone(), &mut cache)?;
                            accumulate_element(body, scope, operator, state[i].clone(), incoming)?
                        };
                        if i < scans.len() {
                            if let Some(output) =
                                body.slot(scope, stage.operation, "prefix", i as i64, 2)?
                            {
                                store(body, output, index.clone(), value.clone())?;
                            }
                            cache.insert(sources[i], value.clone());
                        }
                        next.push(value);
                    }
                    if scans.is_empty() {
                        write_arrays(body, scope, stage.operation, plan, results, index, &mut cache)?;
                    } else {
                        for (role, i, source) in results {
                            if role != "mapped" {
                                continue;
                            }
                            if let Some(output) = body.slot(scope, stage.operation, "mapped", *i, 2)? {
                                let value = element(body, scope, plan, *source, index.clone(), &mut cache)?;
                                store(body, output, index.clone(), value)?;
                            }
                        }
                    }
                    Ok(next)
                })?;
                for (i, value) in state.into_iter().enumerate() {
                    if let Some(output) = body.slot(scope, stage.operation, "partial", i as i64, 2)? {
                        store(body, output, chunk.clone(), value)?;
                    }
                }
                Ok(vec![])
            })?;
        }
        "combine" if scans.is_empty() => {
            let lane = body.op(
                OpTag::Intrinsic {
                    id: catalog().known().local_id,
                    overload_idx: 0,
                },
                vec![],
                Type::Constructed(TypeName::UInt(32), vec![]),
            )?;
            let width = body.literal(&stage.width.to_string(), &chunks.ty)?;
            let extra = body.literal(&(stage.width - 1).to_string(), &chunks.ty)?;
            let count = body.binary(BinaryOperator::Add, chunks.clone(), extra)?;
            let per_lane = body.binary(BinaryOperator::Divide, count, width)?;
            let start = body.binary(BinaryOperator::Multiply, per_lane.clone(), lane.clone())?;
            let bound = body.binary(BinaryOperator::Add, start.clone(), per_lane)?;
            let cond = body.binary(BinaryOperator::Less, bound.clone(), chunks.clone())?;
            let bound = body.select(cond, bound, chunks)?;
            let one = body.literal("1", &start.ty)?;
            let state = body.counted(start, bound, one, initial.clone(), |body, index, state| {
                accumulate_slots(body, scope, stage.operation, &operators, &state, "partial", index)
            })?;
            let zero = body.literal("0", &types::i32())?;
            let count = body.literal(&stage.width.to_string(), &types::i32())?;
            let mut shared = Vec::new();
            for (i, value) in state.into_iter().enumerate() {
                let view = body.op(
                    OpTag::StorageView(PureViewSource::Workgroup {
                        id: i as u32,
                        count: stage.width,
                    }),
                    vec![zero.clone(), count.clone()],
                    interface::view_type(&value.ty, types::no_buffer()),
                )?;
                store(body, view.clone(), lane.clone(), value)?;
                shared.push(view);
            }
            body.builder.push_void_inst(InstKind::ControlBarrier).map_err(builder_error)?;
            for bit in 0..stage.width.trailing_zeros() {
                let stride = 1 << bit;
                let active = body.literal(&(stage.width / (2 * stride)).to_string(), &lane.ty)?;
                let condition = body.binary(BinaryOperator::Less, lane.clone(), active)?;
                body.when(condition, |body| {
                    let scale = body.literal(&(2 * stride).to_string(), &lane.ty)?;
                    let first = body.binary(BinaryOperator::Multiply, lane.clone(), scale)?;
                    let step = body.literal(&stride.to_string(), &lane.ty)?;
                    let second = body.binary(BinaryOperator::Add, first.clone(), step)?;
                    for (i, view) in shared.iter().enumerate() {
                        let a = body.index(view.clone(), first.clone())?;
                        let b = body.index(view.clone(), second.clone())?;
                        let value = combine_accumulator(body, scope, operators[i], a, b)?;
                        store(body, view.clone(), first.clone(), value)?;
                    }
                    Ok(())
                })?;
                body.builder.push_void_inst(InstKind::ControlBarrier).map_err(builder_error)?;
            }
            let condition = body.binary(BinaryOperator::Equal, lane, zero.clone())?;
            body.when(condition, |body| {
                for (i, view) in shared.iter().enumerate() {
                    if let Some(output) = body.slot(scope, stage.operation, "total", i as i64, 2)? {
                        let value = body.index(view.clone(), zero.clone())?;
                        store(body, output, zero.clone(), value)?;
                    }
                }
                Ok(())
            })?;
        }
        "combine" => {
            let zero = body.literal("0", &types::i32())?;
            let one = body.literal("1", &types::i32())?;
            let final_state = body.counted(zero.clone(), chunks, one, initial, |body, index, state| {
                for (i, value) in state.iter().take(scans.len()).enumerate() {
                    if let Some(output) = body.slot(scope, stage.operation, "offset", i as i64, 2)? {
                        store(body, output, index.clone(), value.clone())?;
                    }
                }
                accumulate_slots(body, scope, stage.operation, &operators, &state, "partial", index)
            })?;
            for (i, value) in final_state.into_iter().skip(scans.len()).enumerate() {
                if let Some(output) = body.slot(scope, stage.operation, "total", i as i64, 2)? {
                    store(body, output, zero.clone(), value)?;
                }
            }
        }
        "offsets" => {
            let (start, step) = invocation(body, stage.width)?;
            body.counted(start, n, step, vec![], |body, index, _| {
                let chunk = body.binary(BinaryOperator::Divide, index.clone(), width)?;
                let mut cache = LookupMap::default();
                for (i, &source) in scans.iter().enumerate() {
                    let Some(prefix) = body.slot(scope, stage.operation, "prefix", i as i64, 1)? else {
                        return Err(error("scan prefix is not materialized"));
                    };
                    let Some(offset) = body.slot(scope, stage.operation, "offset", i as i64, 1)? else {
                        return Err(error("scan offset is not materialized"));
                    };
                    let a = body.index(offset, chunk.clone())?;
                    let b = body.index(prefix, index.clone())?;
                    let value = combine_accumulator(body, scope, operators[i], a, b)?;
                    cache.insert(source, value);
                }
                for (role, i, source) in results {
                    if role == "mapped" {
                        if let Some(view) = body.slot(scope, stage.operation, "mapped", *i, 1)? {
                            cache.insert(*source, body.index(view, index.clone())?);
                        }
                    }
                }
                write_arrays(body, scope, stage.operation, plan, results, index, &mut cache)?;
                Ok(vec![])
            })?;
        }
        _ => return Err(error("invalid collective phase")),
    }
    Ok(())
}
fn is_count(body: &Body<'_, '_, '_>, operator: Value) -> bool {
    let Some(source) = body.compiler.plan.source(operator) else {
        return false;
    };
    body.compiler
        .program
        .identities
        .origins
        .get(&source)
        .is_some_and(|(term, _)| matches!(term.kind, TermKind::Soac(SoacOp::Filter { .. })))
}
fn accumulate_element(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operator: Value,
    state: Typed,
    incoming: Typed,
) -> Result<Typed, OptimizeError> {
    if is_count(body, operator) {
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
    if is_count(body, operator) {
        body.binary(BinaryOperator::Add, a, b)
    } else {
        body.callback(scope, operator, vec![a, b])
    }
}
fn accumulate_slots(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operation: Value,
    operators: &[Value],
    state: &[Typed],
    role: &str,
    index: Typed,
) -> Result<Vec<Typed>, OptimizeError> {
    let mut next = Vec::new();
    for (i, &operator) in operators.iter().enumerate() {
        let Some(input) = body.slot(scope, operation, role, i as i64, 1)? else {
            return Err(error("accumulator input is not materialized"));
        };
        let incoming = body.index(input, index.clone())?;
        next.push(combine_accumulator(
            body,
            scope,
            operator,
            state[i].clone(),
            incoming,
        )?);
    }
    Ok(next)
}
pub(super) fn write_arrays(
    body: &mut Body<'_, '_, '_>,
    scope: Value,
    operation: Value,
    plan: Value,
    results: &[(String, i64, Value)],
    index: Typed,
    cache: &mut LookupMap<Value, Typed>,
) -> Result<(), OptimizeError> {
    let mut position = 0;
    for (role, _, source) in results {
        if role != "array" {
            continue;
        }
        if let Some(output) = body.slot(scope, operation, "output", position, 2)? {
            let value = element(body, scope, plan, *source, index.clone(), cache)?;
            store(body, output, index.clone(), value)?;
        }
        position += 1;
    }
    Ok(())
}
