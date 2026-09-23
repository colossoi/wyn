//! Decode operation-local properties and edges into a typed sink.
use super::{
    counts, indexed_demand, input_slices, inputs, produced_input, references, routes, summary, InputSite,
    Kind, Operation, Role, Sink,
};
use crate::egglog::data::{
    body_signature, is_slice, length_source, value_source, Array, ExprKind, Ir, OperationId, OperationKind,
    RegionId, SoacBody,
};
use crate::egglog::dependencies::{safe_body, Dependencies};
use crate::types::{Type, TypeExt, TypeName};
use egglog_engine::Error;
use std::collections::{BTreeMap, BTreeSet};

fn invoke<S: Sink>(
    data: &Ir,
    body: &SoacBody,
    args: Vec<S::Dependency>,
    sink: &mut S,
) -> Result<(bool, Vec<S::Dependency>), Error> {
    if let Some(values) = summary::invoke(data, body, args.clone(), sink)? {
        Ok((true, values))
    } else {
        Ok((false, vec![sink.all(&args)?; body_signature(body).1.len()]))
    }
}

/// Import local call/control-flow edges. Egglog proves the transitive closure;
/// unknown calls, storage reads and authored writes have no scalar proof.
pub(super) fn scalar_regions(
    data: &Ir,
    execution: &Dependencies,
    sink: &mut impl Sink,
) -> Result<(), Error> {
    let definitions: BTreeMap<_, _> = data.definitions.values().map(|d| (d.symbol, d.body)).collect();
    for (&region, body) in &data.regions {
        sink.scalar_region(region, &body.members.iter().copied().collect::<Vec<_>>())?;
        for &op in &body.members {
            if execution.discardable.contains(&op) {
                sink.scalar_operation(op, &[])?;
                continue;
            }
            let kind = &data.operations[op].kind;
            let regions = match kind {
                OperationKind::Call { function, .. } => {
                    let callee = match &data.expressions[value_source(data, *function)].kind {
                        ExprKind::Global(symbol) | ExprKind::Closure { code: symbol, .. } => {
                            definitions.get(symbol).copied()
                        }
                        ExprKind::Lambda(region) => Some(*region),
                        _ => None,
                    };
                    let Some(region) = callee else {
                        continue;
                    };
                    vec![region]
                }
                OperationKind::EvalGlobal(symbol) => {
                    let Some(&region) = definitions.get(symbol) else {
                        continue;
                    };
                    vec![region]
                }
                OperationKind::If { .. } | OperationKind::Loop { .. } => kind.structured_regions(),
                OperationKind::Screma { .. } => kind
                    .callbacks()
                    .into_iter()
                    .filter_map(|body| match body {
                        SoacBody::Apply { region, .. } => Some(*region),
                        _ => None,
                    })
                    .collect(),
                _ => continue,
            };
            sink.scalar_operation(op, &regions)?;
        }
    }
    Ok(())
}

pub(super) fn operations<S: Sink>(
    data: &Ir,
    execution: &Dependencies,
    schedules: &BTreeMap<RegionId, Vec<OperationId>>,
    sink: &mut S,
) -> Result<(), Error> {
    for &id in schedules.values().flatten() {
        let op = &data.operations[id];
        let mut fact = Operation {
            kind: Kind::Other,
            scans: 0,
            reductions: 0,
            arrays: 0,
            pre_projectable: false,
            post_projectable: false,
            predicate_projectable: false,
            safe: execution.discardable.contains(&id),
            movable: execution.movable.contains(&id),
            element_consumer: false,
            demand_limit: 2,
        };
        let args =
            inputs(&op.kind).iter().map(|a| element_value(data, a, sink)).collect::<Result<Vec<_>, _>>()?;
        let values = match &op.kind {
            OperationKind::Screma { form, .. } => {
                fact.kind = Kind::Screma;
                (fact.scans, fact.reductions) = counts(form);
                let (pure, before) = invoke(data, &form.pre, args, sink)?;
                fact.pre_projectable = pure;
                for &value in &before[..fact.scans + fact.reductions] {
                    sink.collective_dependency(id, value)?;
                }
                let after = (0..fact.scans)
                    .map(|_| sink.scan(id))
                    .chain(before[fact.scans + fact.reductions..].iter().copied().map(Ok))
                    .collect::<Result<_, _>>()?;
                let (pure, post) = invoke(data, &form.post, after, sink)?;
                fact.post_projectable = pure;
                fact.arrays = post.len();
                (0..fact.reductions).map(|_| sink.independent()).chain(post).collect::<Vec<_>>()
            }
            OperationKind::Filter { map, body, post, .. } => {
                fact.kind = Kind::Filter;
                let (pure, values) = invoke(data, map, args, sink)?;
                fact.pre_projectable = pure && values.len() == 1;
                let (pure, predicate) = invoke(data, body, values.clone(), sink)?;
                fact.predicate_projectable = pure && predicate.len() == 1;
                let (pure, values) = invoke(data, post, values, sink)?;
                fact.post_projectable = pure && values.len() == 1;
                fact.arrays = 1;
                fact.element_consumer = safe_body(map, &execution.safe_regions)
                    && safe_body(body, &execution.safe_regions)
                    && safe_body(post, &execution.safe_regions);
                values
            }
            OperationKind::ReduceByIndex { map, body, .. } => {
                fact.kind = Kind::Element;
                fact.pre_projectable = invoke(data, map, args, sink)?.0;
                fact.element_consumer =
                    safe_body(map, &execution.safe_regions) && safe_body(body, &execution.safe_regions);
                vec![]
            }
            OperationKind::Scatter { body, .. } | OperationKind::BucketScatter { body, .. } => {
                fact.kind = Kind::Element;
                fact.pre_projectable = invoke(data, body, args, sink)?.0;
                fact.element_consumer = safe_body(body, &execution.safe_regions)
                    && !matches!(&op.kind, OperationKind::BucketScatter { shape, .. } if data.bucket_shapes[*shape].domain_rank != 1);
                vec![]
            }
            _ => vec![],
        };
        for (slot, value) in values.into_iter().enumerate() {
            sink.output_dependency(id, slot, value)?;
        }
        let arrays = inputs(&op.kind);
        fact.demand_limit = arrays.first().and_then(|a| fixed(data, a)).map_or(2, |n| n as usize);
        sink.operation(id, op.region, fact)?;
        if let Some(first) = arrays.first() {
            domain(data, first, id, sink)?;
        }
        for (slot, a) in arrays.iter().enumerate() {
            input_facts(
                data,
                a,
                false,
                id.as_u32() as u64 * (1 << 32) + slot as u64,
                &mut |site| sink.input(id, site),
            )?;
            read_resources(data, a, id, sink)?;
        }
        match &op.kind {
            OperationKind::Scatter { destination, .. }
            | OperationKind::BucketScatter { destination, .. }
            | OperationKind::ReduceByIndex { destination, .. } => {
                sink.write_resource(id, destination.value)?;
            }
            OperationKind::Index { index, .. } => {
                let mut refs = BTreeSet::new();
                references(data, *index, &mut refs);
                for v in refs {
                    if let ExprKind::OperationResult(p) = data.expressions[v].kind {
                        sink.index_address(p, id)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub(super) fn usage(
    data: &Ir,
    producer: OperationId,
    consumer: OperationId,
    role: Role,
    internal: bool,
    sink: &mut impl Sink,
) -> Result<(), Error> {
    sink.usage(producer, consumer, role, internal)?;
    if !internal {
        return Ok(());
    }
    match role {
        Role::Input => {
            let slots = routes(data, producer, consumer);
            if slots.is_empty() {
                sink.blocked_stream(producer, consumer)?;
            } else {
                let Some(slices) = input_slices(data, producer, consumer) else {
                    sink.blocked_stream(producer, consumer)?;
                    return Ok(());
                };
                sink.stream(producer, consumer, &slices)?;
                if matches!(&data.operations[producer].kind, OperationKind::Screma { form, .. } if slots.iter().any(|i| *i < counts(form).1))
                {
                    sink.blocked_stream(producer, consumer)?;
                }
            }
        }
        Role::Length => {
            let direct = length_source(data, &data.operations[consumer].kind).is_some_and(|v| matches!(produced_input(data, &Array::Value(v), producer), Some((0, ref s)) if s.is_empty()));
            sink.length(producer, consumer, data.operations[consumer].ty, direct)?;
        }
        Role::Argument
            if matches!(data.operations[producer].kind, OperationKind::Screma { .. })
                && indexed_demand(data, producer, consumer).is_some() =>
        {
            sink.indexed(producer, consumer)?;
        }
        _ => {}
    }
    Ok(())
}

fn read_resources(data: &Ir, a: &Array, op: OperationId, sink: &mut impl Sink) -> Result<(), Error> {
    match a {
        Array::Value(v) => {
            sink.read_resource(op, *v)?;
            match &data.expressions[*v].kind {
                ExprKind::Coerce(v) => read_resources(data, &Array::Value(*v), op, sink)?,
                ExprKind::Array(a) => read_resources(data, a, op, sink)?,
                _ => {}
            }
        }
        Array::Zip(xs) => {
            for a in xs {
                read_resources(data, a, op, sink)?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn fixed(data: &Ir, a: &Array) -> Option<u64> {
    match a {
        Array::Literal(xs) => Some(xs.len() as u64),
        Array::Value(id) => match data.types[data.expressions[*id].ty].ty.array_size() {
            Some(Type::Constructed(TypeName::Size(n), _)) => Some(*n as u64),
            _ => None,
        },
        _ => None,
    }
}
fn domain(data: &Ir, a: &Array, operation: OperationId, sink: &mut impl Sink) -> Result<(), Error> {
    match a {
        Array::Zip(xs) if !xs.is_empty() => domain(data, &xs[0], operation, sink),
        Array::Value(id) => match &data.expressions[*id].kind {
            ExprKind::Coerce(v) => domain(data, &Array::Value(*v), operation, sink),
            ExprKind::Array(a) => domain(data, a, operation, sink),
            ExprKind::Project { tuple, .. } => {
                if let ExprKind::OperationResult(op) = data.expressions[*tuple].kind {
                    if matches!(&data.operations[op].kind, OperationKind::Screma { inputs, .. } if !inputs.is_empty())
                    {
                        return sink.domain_source(operation, op);
                    }
                }
                sink.domain(operation, a, fixed(data, a))
            }
            _ => sink.domain(operation, a, fixed(data, a)),
        },
        _ => sink.domain(operation, a, fixed(data, a)),
    }
}
fn element_value<S: Sink>(data: &Ir, a: &Array, sink: &mut S) -> Result<S::Dependency, Error> {
    match a {
        Array::Zip(xs) => {
            let values = xs.iter().map(|a| element_value(data, a, sink)).collect::<Result<Vec<_>, _>>()?;
            sink.tuple(&values)
        }
        Array::Value(id) => match &data.expressions[value_source(data, *id)].kind {
            ExprKind::OperationResult(op) => sink.output(*op, 0),
            ExprKind::Coerce(v) => element_value(data, &Array::Value(*v), sink),
            ExprKind::Array(a) => element_value(data, a, sink),
            ExprKind::Project { tuple, index } => {
                if let ExprKind::OperationResult(op) = data.expressions[*tuple].kind {
                    sink.output(op, *index)
                } else {
                    let v = element_value(data, &Array::Value(*tuple), sink)?;
                    sink.field(v, *index)
                }
            }
            ExprKind::PureApp { function, args } if is_slice(data, *function) => {
                element_value(data, &Array::Value(args[0]), sink)
            }
            _ => Ok(sink.independent()),
        },
        _ => Ok(sink.independent()),
    }
}
fn input_facts(
    data: &Ir,
    a: &Array,
    indirect: bool,
    token: u64,
    emit: &mut impl FnMut(InputSite) -> Result<(), Error>,
) -> Result<(), Error> {
    match a {
        Array::Zip(xs) => {
            for x in xs {
                input_facts(data, x, true, token, emit)?;
            }
            Ok(())
        }
        Array::Value(id) => match &data.expressions[value_source(data, *id)].kind {
            ExprKind::OperationResult(op) => {
                emit(if indirect { InputSite::Indirect(*op, token) } else { InputSite::Direct(*op) })
            }
            ExprKind::Coerce(v) => input_facts(data, &Array::Value(*v), indirect, token, emit),
            ExprKind::Array(a) => input_facts(data, a, indirect, token, emit),
            ExprKind::Project { tuple, .. }
                if matches!(data.expressions[*tuple].kind, ExprKind::OperationResult(_)) =>
            {
                let ExprKind::OperationResult(op) = data.expressions[*tuple].kind else {
                    unreachable!("guarded projection {id:?} has a non-operation source {tuple:?}")
                };
                emit(if indirect { InputSite::Indirect(op, token) } else { InputSite::Direct(op) })
            }
            _ => {
                let mut refs = BTreeSet::new();
                references(data, *id, &mut refs);
                let mut found = false;
                for id in refs {
                    if let ExprKind::OperationResult(op) = data.expressions[id].kind {
                        found = true;
                        emit(InputSite::Indirect(op, token))?;
                    }
                }
                if !found {
                    emit(InputSite::External(token))?;
                }
                Ok(())
            }
        },
        _ => emit(InputSite::External(token)),
    }
}
