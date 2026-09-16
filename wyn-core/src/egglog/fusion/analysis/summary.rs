//! Decode local callback dependencies directly into a sink.
use super::sink::Sink;
use crate::egglog::data::{AssociatedData, ExprId, ExprKind, ParameterId, SoacBody};
use std::collections::BTreeMap;

fn scalar<S: Sink>(
    data: &AssociatedData,
    id: ExprId,
    parameters: &BTreeMap<ParameterId, S::Dependency>,
    memo: &mut BTreeMap<ExprId, S::Dependency>,
    sink: &mut S,
) -> Option<S::Dependency> {
    if let Some(&v) = memo.get(&id) {
        return Some(v);
    }
    let v = match &data.expressions[id].kind {
        ExprKind::Parameter(p) => parameters.get(p).copied().unwrap_or_else(|| sink.independent()),
        ExprKind::Project { tuple, index } => {
            let v = scalar(data, *tuple, parameters, memo, sink)?;
            sink.field(v, *index)
        }
        ExprKind::Tuple(xs) | ExprKind::Vector(xs) | ExprKind::PureApp { args: xs, .. } => {
            let vs =
                xs.iter().map(|v| scalar(data, *v, parameters, memo, sink)).collect::<Option<Vec<_>>>()?;
            if matches!(data.expressions[id].kind, ExprKind::Tuple(_)) {
                sink.tuple(&vs)
            } else {
                sink.all(&vs)
            }
        }
        ExprKind::If {
            condition,
            then_value,
            else_value,
        } => {
            let c = scalar(data, *condition, parameters, memo, sink)?;
            let a = scalar(data, *then_value, parameters, memo, sink)?;
            let b = scalar(data, *else_value, parameters, memo, sink)?;
            sink.choice(c, a, b)
        }
        ExprKind::Coerce(v) => {
            let v = scalar(data, *v, parameters, memo, sink)?;
            sink.all(&[v])
        }
        ExprKind::OperationResult(_)
        | ExprKind::Lambda(_)
        | ExprKind::Closure { .. }
        | ExprKind::Array(_) => return None,
        _ => sink.independent(),
    };
    memo.insert(id, v);
    Some(v)
}

pub(super) fn invoke<S: Sink>(
    data: &AssociatedData,
    body: &SoacBody,
    args: Vec<S::Dependency>,
    sink: &mut S,
) -> Option<Vec<S::Dependency>> {
    match body {
        SoacBody::Identity(_) => Some(args),
        SoacBody::Route { indices, .. } => indices.iter().map(|i| args.get(*i).copied()).collect(),
        SoacBody::Compose { first, then } => {
            let args = invoke(data, first, args, sink)?;
            invoke(data, then, args, sink)
        }
        SoacBody::Parallel { left, right } => {
            let mut values = invoke(data, left, args.clone(), sink)?;
            values.extend(invoke(data, right, args, sink)?);
            Some(values)
        }
        SoacBody::Apply { region, captures, .. } => {
            let region = &data.regions[*region];
            if !region.members.is_empty() || region.parameters.len() != args.len() + captures.len() {
                return None;
            }
            let parameters = region
                .parameters
                .iter()
                .copied()
                .zip(args.into_iter().chain(captures.iter().map(|_| sink.independent())))
                .collect();
            let mut memo = BTreeMap::new();
            region.results.iter().map(|v| scalar(data, *v, &parameters, &mut memo, sink)).collect()
        }
    }
}
