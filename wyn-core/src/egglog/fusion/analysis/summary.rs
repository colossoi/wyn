//! Decode local callback dependencies directly into a sink.
use super::sink::Sink;
use crate::egglog::data::{ExprId, ExprKind, Ir, ParameterId, SoacBody};
use egglog_engine::Error;
use std::collections::BTreeMap;

fn scalar<S: Sink>(
    data: &Ir,
    id: ExprId,
    parameters: &BTreeMap<ParameterId, S::Dependency>,
    memo: &mut BTreeMap<ExprId, S::Dependency>,
    sink: &mut S,
) -> Result<Option<S::Dependency>, Error> {
    if let Some(&v) = memo.get(&id) {
        return Ok(Some(v));
    }
    // This summary describes result dependencies, not execution. Both modes
    // retain their condition and alternatives without losing field structure.
    if let Some((choice, _)) = data.conditional_value(id) {
        let (Some(c), Some(a), Some(b)) = (
            scalar(data, choice.condition, parameters, memo, sink)?,
            scalar(data, choice.yes, parameters, memo, sink)?,
            scalar(data, choice.no, parameters, memo, sink)?,
        ) else {
            return Ok(None);
        };
        let v = sink.choice(c, a, b)?;
        memo.insert(id, v);
        return Ok(Some(v));
    }
    let v = match &data.expressions[id].kind {
        ExprKind::Parameter(p) => parameters.get(p).copied().unwrap_or_else(|| sink.independent()),
        ExprKind::Project { tuple, index } => {
            let Some(v) = scalar(data, *tuple, parameters, memo, sink)? else {
                return Ok(None);
            };
            sink.field(v, *index)?
        }
        ExprKind::Tuple(xs) | ExprKind::Vector(xs) | ExprKind::PureApp { args: xs, .. } => {
            let mut vs = Vec::with_capacity(xs.len());
            for &v in xs {
                let Some(v) = scalar(data, v, parameters, memo, sink)? else {
                    return Ok(None);
                };
                vs.push(v);
            }
            if matches!(data.expressions[id].kind, ExprKind::Tuple(_)) {
                sink.tuple(&vs)?
            } else {
                sink.all(&vs)?
            }
        }
        ExprKind::Coerce(v) => {
            let Some(v) = scalar(data, *v, parameters, memo, sink)? else {
                return Ok(None);
            };
            sink.all(&[v])?
        }
        ExprKind::OperationResult(_)
        | ExprKind::Lambda(_)
        | ExprKind::Closure { .. }
        | ExprKind::Array(_) => return Ok(None),
        _ => sink.independent(),
    };
    memo.insert(id, v);
    Ok(Some(v))
}

#[cfg(test)]
#[path = "summary_tests.rs"]
mod tests;

pub(super) fn invoke<S: Sink>(
    data: &Ir,
    body: &SoacBody,
    args: Vec<S::Dependency>,
    sink: &mut S,
) -> Result<Option<Vec<S::Dependency>>, Error> {
    match body {
        SoacBody::Identity(_) => Ok(Some(args)),
        SoacBody::Route { indices, .. } => Ok(indices.iter().map(|i| args.get(*i).copied()).collect()),
        SoacBody::Compose { first, then } => {
            let Some(args) = invoke(data, first, args, sink)? else {
                return Ok(None);
            };
            invoke(data, then, args, sink)
        }
        SoacBody::Parallel { left, right } => {
            let Some(mut values) = invoke(data, left, args.clone(), sink)? else {
                return Ok(None);
            };
            let Some(right) = invoke(data, right, args, sink)? else {
                return Ok(None);
            };
            values.extend(right);
            Ok(Some(values))
        }
        SoacBody::Apply { region, captures, .. } => {
            let region = &data.regions[*region];
            if !region.members.is_empty() || region.parameters.len() != args.len() + captures.len() {
                return Ok(None);
            }
            let parameters = region
                .parameters
                .iter()
                .copied()
                .zip(args.into_iter().chain(captures.iter().map(|_| sink.independent())))
                .collect();
            let mut memo = BTreeMap::new();
            let mut results = Vec::with_capacity(region.results.len());
            for &v in &region.results {
                let Some(v) = scalar(data, v, &parameters, &mut memo, sink)? else {
                    return Ok(None);
                };
                results.push(v);
            }
            Ok(Some(results))
        }
    }
}
