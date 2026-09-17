//! Construct callback regions and project their pure scalar results.
use super::super::analysis::references;
use crate::ast::Span;
use crate::egglog::data::{
    intern_expr, intern_type, ExprData, ExprId, ExprKind, Ir, OperationData, OperationKind, ParameterData,
    RegionData, RegionId, SoacBody, TypeId,
};
use crate::egglog::rewrite::Rewriter;
use crate::types;
use crate::types::function;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn tuple(data: &mut Ir, parent: RegionId, parameters: Vec<TypeId>) -> SoacBody {
    let (region, args) = region(data, parent, &parameters);
    let t = intern_type(
        data,
        types::tuple(parameters.iter().map(|t| data.types[*t].ty.clone()).collect()),
    );
    let value = intern_expr(data, t, ExprKind::Tuple(args));
    finish(data, region, parameters, vec![value])
}
pub(super) fn region(data: &mut Ir, parent: RegionId, types: &[TypeId]) -> (RegionId, Vec<ExprId>) {
    let id = data.regions.alloc(RegionData {
        definition: data.regions[parent].definition,
        parent: Some(parent),
        parameters: vec![],
        members: Default::default(),
        results: vec![],
    });
    // Diagnostic name only; identity is the newly allocated ParameterId.
    let symbol = data.definitions[data.regions[parent].definition].symbol;
    let mut args = vec![];
    for &ty in types {
        let p = data.parameters.alloc(ParameterData {
            region: id,
            symbol,
            ty,
        });
        data.regions[id].parameters.push(p);
        args.push(intern_expr(data, ty, ExprKind::Parameter(p)));
    }
    (id, args)
}
pub(super) fn finish(
    data: &mut Ir,
    region: RegionId,
    parameters: Vec<TypeId>,
    values: Vec<ExprId>,
) -> SoacBody {
    let results = values.iter().map(|v| data.expressions[*v].ty).collect();
    let mut leaves = BTreeSet::new();
    for &v in &values {
        references(data, v, &mut leaves);
    }
    for &op in &data.regions[region].members {
        match &data.operations[op].kind {
            OperationKind::Call { args, .. } => {
                for &v in args {
                    references(data, v, &mut leaves);
                }
            }
            OperationKind::Index { array, index } => {
                references(data, *array, &mut leaves);
                references(data, *index, &mut leaves);
            }
            _ => {}
        }
    }
    let captures: Vec<_> = leaves
        .into_iter()
        .filter(|v| match data.expressions[*v].kind {
            ExprKind::Parameter(p) => data.parameters[p].region != region,
            ExprKind::OperationResult(op) => data.operations[op].region != region,
            _ => false,
        })
        .collect();
    let mut replacements = BTreeMap::new();
    let symbol = data.definitions[data.regions[region].definition].symbol;
    for &v in &captures {
        let ty = data.expressions[v].ty;
        let p = data.parameters.alloc(ParameterData { region, symbol, ty });
        data.regions[region].parameters.push(p);
        let e = intern_expr(data, ty, ExprKind::Parameter(p));
        replacements.insert(v, e);
    }
    let mut rewrite = Rewriter::new(data);
    for op in data.regions[region].members.clone() {
        let mut kind = data.operations[op].kind.clone();
        rewrite.operation(data, &mut kind, &mut replacements);
        data.operations[op].kind = kind;
    }
    data.regions[region].results =
        values.into_iter().map(|v| rewrite.value(data, v, &mut replacements)).collect();
    SoacBody::Apply {
        region,
        parameters,
        results,
        captures,
    }
}

/// Materialize opaque body calls into a new region without inspecting their
/// scalar implementations. Used when projection is unnecessary.
pub(super) fn call(
    data: &mut Ir,
    region: RegionId,
    body: &SoacBody,
    args: Vec<ExprId>,
) -> Option<Vec<ExprId>> {
    match body {
        SoacBody::Identity(_) => Some(args),
        SoacBody::Route { indices, .. } => indices.iter().map(|&i| args.get(i).copied()).collect(),
        SoacBody::Compose { first, then } => {
            let vs = call(data, region, first, args)?;
            call(data, region, then, vs)
        }
        SoacBody::Parallel { left, right } => {
            let mut vs = call(data, region, left, args.clone())?;
            vs.extend(call(data, region, right, args)?);
            Some(vs)
        }
        SoacBody::Apply {
            region: target,
            captures,
            results,
            ..
        } => {
            let args: Vec<_> = args.into_iter().chain(captures.iter().copied()).collect();
            let t = match results.as_slice() {
                [t] => *t,
                _ => intern_type(
                    data,
                    types::tuple(results.iter().map(|t| data.types[*t].ty.clone()).collect()),
                ),
            };
            let fty = args.iter().rev().fold(data.types[t].ty.clone(), |r, a| {
                function(data.types[data.expressions[*a].ty].ty.clone(), r)
            });
            let fty = intern_type(data, fty);
            let f = intern_expr(data, fty, ExprKind::Lambda(*target));
            let v = operation(data, region, t, OperationKind::Call { function: f, args });
            Some(if results.len() == 1 {
                vec![v]
            } else {
                results
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| intern_expr(data, t, ExprKind::Project { tuple: v, index: i }))
                    .collect()
            })
        }
    }
}
pub(super) fn operation(data: &mut Ir, region: RegionId, ty: TypeId, kind: OperationKind) -> ExprId {
    let id = data.operations.alloc(OperationData {
        region,
        ty,
        kind,
        source_position: data.regions[region].members.len(),
        span: Span::generated(),
    });
    data.regions[region].members.insert(id);
    intern_expr(data, ty, ExprKind::OperationResult(id))
}
fn field(data: &mut Ir, v: ExprId, i: usize, t: TypeId) -> ExprId {
    match data.expressions[v].kind.clone() {
        ExprKind::Tuple(vs) => vs[i],
        ExprKind::If {
            condition,
            then_value,
            else_value,
        } => {
            let a = field(data, then_value, i, t);
            let b = field(data, else_value, i, t);
            intern_expr(
                data,
                t,
                ExprKind::If {
                    condition,
                    then_value: a,
                    else_value: b,
                },
            )
        }
        _ => intern_expr(data, t, ExprKind::Project { tuple: v, index: i }),
    }
}
fn scalar(data: &mut Ir, id: ExprId, map: &mut BTreeMap<ExprId, ExprId>) -> Option<ExprId> {
    if let Some(&v) = map.get(&id) {
        return Some(v);
    }
    let ExprData { ty: t, kind } = data.expressions[id].clone();
    let result = match kind {
        ExprKind::Project { tuple, index } => {
            let v = scalar(data, tuple, map)?;
            field(data, v, index, t)
        }
        ExprKind::Tuple(vs) | ExprKind::Vector(vs) => {
            let vector = matches!(data.expressions[id].kind, ExprKind::Vector(_));
            let vs = vs.into_iter().map(|v| scalar(data, v, map)).collect::<Option<Vec<_>>>()?;
            intern_expr(
                data,
                t,
                if vector { ExprKind::Vector(vs) } else { ExprKind::Tuple(vs) },
            )
        }
        ExprKind::PureApp { function, args } => {
            let args = args.into_iter().map(|v| scalar(data, v, map)).collect::<Option<Vec<_>>>()?;
            intern_expr(data, t, ExprKind::PureApp { function, args })
        }
        ExprKind::If {
            condition,
            then_value,
            else_value,
        } => {
            let c = scalar(data, condition, map)?;
            let a = scalar(data, then_value, map)?;
            let b = scalar(data, else_value, map)?;
            intern_expr(
                data,
                t,
                ExprKind::If {
                    condition: c,
                    then_value: a,
                    else_value: b,
                },
            )
        }
        ExprKind::Coerce(v) => {
            let v = scalar(data, v, map)?;
            intern_expr(data, t, ExprKind::Coerce(v))
        }
        ExprKind::OperationResult(_)
        | ExprKind::Lambda(_)
        | ExprKind::Closure { .. }
        | ExprKind::Array(_) => return None,
        _ => id,
    };
    map.insert(id, result);
    Some(result)
}
pub(super) fn invoke(data: &mut Ir, body: &SoacBody, args: Vec<ExprId>) -> Option<Vec<ExprId>> {
    match body {
        SoacBody::Identity(_) => Some(args),
        SoacBody::Route { indices, .. } => indices.iter().map(|&i| args.get(i).copied()).collect(),
        SoacBody::Compose { first, then } => {
            let vs = invoke(data, first, args)?;
            invoke(data, then, vs)
        }
        SoacBody::Parallel { left, right } => {
            let mut vs = invoke(data, left, args.clone())?;
            vs.extend(invoke(data, right, args)?);
            Some(vs)
        }
        SoacBody::Apply { region, captures, .. } => {
            let record = data.regions[*region].clone();
            if !record.members.is_empty() {
                return None;
            }
            let args: Vec<_> = args.into_iter().chain(captures.iter().copied()).collect();
            if args.len() != record.parameters.len() {
                return None;
            }
            let substitutions: BTreeMap<_, _> = record.parameters.into_iter().zip(args).collect();
            let mut memo = data
                .expressions
                .iter()
                .filter_map(|(&id, e)| {
                    if let ExprKind::Parameter(p) = e.kind {
                        substitutions.get(&p).map(|&v| (id, v))
                    } else {
                        None
                    }
                })
                .collect();
            record.results.into_iter().map(|v| scalar(data, v, &mut memo)).collect()
        }
    }
}
