use super::*;

pub(super) fn all(data: &mut AssociatedData, replacements: &BTreeMap<ExprId, ExprId>) {
    let mut memo = replacements.clone();
    let operations: Vec<_> = data.operations.iter().map(|(&id, o)| (id, o.kind.clone())).collect();
    for (id, mut kind) in operations {
        operation(data, &mut kind, &mut memo);
        data.operations[id].kind = kind;
    }
    let regions: Vec<_> = data.regions.iter().map(|(&id, r)| (id, r.results.clone())).collect();
    for (id, values) in regions {
        data.regions[id].results = values.into_iter().map(|v| value(data, v, &mut memo)).collect();
    }
}
pub(super) fn value(data: &mut AssociatedData, id: ExprId, memo: &mut BTreeMap<ExprId, ExprId>) -> ExprId {
    if let Some(&v) = memo.get(&id) {
        return v;
    }
    let ExprData { ty, mut kind } = data.expressions[id].clone();
    match &mut kind {
        ExprKind::PureApp { function, args } => {
            *function = value(data, *function, memo);
            values(data, args, memo);
        }
        ExprKind::Closure { captures, .. } | ExprKind::Tuple(captures) | ExprKind::Vector(captures) => {
            values(data, captures, memo)
        }
        ExprKind::Coerce(v) => *v = value(data, *v, memo),
        ExprKind::Project { tuple, index } => {
            *tuple = value(data, *tuple, memo);
            if let ExprKind::Tuple(fields) = &data.expressions[*tuple].kind {
                if let Some(&v) = fields.get(*index) {
                    memo.insert(id, v);
                    return v;
                }
            }
        }
        ExprKind::If {
            condition,
            then_value,
            else_value,
        } => {
            *condition = value(data, *condition, memo);
            *then_value = value(data, *then_value, memo);
            *else_value = value(data, *else_value, memo);
        }
        ExprKind::Array(a) => array(data, a, memo),
        _ => {}
    }
    let v = expr(data, ty, kind);
    memo.insert(id, v);
    v
}
fn values(data: &mut AssociatedData, values: &mut [ExprId], memo: &mut BTreeMap<ExprId, ExprId>) {
    for v in values {
        *v = value(data, *v, memo);
    }
}
fn array(data: &mut AssociatedData, a: &mut Array, memo: &mut BTreeMap<ExprId, ExprId>) {
    match a {
        Array::Value(v) => *v = value(data, *v, memo),
        Array::Literal(vs) => values(data, vs, memo),
        Array::Zip(xs) => {
            for x in xs {
                array(data, x, memo)
            }
        }
        Array::Range { start, len, step } => {
            *start = value(data, *start, memo);
            *len = value(data, *len, memo);
            if let Some(s) = step {
                *s = value(data, *s, memo);
            }
        }
    }
}
fn body(data: &mut AssociatedData, b: &mut SoacBody, memo: &mut BTreeMap<ExprId, ExprId>) {
    match b {
        SoacBody::Apply { captures, .. } => values(data, captures, memo),
        SoacBody::Compose { first, then } => {
            body(data, first, memo);
            body(data, then, memo);
        }
        SoacBody::Parallel { left, right } => {
            body(data, left, memo);
            body(data, right, memo);
        }
        _ => {}
    }
}
pub(super) fn operation(
    data: &mut AssociatedData,
    k: &mut OperationKind,
    memo: &mut BTreeMap<ExprId, ExprId>,
) {
    match k {
        OperationKind::Call { function, args } => {
            *function = value(data, *function, memo);
            values(data, args, memo);
        }
        OperationKind::EvalGlobal(_) => {}
        OperationKind::If { condition, .. } => *condition = value(data, *condition, memo),
        OperationKind::Loop { init, kind, .. } => {
            *init = value(data, *init, memo);
            if let LoopKind::For(v) | LoopKind::ForRange(v) = kind {
                *v = value(data, *v, memo);
            }
        }
        OperationKind::Index { array, index } => {
            *array = value(data, *array, memo);
            *index = value(data, *index, memo);
        }
        OperationKind::Screma { form, inputs, .. } => {
            for a in inputs {
                array(data, a, memo);
            }
            body(data, &mut form.pre, memo);
            body(data, &mut form.post, memo);
            for s in &mut form.scans {
                body(data, &mut s.operator, memo);
                values(data, &mut s.neutral, memo);
            }
            for r in &mut form.reductions {
                body(data, &mut r.operator, memo);
                values(data, &mut r.neutral, memo);
            }
        }
        OperationKind::Filter {
            body: b, map, inputs, ..
        } => {
            body(data, b, memo);
            body(data, map, memo);
            for a in inputs {
                array(data, a, memo);
            }
        }
        OperationKind::Scatter {
            destination,
            body: b,
            inputs,
        }
        | OperationKind::BucketScatter {
            destination,
            body: b,
            inputs,
            ..
        } => {
            destination.value = value(data, destination.value, memo);
            body(data, b, memo);
            for a in inputs {
                array(data, a, memo);
            }
        }
        OperationKind::ReduceByIndex {
            destination,
            map,
            body: b,
            neutral,
            inputs,
        } => {
            destination.value = value(data, destination.value, memo);
            body(data, map, memo);
            body(data, b, memo);
            *neutral = value(data, *neutral, memo);
            for a in inputs {
                array(data, a, memo);
            }
        }
    }
}
