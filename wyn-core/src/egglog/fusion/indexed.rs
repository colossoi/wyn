use super::*;

/// Indexed replacement is profitable only when every observer is an element
/// demand and the number of demands is small. Keep complete-array observers.
pub(in crate::egglog) fn demands(
    data: &AssociatedData,
    producer: OperationId,
) -> Option<Vec<(OperationId, usize, Vec<usize>)>> {
    let OperationKind::Screma { form, inputs, .. } = &data.operations[producer].kind else {
        return None;
    };
    if counts(form) != (0, 0) {
        return None;
    }
    let summary = super::super::snapshot::fusion(data);
    if summary.observed.contains(&producer) {
        return None;
    }
    let mut demands = BTreeMap::new();
    for &(p, c, _) in &summary.uses {
        if p != producer {
            continue;
        }
        if data.operations[c].region != data.operations[producer].region {
            return None;
        }
        let OperationKind::Index { array, index } = &data.operations[c].kind else {
            return None;
        };
        let (slot, path) = projection(data, *array, producer)?;
        // An index computed from the producer would recreate a dependency cycle.
        let mut seen = BTreeSet::new();
        fn uses(data: &AssociatedData, v: ExprId, op: OperationId, seen: &mut BTreeSet<ExprId>) -> bool {
            if !seen.insert(v) {
                return false;
            }
            match &data.expressions[v].kind {
                ExprKind::OperationResult(p) => *p == op,
                ExprKind::Project { tuple, .. } | ExprKind::Coerce(tuple) => uses(data, *tuple, op, seen),
                ExprKind::Tuple(vs) | ExprKind::Vector(vs) | ExprKind::PureApp { args: vs, .. } => {
                    vs.iter().any(|v| uses(data, *v, op, seen))
                }
                ExprKind::If {
                    condition,
                    then_value,
                    else_value,
                } => [condition, then_value, else_value].into_iter().any(|v| uses(data, *v, op, seen)),
                _ => false,
            }
        }
        if uses(data, *index, producer, &mut seen) {
            return None;
        }
        demands.insert(c, (slot, path));
    }
    let limit = inputs
        .first()
        .and_then(|a| match a {
            Array::Literal(xs) => Some(xs.len()),
            Array::Value(v) => match data.types[data.expressions[*v].ty].ty.array_size() {
                Some(types::Type::Constructed(types::TypeName::Size(n), _)) => Some(*n as usize),
                _ => None,
            },
            _ => None,
        })
        .unwrap_or(2);
    (!demands.is_empty() && demands.len() <= limit)
        .then(|| demands.into_iter().map(|(op, (slot, path))| (op, slot, path)).collect())
}

// TLC may represent an array of tuples as projected component arrays. The
// first projection selects the operation result; the rest select its element.
fn projection(data: &AssociatedData, value: ExprId, producer: OperationId) -> Option<(usize, Vec<usize>)> {
    fn walk(
        data: &AssociatedData,
        value: ExprId,
        producer: OperationId,
        path: &mut Vec<usize>,
    ) -> Option<()> {
        match data.expressions[value].kind {
            ExprKind::Coerce(v) => walk(data, v, producer, path),
            ExprKind::Project { tuple, index } => {
                walk(data, tuple, producer, path)?;
                path.push(index);
                Some(())
            }
            ExprKind::OperationResult(op) if op == producer => Some(()),
            _ => None,
        }
    }
    let mut path = vec![];
    walk(data, value, producer, &mut path)?;
    let (&slot, path) = path.split_first()?;
    Some((slot, path.to_vec()))
}
fn read(data: &mut AssociatedData, region: RegionId, array: &Array, index: ExprId) -> Option<ExprId> {
    if let Array::Zip(xs) = array {
        let values = xs.iter().map(|a| read(data, region, a, index)).collect::<Option<Vec<_>>>()?;
        let t = ty(
            data,
            types::tuple(values.iter().map(|v| data.types[data.expressions[*v].ty].ty.clone()).collect()),
        );
        return Some(expr(data, t, ExprKind::Tuple(values)));
    }
    let t = element(data, array)?;
    let v = match array {
        Array::Value(v) => *v,
        _ => {
            let array_ty = match array {
                Array::Literal(xs) => types::sized_array(xs.len(), data.types[t].ty.clone()),
                _ => types::make_array1(
                    data.types[t].ty.clone(),
                    types::Type::Constructed(types::TypeName::ArrayVariantVirtual, vec![]),
                    types::Type::Constructed(types::TypeName::SizePlaceholder, vec![]),
                    types::no_buffer(),
                ),
            };
            let array_ty = ty(data, array_ty);
            expr(data, array_ty, ExprKind::Array(array.clone()))
        }
    };
    Some(project::operation(
        data,
        region,
        t,
        OperationKind::Index { array: v, index },
    ))
}
pub(in crate::egglog) fn indexed(data: &mut AssociatedData, producer: OperationId) -> Option<()> {
    let demands = demands(data, producer)?;
    let OperationKind::Screma { form, inputs, .. } = data.operations[producer].kind.clone() else {
        return None;
    };
    let parent = data.operations[producer].region;
    for (consumer, slot, path) in demands {
        let OperationKind::Index { index, .. } = data.operations[consumer].kind else {
            return None;
        };
        let (r, _) = project::region(data, parent, &[]);
        let args = inputs.iter().map(|a| read(data, r, a, index)).collect::<Option<Vec<_>>>()?;
        let pre = project::call(data, r, &form.pre, args)?;
        let post = project::call(data, r, &form.post, pre)?;
        let mut value = *post.get(slot)?;
        for index in path {
            let types::Type::Constructed(types::TypeName::Tuple(_), fields) =
                &data.types[data.expressions[value].ty].ty
            else {
                return None;
            };
            let field = fields.get(index)?.clone();
            let t = ty(data, field);
            value = expr(data, t, ExprKind::Project { tuple: value, index });
        }
        let body = project::finish(data, r, vec![], vec![value]);
        let SoacBody::Apply { captures, .. } = body else {
            return None;
        };
        let ret = data.types[data.operations[consumer].ty].ty.clone();
        let fty = captures.iter().rev().fold(ret, |r, a| {
            types::function(data.types[data.expressions[*a].ty].ty.clone(), r)
        });
        let t = ty(data, fty);
        let function = expr(data, t, ExprKind::Lambda(r));
        data.operations[consumer].kind = OperationKind::Call {
            function,
            args: captures,
        };
    }
    data.regions[parent].members.remove(&producer);
    Some(())
}
