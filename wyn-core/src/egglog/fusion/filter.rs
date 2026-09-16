use super::*;

pub(in crate::egglog) fn length_source(data: &AssociatedData, kind: &OperationKind) -> Option<ExprId> {
    let OperationKind::Call { function, args } = kind else {
        return None;
    };
    let [array] = args.as_slice() else {
        return None;
    };
    let mut function = *function;
    while let ExprKind::Coerce(inner) = data.expressions[function].kind {
        function = inner;
    }
    let ExprKind::Builtin(id) = data.expressions[function].kind else {
        return None;
    };
    (data.builtins[id].builtin == crate::builtins::catalog().known().length).then_some(*array)
}

/// Mask reduction inputs and, when needed, append a shared count reduction.
/// Passing the filter as both IDs requests the length-only transformation.
pub(in crate::egglog) fn masked(
    data: &mut AssociatedData,
    producer: OperationId,
    consumer: OperationId,
) -> Option<()> {
    let OperationKind::Filter {
        map, body, inputs, ..
    } = data.operations[producer].kind.clone()
    else {
        return None;
    };
    let parent = data.operations[producer].region;
    let only_count = producer == consumer;
    let summary = super::super::snapshot::analyze(data);
    let mut lengths = BTreeSet::new();
    for &(p, c, role) in &summary.uses {
        if p != producer || role != super::super::snapshot::Role::Length {
            continue;
        }
        let array = length_source(data, &data.operations[c].kind)?;
        if data.operations[c].region != parent
            || !matches!(input(data, &Array::Value(array), Some(producer)), Input::Produced(0, ref s) if s.is_empty())
        {
            return None;
        }
        lengths.insert(c);
    }
    if only_count && lengths.is_empty() {
        return None;
    }
    let (mut form, consumer_inputs) = if only_count {
        (
            ScremaForm {
                pre: SoacBody::Identity(vec![]),
                scans: vec![],
                reductions: vec![],
                post: SoacBody::Identity(vec![]),
            },
            0,
        )
    } else {
        let OperationKind::Screma { form, inputs, .. } = data.operations[consumer].kind.clone() else {
            return None;
        };
        if !form.scans.is_empty()
            || form.reductions.is_empty()
            || !signature(&form.post).1.is_empty()
            || inputs.is_empty()
            || inputs.iter().any(
                |a| !matches!(input(data, a, Some(producer)), Input::Produced(0, ref s) if s.is_empty()),
            )
        {
            return None;
        }
        (form, inputs.len())
    };
    let parameters = signature(&map).0;
    let (region, args) = project::region(data, parent, &parameters);
    let mapped = project::invoke(data, &map, args)?;
    let predicate = project::invoke(data, &body, mapped.clone())?;
    let [condition] = predicate.as_slice() else {
        return None;
    };
    let [mapped] = mapped.as_slice() else {
        return None;
    };
    let values =
        if only_count { vec![] } else { project::invoke(data, &form.pre, vec![*mapped; consumer_inputs])? };
    let neutrals: Vec<_> = form.reductions.iter().flat_map(|r| &r.neutral).copied().collect();
    if values.len() != neutrals.len() {
        return None;
    }
    let mut values: Vec<_> = values
        .into_iter()
        .zip(neutrals)
        .map(|(v, n)| {
            expr(
                data,
                data.expressions[v].ty,
                ExprKind::If {
                    condition: *condition,
                    then_value: v,
                    else_value: n,
                },
            )
        })
        .collect();
    let mut fields = if only_count { vec![] } else { result_types(data, consumer) };
    let count_slot = fields.len();
    if let Some(&length) = lengths.first() {
        let count_ty = data.operations[length].ty;
        if lengths.iter().any(|&id| data.operations[id].ty != count_ty) {
            return None;
        }
        let zero = expr(data, count_ty, ExprKind::Int("0".into()));
        let one = expr(data, count_ty, ExprKind::Int("1".into()));
        values.push(expr(
            data,
            count_ty,
            ExprKind::If {
                condition: *condition,
                then_value: one,
                else_value: zero,
            },
        ));
        let (combine, args) = project::region(data, parent, &[count_ty, count_ty]);
        let t = data.types[count_ty].ty.clone();
        let function_ty = ty(data, types::function(t.clone(), types::function(t.clone(), t)));
        let function = expr(data, function_ty, ExprKind::BinOp("+".into()));
        let sum = expr(data, count_ty, ExprKind::PureApp { function, args });
        let operator = project::finish(data, combine, vec![count_ty, count_ty], vec![sum]);
        form.reductions.push(Reduction {
            operator,
            neutral: vec![zero],
            commutative: true,
        });
        fields.push(count_ty);
    }
    form.pre = project::finish(data, region, parameters, values);

    let old_ty = data.operations[consumer].ty;
    let result_ty = ty(
        data,
        types::tuple(fields.iter().map(|t| data.types[*t].ty.clone()).collect()),
    );
    let old_values: Vec<_> = data.expressions.iter().map(|(&id, e)| (id, e.kind.clone())).collect();
    let result = expr(data, result_ty, ExprKind::OperationResult(consumer));
    let outputs: Vec<_> = fields
        .iter()
        .enumerate()
        .map(|(index, &t)| expr(data, t, ExprKind::Project { tuple: result, index }))
        .collect();
    let previous =
        (!only_count).then(|| expr(data, old_ty, ExprKind::Tuple(outputs[..count_slot].to_vec())));
    let mut substitutions = BTreeMap::new();
    for (id, kind) in old_values {
        if id == result {
            continue;
        }
        if let ExprKind::OperationResult(op) = kind {
            if lengths.contains(&op) {
                substitutions.insert(id, outputs[count_slot]);
            } else if op == consumer {
                if let Some(previous) = previous {
                    substitutions.insert(id, previous);
                }
            }
        }
    }
    data.operations[consumer].ty = result_ty;
    data.operations[consumer].kind = OperationKind::Screma {
        form,
        inputs,
        ownership: vec![types::SoacOwnership::Fresh; fields.len()],
    };
    if !only_count {
        data.regions[parent].members.remove(&producer);
    }
    for id in lengths {
        data.regions[parent].members.remove(&id);
    }
    rewrite::all(data, &substitutions);
    Some(())
}
