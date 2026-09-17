use super::body::{finish, invoke};
use super::{body, result_types};
use crate::egglog::data::{
    body_signature, intern_expr, intern_type, ExprKind, Ir, OperationId, OperationKind, Reduction,
    ScremaForm, SoacBody,
};
use crate::egglog::rewrite::all;
use crate::types::{function, tuple};
use std::collections::{BTreeMap, BTreeSet};

/// Mask reduction inputs and, when needed, append a shared count reduction.
/// Passing the filter as both IDs requests the length-only transformation.
pub(super) fn masked(
    data: &mut Ir,
    producer: OperationId,
    consumer: OperationId,
    lengths: &BTreeSet<OperationId>,
) -> Option<()> {
    let OperationKind::Filter {
        map, body, inputs, ..
    } = data.operations[producer].kind.clone()
    else {
        return None;
    };
    let parent = data.operations[producer].region;
    let only_count = producer == consumer;
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
        (form, inputs.len())
    };
    let parameters = body_signature(&map).0;
    let (region, args) = body::region(data, parent, &parameters);
    let mapped = invoke(data, &map, args)?;
    let predicate = invoke(data, &body, mapped.clone())?;
    let [condition] = predicate.as_slice() else {
        return None;
    };
    let [mapped] = mapped.as_slice() else {
        return None;
    };
    let values = if only_count { vec![] } else { invoke(data, &form.pre, vec![*mapped; consumer_inputs])? };
    let neutrals: Vec<_> = form.reductions.iter().flat_map(|r| &r.neutral).copied().collect();
    if values.len() != neutrals.len() {
        return None;
    }
    let mut values: Vec<_> = values
        .into_iter()
        .zip(neutrals)
        .map(|(v, n)| {
            intern_expr(
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
        let zero = intern_expr(data, count_ty, ExprKind::Int("0".into()));
        let one = intern_expr(data, count_ty, ExprKind::Int("1".into()));
        values.push(intern_expr(
            data,
            count_ty,
            ExprKind::If {
                condition: *condition,
                then_value: one,
                else_value: zero,
            },
        ));
        let (combine, args) = body::region(data, parent, &[count_ty, count_ty]);
        let t = data.types[count_ty].ty.clone();
        let function_ty = intern_type(data, function(t.clone(), function(t.clone(), t)));
        let function = intern_expr(data, function_ty, ExprKind::BinOp("+".into()));
        let sum = intern_expr(data, count_ty, ExprKind::PureApp { function, args });
        let operator = finish(data, combine, vec![count_ty, count_ty], vec![sum]);
        form.reductions.push(Reduction {
            operator,
            neutral: vec![zero],
            commutative: true,
        });
        fields.push(count_ty);
    }
    form.pre = finish(data, region, parameters, values);

    let old_ty = data.operations[consumer].ty;
    let result_ty = intern_type(
        data,
        tuple(fields.iter().map(|t| data.types[*t].ty.clone()).collect()),
    );
    let old_values: Vec<_> = data.expressions.iter().map(|(&id, e)| (id, e.kind.clone())).collect();
    let result = intern_expr(data, result_ty, ExprKind::OperationResult(consumer));
    let outputs: Vec<_> = fields
        .iter()
        .enumerate()
        .map(|(index, &t)| intern_expr(data, t, ExprKind::Project { tuple: result, index }))
        .collect();
    let previous =
        (!only_count).then(|| intern_expr(data, old_ty, ExprKind::Tuple(outputs[..count_slot].to_vec())));
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
        reuse_inputs: vec![None; fields.len()],
    };
    if !only_count {
        data.regions[parent].members.remove(&producer);
    }
    for id in lengths {
        data.regions[parent].members.remove(id);
    }
    all(data, &substitutions);
    Some(())
}
