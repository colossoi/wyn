use super::super::analysis::indexed_demand;
use super::{body, element};
use crate::egglog::data::{
    intern_expr as expr, intern_type as ty, Array, AssociatedData, ExprId, ExprKind, OperationId,
    OperationKind, RegionId, SoacBody,
};
use crate::types;
use std::collections::BTreeSet;

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
    Some(body::operation(
        data,
        region,
        t,
        OperationKind::Index { array: v, index },
    ))
}
pub(super) fn indexed(
    data: &mut AssociatedData,
    producer: OperationId,
    demands: &BTreeSet<OperationId>,
) -> Option<()> {
    let OperationKind::Screma { form, inputs, .. } = data.operations[producer].kind.clone() else {
        return None;
    };
    let parent = data.operations[producer].region;
    for &consumer in demands {
        let (slot, path) = indexed_demand(data, producer, consumer)?;
        let OperationKind::Index { index, .. } = data.operations[consumer].kind else {
            return None;
        };
        let (r, _) = body::region(data, parent, &[]);
        let args = inputs.iter().map(|a| read(data, r, a, index)).collect::<Option<Vec<_>>>()?;
        let pre = body::call(data, r, &form.pre, args)?;
        let post = body::call(data, r, &form.post, pre)?;
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
        let body = body::finish(data, r, vec![], vec![value]);
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
