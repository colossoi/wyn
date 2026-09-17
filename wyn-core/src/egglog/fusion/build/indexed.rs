use super::super::analysis::indexed_demand;
use super::body::{call, finish, operation, region};
use super::element;
use crate::egglog::data::{
    intern_expr, intern_type, Array, ExprId, ExprKind, Ir, OperationId, OperationKind, RegionId, SoacBody,
};
use crate::types::{function, make_array1, no_buffer, sized_array, tuple, Type, TypeName};
use std::collections::BTreeSet;

fn read(data: &mut Ir, region: RegionId, array: &Array, index: ExprId) -> Option<ExprId> {
    if let Array::Zip(xs) = array {
        let values = xs.iter().map(|a| read(data, region, a, index)).collect::<Option<Vec<_>>>()?;
        let t = intern_type(
            data,
            tuple(values.iter().map(|v| data.types[data.expressions[*v].ty].ty.clone()).collect()),
        );
        return Some(intern_expr(data, t, ExprKind::Tuple(values)));
    }
    let t = element(data, array)?;
    let v = match array {
        Array::Value(v) => *v,
        _ => {
            let array_ty = match array {
                Array::Literal(xs) => sized_array(xs.len(), data.types[t].ty.clone()),
                _ => make_array1(
                    data.types[t].ty.clone(),
                    Type::Constructed(TypeName::ArrayVariantVirtual, vec![]),
                    Type::Constructed(TypeName::SizePlaceholder, vec![]),
                    no_buffer(),
                ),
            };
            let array_ty = intern_type(data, array_ty);
            intern_expr(data, array_ty, ExprKind::Array(array.clone()))
        }
    };
    Some(operation(
        data,
        region,
        t,
        OperationKind::Index { array: v, index },
    ))
}
pub(super) fn indexed(data: &mut Ir, producer: OperationId, demands: &BTreeSet<OperationId>) -> Option<()> {
    let OperationKind::Screma { form, inputs, .. } = data.operations[producer].kind.clone() else {
        return None;
    };
    let parent = data.operations[producer].region;
    for &consumer in demands {
        let (slot, path) = indexed_demand(data, producer, consumer)?;
        let OperationKind::Index { index, .. } = data.operations[consumer].kind else {
            return None;
        };
        let (r, _) = region(data, parent, &[]);
        let args = inputs.iter().map(|a| read(data, r, a, index)).collect::<Option<Vec<_>>>()?;
        let pre = call(data, r, &form.pre, args)?;
        let post = call(data, r, &form.post, pre)?;
        let mut value = *post.get(slot)?;
        for index in path {
            let Type::Constructed(TypeName::Tuple(_), fields) = &data.types[data.expressions[value].ty].ty
            else {
                return None;
            };
            let field = fields.get(index)?.clone();
            let t = intern_type(data, field);
            value = intern_expr(data, t, ExprKind::Project { tuple: value, index });
        }
        let body = finish(data, r, vec![], vec![value]);
        let SoacBody::Apply { captures, .. } = body else {
            return None;
        };
        let ret = data.types[data.operations[consumer].ty].ty.clone();
        let fty = captures.iter().rev().fold(ret, |r, a| {
            function(data.types[data.expressions[*a].ty].ty.clone(), r)
        });
        let t = intern_type(data, fty);
        let function = intern_expr(data, t, ExprKind::Lambda(r));
        data.operations[consumer].kind = OperationKind::Call {
            function,
            args: captures,
        };
    }
    data.regions[parent].members.remove(&producer);
    Some(())
}
