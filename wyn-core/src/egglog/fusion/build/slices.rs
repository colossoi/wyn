use super::element;
use crate::egglog::data::{
    intern_expr as expr, intern_type as ty, Array, AssociatedData, ExprId, ExprKind,
};
use crate::types;

/// Push a common slice chain onto every original input of a pure map.
pub(super) fn apply(
    data: &mut AssociatedData,
    array: &Array,
    transforms: &[(ExprId, ExprId)],
) -> Option<Array> {
    if let Array::Zip(xs) = array {
        return Some(Array::Zip(
            xs.iter().map(|a| apply(data, a, transforms)).collect::<Option<_>>()?,
        ));
    }
    let mut value = match array {
        Array::Value(v) => *v,
        _ => {
            let element = element(data, array)?;
            let size = match array {
                Array::Literal(xs) => types::Type::Constructed(types::TypeName::Size(xs.len()), vec![]),
                _ => types::Type::Constructed(types::TypeName::SizePlaceholder, vec![]),
            };
            let t = ty(
                data,
                types::make_array1(
                    data.types[element].ty.clone(),
                    types::Type::Constructed(types::TypeName::ArrayVariantVirtual, vec![]),
                    size,
                    types::no_buffer(),
                ),
            );
            expr(data, t, ExprKind::Array(array.clone()))
        }
    };
    let builtin = data
        .builtins
        .iter()
        .find_map(|(&id, b)| (b.builtin == crate::builtins::catalog().known().slice).then_some(id))?;
    for &(start, end) in transforms {
        let mut t = types::canonical_storage_buffer_ty(&data.types[data.expressions[value].ty].ty);
        let types::Type::Constructed(types::TypeName::Array, fields) = &mut t else {
            return None;
        };
        let size = match (&data.expressions[start].kind, &data.expressions[end].kind) {
            (ExprKind::Int(a), ExprKind::Int(b)) => {
                a.parse::<usize>().ok().zip(b.parse::<usize>().ok()).and_then(|(a, b)| b.checked_sub(a))
            }
            _ => None,
        };
        *fields.get_mut(2)? = types::Type::Constructed(
            size.map(types::TypeName::Size).unwrap_or(types::TypeName::SizePlaceholder),
            vec![],
        );
        let result_ty = ty(data, t.clone());
        let args = vec![value, start, end];
        let function_ty = args.iter().rev().fold(t, |ret, arg| {
            types::function(data.types[data.expressions[*arg].ty].ty.clone(), ret)
        });
        let function_ty = ty(data, function_ty);
        let function = expr(data, function_ty, ExprKind::Builtin(builtin));
        value = expr(data, result_ty, ExprKind::PureApp { function, args });
    }
    Some(Array::Value(value))
}
