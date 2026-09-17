use super::element;
use crate::builtins::catalog;
use crate::egglog::data::{intern_expr, intern_type, Array, ExprId, ExprKind, Ir};
use crate::types::{canonical_storage_buffer_ty, function, make_array1, no_buffer, Type, TypeName};

/// Push a common slice chain onto every original input of a pure map.
pub(super) fn apply(data: &mut Ir, array: &Array, transforms: &[(ExprId, ExprId)]) -> Option<Array> {
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
                Array::Literal(xs) => Type::Constructed(TypeName::Size(xs.len()), vec![]),
                _ => Type::Constructed(TypeName::SizePlaceholder, vec![]),
            };
            let t = intern_type(
                data,
                make_array1(
                    data.types[element].ty.clone(),
                    Type::Constructed(TypeName::ArrayVariantVirtual, vec![]),
                    size,
                    no_buffer(),
                ),
            );
            intern_expr(data, t, ExprKind::Array(array.clone()))
        }
    };
    let builtin =
        data.builtins.iter().find_map(|(&id, b)| (b.builtin == catalog().known().slice).then_some(id))?;
    for &(start, end) in transforms {
        let mut t = canonical_storage_buffer_ty(&data.types[data.expressions[value].ty].ty);
        let Type::Constructed(TypeName::Array, fields) = &mut t else {
            return None;
        };
        let size = match (&data.expressions[start].kind, &data.expressions[end].kind) {
            (ExprKind::Int(a), ExprKind::Int(b)) => {
                a.parse::<usize>().ok().zip(b.parse::<usize>().ok()).and_then(|(a, b)| b.checked_sub(a))
            }
            _ => None,
        };
        *fields.get_mut(2)? = Type::Constructed(
            size.map(TypeName::Size).unwrap_or(TypeName::SizePlaceholder),
            vec![],
        );
        let result_ty = intern_type(data, t.clone());
        let args = vec![value, start, end];
        let function_ty = args.iter().rev().fold(t, |ret, arg| {
            function(data.types[data.expressions[*arg].ty].ty.clone(), ret)
        });
        let function_ty = intern_type(data, function_ty);
        let function = intern_expr(data, function_ty, ExprKind::Builtin(builtin));
        value = intern_expr(data, result_ty, ExprKind::PureApp { function, args });
    }
    Some(Array::Value(value))
}
