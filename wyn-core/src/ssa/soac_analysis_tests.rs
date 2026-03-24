#![cfg(test)]

use crate::ast::TypeName;
use polytype::Type;

/// Check if a type is an array type.
fn is_array_type(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::Array, _))
}

#[test]
fn test_is_array_type() {
    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![Type::Constructed(TypeName::Float(32), vec![])],
    );
    assert!(is_array_type(&arr_ty));

    let int_ty = Type::Constructed(TypeName::Int(32), vec![]);
    assert!(!is_array_type(&int_ty));
}
