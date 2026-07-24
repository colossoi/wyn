#![cfg(test)]

use super::{format_type_compact, SpecKey};
use crate::ast::TypeName;
use crate::tlc::{apply_type_substitution, TypeSubstitution};
use polytype::Type;

#[test]
fn test_format_type_compact() {
    let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);
    assert_eq!(format_type_compact(&f32_ty), "f32");

    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    assert_eq!(format_type_compact(&i32_ty), "i32");

    let size_ty = Type::Constructed(TypeName::Size(4), vec![]);
    assert_eq!(format_type_compact(&size_ty), "n4");
}

#[test]
fn test_spec_key_empty() {
    let key = SpecKey::empty();
    assert!(!key.needs_specialization());
}

#[test]
fn test_spec_key_with_subst() {
    let mut subst = TypeSubstitution::new();
    subst.insert(0, Type::Constructed(TypeName::Float(32), vec![]));
    subst.insert(1, Type::Constructed(TypeName::Bool, vec![]));

    let key = SpecKey::new(&subst);
    assert!(key.needs_specialization());
    assert_eq!(key.type_subst.to_subst(), subst);
}

#[test]
fn shared_type_substitution_reaches_sum_variant_payloads() {
    let mut subst = TypeSubstitution::new();
    let concrete = Type::Constructed(TypeName::UInt(32), vec![]);
    subst.insert(0, concrete.clone());
    let sum = Type::Constructed(
        TypeName::Sum(vec![("some".to_string(), vec![Type::Variable(0)])]),
        vec![],
    );

    let rewritten = apply_type_substitution(&sum, &subst);
    let Type::Constructed(TypeName::Sum(variants), _) = rewritten else {
        panic!("expected sum type");
    };
    assert_eq!(variants[0].1, vec![concrete]);
}
