#![cfg(test)]

use crate::ast::{Span, TypeName};
use crate::tlc::monomorphize::{Monomorphizer, SpecKey, Substitution, format_type_compact};
use crate::tlc::{Term, TermIdSource, TermKind};
use polytype::Type;

fn dummy_span() -> Span {
    Span {
        start_line: 0,
        start_col: 0,
        end_line: 0,
        end_col: 0,
    }
}

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
    let mut subst = Substitution::new();
    subst.insert(0, Type::Constructed(TypeName::Float(32), vec![]));

    let key = SpecKey::new(&subst);
    assert!(key.needs_specialization());
}

#[test]
fn test_collect_application_spine() {
    // Build: f(a, b) as App(App(Var(f), a), b)
    let mut ids = TermIdSource::new();
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    let f_var = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: dummy_span(),
        kind: TermKind::Var("f".to_string()),
    };

    let a_var = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: dummy_span(),
        kind: TermKind::Var("a".to_string()),
    };

    let b_var = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: dummy_span(),
        kind: TermKind::Var("b".to_string()),
    };

    let app1 = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: dummy_span(),
        kind: TermKind::App {
            func: Box::new(f_var),
            arg: Box::new(a_var.clone()),
        },
    };

    let (base, args) = Monomorphizer::collect_application_spine(&app1, &b_var);

    // Check base is f
    assert!(matches!(&base.kind, TermKind::Var(name) if name == "f"));

    // Check args are [a, b]
    assert_eq!(args.len(), 2);
    assert!(matches!(&args[0].kind, TermKind::Var(name) if name == "a"));
    assert!(matches!(&args[1].kind, TermKind::Var(name) if name == "b"));
}
