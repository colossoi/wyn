#![cfg(test)]

use crate::ast::{Span, TypeName};
use crate::tlc::specialize::specialize;
use crate::tlc::{Def, DefMeta, Program, Term, TermIdSource, TermKind};
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
fn test_specialize_sign_f32() {
    let mut ids = TermIdSource::new();
    let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);

    // Build: sign(x) where x: f32
    let x_var = Term {
        id: ids.next_id(),
        ty: f32_ty.clone(),
        span: dummy_span(),
        kind: TermKind::Var("x".to_string()),
    };

    let sign_var = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![f32_ty.clone(), f32_ty.clone()]),
        span: dummy_span(),
        kind: TermKind::Var("sign".to_string()),
    };

    let sign_call = Term {
        id: ids.next_id(),
        ty: f32_ty.clone(),
        span: dummy_span(),
        kind: TermKind::App {
            func: Box::new(sign_var),
            arg: Box::new(x_var),
        },
    };

    let program = Program {
        defs: vec![Def {
            name: "test".to_string(),
            ty: f32_ty.clone(),
            body: sign_call,
            meta: DefMeta::Function,
            arity: 0,
        }],
        uniforms: vec![],
        storage: vec![],
    };

    let specialized = specialize(program);

    // Check that sign became f32.sign
    match &specialized.defs[0].body.kind {
        TermKind::App { func, .. } => match &func.kind {
            TermKind::Var(name) => assert_eq!(name, "f32.sign"),
            _ => panic!("Expected Var, got {:?}", func.kind),
        },
        _ => panic!("Expected App"),
    }
}

#[test]
fn test_specialize_min_i32() {
    let mut ids = TermIdSource::new();
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    // Build: min(a, b) where a, b: i32
    // In curried form: App(App(Var("min"), a), b)
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

    let partial_ty = Type::Constructed(TypeName::Arrow, vec![i32_ty.clone(), i32_ty.clone()]);
    let func_ty = Type::Constructed(TypeName::Arrow, vec![i32_ty.clone(), partial_ty.clone()]);

    let min_var = Term {
        id: ids.next_id(),
        ty: func_ty,
        span: dummy_span(),
        kind: TermKind::Var("min".to_string()),
    };

    let min_a = Term {
        id: ids.next_id(),
        ty: partial_ty,
        span: dummy_span(),
        kind: TermKind::App {
            func: Box::new(min_var),
            arg: Box::new(a_var),
        },
    };

    let min_a_b = Term {
        id: ids.next_id(),
        ty: i32_ty.clone(),
        span: dummy_span(),
        kind: TermKind::App {
            func: Box::new(min_a),
            arg: Box::new(b_var),
        },
    };

    let program = Program {
        defs: vec![Def {
            name: "test".to_string(),
            ty: i32_ty.clone(),
            body: min_a_b,
            meta: DefMeta::Function,
            arity: 0,
        }],
        uniforms: vec![],
        storage: vec![],
    };

    let specialized = specialize(program);

    // Check that min became i32.min in the inner application
    match &specialized.defs[0].body.kind {
        TermKind::App { func, .. } => match &func.kind {
            TermKind::App { func: inner_func, .. } => match &inner_func.kind {
                TermKind::Var(name) => assert_eq!(name, "i32.min"),
                _ => panic!("Expected Var, got {:?}", inner_func.kind),
            },
            _ => panic!("Expected inner App, got {:?}", func.kind),
        },
        _ => panic!("Expected App"),
    }
}
