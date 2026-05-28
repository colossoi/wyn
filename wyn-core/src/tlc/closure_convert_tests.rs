//! Tests for closure_convert and verify_closure_converted.

use super::*;
use crate::SymbolTable;
use crate::ast::{Span, TypeName};
use crate::tlc::{Def, DefMeta, Lambda, Program, Term, TermId, TermKind};
use polytype::Type;
use std::collections::HashMap;

fn span() -> Span {
    Span::dummy()
}

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn empty_program() -> Program {
    Program {
        defs: vec![],
        symbols: SymbolTable::new(),
        def_syms: HashMap::new(),

        view_lengths: Default::default(),
    }
}

fn term(kind: TermKind, ty: Type<TypeName>) -> Term {
    Term {
        id: TermId(0),
        ty,
        span: span(),
        kind,
    }
}

#[test]
fn empty_program_passes_verifier() {
    let program = empty_program();
    assert!(verify_closure_converted(&program).is_ok());
}

#[test]
fn def_with_simple_body_passes_verifier() {
    let mut program = empty_program();
    let sym = program.symbols.alloc("f".to_string());
    program.defs.push(Def {
        name: sym,
        ty: unit_ty(),
        body: term(TermKind::IntLit("0".into()), unit_ty()),
        meta: DefMeta::Function,
        arity: 0,
    });
    assert!(verify_closure_converted(&program).is_ok());
}

#[test]
fn unlifted_lambda_in_body_fails_verifier() {
    let mut program = empty_program();
    let sym = program.symbols.alloc("f".to_string());
    let nested_lam = term(
        TermKind::Lambda(Lambda {
            params: vec![],
            body: Box::new(term(TermKind::IntLit("0".into()), unit_ty())),
            ret_ty: unit_ty(),
        }),
        unit_ty(),
    );
    let body = term(
        TermKind::Let {
            name: program.symbols.alloc("x".to_string()),
            name_ty: unit_ty(),
            rhs: Box::new(nested_lam),
            body: Box::new(term(TermKind::IntLit("0".into()), unit_ty())),
        },
        unit_ty(),
    );
    program.defs.push(Def {
        name: sym,
        ty: unit_ty(),
        body,
        meta: DefMeta::Function,
        arity: 0,
    });
    let err = verify_closure_converted(&program).unwrap_err();
    assert!(matches!(err, ClosureConvertError::UnliftedLambda { .. }));
}

#[test]
fn append_capture_params_extends_param_list() {
    let mut symbols = SymbolTable::new();
    let mut ids = crate::tlc::TermIdSource::new();

    let x = symbols.alloc("x".into());
    let cap_a = symbols.alloc("a".into());
    let cap_b = symbols.alloc("b".into());

    let inner_body = term(TermKind::Var(VarRef::Symbol(x)), unit_ty());
    let lam_ty = Type::Constructed(TypeName::Arrow, vec![unit_ty(), unit_ty()]);
    let lam = Term {
        id: ids.next_id(),
        ty: lam_ty,
        span: span(),
        kind: TermKind::Lambda(Lambda {
            params: vec![(x, unit_ty())],
            body: Box::new(inner_body),
            ret_ty: unit_ty(),
        }),
    };

    let captures: Vec<(crate::SymbolId, Type<TypeName>)> = vec![(cap_a, unit_ty()), (cap_b, unit_ty())];

    let out = append_capture_params(lam, &captures, span(), &mut ids);

    let TermKind::Lambda(Lambda { params, body, .. }) = out.kind else {
        panic!("expected Lambda result");
    };
    assert_eq!(
        params,
        vec![(x, unit_ty()), (cap_a, unit_ty()), (cap_b, unit_ty())]
    );
    assert!(matches!(body.kind, TermKind::Var(VarRef::Symbol(s)) if s == x));
}

#[test]
fn param_spine_lambdas_are_skipped() {
    let mut program = empty_program();
    let sym = program.symbols.alloc("f".to_string());
    let p = program.symbols.alloc("p".to_string());
    let inner = term(TermKind::Var(VarRef::Symbol(p)), unit_ty());
    let body = term(
        TermKind::Lambda(Lambda {
            params: vec![(p, unit_ty())],
            body: Box::new(inner),
            ret_ty: unit_ty(),
        }),
        unit_ty(),
    );
    program.defs.push(Def {
        name: sym,
        ty: unit_ty(),
        body,
        meta: DefMeta::Function,
        arity: 1,
    });
    assert!(verify_closure_converted(&program).is_ok());
}
