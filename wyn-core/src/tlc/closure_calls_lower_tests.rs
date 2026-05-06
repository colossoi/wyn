//! Tests for verify_closure_calls_lowered.

use super::*;
use crate::SymbolTable;
use crate::ast::{Span, TypeName};
use crate::tlc::{Def, DefMeta, Program, Term, TermId, TermKind};
use polytype::Type;
use std::collections::HashMap;

fn empty_program() -> Program {
    Program {
        defs: vec![],
        uniforms: vec![],
        storage: vec![],
        symbols: SymbolTable::new(),
        def_syms: HashMap::new(),
    }
}

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn term(kind: TermKind) -> Term {
    Term {
        id: TermId(0),
        ty: unit_ty(),
        span: Span::dummy(),
        kind,
    }
}

#[test]
fn empty_program_passes() {
    assert!(verify_closure_calls_lowered(&empty_program()).is_ok());
}

#[test]
fn direct_call_passes() {
    let mut p = empty_program();
    let f = p.symbols.alloc("f".into());
    let g = p.symbols.alloc("g".into());
    let body = term(TermKind::App {
        func: Box::new(term(TermKind::Var(crate::tlc::VarRef::Symbol(g)))),
        args: vec![term(TermKind::IntLit("0".into()))],
    });
    p.defs.push(Def {
        name: f,
        ty: unit_ty(),
        body,
        meta: DefMeta::Function,
        arity: 0,
    });
    assert!(verify_closure_calls_lowered(&p).is_ok());
}

#[test]
fn arity_mismatch_fails() {
    let mut p = empty_program();
    let f = p.symbols.alloc("f".into());
    let g = p.symbols.alloc("g".into());
    // g is a defined function with arity 2.
    p.defs.push(Def {
        name: g,
        ty: unit_ty(),
        body: term(TermKind::IntLit("0".into())),
        meta: DefMeta::Function,
        arity: 2,
    });
    // f calls g with only 1 arg — wrong.
    let body = term(TermKind::App {
        func: Box::new(term(TermKind::Var(crate::tlc::VarRef::Symbol(g)))),
        args: vec![term(TermKind::IntLit("0".into()))],
    });
    p.defs.push(Def {
        name: f,
        ty: unit_ty(),
        body,
        meta: DefMeta::Function,
        arity: 0,
    });
    let err = verify_closure_calls_lowered(&p).unwrap_err();
    assert!(
        matches!(
            err,
            ClosureCallsLowerError::ArityMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ),
        "got {:?}",
        err
    );
}

#[test]
fn nested_app_in_func_position_fails() {
    let mut p = empty_program();
    let f = p.symbols.alloc("f".into());
    let g = p.symbols.alloc("g".into());
    let nested = term(TermKind::App {
        func: Box::new(term(TermKind::Var(crate::tlc::VarRef::Symbol(g)))),
        args: vec![],
    });
    let body = term(TermKind::App {
        func: Box::new(nested),
        args: vec![],
    });
    p.defs.push(Def {
        name: f,
        ty: unit_ty(),
        body,
        meta: DefMeta::Function,
        arity: 0,
    });
    let err = verify_closure_calls_lowered(&p).unwrap_err();
    assert!(matches!(err, ClosureCallsLowerError::IndirectCall { .. }));
}
