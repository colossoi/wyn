//! Tests for verify_hof_specialized.

use super::*;
use crate::ast::TypeName;
use crate::tlc::{Def, DefMeta, Program, Term, TermId, TermKind};
use crate::SymbolTable;
use polytype::Type;
use std::collections::HashMap;

fn empty_program() -> Program {
    Program {
        defs: vec![],
        symbols: SymbolTable::new(),
        def_syms: HashMap::new(),
    }
}

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn arrow(a: Type<TypeName>, b: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(TypeName::Arrow, vec![a, b])
}

fn dummy_body() -> Term {
    Term {
        id: TermId(0),
        ty: unit_ty(),
        span: crate::ast::Span::dummy(),
        kind: TermKind::IntLit("0".into()),
    }
}

#[test]
fn empty_program_passes() {
    assert!(verify_hof_specialized(&empty_program()).is_ok());
}

#[test]
fn def_with_no_arrow_params_passes() {
    let mut p = empty_program();
    let sym = p.symbols.alloc("f".into());
    p.defs.push(Def {
        name: sym,
        ty: arrow(unit_ty(), unit_ty()),
        body: dummy_body(),
        meta: DefMeta::Function,
        arity: 1,
    });
    assert!(verify_hof_specialized(&p).is_ok());
}

#[test]
fn def_with_function_typed_param_fails() {
    let mut p = empty_program();
    let sym = p.symbols.alloc("hof".into());
    let arrow_ty = arrow(unit_ty(), unit_ty());
    p.defs.push(Def {
        name: sym,
        ty: arrow(arrow_ty, unit_ty()),
        body: dummy_body(),
        meta: DefMeta::Function,
        arity: 1,
    });
    let err = verify_hof_specialized(&p).unwrap_err();
    assert!(matches!(
        err,
        HofSpecializeError::FunctionTypedParam { param_index: 0, .. }
    ));
}

#[test]
fn function_typed_return_is_ignored() {
    let mut p = empty_program();
    let sym = p.symbols.alloc("returns_fn".into());
    let arrow_ty = arrow(unit_ty(), unit_ty());
    p.defs.push(Def {
        name: sym,
        ty: arrow(unit_ty(), arrow_ty),
        body: dummy_body(),
        meta: DefMeta::Function,
        arity: 1,
    });
    assert!(verify_hof_specialized(&p).is_ok());
}
