//! Tests for verify_hof_specialized.

use super::*;
use crate::ast::TypeName;
use crate::tlc::{Def, DefMeta, Program, Term, TermIdSource, TermKind};
use crate::SymbolTable;
use polytype::Type;
use std::collections::HashMap;

fn empty_program() -> Program {
    Program::from_parts(vec![], SymbolTable::new(), HashMap::new(), TermIdSource::new())
}

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn arrow(a: Type<TypeName>, b: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(TypeName::Arrow, vec![a, b])
}

fn dummy_body(program: &mut Program) -> Term {
    Term {
        id: program.next_term_id(),
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
    let body = dummy_body(&mut p);
    p.defs.push(Def {
        name: sym,
        ty: arrow(unit_ty(), unit_ty()),
        body,
        meta: DefMeta::Function,
        arity: 1,
        param_diets: vec![],
        return_diet: crate::types::Diet::observing(),
    });
    assert!(verify_hof_specialized(&p).is_ok());
}

#[test]
fn def_with_function_typed_param_fails() {
    let mut p = empty_program();
    let sym = p.symbols.alloc("hof".into());
    let arrow_ty = arrow(unit_ty(), unit_ty());
    let body = dummy_body(&mut p);
    p.defs.push(Def {
        name: sym,
        ty: arrow(arrow_ty, unit_ty()),
        body,
        meta: DefMeta::Function,
        arity: 1,
        param_diets: vec![],
        return_diet: crate::types::Diet::observing(),
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
    let body = dummy_body(&mut p);
    p.defs.push(Def {
        name: sym,
        ty: arrow(unit_ty(), arrow_ty),
        body,
        meta: DefMeta::Function,
        arity: 1,
        param_diets: vec![],
        return_diet: crate::types::Diet::observing(),
    });
    assert!(verify_hof_specialized(&p).is_ok());
}
