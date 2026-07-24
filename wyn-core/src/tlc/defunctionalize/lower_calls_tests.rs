//! Tests for direct-call lowering verification.

use super::*;
use crate::ast::{Span, TypeName};
use crate::tlc::{Def, DefMeta, Program, Term, TermIdSource, TermKind};
use crate::SymbolTable;
use polytype::Type;
use std::collections::HashMap;

fn empty_program() -> Program<Defunctionalized> {
    Program::from_parts(
        vec![],
        SymbolTable::new(),
        HashMap::new(),
        TermIdSource::new(),
        crate::tlc::context::PostClosureGlobal {
            auto_storage_binding_ids: crate::IdSource::new(),
        },
    )
}

fn unit_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Unit, vec![])
}

fn term(
    program: &mut Program<Defunctionalized>,
    kind: TermKind<ExplicitClosurePayload, ExplicitCapturesPayload>,
) -> Term<ExplicitClosurePayload, ExplicitCapturesPayload> {
    Term {
        id: program.next_term_id(),
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
    let func = term(&mut p, TermKind::Var(VarRef::Symbol(g)));
    let arg = term(&mut p, TermKind::IntLit("0".into()));
    let body = term(
        &mut p,
        TermKind::App {
            func: Box::new(func),
            args: vec![arg],
        },
    );
    p.defs.push(Def {
        data: (),
        name: f,
        ty: unit_ty(),
        body,
        meta: DefMeta::Function,
        arity: 0,
        param_diets: vec![],
        return_diet: crate::types::Diet::observing(),
    });
    assert!(verify_closure_calls_lowered(&p).is_ok());
}

#[test]
fn arity_mismatch_fails() {
    let mut p = empty_program();
    let f = p.symbols.alloc("f".into());
    let g = p.symbols.alloc("g".into());
    // g is a defined function with arity 2.
    let g_body = term(&mut p, TermKind::IntLit("0".into()));
    p.defs.push(Def {
        data: (),
        name: g,
        ty: unit_ty(),
        body: g_body,
        meta: DefMeta::Function,
        arity: 2,
        param_diets: vec![],
        return_diet: crate::types::Diet::observing(),
    });
    // f calls g with only 1 arg — wrong.
    let func = term(&mut p, TermKind::Var(VarRef::Symbol(g)));
    let arg = term(&mut p, TermKind::IntLit("0".into()));
    let body = term(
        &mut p,
        TermKind::App {
            func: Box::new(func),
            args: vec![arg],
        },
    );
    p.defs.push(Def {
        data: (),
        name: f,
        ty: unit_ty(),
        body,
        meta: DefMeta::Function,
        arity: 0,
        param_diets: vec![],
        return_diet: crate::types::Diet::observing(),
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
    let nested_func = term(&mut p, TermKind::Var(VarRef::Symbol(g)));
    let nested = term(
        &mut p,
        TermKind::App {
            func: Box::new(nested_func),
            args: vec![],
        },
    );
    let body = term(
        &mut p,
        TermKind::App {
            func: Box::new(nested),
            args: vec![],
        },
    );
    p.defs.push(Def {
        data: (),
        name: f,
        ty: unit_ty(),
        body,
        meta: DefMeta::Function,
        arity: 0,
        param_diets: vec![],
        return_diet: crate::types::Diet::observing(),
    });
    let err = verify_closure_calls_lowered(&p).unwrap_err();
    assert!(matches!(err, ClosureCallsLowerError::IndirectCall { .. }));
}
