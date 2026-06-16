//! Tests for closure_convert and verify_closure_converted.

use super::*;
use crate::SymbolTable;
use crate::ast::{Span, TypeName};
use crate::tlc::{Def, DefMeta, Lambda, Place, Program, Shape, SoacBody, SoacOp, Term, TermId, TermKind};
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

#[test]
fn unlifted_scatter_envelope_fails_verifier() {
    let mut program = empty_program();
    let sym = program.symbols.alloc("f".to_string());
    let dest = program.symbols.alloc("dest".to_string());
    let scatter = term(
        TermKind::Soac(SoacOp::Scatter {
            dest: Place::LocalArray {
                id: dest,
                shape: Shape(vec![]),
                elem_ty: unit_ty(),
            },
            lam: SoacBody {
                lam: Lambda {
                    params: vec![],
                    body: Box::new(term(TermKind::UnitLit, unit_ty())),
                    ret_ty: unit_ty(),
                },
                captures: vec![],
            },
            inputs: vec![],
        }),
        unit_ty(),
    );
    program.defs.push(Def {
        name: sym,
        ty: unit_ty(),
        body: scatter,
        meta: DefMeta::Function,
        arity: 0,
    });

    let err = verify_closure_converted(&program).unwrap_err();
    assert!(matches!(err, ClosureConvertError::SoacLambdaNotLifted { .. }));
}

#[test]
fn scatter_envelope_params_count_as_bound_symbols() {
    let mut symbols = SymbolTable::new();
    let mut ids = crate::tlc::TermIdSource::new();
    let dest = symbols.alloc("dest".to_string());
    let param = symbols.alloc("p".to_string());
    let scatter = Term {
        id: ids.next_id(),
        ty: unit_ty(),
        span: span(),
        kind: TermKind::Soac(SoacOp::Scatter {
            dest: Place::LocalArray {
                id: dest,
                shape: Shape(vec![]),
                elem_ty: unit_ty(),
            },
            lam: SoacBody {
                lam: Lambda {
                    params: vec![(param, unit_ty())],
                    body: Box::new(term(TermKind::Var(VarRef::Symbol(param)), unit_ty())),
                    ret_ty: unit_ty(),
                },
                captures: vec![],
            },
            inputs: vec![],
        }),
    };

    let bound = collect_bound_syms(&scatter);
    assert!(
        bound.contains(&param),
        "scatter envelope param should be treated like other SOAC lambda params"
    );
}

// --- Eta-reduction of SOAC operators -----------------------------------

/// `a -> a -> a`, the type of a binary SOAC operator over `elem`.
fn binop_ty(elem: &Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Arrow,
        vec![
            elem.clone(),
            Type::Constructed(TypeName::Arrow, vec![elem.clone(), elem.clone()]),
        ],
    )
}

/// Build `main(arr) = reduce(<op>, (), arr)` plus a top-level binary
/// function `g`, run closure conversion, and return the converted program
/// alongside `g` and `main`'s symbols. `op_args` chooses how the operator
/// envelope forwards its parameters to `g` (in order → eta-reducible).
fn reduce_with_operator(
    op_args: impl Fn(crate::SymbolId, crate::SymbolId) -> [crate::SymbolId; 2],
) -> (Program, ClosureInfo, crate::SymbolId, crate::SymbolId) {
    let mut program = empty_program();
    let elem = unit_ty();
    let op_ty = binop_ty(&elem);

    let g = program.symbols.alloc("g".to_string());
    let ga = program.symbols.alloc("ga".to_string());
    let gb = program.symbols.alloc("gb".to_string());
    program.defs.push(Def {
        name: g,
        ty: op_ty.clone(),
        body: term(
            TermKind::Lambda(Lambda {
                params: vec![(ga, elem.clone()), (gb, elem.clone())],
                body: Box::new(term(TermKind::Var(VarRef::Symbol(ga)), elem.clone())),
                ret_ty: elem.clone(),
            }),
            op_ty.clone(),
        ),
        meta: DefMeta::Function,
        arity: 2,
    });

    let p0 = program.symbols.alloc("p0".to_string());
    let p1 = program.symbols.alloc("p1".to_string());
    let [a0, a1] = op_args(p0, p1);
    let op = crate::tlc::SoacBody {
        lam: Lambda {
            params: vec![(p0, elem.clone()), (p1, elem.clone())],
            body: Box::new(term(
                TermKind::App {
                    func: Box::new(term(TermKind::Var(VarRef::Symbol(g)), op_ty.clone())),
                    args: vec![
                        term(TermKind::Var(VarRef::Symbol(a0)), elem.clone()),
                        term(TermKind::Var(VarRef::Symbol(a1)), elem.clone()),
                    ],
                },
                elem.clone(),
            )),
            ret_ty: elem.clone(),
        },
        captures: vec![],
    };

    let arr = program.symbols.alloc("arr".to_string());
    let reduce = term(
        TermKind::Soac(SoacOp::Reduce {
            op,
            ne: Box::new(term(TermKind::UnitLit, elem.clone())),
            input: crate::tlc::ArrayExpr::Ref(Box::new(term(
                TermKind::Var(VarRef::Symbol(arr)),
                elem.clone(),
            ))),
        }),
        elem.clone(),
    );
    let main = program.symbols.alloc("main".to_string());
    let main_ty = Type::Constructed(TypeName::Arrow, vec![elem.clone(), elem.clone()]);
    program.defs.push(Def {
        name: main,
        ty: main_ty.clone(),
        body: term(
            TermKind::Lambda(Lambda {
                params: vec![(arr, elem.clone())],
                body: Box::new(reduce),
                ret_ty: elem.clone(),
            }),
            main_ty,
        ),
        meta: DefMeta::Function,
        arity: 1,
    });

    let (result, info) = run(program, &std::collections::HashSet::new());
    (result, info, g, main)
}

/// Peel `main`'s parameter spine and return its tail `reduce` operator body.
fn reduce_operator_body(program: &Program, main: crate::SymbolId) -> Term {
    let def = program.defs.iter().find(|d| d.name == main).expect("main def");
    let mut body = &def.body;
    while let TermKind::Lambda(Lambda { body: inner, .. }) = &body.kind {
        body = inner;
    }
    let TermKind::Soac(SoacOp::Reduce { op, .. }) = &body.kind else {
        panic!("expected reduce soac, got {:?}", body.kind);
    };
    (*op.lam.body).clone()
}

#[test]
fn eta_wrapper_operator_references_function_directly() {
    // `λ(p0,p1). g(p0,p1)` is a pure forwarder — the operator is just `g`.
    let (result, _info, g, main) = reduce_with_operator(|p0, p1| [p0, p1]);

    assert_eq!(
        result.defs.iter().filter(|d| matches!(d.meta, DefMeta::LiftedLambda)).count(),
        0,
        "an eta-wrapper operator must not lift a forwarder def"
    );
    let op_body = reduce_operator_body(&result, main);
    assert!(
        matches!(op_body.kind, TermKind::Var(VarRef::Symbol(s)) if s == g),
        "reduce operator should reference g directly, got {:?}",
        op_body.kind
    );
}

#[test]
fn non_forwarding_operator_is_lifted() {
    // `λ(p0,p1). g(p1,p0)` swaps its arguments — not eta-reducible, so the
    // envelope must still be lifted to its own def.
    let (result, _info, g, main) = reduce_with_operator(|p0, p1| [p1, p0]);

    assert_eq!(
        result.defs.iter().filter(|d| matches!(d.meta, DefMeta::LiftedLambda)).count(),
        1,
        "a non-forwarding operator must be lifted"
    );
    let op_body = reduce_operator_body(&result, main);
    let TermKind::Var(VarRef::Symbol(s)) = op_body.kind else {
        panic!("expected operator to be a Var, got {:?}", op_body.kind);
    };
    assert_ne!(s, g, "swapped-arg operator must not collapse to g");
}
