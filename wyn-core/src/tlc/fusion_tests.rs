use super::super::{DefMeta, SoacOp};
use crate::ast::Span;
use crate::tlc::fusion::*;
use crate::tlc::{SoacBody, SoacDestination};
use std::collections::HashMap;

fn dummy_span() -> Span {
    Span::new(0, 0, 0, 0)
}

fn mk_term(kind: TermKind, ty: Type<TypeName>, term_ids: &mut TermIdSource) -> Term {
    Term {
        id: term_ids.next_id(),
        ty,
        span: dummy_span(),
        kind,
    }
}

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn f32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn array_ty(elem: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            elem,
            Type::Variable(0), // size
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
        ],
    )
}

/// Build a simple map: `map(lam, [input])`
fn mk_map(lam: Lambda, input: Term, result_ty: Type<TypeName>, term_ids: &mut TermIdSource) -> Term {
    mk_term(
        TermKind::Soac(SoacOp::Map {
            lam: mk_soac_body(lam),
            inputs: vec![ArrayExpr::Ref(Box::new(input))],
            destination: SoacDestination::Fresh,
        }),
        result_ty,
        term_ids,
    )
}

/// Build a lambda with one parameter
fn mk_lambda1(param: SymbolId, param_ty: Type<TypeName>, body: Term, ret_ty: Type<TypeName>) -> Lambda {
    Lambda {
        params: vec![(param, param_ty)],
        body: Box::new(body),
        ret_ty,
    }
}

/// Build a lambda with two parameters
fn mk_lambda2(
    p1: SymbolId,
    p1_ty: Type<TypeName>,
    p2: SymbolId,
    p2_ty: Type<TypeName>,
    body: Term,
    ret_ty: Type<TypeName>,
) -> Lambda {
    Lambda {
        params: vec![(p1, p1_ty), (p2, p2_ty)],
        body: Box::new(body),
        ret_ty,
    }
}

/// Wrap a Lambda in a SoacBody with empty captures (test helper).
fn mk_soac_body(lam: Lambda) -> SoacBody {
    SoacBody {
        lam,
        captures: vec![],
    }
}

/// Build a reduce: `reduce(op, ne, input)`
fn mk_reduce(
    op: Lambda,
    ne: Term,
    input: Term,
    result_ty: Type<TypeName>,
    term_ids: &mut TermIdSource,
) -> Term {
    mk_term(
        TermKind::Soac(SoacOp::Reduce {
            op: mk_soac_body(op),
            ne: Box::new(ne),
            input: ArrayExpr::Ref(Box::new(input)),
        }),
        result_ty,
        term_ids,
    )
}

fn mk_scan(
    op: Lambda,
    ne: Term,
    input: Term,
    result_ty: Type<TypeName>,
    term_ids: &mut TermIdSource,
) -> Term {
    mk_term(
        TermKind::Soac(SoacOp::Scan {
            op: mk_soac_body(op.clone()),
            // A freshly-built scan is non-fused: pure combiner == element step.
            reduce_op: mk_soac_body(op),
            ne: Box::new(ne),
            input: ArrayExpr::Ref(Box::new(input)),
            destination: SoacDestination::Fresh,
        }),
        result_ty,
        term_ids,
    )
}

/// Build a function def wrapping a body in a lambda
fn mk_func_def(
    name: SymbolId,
    params: Vec<(SymbolId, Type<TypeName>)>,
    body: Term,
    ret_ty: Type<TypeName>,
) -> Def {
    let lam_body = if params.is_empty() {
        body
    } else {
        Term {
            id: body.id,
            ty: ret_ty.clone(),
            span: body.span,
            kind: TermKind::Lambda(Lambda {
                params: params.clone(),
                body: Box::new(body),
                ret_ty: ret_ty.clone(),
            }),
        }
    };
    Def {
        name,
        ty: ret_ty,
        body: lam_body,
        meta: DefMeta::Function,
        arity: params.len(),
    }
}

/// Build a function call: `func(arg)`
fn mk_app(func: Term, arg: Term, result_ty: Type<TypeName>, term_ids: &mut TermIdSource) -> Term {
    mk_term(
        TermKind::App {
            func: Box::new(func),
            args: vec![arg],
        },
        result_ty,
        term_ids,
    )
}

// -------------------------------------------------------------------------
// Test: simple map(g, map(f, a)) → map(g∘f, a)
// -------------------------------------------------------------------------
#[test]
fn test_simple_map_fusion() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    // Symbols
    let a_sym = symbols.alloc("a".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let x_sym = symbols.alloc("x".to_string()); // f's param
    let y_sym = symbols.alloc("y".to_string()); // g's param

    // f: i32 → i32 (identity-like, just returns x)
    let f_body = mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids);
    let f = mk_lambda1(x_sym, i32_ty(), f_body, i32_ty());

    // g: i32 → i32 (identity-like, just returns y)
    let g_body = mk_term(TermKind::Var(VarRef::Symbol(y_sym)), i32_ty(), &mut term_ids);
    let g = mk_lambda1(y_sym, i32_ty(), g_body, i32_ty());

    // a: [i32]
    let a = mk_term(
        TermKind::Var(VarRef::Symbol(a_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );

    // Producer: map(f, a)
    let producer = mk_map(f, a, array_ty(i32_ty()), &mut term_ids);

    // Consumer: map(g, b)
    let b_ref = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let consumer = mk_map(g, b_ref, array_ty(i32_ty()), &mut term_ids);

    // let b = map(f, a) in map(g, b)
    let program_body = mk_term(
        TermKind::Let {
            name: b_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(consumer),
        },
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: array_ty(i32_ty()),
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let fused = run(program);

    // The result should be a single Map (no Let binding)
    match &fused.defs[0].body.kind {
        TermKind::Soac(SoacOp::Map { lam, inputs, .. }) => {
            // Input should be 'a' (the original array)
            assert_eq!(inputs.len(), 1);
            match &inputs[0] {
                ArrayExpr::Ref(t) => match &t.kind {
                    TermKind::Var(VarRef::Symbol(s)) => assert_eq!(*s, a_sym),
                    other => panic!("Expected Var(a), got {:?}", other),
                },
                other => panic!("Expected Ref, got {:?}", other),
            }

            // Lambda should have f's param (x)
            assert_eq!(lam.lam.params.len(), 1);
            assert_eq!(lam.lam.params[0].0, x_sym);

            // Body should be: let _fused = x in _fused
            // (g's body is y, substituted to _fused; f's body is x)
            match &lam.lam.body.kind {
                TermKind::Let { rhs, body, .. } => {
                    // rhs is f's body (Var(x))
                    assert!(matches!(&rhs.kind, TermKind::Var(VarRef::Symbol(s)) if *s == x_sym));
                    // body should be Var(_fused) — the fresh symbol
                    assert!(matches!(&body.kind, TermKind::Var(VarRef::Symbol(_))));
                }
                other => panic!("Expected Let (composed body), got {:?}", other),
            }
        }
        other => panic!("Expected fused Soac(Map), got {:?}", other),
    }
}

// -------------------------------------------------------------------------
// Test: chain of three maps fused
// -------------------------------------------------------------------------
#[test]
fn test_chain_of_three_maps() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let a_sym = symbols.alloc("a".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let c_sym = symbols.alloc("c".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());
    let z_sym = symbols.alloc("z".to_string());

    // f, g, h: i32 → i32
    let f = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let g = mk_lambda1(
        y_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(y_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let h = mk_lambda1(
        z_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(z_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );

    let a = mk_term(
        TermKind::Var(VarRef::Symbol(a_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let producer = mk_map(f, a, array_ty(i32_ty()), &mut term_ids);

    let b_ref = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let middle = mk_map(g, b_ref, array_ty(i32_ty()), &mut term_ids);

    let c_ref = mk_term(
        TermKind::Var(VarRef::Symbol(c_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let consumer = mk_map(h, c_ref, array_ty(i32_ty()), &mut term_ids);

    // let b = map(f, a) in let c = map(g, b) in map(h, c)
    let inner_let = mk_term(
        TermKind::Let {
            name: c_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(middle),
            body: Box::new(consumer),
        },
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let outer_let = mk_term(
        TermKind::Let {
            name: b_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(inner_let),
        },
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: array_ty(i32_ty()),
            body: outer_let,
            meta: DefMeta::Function,
            arity: 0,
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let fused = run(program);

    // Should be a single Map with a's input (all three fused)
    match &fused.defs[0].body.kind {
        TermKind::Soac(SoacOp::Map { inputs, lam, .. }) => {
            assert_eq!(inputs.len(), 1);
            match &inputs[0] {
                ArrayExpr::Ref(t) => {
                    assert!(matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if *s == a_sym))
                }
                other => panic!("Expected Ref(a), got {:?}", other),
            }
            // Lambda param should be f's original param (x)
            assert_eq!(lam.lam.params[0].0, x_sym);
        }
        other => panic!("Expected fully fused Map, got {:?}", other),
    }
}

// -------------------------------------------------------------------------
// Test: multi-use intermediate (no fusion)
// -------------------------------------------------------------------------
#[test]
fn test_multi_use_no_fusion() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let a_sym = symbols.alloc("a".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());

    let f = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let g = mk_lambda1(
        y_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(y_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );

    let a = mk_term(
        TermKind::Var(VarRef::Symbol(a_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let producer = mk_map(f, a, array_ty(i32_ty()), &mut term_ids);

    // Consumer uses b
    let b_ref1 = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let consumer = mk_map(g, b_ref1, array_ty(i32_ty()), &mut term_ids);

    // Second use of b (in an App, making count_uses == 2)
    let b_ref2 = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );

    // Body: App(consumer, b) — artificial but creates two uses
    let body = mk_term(
        TermKind::App {
            func: Box::new(consumer),
            args: vec![b_ref2],
        },
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let program_body = mk_term(
        TermKind::Let {
            name: b_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(body),
        },
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: array_ty(i32_ty()),
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let result = run(program);

    // Should still be a Let (no fusion because b is used twice)
    assert!(matches!(&result.defs[0].body.kind, TermKind::Let { .. }));
}

// -------------------------------------------------------------------------
// Test: zip-fused producer: map(g, map(f, zip(a,b)))
// -------------------------------------------------------------------------
#[test]
fn test_zip_fused_producer() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let a_sym = symbols.alloc("a".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let intermediate_sym = symbols.alloc("inter".to_string());
    let x1_sym = symbols.alloc("x1".to_string());
    let x2_sym = symbols.alloc("x2".to_string());
    let y_sym = symbols.alloc("y".to_string());

    let _tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), f32_ty()]);

    // f: (i32, f32) → i32 — takes two params (zip-fused)
    let f = Lambda {
        params: vec![(x1_sym, i32_ty()), (x2_sym, f32_ty())],
        body: Box::new(mk_term(
            TermKind::Var(VarRef::Symbol(x1_sym)),
            i32_ty(),
            &mut term_ids,
        )),
        ret_ty: i32_ty(),
    };

    // Producer: map(f, [a, b]) with zip-fused inputs
    let a = mk_term(
        TermKind::Var(VarRef::Symbol(a_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let b = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(f32_ty()),
        &mut term_ids,
    );
    let producer = mk_term(
        TermKind::Soac(SoacOp::Map {
            lam: mk_soac_body(f),
            inputs: vec![ArrayExpr::Ref(Box::new(a)), ArrayExpr::Ref(Box::new(b))],
            destination: SoacDestination::Fresh,
        }),
        array_ty(i32_ty()),
        &mut term_ids,
    );

    // g: i32 → i32
    let g = mk_lambda1(
        y_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(y_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );

    // Consumer: map(g, inter) with single input
    let inter_ref = mk_term(
        TermKind::Var(VarRef::Symbol(intermediate_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let consumer = mk_map(g, inter_ref, array_ty(i32_ty()), &mut term_ids);

    // let inter = map(f, [a, b]) in map(g, inter)
    let program_body = mk_term(
        TermKind::Let {
            name: intermediate_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(consumer),
        },
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: array_ty(i32_ty()),
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let fused = run(program);

    // Should be a Map with [a, b] inputs (producer's multi-inputs preserved)
    match &fused.defs[0].body.kind {
        TermKind::Soac(SoacOp::Map { lam, inputs, .. }) => {
            assert_eq!(inputs.len(), 2);
            // Lambda should have f's params (x1, x2)
            assert_eq!(lam.lam.params.len(), 2);
            assert_eq!(lam.lam.params[0].0, x1_sym);
            assert_eq!(lam.lam.params[1].0, x2_sym);
        }
        other => panic!("Expected fused Map with 2 inputs, got {:?}", other),
    }
}

// -------------------------------------------------------------------------
// Test: consumer with multiple inputs (no fusion)
// -------------------------------------------------------------------------
#[test]
fn test_consumer_multi_input_no_fusion() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let a_sym = symbols.alloc("a".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let other_sym = symbols.alloc("other".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y1_sym = symbols.alloc("y1".to_string());
    let y2_sym = symbols.alloc("y2".to_string());

    let f = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );

    let a = mk_term(
        TermKind::Var(VarRef::Symbol(a_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let producer = mk_map(f, a, array_ty(i32_ty()), &mut term_ids);

    // g: (i32, i32) → i32, takes two inputs
    let g = Lambda {
        params: vec![(y1_sym, i32_ty()), (y2_sym, i32_ty())],
        body: Box::new(mk_term(
            TermKind::Var(VarRef::Symbol(y1_sym)),
            i32_ty(),
            &mut term_ids,
        )),
        ret_ty: i32_ty(),
    };

    // Consumer: map(g, [b, other]) — b plus another array
    let b_ref = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let other = mk_term(
        TermKind::Var(VarRef::Symbol(other_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let consumer = mk_term(
        TermKind::Soac(SoacOp::Map {
            lam: mk_soac_body(g),
            inputs: vec![ArrayExpr::Ref(Box::new(b_ref)), ArrayExpr::Ref(Box::new(other))],
            destination: SoacDestination::Fresh,
        }),
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let program_body = mk_term(
        TermKind::Let {
            name: b_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(consumer),
        },
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: array_ty(i32_ty()),
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let result = run(program);

    // Should NOT fuse — consumer has multiple inputs
    assert!(matches!(&result.defs[0].body.kind, TermKind::Let { .. }));
}

// -------------------------------------------------------------------------
// Test: inline map(f, map(g, a)) — no Let binding
// -------------------------------------------------------------------------
#[test]
fn test_inline_map_fusion() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let a_sym = symbols.alloc("a".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());

    // Inner: map(g, a) where g(x) = x
    let g = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let a = mk_term(
        TermKind::Var(VarRef::Symbol(a_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let inner_map = mk_map(g, a, array_ty(i32_ty()), &mut term_ids);

    // Outer: map(f, inner_map) where f(y) = y — inner_map is inline (Ref)
    let f = mk_lambda1(
        y_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(y_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let outer = mk_term(
        TermKind::Soac(SoacOp::Map {
            lam: mk_soac_body(f),
            inputs: vec![ArrayExpr::Ref(Box::new(inner_map))],
            destination: SoacDestination::Fresh,
        }),
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: array_ty(i32_ty()),
            body: outer,
            meta: DefMeta::Function,
            arity: 0,
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let fused = run(program);

    // Should be a single Map with a's input, param x (inner g's param)
    match &fused.defs[0].body.kind {
        TermKind::Soac(SoacOp::Map { lam, inputs, .. }) => {
            assert_eq!(inputs.len(), 1);
            match &inputs[0] {
                ArrayExpr::Ref(t) => {
                    assert!(matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if *s == a_sym))
                }
                other => panic!("Expected Ref(a), got {:?}", other),
            }
            // Lambda should have g's param (x), not f's param (y)
            assert_eq!(lam.lam.params.len(), 1);
            assert_eq!(lam.lam.params[0].0, x_sym);
        }
        other => panic!("Expected fused Map, got {:?}", other),
    }
}

// -------------------------------------------------------------------------
// Test: scan(op, ne, map(g, a)) fuses g into the per-element step but keeps
// the original pure combiner as `reduce_op` (needed by the parallel scan's
// phase 2, which merges already-transformed block sums). Regression guard
// for the fused-scan `OpFunctionCall` type mismatch.
// -------------------------------------------------------------------------
#[test]
fn test_map_into_scan_keeps_pure_reduce_op() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let a_sym = symbols.alloc("a".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let acc_sym = symbols.alloc("acc".to_string());
    let b_sym = symbols.alloc("b".to_string());

    // Inner: map(g, a) where g(x) = x
    let g = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let a = mk_term(
        TermKind::Var(VarRef::Symbol(a_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let inner_map = mk_map(g, a, array_ty(i32_ty()), &mut term_ids);

    // Outer: scan(plus, 0, inner_map) where plus(acc, b) = acc
    let plus = mk_lambda2(
        acc_sym,
        i32_ty(),
        b_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(acc_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let ne = mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids);
    let outer = mk_scan(plus, ne, inner_map, array_ty(i32_ty()), &mut term_ids);

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: array_ty(i32_ty()),
            body: outer,
            meta: DefMeta::Function,
            arity: 0,
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let fused = run(program);

    match &fused.defs[0].body.kind {
        TermKind::Soac(SoacOp::Scan {
            op, reduce_op, input, ..
        }) => {
            // The map fused away — the scan now reads `a` directly.
            match input {
                ArrayExpr::Ref(t) => {
                    assert!(matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if *s == a_sym))
                }
                other => panic!("Expected Ref(a), got {:?}", other),
            }
            // The pure combiner (phase 2) is preserved verbatim — still
            // `plus`'s params — rather than being lost to the fused step.
            assert_eq!(reduce_op.lam.params.len(), 2);
            assert_eq!(reduce_op.lam.params[0].0, acc_sym);
            assert_eq!(reduce_op.lam.params[1].0, b_sym);
            // The element step exists independently (it folds `g` in).
            assert_eq!(op.lam.params.len(), 2);
        }
        other => panic!("Expected fused Scan, got {:?}", other),
    }
}

// -------------------------------------------------------------------------
// Test: inline chain map(f, map(g, map(h, a))) — no Let bindings
// -------------------------------------------------------------------------
#[test]
fn test_inline_chain_of_three() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let a_sym = symbols.alloc("a".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());
    let z_sym = symbols.alloc("z".to_string());

    // h(x) = x
    let h = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let a = mk_term(
        TermKind::Var(VarRef::Symbol(a_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let inner = mk_map(h, a, array_ty(i32_ty()), &mut term_ids);

    // g(y) = y
    let g = mk_lambda1(
        y_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(y_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let middle = mk_term(
        TermKind::Soac(SoacOp::Map {
            lam: mk_soac_body(g),
            inputs: vec![ArrayExpr::Ref(Box::new(inner))],
            destination: SoacDestination::Fresh,
        }),
        array_ty(i32_ty()),
        &mut term_ids,
    );

    // f(z) = z
    let f = mk_lambda1(
        z_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(z_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let outer = mk_term(
        TermKind::Soac(SoacOp::Map {
            lam: mk_soac_body(f),
            inputs: vec![ArrayExpr::Ref(Box::new(middle))],
            destination: SoacDestination::Fresh,
        }),
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: array_ty(i32_ty()),
            body: outer,
            meta: DefMeta::Function,
            arity: 0,
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let fused = run(program);

    // All three fused into one Map over a
    match &fused.defs[0].body.kind {
        TermKind::Soac(SoacOp::Map { lam, inputs, .. }) => {
            assert_eq!(inputs.len(), 1);
            match &inputs[0] {
                ArrayExpr::Ref(t) => {
                    assert!(matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if *s == a_sym))
                }
                other => panic!("Expected Ref(a), got {:?}", other),
            }
            // Innermost param (h's param x) should be the lambda param
            assert_eq!(lam.lam.params.len(), 1);
            assert_eq!(lam.lam.params[0].0, x_sym);
        }
        other => panic!("Expected fully fused Map, got {:?}", other),
    }
}

// -------------------------------------------------------------------------
// Test: zip-fused consumer — map(f, zip(map(g, a), b))
// One input is an inline nested map, the other is not
// -------------------------------------------------------------------------
#[test]
fn test_zip_fused_consumer_inline() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let a_sym = symbols.alloc("a".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y1_sym = symbols.alloc("y1".to_string());
    let y2_sym = symbols.alloc("y2".to_string());

    // Inner: map(g, a) where g(x) = x
    let g = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let a = mk_term(
        TermKind::Var(VarRef::Symbol(a_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let inner_map = mk_map(g, a, array_ty(i32_ty()), &mut term_ids);

    // Outer: map(f, zip(map(g, a), b)) — zip already absorbed into multi-input
    // f(y1, y2) = y1  (takes two params from zip)
    let f = Lambda {
        params: vec![(y1_sym, i32_ty()), (y2_sym, f32_ty())],
        body: Box::new(mk_term(
            TermKind::Var(VarRef::Symbol(y1_sym)),
            i32_ty(),
            &mut term_ids,
        )),
        ret_ty: i32_ty(),
    };
    let b = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(f32_ty()),
        &mut term_ids,
    );
    let outer = mk_term(
        TermKind::Soac(SoacOp::Map {
            lam: mk_soac_body(f),
            inputs: vec![
                ArrayExpr::Ref(Box::new(inner_map)), // will be fused
                ArrayExpr::Ref(Box::new(b)),         // stays as-is
            ],
            destination: SoacDestination::Fresh,
        }),
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: array_ty(i32_ty()),
            body: outer,
            meta: DefMeta::Function,
            arity: 0,
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let fused = run(program);

    // The inner map should be fused: y1's slot replaced by g's param x,
    // input[0] is now Ref(a), input[1] is Ref(b)
    match &fused.defs[0].body.kind {
        TermKind::Soac(SoacOp::Map { lam, inputs, .. }) => {
            assert_eq!(inputs.len(), 2);
            // First input: a (was map(g, a), now fused)
            match &inputs[0] {
                ArrayExpr::Ref(t) => {
                    assert!(matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if *s == a_sym))
                }
                other => panic!("Expected Ref(a), got {:?}", other),
            }
            // Second input: b (unchanged)
            match &inputs[1] {
                ArrayExpr::Ref(t) => {
                    assert!(matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if *s == b_sym))
                }
                other => panic!("Expected Ref(b), got {:?}", other),
            }
            // Params: x (from g), y2 (from f — kept)
            assert_eq!(lam.lam.params.len(), 2);
            assert_eq!(lam.lam.params[0].0, x_sym);
            assert_eq!(lam.lam.params[1].0, y2_sym);
        }
        other => panic!("Expected fused Map with 2 inputs, got {:?}", other),
    }
}

// -------------------------------------------------------------------------
// Test: map-zip-map — map(f, zip(map(g, a), map(h, b)))
// Both zip inputs are inline maps, both should be fused
// -------------------------------------------------------------------------
#[test]
fn test_map_zip_map() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let a_sym = symbols.alloc("a".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let x_sym = symbols.alloc("x".to_string()); // g's param
    let w_sym = symbols.alloc("w".to_string()); // h's param
    let y1_sym = symbols.alloc("y1".to_string()); // f's param 1
    let y2_sym = symbols.alloc("y2".to_string()); // f's param 2

    // g(x) = x : i32 → i32
    let g = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let a = mk_term(
        TermKind::Var(VarRef::Symbol(a_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let map_g_a = mk_map(g, a, array_ty(i32_ty()), &mut term_ids);

    // h(w) = w : f32 → f32
    let h = mk_lambda1(
        w_sym,
        f32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(w_sym)), f32_ty(), &mut term_ids),
        f32_ty(),
    );
    let b = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(f32_ty()),
        &mut term_ids,
    );
    let map_h_b = mk_map(h, b, array_ty(f32_ty()), &mut term_ids);

    // f(y1: i32, y2: f32) = y1
    let f = Lambda {
        params: vec![(y1_sym, i32_ty()), (y2_sym, f32_ty())],
        body: Box::new(mk_term(
            TermKind::Var(VarRef::Symbol(y1_sym)),
            i32_ty(),
            &mut term_ids,
        )),
        ret_ty: i32_ty(),
    };

    // map(f, zip(map(g, a), map(h, b))) — zip absorbed
    let outer = mk_term(
        TermKind::Soac(SoacOp::Map {
            lam: mk_soac_body(f),
            inputs: vec![
                ArrayExpr::Ref(Box::new(map_g_a)),
                ArrayExpr::Ref(Box::new(map_h_b)),
            ],
            destination: SoacDestination::Fresh,
        }),
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: array_ty(i32_ty()),
            body: outer,
            meta: DefMeta::Function,
            arity: 0,
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let fused = run(program);

    match &fused.defs[0].body.kind {
        TermKind::Soac(SoacOp::Map { lam, inputs, .. }) => {
            // Both intermediates eliminated: inputs are [a, b]
            assert_eq!(inputs.len(), 2);
            match &inputs[0] {
                ArrayExpr::Ref(t) => {
                    assert!(matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if *s == a_sym))
                }
                other => panic!("Expected Ref(a), got {:?}", other),
            }
            match &inputs[1] {
                ArrayExpr::Ref(t) => {
                    assert!(matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if *s == b_sym))
                }
                other => panic!("Expected Ref(b), got {:?}", other),
            }
            // Params: x (from g), w (from h)
            assert_eq!(lam.lam.params.len(), 2);
            assert_eq!(lam.lam.params[0].0, x_sym);
            assert_eq!(lam.lam.params[1].0, w_sym);
        }
        other => panic!("Expected fused Map, got {:?}", other),
    }
}

// =========================================================================
// Progressive raytrace-like tests
// =========================================================================

// Test 1: Intraprocedural map-reduce fusion
// let tmp = map(f, xs) in reduce(op, ne, tmp)  →  reduce(op∘f, ne, xs)
#[test]
fn test_raytrace_step1_local_map_reduce() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let tmp_sym = symbols.alloc("tmp".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let acc_sym = symbols.alloc("acc".to_string());
    let elem_sym = symbols.alloc("elem".to_string());

    let f = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let op = mk_lambda2(
        acc_sym,
        i32_ty(),
        elem_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(acc_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );

    let xs = mk_term(
        TermKind::Var(VarRef::Symbol(xs_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let ne = mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids);
    let producer = mk_map(f, xs, array_ty(i32_ty()), &mut term_ids);
    let tmp_ref = mk_term(
        TermKind::Var(VarRef::Symbol(tmp_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let consumer = mk_reduce(op, ne, tmp_ref, i32_ty(), &mut term_ids);

    let body = mk_term(
        TermKind::Let {
            name: tmp_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(consumer),
        },
        i32_ty(),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: i32_ty(),
            body,
            meta: DefMeta::Function,
            arity: 0,
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let fused = run(program);
    match &fused.defs[0].body.kind {
        TermKind::Soac(SoacOp::Redomap { op, inputs, .. }) => {
            assert_eq!(inputs.len(), 1);
            match &inputs[0] {
                ArrayExpr::Ref(t) => {
                    assert!(matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if *s == xs_sym))
                }
                other => panic!("Expected Ref(xs), got {:?}", other),
            }
            // Redomap op has (acc, x) params — acc from reduce, x from map
            assert_eq!(op.lam.params.len(), 2);
            assert_eq!(op.lam.params[0].0, acc_sym);
        }
        other => panic!("Expected fused Redomap, got {:?}", other),
    }
}

// Test 2: Reduce in a separate def, consumer recognized via summary
// def myReduce(xs) = reduce(op, ne, xs)
// main: let tmp = map(f, arr) in myReduce(tmp)
#[test]
fn test_raytrace_step2_interprocedural_reduce_consumer() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let arr_sym = symbols.alloc("arr".to_string());
    let tmp_sym = symbols.alloc("tmp".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let acc_sym = symbols.alloc("acc".to_string());
    let elem_sym = symbols.alloc("elem".to_string());
    let my_reduce_sym = symbols.alloc("myReduce".to_string());
    let main_sym = symbols.alloc("main".to_string());

    // def myReduce(xs) = reduce(op, ne, xs)
    let op = mk_lambda2(
        acc_sym,
        i32_ty(),
        elem_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(acc_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let ne = mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids);
    let reduce_body = mk_reduce(
        op,
        ne,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        i32_ty(),
        &mut term_ids,
    );
    let my_reduce_def = mk_func_def(
        my_reduce_sym,
        vec![(xs_sym, array_ty(i32_ty()))],
        reduce_body,
        i32_ty(),
    );

    // main: let tmp = map(f, arr) in myReduce(tmp)
    let f = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let arr = mk_term(
        TermKind::Var(VarRef::Symbol(arr_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let producer = mk_map(f, arr, array_ty(i32_ty()), &mut term_ids);
    let consumer_call = mk_app(
        mk_term(
            TermKind::Var(VarRef::Symbol(my_reduce_sym)),
            i32_ty(),
            &mut term_ids,
        ),
        mk_term(
            TermKind::Var(VarRef::Symbol(tmp_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        i32_ty(),
        &mut term_ids,
    );

    let main_body = mk_term(
        TermKind::Let {
            name: tmp_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(consumer_call),
        },
        i32_ty(),
        &mut term_ids,
    );
    let main_def = Def {
        name: main_sym,
        ty: i32_ty(),
        body: main_body,
        meta: DefMeta::Function,
        arity: 0,
    };

    let program = Program {
        defs: vec![my_reduce_def, main_def],
        symbols,
        def_syms: HashMap::new(),
    };

    let fused = run(program);
    let main = fused.defs.iter().find(|d| d.name == main_sym).unwrap();
    match &main.body.kind {
        TermKind::Soac(SoacOp::Redomap { op, inputs, .. }) => {
            assert_eq!(inputs.len(), 1);
            match &inputs[0] {
                ArrayExpr::Ref(t) => {
                    assert!(matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if *s == arr_sym))
                }
                other => panic!("Expected Ref(arr), got {:?}", other),
            }
            assert_eq!(op.lam.params.len(), 2);
        }
        other => panic!("Expected fused Redomap in main, got {:?}", other),
    }
}

// Test 3: Map in a separate def, producer recognized via summary
// def myMap(xs) = map(f, xs)
// main: let tmp = myMap(arr) in reduce(op, ne, tmp)
#[test]
fn test_raytrace_step3_interprocedural_map_producer() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let arr_sym = symbols.alloc("arr".to_string());
    let tmp_sym = symbols.alloc("tmp".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let acc_sym = symbols.alloc("acc".to_string());
    let elem_sym = symbols.alloc("elem".to_string());
    let my_map_sym = symbols.alloc("myMap".to_string());
    let main_sym = symbols.alloc("main".to_string());

    // def myMap(xs) = map(f, xs)
    let f = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let map_body = mk_map(
        f,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let my_map_def = mk_func_def(
        my_map_sym,
        vec![(xs_sym, array_ty(i32_ty()))],
        map_body,
        array_ty(i32_ty()),
    );

    // main: let tmp = myMap(arr) in reduce(op, ne, tmp)
    let producer_call = mk_app(
        mk_term(
            TermKind::Var(VarRef::Symbol(my_map_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        mk_term(
            TermKind::Var(VarRef::Symbol(arr_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let op = mk_lambda2(
        acc_sym,
        i32_ty(),
        elem_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(acc_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let ne = mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids);
    let consumer = mk_reduce(
        op,
        ne,
        mk_term(
            TermKind::Var(VarRef::Symbol(tmp_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        i32_ty(),
        &mut term_ids,
    );

    let main_body = mk_term(
        TermKind::Let {
            name: tmp_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer_call),
            body: Box::new(consumer),
        },
        i32_ty(),
        &mut term_ids,
    );
    let main_def = Def {
        name: main_sym,
        ty: i32_ty(),
        body: main_body,
        meta: DefMeta::Function,
        arity: 0,
    };

    let program = Program {
        defs: vec![my_map_def, main_def],
        symbols,
        def_syms: HashMap::new(),
    };

    let fused = run(program);
    let main = fused.defs.iter().find(|d| d.name == main_sym).unwrap();
    match &main.body.kind {
        TermKind::Soac(SoacOp::Redomap { op, inputs, .. }) => {
            assert_eq!(inputs.len(), 1);
            match &inputs[0] {
                ArrayExpr::Ref(t) => {
                    assert!(matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if *s == arr_sym))
                }
                other => panic!("Expected Ref(arr), got {:?}", other),
            }
            assert_eq!(op.lam.params.len(), 2);
        }
        other => panic!("Expected fused Redomap in main, got {:?}", other),
    }
}

// Test 4: Both map and reduce in separate defs (full interprocedural)
// def myMap(xs) = map(f, xs)
// def myReduce(xs) = reduce(op, ne, xs)
// main: let tmp = myMap(arr) in myReduce(tmp)
#[test]
fn test_raytrace_step4_both_interprocedural() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let xs2_sym = symbols.alloc("xs2".to_string());
    let arr_sym = symbols.alloc("arr".to_string());
    let tmp_sym = symbols.alloc("tmp".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let acc_sym = symbols.alloc("acc".to_string());
    let elem_sym = symbols.alloc("elem".to_string());
    let my_map_sym = symbols.alloc("myMap".to_string());
    let my_reduce_sym = symbols.alloc("myReduce".to_string());
    let main_sym = symbols.alloc("main".to_string());

    // def myMap(xs) = map(f, xs)
    let f = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let map_body = mk_map(
        f,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let my_map_def = mk_func_def(
        my_map_sym,
        vec![(xs_sym, array_ty(i32_ty()))],
        map_body,
        array_ty(i32_ty()),
    );

    // def myReduce(xs2) = reduce(op, ne, xs2)
    let op = mk_lambda2(
        acc_sym,
        i32_ty(),
        elem_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(acc_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let ne = mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids);
    let reduce_body = mk_reduce(
        op,
        ne,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs2_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        i32_ty(),
        &mut term_ids,
    );
    let my_reduce_def = mk_func_def(
        my_reduce_sym,
        vec![(xs2_sym, array_ty(i32_ty()))],
        reduce_body,
        i32_ty(),
    );

    // main: let tmp = myMap(arr) in myReduce(tmp)
    let producer_call = mk_app(
        mk_term(
            TermKind::Var(VarRef::Symbol(my_map_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        mk_term(
            TermKind::Var(VarRef::Symbol(arr_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let consumer_call = mk_app(
        mk_term(
            TermKind::Var(VarRef::Symbol(my_reduce_sym)),
            i32_ty(),
            &mut term_ids,
        ),
        mk_term(
            TermKind::Var(VarRef::Symbol(tmp_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        i32_ty(),
        &mut term_ids,
    );

    let main_body = mk_term(
        TermKind::Let {
            name: tmp_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer_call),
            body: Box::new(consumer_call),
        },
        i32_ty(),
        &mut term_ids,
    );
    let main_def = Def {
        name: main_sym,
        ty: i32_ty(),
        body: main_body,
        meta: DefMeta::Function,
        arity: 0,
    };

    let program = Program {
        defs: vec![my_map_def, my_reduce_def, main_def],
        symbols,
        def_syms: HashMap::new(),
    };

    let fused = run(program);
    let main = fused.defs.iter().find(|d| d.name == main_sym).unwrap();
    match &main.body.kind {
        TermKind::Soac(SoacOp::Redomap { op, inputs, .. }) => {
            assert_eq!(inputs.len(), 1);
            match &inputs[0] {
                ArrayExpr::Ref(t) => {
                    assert!(matches!(&t.kind, TermKind::Var(VarRef::Symbol(s)) if *s == arr_sym))
                }
                other => panic!("Expected Ref(arr), got {:?}", other),
            }
            assert_eq!(op.lam.params.len(), 2);
        }
        other => panic!("Expected fused Redomap in main, got {:?}", other),
    }
}

// Test 5: Raytrace pattern — map inputs from globals, not params
// def intersectAll(ro, rd) = map(f, globalArr)  ← NOT from params
// def findClosest(hits) = reduce(op, ne, hits)  ← from param
// main: let tmp = intersectAll(r, d) in findClosest(tmp)
//
// intersectAll gets Unknown summary. This test documents the current limitation.
#[test]
fn test_raytrace_step5_globals_pattern_fused() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let ro_sym = symbols.alloc("ro".to_string());
    let rd_sym = symbols.alloc("rd".to_string());
    let hits_sym = symbols.alloc("hits".to_string());
    let tmp_sym = symbols.alloc("tmp".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let acc_sym = symbols.alloc("acc".to_string());
    let elem_sym = symbols.alloc("elem".to_string());
    let global_arr_sym = symbols.alloc("globalArr".to_string());
    let intersect_sym = symbols.alloc("intersectAll".to_string());
    let find_closest_sym = symbols.alloc("findClosest".to_string());
    let main_sym = symbols.alloc("main".to_string());

    // def intersectAll(ro, rd) = map(f, globalArr)
    let f = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let global_ref = mk_term(
        TermKind::Var(VarRef::Symbol(global_arr_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let map_body = mk_map(f, global_ref, array_ty(i32_ty()), &mut term_ids);
    let intersect_def = mk_func_def(
        intersect_sym,
        vec![(ro_sym, f32_ty()), (rd_sym, f32_ty())],
        map_body,
        array_ty(i32_ty()),
    );

    // def findClosest(hits) = reduce(op, ne, hits)
    let op = mk_lambda2(
        acc_sym,
        i32_ty(),
        elem_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(acc_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let ne = mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids);
    let reduce_body = mk_reduce(
        op,
        ne,
        mk_term(
            TermKind::Var(VarRef::Symbol(hits_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        i32_ty(),
        &mut term_ids,
    );
    let find_closest_def = mk_func_def(
        find_closest_sym,
        vec![(hits_sym, array_ty(i32_ty()))],
        reduce_body,
        i32_ty(),
    );

    // main: let tmp = intersectAll(ro, rd) in findClosest(tmp)
    let producer_call = mk_term(
        TermKind::App {
            func: Box::new(mk_term(
                TermKind::Var(VarRef::Symbol(intersect_sym)),
                array_ty(i32_ty()),
                &mut term_ids,
            )),
            args: vec![
                mk_term(TermKind::Var(VarRef::Symbol(ro_sym)), f32_ty(), &mut term_ids),
                mk_term(TermKind::Var(VarRef::Symbol(rd_sym)), f32_ty(), &mut term_ids),
            ],
        },
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let consumer_call = mk_app(
        mk_term(
            TermKind::Var(VarRef::Symbol(find_closest_sym)),
            i32_ty(),
            &mut term_ids,
        ),
        mk_term(
            TermKind::Var(VarRef::Symbol(tmp_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        i32_ty(),
        &mut term_ids,
    );

    let main_body = mk_term(
        TermKind::Let {
            name: tmp_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer_call),
            body: Box::new(consumer_call),
        },
        i32_ty(),
        &mut term_ids,
    );
    let main_def = Def {
        name: main_sym,
        ty: i32_ty(),
        body: main_body,
        meta: DefMeta::Function,
        arity: 0,
    };

    let program = Program {
        defs: vec![intersect_def, find_closest_def, main_def],
        symbols,
        def_syms: HashMap::new(),
    };

    let fused = run(program);
    let main = fused.defs.iter().find(|d| d.name == main_sym).unwrap();
    // Should fuse: intersectAll produces a Map (ProducesMap summary)
    match &main.body.kind {
        TermKind::Soac(SoacOp::Redomap { op, .. }) => {
            assert_eq!(op.lam.params.len(), 2);
        }
        other => panic!("Expected fused Redomap, got {:?}", other),
    }
}
