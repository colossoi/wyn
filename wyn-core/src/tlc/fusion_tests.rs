use super::super::{DefMeta, SoacOp};
use crate::ast::Span;
use crate::tlc::fusion::*;
use crate::tlc::{ScremaAccumulator, SoacBody, SoacDestination};
use std::collections::HashMap;

fn dummy_span() -> Span {
    Span::new(0, 0, 0, 0)
}

/// A fused `map → reduce` is a scalar-output single-`Reduce`-accumulator,
/// no-map `Screma` (no `TupleProj` wrapper). Destructure that shape, returning
/// the step lambda (the composed `(acc, x) -> acc'`) and the SOAC inputs.
fn as_fused_map_reduce(body: &Term) -> (SoacBody, Vec<ArrayExpr>) {
    match fused_soac(body) {
        SoacOp::Screma {
            lanes,
            accumulators,
            inputs,
        } if lanes.is_empty()
            && accumulators.len() == 1
            && matches!(accumulators[0].kind, ScremaAccumulator::Reduce) =>
        {
            (accumulators[0].step_lam.clone(), inputs)
        }
        other => panic!("Expected fused single-reduce Screma, got {:?}", other),
    }
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
            crate::types::no_buffer(),
        ],
    )
}

/// Wrap a variable `Term` as an ANF SOAC input atom.
fn input_ae(input: Box<Term>) -> ArrayExpr {
    let t = *input;
    match t.kind {
        TermKind::Var(vr) => ArrayExpr::Var(vr, t.ty),
        other => panic!("test SOAC input must be a variable, got {other:?}"),
    }
}

/// Build a simple map: `map(lam, [input])`
fn mk_map(lam: Lambda, input: Term, result_ty: Type<TypeName>, term_ids: &mut TermIdSource) -> Term {
    mk_term(
        TermKind::Soac(SoacOp::Map {
            lam: mk_soac_body(lam),
            inputs: vec![input_ae(Box::new(input))],
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
            input: input_ae(Box::new(input)),
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
            input: input_ae(Box::new(input)),
            destination: SoacDestination::Fresh,
        }),
        result_ty,
        term_ids,
    )
}

fn mk_filter(pred: Lambda, input: Term, result_ty: Type<TypeName>, term_ids: &mut TermIdSource) -> Term {
    mk_term(
        TermKind::Soac(SoacOp::Filter {
            map_lam: None,
            pred: mk_soac_body(pred),
            input: input_ae(Box::new(input)),
            destination: SoacDestination::Fresh,
        }),
        result_ty,
        term_ids,
    )
}

fn mk_length(input: Term, term_ids: &mut TermIdSource) -> Term {
    let func = mk_term(
        TermKind::Var(VarRef::Builtin {
            id: crate::builtins::catalog().known().length,
            overload_idx: 0,
        }),
        Type::Variable(0),
        term_ids,
    );
    mk_term(
        TermKind::App {
            func: Box::new(func),
            args: vec![input],
        },
        i32_ty(),
        term_ids,
    )
}

fn contains_screma(term: &Term) -> bool {
    match &term.kind {
        TermKind::Soac(SoacOp::Screma { .. }) => true,
        _ => {
            let mut found = false;
            term.for_each_child(&mut |child| {
                if !found && contains_screma(child) {
                    found = true;
                }
            });
            found
        }
    }
}

fn contains_filter(term: &Term) -> bool {
    match &term.kind {
        TermKind::Soac(SoacOp::Filter { .. }) => true,
        _ => {
            let mut found = false;
            term.for_each_child(&mut |child| {
                if !found && contains_filter(child) {
                    found = true;
                }
            });
            found
        }
    }
}

fn find_first_screma(term: &Term) -> Option<SoacOp> {
    match &term.kind {
        TermKind::Soac(op @ SoacOp::Screma { .. }) => Some(op.clone()),
        _ => {
            let mut found = None;
            term.for_each_child(&mut |child| {
                if found.is_none() {
                    found = find_first_screma(child);
                }
            });
            found
        }
    }
}

/// The single fused SOAC in a result body, regardless of how the lowering
/// spells it: a bare tail SOAC, or a `let s = SOAC in s` / `… in s.k` wrapper
/// (the union path's single-output shape). Peels the let-chain and asserts
/// exactly one SOAC region is present. Use this instead of matching
/// `TermKind::Soac(SoacOp::Map { .. })` directly, so a test asserts *that fusion
/// happened*, not *how the fused region is currently spelled*.
fn fused_soac(body: &Term) -> SoacOp {
    let (bindings, tail) = crate::tlc::LetChain::from_term(body.clone()).into_parts();
    if let TermKind::Soac(op) = &tail.kind {
        return op.clone();
    }
    let soacs: Vec<SoacOp> = bindings
        .iter()
        .filter_map(|b| match &b.rhs.kind {
            TermKind::Soac(op) => Some(op.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        soacs.len(),
        1,
        "expected exactly one fused SOAC region, got {}",
        soacs.len()
    );
    soacs.into_iter().next().unwrap()
}

/// Count SOAC nodes anywhere in a term. "Fused to one region" is `== 1`; "these
/// producers stayed separate" is a count that didn't collapse.
fn count_soacs(term: &Term) -> usize {
    let mut n = usize::from(matches!(term.kind, TermKind::Soac(_)));
    term.for_each_child(&mut |c| n += count_soacs(c));
    n
}

/// The source symbols a SOAC reads, across every SOAC family — variant-agnostic,
/// so a test can assert "the fused region reads the original input(s)" without
/// caring whether it's a `Map`, `Scan`, `Reduce`, or `Screma`.
fn soac_input_syms(op: &SoacOp) -> Vec<SymbolId> {
    let inputs: Vec<&ArrayExpr> = match op {
        SoacOp::Map { inputs, .. } | SoacOp::Screma { inputs, .. } | SoacOp::Scatter { inputs, .. } => {
            inputs.iter().collect()
        }
        SoacOp::Reduce { input, .. } | SoacOp::Scan { input, .. } | SoacOp::Filter { input, .. } => {
            vec![input]
        }
        SoacOp::ReduceByIndex { indices, values, .. } => vec![indices, values],
    };
    inputs
        .into_iter()
        .filter_map(|ae| match ae {
            ArrayExpr::Var(VarRef::Symbol(s), _) => Some(*s),
            _ => None,
        })
        .collect()
}

/// The widest `Screma` lane-count anywhere in a term. `<= 1` means no horizontal
/// map fusion occurred (a single `map∘map` compose is one lane; merging two maps
/// is two).
fn max_screma_lanes(term: &Term) -> usize {
    let mut m = match &term.kind {
        TermKind::Soac(SoacOp::Screma { lanes, .. }) => lanes.len(),
        _ => 0,
    };
    term.for_each_child(&mut |c| m = m.max(max_screma_lanes(c)));
    m
}

fn projection_idx_for_binding(term: &Term, sym: SymbolId) -> Option<usize> {
    let (bindings, _) = crate::tlc::LetChain::from_term(term.clone()).into_parts();
    bindings.into_iter().find_map(|binding| {
        if binding.name != sym {
            return None;
        }
        match binding.rhs.kind {
            TermKind::TupleProj { idx, .. } => Some(idx),
            _ => None,
        }
    })
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
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);

    // Fusion eliminated the intermediate `b`: the two maps collapse to a single
    // SOAC region reading the original input `a` — regardless of whether that
    // region is spelled as a bare `Map` or a one-lane `Screma`.
    let body = &fused.defs[0].body;
    assert_eq!(count_soacs(body), 1, "the two maps must fuse to one region");
    assert_eq!(
        soac_input_syms(&fused_soac(body)),
        vec![a_sym],
        "the fused region reads the original input a, not the intermediate b"
    );
}

// -------------------------------------------------------------------------
// Test: reduce(op, ne, filter(p, xs)) → masked single-accumulator Screma (step op∘mask, combiner op, ne, xs)
// -------------------------------------------------------------------------
#[test]
fn test_horizontal_sibling_maps_merge_to_multi_output_screma() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let c_sym = symbols.alloc("c".to_string());
    let d_sym = symbols.alloc("d".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());

    let f_body = mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids);
    let f = mk_lambda1(x_sym, i32_ty(), f_body, i32_ty());
    let g_body = mk_term(TermKind::Var(VarRef::Symbol(y_sym)), i32_ty(), &mut term_ids);
    let g = mk_lambda1(y_sym, i32_ty(), g_body, i32_ty());

    let xs_for_c = mk_term(
        TermKind::Var(VarRef::Symbol(xs_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let xs_for_d = mk_term(
        TermKind::Var(VarRef::Symbol(xs_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let c_map = mk_map(f, xs_for_c, array_ty(i32_ty()), &mut term_ids);
    let d_map = mk_map(g, xs_for_d, array_ty(i32_ty()), &mut term_ids);

    let c_ref = mk_term(
        TermKind::Var(VarRef::Symbol(c_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let d_ref = mk_term(
        TermKind::Var(VarRef::Symbol(d_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![array_ty(i32_ty()), array_ty(i32_ty())]);
    let tail = mk_term(
        TermKind::Tuple(vec![c_ref, d_ref]),
        tuple_ty.clone(),
        &mut term_ids,
    );

    let inner = mk_term(
        TermKind::Let {
            name: d_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(d_map),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: c_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(c_map),
            body: Box::new(inner),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);

    let TermKind::Let {
        name: merged_sym,
        rhs: merged_rhs,
        body: c_let,
        ..
    } = &fused.defs[0].body.kind
    else {
        panic!("expected merged map let");
    };

    match &merged_rhs.kind {
        TermKind::Soac(SoacOp::Screma {
            lanes,
            accumulators,
            inputs,
        }) => {
            assert_eq!(inputs.len(), 1);
            assert!(matches!(
                &inputs[0],
                ArrayExpr::Var(VarRef::Symbol(sym), _) if *sym == xs_sym
            ));
            assert!(accumulators.is_empty());
            assert_eq!(lanes.len(), 2);
            assert_eq!(lanes[0].lam.lam.params.len(), 1);
            assert_eq!(lanes[0].lam.lam.params[0].0, x_sym);
            assert_eq!(lanes[1].lam.lam.params.len(), 1);
            assert_eq!(lanes[1].lam.lam.params[0].0, y_sym);
            assert!(
                matches!(&lanes[0].lam.lam.body.kind, TermKind::Var(VarRef::Symbol(sym)) if *sym == x_sym)
            );
            assert!(
                matches!(&lanes[1].lam.lam.body.kind, TermKind::Var(VarRef::Symbol(sym)) if *sym == y_sym)
            );
        }
        other => panic!("expected merged Screma rhs, got {:?}", other),
    }

    let TermKind::Let {
        name: c_name,
        rhs: c_rhs,
        body: d_let,
        ..
    } = &c_let.kind
    else {
        panic!("expected c projection let");
    };
    assert_eq!(*c_name, c_sym);
    assert!(matches!(
        &c_rhs.kind,
        TermKind::TupleProj { tuple, idx: 0 }
            if matches!(&tuple.kind, TermKind::Var(VarRef::Symbol(sym)) if sym == merged_sym)
    ));

    let TermKind::Let {
        name: d_name,
        rhs: d_rhs,
        ..
    } = &d_let.kind
    else {
        panic!("expected d projection let");
    };
    assert_eq!(*d_name, d_sym);
    assert!(matches!(
        &d_rhs.kind,
        TermKind::TupleProj { tuple, idx: 1 }
            if matches!(&tuple.kind, TermKind::Var(VarRef::Symbol(sym)) if sym == merged_sym)
    ));
}

#[test]
fn test_horizontal_sibling_maps_keep_different_inputs_separate() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let ys_sym = symbols.alloc("ys".to_string());
    let c_sym = symbols.alloc("c".to_string());
    let d_sym = symbols.alloc("d".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());

    let f_body = mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids);
    let f = mk_lambda1(x_sym, i32_ty(), f_body, i32_ty());
    let g_body = mk_term(TermKind::Var(VarRef::Symbol(y_sym)), i32_ty(), &mut term_ids);
    let g = mk_lambda1(y_sym, i32_ty(), g_body, i32_ty());

    let xs = mk_term(
        TermKind::Var(VarRef::Symbol(xs_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let ys = mk_term(
        TermKind::Var(VarRef::Symbol(ys_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let c_map = mk_map(f, xs, array_ty(i32_ty()), &mut term_ids);
    let d_map = mk_map(g, ys, array_ty(i32_ty()), &mut term_ids);

    let c_ref = mk_term(
        TermKind::Var(VarRef::Symbol(c_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let d_ref = mk_term(
        TermKind::Var(VarRef::Symbol(d_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![array_ty(i32_ty()), array_ty(i32_ty())]);
    let tail = mk_term(
        TermKind::Tuple(vec![c_ref, d_ref]),
        tuple_ty.clone(),
        &mut term_ids,
    );

    let inner = mk_term(
        TermKind::Let {
            name: d_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(d_map),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: c_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(c_map),
            body: Box::new(inner),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);
    let TermKind::Let { name, rhs, body, .. } = &fused.defs[0].body.kind else {
        panic!("expected original outer let");
    };
    assert_eq!(*name, c_sym);
    assert!(matches!(&rhs.kind, TermKind::Soac(SoacOp::Map { .. })));
    assert!(matches!(
        &body.kind,
        TermKind::Let { name, rhs, .. }
            if *name == d_sym && matches!(&rhs.kind, TermKind::Soac(SoacOp::Map { .. }))
    ));
}

#[test]
fn test_horizontal_sibling_maps_merge_same_input_vector() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let ys_sym = symbols.alloc("ys".to_string());
    let c_sym = symbols.alloc("c".to_string());
    let d_sym = symbols.alloc("d".to_string());
    let x1_sym = symbols.alloc("x1".to_string());
    let y1_sym = symbols.alloc("y1".to_string());
    let x2_sym = symbols.alloc("x2".to_string());
    let y2_sym = symbols.alloc("y2".to_string());

    let f = Lambda {
        params: vec![(x1_sym, i32_ty()), (y1_sym, i32_ty())],
        body: Box::new(mk_term(
            TermKind::Var(VarRef::Symbol(x1_sym)),
            i32_ty(),
            &mut term_ids,
        )),
        ret_ty: i32_ty(),
    };
    let g = Lambda {
        params: vec![(x2_sym, i32_ty()), (y2_sym, i32_ty())],
        body: Box::new(mk_term(
            TermKind::Var(VarRef::Symbol(y2_sym)),
            i32_ty(),
            &mut term_ids,
        )),
        ret_ty: i32_ty(),
    };

    let mk_input = |sym, term_ids: &mut TermIdSource| {
        mk_term(TermKind::Var(VarRef::Symbol(sym)), array_ty(i32_ty()), term_ids)
    };
    let c_map = mk_term(
        TermKind::Soac(SoacOp::Map {
            lam: mk_soac_body(f),
            inputs: vec![
                input_ae(Box::new(mk_input(xs_sym, &mut term_ids))),
                input_ae(Box::new(mk_input(ys_sym, &mut term_ids))),
            ],
            destination: SoacDestination::Fresh,
        }),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let d_map = mk_term(
        TermKind::Soac(SoacOp::Map {
            lam: mk_soac_body(g),
            inputs: vec![
                input_ae(Box::new(mk_input(xs_sym, &mut term_ids))),
                input_ae(Box::new(mk_input(ys_sym, &mut term_ids))),
            ],
            destination: SoacDestination::Fresh,
        }),
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let c_ref = mk_term(
        TermKind::Var(VarRef::Symbol(c_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let d_ref = mk_term(
        TermKind::Var(VarRef::Symbol(d_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![array_ty(i32_ty()), array_ty(i32_ty())]);
    let tail = mk_term(
        TermKind::Tuple(vec![c_ref, d_ref]),
        tuple_ty.clone(),
        &mut term_ids,
    );

    let inner = mk_term(
        TermKind::Let {
            name: d_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(d_map),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: c_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(c_map),
            body: Box::new(inner),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);
    let TermKind::Let { rhs: merged_rhs, .. } = &fused.defs[0].body.kind else {
        panic!("expected merged map let");
    };

    match &merged_rhs.kind {
        TermKind::Soac(SoacOp::Screma {
            lanes,
            accumulators,
            inputs,
        }) => {
            assert_eq!(inputs.len(), 2);
            assert!(accumulators.is_empty());
            assert_eq!(lanes.len(), 2);
            assert_eq!(lanes[0].lam.lam.params.len(), 2);
            assert_eq!(lanes[0].lam.lam.params[0].0, x1_sym);
            assert_eq!(lanes[0].lam.lam.params[1].0, y1_sym);
            assert_eq!(lanes[1].lam.lam.params.len(), 2);
            assert_eq!(lanes[1].lam.lam.params[0].0, x2_sym);
            assert_eq!(lanes[1].lam.lam.params[1].0, y2_sym);
            assert!(
                matches!(&lanes[0].lam.lam.body.kind, TermKind::Var(VarRef::Symbol(sym)) if *sym == x1_sym)
            );
            assert!(
                matches!(&lanes[1].lam.lam.body.kind, TermKind::Var(VarRef::Symbol(sym)) if *sym == y2_sym)
            );
        }
        other => panic!("expected merged Screma rhs, got {:?}", other),
    }
}

#[test]
fn test_horizontal_sibling_maps_enable_shared_producer_vertical_fusion() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let c_sym = symbols.alloc("c".to_string());
    let d_sym = symbols.alloc("d".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());
    let z_sym = symbols.alloc("z".to_string());

    let f_body = mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids);
    let f = mk_lambda1(x_sym, i32_ty(), f_body, i32_ty());
    let g_body = mk_term(TermKind::Var(VarRef::Symbol(y_sym)), i32_ty(), &mut term_ids);
    let g = mk_lambda1(y_sym, i32_ty(), g_body, i32_ty());
    let h_body = mk_term(TermKind::Var(VarRef::Symbol(z_sym)), i32_ty(), &mut term_ids);
    let h = mk_lambda1(z_sym, i32_ty(), h_body, i32_ty());

    let xs = mk_term(
        TermKind::Var(VarRef::Symbol(xs_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let b_map = mk_map(f, xs, array_ty(i32_ty()), &mut term_ids);
    let b_for_c = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let b_for_d = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let c_map = mk_map(g, b_for_c, array_ty(i32_ty()), &mut term_ids);
    let d_map = mk_map(h, b_for_d, array_ty(i32_ty()), &mut term_ids);

    let c_ref = mk_term(
        TermKind::Var(VarRef::Symbol(c_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let d_ref = mk_term(
        TermKind::Var(VarRef::Symbol(d_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![array_ty(i32_ty()), array_ty(i32_ty())]);
    let tail = mk_term(
        TermKind::Tuple(vec![c_ref, d_ref]),
        tuple_ty.clone(),
        &mut term_ids,
    );

    let d_let = mk_term(
        TermKind::Let {
            name: d_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(d_map),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let c_let = mk_term(
        TermKind::Let {
            name: c_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(c_map),
            body: Box::new(d_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: b_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(b_map),
            body: Box::new(c_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);
    let TermKind::Let {
        name: merged_sym,
        rhs: merged_rhs,
        body: c_let,
        ..
    } = &fused.defs[0].body.kind
    else {
        panic!("expected shared producer to fuse into merged map let");
    };
    assert_ne!(
        *merged_sym, b_sym,
        "the original shared producer binding should be gone"
    );

    match &merged_rhs.kind {
        TermKind::Soac(SoacOp::Screma {
            lanes,
            accumulators,
            inputs,
        }) => {
            assert_eq!(inputs.len(), 1);
            assert!(matches!(
                &inputs[0],
                ArrayExpr::Var(VarRef::Symbol(sym), _) if *sym == xs_sym
            ));
            assert!(accumulators.is_empty());
            assert_eq!(lanes.len(), 2);
            assert_eq!(lanes[0].lam.lam.params.len(), 1);
            assert_eq!(lanes[0].lam.lam.params[0].0, x_sym);
            assert_eq!(lanes[1].lam.lam.params.len(), 1);
            assert_eq!(lanes[1].lam.lam.params[0].0, x_sym);
            assert!(matches!(&lanes[0].lam.lam.body.kind, TermKind::Let { .. }));
            assert!(matches!(&lanes[1].lam.lam.body.kind, TermKind::Let { .. }));
        }
        other => panic!("expected merged Screma rhs, got {:?}", other),
    }

    assert!(matches!(
        &c_let.kind,
        TermKind::Let { name, rhs, .. }
            if *name == c_sym && matches!(&rhs.kind, TermKind::TupleProj { idx: 0, .. })
    ));
}

#[test]
fn test_screma_fuses_shared_map_producer_into_map_and_reduce() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let c_sym = symbols.alloc("c".to_string());
    let d_sym = symbols.alloc("d".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());
    let acc_sym = symbols.alloc("acc".to_string());
    let elem_sym = symbols.alloc("elem".to_string());

    let f_body = mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids);
    let f = mk_lambda1(x_sym, i32_ty(), f_body, i32_ty());
    let g_body = mk_term(TermKind::Var(VarRef::Symbol(y_sym)), i32_ty(), &mut term_ids);
    let g = mk_lambda1(y_sym, i32_ty(), g_body, i32_ty());
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
    let b_map = mk_map(f, xs, array_ty(i32_ty()), &mut term_ids);
    let b_for_c = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let b_for_d = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let c_map = mk_map(g, b_for_c, array_ty(i32_ty()), &mut term_ids);
    let ne = mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids);
    let d_reduce = mk_reduce(op, ne, b_for_d, i32_ty(), &mut term_ids);

    let c_ref = mk_term(
        TermKind::Var(VarRef::Symbol(c_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let d_ref = mk_term(TermKind::Var(VarRef::Symbol(d_sym)), i32_ty(), &mut term_ids);
    let tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![array_ty(i32_ty()), i32_ty()]);
    let tail = mk_term(
        TermKind::Tuple(vec![c_ref, d_ref]),
        tuple_ty.clone(),
        &mut term_ids,
    );

    let d_let = mk_term(
        TermKind::Let {
            name: d_sym,
            name_ty: i32_ty(),
            rhs: Box::new(d_reduce),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let c_let = mk_term(
        TermKind::Let {
            name: c_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(c_map),
            body: Box::new(d_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: b_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(b_map),
            body: Box::new(c_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);
    let TermKind::Let {
        name: screma_sym,
        rhs: screma_rhs,
        body: c_let,
        ..
    } = &fused.defs[0].body.kind
    else {
        panic!("expected screma let");
    };
    assert_ne!(*screma_sym, b_sym);
    match &screma_rhs.kind {
        TermKind::Soac(SoacOp::Screma {
            lanes,
            accumulators,
            inputs,
            ..
        }) => {
            assert_eq!(inputs.len(), 1);
            assert!(matches!(
                &inputs[0],
                ArrayExpr::Var(VarRef::Symbol(sym), _) if *sym == xs_sym
            ));
            assert_eq!(lanes.len(), 1);
            assert_eq!(accumulators.len(), 1);
            let map_lam = &lanes[0].lam;
            let reduce_lam = &accumulators[0].step_lam;
            assert_eq!(map_lam.lam.params.len(), 1);
            assert_eq!(map_lam.lam.params[0].0, x_sym);
            assert_eq!(reduce_lam.lam.params.len(), 2);
            assert_eq!(reduce_lam.lam.params[1].0, x_sym);
        }
        other => panic!("expected Screma rhs, got {:?}", other),
    }

    assert!(matches!(
        &c_let.kind,
        TermKind::Let { name, rhs, .. }
            if *name == c_sym && matches!(&rhs.kind, TermKind::TupleProj { idx: 0, .. })
    ));
}

#[test]
fn test_screma_fuses_shared_map_producer_into_map_and_scan() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let c_sym = symbols.alloc("c".to_string());
    let d_sym = symbols.alloc("d".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());
    let acc_sym = symbols.alloc("acc".to_string());
    let z_sym = symbols.alloc("z".to_string());

    let xs_ref = mk_term(
        TermKind::Var(VarRef::Symbol(xs_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let f_body = mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids);
    let f = mk_lambda1(x_sym, i32_ty(), f_body, i32_ty());
    let b_map = mk_map(f, xs_ref, array_ty(i32_ty()), &mut term_ids);

    let b_ref_for_map = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let g_body = mk_term(TermKind::Var(VarRef::Symbol(y_sym)), i32_ty(), &mut term_ids);
    let g = mk_lambda1(y_sym, i32_ty(), g_body, i32_ty());
    let c_map = mk_map(g, b_ref_for_map, array_ty(i32_ty()), &mut term_ids);

    let b_ref_for_scan = mk_term(
        TermKind::Var(VarRef::Symbol(b_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let scan_body = mk_term(TermKind::Var(VarRef::Symbol(acc_sym)), i32_ty(), &mut term_ids);
    let scan_op = mk_lambda2(acc_sym, i32_ty(), z_sym, i32_ty(), scan_body, i32_ty());
    let ne = mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids);
    let d_scan = mk_scan(scan_op, ne, b_ref_for_scan, array_ty(i32_ty()), &mut term_ids);

    let tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![array_ty(i32_ty()), array_ty(i32_ty())]);
    let c_ref = mk_term(
        TermKind::Var(VarRef::Symbol(c_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let d_ref = mk_term(
        TermKind::Var(VarRef::Symbol(d_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let tail = mk_term(
        TermKind::Tuple(vec![c_ref, d_ref]),
        tuple_ty.clone(),
        &mut term_ids,
    );
    let d_let = mk_term(
        TermKind::Let {
            name: d_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(d_scan),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let c_let = mk_term(
        TermKind::Let {
            name: c_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(c_map),
            body: Box::new(d_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: b_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(b_map),
            body: Box::new(c_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);
    let TermKind::Let {
        rhs: screma_rhs,
        body: c_let,
        ..
    } = &fused.defs[0].body.kind
    else {
        panic!("expected screma let");
    };
    match &screma_rhs.kind {
        TermKind::Soac(SoacOp::Screma { accumulators, .. }) => {
            assert_eq!(accumulators.len(), 1);
            assert_eq!(accumulators[0].kind, ScremaAccumulator::Scan);
            let reduce_lam = &accumulators[0].step_lam;
            assert_eq!(reduce_lam.lam.params.len(), 2);
            assert_eq!(reduce_lam.lam.params[1].0, x_sym);
        }
        other => panic!("expected Screma rhs, got {:?}", other),
    }

    assert!(matches!(
        &c_let.kind,
        TermKind::Let { body, .. }
            if matches!(&body.kind, TermKind::Let { name, rhs, .. }
                if *name == d_sym && matches!(&rhs.kind, TermKind::TupleProj { idx: 1, .. }))
    ));
}

#[test]
fn test_screma_fuses_across_independent_let() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let k_sym = symbols.alloc("k".to_string());
    let c_sym = symbols.alloc("c".to_string());
    let d_sym = symbols.alloc("d".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());
    let acc_sym = symbols.alloc("acc".to_string());
    let elem_sym = symbols.alloc("elem".to_string());

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
    let op = mk_lambda2(
        acc_sym,
        i32_ty(),
        elem_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(acc_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );

    let b_map = mk_map(
        f,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let k_rhs = mk_term(TermKind::IntLit("7".to_string()), i32_ty(), &mut term_ids);
    let c_map = mk_map(
        g,
        mk_term(
            TermKind::Var(VarRef::Symbol(b_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let d_reduce = mk_reduce(
        op,
        mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids),
        mk_term(
            TermKind::Var(VarRef::Symbol(b_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        i32_ty(),
        &mut term_ids,
    );

    let tuple_ty = Type::Constructed(TypeName::Tuple(3), vec![i32_ty(), array_ty(i32_ty()), i32_ty()]);
    let tail = mk_term(
        TermKind::Tuple(vec![
            mk_term(TermKind::Var(VarRef::Symbol(k_sym)), i32_ty(), &mut term_ids),
            mk_term(
                TermKind::Var(VarRef::Symbol(c_sym)),
                array_ty(i32_ty()),
                &mut term_ids,
            ),
            mk_term(TermKind::Var(VarRef::Symbol(d_sym)), i32_ty(), &mut term_ids),
        ]),
        tuple_ty.clone(),
        &mut term_ids,
    );
    let d_let = mk_term(
        TermKind::Let {
            name: d_sym,
            name_ty: i32_ty(),
            rhs: Box::new(d_reduce),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let c_let = mk_term(
        TermKind::Let {
            name: c_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(c_map),
            body: Box::new(d_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let k_let = mk_term(
        TermKind::Let {
            name: k_sym,
            name_ty: i32_ty(),
            rhs: Box::new(k_rhs),
            body: Box::new(c_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: b_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(b_map),
            body: Box::new(k_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);
    let TermKind::Let {
        rhs: screma_rhs,
        body: k_let,
        ..
    } = &fused.defs[0].body.kind
    else {
        panic!("expected Screma let before independent binding");
    };
    assert!(matches!(
        &screma_rhs.kind,
        TermKind::Soac(SoacOp::Screma {
            lanes,
            accumulators,
            ..
        }) if lanes.len() == 1 && accumulators.len() == 1
    ));
    assert!(matches!(
        &k_let.kind,
        TermKind::Let { name, body, .. }
            if *name == k_sym
                && matches!(&body.kind, TermKind::Let { name, rhs, .. }
                    if *name == c_sym && matches!(&rhs.kind, TermKind::TupleProj { idx: 0, .. }))
    ));
}

#[test]
fn test_screma_rejects_dependent_sibling() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let c_sym = symbols.alloc("c".to_string());
    let d_sym = symbols.alloc("d".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());
    let acc_sym = symbols.alloc("acc".to_string());
    let elem_sym = symbols.alloc("elem".to_string());

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
    let op = mk_lambda2(
        acc_sym,
        i32_ty(),
        elem_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(c_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );

    let b_map = mk_map(
        f,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let c_map = mk_map(
        g,
        mk_term(
            TermKind::Var(VarRef::Symbol(b_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let d_reduce = mk_reduce(
        op,
        mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids),
        mk_term(
            TermKind::Var(VarRef::Symbol(b_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        i32_ty(),
        &mut term_ids,
    );
    let tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![array_ty(i32_ty()), i32_ty()]);
    let tail = mk_term(
        TermKind::Tuple(vec![
            mk_term(
                TermKind::Var(VarRef::Symbol(c_sym)),
                array_ty(i32_ty()),
                &mut term_ids,
            ),
            mk_term(TermKind::Var(VarRef::Symbol(d_sym)), i32_ty(), &mut term_ids),
        ]),
        tuple_ty.clone(),
        &mut term_ids,
    );
    let d_let = mk_term(
        TermKind::Let {
            name: d_sym,
            name_ty: i32_ty(),
            rhs: Box::new(d_reduce),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let c_let = mk_term(
        TermKind::Let {
            name: c_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(c_map),
            body: Box::new(d_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: b_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(b_map),
            body: Box::new(c_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);
    assert!(!contains_screma(&fused.defs[0].body));
}

#[test]
fn test_screma_rejects_unsupported_filter_consumer() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let c_sym = symbols.alloc("c".to_string());
    let d_sym = symbols.alloc("d".to_string());
    let e_sym = symbols.alloc("e".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());
    let acc_sym = symbols.alloc("acc".to_string());
    let elem_sym = symbols.alloc("elem".to_string());
    let pred_sym = symbols.alloc("pred".to_string());

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
    let op = mk_lambda2(
        acc_sym,
        i32_ty(),
        elem_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(acc_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let pred = mk_lambda1(
        pred_sym,
        i32_ty(),
        mk_term(
            TermKind::BoolLit(true),
            Type::Constructed(TypeName::Bool, vec![]),
            &mut term_ids,
        ),
        Type::Constructed(TypeName::Bool, vec![]),
    );

    let b_map = mk_map(
        f,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let c_map = mk_map(
        g,
        mk_term(
            TermKind::Var(VarRef::Symbol(b_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let d_reduce = mk_reduce(
        op,
        mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids),
        mk_term(
            TermKind::Var(VarRef::Symbol(b_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        i32_ty(),
        &mut term_ids,
    );
    let e_filter = mk_filter(
        pred,
        mk_term(
            TermKind::Var(VarRef::Symbol(b_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let tuple_ty = Type::Constructed(
        TypeName::Tuple(3),
        vec![array_ty(i32_ty()), i32_ty(), array_ty(i32_ty())],
    );
    let tail = mk_term(
        TermKind::Tuple(vec![
            mk_term(
                TermKind::Var(VarRef::Symbol(c_sym)),
                array_ty(i32_ty()),
                &mut term_ids,
            ),
            mk_term(TermKind::Var(VarRef::Symbol(d_sym)), i32_ty(), &mut term_ids),
            mk_term(
                TermKind::Var(VarRef::Symbol(e_sym)),
                array_ty(i32_ty()),
                &mut term_ids,
            ),
        ]),
        tuple_ty.clone(),
        &mut term_ids,
    );
    let e_let = mk_term(
        TermKind::Let {
            name: e_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(e_filter),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let d_let = mk_term(
        TermKind::Let {
            name: d_sym,
            name_ty: i32_ty(),
            rhs: Box::new(d_reduce),
            body: Box::new(e_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let c_let = mk_term(
        TermKind::Let {
            name: c_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(c_map),
            body: Box::new(d_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: b_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(b_map),
            body: Box::new(c_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);
    assert!(!contains_screma(&fused.defs[0].body));
}

#[test]
fn test_screma_rejects_different_producers() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let ys_sym = symbols.alloc("ys".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let c_sym = symbols.alloc("c".to_string());
    let e_sym = symbols.alloc("e".to_string());
    let d_sym = symbols.alloc("d".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());
    let z_sym = symbols.alloc("z".to_string());
    let acc_sym = symbols.alloc("acc".to_string());
    let elem_sym = symbols.alloc("elem".to_string());

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
    let op = mk_lambda2(
        acc_sym,
        i32_ty(),
        elem_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(acc_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );

    let b_map = mk_map(
        f,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let c_map = mk_map(
        g,
        mk_term(
            TermKind::Var(VarRef::Symbol(b_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let e_map = mk_map(
        h,
        mk_term(
            TermKind::Var(VarRef::Symbol(ys_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let d_reduce = mk_reduce(
        op,
        mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids),
        mk_term(
            TermKind::Var(VarRef::Symbol(e_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        i32_ty(),
        &mut term_ids,
    );

    let tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![array_ty(i32_ty()), i32_ty()]);
    let tail = mk_term(
        TermKind::Tuple(vec![
            mk_term(
                TermKind::Var(VarRef::Symbol(c_sym)),
                array_ty(i32_ty()),
                &mut term_ids,
            ),
            mk_term(TermKind::Var(VarRef::Symbol(d_sym)), i32_ty(), &mut term_ids),
        ]),
        tuple_ty.clone(),
        &mut term_ids,
    );
    let d_let = mk_term(
        TermKind::Let {
            name: d_sym,
            name_ty: i32_ty(),
            rhs: Box::new(d_reduce),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let e_let = mk_term(
        TermKind::Let {
            name: e_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(e_map),
            body: Box::new(d_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let c_let = mk_term(
        TermKind::Let {
            name: c_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(c_map),
            body: Box::new(e_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: b_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(b_map),
            body: Box::new(c_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);
    // Vertical fusions are fine and expected (`c = map(g, b)` composes with its
    // producer `b`; `d = reduce(op, 0, e)` folds its producer `e`). What must NOT
    // happen is a *horizontal* merge of the two distinct producers (the
    // xs-derived chain vs the ys-derived chain) into one multi-lane Screma — a
    // single map∘map compose is one lane, so the rejection signature is ≥2 lanes.
    assert!(
        max_screma_lanes(&fused.defs[0].body) <= 1,
        "maps over different producers must not horizontally fuse into a multi-lane Screma"
    );
}

#[test]
fn test_screma_rejects_tail_use_of_producer() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let c_sym = symbols.alloc("c".to_string());
    let d_sym = symbols.alloc("d".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let y_sym = symbols.alloc("y".to_string());
    let acc_sym = symbols.alloc("acc".to_string());
    let elem_sym = symbols.alloc("elem".to_string());

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
    let op = mk_lambda2(
        acc_sym,
        i32_ty(),
        elem_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(acc_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );

    let b_map = mk_map(
        f,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let c_map = mk_map(
        g,
        mk_term(
            TermKind::Var(VarRef::Symbol(b_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let d_reduce = mk_reduce(
        op,
        mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids),
        mk_term(
            TermKind::Var(VarRef::Symbol(b_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        i32_ty(),
        &mut term_ids,
    );

    let tuple_ty = Type::Constructed(
        TypeName::Tuple(3),
        vec![array_ty(i32_ty()), i32_ty(), array_ty(i32_ty())],
    );
    let tail = mk_term(
        TermKind::Tuple(vec![
            mk_term(
                TermKind::Var(VarRef::Symbol(c_sym)),
                array_ty(i32_ty()),
                &mut term_ids,
            ),
            mk_term(TermKind::Var(VarRef::Symbol(d_sym)), i32_ty(), &mut term_ids),
            mk_term(
                TermKind::Var(VarRef::Symbol(b_sym)),
                array_ty(i32_ty()),
                &mut term_ids,
            ),
        ]),
        tuple_ty.clone(),
        &mut term_ids,
    );
    let d_let = mk_term(
        TermKind::Let {
            name: d_sym,
            name_ty: i32_ty(),
            rhs: Box::new(d_reduce),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let c_let = mk_term(
        TermKind::Let {
            name: c_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(c_map),
            body: Box::new(d_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: b_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(b_map),
            body: Box::new(c_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);
    assert!(!contains_screma(&fused.defs[0].body));
}

#[test]
fn test_filter_into_reduce_fuses_to_masked_screma() {
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let k_sym = symbols.alloc("k".to_string()); // let-bound filter result
    let x_sym = symbols.alloc("x".to_string()); // pred param
    let a_sym = symbols.alloc("a".to_string()); // reduce acc
    let b_sym = symbols.alloc("b".to_string()); // reduce elem

    // pred: λx. true  (body content is irrelevant to fusion structure)
    let pred_body = mk_term(TermKind::BoolLit(true), bool_ty.clone(), &mut term_ids);
    let pred = mk_lambda1(x_sym, i32_ty(), pred_body, bool_ty);

    // producer: filter(pred, xs)
    let xs = mk_term(
        TermKind::Var(VarRef::Symbol(xs_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let producer = mk_filter(pred, xs, array_ty(i32_ty()), &mut term_ids);

    // reduce op: λ(a, b). a  (binary; body Var(a))
    let op_body = mk_term(TermKind::Var(VarRef::Symbol(a_sym)), i32_ty(), &mut term_ids);
    let op = mk_lambda2(a_sym, i32_ty(), b_sym, i32_ty(), op_body, i32_ty());
    let ne = mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids);

    // consumer: reduce(op, 0, k)
    let k_ref = mk_term(
        TermKind::Var(VarRef::Symbol(k_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let consumer = mk_reduce(op, ne, k_ref, i32_ty(), &mut term_ids);

    // let k = filter(pred, xs) in reduce(op, 0, k)
    let program_body = mk_term(
        TermKind::Let {
            name: k_sym,
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
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);

    assert!(!contains_filter(&fused.defs[0].body));
    match find_first_screma(&fused.defs[0].body).expect("expected fused Screma") {
        SoacOp::Screma {
            lanes,
            accumulators,
            inputs,
        } => {
            assert!(lanes.is_empty());
            assert_eq!(accumulators.len(), 1);
            assert_eq!(inputs.len(), 1);
            match &inputs[0] {
                ArrayExpr::Var(vr, _) => {
                    assert!(matches!(vr, VarRef::Symbol(s) if *s == xs_sym))
                }
                other => panic!("expected Ref(Var(xs)), got {other:?}"),
            }

            let acc = &accumulators[0];
            assert_eq!(acc.step_lam.lam.params.len(), 2);
            assert_eq!(
                acc.step_lam.lam.params[1].0, x_sym,
                "element param is the pred's param"
            );
            assert!(
                matches!(&acc.step_lam.lam.body.kind, TermKind::If { .. }),
                "filtered reduction step should be guarded by the predicate"
            );
            assert_eq!(acc.reduce_op.lam.params.len(), 2);
            assert!(
                matches!(&acc.reduce_op.lam.body.kind, TermKind::Var(VarRef::Symbol(s)) if *s == a_sym)
            );
            assert!(matches!(&acc.ne.kind, TermKind::IntLit(s) if s == "0"));
        }
        other => panic!("expected fused Soac(Screma), got {other:?}"),
    }
}

#[test]
fn test_filter_into_reduce_and_length_fuses_to_one_screma() {
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let k_sym = symbols.alloc("k".to_string());
    let r_sym = symbols.alloc("r".to_string());
    let n_sym = symbols.alloc("n".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let a_sym = symbols.alloc("a".to_string());
    let b_sym = symbols.alloc("b".to_string());

    let pred = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::BoolLit(true), bool_ty.clone(), &mut term_ids),
        bool_ty,
    );
    let producer = mk_filter(
        pred,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let op = mk_lambda2(
        a_sym,
        i32_ty(),
        b_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(a_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let reduce = mk_reduce(
        op,
        mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids),
        mk_term(
            TermKind::Var(VarRef::Symbol(k_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        i32_ty(),
        &mut term_ids,
    );
    let len = mk_length(
        mk_term(
            TermKind::Var(VarRef::Symbol(k_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        &mut term_ids,
    );
    let tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), i32_ty()]);
    let tail = mk_term(
        TermKind::Tuple(vec![
            mk_term(TermKind::Var(VarRef::Symbol(r_sym)), i32_ty(), &mut term_ids),
            mk_term(TermKind::Var(VarRef::Symbol(n_sym)), i32_ty(), &mut term_ids),
        ]),
        tuple_ty.clone(),
        &mut term_ids,
    );
    let n_let = mk_term(
        TermKind::Let {
            name: n_sym,
            name_ty: i32_ty(),
            rhs: Box::new(len),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let r_let = mk_term(
        TermKind::Let {
            name: r_sym,
            name_ty: i32_ty(),
            rhs: Box::new(reduce),
            body: Box::new(n_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: k_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(r_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let mut fused = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };
    run(&mut fused);

    assert!(!contains_filter(&fused.defs[0].body));
    match find_first_screma(&fused.defs[0].body).expect("expected filtered Screma") {
        SoacOp::Screma {
            lanes,
            accumulators,
            inputs,
        } => {
            assert!(lanes.is_empty());
            assert_eq!(accumulators.len(), 2);
            assert_eq!(inputs.len(), 1);
        }
        other => panic!("expected Screma, got {other:?}"),
    }
    assert_ne!(
        projection_idx_for_binding(&fused.defs[0].body, r_sym),
        projection_idx_for_binding(&fused.defs[0].body, n_sym)
    );
}

#[test]
fn test_filter_multiple_reduces_and_lengths_reuses_one_count_accumulator() {
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let k_sym = symbols.alloc("k".to_string());
    let r1_sym = symbols.alloc("r1".to_string());
    let r2_sym = symbols.alloc("r2".to_string());
    let n1_sym = symbols.alloc("n1".to_string());
    let n2_sym = symbols.alloc("n2".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let a_sym = symbols.alloc("a".to_string());
    let b_sym = symbols.alloc("b".to_string());
    let c_sym = symbols.alloc("c".to_string());
    let d_sym = symbols.alloc("d".to_string());

    let pred = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::BoolLit(true), bool_ty.clone(), &mut term_ids),
        bool_ty,
    );
    let producer = mk_filter(
        pred,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let op1 = mk_lambda2(
        a_sym,
        i32_ty(),
        b_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(a_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let op2 = mk_lambda2(
        c_sym,
        i32_ty(),
        d_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(d_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let k_ref = |term_ids: &mut TermIdSource| {
        mk_term(TermKind::Var(VarRef::Symbol(k_sym)), array_ty(i32_ty()), term_ids)
    };
    let r1 = mk_reduce(
        op1,
        mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids),
        k_ref(&mut term_ids),
        i32_ty(),
        &mut term_ids,
    );
    let r2 = mk_reduce(
        op2,
        mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids),
        k_ref(&mut term_ids),
        i32_ty(),
        &mut term_ids,
    );
    let n1 = mk_length(k_ref(&mut term_ids), &mut term_ids);
    let n2 = mk_length(k_ref(&mut term_ids), &mut term_ids);

    let tuple_ty = Type::Constructed(TypeName::Tuple(4), vec![i32_ty(), i32_ty(), i32_ty(), i32_ty()]);
    let tail = mk_term(
        TermKind::Tuple(vec![
            mk_term(TermKind::Var(VarRef::Symbol(r1_sym)), i32_ty(), &mut term_ids),
            mk_term(TermKind::Var(VarRef::Symbol(r2_sym)), i32_ty(), &mut term_ids),
            mk_term(TermKind::Var(VarRef::Symbol(n1_sym)), i32_ty(), &mut term_ids),
            mk_term(TermKind::Var(VarRef::Symbol(n2_sym)), i32_ty(), &mut term_ids),
        ]),
        tuple_ty.clone(),
        &mut term_ids,
    );
    let n2_let = mk_term(
        TermKind::Let {
            name: n2_sym,
            name_ty: i32_ty(),
            rhs: Box::new(n2),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let n1_let = mk_term(
        TermKind::Let {
            name: n1_sym,
            name_ty: i32_ty(),
            rhs: Box::new(n1),
            body: Box::new(n2_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let r2_let = mk_term(
        TermKind::Let {
            name: r2_sym,
            name_ty: i32_ty(),
            rhs: Box::new(r2),
            body: Box::new(n1_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let r1_let = mk_term(
        TermKind::Let {
            name: r1_sym,
            name_ty: i32_ty(),
            rhs: Box::new(r1),
            body: Box::new(r2_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: k_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(r1_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let mut fused = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };
    run(&mut fused);

    match find_first_screma(&fused.defs[0].body).expect("expected filtered Screma") {
        SoacOp::Screma { accumulators, .. } => assert_eq!(accumulators.len(), 3),
        other => panic!("expected Screma, got {other:?}"),
    }
    assert_eq!(
        projection_idx_for_binding(&fused.defs[0].body, n1_sym),
        projection_idx_for_binding(&fused.defs[0].body, n2_sym),
        "all length calls should project the shared count accumulator"
    );
}

fn contains_length(term: &Term) -> bool {
    match &term.kind {
        TermKind::Var(VarRef::Builtin { id, .. }) if *id == crate::builtins::catalog().known().length => {
            true
        }
        _ => {
            let mut found = false;
            term.for_each_child(&mut |child| {
                if !found && contains_length(child) {
                    found = true;
                }
            });
            found
        }
    }
}

// Risk 1 (field-ordering): the shared `length` count is always the LAST
// accumulator field, even when the `length` binding precedes the `reduce` in
// source order. The filter builder collects reductions and lengths separately
// and emits reductions before the count, so `lower_fused_screma` assigns the
// reduction field 0 and the count field 1 regardless of `uses` order. If a
// future refactor folded these into one source-order loop, a length-first
// program would shift the count field and mis-route every reduction projection.
#[test]
fn test_filter_count_field_is_last_when_length_precedes_reduce() {
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let k_sym = symbols.alloc("k".to_string());
    let n_sym = symbols.alloc("n".to_string()); // length binding — comes FIRST
    let r_sym = symbols.alloc("r".to_string()); // reduce binding — comes SECOND
    let x_sym = symbols.alloc("x".to_string());
    let a_sym = symbols.alloc("a".to_string());
    let b_sym = symbols.alloc("b".to_string());

    let pred = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::BoolLit(true), bool_ty.clone(), &mut term_ids),
        bool_ty,
    );
    let producer = mk_filter(
        pred,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let op = mk_lambda2(
        a_sym,
        i32_ty(),
        b_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(a_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let k_ref = |term_ids: &mut TermIdSource| {
        mk_term(TermKind::Var(VarRef::Symbol(k_sym)), array_ty(i32_ty()), term_ids)
    };
    let len = mk_length(k_ref(&mut term_ids), &mut term_ids);
    let reduce = mk_reduce(
        op,
        mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids),
        k_ref(&mut term_ids),
        i32_ty(),
        &mut term_ids,
    );
    let tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), i32_ty()]);
    let tail = mk_term(
        TermKind::Tuple(vec![
            mk_term(TermKind::Var(VarRef::Symbol(r_sym)), i32_ty(), &mut term_ids),
            mk_term(TermKind::Var(VarRef::Symbol(n_sym)), i32_ty(), &mut term_ids),
        ]),
        tuple_ty.clone(),
        &mut term_ids,
    );
    let r_let = mk_term(
        TermKind::Let {
            name: r_sym,
            name_ty: i32_ty(),
            rhs: Box::new(reduce),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let n_let = mk_term(
        TermKind::Let {
            name: n_sym,
            name_ty: i32_ty(),
            rhs: Box::new(len),
            body: Box::new(r_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: k_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(n_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let mut fused = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };
    run(&mut fused);

    assert!(!contains_filter(&fused.defs[0].body));
    match find_first_screma(&fused.defs[0].body).expect("expected filtered Screma") {
        SoacOp::Screma { accumulators, .. } => assert_eq!(accumulators.len(), 2),
        other => panic!("expected Screma, got {other:?}"),
    }
    // The reduction is field 0, the count field 1 — *despite* the length binding
    // appearing before the reduce binding in source order.
    assert_eq!(projection_idx_for_binding(&fused.defs[0].body, r_sym), Some(0));
    assert_eq!(projection_idx_for_binding(&fused.defs[0].body, n_sym), Some(1));
}

// Risk 2 (consumer routing): a `length` that is the whole entry tail is routed
// through the `term_replacements` (NestedTerm) sink, not `tail_projection` —
// `replace_projection_terms` rewrites it in place. Exercises the term-
// replacement path for a tail-owned use, which the binding-owned length tests
// do not cover.
#[test]
fn test_filter_whole_tail_length_rewrites_in_place() {
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let k_sym = symbols.alloc("k".to_string());
    let r_sym = symbols.alloc("r".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let a_sym = symbols.alloc("a".to_string());
    let b_sym = symbols.alloc("b".to_string());

    let pred = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::BoolLit(true), bool_ty.clone(), &mut term_ids),
        bool_ty,
    );
    let producer = mk_filter(
        pred,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let op = mk_lambda2(
        a_sym,
        i32_ty(),
        b_sym,
        i32_ty(),
        mk_term(TermKind::Var(VarRef::Symbol(a_sym)), i32_ty(), &mut term_ids),
        i32_ty(),
    );
    let k_ref = |term_ids: &mut TermIdSource| {
        mk_term(TermKind::Var(VarRef::Symbol(k_sym)), array_ty(i32_ty()), term_ids)
    };
    // A reduce keeps the result a 2-output tuple (not a scalar-reduce Screma),
    // so the whole-tail length projects field 1 via a real TupleProj.
    let reduce = mk_reduce(
        op,
        mk_term(TermKind::IntLit("0".to_string()), i32_ty(), &mut term_ids),
        k_ref(&mut term_ids),
        i32_ty(),
        &mut term_ids,
    );
    // The entry tail IS `length(k)`.
    let tail = mk_length(k_ref(&mut term_ids), &mut term_ids);
    let r_let = mk_term(
        TermKind::Let {
            name: r_sym,
            name_ty: i32_ty(),
            rhs: Box::new(reduce),
            body: Box::new(tail),
        },
        i32_ty(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: k_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(r_let),
        },
        i32_ty(),
        &mut term_ids,
    );

    let mut fused = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: i32_ty(),
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };
    run(&mut fused);

    assert!(!contains_filter(&fused.defs[0].body));
    // Both the reduce and the whole-tail length folded in: the filter and the
    // `length` builtin are gone, replaced by a 2-accumulator Screma + projection.
    assert!(
        !contains_length(&fused.defs[0].body),
        "whole-tail length must be rewritten to a projection"
    );
    match find_first_screma(&fused.defs[0].body).expect("expected filtered Screma") {
        SoacOp::Screma {
            lanes, accumulators, ..
        } => {
            assert!(lanes.is_empty());
            assert_eq!(accumulators.len(), 2);
        }
        other => panic!("expected Screma, got {other:?}"),
    }
    assert_eq!(projection_idx_for_binding(&fused.defs[0].body, r_sym), Some(0));
}

#[test]
fn test_filter_into_length_only_fuses_to_count_screma() {
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let k_sym = symbols.alloc("k".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let pred = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::BoolLit(true), bool_ty.clone(), &mut term_ids),
        bool_ty,
    );
    let producer = mk_filter(
        pred,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let len = mk_length(
        mk_term(
            TermKind::Var(VarRef::Symbol(k_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: k_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(len),
        },
        i32_ty(),
        &mut term_ids,
    );

    let mut fused = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: i32_ty(),
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };
    run(&mut fused);

    assert!(!contains_filter(&fused.defs[0].body));
    match find_first_screma(&fused.defs[0].body).expect("expected count Screma") {
        SoacOp::Screma {
            lanes, accumulators, ..
        } => {
            assert!(lanes.is_empty());
            assert_eq!(accumulators.len(), 1);
        }
        other => panic!("expected Screma, got {other:?}"),
    }
}

#[test]
fn test_filter_escape_blocks_scalar_fusion() {
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let k_sym = symbols.alloc("k".to_string());
    let n_sym = symbols.alloc("n".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let pred = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::BoolLit(true), bool_ty.clone(), &mut term_ids),
        bool_ty,
    );
    let producer = mk_filter(
        pred,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let len = mk_length(
        mk_term(
            TermKind::Var(VarRef::Symbol(k_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        &mut term_ids,
    );
    let tuple_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), array_ty(i32_ty())]);
    let tail = mk_term(
        TermKind::Tuple(vec![
            mk_term(TermKind::Var(VarRef::Symbol(n_sym)), i32_ty(), &mut term_ids),
            mk_term(
                TermKind::Var(VarRef::Symbol(k_sym)),
                array_ty(i32_ty()),
                &mut term_ids,
            ),
        ]),
        tuple_ty.clone(),
        &mut term_ids,
    );
    let n_let = mk_term(
        TermKind::Let {
            name: n_sym,
            name_ty: i32_ty(),
            rhs: Box::new(len),
            body: Box::new(tail),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: k_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(n_let),
        },
        tuple_ty.clone(),
        &mut term_ids,
    );

    let mut fused = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: tuple_ty,
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };
    run(&mut fused);

    assert!(contains_filter(&fused.defs[0].body));
    assert!(!contains_screma(&fused.defs[0].body));
}

#[test]
fn test_shadowed_filter_symbol_inside_lambda_is_not_an_escape() {
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let xs_sym = symbols.alloc("xs".to_string());
    let k_sym = symbols.alloc("k".to_string());
    let shadow_sym = symbols.alloc("shadow".to_string());
    let x_sym = symbols.alloc("x".to_string());
    let pred = mk_lambda1(
        x_sym,
        i32_ty(),
        mk_term(TermKind::BoolLit(true), bool_ty.clone(), &mut term_ids),
        bool_ty,
    );
    let producer = mk_filter(
        pred,
        mk_term(
            TermKind::Var(VarRef::Symbol(xs_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let shadow_lam_ty = Type::Constructed(TypeName::Arrow, vec![i32_ty(), i32_ty()]);
    let shadow_lam = mk_term(
        TermKind::Lambda(Lambda {
            params: vec![(k_sym, i32_ty())],
            body: Box::new(mk_term(
                TermKind::Var(VarRef::Symbol(k_sym)),
                i32_ty(),
                &mut term_ids,
            )),
            ret_ty: i32_ty(),
        }),
        shadow_lam_ty.clone(),
        &mut term_ids,
    );
    let len = mk_length(
        mk_term(
            TermKind::Var(VarRef::Symbol(k_sym)),
            array_ty(i32_ty()),
            &mut term_ids,
        ),
        &mut term_ids,
    );
    let shadow_let = mk_term(
        TermKind::Let {
            name: shadow_sym,
            name_ty: shadow_lam_ty,
            rhs: Box::new(shadow_lam),
            body: Box::new(len),
        },
        i32_ty(),
        &mut term_ids,
    );
    let program_body = mk_term(
        TermKind::Let {
            name: k_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(shadow_let),
        },
        i32_ty(),
        &mut term_ids,
    );

    let mut fused = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: i32_ty(),
            body: program_body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };
    run(&mut fused);

    assert!(!contains_filter(&fused.defs[0].body));
    assert!(contains_screma(&fused.defs[0].body));
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
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);

    // All three maps collapse to one SOAC region reading the original input `a`.
    let body = &fused.defs[0].body;
    assert_eq!(
        count_soacs(body),
        1,
        "all three chained maps must fuse to one region"
    );
    assert_eq!(
        soac_input_syms(&fused_soac(body)),
        vec![a_sym],
        "the fused region reads the original input a"
    );
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
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut result = program;
    run(&mut result);

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
            inputs: vec![input_ae(Box::new(a)), input_ae(Box::new(b))],
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
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);

    // map over a zip fuses to one region that preserves both source inputs.
    let body = &fused.defs[0].body;
    assert_eq!(count_soacs(body), 1, "map over a zip must fuse to one region");
    assert_eq!(
        soac_input_syms(&fused_soac(body)).len(),
        2,
        "the fused region preserves both zip inputs"
    );
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
            inputs: vec![input_ae(Box::new(b_ref)), input_ae(Box::new(other))],
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
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut result = program;
    run(&mut result);

    // Should NOT fuse — consumer has multiple inputs
    assert!(matches!(&result.defs[0].body.kind, TermKind::Let { .. }));
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
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);
    let (op, inputs) = as_fused_map_reduce(&fused.defs[0].body);
    assert_eq!(inputs.len(), 1);
    match &inputs[0] {
        ArrayExpr::Var(vr, _) => {
            assert!(matches!(vr, VarRef::Symbol(s) if *s == xs_sym))
        }
        other => panic!("Expected Ref(xs), got {:?}", other),
    }
    // step has (acc, x) params — acc from reduce, x from map
    assert_eq!(op.lam.params.len(), 2);
    assert_eq!(op.lam.params[0].0, acc_sym);
}

// -------------------------------------------------------------------------
// Test: let idxs = map(f, pts) in scatter(fb, idxs, vals)
//   → scatter(fb, pts, vals), f composed into the envelope's index slot.
// -------------------------------------------------------------------------
#[test]
fn test_map_into_scatter_fuses() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let pts_sym = symbols.alloc("pts".to_string());
    let idxs_sym = symbols.alloc("idxs".to_string());
    let vals_sym = symbols.alloc("vals".to_string());
    let fb_sym = symbols.alloc("fb".to_string());
    let x_sym = symbols.alloc("x".to_string()); // f's param
    let i_sym = symbols.alloc("i".to_string()); // envelope index param
    let v_sym = symbols.alloc("v".to_string()); // envelope value param

    // f: i32 → i32 (returns x); producer map(f, pts).
    let f_body = mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids);
    let f = mk_lambda1(x_sym, i32_ty(), f_body, i32_ty());
    let pts = mk_term(
        TermKind::Var(VarRef::Symbol(pts_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let producer = mk_map(f, pts, array_ty(i32_ty()), &mut term_ids);

    // Identity envelope λ(i, v) → (i, v).
    let i_var = mk_term(TermKind::Var(VarRef::Symbol(i_sym)), i32_ty(), &mut term_ids);
    let v_var = mk_term(TermKind::Var(VarRef::Symbol(v_sym)), i32_ty(), &mut term_ids);
    let tup_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), i32_ty()]);
    let env_body = mk_term(TermKind::Tuple(vec![i_var, v_var]), tup_ty.clone(), &mut term_ids);
    let env = mk_lambda2(i_sym, i32_ty(), v_sym, i32_ty(), env_body, tup_ty);

    // scatter(fb, idxs, vals).
    let idxs_ref = mk_term(
        TermKind::Var(VarRef::Symbol(idxs_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let vals_ref = mk_term(
        TermKind::Var(VarRef::Symbol(vals_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let scatter = mk_term(
        TermKind::Soac(SoacOp::Scatter {
            dest: crate::tlc::Place::LocalArray {
                id: fb_sym,
                shape: crate::tlc::Shape(vec![]),
                elem_ty: i32_ty(),
            },
            lam: mk_soac_body(env),
            inputs: vec![input_ae(Box::new(idxs_ref)), input_ae(Box::new(vals_ref))],
        }),
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let body = mk_term(
        TermKind::Let {
            name: idxs_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(producer),
            body: Box::new(scatter),
        },
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: array_ty(i32_ty()),
            body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);

    match &fused.defs[0].body.kind {
        TermKind::Soac(SoacOp::Scatter { lam, inputs, .. }) => {
            // Slot 0 input is now `pts`; slot 1 stays `vals`.
            assert_eq!(inputs.len(), 2);
            assert!(
                matches!(&inputs[0], ArrayExpr::Var(VarRef::Symbol(s), _) if *s == pts_sym),
                "index slot input should be pts, got {:?}",
                inputs[0]
            );
            assert!(
                matches!(&inputs[1], ArrayExpr::Var(VarRef::Symbol(s), _) if *s == vals_sym),
                "value slot input should stay vals, got {:?}",
                inputs[1]
            );

            // f's param replaces the index slot; the value param stays.
            assert_eq!(lam.lam.params.len(), 2);
            assert_eq!(lam.lam.params[0].0, x_sym);
            assert_eq!(lam.lam.params[1].0, v_sym);

            // Body: let _fused = x in (_fused, v).
            match &lam.lam.body.kind {
                TermKind::Let { rhs, body, .. } => {
                    assert!(matches!(&rhs.kind, TermKind::Var(VarRef::Symbol(s)) if *s == x_sym));
                    match &body.kind {
                        TermKind::Tuple(parts) => {
                            assert_eq!(parts.len(), 2);
                            assert!(
                                matches!(&parts[1].kind, TermKind::Var(VarRef::Symbol(s)) if *s == v_sym),
                                "tuple value slot should still be v"
                            );
                        }
                        other => panic!("expected tuple body, got {other:?}"),
                    }
                }
                other => panic!("expected composed Let body, got {other:?}"),
            }
        }
        other => panic!("expected a single fused Scatter, got {:?}", other),
    }
}

// -------------------------------------------------------------------------
// Test (particles shape): both the index and value arrays are maps over the
// same base. Both compose into the envelope, and the shared base is deduped to
// a single input slot — the scatter reads `pts` once, computing both channels.
// -------------------------------------------------------------------------
#[test]
fn test_map_scatter_both_producers_fuse() {
    let mut symbols = SymbolTable::default();
    let mut term_ids = TermIdSource::new();

    let pts_sym = symbols.alloc("pts".to_string());
    let idxs_sym = symbols.alloc("idxs".to_string());
    let vals_sym = symbols.alloc("vals".to_string());
    let fb_sym = symbols.alloc("fb".to_string());
    let x_sym = symbols.alloc("x".to_string()); // f's param (index producer)
    let y_sym = symbols.alloc("y".to_string()); // g's param (value producer)
    let i_sym = symbols.alloc("i".to_string());
    let v_sym = symbols.alloc("v".to_string());

    let mk_pts = |term_ids: &mut TermIdSource| {
        mk_term(
            TermKind::Var(VarRef::Symbol(pts_sym)),
            array_ty(i32_ty()),
            term_ids,
        )
    };

    // idxs = map(f, pts), vals = map(g, pts).
    let f_body = mk_term(TermKind::Var(VarRef::Symbol(x_sym)), i32_ty(), &mut term_ids);
    let f = mk_lambda1(x_sym, i32_ty(), f_body, i32_ty());
    let idxs_producer = mk_map(f, mk_pts(&mut term_ids), array_ty(i32_ty()), &mut term_ids);

    let g_body = mk_term(TermKind::Var(VarRef::Symbol(y_sym)), i32_ty(), &mut term_ids);
    let g = mk_lambda1(y_sym, i32_ty(), g_body, i32_ty());
    let vals_producer = mk_map(g, mk_pts(&mut term_ids), array_ty(i32_ty()), &mut term_ids);

    // Identity envelope λ(i, v) → (i, v).
    let i_var = mk_term(TermKind::Var(VarRef::Symbol(i_sym)), i32_ty(), &mut term_ids);
    let v_var = mk_term(TermKind::Var(VarRef::Symbol(v_sym)), i32_ty(), &mut term_ids);
    let tup_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), i32_ty()]);
    let env_body = mk_term(TermKind::Tuple(vec![i_var, v_var]), tup_ty.clone(), &mut term_ids);
    let env = mk_lambda2(i_sym, i32_ty(), v_sym, i32_ty(), env_body, tup_ty);

    let idxs_ref = mk_term(
        TermKind::Var(VarRef::Symbol(idxs_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let vals_ref = mk_term(
        TermKind::Var(VarRef::Symbol(vals_sym)),
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let scatter = mk_term(
        TermKind::Soac(SoacOp::Scatter {
            dest: crate::tlc::Place::LocalArray {
                id: fb_sym,
                shape: crate::tlc::Shape(vec![]),
                elem_ty: i32_ty(),
            },
            lam: mk_soac_body(env),
            inputs: vec![input_ae(Box::new(idxs_ref)), input_ae(Box::new(vals_ref))],
        }),
        array_ty(i32_ty()),
        &mut term_ids,
    );

    // let idxs = map(f, pts) in let vals = map(g, pts) in scatter(fb, idxs, vals)
    let inner_let = mk_term(
        TermKind::Let {
            name: vals_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(vals_producer),
            body: Box::new(scatter),
        },
        array_ty(i32_ty()),
        &mut term_ids,
    );
    let body = mk_term(
        TermKind::Let {
            name: idxs_sym,
            name_ty: array_ty(i32_ty()),
            rhs: Box::new(idxs_producer),
            body: Box::new(inner_let),
        },
        array_ty(i32_ty()),
        &mut term_ids,
    );

    let program = Program {
        defs: vec![Def {
            name: symbols.alloc("main".to_string()),
            ty: array_ty(i32_ty()),
            body,
            meta: DefMeta::Function,
            arity: 0,
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut fused = program;
    run(&mut fused);

    match &fused.defs[0].body.kind {
        TermKind::Soac(SoacOp::Scatter { lam, inputs, .. }) => {
            // Both producers fused away (no `idxs`/`vals` Let bindings), and the
            // shared base is deduped to a SINGLE input slot — pts read once.
            assert_eq!(inputs.len(), 1, "duplicate pts slots collapse to one");
            assert!(
                matches!(&inputs[0], ArrayExpr::Var(VarRef::Symbol(s), _) if *s == pts_sym),
                "the one input slot reads pts, got {:?}",
                inputs[0]
            );
            // One param feeds both channels; g's param was rewritten to f's by
            // the dedup, so x_sym is kept.
            assert_eq!(lam.lam.params.len(), 1, "one param after dedup");
            assert_eq!(lam.lam.params[0].0, x_sym);
            // Two composed Lets (index then value) nest around the result tuple.
            match &lam.lam.body.kind {
                TermKind::Let { body: outer_body, .. } => {
                    assert!(
                        matches!(&outer_body.kind, TermKind::Let { .. }),
                        "expected two nested composed Lets"
                    );
                }
                other => panic!("expected nested composed Lets, got {other:?}"),
            }
        }
        other => panic!("expected a single fused Scatter, got {:?}", other),
    }
}
