//! Tests for TLC partial evaluation.

use super::partial_eval::PartialEvaluator;
use super::{Def, DefMeta, FunctionName, Program, Term, TermIdSource, TermKind};
use crate::ast::{BinaryOp, Span, TypeName};
use polytype::Type;

fn make_span() -> Span {
    Span {
        start_line: 1,
        start_col: 1,
        end_line: 1,
        end_col: 1,
    }
}

fn make_program(name: &str, body: Term) -> Program {
    Program {
        defs: vec![Def {
            name: name.to_string(),
            ty: body.ty.clone(),
            body,
            meta: DefMeta::Function,
            arity: 0,
        }],
        uniforms: vec![],
        storage: vec![],
    }
}

fn make_int(ids: &mut TermIdSource, n: i64) -> Term {
    Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: make_span(),
        kind: TermKind::IntLit(n.to_string()),
    }
}

fn make_bool(ids: &mut TermIdSource, b: bool) -> Term {
    Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Str("bool"), vec![]),
        span: make_span(),
        kind: TermKind::BoolLit(b),
    }
}

fn make_binop(ids: &mut TermIdSource, op: &str, lhs: Term, rhs: Term) -> Term {
    let partial = Term {
        id: ids.next_id(),
        ty: lhs.ty.clone(),
        span: make_span(),
        kind: TermKind::App {
            func: Box::new(FunctionName::BinOp(BinaryOp { op: op.to_string() })),
            arg: Box::new(lhs),
        },
    };
    Term {
        id: ids.next_id(),
        ty: partial.ty.clone(),
        span: make_span(),
        kind: TermKind::App {
            func: Box::new(FunctionName::Term(Box::new(partial))),
            arg: Box::new(rhs),
        },
    }
}

#[test]
fn test_constant_folding_add() {
    let mut ids = TermIdSource::new();
    let lhs = make_int(&mut ids, 2);
    let rhs = make_int(&mut ids, 3);
    let term = make_binop(&mut ids, "+", lhs, rhs);

    let program = make_program("test", term);

    let result = PartialEvaluator::partial_eval(program);
    assert_eq!(result.defs.len(), 1);

    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "5"),
        other => panic!("Expected IntLit(5), got {:?}", other),
    }
}

#[test]
fn test_constant_folding_mul() {
    let mut ids = TermIdSource::new();
    let lhs = make_int(&mut ids, 4);
    let rhs = make_int(&mut ids, 7);
    let term = make_binop(&mut ids, "*", lhs, rhs);

    let program = make_program("test", term);

    let result = PartialEvaluator::partial_eval(program);

    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "28"),
        other => panic!("Expected IntLit(28), got {:?}", other),
    }
}

#[test]
fn test_algebraic_add_zero() {
    let mut ids = TermIdSource::new();
    let x = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: make_span(),
        kind: TermKind::Var("x".to_string()),
    };
    let zero = make_int(&mut ids, 0);
    let term = make_binop(&mut ids, "+", x, zero);

    let program = make_program("test", term);

    let result = PartialEvaluator::partial_eval(program);

    // x + 0 should simplify to just x
    match &result.defs[0].body.kind {
        TermKind::Var(name) => assert_eq!(name, "x"),
        other => panic!("Expected Var(x), got {:?}", other),
    }
}

#[test]
fn test_algebraic_mul_one() {
    let mut ids = TermIdSource::new();
    let one = make_int(&mut ids, 1);
    let x = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: make_span(),
        kind: TermKind::Var("x".to_string()),
    };
    let term = make_binop(&mut ids, "*", one, x);

    let program = make_program("test", term);

    let result = PartialEvaluator::partial_eval(program);

    // 1 * x should simplify to just x
    match &result.defs[0].body.kind {
        TermKind::Var(name) => assert_eq!(name, "x"),
        other => panic!("Expected Var(x), got {:?}", other),
    }
}

#[test]
fn test_algebraic_mul_zero() {
    let mut ids = TermIdSource::new();
    let x = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: make_span(),
        kind: TermKind::Var("x".to_string()),
    };
    let zero = make_int(&mut ids, 0);
    let term = make_binop(&mut ids, "*", x, zero);

    let program = make_program("test", term);

    let result = PartialEvaluator::partial_eval(program);

    // x * 0 should simplify to 0
    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "0"),
        other => panic!("Expected IntLit(0), got {:?}", other),
    }
}

#[test]
fn test_if_true_elimination() {
    let mut ids = TermIdSource::new();
    let cond = make_bool(&mut ids, true);
    let then_branch = make_int(&mut ids, 1);
    let else_branch = make_int(&mut ids, 2);
    let term = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: make_span(),
        kind: TermKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
    };

    let program = make_program("test", term);

    let result = PartialEvaluator::partial_eval(program);

    // if true then 1 else 2 should simplify to 1
    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "1"),
        other => panic!("Expected IntLit(1), got {:?}", other),
    }
}

#[test]
fn test_if_false_elimination() {
    let mut ids = TermIdSource::new();
    let cond = make_bool(&mut ids, false);
    let then_branch = make_int(&mut ids, 1);
    let else_branch = make_int(&mut ids, 2);
    let term = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: make_span(),
        kind: TermKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
    };

    let program = make_program("test", term);

    let result = PartialEvaluator::partial_eval(program);

    // if false then 1 else 2 should simplify to 2
    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "2"),
        other => panic!("Expected IntLit(2), got {:?}", other),
    }
}

#[test]
fn test_let_constant_propagation() {
    let mut ids = TermIdSource::new();
    // let x = 5 in x + 3
    let rhs = make_int(&mut ids, 5);
    let x_var = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: make_span(),
        kind: TermKind::Var("x".to_string()),
    };
    let three = make_int(&mut ids, 3);
    let body_expr = make_binop(&mut ids, "+", x_var, three);
    let term = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: make_span(),
        kind: TermKind::Let {
            name: "x".to_string(),
            name_ty: Type::Constructed(TypeName::Int(32), vec![]),
            rhs: Box::new(rhs),
            body: Box::new(body_expr),
        },
    };

    let program = make_program("test", term);

    let result = PartialEvaluator::partial_eval(program);

    // let x = 5 in x + 3 should simplify to 8
    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "8"),
        other => panic!("Expected IntLit(8), got {:?}", other),
    }
}

#[test]
fn test_function_inlining() {
    // def foo(a, b) = a + b
    // def bar = foo(8, 9)
    // bar should evaluate to 17
    let mut ids = TermIdSource::new();
    let int_ty = Type::Constructed(TypeName::Int(32), vec![]);

    // Build foo: |a| |b| a + b
    let a_var = Term {
        id: ids.next_id(),
        ty: int_ty.clone(),
        span: make_span(),
        kind: TermKind::Var("a".to_string()),
    };
    let b_var = Term {
        id: ids.next_id(),
        ty: int_ty.clone(),
        span: make_span(),
        kind: TermKind::Var("b".to_string()),
    };
    let a_plus_b = make_binop(&mut ids, "+", a_var, b_var);

    let inner_lam = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![int_ty.clone(), int_ty.clone()]),
        span: make_span(),
        kind: TermKind::Lam {
            param: "b".to_string(),
            param_ty: int_ty.clone(),
            body: Box::new(a_plus_b),
        },
    };

    let foo_body = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![int_ty.clone(), inner_lam.ty.clone()]),
        span: make_span(),
        kind: TermKind::Lam {
            param: "a".to_string(),
            param_ty: int_ty.clone(),
            body: Box::new(inner_lam),
        },
    };

    // Build bar: foo 8 9
    let eight = make_int(&mut ids, 8);
    let nine = make_int(&mut ids, 9);

    let foo_ref = Term {
        id: ids.next_id(),
        ty: foo_body.ty.clone(),
        span: make_span(),
        kind: TermKind::Var("foo".to_string()),
    };

    let foo_8 = Term {
        id: ids.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![int_ty.clone(), int_ty.clone()]),
        span: make_span(),
        kind: TermKind::App {
            func: Box::new(FunctionName::Term(Box::new(foo_ref))),
            arg: Box::new(eight),
        },
    };

    let bar_body = Term {
        id: ids.next_id(),
        ty: int_ty.clone(),
        span: make_span(),
        kind: TermKind::App {
            func: Box::new(FunctionName::Term(Box::new(foo_8))),
            arg: Box::new(nine),
        },
    };

    let program = Program {
        defs: vec![
            Def {
                name: "foo".to_string(),
                ty: foo_body.ty.clone(),
                body: foo_body,
                meta: DefMeta::Function,
                arity: 2,
            },
            Def {
                name: "bar".to_string(),
                ty: int_ty.clone(),
                body: bar_body,
                meta: DefMeta::Function,
                arity: 0,
            },
        ],
        uniforms: vec![],
        storage: vec![],
    };

    let result = PartialEvaluator::partial_eval(program);

    // bar should be inlined and folded to 17
    let bar_def = result.defs.iter().find(|d| d.name == "bar").unwrap();
    match &bar_def.body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "17"),
        other => panic!("Expected IntLit(17), got {:?}", other),
    }
}

fn int_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn arrow_ty(from: Type<TypeName>, to: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(TypeName::Arrow, vec![from, to])
}

/// Test that `let f = g in f x` is inlined to `g x`
#[test]
fn test_function_alias_inlining() {
    let mut ids = TermIdSource::new();
    let span = make_span();

    // Build: def g = |y| y  (identity function)
    let g_body = Term {
        id: ids.next_id(),
        ty: arrow_ty(int_ty(), int_ty()),
        span,
        kind: TermKind::Lam {
            param: "y".to_string(),
            param_ty: int_ty(),
            body: Box::new(Term {
                id: ids.next_id(),
                ty: int_ty(),
                span,
                kind: TermKind::Var("y".to_string()),
            }),
        },
    };

    // Build: def main = let f = g in f 42
    // This is: Let { name: "f", rhs: Var("g"), body: App(Var("f"), 42) }
    let main_body = Term {
        id: ids.next_id(),
        ty: int_ty(),
        span,
        kind: TermKind::Let {
            name: "f".to_string(),
            name_ty: arrow_ty(int_ty(), int_ty()),
            rhs: Box::new(Term {
                id: ids.next_id(),
                ty: arrow_ty(int_ty(), int_ty()),
                span,
                kind: TermKind::Var("g".to_string()),
            }),
            body: Box::new(Term {
                id: ids.next_id(),
                ty: int_ty(),
                span,
                kind: TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(Term {
                        id: ids.next_id(),
                        ty: arrow_ty(int_ty(), int_ty()),
                        span,
                        kind: TermKind::Var("f".to_string()),
                    }))),
                    arg: Box::new(Term {
                        id: ids.next_id(),
                        ty: int_ty(),
                        span,
                        kind: TermKind::IntLit("42".to_string()),
                    }),
                },
            }),
        },
    };

    let program = Program {
        defs: vec![
            Def {
                name: "g".to_string(),
                ty: g_body.ty.clone(),
                body: g_body,
                meta: DefMeta::Function,
                arity: 1,
            },
            Def {
                name: "main".to_string(),
                ty: int_ty(),
                body: main_body,
                meta: DefMeta::Function,
                arity: 0,
            },
        ],
        uniforms: vec![],
        storage: vec![],
    };

    let result = PartialEvaluator::partial_eval(program);

    // Find main's body - it should be simplified to just `42`
    // because g is identity and f aliases g, so f 42 = g 42 = 42
    let main_def = result.defs.iter().find(|d| d.name == "main").unwrap();

    // The result should be IntLit("42") since g is identity
    match &main_def.body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "42"),
        other => panic!("Expected IntLit(42), got {:?}", other),
    }
}

/// Test that function alias without full application still uses correct name
#[test]
fn test_function_alias_partial_application() {
    let mut ids = TermIdSource::new();
    let span = make_span();

    // Build: def g = |x| |y| x  (const function, arity 2)
    let g_body = Term {
        id: ids.next_id(),
        ty: arrow_ty(int_ty(), arrow_ty(int_ty(), int_ty())),
        span,
        kind: TermKind::Lam {
            param: "x".to_string(),
            param_ty: int_ty(),
            body: Box::new(Term {
                id: ids.next_id(),
                ty: arrow_ty(int_ty(), int_ty()),
                span,
                kind: TermKind::Lam {
                    param: "y".to_string(),
                    param_ty: int_ty(),
                    body: Box::new(Term {
                        id: ids.next_id(),
                        ty: int_ty(),
                        span,
                        kind: TermKind::Var("x".to_string()),
                    }),
                },
            }),
        },
    };

    // Build: def main = let f = g in f 1 2
    // f aliases g, so f 1 2 should become g 1 2 = 1
    let main_body = Term {
        id: ids.next_id(),
        ty: int_ty(),
        span,
        kind: TermKind::Let {
            name: "f".to_string(),
            name_ty: arrow_ty(int_ty(), arrow_ty(int_ty(), int_ty())),
            rhs: Box::new(Term {
                id: ids.next_id(),
                ty: arrow_ty(int_ty(), arrow_ty(int_ty(), int_ty())),
                span,
                kind: TermKind::Var("g".to_string()),
            }),
            body: Box::new(Term {
                id: ids.next_id(),
                ty: int_ty(),
                span,
                kind: TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(Term {
                        id: ids.next_id(),
                        ty: arrow_ty(int_ty(), int_ty()),
                        span,
                        kind: TermKind::App {
                            func: Box::new(FunctionName::Term(Box::new(Term {
                                id: ids.next_id(),
                                ty: arrow_ty(int_ty(), arrow_ty(int_ty(), int_ty())),
                                span,
                                kind: TermKind::Var("f".to_string()),
                            }))),
                            arg: Box::new(Term {
                                id: ids.next_id(),
                                ty: int_ty(),
                                span,
                                kind: TermKind::IntLit("1".to_string()),
                            }),
                        },
                    }))),
                    arg: Box::new(Term {
                        id: ids.next_id(),
                        ty: int_ty(),
                        span,
                        kind: TermKind::IntLit("2".to_string()),
                    }),
                },
            }),
        },
    };

    let program = Program {
        defs: vec![
            Def {
                name: "g".to_string(),
                ty: g_body.ty.clone(),
                body: g_body,
                meta: DefMeta::Function,
                arity: 2,
            },
            Def {
                name: "main".to_string(),
                ty: int_ty(),
                body: main_body,
                meta: DefMeta::Function,
                arity: 0,
            },
        ],
        uniforms: vec![],
        storage: vec![],
    };

    let result = PartialEvaluator::partial_eval(program);

    // Find main's body - it should be simplified to `1`
    // because g x y = x, so f 1 2 = g 1 2 = 1
    let main_def = result.defs.iter().find(|d| d.name == "main").unwrap();

    match &main_def.body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "1"),
        other => panic!("Expected IntLit(1), got {:?}", other),
    }
}

/// Test that `let f = f32.sin in f x` becomes `f32.sin x`
#[test]
fn test_intrinsic_alias_inlining() {
    let mut ids = TermIdSource::new();
    let span = make_span();
    let float_ty = Type::Constructed(TypeName::Float(32), vec![]);

    // Build: def main = let f = f32.sin in f 0.5
    // f32.sin is an intrinsic (not in defs), so it evaluates to Unknown(Var("f32.sin"))
    let main_body = Term {
        id: ids.next_id(),
        ty: float_ty.clone(),
        span,
        kind: TermKind::Let {
            name: "f".to_string(),
            name_ty: arrow_ty(float_ty.clone(), float_ty.clone()),
            rhs: Box::new(Term {
                id: ids.next_id(),
                ty: arrow_ty(float_ty.clone(), float_ty.clone()),
                span,
                kind: TermKind::Var("f32.sin".to_string()),
            }),
            body: Box::new(Term {
                id: ids.next_id(),
                ty: float_ty.clone(),
                span,
                kind: TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(Term {
                        id: ids.next_id(),
                        ty: arrow_ty(float_ty.clone(), float_ty.clone()),
                        span,
                        kind: TermKind::Var("f".to_string()),
                    }))),
                    arg: Box::new(Term {
                        id: ids.next_id(),
                        ty: float_ty.clone(),
                        span,
                        kind: TermKind::FloatLit(0.5),
                    }),
                },
            }),
        },
    };

    let program = Program {
        defs: vec![Def {
            name: "main".to_string(),
            ty: float_ty.clone(),
            body: main_body,
            meta: DefMeta::Function,
            arity: 0,
        }],
        uniforms: vec![],
        storage: vec![],
    };

    let result = PartialEvaluator::partial_eval(program);
    let main_def = result.defs.iter().find(|d| d.name == "main").unwrap();

    // The result should be App(f32.sin, 0.5) - the alias `f` should be resolved to `f32.sin`
    match &main_def.body.kind {
        TermKind::App { func, arg } => {
            // Check that the function is now f32.sin (not f)
            match func.as_ref() {
                FunctionName::Term(t) => match &t.kind {
                    TermKind::Var(name) => assert_eq!(name, "f32.sin"),
                    other => panic!("Expected Var(f32.sin), got {:?}", other),
                },
                other => panic!("Expected FunctionName::Term, got {:?}", other),
            }
            // Check the argument is still 0.5
            match &arg.kind {
                TermKind::FloatLit(f) => assert!((*f - 0.5).abs() < 0.001),
                other => panic!("Expected FloatLit(0.5), got {:?}", other),
            }
        }
        other => panic!("Expected App, got {:?}", other),
    }
}
