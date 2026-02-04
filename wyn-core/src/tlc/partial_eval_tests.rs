//! Tests for TLC partial evaluation.

use super::partial_eval::PartialEvaluator;
use super::{Def, DefMeta, Program, Term, TermIdSource, TermKind};
use crate::ast::{BinaryOp, Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;

/// Test helper that manages symbol table and term ID generation.
struct TestBuilder {
    symbols: SymbolTable,
    ids: TermIdSource,
}

impl TestBuilder {
    fn new() -> Self {
        TestBuilder {
            symbols: SymbolTable::new(),
            ids: TermIdSource::new(),
        }
    }

    fn sym(&mut self, name: &str) -> SymbolId {
        self.symbols.alloc(name.to_string())
    }

    fn next_id(&mut self) -> super::TermId {
        self.ids.next_id()
    }

    fn span(&self) -> Span {
        Span::dummy()
    }

    fn finish(self) -> SymbolTable {
        self.symbols
    }
}

fn make_span() -> Span {
    Span::dummy()
}

fn make_program(name_sym: SymbolId, body: Term, symbols: SymbolTable) -> Program {
    Program {
        defs: vec![Def {
            name: name_sym,
            ty: body.ty.clone(),
            body,
            meta: DefMeta::Function,
            arity: 0,
        }],
        uniforms: vec![],
        storage: vec![],
        symbols,
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
    let result_ty = lhs.ty.clone();
    let partial_ty = Type::Constructed(TypeName::Arrow, vec![result_ty.clone(), result_ty.clone()]);
    let binop_ty = Type::Constructed(TypeName::Arrow, vec![result_ty.clone(), partial_ty.clone()]);

    let partial = Term {
        id: ids.next_id(),
        ty: partial_ty,
        span: make_span(),
        kind: TermKind::App {
            func: Box::new(Term {
                id: ids.next_id(),
                ty: binop_ty,
                span: make_span(),
                kind: TermKind::BinOp(BinaryOp { op: op.to_string() }),
            }),
            arg: Box::new(lhs),
        },
    };
    Term {
        id: ids.next_id(),
        ty: result_ty,
        span: make_span(),
        kind: TermKind::App {
            func: Box::new(partial),
            arg: Box::new(rhs),
        },
    }
}

#[test]
fn test_constant_folding_add() {
    let mut b = TestBuilder::new();
    let test_sym = b.sym("test");

    let lhs = make_int(&mut b.ids, 2);
    let rhs = make_int(&mut b.ids, 3);
    let term = make_binop(&mut b.ids, "+", lhs, rhs);

    let program = make_program(test_sym, term, b.finish());

    let result = PartialEvaluator::partial_eval(program);
    assert_eq!(result.defs.len(), 1);

    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "5"),
        other => panic!("Expected IntLit(5), got {:?}", other),
    }
}

#[test]
fn test_constant_folding_mul() {
    let mut b = TestBuilder::new();
    let test_sym = b.sym("test");

    let lhs = make_int(&mut b.ids, 4);
    let rhs = make_int(&mut b.ids, 7);
    let term = make_binop(&mut b.ids, "*", lhs, rhs);

    let program = make_program(test_sym, term, b.finish());

    let result = PartialEvaluator::partial_eval(program);

    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "28"),
        other => panic!("Expected IntLit(28), got {:?}", other),
    }
}

#[test]
fn test_algebraic_add_zero() {
    let mut b = TestBuilder::new();
    let x_sym = b.sym("x");
    let test_sym = b.sym("test");

    let x = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::Var(x_sym),
    };
    let zero = make_int(&mut b.ids, 0);
    let term = make_binop(&mut b.ids, "+", x, zero);

    let program = make_program(test_sym, term, b.finish());

    let result = PartialEvaluator::partial_eval(program);

    // x + 0 should simplify to just x
    match &result.defs[0].body.kind {
        TermKind::Var(sym) => {
            let name = result.symbols.get(*sym).expect("BUG: symbol not in table");
            assert_eq!(name, "x");
        }
        other => panic!("Expected Var(x), got {:?}", other),
    }
}

#[test]
fn test_algebraic_mul_one() {
    let mut b = TestBuilder::new();
    let x_sym = b.sym("x");
    let test_sym = b.sym("test");

    let one = make_int(&mut b.ids, 1);
    let x = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::Var(x_sym),
    };
    let term = make_binop(&mut b.ids, "*", one, x);

    let program = make_program(test_sym, term, b.finish());

    let result = PartialEvaluator::partial_eval(program);

    // 1 * x should simplify to just x
    match &result.defs[0].body.kind {
        TermKind::Var(sym) => {
            let name = result.symbols.get(*sym).expect("BUG: symbol not in table");
            assert_eq!(name, "x");
        }
        other => panic!("Expected Var(x), got {:?}", other),
    }
}

#[test]
fn test_algebraic_mul_zero() {
    let mut b = TestBuilder::new();
    let x_sym = b.sym("x");
    let test_sym = b.sym("test");

    let x = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::Var(x_sym),
    };
    let zero = make_int(&mut b.ids, 0);
    let term = make_binop(&mut b.ids, "*", x, zero);

    let program = make_program(test_sym, term, b.finish());

    let result = PartialEvaluator::partial_eval(program);

    // x * 0 should simplify to 0
    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "0"),
        other => panic!("Expected IntLit(0), got {:?}", other),
    }
}

#[test]
fn test_if_true_elimination() {
    let mut b = TestBuilder::new();
    let test_sym = b.sym("test");

    let cond = make_bool(&mut b.ids, true);
    let then_branch = make_int(&mut b.ids, 1);
    let else_branch = make_int(&mut b.ids, 2);
    let term = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
    };

    let program = make_program(test_sym, term, b.finish());

    let result = PartialEvaluator::partial_eval(program);

    // if true then 1 else 2 should simplify to 1
    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "1"),
        other => panic!("Expected IntLit(1), got {:?}", other),
    }
}

#[test]
fn test_if_false_elimination() {
    let mut b = TestBuilder::new();
    let test_sym = b.sym("test");

    let cond = make_bool(&mut b.ids, false);
    let then_branch = make_int(&mut b.ids, 1);
    let else_branch = make_int(&mut b.ids, 2);
    let term = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
    };

    let program = make_program(test_sym, term, b.finish());

    let result = PartialEvaluator::partial_eval(program);

    // if false then 1 else 2 should simplify to 2
    match &result.defs[0].body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "2"),
        other => panic!("Expected IntLit(2), got {:?}", other),
    }
}

#[test]
fn test_let_constant_propagation() {
    let mut b = TestBuilder::new();
    let x_sym = b.sym("x");
    let test_sym = b.sym("test");

    // let x = 5 in x + 3
    let rhs = make_int(&mut b.ids, 5);
    let x_var = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::Var(x_sym),
    };
    let three = make_int(&mut b.ids, 3);
    let body_expr = make_binop(&mut b.ids, "+", x_var, three);
    let term = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Int(32), vec![]),
        span: b.span(),
        kind: TermKind::Let {
            name: x_sym,
            name_ty: Type::Constructed(TypeName::Int(32), vec![]),
            rhs: Box::new(rhs),
            body: Box::new(body_expr),
        },
    };

    let program = make_program(test_sym, term, b.finish());

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
    let mut b = TestBuilder::new();
    let int_ty = Type::Constructed(TypeName::Int(32), vec![]);

    let a_sym = b.sym("a");
    let b_sym = b.sym("b");
    let foo_sym = b.sym("foo");
    let bar_sym = b.sym("bar");

    // Build foo: |a| |b| a + b
    let a_var = Term {
        id: b.next_id(),
        ty: int_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(a_sym),
    };
    let b_var = Term {
        id: b.next_id(),
        ty: int_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(b_sym),
    };
    let a_plus_b = make_binop(&mut b.ids, "+", a_var, b_var);

    let inner_lam = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![int_ty.clone(), int_ty.clone()]),
        span: b.span(),
        kind: TermKind::Lam {
            param: b_sym,
            param_ty: int_ty.clone(),
            body: Box::new(a_plus_b),
        },
    };

    let foo_body = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![int_ty.clone(), inner_lam.ty.clone()]),
        span: b.span(),
        kind: TermKind::Lam {
            param: a_sym,
            param_ty: int_ty.clone(),
            body: Box::new(inner_lam),
        },
    };

    // Build bar: foo 8 9
    let eight = make_int(&mut b.ids, 8);
    let nine = make_int(&mut b.ids, 9);

    let foo_ref = Term {
        id: b.next_id(),
        ty: foo_body.ty.clone(),
        span: b.span(),
        kind: TermKind::Var(foo_sym),
    };

    let foo_8 = Term {
        id: b.next_id(),
        ty: Type::Constructed(TypeName::Arrow, vec![int_ty.clone(), int_ty.clone()]),
        span: b.span(),
        kind: TermKind::App {
            func: Box::new(foo_ref),
            arg: Box::new(eight),
        },
    };

    let bar_body = Term {
        id: b.next_id(),
        ty: int_ty.clone(),
        span: b.span(),
        kind: TermKind::App {
            func: Box::new(foo_8),
            arg: Box::new(nine),
        },
    };

    let symbols = b.finish();

    let program = Program {
        defs: vec![
            Def {
                name: foo_sym,
                ty: foo_body.ty.clone(),
                body: foo_body,
                meta: DefMeta::Function,
                arity: 2,
            },
            Def {
                name: bar_sym,
                ty: int_ty.clone(),
                body: bar_body,
                meta: DefMeta::Function,
                arity: 0,
            },
        ],
        uniforms: vec![],
        storage: vec![],
        symbols,
    };

    let result = PartialEvaluator::partial_eval(program);

    // bar should be inlined and folded to 17
    let bar_def = result
        .defs
        .iter()
        .find(|d| result.symbols.get(d.name).expect("BUG: symbol not in table") == "bar")
        .unwrap();
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
    let mut b = TestBuilder::new();

    let y_sym = b.sym("y");
    let f_sym = b.sym("f");
    let g_sym = b.sym("g");
    let main_sym = b.sym("main");

    let span = b.span();

    // Build: def g = |y| y  (identity function)
    let g_body = Term {
        id: b.next_id(),
        ty: arrow_ty(int_ty(), int_ty()),
        span,
        kind: TermKind::Lam {
            param: y_sym,
            param_ty: int_ty(),
            body: Box::new(Term {
                id: b.next_id(),
                ty: int_ty(),
                span,
                kind: TermKind::Var(y_sym),
            }),
        },
    };

    // Build: def main = let f = g in f 42
    // This is: Let { name: "f", rhs: Var("g"), body: App(Var("f"), 42) }
    let main_body = Term {
        id: b.next_id(),
        ty: int_ty(),
        span,
        kind: TermKind::Let {
            name: f_sym,
            name_ty: arrow_ty(int_ty(), int_ty()),
            rhs: Box::new(Term {
                id: b.next_id(),
                ty: arrow_ty(int_ty(), int_ty()),
                span,
                kind: TermKind::Var(g_sym),
            }),
            body: Box::new(Term {
                id: b.next_id(),
                ty: int_ty(),
                span,
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: b.next_id(),
                        ty: arrow_ty(int_ty(), int_ty()),
                        span,
                        kind: TermKind::Var(f_sym),
                    }),
                    arg: Box::new(Term {
                        id: b.next_id(),
                        ty: int_ty(),
                        span,
                        kind: TermKind::IntLit("42".to_string()),
                    }),
                },
            }),
        },
    };

    let symbols = b.finish();

    let program = Program {
        defs: vec![
            Def {
                name: g_sym,
                ty: g_body.ty.clone(),
                body: g_body,
                meta: DefMeta::Function,
                arity: 1,
            },
            Def {
                name: main_sym,
                ty: int_ty(),
                body: main_body,
                meta: DefMeta::Function,
                arity: 0,
            },
        ],
        uniforms: vec![],
        storage: vec![],
        symbols,
    };

    let result = PartialEvaluator::partial_eval(program);

    // Find main's body - it should be simplified to just `42`
    // because g is identity and f aliases g, so f 42 = g 42 = 42
    let main_def = result
        .defs
        .iter()
        .find(|d| result.symbols.get(d.name).expect("BUG: symbol not in table") == "main")
        .unwrap();

    // The result should be IntLit("42") since g is identity
    match &main_def.body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "42"),
        other => panic!("Expected IntLit(42), got {:?}", other),
    }
}

/// Test that function alias without full application still uses correct name
#[test]
fn test_function_alias_partial_application() {
    let mut b = TestBuilder::new();

    let x_sym = b.sym("x");
    let y_sym = b.sym("y");
    let f_sym = b.sym("f");
    let g_sym = b.sym("g");
    let main_sym = b.sym("main");

    let span = b.span();

    // Build: def g = |x| |y| x  (const function, arity 2)
    let g_body = Term {
        id: b.next_id(),
        ty: arrow_ty(int_ty(), arrow_ty(int_ty(), int_ty())),
        span,
        kind: TermKind::Lam {
            param: x_sym,
            param_ty: int_ty(),
            body: Box::new(Term {
                id: b.next_id(),
                ty: arrow_ty(int_ty(), int_ty()),
                span,
                kind: TermKind::Lam {
                    param: y_sym,
                    param_ty: int_ty(),
                    body: Box::new(Term {
                        id: b.next_id(),
                        ty: int_ty(),
                        span,
                        kind: TermKind::Var(x_sym),
                    }),
                },
            }),
        },
    };

    // Build: def main = let f = g in f 1 2
    // f aliases g, so f 1 2 should become g 1 2 = 1
    let main_body = Term {
        id: b.next_id(),
        ty: int_ty(),
        span,
        kind: TermKind::Let {
            name: f_sym,
            name_ty: arrow_ty(int_ty(), arrow_ty(int_ty(), int_ty())),
            rhs: Box::new(Term {
                id: b.next_id(),
                ty: arrow_ty(int_ty(), arrow_ty(int_ty(), int_ty())),
                span,
                kind: TermKind::Var(g_sym),
            }),
            body: Box::new(Term {
                id: b.next_id(),
                ty: int_ty(),
                span,
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: b.next_id(),
                        ty: arrow_ty(int_ty(), int_ty()),
                        span,
                        kind: TermKind::App {
                            func: Box::new(Term {
                                id: b.next_id(),
                                ty: arrow_ty(int_ty(), arrow_ty(int_ty(), int_ty())),
                                span,
                                kind: TermKind::Var(f_sym),
                            }),
                            arg: Box::new(Term {
                                id: b.next_id(),
                                ty: int_ty(),
                                span,
                                kind: TermKind::IntLit("1".to_string()),
                            }),
                        },
                    }),
                    arg: Box::new(Term {
                        id: b.next_id(),
                        ty: int_ty(),
                        span,
                        kind: TermKind::IntLit("2".to_string()),
                    }),
                },
            }),
        },
    };

    let symbols = b.finish();

    let program = Program {
        defs: vec![
            Def {
                name: g_sym,
                ty: g_body.ty.clone(),
                body: g_body,
                meta: DefMeta::Function,
                arity: 2,
            },
            Def {
                name: main_sym,
                ty: int_ty(),
                body: main_body,
                meta: DefMeta::Function,
                arity: 0,
            },
        ],
        uniforms: vec![],
        storage: vec![],
        symbols,
    };

    let result = PartialEvaluator::partial_eval(program);

    // Find main's body - it should be simplified to `1`
    // because g x y = x, so f 1 2 = g 1 2 = 1
    let main_def = result
        .defs
        .iter()
        .find(|d| result.symbols.get(d.name).expect("BUG: symbol not in table") == "main")
        .unwrap();

    match &main_def.body.kind {
        TermKind::IntLit(s) => assert_eq!(s, "1"),
        other => panic!("Expected IntLit(1), got {:?}", other),
    }
}

/// Test that `let f = f32.sin in f x` becomes `f32.sin x`
#[test]
fn test_intrinsic_alias_inlining() {
    let mut b = TestBuilder::new();

    let f_sym = b.sym("f");
    let f32_sin_sym = b.sym("f32.sin");
    let main_sym = b.sym("main");

    let span = b.span();
    let float_ty = Type::Constructed(TypeName::Float(32), vec![]);

    // Build: def main = let f = f32.sin in f 0.5
    // f32.sin is an intrinsic (not in defs), so it evaluates to Unknown(Var("f32.sin"))
    let main_body = Term {
        id: b.next_id(),
        ty: float_ty.clone(),
        span,
        kind: TermKind::Let {
            name: f_sym,
            name_ty: arrow_ty(float_ty.clone(), float_ty.clone()),
            rhs: Box::new(Term {
                id: b.next_id(),
                ty: arrow_ty(float_ty.clone(), float_ty.clone()),
                span,
                kind: TermKind::Var(f32_sin_sym),
            }),
            body: Box::new(Term {
                id: b.next_id(),
                ty: float_ty.clone(),
                span,
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: b.next_id(),
                        ty: arrow_ty(float_ty.clone(), float_ty.clone()),
                        span,
                        kind: TermKind::Var(f_sym),
                    }),
                    arg: Box::new(Term {
                        id: b.next_id(),
                        ty: float_ty.clone(),
                        span,
                        kind: TermKind::FloatLit(0.5),
                    }),
                },
            }),
        },
    };

    let symbols = b.finish();

    let program = Program {
        defs: vec![Def {
            name: main_sym,
            ty: float_ty.clone(),
            body: main_body,
            meta: DefMeta::Function,
            arity: 0,
        }],
        uniforms: vec![],
        storage: vec![],
        symbols,
    };

    let result = PartialEvaluator::partial_eval(program);
    let main_def = result
        .defs
        .iter()
        .find(|d| result.symbols.get(d.name).expect("BUG: symbol not in table") == "main")
        .unwrap();

    // The result should be App(f32.sin, 0.5) - the alias `f` should be resolved to `f32.sin`
    match &main_def.body.kind {
        TermKind::App { func, arg } => {
            // Check that the function is now f32.sin (not f)
            match &func.kind {
                TermKind::Var(sym) => {
                    let name = result.symbols.get(*sym).expect("BUG: symbol not in table");
                    assert_eq!(name, "f32.sin");
                }
                other => panic!("Expected Var(f32.sin), got {:?}", other),
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
