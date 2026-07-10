//! Tests for TLC partial evaluation.

use super::{PartialEvaluator, VarRef};
use crate::ast::{BinaryOp, Span, TypeName};
use crate::tlc::{Def, DefMeta, Lambda, Program, Term, TermId, TermIdSource, TermKind};
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::HashMap;

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

    fn next_id(&mut self) -> TermId {
        self.ids.next_id()
    }

    fn span(&self) -> Span {
        Span::dummy()
    }

    fn finish(self) -> SymbolTable {
        self.symbols
    }
}

fn input_ae(boxed: Box<crate::tlc::Term>) -> crate::tlc::ArrayExpr {
    use crate::tlc::{ArrayExpr, TermKind};
    let t = *boxed;
    match t.kind {
        TermKind::Var(vr) => ArrayExpr::Var(vr, t.ty),
        TermKind::ArrayExpr(ae) => ae,
        other => panic!("test SOAC input must be a variable or array expr, got {other:?}"),
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
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
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
        ty: Type::Constructed(TypeName::Bool, vec![]),
        span: make_span(),
        kind: TermKind::BoolLit(b),
    }
}

fn make_binop(ids: &mut TermIdSource, op: &str, lhs: Term, rhs: Term) -> Term {
    let result_ty = lhs.ty.clone();
    let partial_ty = Type::Constructed(TypeName::Arrow, vec![result_ty.clone(), result_ty.clone()]);
    let binop_ty = Type::Constructed(TypeName::Arrow, vec![result_ty.clone(), partial_ty.clone()]);

    Term {
        id: ids.next_id(),
        ty: result_ty,
        span: make_span(),
        kind: TermKind::App {
            func: Box::new(Term {
                id: ids.next_id(),
                ty: binop_ty,
                span: make_span(),
                kind: TermKind::BinOp(BinaryOp { op: op.to_string() }),
            }),
            args: vec![lhs, rhs],
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

    let mut result = program;
    PartialEvaluator::partial_eval(&mut result);
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

    let mut result = program;
    PartialEvaluator::partial_eval(&mut result);

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
        kind: TermKind::Var(VarRef::Symbol(x_sym)),
    };
    let zero = make_int(&mut b.ids, 0);
    let term = make_binop(&mut b.ids, "+", x, zero);

    let program = make_program(test_sym, term, b.finish());

    let mut result = program;
    PartialEvaluator::partial_eval(&mut result);

    // x + 0 should simplify to just x
    match &result.defs[0].body.kind {
        TermKind::Var(VarRef::Symbol(sym)) => {
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
        kind: TermKind::Var(VarRef::Symbol(x_sym)),
    };
    let term = make_binop(&mut b.ids, "*", one, x);

    let program = make_program(test_sym, term, b.finish());

    let mut result = program;
    PartialEvaluator::partial_eval(&mut result);

    // 1 * x should simplify to just x
    match &result.defs[0].body.kind {
        TermKind::Var(VarRef::Symbol(sym)) => {
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
        kind: TermKind::Var(VarRef::Symbol(x_sym)),
    };
    let zero = make_int(&mut b.ids, 0);
    let term = make_binop(&mut b.ids, "*", x, zero);

    let program = make_program(test_sym, term, b.finish());

    let mut result = program;
    PartialEvaluator::partial_eval(&mut result);

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

    let mut result = program;
    PartialEvaluator::partial_eval(&mut result);

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

    let mut result = program;
    PartialEvaluator::partial_eval(&mut result);

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
        kind: TermKind::Var(VarRef::Symbol(x_sym)),
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

    let mut result = program;
    PartialEvaluator::partial_eval(&mut result);

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
        kind: TermKind::Var(VarRef::Symbol(a_sym)),
    };
    let b_var = Term {
        id: b.next_id(),
        ty: int_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(b_sym)),
    };
    let a_plus_b = make_binop(&mut b.ids, "+", a_var, b_var);

    let foo_body = Term {
        id: b.next_id(),
        ty: Type::Constructed(
            TypeName::Arrow,
            vec![
                int_ty.clone(),
                Type::Constructed(TypeName::Arrow, vec![int_ty.clone(), int_ty.clone()]),
            ],
        ),
        span: b.span(),
        kind: TermKind::Lambda(Lambda {
            params: vec![(a_sym, int_ty.clone()), (b_sym, int_ty.clone())],
            body: Box::new(a_plus_b),
            ret_ty: int_ty.clone(),
        }),
    };

    // Build bar: foo 8 9
    let eight = make_int(&mut b.ids, 8);
    let nine = make_int(&mut b.ids, 9);

    let foo_ref = Term {
        id: b.next_id(),
        ty: foo_body.ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(foo_sym)),
    };

    let bar_body = Term {
        id: b.next_id(),
        ty: int_ty.clone(),
        span: b.span(),
        kind: TermKind::App {
            func: Box::new(foo_ref),
            args: vec![eight, nine],
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
                param_diets: vec![],
                return_diet: crate::types::Diet::observing(),
            },
            Def {
                name: bar_sym,
                ty: int_ty.clone(),
                body: bar_body,
                meta: DefMeta::Function,
                arity: 0,
                param_diets: vec![],
                return_diet: crate::types::Diet::observing(),
            },
        ],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut result = program;
    PartialEvaluator::partial_eval(&mut result);

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
        kind: TermKind::Lambda(Lambda {
            params: vec![(y_sym, int_ty())],
            body: Box::new(Term {
                id: b.next_id(),
                ty: int_ty(),
                span,
                kind: TermKind::Var(VarRef::Symbol(y_sym)),
            }),
            ret_ty: int_ty(),
        }),
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
                kind: TermKind::Var(VarRef::Symbol(g_sym)),
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
                        kind: TermKind::Var(VarRef::Symbol(f_sym)),
                    }),
                    args: vec![Term {
                        id: b.next_id(),
                        ty: int_ty(),
                        span,
                        kind: TermKind::IntLit("42".to_string()),
                    }],
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
                param_diets: vec![],
                return_diet: crate::types::Diet::observing(),
            },
            Def {
                name: main_sym,
                ty: int_ty(),
                body: main_body,
                meta: DefMeta::Function,
                arity: 0,
                param_diets: vec![],
                return_diet: crate::types::Diet::observing(),
            },
        ],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut result = program;
    PartialEvaluator::partial_eval(&mut result);

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
        kind: TermKind::Lambda(Lambda {
            params: vec![(x_sym, int_ty()), (y_sym, int_ty())],
            body: Box::new(Term {
                id: b.next_id(),
                ty: int_ty(),
                span,
                kind: TermKind::Var(VarRef::Symbol(x_sym)),
            }),
            ret_ty: int_ty(),
        }),
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
                kind: TermKind::Var(VarRef::Symbol(g_sym)),
            }),
            body: Box::new(Term {
                id: b.next_id(),
                ty: int_ty(),
                span,
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: b.next_id(),
                        ty: arrow_ty(int_ty(), arrow_ty(int_ty(), int_ty())),
                        span,
                        kind: TermKind::Var(VarRef::Symbol(f_sym)),
                    }),
                    args: vec![
                        Term {
                            id: b.next_id(),
                            ty: int_ty(),
                            span,
                            kind: TermKind::IntLit("1".to_string()),
                        },
                        Term {
                            id: b.next_id(),
                            ty: int_ty(),
                            span,
                            kind: TermKind::IntLit("2".to_string()),
                        },
                    ],
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
                param_diets: vec![],
                return_diet: crate::types::Diet::observing(),
            },
            Def {
                name: main_sym,
                ty: int_ty(),
                body: main_body,
                meta: DefMeta::Function,
                arity: 0,
                param_diets: vec![],
                return_diet: crate::types::Diet::observing(),
            },
        ],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut result = program;
    PartialEvaluator::partial_eval(&mut result);

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
                kind: TermKind::Var(VarRef::Symbol(f32_sin_sym)),
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
                        kind: TermKind::Var(VarRef::Symbol(f_sym)),
                    }),
                    args: vec![Term {
                        id: b.next_id(),
                        ty: float_ty.clone(),
                        span,
                        kind: TermKind::FloatLit(0.5),
                    }],
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
            param_diets: vec![],
            return_diet: crate::types::Diet::observing(),
        }],
        symbols,
        def_syms: HashMap::new(),
    };

    let mut result = program;
    PartialEvaluator::partial_eval(&mut result);
    let main_def = result
        .defs
        .iter()
        .find(|d| result.symbols.get(d.name).expect("BUG: symbol not in table") == "main")
        .unwrap();

    // The result should be App(f32.sin, 0.5) - the alias `f` should be resolved to `f32.sin`
    match &main_def.body.kind {
        TermKind::App { func, args } => {
            // Check that the function is now f32.sin (not f)
            match &func.kind {
                TermKind::Var(VarRef::Symbol(sym)) => {
                    let name = result.symbols.get(*sym).expect("BUG: symbol not in table");
                    assert_eq!(name, "f32.sin");
                }
                other => panic!("Expected Var(f32.sin), got {:?}", other),
            }
            // Check the argument is still 0.5
            assert_eq!(args.len(), 1);
            match &args[0].kind {
                TermKind::FloatLit(f) => assert!((*f - 0.5).abs() < 0.001),
                other => panic!("Expected FloatLit(0.5), got {:?}", other),
            }
        }
        other => panic!("Expected App, got {:?}", other),
    }
}

// =============================================================================
// Substitution survives through SOAC residualization
// =============================================================================
//
// Regression: `let m = lit in map(f, m)` — partial_eval evaluates the let-rhs
// and binds `m` in env, then descends into the body. Hitting the SOAC at
// `body`, the previous code returned `Value::Unknown(term.clone())` *without
// substituting let-bound vars into the SOAC's sub-terms*, so the resulting
// program had a dangling free `Var(m_sym)` inside the SOAC's input expression.
//
// The corrected behavior: when residualizing a SOAC (or ArrayExpr), free
// Vars referring to env-bound symbols must be substituted with the let-rhs
// term (binder-aware — Lambda params and inner Let-bound names shadow env).

use crate::tlc::{ArrayExpr, SoacBody, SoacDestination, SoacOp};

/// Assert that `target_sym` does not occur as a free `Var` anywhere
/// in `term`. Recurses via `map_children`, so it sees through Soac,
/// ArrayExpr, App, Tuple, etc. without needing per-variant handling.
/// Binder-aware via the `bound` set: Lambda params, Let names, etc.
/// shadow the assertion (we treat any matching symbol added by an
/// enclosing binder as a different identifier).
fn assert_no_free_reference_to(term: &crate::tlc::Term, target_sym: SymbolId) {
    fn walk(t: &crate::tlc::Term, target: SymbolId, shadowed: bool) {
        if shadowed {
            return;
        }
        match &t.kind {
            TermKind::Var(VarRef::Symbol(sym)) => {
                assert!(
                    *sym != target,
                    "free Var(sym={:?}) — let-bound symbol not substituted through SOAC",
                    sym.0,
                );
            }
            TermKind::Let { name, rhs, body, .. } => {
                walk(rhs, target, false);
                walk(body, target, *name == target);
            }
            TermKind::Lambda(lam) => {
                let shadow = lam.params.iter().any(|(p, _)| *p == target);
                walk(&lam.body, target, shadow);
            }
            _ => {
                let cloned = crate::tlc::Term {
                    id: t.id,
                    ty: t.ty.clone(),
                    span: t.span,
                    kind: t.kind.clone(),
                };
                let _ = cloned.map_children(&mut |child| {
                    walk(&child, target, false);
                    child
                });
            }
        }
    }
    walk(term, target_sym, false);
}

#[test]
fn let_bound_array_substituted_through_soac_input() {
    // Construct (paraphrased):
    //   def test() [3]i32 =
    //       let m: [3]i32 = [1, 2, 3] in
    //       map(|x: i32| x, m)
    //
    // After partial_eval, the let may be eliminated — but if so, every
    // reference to `m` inside the SOAC must be substituted with the literal.
    let mut b = TestBuilder::new();
    let m_sym = b.sym("m");
    let x_sym = b.sym("x");
    let test_sym = b.sym("test");

    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            i32_ty.clone(),
            Type::Constructed(TypeName::Size(3), vec![]),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            crate::types::no_buffer(),
        ],
    );

    // [1, 2, 3] as ArrayExpr::Literal
    let lit_elems = vec![
        make_int(&mut b.ids, 1),
        make_int(&mut b.ids, 2),
        make_int(&mut b.ids, 3),
    ];
    let arr_lit_term = Term {
        id: b.next_id(),
        ty: arr_ty.clone(),
        span: b.span(),
        kind: TermKind::ArrayExpr(ArrayExpr::Literal(lit_elems)),
    };

    // `m`-typed var reference inside the SOAC input
    let m_var = Term {
        id: b.next_id(),
        ty: arr_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(m_sym)),
    };

    // Identity lambda |x: i32| x
    let x_var = Term {
        id: b.next_id(),
        ty: i32_ty.clone(),
        span: b.span(),
        kind: TermKind::Var(VarRef::Symbol(x_sym)),
    };
    let lam = Lambda {
        params: vec![(x_sym, i32_ty.clone())],
        body: Box::new(x_var),
        ret_ty: i32_ty.clone(),
    };

    let soac_term = Term {
        id: b.next_id(),
        ty: arr_ty.clone(),
        span: b.span(),
        kind: TermKind::Soac(SoacOp::Map {
            lam: SoacBody {
                lam,
                captures: vec![],
            },
            inputs: vec![input_ae(Box::new(m_var))],
            destination: SoacDestination::Fresh,
        }),
    };

    let let_term = Term {
        id: b.next_id(),
        ty: arr_ty.clone(),
        span: b.span(),
        kind: TermKind::Let {
            name: m_sym,
            name_ty: arr_ty.clone(),
            rhs: Box::new(arr_lit_term),
            body: Box::new(soac_term),
        },
    };

    let program = make_program(test_sym, let_term, b.finish());
    let mut result = program;
    PartialEvaluator::partial_eval(&mut result);

    assert_eq!(result.defs.len(), 1);
    assert_no_free_reference_to(&result.defs[0].body, m_sym);
}
