use super::defunctionalize::defunctionalize;
use super::{Def, DefMeta, LoopKind, Program, Term, TermIdSource, TermKind};
use crate::ast::{BinaryOp, Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::HashSet;

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

    fn lookup(&self, sym: SymbolId) -> &str {
        self.symbols.get(sym).expect("BUG: symbol not in table")
    }

    fn finish(self) -> SymbolTable {
        self.symbols
    }
}

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

#[allow(dead_code)]
fn array_ty(elem: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(TypeName::Array, vec![elem])
}

fn arrow(from: Type<TypeName>, to: Type<TypeName>) -> Type<TypeName> {
    Type::Constructed(TypeName::Arrow, vec![from, to])
}

/// Helper to print TLC term structure for debugging
#[allow(dead_code)]
fn print_term(term: &Term, symbols: &SymbolTable, indent: usize) -> String {
    let unknown = "<unknown>".to_string();
    let pad = "  ".repeat(indent);
    match &term.kind {
        TermKind::Var(sym) => {
            let name = symbols.get(*sym).unwrap_or(&unknown);
            format!("{}Var({})", pad, name)
        }
        TermKind::IntLit(n) => format!("{}Int({})", pad, n),
        TermKind::FloatLit(f) => format!("{}Float({})", pad, f),
        TermKind::BoolLit(b) => format!("{}Bool({})", pad, b),
        TermKind::StringLit(s) => format!("{}String({})", pad, s),
        TermKind::Lam { param, body, .. } => {
            let name = symbols.get(*param).unwrap_or(&unknown);
            format!("{}Lam({})\n{}", pad, name, print_term(body, symbols, indent + 1))
        }
        TermKind::App { func, arg } => {
            format!(
                "{}App\n{}  func:\n{}\n{}  arg:\n{}",
                pad,
                pad,
                print_term(func, symbols, indent + 2),
                pad,
                print_term(arg, symbols, indent + 2)
            )
        }
        TermKind::BinOp(op) => format!("{}BinOp({})", pad, op.op),
        TermKind::UnOp(op) => format!("{}UnOp({})", pad, op.op),
        TermKind::Let { name, rhs, body, .. } => {
            let n = symbols.get(*name).unwrap_or(&unknown);
            format!(
                "{}Let {} =\n{}\n{}in\n{}",
                pad,
                n,
                print_term(rhs, symbols, indent + 1),
                pad,
                print_term(body, symbols, indent + 1)
            )
        }
        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            format!(
                "{}If\n{}\n{}then\n{}\n{}else\n{}",
                pad,
                print_term(cond, symbols, indent + 1),
                pad,
                print_term(then_branch, symbols, indent + 1),
                pad,
                print_term(else_branch, symbols, indent + 1)
            )
        }
        TermKind::Loop {
            loop_var,
            init,
            kind,
            body,
            ..
        } => {
            let lv = symbols.get(*loop_var).unwrap_or(&unknown);
            let kind_str = match kind {
                LoopKind::For { var, .. } => {
                    let v = symbols.get(*var).unwrap_or(&unknown);
                    format!("for {} in ...", v)
                }
                LoopKind::ForRange { var, .. } => {
                    let v = symbols.get(*var).unwrap_or(&unknown);
                    format!("for {} < ...", v)
                }
                LoopKind::While { .. } => "while ...".to_string(),
            };
            format!(
                "{}Loop {} = {} ({})\n{}body:\n{}",
                pad,
                lv,
                print_term(init, symbols, 0).trim(),
                kind_str,
                pad,
                print_term(body, symbols, indent + 1)
            )
        }
        TermKind::Extern(linkage) => format!("{}Extern(\"{}\")", pad, linkage),
    }
}

/// Helper to print all defs in a program
#[allow(dead_code)]
fn print_program(program: &Program) -> String {
    let unknown = "<unknown>".to_string();
    let mut out = String::new();
    for def in &program.defs {
        let name = program.symbols.get(def.name).unwrap_or(&unknown);
        out.push_str(&format!("\n=== {} (arity {}) ===\n", name, def.arity));
        out.push_str(&print_term(&def.body, &program.symbols, 0));
        out.push('\n');
    }
    out
}

#[test]
fn test_defunc_simple_lambda_no_capture() {
    // def f = |x| x
    let mut b = TestBuilder::new();

    let x_sym = b.sym("x");
    let f_sym = b.sym("f");

    let lam = Term {
        id: b.next_id(),
        ty: arrow(i32_ty(), i32_ty()),
        span: b.span(),
        kind: TermKind::Lam {
            param: x_sym,
            param_ty: i32_ty(),
            body: Box::new(Term {
                id: b.next_id(),
                ty: i32_ty(),
                span: b.span(),
                kind: TermKind::Var(x_sym),
            }),
        },
    };

    let program = Program {
        defs: vec![Def {
            name: f_sym,
            ty: lam.ty.clone(),
            body: lam,
            meta: DefMeta::Function,
            arity: 1,
        }],
        uniforms: vec![],
        storage: vec![],
        symbols: b.finish(),
    };

    let known_defs = HashSet::new();
    let result = defunctionalize(program, &known_defs);

    // Should preserve the parameter lambda (not lift it)
    assert_eq!(result.defs.len(), 1);
    assert!(matches!(result.defs[0].body.kind, TermKind::Lam { .. }));
}

#[test]
fn test_defunc_lambda_with_capture() {
    // def f y = let g = |x| x + y in g
    // The lambda |x| x + y captures y
    let mut b = TestBuilder::new();

    let x_sym = b.sym("x");
    let y_sym = b.sym("y");
    let g_sym = b.sym("g");
    let f_sym = b.sym("f");

    // Build: |x| x + y  (where y is free)
    let inner_lam = Term {
        id: b.next_id(),
        ty: arrow(i32_ty(), i32_ty()),
        span: b.span(),
        kind: TermKind::Lam {
            param: x_sym,
            param_ty: i32_ty(),
            body: Box::new(Term {
                id: b.next_id(),
                ty: i32_ty(),
                span: b.span(),
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: b.next_id(),
                        ty: arrow(i32_ty(), arrow(i32_ty(), i32_ty())),
                        span: b.span(),
                        kind: TermKind::BinOp(BinaryOp { op: "+".to_string() }),
                    }),
                    arg: Box::new(Term {
                        id: b.next_id(),
                        ty: i32_ty(),
                        span: b.span(),
                        kind: TermKind::Var(y_sym),
                    }),
                },
            }),
        },
    };

    // let g = inner_lam in g
    let let_expr = Term {
        id: b.next_id(),
        ty: arrow(i32_ty(), i32_ty()),
        span: b.span(),
        kind: TermKind::Let {
            name: g_sym,
            name_ty: arrow(i32_ty(), i32_ty()),
            rhs: Box::new(inner_lam),
            body: Box::new(Term {
                id: b.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span: b.span(),
                kind: TermKind::Var(g_sym),
            }),
        },
    };

    // |y| let_expr
    let outer_lam = Term {
        id: b.next_id(),
        ty: arrow(i32_ty(), arrow(i32_ty(), i32_ty())),
        span: b.span(),
        kind: TermKind::Lam {
            param: y_sym,
            param_ty: i32_ty(),
            body: Box::new(let_expr),
        },
    };

    let symbols = b.finish();

    let program = Program {
        defs: vec![Def {
            name: f_sym,
            ty: outer_lam.ty.clone(),
            body: outer_lam,
            meta: DefMeta::Function,
            arity: 1,
        }],
        uniforms: vec![],
        storage: vec![],
        symbols,
    };

    let known_defs = HashSet::new();
    let result = defunctionalize(program, &known_defs);

    // Should have lifted the inner lambda
    assert!(result.defs.len() >= 2, "Expected lifted lambda def");

    // Find the lifted lambda
    let lifted = result.defs.iter().find(|d| {
        let name = result.symbols.get(d.name).expect("BUG: symbol not in table");
        name.starts_with("_w_lambda_")
    });
    assert!(lifted.is_some(), "Should have a _w_lambda_ definition");
}

/// Test that specialized HOF bodies are properly defunctionalized.
///
/// This is a stricter test: hof_outer doesn't have a lambda in its body,
/// it just passes `g` directly. But the CALL SITE wraps `g` in a new lambda.
///
/// hof_inner(f, x) = f(x)
/// hof_outer(g, y) = hof_inner(g, y)  // No lambda here, just passes g through
/// main(cap) = hof_outer(|a| a + cap, 5)
///
/// When hof_outer is specialized for _w_lambda_0:
/// - hof_outer$0 body becomes: hof_inner(_w_lambda_0, y)
/// - This should trigger specialization of hof_inner for _w_lambda_0
///
/// This test fails if specialized bodies aren't processed (fixpoint issue).
#[test]
fn test_nested_hof_passthrough() {
    let mut b = TestBuilder::new();

    let f_sym = b.sym("f");
    let x_sym = b.sym("x");
    let g_sym = b.sym("g");
    let y_sym = b.sym("y");
    let hof_inner_sym = b.sym("hof_inner");
    let hof_outer_sym = b.sym("hof_outer");
    let main_sym = b.sym("main");
    let a_sym = b.sym("a");
    let cap_sym = b.sym("cap");

    let span = b.span();

    // hof_inner = |f| |x| f(x)
    let hof_inner_body = Term {
        id: b.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::App {
            func: Box::new(Term {
                id: b.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span,
                kind: TermKind::Var(f_sym),
            }),
            arg: Box::new(Term {
                id: b.next_id(),
                ty: i32_ty(),
                span,
                kind: TermKind::Var(x_sym),
            }),
        },
    };

    let hof_inner = Term {
        id: b.next_id(),
        ty: arrow(arrow(i32_ty(), i32_ty()), arrow(i32_ty(), i32_ty())),
        span,
        kind: TermKind::Lam {
            param: f_sym,
            param_ty: arrow(i32_ty(), i32_ty()),
            body: Box::new(Term {
                id: b.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span,
                kind: TermKind::Lam {
                    param: x_sym,
                    param_ty: i32_ty(),
                    body: Box::new(hof_inner_body),
                },
            }),
        },
    };

    // hof_outer = |g| |y| hof_inner g y
    // Just passes g directly to hof_inner, no lambda wrapping
    let hof_inner_call = Term {
        id: b.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::App {
            func: Box::new(Term {
                id: b.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span,
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: b.next_id(),
                        ty: arrow(arrow(i32_ty(), i32_ty()), arrow(i32_ty(), i32_ty())),
                        span,
                        kind: TermKind::Var(hof_inner_sym),
                    }),
                    arg: Box::new(Term {
                        id: b.next_id(),
                        ty: arrow(i32_ty(), i32_ty()),
                        span,
                        kind: TermKind::Var(g_sym), // <-- Just passes g, no lambda
                    }),
                },
            }),
            arg: Box::new(Term {
                id: b.next_id(),
                ty: i32_ty(),
                span,
                kind: TermKind::Var(y_sym),
            }),
        },
    };

    let hof_outer = Term {
        id: b.next_id(),
        ty: arrow(arrow(i32_ty(), i32_ty()), arrow(i32_ty(), i32_ty())),
        span,
        kind: TermKind::Lam {
            param: g_sym,
            param_ty: arrow(i32_ty(), i32_ty()),
            body: Box::new(Term {
                id: b.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span,
                kind: TermKind::Lam {
                    param: y_sym,
                    param_ty: i32_ty(),
                    body: Box::new(hof_inner_call),
                },
            }),
        },
    };

    // main = |cap| hof_outer (|a| a + cap) 5
    let a_plus_cap = Term {
        id: b.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::App {
            func: Box::new(Term {
                id: b.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span,
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: b.next_id(),
                        ty: arrow(i32_ty(), arrow(i32_ty(), i32_ty())),
                        span,
                        kind: TermKind::BinOp(BinaryOp { op: "+".to_string() }),
                    }),
                    arg: Box::new(Term {
                        id: b.next_id(),
                        ty: i32_ty(),
                        span,
                        kind: TermKind::Var(a_sym),
                    }),
                },
            }),
            arg: Box::new(Term {
                id: b.next_id(),
                ty: i32_ty(),
                span,
                kind: TermKind::Var(cap_sym),
            }),
        },
    };

    let capturing_lambda = Term {
        id: b.next_id(),
        ty: arrow(i32_ty(), i32_ty()),
        span,
        kind: TermKind::Lam {
            param: a_sym,
            param_ty: i32_ty(),
            body: Box::new(a_plus_cap),
        },
    };

    let main_body = Term {
        id: b.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::App {
            func: Box::new(Term {
                id: b.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span,
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: b.next_id(),
                        ty: arrow(arrow(i32_ty(), i32_ty()), arrow(i32_ty(), i32_ty())),
                        span,
                        kind: TermKind::Var(hof_outer_sym),
                    }),
                    arg: Box::new(capturing_lambda),
                },
            }),
            arg: Box::new(Term {
                id: b.next_id(),
                ty: i32_ty(),
                span,
                kind: TermKind::IntLit("5".to_string()),
            }),
        },
    };

    let main_def = Term {
        id: b.next_id(),
        ty: arrow(i32_ty(), i32_ty()),
        span,
        kind: TermKind::Lam {
            param: cap_sym,
            param_ty: i32_ty(),
            body: Box::new(main_body),
        },
    };

    let symbols = b.finish();

    let program = Program {
        defs: vec![
            Def {
                name: hof_inner_sym,
                ty: hof_inner.ty.clone(),
                body: hof_inner,
                meta: DefMeta::Function,
                arity: 2,
            },
            Def {
                name: hof_outer_sym,
                ty: hof_outer.ty.clone(),
                body: hof_outer,
                meta: DefMeta::Function,
                arity: 2,
            },
            Def {
                name: main_sym,
                ty: main_def.ty.clone(),
                body: main_def,
                meta: DefMeta::Function,
                arity: 1,
            },
        ],
        uniforms: vec![],
        storage: vec![],
        symbols,
    };

    let known_defs = HashSet::new();
    let result = defunctionalize(program, &known_defs);

    let def_names: Vec<&str> = result
        .defs
        .iter()
        .map(|d| result.symbols.get(d.name).expect("BUG: symbol not in table").as_str())
        .collect();
    eprintln!(
        "Defs after defunctionalization (passthrough test): {:?}",
        def_names
    );

    // Expected:
    // - hof_inner (original)
    // - hof_outer (original)
    // - main (original)
    // - _w_lambda_0 (lifted from main, captures cap)
    // - hof_outer$0 (specialized for _w_lambda_0)
    // - hof_inner$0 (specialized for _w_lambda_0, triggered by processing hof_outer$0's body)
    //
    // The key assertion: hof_inner MUST be specialized even though the original
    // hof_outer doesn't have a lambda - the specialized body hof_inner(_w_lambda_0, y)
    // contains a HOF call that needs specialization.

    let hof_outer_specialized = def_names.iter().any(|n| n.starts_with("hof_outer$"));
    assert!(
        hof_outer_specialized,
        "hof_outer should be specialized. Defs: {:?}",
        def_names
    );

    // THIS IS THE CRITICAL ASSERTION:
    // hof_inner must be specialized because hof_outer$0's body is: hof_inner(_w_lambda_0, y)
    // If we don't process hof_outer$0's body (fixpoint issue), hof_inner won't be specialized
    let hof_inner_specialized = def_names.iter().any(|n| n.starts_with("hof_inner$"));
    assert!(
        hof_inner_specialized,
        "hof_inner should be specialized (from processing hof_outer$0's body). Defs: {:?}",
        def_names
    );
}
