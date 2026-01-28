use super::defunctionalize::defunctionalize;
use super::{Def, DefMeta, LoopKind, Program, Term, TermIdSource, TermKind};
use crate::ast::{BinaryOp, Span, TypeName};
use polytype::Type;
use std::collections::HashSet;

fn dummy_span() -> Span {
    Span::dummy()
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
fn print_term(term: &Term, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match &term.kind {
        TermKind::Var(name) => format!("{}Var({})", pad, name),
        TermKind::IntLit(n) => format!("{}Int({})", pad, n),
        TermKind::FloatLit(f) => format!("{}Float({})", pad, f),
        TermKind::BoolLit(b) => format!("{}Bool({})", pad, b),
        TermKind::StringLit(s) => format!("{}String({})", pad, s),
        TermKind::Lam { param, body, .. } => {
            format!("{}Lam({})\n{}", pad, param, print_term(body, indent + 1))
        }
        TermKind::App { func, arg } => {
            format!(
                "{}App\n{}  func:\n{}\n{}  arg:\n{}",
                pad,
                pad,
                print_term(func, indent + 2),
                pad,
                print_term(arg, indent + 2)
            )
        }
        TermKind::BinOp(op) => format!("{}BinOp({})", pad, op.op),
        TermKind::UnOp(op) => format!("{}UnOp({})", pad, op.op),
        TermKind::Let { name, rhs, body, .. } => {
            format!(
                "{}Let {} =\n{}\n{}in\n{}",
                pad,
                name,
                print_term(rhs, indent + 1),
                pad,
                print_term(body, indent + 1)
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
                print_term(cond, indent + 1),
                pad,
                print_term(then_branch, indent + 1),
                pad,
                print_term(else_branch, indent + 1)
            )
        }
        TermKind::Loop {
            loop_var,
            init,
            kind,
            body,
            ..
        } => {
            let kind_str = match kind {
                LoopKind::For { var, .. } => format!("for {} in ...", var),
                LoopKind::ForRange { var, .. } => format!("for {} < ...", var),
                LoopKind::While { .. } => "while ...".to_string(),
            };
            format!(
                "{}Loop {} = {} ({})\n{}body:\n{}",
                pad,
                loop_var,
                print_term(init, 0).trim(),
                kind_str,
                pad,
                print_term(body, indent + 1)
            )
        }
        TermKind::Extern(linkage) => format!("{}Extern(\"{}\")", pad, linkage),
    }
}

/// Helper to print all defs in a program
#[allow(dead_code)]
fn print_program(program: &Program) -> String {
    let mut out = String::new();
    for def in &program.defs {
        out.push_str(&format!("\n=== {} (arity {}) ===\n", def.name, def.arity));
        out.push_str(&print_term(&def.body, 0));
        out.push('\n');
    }
    out
}

#[test]
fn test_defunc_simple_lambda_no_capture() {
    // def f = |x| x
    let mut ids = TermIdSource::new();

    let lam = Term {
        id: ids.next_id(),
        ty: arrow(i32_ty(), i32_ty()),
        span: dummy_span(),
        kind: TermKind::Lam {
            param: "x".to_string(),
            param_ty: i32_ty(),
            body: Box::new(Term {
                id: ids.next_id(),
                ty: i32_ty(),
                span: dummy_span(),
                kind: TermKind::Var("x".to_string()),
            }),
        },
    };

    let program = Program {
        defs: vec![Def {
            name: "f".to_string(),
            ty: lam.ty.clone(),
            body: lam,
            meta: DefMeta::Function,
            arity: 1,
        }],
        uniforms: vec![],
        storage: vec![],
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
    let mut ids = TermIdSource::new();

    // Build: |x| x + y  (where y is free)
    let inner_lam = Term {
        id: ids.next_id(),
        ty: arrow(i32_ty(), i32_ty()),
        span: dummy_span(),
        kind: TermKind::Lam {
            param: "x".to_string(),
            param_ty: i32_ty(),
            body: Box::new(Term {
                id: ids.next_id(),
                ty: i32_ty(),
                span: dummy_span(),
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: ids.next_id(),
                        ty: arrow(i32_ty(), arrow(i32_ty(), i32_ty())),
                        span: dummy_span(),
                        kind: TermKind::BinOp(BinaryOp { op: "+".to_string() }),
                    }),
                    arg: Box::new(Term {
                        id: ids.next_id(),
                        ty: i32_ty(),
                        span: dummy_span(),
                        kind: TermKind::Var("y".to_string()),
                    }),
                },
            }),
        },
    };

    // let g = inner_lam in g
    let let_expr = Term {
        id: ids.next_id(),
        ty: arrow(i32_ty(), i32_ty()),
        span: dummy_span(),
        kind: TermKind::Let {
            name: "g".to_string(),
            name_ty: arrow(i32_ty(), i32_ty()),
            rhs: Box::new(inner_lam),
            body: Box::new(Term {
                id: ids.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span: dummy_span(),
                kind: TermKind::Var("g".to_string()),
            }),
        },
    };

    // |y| let_expr
    let outer_lam = Term {
        id: ids.next_id(),
        ty: arrow(i32_ty(), arrow(i32_ty(), i32_ty())),
        span: dummy_span(),
        kind: TermKind::Lam {
            param: "y".to_string(),
            param_ty: i32_ty(),
            body: Box::new(let_expr),
        },
    };

    let program = Program {
        defs: vec![Def {
            name: "f".to_string(),
            ty: outer_lam.ty.clone(),
            body: outer_lam,
            meta: DefMeta::Function,
            arity: 1,
        }],
        uniforms: vec![],
        storage: vec![],
    };

    let known_defs = HashSet::new();
    let result = defunctionalize(program, &known_defs);

    // Should have lifted the inner lambda
    assert!(result.defs.len() >= 2, "Expected lifted lambda def");

    // Find the lifted lambda
    let lifted = result.defs.iter().find(|d| d.name.starts_with("_lambda_"));
    assert!(lifted.is_some(), "Should have a _lambda_ definition");
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
/// When hof_outer is specialized for _lambda_0:
/// - hof_outer$0 body becomes: hof_inner(_lambda_0, y)
/// - This should trigger specialization of hof_inner for _lambda_0
///
/// This test fails if specialized bodies aren't processed (fixpoint issue).
#[test]
fn test_nested_hof_passthrough() {
    let mut ids = TermIdSource::new();
    let span = dummy_span();

    // hof_inner = |f| |x| f(x)
    let hof_inner_body = Term {
        id: ids.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::App {
            func: Box::new(Term {
                id: ids.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span,
                kind: TermKind::Var("f".to_string()),
            }),
            arg: Box::new(Term {
                id: ids.next_id(),
                ty: i32_ty(),
                span,
                kind: TermKind::Var("x".to_string()),
            }),
        },
    };

    let hof_inner = Term {
        id: ids.next_id(),
        ty: arrow(arrow(i32_ty(), i32_ty()), arrow(i32_ty(), i32_ty())),
        span,
        kind: TermKind::Lam {
            param: "f".to_string(),
            param_ty: arrow(i32_ty(), i32_ty()),
            body: Box::new(Term {
                id: ids.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span,
                kind: TermKind::Lam {
                    param: "x".to_string(),
                    param_ty: i32_ty(),
                    body: Box::new(hof_inner_body),
                },
            }),
        },
    };

    // hof_outer = |g| |y| hof_inner g y
    // Just passes g directly to hof_inner, no lambda wrapping
    let hof_inner_call = Term {
        id: ids.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::App {
            func: Box::new(Term {
                id: ids.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span,
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: ids.next_id(),
                        ty: arrow(arrow(i32_ty(), i32_ty()), arrow(i32_ty(), i32_ty())),
                        span,
                        kind: TermKind::Var("hof_inner".to_string()),
                    }),
                    arg: Box::new(Term {
                        id: ids.next_id(),
                        ty: arrow(i32_ty(), i32_ty()),
                        span,
                        kind: TermKind::Var("g".to_string()), // <-- Just passes g, no lambda
                    }),
                },
            }),
            arg: Box::new(Term {
                id: ids.next_id(),
                ty: i32_ty(),
                span,
                kind: TermKind::Var("y".to_string()),
            }),
        },
    };

    let hof_outer = Term {
        id: ids.next_id(),
        ty: arrow(arrow(i32_ty(), i32_ty()), arrow(i32_ty(), i32_ty())),
        span,
        kind: TermKind::Lam {
            param: "g".to_string(),
            param_ty: arrow(i32_ty(), i32_ty()),
            body: Box::new(Term {
                id: ids.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span,
                kind: TermKind::Lam {
                    param: "y".to_string(),
                    param_ty: i32_ty(),
                    body: Box::new(hof_inner_call),
                },
            }),
        },
    };

    // main = |cap| hof_outer (|a| a + cap) 5
    let a_plus_cap = Term {
        id: ids.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::App {
            func: Box::new(Term {
                id: ids.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span,
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: ids.next_id(),
                        ty: arrow(i32_ty(), arrow(i32_ty(), i32_ty())),
                        span,
                        kind: TermKind::BinOp(BinaryOp { op: "+".to_string() }),
                    }),
                    arg: Box::new(Term {
                        id: ids.next_id(),
                        ty: i32_ty(),
                        span,
                        kind: TermKind::Var("a".to_string()),
                    }),
                },
            }),
            arg: Box::new(Term {
                id: ids.next_id(),
                ty: i32_ty(),
                span,
                kind: TermKind::Var("cap".to_string()),
            }),
        },
    };

    let capturing_lambda = Term {
        id: ids.next_id(),
        ty: arrow(i32_ty(), i32_ty()),
        span,
        kind: TermKind::Lam {
            param: "a".to_string(),
            param_ty: i32_ty(),
            body: Box::new(a_plus_cap),
        },
    };

    let main_body = Term {
        id: ids.next_id(),
        ty: i32_ty(),
        span,
        kind: TermKind::App {
            func: Box::new(Term {
                id: ids.next_id(),
                ty: arrow(i32_ty(), i32_ty()),
                span,
                kind: TermKind::App {
                    func: Box::new(Term {
                        id: ids.next_id(),
                        ty: arrow(arrow(i32_ty(), i32_ty()), arrow(i32_ty(), i32_ty())),
                        span,
                        kind: TermKind::Var("hof_outer".to_string()),
                    }),
                    arg: Box::new(capturing_lambda),
                },
            }),
            arg: Box::new(Term {
                id: ids.next_id(),
                ty: i32_ty(),
                span,
                kind: TermKind::IntLit("5".to_string()),
            }),
        },
    };

    let main_def = Term {
        id: ids.next_id(),
        ty: arrow(i32_ty(), i32_ty()),
        span,
        kind: TermKind::Lam {
            param: "cap".to_string(),
            param_ty: i32_ty(),
            body: Box::new(main_body),
        },
    };

    let program = Program {
        defs: vec![
            Def {
                name: "hof_inner".to_string(),
                ty: hof_inner.ty.clone(),
                body: hof_inner,
                meta: DefMeta::Function,
                arity: 2,
            },
            Def {
                name: "hof_outer".to_string(),
                ty: hof_outer.ty.clone(),
                body: hof_outer,
                meta: DefMeta::Function,
                arity: 2,
            },
            Def {
                name: "main".to_string(),
                ty: main_def.ty.clone(),
                body: main_def,
                meta: DefMeta::Function,
                arity: 1,
            },
        ],
        uniforms: vec![],
        storage: vec![],
    };

    let known_defs = HashSet::new();
    let result = defunctionalize(program, &known_defs);

    let def_names: Vec<&str> = result.defs.iter().map(|d| d.name.as_str()).collect();
    eprintln!(
        "Defs after defunctionalization (passthrough test): {:?}",
        def_names
    );

    // Expected:
    // - hof_inner (original)
    // - hof_outer (original)
    // - main (original)
    // - _lambda_0 (lifted from main, captures cap)
    // - hof_outer$0 (specialized for _lambda_0)
    // - hof_inner$0 (specialized for _lambda_0, triggered by processing hof_outer$0's body)
    //
    // The key assertion: hof_inner MUST be specialized even though the original
    // hof_outer doesn't have a lambda - the specialized body hof_inner(_lambda_0, y)
    // contains a HOF call that needs specialization.

    let hof_outer_specialized = result.defs.iter().any(|d| d.name.starts_with("hof_outer$"));
    assert!(
        hof_outer_specialized,
        "hof_outer should be specialized. Defs: {:?}",
        def_names
    );

    // THIS IS THE CRITICAL ASSERTION:
    // hof_inner must be specialized because hof_outer$0's body is: hof_inner(_lambda_0, y)
    // If we don't process hof_outer$0's body (fixpoint issue), hof_inner won't be specialized
    let hof_inner_specialized = result.defs.iter().any(|d| d.name.starts_with("hof_inner$"));
    assert!(
        hof_inner_specialized,
        "hof_inner should be specialized (from processing hof_outer$0's body). Defs: {:?}",
        def_names
    );
}
