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

    let builtins = HashSet::new();
    let result = defunctionalize(program, &builtins);

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

    let builtins = HashSet::new();
    let result = defunctionalize(program, &builtins);

    // Should have lifted the inner lambda
    assert!(result.defs.len() >= 2, "Expected lifted lambda def");

    // Find the lifted lambda
    let lifted = result.defs.iter().find(|d| d.name.starts_with("_lambda_"));
    assert!(lifted.is_some(), "Should have a _lambda_ definition");
}
