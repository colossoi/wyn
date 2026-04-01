//! ANF-ish normalization pass for TLC.
//!
//! Lifts nested SOAC expressions out of non-let positions into let bindings,
//! so that all SOACs appear as the RHS of a let or as the tail expression
//! of a function body. This makes subsequent summary extraction and
//! interprocedural fusion reliable.
//!
//! Example:
//! ```text
//! f(map(g, xs))  =>  let tmp = map(g, xs) in f(tmp)
//! ```

use crate::SymbolTable;

use super::{Def, Lambda, Program, SoacOp, Term, TermIdSource, TermKind};

/// Normalize a TLC program into ANF-ish form for fusion analysis.
pub fn normalize(program: Program) -> Program {
    let mut symbols = program.symbols;
    let mut term_ids = TermIdSource::new();

    let defs = program
        .defs
        .into_iter()
        .map(|def| {
            let body = normalize_term(def.body, &mut symbols, &mut term_ids);
            Def { body, ..def }
        })
        .collect();

    Program {
        defs,
        uniforms: program.uniforms,
        storage: program.storage,
        symbols,
    }
}

/// Normalize a single term, lifting nested SOACs into let bindings.
fn normalize_term(term: Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> Term {
    // Bottom-up: normalize children first, then check if this term needs lifting.
    let term = normalize_children(term, symbols, term_ids);
    term
}

/// Recursively normalize all children of a term.
fn normalize_children(term: Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> Term {
    let kind = match term.kind {
        TermKind::App { func, arg } => {
            let func = Box::new(normalize_term(*func, symbols, term_ids));
            let arg = Box::new(normalize_term(*arg, symbols, term_ids));

            // If arg is a SOAC, lift it into a let binding
            if is_soac(&arg) {
                let fresh = symbols.alloc("_anf".to_string());
                let arg_ty = arg.ty.clone();
                let span = arg.span;
                return Term {
                    id: term_ids.next_id(),
                    ty: term.ty.clone(),
                    span: term.span,
                    kind: TermKind::Let {
                        name: fresh,
                        name_ty: arg_ty.clone(),
                        rhs: arg,
                        body: Box::new(Term {
                            id: term_ids.next_id(),
                            ty: term.ty.clone(),
                            span,
                            kind: TermKind::App {
                                func,
                                arg: Box::new(Term {
                                    id: term_ids.next_id(),
                                    ty: arg_ty,
                                    span,
                                    kind: TermKind::Var(fresh),
                                }),
                            },
                        }),
                    },
                };
            }

            TermKind::App { func, arg }
        }

        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let rhs = Box::new(normalize_term(*rhs, symbols, term_ids));
            let body = Box::new(normalize_term(*body, symbols, term_ids));
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            }
        }

        TermKind::Lambda(lam) => {
            let body = Box::new(normalize_term(*lam.body, symbols, term_ids));
            TermKind::Lambda(Lambda { body, ..lam })
        }

        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let cond = Box::new(normalize_term(*cond, symbols, term_ids));
            let then_branch = Box::new(normalize_term(*then_branch, symbols, term_ids));
            let else_branch = Box::new(normalize_term(*else_branch, symbols, term_ids));
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            }
        }

        TermKind::Soac(soac) => TermKind::Soac(normalize_soac(soac, symbols, term_ids)),

        // Leaf terms and terms we don't need to normalize
        other => other,
    };

    Term { kind, ..term }
}

/// Normalize SOAC subterms (lambda bodies and inputs).
fn normalize_soac(soac: SoacOp, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> SoacOp {
    match soac {
        SoacOp::Map { lam, inputs } => {
            let body = Box::new(normalize_term(*lam.body, symbols, term_ids));
            SoacOp::Map {
                lam: Lambda { body, ..lam },
                inputs,
            }
        }
        SoacOp::Reduce { op, ne, input, props } => {
            let body = Box::new(normalize_term(*op.body, symbols, term_ids));
            let ne = Box::new(normalize_term(*ne, symbols, term_ids));
            SoacOp::Reduce {
                op: Lambda { body, ..op },
                ne,
                input,
                props,
            }
        }
        SoacOp::Scan { op, ne, input } => {
            let body = Box::new(normalize_term(*op.body, symbols, term_ids));
            let ne = Box::new(normalize_term(*ne, symbols, term_ids));
            SoacOp::Scan {
                op: Lambda { body, ..op },
                ne,
                input,
            }
        }
        SoacOp::Filter { pred, input } => {
            let body = Box::new(normalize_term(*pred.body, symbols, term_ids));
            SoacOp::Filter {
                pred: Lambda { body, ..pred },
                input,
            }
        }
        other => other,
    }
}

/// Check if a term is a SOAC expression.
fn is_soac(term: &Term) -> bool {
    matches!(term.kind, TermKind::Soac(_))
}
