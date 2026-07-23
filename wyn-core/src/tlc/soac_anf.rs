//! Normalize nested SOAC expressions into flat let chains.
//!
//! This is a source-shape pass, not a fusion pass: EGIR still makes every
//! producer/consumer decision. The flat form ensures TLC-to-EGIR conversion
//! emits each SOAC as an explicit side effect and therefore preserves the
//! semantic value edges that EGIR needs.

use super::{Def, Program, Term, TermIdSource, TermKind, VarRef};
use crate::SymbolTable;

pub fn run(program: &mut Program) {
    let term_ids = &mut program.term_ids;
    crate::map_in_place(&mut program.defs, |def| {
        let body = normalize_term(def.body, &mut program.symbols, term_ids);
        Def { body, ..def }
    });
    debug_assert!(
        verify_flattened(program).is_ok(),
        "SOAC ANF normalization left a nested let rhs"
    );
}

fn normalize_term(term: Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> Term {
    let term = term.map_children(&mut |child| normalize_term(child, symbols, term_ids));
    if let TermKind::App { args, .. } = &term.kind {
        if args.iter().any(|arg| matches!(arg.kind, TermKind::Soac(_))) {
            let TermKind::App { func, args } = term.kind else {
                unreachable!();
            };
            let span = term.span;
            let mut new_args = Vec::with_capacity(args.len());
            let mut bindings = Vec::new();
            for arg in args {
                if matches!(arg.kind, TermKind::Soac(_)) {
                    let name = symbols.alloc("_anf".to_string());
                    let ty = arg.ty.clone();
                    bindings.push((name, arg));
                    new_args.push(Term {
                        id: term_ids.next_id(),
                        ty,
                        span,
                        kind: TermKind::Var(VarRef::Symbol(name)),
                    });
                } else {
                    new_args.push(arg);
                }
            }
            let mut result = Term {
                id: term_ids.next_id(),
                ty: term.ty,
                span,
                kind: TermKind::App { func, args: new_args },
            };
            for (name, rhs) in bindings.into_iter().rev() {
                let name_ty = rhs.ty.clone();
                result = Term {
                    id: term_ids.next_id(),
                    ty: result.ty.clone(),
                    span,
                    kind: TermKind::Let {
                        name,
                        name_ty,
                        rhs: Box::new(rhs),
                        body: Box::new(result),
                    },
                };
            }
            return result;
        }
    }
    flatten_let(term, term_ids)
}

fn flatten_let(term: Term, term_ids: &mut TermIdSource) -> Term {
    let TermKind::Let {
        name,
        name_ty,
        rhs,
        body,
    } = term.kind
    else {
        return term;
    };
    if !matches!(rhs.kind, TermKind::Let { .. }) {
        return Term {
            kind: TermKind::Let {
                name,
                name_ty,
                rhs,
                body: Box::new(flatten_let(*body, term_ids)),
            },
            ..term
        };
    }
    let TermKind::Let {
        name: inner_name,
        name_ty: inner_ty,
        rhs: inner_rhs,
        body: inner_body,
    } = rhs.kind
    else {
        unreachable!();
    };
    let nested = Term {
        id: term_ids.next_id(),
        ty: term.ty.clone(),
        span: term.span,
        kind: TermKind::Let {
            name,
            name_ty,
            rhs: inner_body,
            body,
        },
    };
    flatten_let(
        Term {
            id: term_ids.next_id(),
            ty: term.ty,
            span: term.span,
            kind: TermKind::Let {
                name: inner_name,
                name_ty: inner_ty,
                rhs: inner_rhs,
                body: Box::new(nested),
            },
        },
        term_ids,
    )
}

fn verify_flattened(program: &Program) -> Result<(), ()> {
    fn walk(term: &Term) -> Result<(), ()> {
        if matches!(&term.kind, TermKind::Let { rhs, .. } if matches!(rhs.kind, TermKind::Let { .. })) {
            return Err(());
        }
        let mut result = Ok(());
        term.for_each_child(&mut |child| {
            if result.is_ok() {
                result = walk(child);
            }
        });
        result
    }
    program.defs.iter().try_for_each(|def| walk(&def.body))
}
