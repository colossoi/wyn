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
//!
//! It also flattens nested `let`-in-`let`-rhs so every binding lives on one
//! top-level chain:
//! ```text
//! let x = (let y = a in b) in rest  =>  let y = a in let x = b in rest
//! ```
//! Lifting a SOAC out of a binding's rhs (e.g. the `reduce` in
//! `let r = reduce(..) * k`) otherwise leaves it one level deep
//! (`let r = (let _anf = reduce(..) in _anf * k)`), where the fusion driver —
//! which only walks the top-level let chain — can't see it to fuse.

use super::VarRef;
use crate::SymbolTable;

use super::{Def, Program, Term, TermIdSource, TermKind};

#[cfg(test)]
#[path = "normalize_tests.rs"]
mod normalize_tests;

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
        symbols,
        ..program
    }
}

/// Bottom-up: recurse into children, then lift SOAC args in App nodes.
fn normalize_term(term: Term, symbols: &mut SymbolTable, term_ids: &mut TermIdSource) -> Term {
    let term = term.map_children(&mut |child| normalize_term(child, symbols, term_ids));

    // After children are normalized, check if this App has any SOAC args to lift.
    if let TermKind::App { ref args, .. } = term.kind {
        if args.iter().any(|a| matches!(a.kind, TermKind::Soac(_))) {
            let TermKind::App { func, args } = term.kind else {
                unreachable!()
            };
            // Wrap the App in let bindings for each SOAC arg (inside-out).
            let span = term.span;
            let mut new_args = Vec::with_capacity(args.len());
            let mut lets: Vec<(crate::SymbolId, Term)> = Vec::new();
            for arg in args {
                if matches!(arg.kind, TermKind::Soac(_)) {
                    let fresh = symbols.alloc("_anf".to_string());
                    let arg_ty = arg.ty.clone();
                    lets.push((fresh, arg));
                    new_args.push(Term {
                        id: term_ids.next_id(),
                        ty: arg_ty,
                        span,
                        kind: TermKind::Var(VarRef::Symbol(fresh)),
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
            for (fresh, soac_arg) in lets.into_iter().rev() {
                let arg_ty = soac_arg.ty.clone();
                result = Term {
                    id: term_ids.next_id(),
                    ty: result.ty.clone(),
                    span,
                    kind: TermKind::Let {
                        name: fresh,
                        name_ty: arg_ty,
                        rhs: Box::new(soac_arg),
                        body: Box::new(result),
                    },
                };
            }
            return result;
        }
    }

    // Flatten `let x = (let y = a in b) in rest` => `let y = a in let x = b in
    // rest`. Children are already normalized (bottom-up), so a binding whose rhs
    // is a `let` came from a SOAC lifted out of that rhs; hoisting it onto the
    // outer chain is what makes the fusion driver (which only scans the
    // top-level let chain) see the producer/consumer edge. Names are fresh, so
    // reordering can't capture.
    flatten_let(term, term_ids)
}

/// Rotate a `let` whose rhs is itself a `let` up onto the enclosing chain,
/// repeating while the new rhs is still a `let` (e.g. several SOACs lifted out
/// of one expression). Non-`let` terms pass through unchanged.
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
                body,
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
        unreachable!("checked rhs is a Let above")
    };
    // `let name = (let inner_name = inner_rhs in inner_body) in body`
    //   => `let inner_name = inner_rhs in let name = inner_body in body`
    let new_inner = Term {
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
    let rotated = Term {
        id: term_ids.next_id(),
        ty: term.ty,
        span: term.span,
        kind: TermKind::Let {
            name: inner_name,
            name_ty: inner_ty,
            rhs: inner_rhs,
            body: Box::new(new_inner),
        },
    };
    flatten_let(rotated, term_ids)
}
