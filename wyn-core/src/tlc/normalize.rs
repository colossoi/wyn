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
pub fn normalize(program: &mut Program) {
    let mut term_ids = TermIdSource::new();

    crate::map_in_place(&mut program.defs, |def| {
        let body = normalize_term(def.body, &mut program.symbols, &mut term_ids);
        Def { body, ..def }
    });

    debug_assert!(
        verify_flattened(program).is_ok(),
        "normalize postcondition: {}",
        verify_flattened(program).unwrap_err()
    );
}

/// `debug_assert` the flat-chain invariant at a pass boundary (no-op in
/// release). Sprinkle at the end of any pass that fusion/parallelize rely on to
/// see producer/consumer edges: a violation localizes exactly which pass buried
/// an edge inside a binding's rhs.
pub fn debug_check_flattened(program: &Program, stage: &'static str) {
    debug_assert!(
        verify_flattened(program).is_ok(),
        "flat-chain invariant violated after {}: {}",
        stage,
        verify_flattened(program).err().unwrap_or("")
    );
}

/// Postcondition of `normalize`: every binding lives on a single flat chain, so
/// no `let`'s rhs is itself a `let`. The chain-walking fusion driver only sees
/// producer/consumer edges that sit on one chain, so a surviving `let x = (let y
/// = .. in ..)` silently hides a fusion (the bug this guards against). A
/// `debug_assert` only — in a correct pass it never fires.
pub fn verify_flattened(program: &Program) -> Result<(), &'static str> {
    fn walk(t: &Term) -> Result<(), &'static str> {
        if let TermKind::Let { rhs, .. } = &t.kind {
            if matches!(rhs.kind, TermKind::Let { .. }) {
                return Err("a let binding has a let-expression as its rhs (chain not flattened)");
            }
        }
        let mut result = Ok(());
        t.for_each_child(&mut |c| {
            if result.is_ok() {
                result = walk(c);
            }
        });
        result
    }
    for def in &program.defs {
        walk(&def.body)?;
    }
    Ok(())
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
/// of one expression). Once a binding's rhs is non-`let`, the chain continues
/// in its body, so flatten that too — a rotation can leave a freshly nested
/// `let` deeper in the body (`let s2 = (let _m = … in scan _m)`), and the fusion
/// driver only sees producer/consumer edges on one flat top-level chain.
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
