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

use super::{Def, Program, Term, TermIdSource, TermKind};

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
        def_syms: program.def_syms,
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
                        kind: TermKind::Var(crate::tlc::VarRef::Symbol(fresh)),
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

    term
}
