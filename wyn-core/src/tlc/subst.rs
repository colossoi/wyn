//! Shadow-correct symbol substitution over TLC terms.
//!
//! One traversal ([`substitute_core`]) handles binder shadowing for `Let`,
//! `Lambda`, and `Loop` exactly once; the two public entry points differ only
//! in what a matched `Var(old)` occurrence becomes. Passes that rename or
//! replace a symbol use these instead of re-rolling the recursion.

use super::{LoopKind, Term, TermIdSource, TermKind, VarRef};
use crate::SymbolId;

/// Shadow-correct substitution traversal shared by [`substitute_term_expr`]
/// (replace a symbol with an arbitrary term) and [`substitute_sym`] (rename a
/// symbol). The only difference between the two is what a matched occurrence
/// becomes; `make_replacement` produces that, given the matched `Var` term so a
/// rename can preserve the occurrence's own type/span. Shadowing by `Let`,
/// `Lambda`, and `Loop` binders is handled here once, so neither caller can grow
/// a capture bug independently.
fn substitute_core<F>(
    term: Term,
    old: SymbolId,
    make_replacement: &mut F,
    term_ids: &mut TermIdSource,
) -> Term
where
    F: FnMut(Term, &mut TermIdSource) -> Term,
{
    match term.kind {
        TermKind::Var(VarRef::Symbol(sym)) if sym == old => make_replacement(term, term_ids),
        TermKind::Lambda(lam) if lam.params.iter().any(|(param, _)| *param == old) => Term {
            kind: TermKind::Lambda(lam),
            ..term
        },
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let rhs = substitute_core(*rhs, old, make_replacement, term_ids);
            let body =
                if name == old { *body } else { substitute_core(*body, old, make_replacement, term_ids) };
            Term {
                id: term_ids.next_id(),
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(rhs),
                    body: Box::new(body),
                },
                ..term
            }
        }
        TermKind::Loop {
            loop_var,
            loop_var_ty,
            init,
            init_bindings,
            kind,
            body,
        } => {
            let init = substitute_core(*init, old, make_replacement, term_ids);
            let init_binding_shadows = init_bindings.iter().any(|(sym, _, _)| *sym == old);
            let init_bindings = init_bindings
                .into_iter()
                .map(|(sym, ty, extraction)| {
                    if init_binding_shadows {
                        (sym, ty, extraction)
                    } else {
                        (
                            sym,
                            ty,
                            substitute_core(extraction, old, make_replacement, term_ids),
                        )
                    }
                })
                .collect();
            let (kind, kind_shadows) = substitute_loop_kind_core(kind, old, make_replacement, term_ids);
            let body = if loop_var == old || init_binding_shadows || kind_shadows {
                *body
            } else {
                substitute_core(*body, old, make_replacement, term_ids)
            };
            Term {
                id: term_ids.next_id(),
                kind: TermKind::Loop {
                    loop_var,
                    loop_var_ty,
                    init: Box::new(init),
                    init_bindings,
                    kind,
                    body: Box::new(body),
                },
                ..term
            }
        }
        other => Term { kind: other, ..term }
            .map_children(&mut |child| substitute_core(child, old, make_replacement, term_ids)),
    }
}

fn substitute_loop_kind_core<F>(
    kind: LoopKind,
    old: SymbolId,
    make_replacement: &mut F,
    term_ids: &mut TermIdSource,
) -> (LoopKind, bool)
where
    F: FnMut(Term, &mut TermIdSource) -> Term,
{
    match kind {
        LoopKind::For { var, var_ty, iter } => (
            LoopKind::For {
                var,
                var_ty,
                iter: Box::new(substitute_core(*iter, old, make_replacement, term_ids)),
            },
            var == old,
        ),
        LoopKind::ForRange { var, var_ty, bound } => (
            LoopKind::ForRange {
                var,
                var_ty,
                bound: Box::new(substitute_core(*bound, old, make_replacement, term_ids)),
            },
            var == old,
        ),
        LoopKind::While { cond } => (
            LoopKind::While {
                cond: Box::new(substitute_core(*cond, old, make_replacement, term_ids)),
            },
            false,
        ),
    }
}

/// Rename all free occurrences of `old` to `new` in a term, respecting
/// shadowing by Let names, Lambda params, and Loop vars. Each renamed
/// occurrence keeps its own type/span (a rename doesn't change the value's
/// type); only the symbol changes.
pub(crate) fn substitute_sym(
    term: Term,
    old: SymbolId,
    new: SymbolId,
    term_ids: &mut TermIdSource,
) -> Term {
    substitute_core(
        term,
        old,
        &mut |occurrence, ids| Term {
            id: ids.next_id(),
            kind: TermKind::Var(VarRef::Symbol(new)),
            ..occurrence
        },
        term_ids,
    )
}
