//! Shadow-correct symbol substitution over TLC terms.
//!
//! One traversal ([`substitute_with`]) handles binder shadowing for `Let`,
//! `Lambda`, and `Loop` exactly once. Passes supply what a matched `Var(old)`
//! occurrence becomes instead of re-rolling the recursion; symbol renaming is
//! the common specialization below.

use super::{LoopKind, Payload, Term, TermIdSource, TermKind, VarRef};
use crate::SymbolId;

/// Shadow-correct substitution traversal. `make_replacement` receives the
/// matched variable occurrence, allowing callers to preserve its type/span or
/// replace it with arbitrary syntax. Shadowing by `Let`, `Lambda`, and `Loop`
/// binders is handled here once.
pub(crate) fn substitute_with<C, S, F>(
    term: Term<C, S>,
    old: SymbolId,
    make_replacement: &mut F,
    term_ids: &mut TermIdSource,
) -> Term<C, S>
where
    C: Payload,
    S: Payload,
    F: FnMut(Term<C, S>, &mut TermIdSource) -> Term<C, S>,
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
            let rhs = substitute_with(*rhs, old, make_replacement, term_ids);
            let body =
                if name == old { *body } else { substitute_with(*body, old, make_replacement, term_ids) };
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
            let init = substitute_with(*init, old, make_replacement, term_ids);
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
                            substitute_with(extraction, old, make_replacement, term_ids),
                        )
                    }
                })
                .collect();
            let (kind, kind_shadows) = substitute_loop_kind_core(kind, old, make_replacement, term_ids);
            let body = if loop_var == old || init_binding_shadows || kind_shadows {
                *body
            } else {
                substitute_with(*body, old, make_replacement, term_ids)
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
        other => {
            let fresh_id = term_ids.next_id();
            Term { kind: other, ..term }.map_children(fresh_id, &mut |child| {
                substitute_with(child, old, make_replacement, term_ids)
            })
        }
    }
}

fn substitute_loop_kind_core<C, S, F>(
    kind: LoopKind<C, S>,
    old: SymbolId,
    make_replacement: &mut F,
    term_ids: &mut TermIdSource,
) -> (LoopKind<C, S>, bool)
where
    C: Payload,
    S: Payload,
    F: FnMut(Term<C, S>, &mut TermIdSource) -> Term<C, S>,
{
    match kind {
        LoopKind::For { var, var_ty, iter } => (
            LoopKind::For {
                var,
                var_ty,
                iter: Box::new(substitute_with(*iter, old, make_replacement, term_ids)),
            },
            var == old,
        ),
        LoopKind::ForRange { var, var_ty, bound } => (
            LoopKind::ForRange {
                var,
                var_ty,
                bound: Box::new(substitute_with(*bound, old, make_replacement, term_ids)),
            },
            var == old,
        ),
        LoopKind::While { cond } => (
            LoopKind::While {
                cond: Box::new(substitute_with(*cond, old, make_replacement, term_ids)),
            },
            false,
        ),
    }
}

/// Rename all free occurrences of `old` to `new` in a term, respecting
/// shadowing by Let names, Lambda params, and Loop vars. Each renamed
/// occurrence keeps its own type/span (a rename doesn't change the value's
/// type); only the symbol changes.
pub(crate) fn substitute_sym<C: Payload, S: Payload>(
    term: Term<C, S>,
    old: SymbolId,
    new: SymbolId,
    term_ids: &mut TermIdSource,
) -> Term<C, S> {
    substitute_with(
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
