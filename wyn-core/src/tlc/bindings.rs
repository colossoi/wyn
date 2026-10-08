//! Construction of ordered bindings within one lexical evaluation scope.
//!
//! Only leading lets are spliced. Branches, loops, and callbacks remain opaque;
//! callers must finish a builder inside the scope in which its terms execute.
//! Moving a producer across such a boundary requires a separate legality check.

use super::{ArrayExpr, Payload, Term, TermIdSource, TermKind, VarRef};
use crate::ast::{Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;

#[derive(Debug, Clone)]
pub(crate) struct LetBinding<C: Payload, S: Payload> {
    pub name: SymbolId,
    pub name_ty: Type<TypeName>,
    pub rhs: Term<C, S>,
    pub span: Span,
}

/// Pending bindings in evaluation order. Each pending RHS is a non-let term.
pub(crate) struct Bindings<C: Payload, S: Payload> {
    pending: Vec<LetBinding<C, S>>,
}

impl<C: Payload, S: Payload> Bindings<C, S> {
    pub fn new() -> Self {
        Self { pending: Vec::new() }
    }

    /// Append a binding, evaluating its leading lets before its final RHS.
    pub fn push(&mut self, binding: LetBinding<C, S>) {
        let rhs = self.append(binding.rhs);
        self.pending.push(LetBinding { rhs, ..binding });
    }

    /// Splice a term's leading bindings into this scope and return its value.
    /// No traversal enters the returned expression's children.
    pub fn append(&mut self, mut term: Term<C, S>) -> Term<C, S> {
        while let TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } = term.kind
        {
            self.push(LetBinding {
                name,
                name_ty,
                rhs: *rhs,
                span: term.span,
            });
            term = *body;
        }
        term
    }

    /// Obtain a reference to a value without duplicating its computation.
    /// An existing variable needs no additional binding.
    pub fn name(
        &mut self,
        term: Term<C, S>,
        hint: &str,
        symbols: &mut SymbolTable,
        ids: &mut TermIdSource,
    ) -> Term<C, S> {
        let term = self.append(term);
        if matches!(term.kind, TermKind::Var(_)) {
            return term;
        }
        let name = symbols.alloc(hint.to_owned());
        let ty = term.ty.clone();
        let span = term.span;
        self.push(LetBinding {
            name,
            name_ty: ty.clone(),
            rhs: term,
            span,
        });
        Term::fresh(ids, ty, span, TermKind::Var(VarRef::Symbol(name)))
    }

    /// Obtain a SOAC input, naming producers while retaining array atoms.
    pub fn input(
        &mut self,
        term: Term<C, S>,
        symbols: &mut SymbolTable,
        ids: &mut TermIdSource,
    ) -> ArrayExpr<C, S> {
        let term = self.append(term);
        if let TermKind::ArrayExpr(array) = term.kind {
            return array;
        }
        let value = self.name(term, "_anf", symbols, ids);
        let TermKind::Var(reference) = value.kind else {
            unreachable!("named value")
        };
        ArrayExpr::Var(reference, value.ty)
    }

    /// Finish this scope. The tail's children and existing bindings are intact.
    pub fn finish(self, mut body: Term<C, S>, ids: &mut TermIdSource) -> Term<C, S> {
        for binding in self.pending.into_iter().rev() {
            body = Term::fresh(
                ids,
                body.ty.clone(),
                binding.span,
                TermKind::Let {
                    name: binding.name,
                    name_ty: binding.name_ty,
                    rhs: Box::new(binding.rhs),
                    body: Box::new(body),
                },
            );
        }
        body
    }
}

/// Materialize bindings whose placement has already been chosen by the caller.
pub(crate) fn wrap_let_bindings<C: Payload, S: Payload>(
    bindings: Vec<LetBinding<C, S>>,
    body: Term<C, S>,
    ids: &mut TermIdSource,
) -> Term<C, S> {
    let mut scope = Bindings::new();
    for binding in bindings {
        scope.push(binding);
    }
    scope.finish(body, ids)
}

/// Normalize an existing let with a let RHS using the same construction rules.
/// Unchanged terms retain their identity; rebuilt bindings get fresh IDs.
pub(super) fn flatten_nested_let<C: Payload, S: Payload>(
    term: Term<C, S>,
    ids: &mut TermIdSource,
) -> (Term<C, S>, bool) {
    if !matches!(&term.kind, TermKind::Let { rhs, .. } if matches!(rhs.kind, TermKind::Let { .. })) {
        return (term, false);
    }
    let mut bindings = Bindings::new();
    let body = bindings.append(term);
    (bindings.finish(body, ids), true)
}

#[cfg(test)]
#[path = "bindings_tests.rs"]
mod tests;
