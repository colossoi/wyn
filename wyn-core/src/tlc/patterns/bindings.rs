//! AST→TLC pattern destructuring for `let p = e in body` and lambda
//! parameters. Walks a pattern + scrutinee term and produces a list
//! of `PendingBinding`s that the caller applies via
//! `apply_bindings_around`.
//!
//! Refutable patterns (`Literal`, `Constructor` against a
//! multi-variant sum) are rejected upstream during pattern binding.
//! Single-variant Constructor and `Unit` are valid irrefutable
//! patterns; they're handled here.

use crate::ast::{self, PatternKind, Span, TypeName};
use crate::tlc::{PendingBinding, Term, TermKind, Transformer, VarRef};
use crate::SymbolId;
use polytype::Type;

impl<'a> Transformer<'a> {
    /// Top-level entry: compute the binding plan for a pattern against
    /// a scrutinee. Compound patterns produce a chain of
    /// projections; simple Name/Wildcard at top level produce no
    /// bindings (caller wraps with a `Let` directly).
    pub(in crate::tlc) fn compute_pattern_bindings<A>(
        &mut self,
        pattern: &ast::Pattern<ast::HolesResolvedTree, A>,
        scrutinee: Term,
        span: Span,
    ) -> (SymbolId, Vec<PendingBinding>) {
        self.compute_pattern_bindings_inner(pattern, scrutinee, span, true)
    }

    /// Inner implementation tracking nesting depth.
    /// At top level, Name/Wildcard don't create bindings (caller handles).
    /// Nested Name/Wildcard DO create bindings (needed for component extraction).
    pub(in crate::tlc) fn compute_pattern_bindings_inner<A>(
        &mut self,
        pattern: &ast::Pattern<ast::HolesResolvedTree, A>,
        scrutinee: Term,
        span: Span,
        is_top_level: bool,
    ) -> (SymbolId, Vec<PendingBinding>) {
        match &pattern.kind {
            PatternKind::Name(name) => {
                let sym = name.symbol;
                if is_top_level {
                    (sym, vec![])
                } else {
                    let binding = PendingBinding {
                        name: sym,
                        ty: scrutinee.ty.clone(),
                        expr: scrutinee,
                    };
                    (sym, vec![binding])
                }
            }
            PatternKind::Wildcard => {
                let fresh_name = format!("_w_wild_{}", self.term_ids.next_id());
                let sym = self.fresh(&fresh_name);
                if is_top_level {
                    (sym, vec![])
                } else {
                    let binding = PendingBinding {
                        name: sym,
                        ty: scrutinee.ty.clone(),
                        expr: scrutinee,
                    };
                    (sym, vec![binding])
                }
            }
            PatternKind::Typed(inner, _) | PatternKind::Attributed(_, inner) => {
                self.compute_pattern_bindings_inner(inner, scrutinee, span, is_top_level)
            }
            PatternKind::Tuple(patterns)
            | PatternKind::Vec(patterns)
            | PatternKind::Constructor(_, patterns) => {
                let fresh_name = format!("_w_pattern_{}", self.term_ids.next_id());
                let fresh_sym = self.fresh(&fresh_name);
                let scrutinee_ty = scrutinee.ty.clone();
                let mut bindings = vec![PendingBinding {
                    name: fresh_sym,
                    ty: scrutinee_ty.clone(),
                    expr: scrutinee,
                }];
                // Irrefutable constructors have one variant, with payload after the tag.
                let offset = usize::from(matches!(&pattern.kind, PatternKind::Constructor(..)));
                for (i, sub_pattern) in patterns.iter().enumerate() {
                    let proj = self.build_tuple_projection(
                        fresh_sym,
                        &scrutinee_ty,
                        offset + i,
                        Self::type_of(&sub_pattern.h),
                        span,
                    );
                    let (_, sub_bindings) =
                        self.compute_pattern_bindings_inner(sub_pattern, proj, span, false);
                    bindings.extend(sub_bindings);
                }
                (fresh_sym, bindings)
            }
            PatternKind::Unit => {
                // Single-inhabitant type: bind a fresh symbol to the
                // scrutinee so any downstream `Let` chain has somewhere
                // to anchor. No sub-bindings.
                let fresh_name = format!("_w_unit_{}", self.term_ids.next_id());
                let sym = self.fresh(&fresh_name);
                if is_top_level {
                    (sym, vec![])
                } else {
                    let binding = PendingBinding {
                        name: sym,
                        ty: scrutinee.ty.clone(),
                        expr: scrutinee,
                    };
                    (sym, vec![binding])
                }
            }
            PatternKind::Literal(_) | PatternKind::Record(_) => unreachable!(),
        }
    }

    /// Returns Some(name) for simple patterns (Name, Wildcard, or
    /// wrapped versions), None for complex patterns that need
    /// destructuring. Used by `transform_expr`'s Let arm to fast-path
    /// `let x = …` without the full binding-list machinery.
    pub(in crate::tlc) fn simple_pattern_symbol<A>(
        &mut self,
        pattern: &ast::Pattern<ast::HolesResolvedTree, A>,
    ) -> Option<SymbolId> {
        match &pattern.kind {
            PatternKind::Name(binding) => Some(binding.symbol),
            PatternKind::Wildcard => {
                let id = self.term_ids.next_id();
                Some(self.fresh(format!("_w_wild_{id}")))
            }
            PatternKind::Typed(inner, _) | PatternKind::Attributed(_, inner) => {
                self.simple_pattern_symbol(inner)
            }
            PatternKind::Tuple(_)
            | PatternKind::Vec(_)
            | PatternKind::Record(_)
            | PatternKind::Unit
            | PatternKind::Literal(_)
            | PatternKind::Constructor(_, _) => None,
        }
    }

    /// Build a `TupleProj` of `var_sym.idx` typed `result_ty`.
    pub(in crate::tlc) fn build_tuple_projection(
        &mut self,
        var_sym: SymbolId,
        var_ty: &Type<TypeName>,
        index: usize,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let var_term = self.mk_term(var_ty.clone(), span, TermKind::Var(VarRef::Symbol(var_sym)));
        self.mk_tuple_proj(var_term, index, result_ty, span)
    }
}
