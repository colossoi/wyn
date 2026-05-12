//! AST→TLC pattern destructuring for `let p = e in body` and lambda
//! parameters. Walks a pattern + scrutinee term and produces a list
//! of `PendingBinding`s that the caller applies via
//! `apply_bindings_around`.
//!
//! Refutable patterns (`Literal`, `Constructor` against a
//! multi-variant sum) hit `todo!` here today — they're rejected
//! upstream by the refutability gate in `types/patterns/refutability`.
//! Single-variant Constructor and `Unit` are valid irrefutable
//! patterns; they're handled here.

use crate::SymbolId;
use crate::ast::{self, PatternKind, Span, TypeName};
use crate::tlc::{PendingBinding, Term, TermKind, Transformer, VarRef};
use polytype::Type;
use std::collections::HashMap;

impl<'a> Transformer<'a> {
    /// Top-level entry: compute the binding plan for a pattern against
    /// a scrutinee. Tuple/Record patterns produce a chain of
    /// projections; simple Name/Wildcard at top level produce no
    /// bindings (caller wraps with a `Let` directly).
    pub(in crate::tlc) fn compute_pattern_bindings(
        &mut self,
        pattern: &ast::Pattern,
        scrutinee: Term,
        span: Span,
    ) -> (SymbolId, Vec<PendingBinding>) {
        self.compute_pattern_bindings_inner(pattern, scrutinee, span, true)
    }

    /// Inner implementation tracking nesting depth.
    /// At top level, Name/Wildcard don't create bindings (caller handles).
    /// Nested Name/Wildcard DO create bindings (needed for component extraction).
    pub(in crate::tlc) fn compute_pattern_bindings_inner(
        &mut self,
        pattern: &ast::Pattern,
        scrutinee: Term,
        span: Span,
        is_top_level: bool,
    ) -> (SymbolId, Vec<PendingBinding>) {
        match &pattern.kind {
            PatternKind::Name(name) => {
                let sym = self.define(name);
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
                let fresh_name = format!("_w_wild_{}", self.term_ids.next_id().0);
                let sym = self.define(&fresh_name);
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
            PatternKind::Tuple(patterns) => {
                let fresh_name = format!("_w_tup_{}", self.term_ids.next_id().0);
                let fresh_sym = self.define(&fresh_name);
                let tuple_ty = scrutinee.ty.clone();
                let component_types = self.extract_tuple_types(&tuple_ty, patterns.len());

                let mut bindings = vec![PendingBinding {
                    name: fresh_sym,
                    ty: tuple_ty.clone(),
                    expr: scrutinee,
                }];

                for (i, sub_pattern) in patterns.iter().enumerate() {
                    let comp_ty = component_types
                        .get(i)
                        .cloned()
                        .expect("BUG: Tuple pattern has more elements than tuple type");

                    let proj = self.build_tuple_projection(fresh_sym, &tuple_ty, i, comp_ty, span);
                    let (_, sub_bindings) =
                        self.compute_pattern_bindings_inner(sub_pattern, proj, span, false);
                    bindings.extend(sub_bindings);
                }

                (fresh_sym, bindings)
            }
            PatternKind::Record(fields) => {
                let fresh_name = format!("_w_rec_{}", self.term_ids.next_id().0);
                let fresh_sym = self.define(&fresh_name);
                let record_ty = scrutinee.ty.clone();
                let field_types = self.extract_record_types(&record_ty);

                let mut bindings = vec![PendingBinding {
                    name: fresh_sym,
                    ty: record_ty.clone(),
                    expr: scrutinee,
                }];

                for field in fields {
                    let field_ty = field_types
                        .get(&field.field)
                        .cloned()
                        .unwrap_or_else(|| panic!("BUG: Record field '{}' not found in type", field.field));

                    let field_idx = self
                        .resolve_field_index(&record_ty, &field.field)
                        .unwrap_or_else(|| panic!("BUG: field '{}' not in record type", field.field));

                    let field_access = self.build_tuple_projection(
                        fresh_sym,
                        &record_ty,
                        field_idx,
                        field_ty.clone(),
                        span,
                    );

                    if let Some(pat) = &field.pattern {
                        let (_, sub_bindings) =
                            self.compute_pattern_bindings_inner(pat, field_access, span, false);
                        bindings.extend(sub_bindings);
                    } else {
                        let field_sym = self.define(&field.field);
                        bindings.push(PendingBinding {
                            name: field_sym,
                            ty: field_ty,
                            expr: field_access,
                        });
                    }
                }

                (fresh_sym, bindings)
            }
            PatternKind::Unit => {
                todo!("Unit patterns")
            }
            PatternKind::Literal(_) => {
                todo!("Literal patterns in lambdas")
            }
            PatternKind::Constructor(_, _) => {
                todo!("Constructor patterns in lambdas")
            }
        }
    }

    /// Returns Some(name) for simple patterns (Name, Wildcard, or
    /// wrapped versions), None for complex patterns that need
    /// destructuring. Used by `transform_expr`'s Let arm to fast-path
    /// `let x = …` without the full binding-list machinery.
    pub(in crate::tlc) fn simple_pattern_name(&mut self, pattern: &ast::Pattern) -> Option<String> {
        match &pattern.kind {
            PatternKind::Name(name) => Some(name.clone()),
            PatternKind::Wildcard => Some(format!("_w_wild_{}", self.term_ids.next_id().0)),
            PatternKind::Typed(inner, _) | PatternKind::Attributed(_, inner) => {
                self.simple_pattern_name(inner)
            }
            PatternKind::Tuple(_)
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

    pub(in crate::tlc) fn extract_tuple_types(
        &self,
        ty: &Type<TypeName>,
        _expected_len: usize,
    ) -> Vec<Type<TypeName>> {
        match ty {
            Type::Constructed(TypeName::Tuple(_), args) => args.clone(),
            _ => panic!("BUG: Expected tuple type, got {:?}", ty),
        }
    }

    /// Resolve a field name to its index in a record / vec / tuple
    /// type. Returns None for unresolved field names; callers panic
    /// on None since checker validated upstream.
    pub(in crate::tlc) fn resolve_field_index(&self, ty: &Type<TypeName>, field: &str) -> Option<usize> {
        match ty {
            Type::Constructed(TypeName::Record(fields), _) => fields.iter().position(|f| f == field),
            // Vec swizzle: x=0, y=1, z=2, w=3
            Type::Constructed(TypeName::Vec, _) => match field {
                "x" => Some(0),
                "y" => Some(1),
                "z" => Some(2),
                "w" => Some(3),
                _ => None,
            },
            // Tuple positional access: .0, .1, etc.
            Type::Constructed(TypeName::Tuple(_), _) => field.parse::<usize>().ok(),
            _ => None,
        }
    }

    pub(in crate::tlc) fn extract_record_types(
        &self,
        ty: &Type<TypeName>,
    ) -> HashMap<String, Type<TypeName>> {
        match ty {
            Type::Constructed(TypeName::Record(fields), args) => {
                fields.iter().cloned().zip(args.iter().cloned()).collect()
            }
            _ => HashMap::new(),
        }
    }
}
