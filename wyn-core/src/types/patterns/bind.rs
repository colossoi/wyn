//! Type-checking pattern bindings. Extends `TypeChecker` with
//! `fresh_type_for_pattern` (used by lambda parameter inference) and
//! `bind_pattern` (used by `let` and `match` arms).

use super::refutability::is_irrefutable;
use crate::ast::{Pattern, PatternKind, PatternLiteral};
use crate::error::CompilerError;
use crate::scope::IdentifierKind;
use crate::types::checker::TypeChecker;
use crate::types::{bool_type, f32, i32, tuple, vec, Type, TypeName, TypeScheme};
use crate::{bail_type_at, err_type_at};

type Result<T> = std::result::Result<T, CompilerError>;

impl<'a> TypeChecker<'a> {
    /// Create a fresh type for a pattern based on its structure.
    /// Tuple patterns produce a tuple of fresh variables; everything
    /// else produces a single fresh variable (or the annotation, if
    /// the pattern is `Typed`).
    pub(crate) fn fresh_type_for_pattern(&mut self, pattern: &Pattern) -> Result<Type> {
        Ok(match &pattern.kind {
            PatternKind::Tuple(patterns) => {
                let elem_types: Result<Vec<Type>> =
                    patterns.iter().map(|p| self.fresh_type_for_pattern(p)).collect();
                tuple(elem_types?)
            }
            PatternKind::Vec(patterns) => {
                // All sub-patterns share the same scalar element type;
                // make a fresh tvar and a vec of the pattern's arity.
                let elem = self.context.new_variable();
                vec(patterns.len(), elem)
            }
            PatternKind::Typed(_, annotated_type) => {
                self.normalize_annotation_type(annotated_type, self.current_module.as_deref())?
            }
            PatternKind::Attributed(_, inner_pattern) => self.fresh_type_for_pattern(inner_pattern)?,
            _ => self.context.new_variable(),
        })
    }

    /// Bind a pattern that must be irrefutable against `expected_type`.
    /// Used at let-binding, lambda-parameter, and loop-variable sites
    /// where the pattern may not fail to match. Refutable patterns
    /// produce a compile error pointing at the smallest offending
    /// sub-pattern.
    pub(crate) fn bind_irrefutable_pattern(
        &mut self,
        pattern: &Pattern,
        expected_type: &Type,
        generalize: bool,
    ) -> Result<Type> {
        let result = self.bind_pattern(pattern, expected_type, generalize)?;
        let applied = expected_type.apply(&self.context);
        if let Err(e) = is_irrefutable(pattern, &applied) {
            bail_type_at!(
                e.culprit,
                "refutable pattern in irrefutable position: {}",
                e.reason
            );
        }
        Ok(result)
    }

    /// Bind a pattern at the given type, adding the bound names to the
    /// current scope. If `generalize` is true, generalizes types for
    /// polymorphism (used at let bindings).
    pub(crate) fn bind_pattern(
        &mut self,
        pattern: &Pattern,
        expected_type: &Type,
        generalize: bool,
    ) -> Result<Type> {
        match &pattern.kind {
            PatternKind::Name(name) => {
                let type_scheme = if generalize {
                    self.generalize(expected_type)
                } else {
                    TypeScheme::Monotype(expected_type.clone())
                };
                self.define(name.clone(), IdentifierKind::Local, type_scheme);
                self.type_table.insert(
                    pattern.h.id,
                    TypeScheme::Monotype(expected_type.apply(&self.context)),
                );
                Ok(expected_type.clone())
            }
            PatternKind::Wildcard => {
                self.type_table.insert(
                    pattern.h.id,
                    TypeScheme::Monotype(expected_type.apply(&self.context)),
                );
                Ok(expected_type.clone())
            }
            PatternKind::Tuple(patterns) => {
                let expected_applied = expected_type.apply(&self.context);
                match expected_applied {
                    Type::Constructed(TypeName::Tuple(_), ref elem_types) => {
                        if elem_types.len() != patterns.len() {
                            bail_type_at!(
                                pattern.h.span,
                                "Tuple pattern has {} elements but type has {}",
                                patterns.len(),
                                elem_types.len()
                            );
                        }
                        for (sub_pattern, elem_type) in patterns.iter().zip(elem_types.iter()) {
                            self.bind_pattern(sub_pattern, elem_type, generalize)?;
                        }
                        self.type_table.insert(
                            pattern.h.id,
                            TypeScheme::Monotype(expected_type.apply(&self.context)),
                        );
                        Ok(expected_type.clone())
                    }
                    _ => Err(err_type_at!(
                        pattern.h.span,
                        "Expected tuple type for tuple pattern, got {}",
                        self.format_type(&expected_applied)
                    )),
                }
            }
            PatternKind::Vec(patterns) => {
                // Force the scrutinee to be a vec of this pattern's arity
                // with a fresh element tvar. Unify so the user can write
                // `let @[a, b] = some_polymorphic_value in …` and have
                // inference propagate.
                let elem_tvar = self.context.new_variable();
                let pattern_vec_ty = vec(patterns.len(), elem_tvar.clone());
                self.context.unify(&pattern_vec_ty, expected_type).map_err(|_| {
                    err_type_at!(
                        pattern.h.span,
                        "`@[{}]` pattern requires a vec{} but got {}",
                        std::iter::repeat("_").take(patterns.len()).collect::<Vec<_>>().join(", "),
                        patterns.len(),
                        self.format_type(&expected_type.apply(&self.context))
                    )
                })?;
                let elem_ty = elem_tvar.apply(&self.context);
                for sub_pattern in patterns {
                    self.bind_pattern(sub_pattern, &elem_ty, generalize)?;
                }
                self.type_table.insert(
                    pattern.h.id,
                    TypeScheme::Monotype(expected_type.apply(&self.context)),
                );
                Ok(expected_type.clone())
            }
            PatternKind::Typed(inner_pattern, annotated_type) => {
                let normalized =
                    self.normalize_annotation_type(annotated_type, self.current_module.as_deref())?;
                self.context.unify(&normalized, expected_type).map_err(|_| {
                    err_type_at!(
                        pattern.h.span,
                        "Pattern type annotation {} doesn't match expected type {}",
                        self.format_type(&normalized),
                        self.format_type(expected_type)
                    )
                })?;
                let result = self.bind_pattern(inner_pattern, &normalized, generalize)?;
                let resolved = normalized.apply(&self.context);
                self.type_table.insert(pattern.h.id, TypeScheme::Monotype(resolved));
                Ok(result)
            }
            PatternKind::Attributed(_, inner_pattern) => {
                let result = self.bind_pattern(inner_pattern, expected_type, generalize)?;
                self.type_table.insert(
                    pattern.h.id,
                    TypeScheme::Monotype(expected_type.apply(&self.context)),
                );
                Ok(result)
            }
            PatternKind::Unit => {
                let unit_type = tuple(vec![]);
                self.context.unify(&unit_type, expected_type).map_err(|_| {
                    err_type_at!(
                        pattern.h.span,
                        "Unit pattern doesn't match expected type {}",
                        self.format_type(expected_type)
                    )
                })?;
                self.type_table.insert(pattern.h.id, TypeScheme::Monotype(unit_type.apply(&self.context)));
                Ok(unit_type)
            }
            PatternKind::Constructor(name, args) => {
                let expected_applied = expected_type.apply(&self.context);
                match expected_applied {
                    Type::Constructed(TypeName::Sum(ref variants), _) => {
                        let payload_types = match variants.iter().find(|(n, _)| n == name) {
                            Some((_, payload)) => payload,
                            None => bail_type_at!(
                                pattern.h.span,
                                "constructor `#{}` not found in sum type {}",
                                name,
                                self.format_type(&expected_applied)
                            ),
                        };
                        if args.len() != payload_types.len() {
                            bail_type_at!(
                                pattern.h.span,
                                "constructor `#{}` expects {} payload value{}, got {}",
                                name,
                                payload_types.len(),
                                if payload_types.len() == 1 { "" } else { "s" },
                                args.len()
                            );
                        }
                        for (sub_pattern, payload_ty) in args.iter().zip(payload_types.iter()) {
                            self.bind_pattern(sub_pattern, payload_ty, generalize)?;
                        }
                        self.type_table.insert(
                            pattern.h.id,
                            TypeScheme::Monotype(expected_type.apply(&self.context)),
                        );
                        Ok(expected_type.clone())
                    }
                    _ => Err(err_type_at!(
                        pattern.h.span,
                        "constructor pattern `#{}` requires a sum-typed scrutinee, got {}",
                        name,
                        self.format_type(&expected_applied)
                    )),
                }
            }
            PatternKind::Literal(lit) => {
                // Literal patterns bind no names; they constrain the
                // value at runtime. Type-check by unifying the literal's
                // numeric tower / bool with the expected scrutinee type.
                let applied = expected_type.apply(&self.context);
                let lit_ty = match lit {
                    PatternLiteral::Int(_) => match &applied {
                        Type::Constructed(TypeName::Int(_), _)
                        | Type::Constructed(TypeName::UInt(_), _) => expected_type.clone(),
                        _ => {
                            let inferred = i32();
                            self.context.unify(&inferred, expected_type).map_err(|_| {
                                err_type_at!(
                                    pattern.h.span,
                                    "Int literal pattern doesn't match expected type {}",
                                    self.format_type(expected_type)
                                )
                            })?;
                            inferred
                        }
                    },
                    PatternLiteral::Float(_) => match &applied {
                        Type::Constructed(TypeName::Float(_), _) => expected_type.clone(),
                        _ => {
                            let inferred = f32();
                            self.context.unify(&inferred, expected_type).map_err(|_| {
                                err_type_at!(
                                    pattern.h.span,
                                    "Float literal pattern doesn't match expected type {}",
                                    self.format_type(expected_type)
                                )
                            })?;
                            inferred
                        }
                    },
                    PatternLiteral::Bool(_) => {
                        let inferred = bool_type();
                        self.context.unify(&inferred, expected_type).map_err(|_| {
                            err_type_at!(
                                pattern.h.span,
                                "Bool literal pattern doesn't match expected type {}",
                                self.format_type(expected_type)
                            )
                        })?;
                        inferred
                    }
                };
                self.type_table.insert(pattern.h.id, TypeScheme::Monotype(lit_ty.apply(&self.context)));
                Ok(lit_ty)
            }
            PatternKind::Record(_) => Err(err_type_at!(
                pattern.h.span,
                "Record patterns in match/let are not yet supported"
            )),
        }
    }
}

#[cfg(test)]
#[path = "bind_tests.rs"]
mod bind_tests;
