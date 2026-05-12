//! Type-checking pattern bindings. Extends `TypeChecker` with
//! `fresh_type_for_pattern` (used by lambda parameter inference) and
//! `bind_pattern` (used by `let` and `match` arms).

use crate::ast::{Pattern, PatternKind};
use crate::error::CompilerError;
use crate::scope::IdentifierKind;
use crate::types::checker::TypeChecker;
use crate::types::{Type, TypeName, TypeScheme, tuple};
use crate::{bail_type_at, err_type_at};

type Result<T> = std::result::Result<T, CompilerError>;

impl<'a> TypeChecker<'a> {
    /// Create a fresh type for a pattern based on its structure.
    /// Tuple patterns produce a tuple of fresh variables; everything
    /// else produces a single fresh variable (or the annotation, if
    /// the pattern is `Typed`).
    pub(crate) fn fresh_type_for_pattern(&mut self, pattern: &Pattern) -> Type {
        match &pattern.kind {
            PatternKind::Tuple(patterns) => {
                let elem_types: Vec<Type> =
                    patterns.iter().map(|p| self.fresh_type_for_pattern(p)).collect();
                tuple(elem_types)
            }
            PatternKind::Typed(_, annotated_type) => {
                self.normalize_annotation_type(annotated_type, self.current_module.as_deref())
            }
            PatternKind::Attributed(_, inner_pattern) => self.fresh_type_for_pattern(inner_pattern),
            _ => self.context.new_variable(),
        }
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
            PatternKind::Typed(inner_pattern, annotated_type) => {
                let normalized =
                    self.normalize_annotation_type(annotated_type, self.current_module.as_deref());
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
            _ => Err(err_type_at!(
                pattern.h.span,
                "Pattern {:?} not yet supported in lambda parameters",
                pattern.kind
            )),
        }
    }
}

#[cfg(test)]
#[path = "bind_tests.rs"]
mod bind_tests;
