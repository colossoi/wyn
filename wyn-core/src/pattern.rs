//! Pattern matching utilities shared across compiler passes.
//!
//! This module provides a generic API for extracting bindings from patterns,
//! which can be used in type checking, code generation, and other passes.

use crate::ast::{Pattern, PatternKind};

/// A binding extracted from a pattern: (name, value).
pub type Binding<T> = (String, T);

/// Trait for values that can be decomposed according to patterns.
///
/// Different passes implement this trait for their value types:
/// - Type checker: implements for `Type`
/// - Code generator: implements for registers/values
pub trait PatternValue: Clone {
    /// Extract the i-th element from a tuple value.
    /// Returns None if the value is not a tuple or index is out of bounds.
    fn tuple_element(&self, index: usize) -> Option<Self>;

    /// Get the number of elements if this is a tuple.
    /// Returns None if not a tuple.
    fn tuple_len(&self) -> Option<usize>;
}

/// Error that can occur during pattern matching.
#[derive(Debug, Clone)]
pub enum PatternError {
    /// Pattern expects a tuple but value is not a tuple
    NotATuple,
    /// Tuple pattern has wrong number of elements
    TupleLengthMismatch {
        expected: usize,
        actual: usize,
    },
    /// Pattern kind not supported
    UnsupportedPattern(String),
}

/// Extract all bindings from a pattern matched against a value.
///
/// Returns a list of (name, value) pairs for all Name patterns in the tree.
/// Wildcards and Units produce no bindings.
///
/// # Example
/// ```ignore
/// // Pattern: (x, (y, z))
/// // Value: (1, (2, 3))
/// // Result: [("x", 1), ("y", 2), ("z", 3)]
/// ```
pub fn extract_bindings<T: PatternValue>(
    pattern: &Pattern,
    value: T,
) -> Result<Vec<Binding<T>>, PatternError> {
    let mut bindings = Vec::new();
    extract_bindings_inner(pattern, value, &mut bindings)?;
    Ok(bindings)
}

fn extract_bindings_inner<T: PatternValue>(
    pattern: &Pattern,
    value: T,
    bindings: &mut Vec<Binding<T>>,
) -> Result<(), PatternError> {
    match &pattern.kind {
        PatternKind::Name(name) => {
            bindings.push((name.clone(), value));
            Ok(())
        }

        PatternKind::Wildcard => {
            // Wildcard binds nothing
            Ok(())
        }

        PatternKind::Unit => {
            // Unit pattern binds nothing
            Ok(())
        }

        PatternKind::Tuple(patterns) => {
            let len = value.tuple_len().ok_or(PatternError::NotATuple)?;
            if len != patterns.len() {
                return Err(PatternError::TupleLengthMismatch {
                    expected: patterns.len(),
                    actual: len,
                });
            }

            for (i, sub_pattern) in patterns.iter().enumerate() {
                let elem = value.tuple_element(i).ok_or(PatternError::NotATuple)?;
                extract_bindings_inner(sub_pattern, elem, bindings)?;
            }
            Ok(())
        }

        PatternKind::Typed(inner, _ty) => {
            // Type annotation doesn't affect binding extraction
            // (type checking happens separately)
            extract_bindings_inner(inner, value, bindings)
        }

        PatternKind::Attributed(_, inner) => {
            // Attributes don't affect binding extraction
            extract_bindings_inner(inner, value, bindings)
        }

        PatternKind::Literal(_) => {
            // Literal patterns don't bind anything (used for matching)
            Ok(())
        }

        PatternKind::Record(_) => Err(PatternError::UnsupportedPattern("Record patterns".to_string())),

        PatternKind::Constructor(_, _) => Err(PatternError::UnsupportedPattern(
            "Constructor patterns".to_string(),
        )),
    }
}

/// Get all names bound by a pattern (without values).
///
/// Useful for checking what variables a pattern introduces.
pub fn bound_names(pattern: &Pattern) -> Vec<String> {
    let mut names = Vec::new();
    collect_names(pattern, &mut names);
    names
}

fn collect_names(pattern: &Pattern, names: &mut Vec<String>) {
    match &pattern.kind {
        PatternKind::Name(name) => {
            names.push(name.clone());
        }
        PatternKind::Wildcard | PatternKind::Unit | PatternKind::Literal(_) => {}
        PatternKind::Tuple(patterns) => {
            for p in patterns {
                collect_names(p, names);
            }
        }
        PatternKind::Typed(inner, _) | PatternKind::Attributed(_, inner) => {
            collect_names(inner, names);
        }
        PatternKind::Record(fields) => {
            for field in fields {
                if let Some(p) = &field.pattern {
                    collect_names(p, names);
                } else {
                    // Shorthand: field name is the binding
                    names.push(field.field.clone());
                }
            }
        }
        PatternKind::Constructor(_, patterns) => {
            for p in patterns {
                collect_names(p, names);
            }
        }
    }
}
