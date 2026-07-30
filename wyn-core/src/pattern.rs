//! Pattern matching utilities shared across compiler passes.
//!
//! This module provides a generic API for extracting bindings from patterns,
//! which can be used in type checking, code generation, and other passes.

use crate::ast::{BindingName, Pattern, PatternKind, RecordPatternTarget, TreeFamily};

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
pub fn extract_bindings<T: PatternValue, F: TreeFamily, A>(
    pattern: &Pattern<F, A>,
    value: T,
) -> Result<Vec<Binding<T>>, PatternError> {
    let mut bindings = Vec::new();
    extract_bindings_inner(pattern, value, &mut bindings)?;
    Ok(bindings)
}

fn extract_bindings_inner<T: PatternValue, F: TreeFamily, A>(
    pattern: &Pattern<F, A>,
    value: T,
    bindings: &mut Vec<Binding<T>>,
) -> Result<(), PatternError> {
    match &pattern.kind {
        PatternKind::Name(name) => {
            bindings.push((name.source_name().to_owned(), value));
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

        PatternKind::Tuple(patterns) | PatternKind::Vec(patterns) => {
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
pub fn bound_names<T: TreeFamily, A>(pattern: &Pattern<T, A>) -> Vec<String> {
    let mut names = Vec::new();
    collect_names(pattern, &mut names);
    names
}

/// A path of tuple indices to reach a binding from the root value.
pub type ProjectionPath = Vec<usize>;

/// Information about a binding extracted from a pattern.
#[derive(Debug, Clone)]
pub struct BindingPath {
    /// The name being bound.
    pub name: String,
    /// Sequence of tuple indices to project from root to reach this binding.
    /// Empty if the pattern is just a name (no projection needed).
    /// Stable semantic identity after name resolution. Source-only patterns
    /// carry `None`; downstream semantic consumers require `Some`.
    pub symbol: Option<crate::SymbolId>,
    pub path: ProjectionPath,
}

/// Extract binding paths from a pattern.
///
/// For each name in the pattern, returns the sequence of tuple projections
/// needed to extract it from the root value.
///
/// # Example
/// ```ignore
/// // Pattern: (x, (y, z))
/// // Result: [
/// //   BindingPath { name: "x", path: [0] },
/// //   BindingPath { name: "y", path: [1, 0] },
/// //   BindingPath { name: "z", path: [1, 1] },
/// // ]
/// ```
pub fn binding_paths<T: TreeFamily, A>(pattern: &Pattern<T, A>) -> Vec<BindingPath> {
    let mut bindings = Vec::new();
    collect_binding_paths(pattern, &[], &mut bindings);
    bindings
}

fn collect_binding_paths<T: TreeFamily, A>(
    pattern: &Pattern<T, A>,
    path: &[usize],
    bindings: &mut Vec<BindingPath>,
) {
    match &pattern.kind {
        PatternKind::Name(name) => {
            bindings.push(BindingPath {
                name: name.source_name().to_owned(),
                symbol: name.symbol(),
                path: path.to_vec(),
            });
        }

        PatternKind::Wildcard | PatternKind::Unit | PatternKind::Literal(_) => {
            // These don't bind anything
        }

        PatternKind::Tuple(patterns) | PatternKind::Vec(patterns) => {
            for (i, p) in patterns.iter().enumerate() {
                let mut new_path = path.to_vec();
                new_path.push(i);
                collect_binding_paths(p, &new_path, bindings);
            }
        }

        PatternKind::Typed(inner, _) | PatternKind::Attributed(_, inner) => {
            collect_binding_paths(inner, path, bindings);
        }

        PatternKind::Record(fields) => {
            for (i, field) in fields.iter().enumerate() {
                let mut new_path = path.to_vec();
                new_path.push(i);
                match &field.target {
                    RecordPatternTarget::Pattern(pattern) => {
                        collect_binding_paths(pattern, &new_path, bindings)
                    }
                    RecordPatternTarget::Shorthand(binding) => bindings.push(BindingPath {
                        name: binding.source_name().to_owned(),
                        symbol: binding.symbol(),
                        path: new_path,
                    }),
                }
            }
        }

        PatternKind::Constructor(_, patterns) => {
            for (i, p) in patterns.iter().enumerate() {
                let mut new_path = path.to_vec();
                new_path.push(i);
                collect_binding_paths(p, &new_path, bindings);
            }
        }
    }
}

fn collect_names<T: TreeFamily, A>(pattern: &Pattern<T, A>, names: &mut Vec<String>) {
    match &pattern.kind {
        PatternKind::Name(name) => {
            names.push(name.source_name().to_owned());
        }
        PatternKind::Wildcard | PatternKind::Unit | PatternKind::Literal(_) => {}
        PatternKind::Tuple(patterns) | PatternKind::Vec(patterns) => {
            for p in patterns {
                collect_names(p, names);
            }
        }
        PatternKind::Typed(inner, _) | PatternKind::Attributed(_, inner) => {
            collect_names(inner, names);
        }
        PatternKind::Record(fields) => {
            for field in fields {
                match &field.target {
                    RecordPatternTarget::Pattern(pattern) => collect_names(pattern, names),
                    RecordPatternTarget::Shorthand(binding) => names.push(binding.source_name().to_owned()),
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

#[cfg(test)]
#[path = "pattern_tests.rs"]
mod pattern_tests;
