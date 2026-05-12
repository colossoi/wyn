//! Refutability check: is a given pattern irrefutable against its
//! declared type? Used to gate `let p = e in body` and lambda
//! parameters against patterns that could fail to match.
//!
//! Refutable patterns in let/lambda positions are a Wyn-specific
//! compile error (Futhark traps at runtime; we don't have a panic
//! intrinsic, and the GPU target makes one awkward).
//!
//! Irrefutable patterns:
//!   - `Name`, `Wildcard` — always match.
//!   - `Typed(p, _)`, `Attributed(_, p)` — delegate to inner.
//!   - `Unit` against the unit type — single-inhabitant.
//!   - `Tuple([p1..pn])` against a tuple type — irrefutable iff every
//!     sub-pattern is irrefutable.
//!   - `Record([{f1=p1; ...}])` against a record type — irrefutable
//!     iff every sub-pattern is irrefutable.
//!   - `Constructor(name, [sub])` against a single-variant sum —
//!     irrefutable iff every sub-pattern is irrefutable against the
//!     constructor's payload type.
//!
//! Refutable patterns:
//!   - `Literal` — constrains the value.
//!   - `Constructor` against a multi-variant sum — doesn't cover other
//!     variants.

use crate::ast::{self, PatternKind, Span};
use crate::types::{Type, TypeName};

#[derive(Debug, Clone)]
pub struct RefutabilityError {
    /// Span of the offending sub-pattern.
    pub culprit: Span,
    /// Human-readable reason (e.g. "constructor `#foo` doesn't cover
    /// `#bar` or `#baz`").
    pub reason: String,
}

/// Returns `Ok(())` if `pat` is irrefutable against `ty`, else an
/// error pointing at the smallest offending sub-pattern.
pub fn is_irrefutable(pat: &ast::Pattern, ty: &Type) -> Result<(), RefutabilityError> {
    match &pat.kind {
        PatternKind::Name(_) | PatternKind::Wildcard => Ok(()),
        PatternKind::Typed(inner, _) | PatternKind::Attributed(_, inner) => is_irrefutable(inner, ty),
        PatternKind::Unit => {
            if matches!(ty, Type::Constructed(TypeName::Unit, _)) {
                Ok(())
            } else {
                // Should be unreachable post-type-check, but guard for it.
                Err(RefutabilityError {
                    culprit: pat.h.span,
                    reason: format!("unit pattern against non-unit type {:?}", ty),
                })
            }
        }
        PatternKind::Literal(_) => Err(RefutabilityError {
            culprit: pat.h.span,
            reason: "literal patterns are refutable — use `match` to dispatch on values".to_string(),
        }),
        PatternKind::Tuple(sub) => {
            if let Type::Constructed(TypeName::Tuple(_), args) = ty {
                for (s, t) in sub.iter().zip(args.iter()) {
                    is_irrefutable(s, t)?;
                }
                Ok(())
            } else {
                Err(RefutabilityError {
                    culprit: pat.h.span,
                    reason: format!("tuple pattern against non-tuple type {:?}", ty),
                })
            }
        }
        PatternKind::Record(fields) => {
            if let Type::Constructed(TypeName::Record(record_names), args) = ty {
                for f in fields {
                    if let Some(sub) = &f.pattern {
                        let idx = record_names.iter().position(|n| n == &f.field).ok_or_else(|| {
                            RefutabilityError {
                                culprit: pat.h.span,
                                reason: format!("field `{}` not in record type", f.field),
                            }
                        })?;
                        is_irrefutable(sub, &args[idx])?;
                    }
                    // Shorthand `{name}` binds without constraint — irrefutable.
                }
                Ok(())
            } else {
                Err(RefutabilityError {
                    culprit: pat.h.span,
                    reason: format!("record pattern against non-record type {:?}", ty),
                })
            }
        }
        PatternKind::Constructor(name, sub) => {
            if let Type::Constructed(TypeName::Sum(variants), _) = ty {
                if variants.len() == 1 {
                    let (only_name, payload_tys) = &variants[0];
                    if only_name != name {
                        return Err(RefutabilityError {
                            culprit: pat.h.span,
                            reason: format!(
                                "constructor `#{}` doesn't match the sum's only variant `#{}`",
                                name, only_name
                            ),
                        });
                    }
                    for (s, t) in sub.iter().zip(payload_tys.iter()) {
                        is_irrefutable(s, t)?;
                    }
                    Ok(())
                } else {
                    let other_variants: Vec<String> = variants
                        .iter()
                        .filter(|(n, _)| n != name)
                        .map(|(n, _)| format!("#{}", n))
                        .collect();
                    Err(RefutabilityError {
                        culprit: pat.h.span,
                        reason: format!(
                            "constructor `#{}` doesn't cover other variants: {} — use `match` instead",
                            name,
                            other_variants.join(", ")
                        ),
                    })
                }
            } else {
                Err(RefutabilityError {
                    culprit: pat.h.span,
                    reason: format!("constructor pattern against non-sum type {:?}", ty),
                })
            }
        }
    }
}

#[cfg(test)]
#[path = "refutability_tests.rs"]
mod refutability_tests;
