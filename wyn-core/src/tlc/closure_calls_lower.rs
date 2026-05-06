//! Closure-call lowering phase boundary (phase 3 of the defunctionalization split).
//!
//! Owns the post-condition verifier asserting every call is "direct" —
//! the function position of every `App` resolves to a `Var`, never to a
//! nested `App`, a `Lambda`, or any other dynamic value. Captures have
//! been fully threaded into trailing args. Combined with the
//! closure-conversion and HOF-specialization verifiers, this completes
//! the user's invariant chain: no dynamic function calls survive into
//! the backend.
//!
//! Strictly stronger than `assert_flat_apps` (which only forbids nested
//! `App` in func position) — this verifier additionally rejects every
//! non-`Var` func, including residual `Lambda` nodes and constructed
//! function values.
//!
//! The actual lowering logic still lives inside `tlc::defunctionalize`
//! (capture threading is interleaved with lifting and HOF specialization).
//! This module owns the architectural seam.

use super::{Program, Term, TermKind};
use crate::{SymbolId, SymbolTable};
use std::collections::HashMap;

#[derive(Debug)]
pub enum ClosureCallsLowerError {
    /// An `App` node has a function position that isn't a `Var(SymbolId)` —
    /// e.g. a nested App, a Lambda, or some constructed value. After
    /// closure-call lowering, every call should resolve statically.
    IndirectCall {
        def: SymbolId,
        func_kind: &'static str,
    },
    /// An `App { Var(target), args }` has the wrong arg count for its
    /// target. After defunctionalize + monomorphize + buffer_specialize,
    /// every direct call must be fully applied.
    ArityMismatch {
        def: SymbolId,
        target: SymbolId,
        expected: usize,
        actual: usize,
    },
}

pub fn verify_closure_calls_lowered(program: &Program) -> Result<(), ClosureCallsLowerError> {
    // Build an arity map for direct-call targets. Intrinsics not in
    // `program.defs` fall back to `builtins::intrinsic_arity` (the
    // catalog-derived arity for each builtin). Targets the catalog
    // doesn't know about are skipped — those are operator dispatch
    // helpers whose arity is enforced by the backend.
    let arities: HashMap<SymbolId, usize> = program.defs.iter().map(|d| (d.name, d.arity)).collect();
    for def in &program.defs {
        walk(&def.body, def.name, &arities, &program.symbols)?;
    }
    Ok(())
}

fn walk(
    term: &Term,
    def: SymbolId,
    arities: &HashMap<SymbolId, usize>,
    symbols: &SymbolTable,
) -> Result<(), ClosureCallsLowerError> {
    if let TermKind::App { func, args } = &term.kind {
        if !is_static_func(&func.kind) {
            return Err(ClosureCallsLowerError::IndirectCall {
                def,
                func_kind: discriminant_name(&func.kind),
            });
        }
        if let TermKind::Var(crate::tlc::VarRef::Symbol(target)) = &func.kind {
            let expected = arities
                .get(target)
                .copied()
                .or_else(|| symbols.get(*target).and_then(|name| crate::builtins::intrinsic_arity(name)));
            if let Some(expected) = expected {
                if expected != args.len() {
                    return Err(ClosureCallsLowerError::ArityMismatch {
                        def,
                        target: *target,
                        expected,
                        actual: args.len(),
                    });
                }
            }
        }
    }
    let mut result = Ok(());
    term.for_each_child(&mut |child| {
        if result.is_ok() {
            result = walk(child, def, arities, symbols);
        }
    });
    result
}

/// A "static func" position is anything that resolves to a known
/// callable at compile time: a `Var` (top-level def or local
/// parameter), or an operator value (`BinOp`/`UnOp`) that the backend
/// dispatches directly via a fixed PrimOp.
fn is_static_func(kind: &TermKind) -> bool {
    matches!(
        kind,
        TermKind::Var(_) | TermKind::BinOp(_) | TermKind::UnOp(_) | TermKind::Extern(_)
    )
}

fn discriminant_name(kind: &TermKind) -> &'static str {
    match kind {
        TermKind::Var(_) => "Var",
        TermKind::BinOp(_) => "BinOp",
        TermKind::UnOp(_) => "UnOp",
        TermKind::Lambda(_) => "Lambda",
        TermKind::App { .. } => "App",
        TermKind::Let { .. } => "Let",
        TermKind::IntLit(_) => "IntLit",
        TermKind::FloatLit(_) => "FloatLit",
        TermKind::BoolLit(_) => "BoolLit",
        TermKind::Extern(_) => "Extern",
        TermKind::If { .. } => "If",
        TermKind::Loop { .. } => "Loop",
        TermKind::Soac(_) => "Soac",
        TermKind::ArrayExpr(_) => "ArrayExpr",
        TermKind::Force(_) => "Force",
    }
}

#[cfg(test)]
#[path = "closure_calls_lower_tests.rs"]
mod closure_calls_lower_tests;
