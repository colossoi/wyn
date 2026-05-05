//! Closure conversion (phase 1 of the defunctionalization split).
//!
//! Lifts every standalone `TermKind::Lambda` to a top-level `Def`, replacing
//! the lambda site with a `Var` referring to the lifted symbol. Captures are
//! threaded as additional trailing parameters on the lifted def. Lambdas
//! embedded inside `SoacOp` envelopes are also lifted, but the envelope
//! itself is preserved as a structural payload (the SOAC's loop body for
//! later lowering).
//!
//! The post-pass shape is validated by `verify_closure_converted`.
//!
//! Phase 1 of the defunctionalization split: this pass owns *only*
//! lambda lifting and free-variable analysis. HOF specialization and
//! call-site capture threading are downstream concerns (phases 2 and 3).

use super::{ArrayExpr, Def, Lambda, Program, SoacOp, Term, TermKind};
use crate::SymbolId;
use std::collections::HashSet;

// =============================================================================
// Verifier
// =============================================================================

/// Errors a `verify_closure_converted` walk can report. Each carries enough
/// context to point at the failing def.
#[derive(Debug)]
pub enum ClosureConvertError {
    /// A `TermKind::Lambda` appeared deep inside a def body (i.e. not in the
    /// top-level parameter spine and not as a SOAC envelope). Closure
    /// conversion should have lifted it.
    UnliftedLambda {
        def: SymbolId,
    },
    /// A SOAC's `lam: Lambda` envelope has a body that isn't `Var(top_level)`.
    /// Phase 1 expects every SOAC lambda to have already been lifted, with
    /// its body reduced to a reference into the top-level def set.
    SoacLambdaNotLifted {
        def: SymbolId,
    },
}

/// Walk every def and assert phase-1 invariants. Returns Ok(()) iff the
/// program is in valid post-closure-converted shape.
pub fn verify_closure_converted(program: &Program) -> Result<(), ClosureConvertError> {
    let top_level: HashSet<SymbolId> = program.defs.iter().map(|d| d.name).collect();
    for def in &program.defs {
        verify_def(def, &top_level)?;
    }
    Ok(())
}

fn verify_def(def: &Def, top_level: &HashSet<SymbolId>) -> Result<(), ClosureConvertError> {
    let inner = strip_param_spine(&def.body);
    walk_no_lambdas(inner, def.name, top_level)
}

/// Skip past leading `TermKind::Lambda` nodes — those are the def's parameter
/// spine, which `defunc_preserving_params` keeps intact for entry points and
/// regular top-level functions. Return the first non-lambda subterm.
fn strip_param_spine(term: &Term) -> &Term {
    let mut current = term;
    while let TermKind::Lambda(Lambda { body, .. }) = &current.kind {
        current = body;
    }
    current
}

/// Recursively walk a term and assert no `TermKind::Lambda` appears anywhere,
/// except inside `SoacOp` / `ArrayExpr::Generate` envelopes where it must
/// have body = `Var(top_level)`.
fn walk_no_lambdas(
    term: &Term,
    def_name: SymbolId,
    top_level: &HashSet<SymbolId>,
) -> Result<(), ClosureConvertError> {
    match &term.kind {
        TermKind::Lambda(_) => return Err(ClosureConvertError::UnliftedLambda { def: def_name }),
        TermKind::Soac(soac) => check_soac_envelopes(soac, def_name, top_level)?,
        TermKind::ArrayExpr(ae) => check_array_expr_envelopes(ae, def_name, top_level)?,
        _ => {}
    }
    let mut result = Ok(());
    term.for_each_child(&mut |child| {
        if result.is_ok() {
            result = walk_no_lambdas(child, def_name, top_level);
        }
    });
    result
}

fn check_soac_envelopes(
    soac: &SoacOp,
    def_name: SymbolId,
    top_level: &HashSet<SymbolId>,
) -> Result<(), ClosureConvertError> {
    let lambdas: Vec<&Lambda> = match soac {
        SoacOp::Map { lam, .. } => vec![lam],
        SoacOp::Reduce { op, .. } => vec![op],
        SoacOp::Scan { op, .. } => vec![op],
        SoacOp::Filter { pred, .. } => vec![pred],
        SoacOp::ReduceByIndex { op, .. } => vec![op],
        SoacOp::Redomap { op, reduce_op, .. } => vec![op, reduce_op],
        SoacOp::Scatter { .. } => vec![],
    };
    for lam in lambdas {
        if !is_lifted_body(&lam.body, top_level) {
            return Err(ClosureConvertError::SoacLambdaNotLifted { def: def_name });
        }
    }
    Ok(())
}

fn check_array_expr_envelopes(
    ae: &ArrayExpr,
    def_name: SymbolId,
    top_level: &HashSet<SymbolId>,
) -> Result<(), ClosureConvertError> {
    if let ArrayExpr::Generate { index_fn, .. } = ae {
        if !is_lifted_body(&index_fn.body, top_level) {
            return Err(ClosureConvertError::SoacLambdaNotLifted { def: def_name });
        }
    }
    Ok(())
}

fn is_lifted_body(body: &Term, top_level: &HashSet<SymbolId>) -> bool {
    matches!(&body.kind, TermKind::Var(sym) if top_level.contains(sym))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "closure_convert_tests.rs"]
mod closure_convert_tests;
