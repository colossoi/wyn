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

use super::{ArrayExpr, Def, Lambda, LoopKind, Program, SoacOp, Term, TermIdSource, TermKind};
use crate::ast::{Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::HashSet;

// =============================================================================
// Lambda construction helpers
// =============================================================================
//
// Pure constructors used by the lambda-lifting logic. Take an explicit
// `&mut TermIdSource` so they can live outside `Defunctionalizer`'s
// state. Operate on the standalone `TermKind::Lambda` form: produce a
// term with a fresh ID and a curried-arrow type built from the
// supplied params.

/// Build a single nested-lambda term from a parameter list and body.
/// The returned term's type is the curried arrow over `params` ending
/// at `body.ty`.
pub fn rebuild_nested_lam(
    params: &[(SymbolId, Type<TypeName>)],
    body: Term,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term {
    let ret_ty = body.ty.clone();
    let mut lam_ty = ret_ty.clone();
    for (_, param_ty) in params.iter().rev() {
        lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), lam_ty]);
    }
    Term {
        id: term_ids.next_id(),
        ty: lam_ty,
        span,
        kind: TermKind::Lambda(Lambda {
            params: params.to_vec(),
            body: Box::new(body),
            ret_ty,
        }),
    }
}

/// Build `App(Var(func_sym), args)` with a fresh ID and a curried-arrow
/// function-position type. Returns just the Var-with-arrow-type if
/// `args` is empty.
pub fn build_app_call(
    func_sym: SymbolId,
    args: Vec<Term>,
    result_ty: Type<TypeName>,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term {
    let mut fn_ty = result_ty.clone();
    for arg in args.iter().rev() {
        fn_ty = Type::Constructed(TypeName::Arrow, vec![arg.ty.clone(), fn_ty]);
    }
    let func_term = Term {
        id: term_ids.next_id(),
        ty: fn_ty,
        span,
        kind: TermKind::Var(func_sym),
    };

    if args.is_empty() {
        return Term {
            ty: result_ty,
            ..func_term
        };
    }

    Term {
        id: term_ids.next_id(),
        ty: result_ty,
        span,
        kind: TermKind::App {
            func: Box::new(func_term),
            args,
        },
    }
}

/// Build `App(func_term, args)` with a fresh ID. Returns `func_term`
/// unchanged if `args` is empty.
pub fn build_app_with_term(
    func_term: Term,
    args: Vec<Term>,
    result_ty: Type<TypeName>,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term {
    if args.is_empty() {
        return func_term;
    }
    Term {
        id: term_ids.next_id(),
        ty: result_ty,
        span,
        kind: TermKind::App {
            func: Box::new(func_term),
            args,
        },
    }
}

/// Append capture parameters to a (possibly nested-Lambda) term and
/// re-flatten into a single Lambda. Captures are supplied as
/// `(SymbolId, Type)` pairs so non-Var captures are unrepresentable.
///
/// Given `|x, y| body` and captures `[(a, A), (b, B)]`, produces
/// `|x, y, a, b| body` typed `X -> Y -> A -> B -> ret`.
pub fn append_capture_params(
    lam: Term,
    captures: &[(SymbolId, Type<TypeName>)],
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term {
    let (orig_params, inner_body) = super::extract_lambda_params(&lam);

    let mut all_params = orig_params;
    all_params.extend(captures.iter().cloned());

    let ret_ty = inner_body.ty.clone();
    let mut lam_ty = ret_ty.clone();
    for (_, param_ty) in all_params.iter().rev() {
        lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), lam_ty]);
    }
    Term {
        id: term_ids.next_id(),
        ty: lam_ty,
        span,
        kind: TermKind::Lambda(Lambda {
            params: all_params,
            body: Box::new(inner_body),
            ret_ty,
        }),
    }
}

// =============================================================================
// Free-variable analysis
// =============================================================================
//
// These helpers walk a term and collect references that aren't bound locally
// or globally. They're pure — no transformation, no allocation of new
// symbols — so closure_convert and `parallelize` (which also needs FV
// analysis for its outlining work) can share them.

/// Compute the free variables of a term given explicit `bound`/`top_level`
/// sets and the registry of intrinsic-style names. Returns one `Term` per
/// distinct free SymbolId (preserving type and span from its first
/// occurrence).
pub fn compute_free_vars(
    term: &Term,
    bound: &HashSet<SymbolId>,
    top_level: &HashSet<SymbolId>,
    known_defs: &HashSet<String>,
    symbols: &SymbolTable,
) -> Vec<Term> {
    let mut free = Vec::new();
    let mut seen = HashSet::new();
    collect_free_vars(term, bound, top_level, known_defs, symbols, &mut free, &mut seen);
    free
}

pub fn collect_free_vars(
    term: &Term,
    bound: &HashSet<SymbolId>,
    top_level: &HashSet<SymbolId>,
    known_defs: &HashSet<String>,
    symbols: &SymbolTable,
    free: &mut Vec<Term>,
    seen: &mut HashSet<SymbolId>,
) {
    match &term.kind {
        TermKind::Var(sym) => {
            let name = symbols.get(*sym).expect("BUG: symbol not in table");
            if !bound.contains(sym)
                && !top_level.contains(sym)
                && !known_defs.contains(name)
                && !name.starts_with("_w_")
                && !seen.contains(sym)
            {
                seen.insert(*sym);
                free.push(term.clone());
            }
        }
        TermKind::Let { name, rhs, body, .. } => {
            collect_free_vars(rhs, bound, top_level, known_defs, symbols, free, seen);
            let mut inner_bound = bound.clone();
            inner_bound.insert(*name);
            collect_free_vars(body, &inner_bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::Lambda(Lambda { params, body, .. }) => {
            let mut inner_bound = bound.clone();
            for (p, _) in params {
                inner_bound.insert(*p);
            }
            collect_free_vars(body, &inner_bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::App { func, args } => {
            collect_free_vars(func, bound, top_level, known_defs, symbols, free, seen);
            for arg in args {
                collect_free_vars(arg, bound, top_level, known_defs, symbols, free, seen);
            }
        }
        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_free_vars(cond, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(then_branch, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(else_branch, bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
            ..
        } => {
            collect_free_vars(init, bound, top_level, known_defs, symbols, free, seen);
            let mut inner_bound = bound.clone();
            inner_bound.insert(*loop_var);
            for (name, _, _) in init_bindings {
                inner_bound.insert(*name);
            }
            match kind {
                LoopKind::For { var, iter, .. } => {
                    collect_free_vars(iter, bound, top_level, known_defs, symbols, free, seen);
                    inner_bound.insert(*var);
                }
                LoopKind::ForRange {
                    var,
                    bound: bound_expr,
                    ..
                } => {
                    collect_free_vars(bound_expr, bound, top_level, known_defs, symbols, free, seen);
                    inner_bound.insert(*var);
                }
                LoopKind::While { cond } => {
                    collect_free_vars(cond, &inner_bound, top_level, known_defs, symbols, free, seen);
                }
            }
            for (_, _, expr) in init_bindings {
                collect_free_vars(expr, &inner_bound, top_level, known_defs, symbols, free, seen);
            }
            collect_free_vars(body, &inner_bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::Extern(_) => {}
        TermKind::Soac(soac) => {
            collect_free_vars_soac(soac, bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::ArrayExpr(ae) => {
            collect_free_vars_array_expr(ae, bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::Force(inner) => {
            collect_free_vars(inner, bound, top_level, known_defs, symbols, free, seen);
        }
    }
}

pub fn collect_free_vars_lambda(
    lam: &Lambda,
    bound: &HashSet<SymbolId>,
    top_level: &HashSet<SymbolId>,
    known_defs: &HashSet<String>,
    symbols: &SymbolTable,
    free: &mut Vec<Term>,
    seen: &mut HashSet<SymbolId>,
) {
    let mut inner_bound = bound.clone();
    for (p, _) in &lam.params {
        inner_bound.insert(*p);
    }
    collect_free_vars(
        &lam.body,
        &inner_bound,
        top_level,
        known_defs,
        symbols,
        free,
        seen,
    );
}

pub fn collect_free_vars_soac_body(
    sb: &super::SoacBody,
    bound: &HashSet<SymbolId>,
    top_level: &HashSet<SymbolId>,
    known_defs: &HashSet<String>,
    symbols: &SymbolTable,
    free: &mut Vec<Term>,
    seen: &mut HashSet<SymbolId>,
) {
    collect_free_vars_lambda(&sb.lam, bound, top_level, known_defs, symbols, free, seen);
    for (_, _, cap_term) in &sb.captures {
        collect_free_vars(cap_term, bound, top_level, known_defs, symbols, free, seen);
    }
}

pub fn collect_free_vars_soac(
    soac: &SoacOp,
    bound: &HashSet<SymbolId>,
    top_level: &HashSet<SymbolId>,
    known_defs: &HashSet<String>,
    symbols: &SymbolTable,
    free: &mut Vec<Term>,
    seen: &mut HashSet<SymbolId>,
) {
    match soac {
        SoacOp::Map { lam, inputs, .. } => {
            collect_free_vars_soac_body(lam, bound, top_level, known_defs, symbols, free, seen);
            for input in inputs {
                collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
            }
        }
        SoacOp::Reduce { op, ne, input, .. } => {
            collect_free_vars_soac_body(op, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(ne, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
        }
        SoacOp::Scan { op, ne, input } => {
            collect_free_vars_soac_body(op, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(ne, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
        }
        SoacOp::Filter { pred, input } => {
            collect_free_vars_soac_body(pred, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
        }
        SoacOp::Scatter { indices, values, .. } => {
            collect_free_vars_array_expr(indices, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(values, bound, top_level, known_defs, symbols, free, seen);
        }
        SoacOp::ReduceByIndex {
            op,
            ne,
            indices,
            values,
            ..
        } => {
            collect_free_vars_soac_body(op, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(ne, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(indices, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(values, bound, top_level, known_defs, symbols, free, seen);
        }
        SoacOp::Redomap { op, ne, inputs, .. } => {
            collect_free_vars_soac_body(op, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(ne, bound, top_level, known_defs, symbols, free, seen);
            for input in inputs {
                collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
            }
        }
    }
}

pub fn collect_free_vars_array_expr(
    ae: &ArrayExpr,
    bound: &HashSet<SymbolId>,
    top_level: &HashSet<SymbolId>,
    known_defs: &HashSet<String>,
    symbols: &SymbolTable,
    free: &mut Vec<Term>,
    seen: &mut HashSet<SymbolId>,
) {
    match ae {
        ArrayExpr::Ref(t) => collect_free_vars(t, bound, top_level, known_defs, symbols, free, seen),
        ArrayExpr::Zip(exprs) => {
            for e in exprs {
                collect_free_vars_array_expr(e, bound, top_level, known_defs, symbols, free, seen);
            }
        }
        ArrayExpr::Soac(op) => {
            collect_free_vars_soac(op, bound, top_level, known_defs, symbols, free, seen)
        }
        ArrayExpr::Generate { index_fn, .. } => {
            collect_free_vars_soac_body(index_fn, bound, top_level, known_defs, symbols, free, seen);
        }
        ArrayExpr::Literal(terms) => {
            for t in terms {
                collect_free_vars(t, bound, top_level, known_defs, symbols, free, seen);
            }
        }
        ArrayExpr::Range { start, len } => {
            collect_free_vars(start, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(len, bound, top_level, known_defs, symbols, free, seen);
        }
        ArrayExpr::StorageBuffer { offset, len, .. } => {
            // set/binding are compile-time u32s; only offset/len carry refs.
            collect_free_vars(offset, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(len, bound, top_level, known_defs, symbols, free, seen);
        }
    }
}

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
    let bodies: Vec<&super::SoacBody> = match soac {
        SoacOp::Map { lam, .. } => vec![lam],
        SoacOp::Reduce { op, .. } => vec![op],
        SoacOp::Scan { op, .. } => vec![op],
        SoacOp::Filter { pred, .. } => vec![pred],
        SoacOp::ReduceByIndex { op, .. } => vec![op],
        SoacOp::Redomap { op, reduce_op, .. } => vec![op, reduce_op],
        SoacOp::Scatter { .. } => vec![],
    };
    for sb in bodies {
        if !is_lifted_body(&sb.lam.body, top_level) {
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
        if !is_lifted_body(&index_fn.lam.body, top_level) {
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
