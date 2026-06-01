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

use super::VarRef;
use super::{ArrayExpr, Def, DefMeta, Lambda, LoopKind, Program, SoacOp, Term, TermIdSource, TermKind};
use crate::ast::{Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::{HashMap, HashSet};

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
        kind: TermKind::Var(VarRef::Symbol(func_sym)),
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
        TermKind::Var(VarRef::Symbol(sym)) => {
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
        TermKind::Var(VarRef::Builtin { .. })
        | TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::UnitLit
        | TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::Extern(_) => {}
        TermKind::Coerce { inner, .. } => {
            collect_free_vars(inner, bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::Soac(soac) => {
            collect_free_vars_soac(soac, bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::ArrayExpr(ae) => {
            collect_free_vars_array_expr(ae, bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
            for p in parts {
                collect_free_vars(p, bound, top_level, known_defs, symbols, free, seen);
            }
        }
        TermKind::TupleProj { tuple, .. } => {
            collect_free_vars(tuple, bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::Index { array, index } => {
            collect_free_vars(array, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(index, bound, top_level, known_defs, symbols, free, seen);
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
        SoacOp::Scan {
            op,
            reduce_op,
            ne,
            input,
            ..
        } => {
            collect_free_vars_soac_body(op, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_soac_body(reduce_op, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(ne, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
        }
        SoacOp::Filter { pred, input, .. } => {
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
        ArrayExpr::Literal(terms) => {
            for t in terms {
                collect_free_vars(t, bound, top_level, known_defs, symbols, free, seen);
            }
        }
        ArrayExpr::Range { start, len, step } => {
            collect_free_vars(start, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(len, bound, top_level, known_defs, symbols, free, seen);
            if let Some(s) = step {
                collect_free_vars(s, bound, top_level, known_defs, symbols, free, seen);
            }
        }
        ArrayExpr::StorageView(crate::tlc::StorageView { offset, len, .. }) => {
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
        TermKind::ArrayExpr(_) => {}
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
        SoacOp::Scan { op, reduce_op, .. } => vec![op, reduce_op],
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

fn is_lifted_body(body: &Term, top_level: &HashSet<SymbolId>) -> bool {
    matches!(&body.kind, TermKind::Var(VarRef::Symbol(sym)) if top_level.contains(sym))
}

// =============================================================================
// CallableValue + ClosureInfo
// =============================================================================
//
// Side-table consumed by `hof_specialize` and `closure_calls_lower`.
// After `closure_convert::run`, every `Var(sym)` that names a callable
// resolves through `closure_info.resolve_callable(sym)` to one of these
// two shapes, eliminating the need for downstream passes to reason about
// inline `Lambda` nodes, let-bound aliases, or top-level def status.

#[derive(Debug, Clone)]
pub enum CallableValue {
    /// Direct reference to a top-level def with no captures. The
    /// surrounding `App(Var(sym), args)` already has the right arg
    /// shape and needs no rewriting at call lowering.
    Direct(SymbolId),
    /// Lifted lambda whose body closes over `captures`. Call lowering
    /// rewrites `App(Var(sym), args)` to `App(Var(code), args ++ captures)`,
    /// but only when `args.len() == param_count` — i.e. the captures
    /// haven't already been threaded (this is the case in cloned HOF
    /// bodies that `hof_specialize` has pre-threaded). `param_count`
    /// is the lifted def's user-facing parameter count, excluding the
    /// appended capture parameters.
    Closure {
        code: SymbolId,
        captures: Vec<Term>,
        param_count: usize,
    },
}

#[derive(Debug, Default)]
pub struct ClosureInfo {
    /// Every callable `Var`-position is keyed here:
    ///   - Lifted lambda symbols allocated by this pass
    ///   - Top-level defs with arrow type
    /// Symbols not in this map are non-callable values.
    pub callable_values: HashMap<SymbolId, CallableValue>,
}

impl ClosureInfo {
    pub fn resolve_callable(&self, sym: SymbolId) -> Option<&CallableValue> {
        self.callable_values.get(&sym)
    }
}

// =============================================================================
// ClosureConverter pass
// =============================================================================

struct ClosureConverter<'a> {
    symbols: SymbolTable,
    top_level: HashSet<SymbolId>,
    known_defs: &'a HashSet<String>,
    lifted_defs: Vec<Def>,
    callable_values: HashMap<SymbolId, CallableValue>,
    lambda_counter: u32,
    term_ids: TermIdSource,
}

impl<'a> ClosureConverter<'a> {
    fn run(program: Program, known_defs: &'a HashSet<String>) -> (Program, ClosureInfo) {
        let top_level: HashSet<SymbolId> = program.defs.iter().map(|d| d.name).collect();

        let mut callable_values = HashMap::new();
        for def in &program.defs {
            if is_arrow_param(&def.ty) {
                callable_values.insert(def.name, CallableValue::Direct(def.name));
            }
        }

        let mut cc = Self {
            symbols: program.symbols,
            top_level,
            known_defs,
            lifted_defs: vec![],
            callable_values,
            lambda_counter: 0,
            term_ids: TermIdSource::new(),
        };

        let transformed: Vec<Def> = program
            .defs
            .into_iter()
            .map(|def| {
                let body = cc.convert_def_body(def.body);
                Def { body, ..def }
            })
            .collect();

        let result_program = Program {
            defs: transformed.into_iter().chain(cc.lifted_defs).collect(),
            symbols: cc.symbols,
            ..program
        };

        let info = ClosureInfo {
            callable_values: cc.callable_values,
        };
        (result_program, info)
    }

    /// Walk a def body, preserving the outer parameter-spine `Lambda`
    /// nodes (those carry the def's named parameters). Anything below
    /// the spine routes through `convert_term`, which lifts every
    /// inline lambda.
    fn convert_def_body(&mut self, term: Term) -> Term {
        match term.kind {
            TermKind::Lambda(Lambda { params, body, ret_ty }) => {
                let new_body = self.convert_def_body(*body);
                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty,
                    span: term.span,
                    kind: TermKind::Lambda(Lambda {
                        params,
                        body: Box::new(new_body),
                        ret_ty,
                    }),
                }
            }
            _ => self.convert_term(term),
        }
    }

    /// Recursively walk a term, lifting every `Lambda` node to a
    /// top-level def and replacing it with `Var(lifted_sym)`. Lets
    /// whose rhs is a callable Var get substituted away — the let-name
    /// is rewritten to the target throughout the body and the let is
    /// dropped.
    fn convert_term(&mut self, term: Term) -> Term {
        let ty = term.ty.clone();
        let span = term.span;
        match term.kind {
            TermKind::Lambda(_) => self.lift_lambda(term).0,

            TermKind::App { func, args } => {
                let func = self.convert_term(*func);
                let args = args.into_iter().map(|a| self.convert_term(a)).collect();
                Term {
                    id: self.term_ids.next_id(),
                    ty,
                    span,
                    kind: TermKind::App {
                        func: Box::new(func),
                        args,
                    },
                }
            }

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                let new_rhs = self.convert_term(*rhs);

                // If the converted rhs is a callable `Var`, substitute
                // `name → target` in the *raw* body before walking it,
                // and drop the let. Substituting before walking is
                // essential: any nested lambda's free-variable analysis
                // would otherwise capture `Var(name)` (an unrelated
                // identifier) into its captures list, which gets frozen
                // in `callable_values`. Threading that stale name into
                // call sites later yields a dangling reference.
                if let TermKind::Var(VarRef::Symbol(target_sym)) = &new_rhs.kind {
                    if self.callable_values.contains_key(target_sym) {
                        let target = *target_sym;
                        let substituted_body =
                            super::hof_specialize::substitute_var(&body, name, target, &mut self.term_ids);
                        return self.convert_term(substituted_body);
                    }
                }

                let new_body = self.convert_term(*body);
                Term {
                    id: self.term_ids.next_id(),
                    ty,
                    span,
                    kind: TermKind::Let {
                        name,
                        name_ty,
                        rhs: Box::new(new_rhs),
                        body: Box::new(new_body),
                    },
                }
            }

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::If {
                    cond: Box::new(self.convert_term(*cond)),
                    then_branch: Box::new(self.convert_term(*then_branch)),
                    else_branch: Box::new(self.convert_term(*else_branch)),
                },
            },

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let init = self.convert_term(*init);
                let init_bindings: Vec<_> =
                    init_bindings.into_iter().map(|(n, ty, e)| (n, ty, self.convert_term(e))).collect();
                let kind = match kind {
                    LoopKind::For { var, var_ty, iter } => LoopKind::For {
                        var,
                        var_ty,
                        iter: Box::new(self.convert_term(*iter)),
                    },
                    LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                        var,
                        var_ty,
                        bound: Box::new(self.convert_term(*bound)),
                    },
                    LoopKind::While { cond } => LoopKind::While {
                        cond: Box::new(self.convert_term(*cond)),
                    },
                };
                let body = self.convert_term(*body);
                Term {
                    id: self.term_ids.next_id(),
                    ty,
                    span,
                    kind: TermKind::Loop {
                        loop_var,
                        loop_var_ty,
                        init: Box::new(init),
                        init_bindings,
                        kind,
                        body: Box::new(body),
                    },
                }
            }

            TermKind::Soac(soac) => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Soac(self.convert_soac(soac, span)),
            },

            TermKind::ArrayExpr(ae) => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::ArrayExpr(self.convert_array_expr(ae, span)),
            },

            // Leaf kinds — no rewriting.
            TermKind::Var(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::UnitLit
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_) => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: term.kind,
            },

            TermKind::Coerce { inner, target_ty } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Coerce {
                    inner: Box::new(self.convert_term(*inner)),
                    target_ty,
                },
            },

            TermKind::Tuple(parts) => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Tuple(parts.into_iter().map(|p| self.convert_term(p)).collect()),
            },
            TermKind::TupleProj { tuple, idx } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::TupleProj {
                    tuple: Box::new(self.convert_term(*tuple)),
                    idx,
                },
            },
            TermKind::Index { array, index } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Index {
                    array: Box::new(self.convert_term(*array)),
                    index: Box::new(self.convert_term(*index)),
                },
            },
            TermKind::VecLit(parts) => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::VecLit(parts.into_iter().map(|p| self.convert_term(p)).collect()),
            },
        }
    }

    /// Lift a `Lambda` term to a fresh top-level def. Returns `(Var(lifted_sym),
    /// captures)`: the replacement term for the original site, and the captures
    /// computed during free-variable analysis. The lifted def is pushed onto
    /// `lifted_defs`, the new symbol is registered in `top_level`, and an entry
    /// is added to `callable_values`.
    fn lift_lambda(&mut self, term: Term) -> (Term, Vec<Term>) {
        let ty = term.ty.clone();
        let span = term.span;

        let (params, inner_body) = super::extract_lambda_params(&term);
        let converted_body = self.convert_term(inner_body);

        let bound: HashSet<SymbolId> = params.iter().map(|(p, _)| *p).collect();
        let captures = self.compute_transitive_captures(&converted_body, &bound);

        let rebuilt = rebuild_nested_lam(&params, converted_body, span, &mut self.term_ids);
        let lifted_sym = self.fresh_lambda_symbol();
        self.top_level.insert(lifted_sym);

        if captures.is_empty() {
            self.lifted_defs.push(Def {
                name: lifted_sym,
                ty: rebuilt.ty.clone(),
                body: rebuilt,
                meta: DefMeta::LiftedLambda,
                arity: params.len(),
            });
            self.callable_values.insert(lifted_sym, CallableValue::Direct(lifted_sym));
        } else {
            let cap_params: Vec<(SymbolId, Type<TypeName>)> = captures
                .iter()
                .map(|cap_term| match &cap_term.kind {
                    TermKind::Var(VarRef::Symbol(sym)) => (*sym, cap_term.ty.clone()),
                    other => panic!(
                        "compute_free_vars contract violated: capture is not a Var: {:?}",
                        other
                    ),
                })
                .collect();
            let wrapped = append_capture_params(rebuilt, &cap_params, span, &mut self.term_ids);
            let arity = params.len() + captures.len();

            self.lifted_defs.push(Def {
                name: lifted_sym,
                ty: wrapped.ty.clone(),
                body: wrapped,
                meta: DefMeta::LiftedLambda,
                arity,
            });
            self.callable_values.insert(
                lifted_sym,
                CallableValue::Closure {
                    code: lifted_sym,
                    captures: captures.clone(),
                    param_count: params.len(),
                },
            );
        }

        let var_term = Term {
            id: self.term_ids.next_id(),
            ty,
            span,
            kind: TermKind::Var(VarRef::Symbol(lifted_sym)),
        };
        (var_term, captures)
    }

    /// Lift a SOAC envelope `Lambda` and produce the matching
    /// `SoacBody`: the body becomes `Var(lifted_sym)`, the captures
    /// triple-list is populated.
    fn lift_soac_lambda(&mut self, lam: Lambda, span: Span) -> super::SoacBody {
        let lam_ty = if lam.params.len() == 1 {
            Type::Constructed(TypeName::Arrow, vec![lam.params[0].1.clone(), lam.ret_ty.clone()])
        } else {
            let mut ty = lam.ret_ty.clone();
            for (_, param_ty) in lam.params.iter().rev() {
                ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), ty]);
            }
            ty
        };

        let original_params = lam.params.clone();
        let original_ret_ty = lam.ret_ty.clone();
        let lam_term = Term {
            id: self.term_ids.next_id(),
            ty: lam_ty,
            span,
            kind: TermKind::Lambda(lam),
        };

        let (var_term, capture_terms) = self.lift_lambda(lam_term);

        let lifted_sym = match &var_term.kind {
            TermKind::Var(VarRef::Symbol(sym)) => *sym,
            _ => panic!("BUG: lift_lambda did not return Var"),
        };

        let captures: Vec<(SymbolId, Type<TypeName>, Term)> = capture_terms
            .into_iter()
            .map(|t| match &t.kind {
                TermKind::Var(VarRef::Symbol(s)) => (*s, t.ty.clone(), t.clone()),
                _ => panic!("BUG: capture term is not a Var: {:?}", t.kind),
            })
            .collect();

        let body = Term {
            id: self.term_ids.next_id(),
            ty: var_term.ty.clone(),
            span,
            kind: TermKind::Var(VarRef::Symbol(lifted_sym)),
        };

        super::SoacBody {
            lam: Lambda {
                params: original_params,
                body: Box::new(body),
                ret_ty: original_ret_ty,
            },
            captures,
        }
    }

    fn convert_soac(&mut self, soac: SoacOp, span: Span) -> SoacOp {
        match soac {
            SoacOp::Map {
                lam,
                inputs,
                consumes_input,
            } => SoacOp::Map {
                lam: self.lift_soac_lambda(lam.lam, span),
                inputs: inputs.into_iter().map(|ae| self.convert_array_expr(ae, span)).collect(),
                consumes_input,
            },
            SoacOp::Reduce { op, ne, input } => SoacOp::Reduce {
                op: self.lift_soac_lambda(op.lam, span),
                ne: Box::new(self.convert_term(*ne)),
                input: self.convert_array_expr(input, span),
            },
            SoacOp::Scan {
                op,
                reduce_op,
                ne,
                input,
                consumes_input,
            } => SoacOp::Scan {
                op: self.lift_soac_lambda(op.lam, span),
                reduce_op: self.lift_soac_lambda(reduce_op.lam, span),
                ne: Box::new(self.convert_term(*ne)),
                input: self.convert_array_expr(input, span),
                consumes_input,
            },
            SoacOp::Filter {
                pred,
                input,
                consumes_input,
            } => SoacOp::Filter {
                pred: self.lift_soac_lambda(pred.lam, span),
                input: self.convert_array_expr(input, span),
                consumes_input,
            },
            SoacOp::Scatter {
                dest,
                indices,
                values,
            } => SoacOp::Scatter {
                dest,
                indices: self.convert_array_expr(indices, span),
                values: self.convert_array_expr(values, span),
            },
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
            } => SoacOp::ReduceByIndex {
                dest,
                op: self.lift_soac_lambda(op.lam, span),
                ne: Box::new(self.convert_term(*ne)),
                indices: self.convert_array_expr(indices, span),
                values: self.convert_array_expr(values, span),
            },
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
            } => SoacOp::Redomap {
                op: self.lift_soac_lambda(op.lam, span),
                reduce_op: self.lift_soac_lambda(reduce_op.lam, span),
                ne: Box::new(self.convert_term(*ne)),
                inputs: inputs.into_iter().map(|ae| self.convert_array_expr(ae, span)).collect(),
            },
        }
    }

    fn convert_array_expr(&mut self, ae: ArrayExpr, span: Span) -> ArrayExpr {
        match ae {
            ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(self.convert_term(*t))),
            ArrayExpr::Zip(exprs) => {
                ArrayExpr::Zip(exprs.into_iter().map(|e| self.convert_array_expr(e, span)).collect())
            }
            ArrayExpr::Soac(op) => ArrayExpr::Soac(Box::new(self.convert_soac(*op, span))),
            ArrayExpr::Literal(terms) => {
                ArrayExpr::Literal(terms.into_iter().map(|t| self.convert_term(t)).collect())
            }
            ArrayExpr::Range { start, len, step } => ArrayExpr::Range {
                start: Box::new(self.convert_term(*start)),
                len: Box::new(self.convert_term(*len)),
                step: step.map(|s| Box::new(self.convert_term(*s))),
            },
            ArrayExpr::StorageView(_) => {
                unreachable!("StorageBuffer introduced after defunctionalization")
            }
        }
    }

    fn fresh_lambda_symbol(&mut self) -> SymbolId {
        let name = format!("_w_lambda_{}", self.lambda_counter);
        self.lambda_counter += 1;
        self.symbols.alloc(name)
    }

    /// Compute the captures of a lambda body, *including* transitive
    /// captures pulled in through `Var(closure_sym)` references.
    ///
    /// The `_w_`-prefix filter in `compute_free_vars` correctly omits
    /// already-lifted lambda symbols from the FV set (those are now
    /// top-level), but the FV pass would also miss the closures'
    /// captures — which `closure_calls_lower` will later thread into
    /// every call site. If those captures are not bound by the
    /// surrounding lambda's params, the surrounding lambda needs to
    /// capture them so they remain in scope after lifting.
    fn compute_transitive_captures(&self, body: &Term, bound: &HashSet<SymbolId>) -> Vec<Term> {
        let mut result = compute_free_vars(body, bound, &self.top_level, self.known_defs, &self.symbols);
        let mut seen: HashSet<SymbolId> = result
            .iter()
            .filter_map(|t| match &t.kind {
                TermKind::Var(VarRef::Symbol(s)) => Some(*s),
                _ => None,
            })
            .collect();

        let mut worklist: Vec<SymbolId> = super::collect_var_refs(body);
        let mut visited: HashSet<SymbolId> = HashSet::new();
        while let Some(sym) = worklist.pop() {
            if !visited.insert(sym) {
                continue;
            }
            let captures = match self.callable_values.get(&sym) {
                Some(CallableValue::Closure { captures, .. }) => captures,
                _ => continue,
            };
            for cap_term in captures {
                let cap_sym = match &cap_term.kind {
                    TermKind::Var(VarRef::Symbol(s)) => *s,
                    _ => continue,
                };
                if bound.contains(&cap_sym) || seen.contains(&cap_sym) {
                    worklist.push(cap_sym);
                    continue;
                }
                let name = self.symbols.get(cap_sym).expect("BUG: capture sym not in table");
                if self.top_level.contains(&cap_sym)
                    || self.known_defs.contains(name)
                    || name.starts_with("_w_")
                {
                    worklist.push(cap_sym);
                    continue;
                }
                seen.insert(cap_sym);
                result.push(cap_term.clone());
                worklist.push(cap_sym);
            }
        }

        result
    }
}

fn is_arrow_param(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::Arrow, _))
}

/// Run closure conversion. Lifts every `Lambda` node to a top-level def,
/// substitutes let-aliased callable Vars away, and returns a side-table
/// describing every callable symbol's captures (or lack thereof).
pub fn run(program: Program, known_defs: &HashSet<String>) -> (Program, ClosureInfo) {
    let result = ClosureConverter::run(program, known_defs);
    verify_closure_converted(&result.0)
        .unwrap_or_else(|e| panic!("closure-conversion verifier failed: {:?}", e));
    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "closure_convert_tests.rs"]
mod closure_convert_tests;

/// Integration tests for the three-phase closure pipeline
/// (`closure_convert::run` → `hof_specialize::run` →
/// `closure_calls_lower::run`). Lives here because closure_convert is
/// the pipeline's entry point.
#[cfg(test)]
#[path = "closure_pipeline_tests.rs"]
mod closure_pipeline_tests;
