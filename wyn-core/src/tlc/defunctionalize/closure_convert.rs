//! Lambda lifting and explicit environment construction.
//!
//! This is the family-changing part of defunctionalization. It consumes the
//! pre-closure tree and rebuilds it as a `ClosureConverted` tree. Closure
//! environments are stored on the closure value and SOAC environments are
//! stored on the SOAC body; no symbol-keyed closure side table survives.

use super::{ClosureConverted, Defunctionalized};
use crate::ast::{Span, TypeName};
use crate::tlc::data::{
    Empty, ExplicitCaptures, ExplicitCapturesPayload, ExplicitClosure, ExplicitClosurePayload,
};
use crate::tlc::{
    self, ArrayExpr, Def, DefMeta, Lambda, LoopKind, Program, SoacBody, SoacOp, Term, TermIdSource,
    TermKind, VarRef,
};
use crate::{LookupMap, LookupSet, SymbolId, SymbolTable};
use polytype::Type;

// =============================================================================
// Shared lambda construction
// =============================================================================

fn append_capture_params(
    lam: Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    captures: &[(SymbolId, Type<TypeName>)],
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term<ExplicitClosurePayload, ExplicitCapturesPayload> {
    let (orig_params, inner_body) = tlc::extract_lambda_params(&lam);
    let mut all_params = orig_params;
    all_params.extend(captures.iter().cloned());
    tlc::rebuild_nested_lam(&all_params, inner_body, span, term_ids)
}

// =============================================================================
// Free-variable analysis
// =============================================================================

/// Compute the free variables of a term. The returned terms are the first
/// variable occurrences for each free symbol, in traversal order.
pub fn compute_free_vars<C, S>(
    term: &Term<C, S>,
    bound: &LookupSet<SymbolId>,
    top_level: &LookupSet<SymbolId>,
    known_defs: &LookupSet<String>,
    symbols: &SymbolTable,
) -> Vec<Term<C, S>>
where
    C: tlc::Payload,
    S: tlc::Payload,
{
    let mut free = Vec::new();
    let mut seen = LookupSet::new();
    collect_free_vars(term, bound, top_level, known_defs, symbols, &mut free, &mut seen);
    free
}

fn collect_free_vars<C, S>(
    term: &Term<C, S>,
    bound: &LookupSet<SymbolId>,
    top_level: &LookupSet<SymbolId>,
    known_defs: &LookupSet<String>,
    symbols: &SymbolTable,
    free: &mut Vec<Term<C, S>>,
    seen: &mut LookupSet<SymbolId>,
) where
    C: tlc::Payload,
    S: tlc::Payload,
{
    match &term.kind {
        TermKind::Var(VarRef::Symbol(sym)) => {
            let name = symbols.get(*sym).expect("BUG: symbol not in table");
            if !bound.contains(sym)
                && !top_level.contains(sym)
                && !known_defs.contains(name)
                && !name.starts_with("_w_")
                && seen.insert(*sym)
            {
                free.push(term.clone());
            }
        }
        TermKind::Let { name, rhs, body, .. } => {
            collect_free_vars(rhs, bound, top_level, known_defs, symbols, free, seen);
            let mut body_bound = bound.clone();
            body_bound.insert(*name);
            collect_free_vars(body, &body_bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::Lambda(lam) => {
            collect_free_vars_lambda(lam, bound, top_level, known_defs, symbols, free, seen);
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
            let mut loop_bound = bound.clone();
            loop_bound.insert(*loop_var);
            loop_bound.extend(init_bindings.iter().map(|(name, _, _)| *name));
            match kind {
                LoopKind::For { var, iter, .. } => {
                    collect_free_vars(iter, bound, top_level, known_defs, symbols, free, seen);
                    loop_bound.insert(*var);
                }
                LoopKind::ForRange {
                    var,
                    bound: range_bound,
                    ..
                } => {
                    collect_free_vars(range_bound, bound, top_level, known_defs, symbols, free, seen);
                    loop_bound.insert(*var);
                }
                LoopKind::While { cond } => {
                    collect_free_vars(cond, &loop_bound, top_level, known_defs, symbols, free, seen);
                }
            }
            for (_, _, expr) in init_bindings {
                collect_free_vars(expr, &loop_bound, top_level, known_defs, symbols, free, seen);
            }
            collect_free_vars(body, &loop_bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::Soac(soac) => {
            collect_free_vars_soac(soac, bound, top_level, known_defs, symbols, free, seen)
        }
        TermKind::Closure(data) => C::for_each(data, &mut |capture| {
            collect_free_vars(capture, bound, top_level, known_defs, symbols, free, seen)
        }),
        TermKind::Var(VarRef::Builtin { .. })
        | TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::UnitLit
        | TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::Extern(_) => {}
        _ => term.for_each_child(&mut |child| {
            collect_free_vars(child, bound, top_level, known_defs, symbols, free, seen)
        }),
    }
}

fn collect_free_vars_lambda<C, S>(
    lam: &Lambda<C, S>,
    bound: &LookupSet<SymbolId>,
    top_level: &LookupSet<SymbolId>,
    known_defs: &LookupSet<String>,
    symbols: &SymbolTable,
    free: &mut Vec<Term<C, S>>,
    seen: &mut LookupSet<SymbolId>,
) where
    C: tlc::Payload,
    S: tlc::Payload,
{
    let mut body_bound = bound.clone();
    body_bound.extend(lam.params.iter().map(|(param, _)| *param));
    collect_free_vars(&lam.body, &body_bound, top_level, known_defs, symbols, free, seen);
}

fn collect_free_vars_soac_body<C, S>(
    body: &SoacBody<C, S>,
    bound: &LookupSet<SymbolId>,
    top_level: &LookupSet<SymbolId>,
    known_defs: &LookupSet<String>,
    symbols: &SymbolTable,
    free: &mut Vec<Term<C, S>>,
    seen: &mut LookupSet<SymbolId>,
) where
    C: tlc::Payload,
    S: tlc::Payload,
{
    collect_free_vars_lambda(&body.lam, bound, top_level, known_defs, symbols, free, seen);
    S::for_each(&body.data, &mut |(_, _, capture)| {
        collect_free_vars(capture, bound, top_level, known_defs, symbols, free, seen)
    });
}

fn collect_free_vars_soac<C, S>(
    soac: &SoacOp<C, S>,
    bound: &LookupSet<SymbolId>,
    top_level: &LookupSet<SymbolId>,
    known_defs: &LookupSet<String>,
    symbols: &SymbolTable,
    free: &mut Vec<Term<C, S>>,
    seen: &mut LookupSet<SymbolId>,
) where
    C: tlc::Payload,
    S: tlc::Payload,
{
    let mut visit_body = |body: &SoacBody<C, S>| {
        collect_free_vars_soac_body(body, bound, top_level, known_defs, symbols, free, seen)
    };
    match soac {
        SoacOp::Map { lam, inputs, .. } => {
            visit_body(lam);
            for input in inputs {
                collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
            }
        }
        SoacOp::Reduce { op, ne, input } | SoacOp::Scan { op, ne, input, .. } => {
            visit_body(op);
            collect_free_vars(ne, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
        }
        SoacOp::Filter { pred, input, .. } => {
            visit_body(pred);
            collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
        }
        SoacOp::Scatter { lam, inputs, .. } => {
            visit_body(lam);
            for input in inputs {
                collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
            }
        }
        SoacOp::ReduceByIndex {
            op,
            ne,
            indices,
            values,
            ..
        } => {
            visit_body(op);
            collect_free_vars(ne, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(indices, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(values, bound, top_level, known_defs, symbols, free, seen);
        }
    }
}

fn collect_free_vars_array_expr<C, S>(
    array: &ArrayExpr<C, S>,
    bound: &LookupSet<SymbolId>,
    top_level: &LookupSet<SymbolId>,
    known_defs: &LookupSet<String>,
    symbols: &SymbolTable,
    free: &mut Vec<Term<C, S>>,
    seen: &mut LookupSet<SymbolId>,
) where
    C: tlc::Payload,
    S: tlc::Payload,
{
    match array {
        ArrayExpr::Var(var, ty) => collect_free_vars(
            &tlc::synthetic_atom_var_term(*var, ty.clone()),
            bound,
            top_level,
            known_defs,
            symbols,
            free,
            seen,
        ),
        ArrayExpr::Zip(parts) => {
            for part in parts {
                collect_free_vars_array_expr(part, bound, top_level, known_defs, symbols, free, seen);
            }
        }
        ArrayExpr::Literal(terms) => {
            for term in terms {
                collect_free_vars(term, bound, top_level, known_defs, symbols, free, seen);
            }
        }
        ArrayExpr::Range { start, len, step } => {
            collect_free_vars(start, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(len, bound, top_level, known_defs, symbols, free, seen);
            if let Some(step) = step {
                collect_free_vars(step, bound, top_level, known_defs, symbols, free, seen);
            }
        }
    }
}

// =============================================================================
// Verifier
// =============================================================================

#[derive(Debug)]
pub enum ClosureConvertError {
    UnliftedLambda {
        def: SymbolId,
    },
    SoacLambdaNotLifted {
        def: SymbolId,
    },
}

pub fn verify_closure_converted<S>(program: &Program<S>) -> Result<(), ClosureConvertError>
where
    S: tlc::Stage<Family = ClosureConverted>,
{
    let top_level: LookupSet<SymbolId> = program.defs.iter().map(|def| def.name).collect();
    for def in &program.defs {
        let mut body = &def.body;
        while let TermKind::Lambda(lam) = &body.kind {
            body = &lam.body;
        }
        verify_no_nested_lambdas(body, def.name, &top_level)?;
    }
    Ok(())
}

fn verify_no_nested_lambdas(
    term: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    def: SymbolId,
    top_level: &LookupSet<SymbolId>,
) -> Result<(), ClosureConvertError> {
    if matches!(&term.kind, TermKind::Lambda(_)) {
        return Err(ClosureConvertError::UnliftedLambda { def });
    }
    if let TermKind::Soac(soac) = &term.kind {
        let check = |body: &SoacBody<ExplicitClosurePayload, ExplicitCapturesPayload>| {
            if matches!(
                &body.lam.body.kind,
                TermKind::Var(VarRef::Symbol(symbol)) if top_level.contains(symbol)
            ) {
                Ok(())
            } else {
                Err(ClosureConvertError::SoacLambdaNotLifted { def })
            }
        };
        match soac {
            SoacOp::Map { lam, .. } | SoacOp::Scatter { lam, .. } => check(lam)?,
            SoacOp::Reduce { op, .. } | SoacOp::Scan { op, .. } | SoacOp::ReduceByIndex { op, .. } => {
                check(op)?
            }
            SoacOp::Filter { pred, .. } => check(pred)?,
        }
    }
    let mut result = Ok(());
    term.for_each_child(&mut |child| {
        if result.is_ok() {
            result = verify_no_nested_lambdas(child, def, top_level);
        }
    });
    result
}

// =============================================================================
// Consuming family translation
// =============================================================================

struct ClosureConverter {
    symbols: SymbolTable,
    top_level: LookupSet<SymbolId>,
    known_def_symbols: LookupSet<SymbolId>,
    callable_symbols: LookupSet<SymbolId>,
    callable_bindings: LookupMap<SymbolId, Term<ExplicitClosurePayload, ExplicitCapturesPayload>>,
    lifted_defs: Vec<Def<ClosureConverted>>,
    lambda_counter: u32,
    term_ids: TermIdSource,
}

impl ClosureConverter {
    fn convert_def(&mut self, def: Def<crate::tlc::family::Monomorphic>) -> Def<ClosureConverted> {
        Def {
            data: def.data,
            name: def.name,
            ty: def.ty,
            body: self.convert_def_body(def.body),
            meta: def.meta,
            arity: def.arity,
            param_diets: def.param_diets,
            return_diet: def.return_diet,
        }
    }

    fn convert_def_body(
        &mut self,
        term: Term<Empty, Empty>,
    ) -> Term<ExplicitClosurePayload, ExplicitCapturesPayload> {
        let Term { ty, span, kind, .. } = term;
        match kind {
            TermKind::Lambda(Lambda { params, body, ret_ty }) => {
                let body = self.convert_def_body(*body);
                Term {
                    id: self.term_ids.next_id(),
                    ty,
                    span,
                    kind: TermKind::Lambda(Lambda {
                        params,
                        body: Box::new(body),
                        ret_ty,
                    }),
                }
            }
            kind => self.convert_term(Term {
                id: tlc::TermId::SYNTHETIC,
                ty,
                span,
                kind,
            }),
        }
    }

    fn convert_term(
        &mut self,
        term: Term<Empty, Empty>,
    ) -> Term<ExplicitClosurePayload, ExplicitCapturesPayload> {
        let Term { ty, span, kind, .. } = term;
        match kind {
            TermKind::Var(VarRef::Symbol(symbol)) => {
                if let Some(replacement) = self.callable_bindings.get(&symbol) {
                    return tlc::clone_term_with_fresh_ids(replacement, &mut self.term_ids);
                }
                self.term(ty, span, TermKind::Var(VarRef::Symbol(symbol)))
            }
            TermKind::Var(var) => self.term(ty, span, TermKind::Var(var)),
            TermKind::Lambda(lambda) => self.lift_lambda(Term {
                id: tlc::TermId::SYNTHETIC,
                ty,
                span,
                kind: TermKind::Lambda(lambda),
            }),
            TermKind::Closure(()) => {
                unreachable!("pre-defunctionalization family cannot contain a closure")
            }
            TermKind::App { func, args } => {
                let func = self.convert_term(*func);
                let args = args.into_iter().map(|arg| self.convert_term(arg)).collect();
                self.term(
                    ty,
                    span,
                    TermKind::App {
                        func: Box::new(func),
                        args,
                    },
                )
            }
            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                let rhs = self.convert_term(*rhs);
                if self.is_callable(&rhs) {
                    let previous = self.callable_bindings.insert(name, rhs);
                    let body = self.convert_term(*body);
                    if let Some(previous) = previous {
                        self.callable_bindings.insert(name, previous);
                    } else {
                        self.callable_bindings.remove(&name);
                    }
                    return body;
                }
                let body = self.convert_term(*body);
                self.term(
                    ty,
                    span,
                    TermKind::Let {
                        name,
                        name_ty,
                        rhs: Box::new(rhs),
                        body: Box::new(body),
                    },
                )
            }
            TermKind::Coerce { inner, target_ty } => {
                let inner = self.convert_term(*inner);
                self.term(
                    ty,
                    span,
                    TermKind::Coerce {
                        inner: Box::new(inner),
                        target_ty,
                    },
                )
            }
            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond = self.convert_term(*cond);
                let then_branch = self.convert_term(*then_branch);
                let else_branch = self.convert_term(*else_branch);
                self.term(
                    ty,
                    span,
                    TermKind::If {
                        cond: Box::new(cond),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                )
            }
            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                let init = Box::new(self.convert_term(*init));
                let init_bindings = init_bindings
                    .into_iter()
                    .map(|(name, ty, value)| (name, ty, self.convert_term(value)))
                    .collect();
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
                let body = Box::new(self.convert_term(*body));
                self.term(
                    ty,
                    span,
                    TermKind::Loop {
                        loop_var,
                        loop_var_ty,
                        init,
                        init_bindings,
                        kind,
                        body,
                    },
                )
            }
            TermKind::Soac(soac) => {
                let soac = self.convert_soac(soac, span);
                self.term(ty, span, TermKind::Soac(soac))
            }
            TermKind::ArrayExpr(array) => {
                let array = self.convert_array_expr(array);
                self.term(ty, span, TermKind::ArrayExpr(array))
            }
            TermKind::Tuple(parts) => {
                let parts = parts.into_iter().map(|part| self.convert_term(part)).collect();
                self.term(ty, span, TermKind::Tuple(parts))
            }
            TermKind::TupleProj { tuple, idx } => {
                let tuple = Box::new(self.convert_term(*tuple));
                self.term(ty, span, TermKind::TupleProj { tuple, idx })
            }
            TermKind::Index { array, index } => {
                let array = Box::new(self.convert_term(*array));
                let index = Box::new(self.convert_term(*index));
                self.term(ty, span, TermKind::Index { array, index })
            }
            TermKind::VecLit(parts) => {
                let parts = parts.into_iter().map(|part| self.convert_term(part)).collect();
                self.term(ty, span, TermKind::VecLit(parts))
            }
            TermKind::BinOp(op) => self.term(ty, span, TermKind::BinOp(op)),
            TermKind::UnOp(op) => self.term(ty, span, TermKind::UnOp(op)),
            TermKind::IntLit(value) => self.term(ty, span, TermKind::IntLit(value)),
            TermKind::FloatLit(value) => self.term(ty, span, TermKind::FloatLit(value)),
            TermKind::BoolLit(value) => self.term(ty, span, TermKind::BoolLit(value)),
            TermKind::UnitLit => self.term(ty, span, TermKind::UnitLit),
            TermKind::Extern(name) => self.term(ty, span, TermKind::Extern(name)),
        }
    }

    fn term(
        &mut self,
        ty: Type<TypeName>,
        span: Span,
        kind: TermKind<ExplicitClosurePayload, ExplicitCapturesPayload>,
    ) -> Term<ExplicitClosurePayload, ExplicitCapturesPayload> {
        Term {
            id: self.term_ids.next_id(),
            ty,
            span,
            kind,
        }
    }

    fn is_callable(&self, term: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>) -> bool {
        match &term.kind {
            TermKind::Closure(_) => true,
            TermKind::Var(VarRef::Symbol(symbol)) => self.callable_symbols.contains(symbol),
            _ => false,
        }
    }

    fn forwarded_top_level_fn(
        &self,
        params: &[(SymbolId, Type<TypeName>)],
        body: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    ) -> Option<SymbolId> {
        let TermKind::App { func, args } = &body.kind else {
            return None;
        };
        let TermKind::Var(VarRef::Symbol(target)) = &func.kind else {
            return None;
        };
        if !self.top_level.contains(target) || args.len() != params.len() {
            return None;
        }
        args.iter()
            .zip(params)
            .all(|(arg, (param, _))| {
                matches!(&arg.kind, TermKind::Var(VarRef::Symbol(symbol)) if symbol == param)
            })
            .then_some(*target)
    }

    fn lift_lambda(
        &mut self,
        term: Term<Empty, Empty>,
    ) -> Term<ExplicitClosurePayload, ExplicitCapturesPayload> {
        let ty = term.ty.clone();
        let span = term.span;
        let (params, body) = tlc::extract_lambda_params(&term);
        let body = self.convert_term(body);
        let bound: LookupSet<SymbolId> = params.iter().map(|(param, _)| *param).collect();
        let captures = self.compute_captures(&body, &bound);

        if captures.is_empty() {
            if let Some(target) = self.forwarded_top_level_fn(&params, &body) {
                return self.term(ty, span, TermKind::Var(VarRef::Symbol(target)));
            }
        }

        let rebuilt = tlc::rebuild_nested_lam(&params, body, span, &mut self.term_ids);
        let lifted_symbol = self.fresh_lambda_symbol();
        self.top_level.insert(lifted_symbol);
        self.callable_symbols.insert(lifted_symbol);

        let (body, arity) = if captures.is_empty() {
            (rebuilt, params.len())
        } else {
            let capture_params = captures
                .iter()
                .map(|capture| match &capture.kind {
                    TermKind::Var(VarRef::Symbol(symbol)) => (*symbol, capture.ty.clone()),
                    _ => panic!("BUG: free-variable analysis returned a non-variable capture"),
                })
                .collect::<Vec<_>>();
            (
                append_capture_params(rebuilt, &capture_params, span, &mut self.term_ids),
                params.len() + captures.len(),
            )
        };

        self.lifted_defs.push(Def {
            data: (),
            name: lifted_symbol,
            ty: body.ty.clone(),
            body,
            meta: DefMeta::LiftedLambda,
            arity,
            param_diets: vec![crate::types::Diet::observing(); arity],
            return_diet: crate::types::Diet::observing(),
        });

        if captures.is_empty() {
            self.term(ty, span, TermKind::Var(VarRef::Symbol(lifted_symbol)))
        } else {
            self.term(
                ty,
                span,
                TermKind::Closure(ExplicitClosure {
                    code: lifted_symbol,
                    captures,
                    param_count: params.len(),
                }),
            )
        }
    }

    fn lift_soac_lambda(
        &mut self,
        lambda: Lambda<Empty, Empty>,
        span: Span,
    ) -> SoacBody<ExplicitClosurePayload, ExplicitCapturesPayload> {
        let original_params = lambda.params.clone();
        let original_ret_ty = lambda.ret_ty.clone();
        let mut lambda_ty = original_ret_ty.clone();
        for (_, param_ty) in original_params.iter().rev() {
            lambda_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), lambda_ty]);
        }
        let lambda_id = self.term_ids.next_id();
        let closure = self.lift_lambda(Term {
            id: lambda_id,
            ty: lambda_ty,
            span,
            kind: TermKind::Lambda(lambda),
        });
        let closure_ty = closure.ty.clone();
        let (code, captures) = match closure.kind {
            TermKind::Var(VarRef::Symbol(code)) => (code, Vec::new()),
            TermKind::Closure(ExplicitClosure { code, captures, .. }) => (code, captures),
            _ => panic!("BUG: lambda lifting produced a non-callable value"),
        };
        let captures = captures
            .into_iter()
            .map(|capture| {
                let symbol = match &capture.kind {
                    TermKind::Var(VarRef::Symbol(symbol)) => *symbol,
                    _ => panic!("BUG: free-variable analysis returned a non-variable capture"),
                };
                (symbol, capture.ty.clone(), capture)
            })
            .collect();
        let body = self.term(closure_ty, span, TermKind::Var(VarRef::Symbol(code)));
        SoacBody {
            lam: Lambda {
                params: original_params,
                body: Box::new(body),
                ret_ty: original_ret_ty,
            },
            data: ExplicitCaptures { captures },
        }
    }

    fn convert_soac(
        &mut self,
        soac: SoacOp<Empty, Empty>,
        span: Span,
    ) -> SoacOp<ExplicitClosurePayload, ExplicitCapturesPayload> {
        match soac {
            SoacOp::Map {
                lam,
                inputs,
                destination,
            } => SoacOp::Map {
                lam: self.lift_soac_lambda(lam.lam, span),
                inputs: inputs.into_iter().map(|input| self.convert_array_expr(input)).collect(),
                destination,
            },
            SoacOp::Reduce { op, ne, input } => SoacOp::Reduce {
                op: self.lift_soac_lambda(op.lam, span),
                ne: Box::new(self.convert_term(*ne)),
                input: self.convert_array_expr(input),
            },
            SoacOp::Scan {
                op,
                ne,
                input,
                destination,
            } => SoacOp::Scan {
                op: self.lift_soac_lambda(op.lam, span),
                ne: Box::new(self.convert_term(*ne)),
                input: self.convert_array_expr(input),
                destination,
            },
            SoacOp::Filter {
                pred,
                input,
                destination,
            } => SoacOp::Filter {
                pred: self.lift_soac_lambda(pred.lam, span),
                input: self.convert_array_expr(input),
                destination,
            },
            SoacOp::Scatter { dest, lam, inputs } => SoacOp::Scatter {
                dest,
                lam: self.lift_soac_lambda(lam.lam, span),
                inputs: inputs.into_iter().map(|input| self.convert_array_expr(input)).collect(),
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
                indices: self.convert_array_expr(indices),
                values: self.convert_array_expr(values),
            },
        }
    }

    fn convert_array_expr(
        &mut self,
        array: ArrayExpr<Empty, Empty>,
    ) -> ArrayExpr<ExplicitClosurePayload, ExplicitCapturesPayload> {
        match array {
            ArrayExpr::Var(var, ty) => ArrayExpr::Var(var, ty),
            ArrayExpr::Zip(parts) => {
                ArrayExpr::Zip(parts.into_iter().map(|part| self.convert_array_expr(part)).collect())
            }
            ArrayExpr::Literal(terms) => {
                ArrayExpr::Literal(terms.into_iter().map(|term| self.convert_term(term)).collect())
            }
            ArrayExpr::Range { start, len, step } => ArrayExpr::Range {
                start: Box::new(self.convert_term(*start)),
                len: Box::new(self.convert_term(*len)),
                step: step.map(|step| Box::new(self.convert_term(*step))),
            },
        }
    }

    fn fresh_lambda_symbol(&mut self) -> SymbolId {
        let symbol = self.symbols.alloc(format!("_w_lambda_{}", self.lambda_counter));
        self.lambda_counter += 1;
        symbol
    }

    fn compute_captures(
        &self,
        body: &Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
        bound: &LookupSet<SymbolId>,
    ) -> Vec<Term<ExplicitClosurePayload, ExplicitCapturesPayload>> {
        let mut result = compute_free_vars(body, bound, &self.top_level, &LookupSet::new(), &self.symbols);
        result.retain(|term| {
            !matches!(
                &term.kind,
                TermKind::Var(VarRef::Symbol(symbol))
                    if self.known_def_symbols.contains(symbol)
            )
        });
        result
    }
}

pub(super) fn run(
    program: Program<crate::tlc::stage::RuntimeIndexProducersFloated>,
) -> Program<Defunctionalized> {
    let Program {
        defs,
        symbols,
        def_syms,
        term_ids,
        global_context,
    } = program;
    let crate::tlc::context::RewriteGlobal {
        known_defs,
        auto_storage_binding_ids,
    } = global_context;
    let top_level = defs.iter().map(|def| def.name).collect();
    let callable_symbols = defs
        .iter()
        .filter(|def| matches!(&def.ty, Type::Constructed(TypeName::Arrow, _)))
        .map(|def| def.name)
        .collect();
    let known_def_symbols =
        def_syms.iter().filter_map(|(name, symbol)| known_defs.contains(name).then_some(*symbol)).collect();
    let mut converter = ClosureConverter {
        symbols,
        top_level,
        known_def_symbols,
        callable_symbols,
        callable_bindings: LookupMap::new(),
        lifted_defs: Vec::new(),
        lambda_counter: 0,
        term_ids,
    };
    let mut defs = defs.into_iter().map(|def| converter.convert_def(def)).collect::<Vec<_>>();
    defs.append(&mut converter.lifted_defs);
    let program = Program::from_parts(
        defs,
        converter.symbols,
        def_syms,
        converter.term_ids,
        crate::tlc::context::PostClosureGlobal {
            auto_storage_binding_ids,
        },
    );
    verify_closure_converted(&program)
        .unwrap_or_else(|error| panic!("closure-conversion verifier failed: {error:?}"));
    program
}

#[cfg(test)]
#[path = "closure_convert_tests.rs"]
mod closure_convert_tests;
