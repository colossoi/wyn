//! Defunctionalization for TLC (Futhark-style).
//!
//! This pass combines lambda lifting with static value tracking to eliminate
//! closures and partial applications. Unlike simple lambda lifting which creates
//! partial applications, this pass:
//!
//! 1. Lifts lambdas to top-level definitions (with captures as extra params at end)
//! 2. Tracks StaticVal alongside term transformation
//! 3. Flattens closure captures as explicit trailing arguments at all call sites
//!
//! Example transformation:
//!   Input:  f(|x| x + y, arr)     where y is captured
//!   Output: f _w_lambda_0 arr y
//!
//! The lifted lambda is: _w_lambda_0 = |x| |y| x + y  (captures at end)

use super::{ArrayExpr, Def, DefMeta, Lambda, LoopKind, Program, SoacOp, Term, TermIdSource, TermKind};
use crate::ast::{Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::{HashMap, HashSet};

// =============================================================================
// Static Value Tracking
// =============================================================================

/// Static value classification for defunctionalization.
///
/// Every term is evaluated to both a residual term AND a StaticVal.
/// The StaticVal tracks compile-time knowledge about function values.
#[derive(Debug, Clone)]
pub enum StaticVal {
    /// Runtime value - no compile-time function knowledge
    Dynamic,

    /// A lambda closure with:
    /// - The lifted function symbol
    /// - Captured terms (already typed, ready to append at call sites)
    Lambda {
        lifted_name: SymbolId,
        captures: Vec<Term>,
    },
}

// =============================================================================
// HOF detection + type substitution
// =============================================================================
//
// Lives in `tlc::hof_specialize` (the phase-2 home); re-export the items
// the lifting / call-site logic in this file still needs.

use super::hof_specialize::{
    HofInfo, TypeSubst, apply_type_subst, apply_type_subst_to_term, build_type_subst, detect_hofs,
    extract_param_types, format_type_for_key, is_arrow_type,
};

// =============================================================================
// Free Variable Analysis
// =============================================================================
//
// The actual implementations live in `tlc::closure_convert`; re-exports
// kept here for the lambda-lifting logic that hasn't migrated yet.

use super::closure_convert::compute_free_vars;

// =============================================================================
// Defunctionalizer
// =============================================================================

/// Result of defunctionalizing a term.
struct DefuncResult {
    term: Term,
    sv: StaticVal,
}

/// Defunctionalizer - combines lambda lifting with static value tracking.
pub struct Defunctionalizer<'a> {
    /// Symbol table for name resolution and allocation (owned during transformation)
    symbols: SymbolTable,
    /// Top-level definition names
    top_level: HashSet<SymbolId>,
    /// Built-in names (intrinsics, prelude functions) - keyed by string for lookup
    known_defs: &'a HashSet<String>,
    /// New definitions created for lifted lambdas
    lifted_defs: Vec<Def>,
    /// Counter for generating unique lambda names
    lambda_counter: u32,
    /// Term ID generator
    term_ids: TermIdSource,
    /// Environment: variable symbol -> StaticVal
    env: HashMap<SymbolId, StaticVal>,
    /// HOF info for both user-defined and intrinsic higher-order functions
    hof_info: HashMap<SymbolId, HofInfo>,
    /// Cache: (hof_name, lambda_name, arg_types) -> specialized_name
    /// Uses SymbolId for names, strings for type keys
    specialization_cache: HashMap<(SymbolId, SymbolId, Vec<String>), SymbolId>,
    /// Counter for generating unique specialization names
    specialization_counter: usize,
    /// Captures for lifted lambdas: lambda symbol -> capture terms
    lifted_lambda_captures: HashMap<SymbolId, Vec<Term>>,
}

impl<'a> Defunctionalizer<'a> {
    /// Defunctionalize a program.
    pub fn run(program: Program, known_defs: &'a HashSet<String>) -> Program {
        // Detect HOFs before defunctionalization (user-defined only;
        // intrinsic SOACs are now first-class SOAC nodes, not HOF calls)
        let hof_info = detect_hofs(&program.defs);
        let top_level: HashSet<SymbolId> = program.defs.iter().map(|d| d.name).collect();

        let mut defunc = Self {
            symbols: program.symbols,
            top_level,
            known_defs,
            lifted_defs: vec![],
            lambda_counter: 0,
            term_ids: TermIdSource::new(),
            env: HashMap::new(),
            hof_info,
            specialization_cache: HashMap::new(),
            specialization_counter: 0,
            lifted_lambda_captures: HashMap::new(),
        };

        let transformed_defs: Vec<_> = program
            .defs
            .into_iter()
            .map(|def| {
                // For all defs, preserve parameter lambdas but defunc the body
                let body = defunc.defunc_preserving_params(def.body);
                Def { body, ..def }
            })
            .collect();

        let (lifted_defs, symbols) = defunc.finish();

        Program {
            defs: transformed_defs.into_iter().chain(lifted_defs).collect(),
            uniforms: program.uniforms,
            storage: program.storage,
            symbols,
            def_syms: program.def_syms,
        }
    }

    /// Consume the defunctionalizer and return the lifted definitions and symbol table.
    fn finish(self) -> (Vec<Def>, SymbolTable) {
        (self.lifted_defs, self.symbols)
    }

    /// Defunctionalize but preserve outermost parameter lambdas (for entry points).
    fn defunc_preserving_params(&mut self, term: Term) -> Term {
        match term.kind {
            TermKind::Lambda(Lambda { params, body, ret_ty }) => {
                // Mark all params as Dynamic in env
                for (param, _) in &params {
                    self.env.insert(*param, StaticVal::Dynamic);
                }
                // Process body — if it's also a Lambda, this handles nested lambdas too
                let defunc_body = self.defunc_preserving_params(*body);
                for (param, _) in &params {
                    self.env.remove(param);
                }

                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty,
                    span: term.span,
                    kind: TermKind::Lambda(Lambda {
                        params,
                        body: Box::new(defunc_body),
                        ret_ty,
                    }),
                }
            }
            // Once we hit a non-lambda, defunc normally
            _ => self.defunc_term(term).term,
        }
    }

    /// Core: defunctionalize a term, returning both transformed term and static value.
    fn defunc_term(&mut self, term: Term) -> DefuncResult {
        let ty = term.ty.clone();
        let span = term.span;

        match term.kind {
            TermKind::Var(crate::tlc::VarRef::Symbol(sym)) => {
                // Look up static value from environment
                let sv = if let Some(sv) = self.env.get(&sym) {
                    sv.clone()
                } else if let Some(captures) = self.lifted_lambda_captures.get(&sym) {
                    // Lifted lambda - we know its captures
                    StaticVal::Lambda {
                        lifted_name: sym,
                        captures: captures.clone(),
                    }
                } else if self.top_level.contains(&sym) && is_arrow_type(&ty) {
                    // Top-level function reference - treat as a Lambda with no captures
                    // This allows HOF specialization to work for named function references
                    StaticVal::Lambda {
                        lifted_name: sym,
                        captures: vec![],
                    }
                } else {
                    StaticVal::Dynamic
                };
                DefuncResult { term, sv }
            }

            // Catalog builtin reference: not a static lambda for defunc
            // purposes; flows through as `Dynamic`.
            TermKind::Var(crate::tlc::VarRef::Builtin(_)) => DefuncResult {
                term,
                sv: StaticVal::Dynamic,
            },

            TermKind::Lambda(..) => self.defunc_lambda(term),

            TermKind::App { func, args } => self.defunc_app(*func, args, ty, span),

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                // Defunc rhs and track its static value
                let rhs_result = self.defunc_term(*rhs);

                // Bind name -> StaticVal in environment
                self.env.insert(name, rhs_result.sv.clone());

                // Defunc body with updated environment
                let body_result = self.defunc_term(*body);

                // Remove binding
                self.env.remove(&name);

                DefuncResult {
                    term: Term {
                        id: self.term_ids.next_id(),
                        ty,
                        span,
                        kind: TermKind::Let {
                            name,
                            name_ty,
                            rhs: Box::new(rhs_result.term),
                            body: Box::new(body_result.term),
                        },
                    },
                    sv: body_result.sv,
                }
            }

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_result = self.defunc_term(*cond);
                let then_result = self.defunc_term(*then_branch);
                let else_result = self.defunc_term(*else_branch);

                DefuncResult {
                    term: Term {
                        id: self.term_ids.next_id(),
                        ty,
                        span,
                        kind: TermKind::If {
                            cond: Box::new(cond_result.term),
                            then_branch: Box::new(then_result.term),
                            else_branch: Box::new(else_result.term),
                        },
                    },
                    sv: StaticVal::Dynamic, // Conditionals are dynamic
                }
            }

            TermKind::Loop {
                loop_var,
                loop_var_ty,
                init,
                init_bindings,
                kind,
                body,
            } => {
                // Defunc init (outside loop scope)
                let init_result = self.defunc_term(*init);

                // Set up environment with loop_var as Dynamic
                self.env.insert(loop_var, StaticVal::Dynamic);

                // Defunc init_bindings and add them to env
                let defunc_init_bindings: Vec<_> = init_bindings
                    .into_iter()
                    .map(|(name, binding_ty, expr)| {
                        let expr_result = self.defunc_term(expr);
                        self.env.insert(name, StaticVal::Dynamic);
                        (name, binding_ty, expr_result.term)
                    })
                    .collect();

                // Defunc kind (iter/bound/cond depending on variant)
                let defunc_kind = match kind {
                    LoopKind::For { var, var_ty, iter } => {
                        let iter_result = self.defunc_term(*iter);
                        self.env.insert(var, StaticVal::Dynamic);
                        LoopKind::For {
                            var,
                            var_ty,
                            iter: Box::new(iter_result.term),
                        }
                    }
                    LoopKind::ForRange { var, var_ty, bound } => {
                        let bound_result = self.defunc_term(*bound);
                        self.env.insert(var, StaticVal::Dynamic);
                        LoopKind::ForRange {
                            var,
                            var_ty,
                            bound: Box::new(bound_result.term),
                        }
                    }
                    LoopKind::While { cond } => {
                        let cond_result = self.defunc_term(*cond);
                        LoopKind::While {
                            cond: Box::new(cond_result.term),
                        }
                    }
                };

                // Defunc body
                let body_result = self.defunc_term(*body);

                // Clean up environment
                self.env.remove(&loop_var);
                for (name, _, _) in &defunc_init_bindings {
                    self.env.remove(name);
                }
                if let LoopKind::For { ref var, .. } | LoopKind::ForRange { ref var, .. } = defunc_kind {
                    self.env.remove(var);
                }

                DefuncResult {
                    term: Term {
                        id: self.term_ids.next_id(),
                        ty,
                        span,
                        kind: TermKind::Loop {
                            loop_var,
                            loop_var_ty,
                            init: Box::new(init_result.term),
                            init_bindings: defunc_init_bindings,
                            kind: defunc_kind,
                            body: Box::new(body_result.term),
                        },
                    },
                    sv: StaticVal::Dynamic, // Loops are dynamic
                }
            }

            // Literals, operators, and extern declarations are dynamic
            TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_) => DefuncResult {
                term,
                sv: StaticVal::Dynamic,
            },

            TermKind::Soac(soac) => self.defunc_soac(soac, ty, span),

            TermKind::ArrayExpr(ae) => self.defunc_array_expr_term(ae, ty, span),

            TermKind::Force(inner) => {
                let result = self.defunc_term(*inner);
                DefuncResult {
                    term: Term {
                        id: self.term_ids.next_id(),
                        ty,
                        span,
                        kind: TermKind::Force(Box::new(result.term)),
                    },
                    sv: StaticVal::Dynamic,
                }
            }
        }
    }

    /// Handle lambda: lift to top-level, return Lambda StaticVal.
    fn defunc_lambda(&mut self, term: Term) -> DefuncResult {
        let ty = term.ty.clone();
        let span = term.span;

        // Extract all nested lambda params
        let (params, inner_body) = self.extract_lambda_params(term);

        // Push params into env as Dynamic
        for (p, _) in &params {
            self.env.insert(*p, StaticVal::Dynamic);
        }

        // Defunc the body
        let body_result = self.defunc_term(inner_body);

        // Pop params from env
        for (p, _) in &params {
            self.env.remove(p);
        }

        // Compute free variables (captures)
        let bound: HashSet<SymbolId> = params.iter().map(|(p, _)| *p).collect();
        let captures = compute_free_vars(
            &body_result.term,
            &bound,
            &self.top_level,
            self.known_defs,
            &self.symbols,
        );

        // Rebuild nested lambdas from inside out
        let rebuilt_lam =
            super::closure_convert::rebuild_nested_lam(&params, body_result.term, span, &mut self.term_ids);

        // Lift to top-level
        let lifted_sym = self.fresh_symbol();

        // Record captures for this lifted lambda (used during HOF specialization)
        self.lifted_lambda_captures.insert(lifted_sym, captures.clone());
        // Register as top-level so subsequent free-var passes don't
        // mis-classify it as a capture of an enclosing lambda.
        self.top_level.insert(lifted_sym);

        if captures.is_empty() {
            // No captures: lift as-is
            self.lifted_defs.push(Def {
                name: lifted_sym,
                ty: rebuilt_lam.ty.clone(),
                body: rebuilt_lam,
                meta: DefMeta::Function,
                arity: params.len(),
            });

            // Return reference to lifted function
            DefuncResult {
                term: Term {
                    id: self.term_ids.next_id(),
                    ty,
                    span,
                    kind: TermKind::Var(crate::tlc::VarRef::Symbol(lifted_sym)),
                },
                sv: StaticVal::Lambda {
                    lifted_name: lifted_sym,
                    captures: vec![],
                },
            }
        } else {
            // Has captures: append captures as additional parameters at the end
            // Build: |original_params...| |captures...| body
            let cap_params: Vec<(SymbolId, Type<TypeName>)> = captures
                .iter()
                .map(|cap_term| match &cap_term.kind {
                    TermKind::Var(crate::tlc::VarRef::Symbol(sym)) => (*sym, cap_term.ty.clone()),
                    other => panic!(
                        "compute_free_vars contract violated: capture is not a Var: {:?}",
                        other
                    ),
                })
                .collect();
            let wrapped = super::closure_convert::append_capture_params(
                rebuilt_lam,
                &cap_params,
                span,
                &mut self.term_ids,
            );
            let arity = params.len() + captures.len();

            self.lifted_defs.push(Def {
                name: lifted_sym,
                ty: wrapped.ty.clone(),
                body: wrapped,
                meta: DefMeta::Function,
                arity,
            });

            // Return reference to lifted function (NOT a partial application!)
            // The StaticVal tracks the captures that need to be applied
            DefuncResult {
                term: Term {
                    id: self.term_ids.next_id(),
                    ty,
                    span,
                    kind: TermKind::Var(crate::tlc::VarRef::Symbol(lifted_sym)),
                },
                sv: StaticVal::Lambda {
                    lifted_name: lifted_sym,
                    captures,
                },
            }
        }
    }

    /// Defunctionalize a SOAC node: lift lambdas inside, resolve captures.
    fn defunc_soac(&mut self, soac: SoacOp, ty: Type<TypeName>, span: Span) -> DefuncResult {
        // Pre-defunc, SoacBody.captures is empty; we drop it and the
        // call below re-derives captures during lambda lifting.
        let new_soac = match soac {
            SoacOp::Map {
                lam,
                inputs,
                consumes_input,
            } => {
                let lam = self.defunc_lambda_in_soac(lam.lam, span);
                let inputs = inputs.into_iter().map(|ae| self.defunc_array_expr(ae)).collect();
                SoacOp::Map {
                    lam,
                    inputs,
                    consumes_input,
                }
            }
            SoacOp::Reduce { op, ne, input, props } => {
                let op = self.defunc_lambda_in_soac(op.lam, span);
                let ne = Box::new(self.defunc_term(*ne).term);
                let input = self.defunc_array_expr(input);
                SoacOp::Reduce { op, ne, input, props }
            }
            SoacOp::Scan { op, ne, input } => {
                let op = self.defunc_lambda_in_soac(op.lam, span);
                let ne = Box::new(self.defunc_term(*ne).term);
                let input = self.defunc_array_expr(input);
                SoacOp::Scan { op, ne, input }
            }
            SoacOp::Filter { pred, input } => {
                let pred = self.defunc_lambda_in_soac(pred.lam, span);
                let input = self.defunc_array_expr(input);
                SoacOp::Filter { pred, input }
            }
            SoacOp::Scatter {
                dest,
                indices,
                values,
            } => {
                let indices = self.defunc_array_expr(indices);
                let values = self.defunc_array_expr(values);
                SoacOp::Scatter {
                    dest,
                    indices,
                    values,
                }
            }
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
                props,
            } => {
                let op = self.defunc_lambda_in_soac(op.lam, span);
                let ne = Box::new(self.defunc_term(*ne).term);
                let indices = self.defunc_array_expr(indices);
                let values = self.defunc_array_expr(values);
                SoacOp::ReduceByIndex {
                    dest,
                    op,
                    ne,
                    indices,
                    values,
                    props,
                }
            }
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
                props,
            } => {
                let op = self.defunc_lambda_in_soac(op.lam, span);
                let reduce_op = self.defunc_lambda_in_soac(reduce_op.lam, span);
                let ne = Box::new(self.defunc_term(*ne).term);
                let inputs = inputs.into_iter().map(|ae| self.defunc_array_expr(ae)).collect();
                SoacOp::Redomap {
                    op,
                    reduce_op,
                    ne,
                    inputs,
                    props,
                }
            }
        };
        DefuncResult {
            term: Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Soac(new_soac),
            },
            sv: StaticVal::Dynamic,
        }
    }

    /// Defunctionalize a lambda within a SOAC node.
    ///
    /// This lifts the lambda body to a top-level Def (same mechanism as defunc_lambda)
    /// and fills `SoacBody.captures` with the resolved capture terms. The resulting
    /// SoacBody has its body replaced with a reference to the lifted function and
    /// captures populated.
    fn defunc_lambda_in_soac(&mut self, lam: Lambda, span: Span) -> super::SoacBody {
        // Build a term from the lambda so we can use defunc_lambda
        let lam_ty = if lam.params.len() == 1 {
            Type::Constructed(TypeName::Arrow, vec![lam.params[0].1.clone(), lam.ret_ty.clone()])
        } else {
            // Multi-param: build nested arrow
            let mut ty = lam.ret_ty.clone();
            for (_, param_ty) in lam.params.iter().rev() {
                ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), ty]);
            }
            ty
        };

        let lam_term = Term {
            id: self.term_ids.next_id(),
            ty: lam_ty,
            span,
            kind: TermKind::Lambda(lam),
        };

        let result = self.defunc_lambda(lam_term);

        // Extract the lifted function name and captures from the StaticVal
        let (lifted_name, capture_terms) = match &result.sv {
            StaticVal::Lambda {
                lifted_name,
                captures,
            } => (*lifted_name, captures.clone()),
            _ => {
                // This shouldn't happen for a lambda, but be safe
                match &result.term.kind {
                    TermKind::Var(crate::tlc::VarRef::Symbol(sym)) => (*sym, vec![]),
                    _ => panic!("BUG: defunc_lambda didn't produce a Var or Lambda StaticVal"),
                }
            }
        };

        // Build the captures list: (sym, ty, term) triples
        let captures: Vec<(SymbolId, Type<TypeName>, Term)> = capture_terms
            .into_iter()
            .map(|t| {
                let sym = match &t.kind {
                    TermKind::Var(crate::tlc::VarRef::Symbol(s)) => *s,
                    _ => panic!("BUG: capture is not a Var: {:?}", t.kind),
                };
                let ty = t.ty.clone();
                (sym, ty, t)
            })
            .collect();

        // Retrieve the lifted def to get the body
        // The lifted lambda body is a reference to the lifted function
        let body = Term {
            id: self.term_ids.next_id(),
            ty: result.term.ty.clone(),
            span,
            kind: TermKind::Var(crate::tlc::VarRef::Symbol(lifted_name)),
        };

        // Get the params from the lifted def
        let lifted_def = self.lifted_defs.iter().find(|d| d.name == lifted_name);
        let params = if let Some(def) = lifted_def {
            // Extract original params (not captures) from the lifted def
            let (all_params, _) = self.extract_lambda_params(def.body.clone());
            // Original params are the first N params, captures are trailing
            let n_captures = captures.len();
            let n_orig = all_params.len().saturating_sub(n_captures);
            all_params[..n_orig].to_vec()
        } else {
            // Fallback — shouldn't happen
            vec![]
        };

        let ret_ty = if let Some(def) = self.lifted_defs.iter().find(|d| d.name == lifted_name) {
            // Walk the def type to get the final return type after all params
            let mut ty = &def.ty;
            for _ in 0..params.len() + captures.len() {
                if let Type::Constructed(TypeName::Arrow, args) = ty {
                    ty = &args[1];
                } else {
                    break;
                }
            }
            ty.clone()
        } else {
            Type::Constructed(TypeName::Unit, vec![])
        };

        super::SoacBody {
            lam: Lambda {
                params,
                body: Box::new(body),
                ret_ty,
            },
            captures,
        }
    }

    /// Defunctionalize an ArrayExpr (returns transformed ArrayExpr).
    fn defunc_array_expr(&mut self, ae: ArrayExpr) -> ArrayExpr {
        match ae {
            ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(self.defunc_term(*t).term)),
            ArrayExpr::Zip(exprs) => {
                ArrayExpr::Zip(exprs.into_iter().map(|e| self.defunc_array_expr(e)).collect())
            }
            ArrayExpr::Soac(op) => {
                // Create a dummy type/span for the recursive call
                let result = self.defunc_soac(
                    *op,
                    Type::Constructed(TypeName::Unit, vec![]),
                    Span::new(0, 0, 0, 0),
                );
                match result.term.kind {
                    TermKind::Soac(s) => ArrayExpr::Soac(Box::new(s)),
                    _ => unreachable!(),
                }
            }
            ArrayExpr::Generate {
                shape,
                index_fn,
                elem_ty,
            } => {
                // Pre-defunc, captures are empty: drop them and re-defunc the lambda.
                let index_fn = self.defunc_lambda_in_soac(index_fn.lam, Span::new(0, 0, 0, 0));
                ArrayExpr::Generate {
                    shape,
                    index_fn,
                    elem_ty,
                }
            }
            ArrayExpr::Literal(terms) => {
                ArrayExpr::Literal(terms.into_iter().map(|t| self.defunc_term(t).term).collect())
            }
            ArrayExpr::Range { start, len } => ArrayExpr::Range {
                start: Box::new(self.defunc_term(*start).term),
                len: Box::new(self.defunc_term(*len).term),
            },
            ArrayExpr::StorageBuffer { .. } => {
                unreachable!("StorageBuffer introduced after defunctionalization")
            }
        }
    }

    /// Defunctionalize an ArrayExpr wrapped in a term.
    fn defunc_array_expr_term(&mut self, ae: ArrayExpr, ty: Type<TypeName>, span: Span) -> DefuncResult {
        let new_ae = self.defunc_array_expr(ae);
        DefuncResult {
            term: Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::ArrayExpr(new_ae),
            },
            sv: StaticVal::Dynamic,
        }
    }

    /// Handle application: flatten captures.
    fn defunc_app(&mut self, func: Term, args: Vec<Term>, ty: Type<TypeName>, span: Span) -> DefuncResult {
        let base_func = func;

        // Defunctionalize all arguments
        let arg_results: Vec<DefuncResult> = args.into_iter().map(|a| self.defunc_term(a)).collect();
        let arg_terms: Vec<Term> = arg_results.iter().map(|ar| ar.term.clone()).collect();

        // Handle based on the function term kind
        match &base_func.kind {
            TermKind::BinOp(_) | TermKind::UnOp(_) => {
                // Operators are preserved as-is - just rebuild the application
                let result_term = super::closure_convert::build_app_with_term(
                    base_func,
                    arg_terms,
                    ty.clone(),
                    span,
                    &mut self.term_ids,
                );
                DefuncResult {
                    term: result_term,
                    sv: StaticVal::Dynamic,
                }
            }

            TermKind::Var(crate::tlc::VarRef::Symbol(sym)) => {
                let sym = *sym; // Copy out of the match to avoid borrow issues
                // Get static value for the callee. Lookup mirrors
                // `defunc_term`'s Var arm: env wins (binding-scoped),
                // then known lifted-lambda captures, then top-level
                // names. Without the `lifted_lambda_captures` step,
                // calls to a lifted closure inside a specialized HOF
                // body would fall through to `Dynamic` and lose the
                // captures that need to be threaded through.
                let callee_sv = self.env.get(&sym).cloned().unwrap_or_else(|| {
                    if let Some(captures) = self.lifted_lambda_captures.get(&sym) {
                        StaticVal::Lambda {
                            lifted_name: sym,
                            captures: captures.clone(),
                        }
                    } else if self.top_level.contains(&sym) && is_arrow_type(&base_func.ty) {
                        StaticVal::Lambda {
                            lifted_name: sym,
                            captures: vec![],
                        }
                    } else {
                        StaticVal::Dynamic
                    }
                });

                // Check if this is a call to a HOF (user-defined or intrinsic) with a lambda argument
                // We specialize for ALL lambda arguments (even non-capturing ones) because
                // SPIRV doesn't support passing functions as values - they must be called directly

                // Check HOF info (user-defined only; intrinsic SOACs are now SOAC nodes)
                if let Some(hof_info) = self.hof_info.get(&sym).cloned() {
                    for &func_param_idx in &hof_info.func_param_indices {
                        if func_param_idx < arg_results.len() {
                            if let StaticVal::Lambda { .. } = &arg_results[func_param_idx].sv {
                                if hof_info.def.is_some() {
                                    return self.handle_hof_call(
                                        sym,
                                        &hof_info,
                                        func_param_idx,
                                        &arg_results,
                                        ty,
                                        span,
                                    );
                                }
                            }
                        }
                    }
                }

                // Use unified apply_callable
                let result_term = self.apply_callable(base_func, &callee_sv, arg_terms, ty.clone(), span);
                DefuncResult {
                    term: result_term,
                    sv: StaticVal::Dynamic,
                }
            }

            _ => {
                // Other computed function term - defunc it and use apply_callable
                let func_result = self.defunc_term(base_func);
                let result_term =
                    self.apply_callable(func_result.term, &func_result.sv, arg_terms, ty.clone(), span);
                DefuncResult {
                    term: result_term,
                    sv: StaticVal::Dynamic,
                }
            }
        }
    }

    /// Unified function call handler.
    ///
    /// If callee_sv is Lambda with captures: emit call to lifted_name with args + capture_terms
    /// Otherwise: emit call to callee_term with args
    fn apply_callable(
        &mut self,
        callee_term: Term,
        callee_sv: &StaticVal,
        args: Vec<Term>,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        match callee_sv {
            StaticVal::Lambda {
                lifted_name,
                captures,
            } if !captures.is_empty() => {
                // Function has captures - call lifted_name with args + captures
                let mut all_args = args;
                all_args.extend(captures.iter().cloned());
                super::closure_convert::build_app_call(
                    *lifted_name,
                    all_args,
                    result_ty,
                    span,
                    &mut self.term_ids,
                )
            }
            _ => {
                // No captures - call callee_term directly with args
                super::closure_convert::build_app_with_term(
                    callee_term,
                    args,
                    result_ty,
                    span,
                    &mut self.term_ids,
                )
            }
        }
    }

    /// Extract all nested lambda parameters from a term.
    fn extract_lambda_params(&self, term: Term) -> (Vec<(SymbolId, Type<TypeName>)>, Term) {
        super::extract_lambda_params(&term)
    }

    // =========================================================================
    // HOF Handling (User-defined and Intrinsic)
    // =========================================================================

    /// Handle a user-defined HOF call with a lambda argument.
    /// User-defined HOFs get specialized by cloning and substituting.
    fn handle_hof_call(
        &mut self,
        hof_sym: SymbolId,
        hof_info: &HofInfo,
        func_param_idx: usize,
        arg_results: &[DefuncResult],
        ty: Type<TypeName>,
        span: Span,
    ) -> DefuncResult {
        // Extract lambda info from function argument
        let (lambda_sym, captures) = match &arg_results[func_param_idx].sv {
            StaticVal::Lambda {
                lifted_name,
                captures,
            } => (*lifted_name, captures.clone()),
            _ => unreachable!("handle_hof_call called without Lambda StaticVal"),
        };

        let def = hof_info.def.as_ref().expect("BUG: user HOF should have def");
        self.specialize_user_hof(
            hof_sym,
            def,
            func_param_idx,
            lambda_sym,
            &captures,
            arg_results,
            ty,
            span,
        )
    }

    /// Specialize a user-defined HOF call by cloning, substituting, and defunctionalizing.
    ///
    /// This is the fixpoint-enabled version: after substituting the function parameter
    /// with the lambda name, we defunc the body. This handles:
    /// - Capture appending for calls to the lambda (via apply_callable)
    /// - Nested HOF calls with lambda args (triggers recursive specialization)
    /// - New lambdas in the specialized body (get lifted)
    fn specialize_user_hof(
        &mut self,
        hof_sym: SymbolId,
        hof_def: &Def,
        func_param_idx: usize,
        lambda_sym: SymbolId,
        captures: &[Term],
        arg_results: &[DefuncResult],
        ty: Type<TypeName>,
        span: Span,
    ) -> DefuncResult {
        // Check cache (include arg types to handle different type instantiations)
        let arg_type_keys: Vec<String> =
            arg_results.iter().map(|ar| format_type_for_key(&ar.term.ty)).collect();
        let cache_key = (hof_sym, lambda_sym, arg_type_keys);
        if let Some(specialized_sym) = self.specialization_cache.get(&cache_key).cloned() {
            let arg_terms: Vec<Term> = arg_results.iter().map(|ar| ar.term.clone()).collect();
            let call_term = super::hof_specialize::build_specialized_call(
                specialized_sym,
                func_param_idx,
                &arg_terms,
                captures,
                ty,
                span,
                &mut self.term_ids,
            );
            return DefuncResult {
                term: call_term,
                sv: StaticVal::Dynamic,
            };
        }

        // Build type substitution from polymorphic params to concrete types
        let mut type_subst = TypeSubst::new();
        let poly_param_types = extract_param_types(&hof_def.ty);
        for (i, poly_ty) in poly_param_types.iter().enumerate() {
            if i < arg_results.len() {
                build_type_subst(poly_ty, &arg_results[i].term.ty, &mut type_subst);
            }
        }

        // Generate specialized variant symbol
        let hof_name = self.symbols.get(hof_sym).expect("BUG: HOF symbol not in table");
        let specialized_name = format!("{}${}", hof_name, self.specialization_counter);
        self.specialization_counter += 1;
        let specialized_sym = self.symbols.alloc(specialized_name.clone());

        // Cache early to prevent infinite recursion on mutually recursive HOFs
        self.specialization_cache.insert(cache_key, specialized_sym);

        // Get the function parameter symbol from the HOF definition
        let func_param_sym = super::hof_specialize::get_func_param_sym(hof_def, func_param_idx);

        // Extract params FIRST, then substitute in the inner body
        // (Substitution must happen AFTER unwrapping lambdas because the func param
        // is bound by one of those lambdas, and substitute_var correctly avoids
        // substituting in shadowed scopes)
        let (params, inner_body) = self.extract_lambda_params(hof_def.body.clone());

        // Simple substitution in inner body: replace func_param_sym with lambda_sym
        let substituted_inner = super::hof_specialize::substitute_var(
            &inner_body,
            func_param_sym,
            lambda_sym,
            &mut self.term_ids,
        );

        // Build new param list: remove function param at func_param_idx, add captures at end.
        //
        // Captures are alpha-renamed: each outer-scope symbol used as a
        // capture gets a fresh symbol id for its role as a parameter
        // of the specialized HOF. Reusing the outer id directly is
        // tempting (the term tree stays internally consistent) but
        // brittle — downstream codegen classifies symbols against the
        // original def/env/global tables, so an entry-point parameter
        // captured into a lifted HOF still "looks like" the entry
        // parameter to SPIR-V lowering and surfaces as
        // `Unknown global`. Fresh local ids give the specialized
        // function its own clean lexical scope.
        //
        // The substitution map is also applied to the defunced body
        // below, so any reference to the outer capture (including
        // those introduced by `apply_callable` when threading captures
        // into a lifted closure call) routes through the fresh local
        // parameter.
        let mut new_params: Vec<(SymbolId, Type<TypeName>)> =
            params.into_iter().enumerate().filter(|(i, _)| *i != func_param_idx).map(|(_, p)| p).collect();

        let mut capture_subst: Vec<(SymbolId, SymbolId)> = Vec::with_capacity(captures.len());
        for cap_term in captures {
            let outer_sym = match &cap_term.kind {
                TermKind::Var(crate::tlc::VarRef::Symbol(sym)) => *sym,
                _ => panic!("BUG: capture term is not a Var: {:?}", cap_term.kind),
            };
            let outer_name =
                self.symbols.get(outer_sym).cloned().unwrap_or_else(|| format!("cap{}", outer_sym.0));
            let fresh_sym = self.symbols.alloc(format!("{}__cap_{}", specialized_name, outer_name));
            capture_subst.push((outer_sym, fresh_sym));
            new_params.push((fresh_sym, cap_term.ty.clone()));
        }

        // Set up environment for defunc with new params as Dynamic
        let old_env = std::mem::take(&mut self.env);
        for (param_sym, _) in &new_params {
            self.env.insert(*param_sym, StaticVal::Dynamic);
        }

        // Defunc the substituted inner body - this handles:
        // - Capture appending for calls to lambda_sym (via apply_callable + lifted_lambda_captures)
        // - Nested HOF calls with lambda args (triggers recursive specialization)
        // - New lambdas (get lifted)
        let mut defunced_body = self.defunc_term(substituted_inner).term;

        // Rewrite outer capture symbols to the fresh local parameters.
        // Done after defunc so the synthesized `f(x, outer_a, outer_b)`
        // calls produced by `apply_callable` capture-threading also get
        // rewritten to use the fresh params. `substitute_var` is
        // binder-aware (Lambda / Let / Loop), so any inner shadowing
        // is respected.
        for (outer_sym, fresh_sym) in &capture_subst {
            defunced_body = super::hof_specialize::substitute_var(
                &defunced_body,
                *outer_sym,
                *fresh_sym,
                &mut self.term_ids,
            );
        }

        // Restore env
        self.env = old_env;

        // Apply type substitution to the defunced body AND to the param types
        let defunced_body = apply_type_subst_to_term(&defunced_body, &type_subst, &mut self.term_ids);
        let new_params: Vec<(SymbolId, Type<TypeName>)> =
            new_params.into_iter().map(|(sym, ty)| (sym, apply_type_subst(&ty, &type_subst))).collect();

        // Rebuild nested lambdas with substituted params
        let rebuilt = super::closure_convert::rebuild_nested_lam(
            &new_params,
            defunced_body,
            hof_def.body.span,
            &mut self.term_ids,
        );

        let specialized_def = Def {
            name: specialized_sym,
            ty: rebuilt.ty.clone(),
            body: rebuilt,
            meta: DefMeta::Function,
            arity: new_params.len(),
        };

        // Register the specialized HOF as top-level so an enclosing
        // lambda's free-var pass doesn't pick it up as a capture (it
        // wasn't in `top_level` at startup because it's a fresh symbol
        // produced by this pass).
        self.top_level.insert(specialized_sym);
        self.lifted_defs.push(specialized_def);

        // Build call to specialized function
        let arg_terms: Vec<Term> = arg_results.iter().map(|ar| ar.term.clone()).collect();
        let call_term = super::hof_specialize::build_specialized_call(
            specialized_sym,
            func_param_idx,
            &arg_terms,
            captures,
            ty,
            span,
            &mut self.term_ids,
        );
        DefuncResult {
            term: call_term,
            sv: StaticVal::Dynamic,
        }
    }

    fn fresh_symbol(&mut self) -> SymbolId {
        let name = format!("_w_lambda_{}", self.lambda_counter);
        self.lambda_counter += 1;
        self.symbols.alloc(name)
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Defunctionalize a TLC program.
///
/// This combines lambda lifting with static value tracking:
/// - All lambdas are lifted to top-level definitions
/// - Captures become extra parameters (appended at end)
/// - All call sites have captures flattened as trailing arguments
pub fn run(program: Program, known_defs: &HashSet<String>) -> Program {
    let result = Defunctionalizer::run(program, known_defs);
    result.assert_flat_apps();
    super::closure_convert::verify_closure_converted(&result).unwrap_or_else(|e| {
        panic!(
            "closure-conversion verifier failed after defunctionalize: {:?}",
            e
        )
    });
    super::closure_calls_lower::verify_closure_calls_lowered(&result).unwrap_or_else(|e| {
        panic!(
            "closure-calls-lowered verifier failed after defunctionalize: {:?}",
            e
        )
    });
    result
}

#[cfg(test)]
#[path = "defunctionalize_tests.rs"]
mod defunctionalize_tests;
