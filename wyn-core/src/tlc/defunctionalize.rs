//! Defunctionalization for TLC (Futhark-style).
//!
//! This pass combines lambda lifting with static value tracking to properly
//! handle SOAC (Second-Order Array Combinator) call sites. Unlike simple
//! lambda lifting which creates partial applications, this pass:
//!
//! 1. Lifts lambdas to top-level definitions (with captures as extra params)
//! 2. Tracks StaticVal alongside term transformation
//! 3. Flattens SOAC closure captures as explicit arguments
//!
//! Example transformation:
//!   Input:  map(|x| x + y, arr)     where y is captured
//!   Output: _w_intrinsic_map _lambda_0 arr y
//!
//! The lifted lambda is: _lambda_0 = |y| |x| x + y

use super::{Def, DefMeta, FunctionName, Program, Term, TermIdSource, TermKind};
use crate::ast::{Span, TypeName};
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
    /// - The lifted function name
    /// - Captured free variables (name, type)
    Lambda {
        lifted_name: String,
        captures: Vec<(String, Type<TypeName>)>,
    },

    /// Tuple/record of static values (for destructuring)
    Record(Vec<StaticVal>),
}

// =============================================================================
// SOAC Detection
// =============================================================================

/// SOAC intrinsics that take a function as their first argument.
const SOAC_INTRINSICS: &[&str] = &[
    "_w_intrinsic_map",
    "_w_intrinsic_map2",
    "_w_intrinsic_map3",
    "_w_intrinsic_reduce",
    "_w_intrinsic_reduce_comm",
    "_w_intrinsic_scan",
    "_w_intrinsic_filter",
    "_w_intrinsic_scatter",
    "_w_intrinsic_hist_1d",
];

fn is_soac_intrinsic(name: &str) -> bool {
    SOAC_INTRINSICS.contains(&name)
}

/// Create a placeholder span for synthesized terms.
fn dummy_span() -> Span {
    Span {
        start_line: 0,
        start_col: 0,
        end_line: 0,
        end_col: 0,
    }
}

// =============================================================================
// Free Variable Analysis (reused from lift.rs)
// =============================================================================

/// Compute free variables of a term, given explicit sets of bound names.
fn compute_free_vars(
    term: &Term,
    bound: &HashSet<String>,
    top_level: &HashSet<String>,
    builtins: &HashSet<String>,
) -> Vec<(String, Type<TypeName>)> {
    let mut free = Vec::new();
    let mut seen = HashSet::new();
    collect_free_vars(term, bound, top_level, builtins, &mut free, &mut seen);
    free
}

fn collect_free_vars(
    term: &Term,
    bound: &HashSet<String>,
    top_level: &HashSet<String>,
    builtins: &HashSet<String>,
    free: &mut Vec<(String, Type<TypeName>)>,
    seen: &mut HashSet<String>,
) {
    match &term.kind {
        TermKind::Var(name) => {
            if !bound.contains(name)
                && !top_level.contains(name)
                && !builtins.contains(name)
                && !name.starts_with("_w_")
                && !seen.contains(name)
            {
                seen.insert(name.clone());
                free.push((name.clone(), term.ty.clone()));
            }
        }
        TermKind::Let { name, rhs, body, .. } => {
            collect_free_vars(rhs, bound, top_level, builtins, free, seen);
            let mut inner_bound = bound.clone();
            inner_bound.insert(name.clone());
            collect_free_vars(body, &inner_bound, top_level, builtins, free, seen);
        }
        TermKind::Lam { param, body, .. } => {
            let mut inner_bound = bound.clone();
            inner_bound.insert(param.clone());
            collect_free_vars(body, &inner_bound, top_level, builtins, free, seen);
        }
        TermKind::App { func, arg } => {
            collect_free_vars_func(func, bound, top_level, builtins, free, seen);
            collect_free_vars(arg, bound, top_level, builtins, free, seen);
        }
        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_free_vars(cond, bound, top_level, builtins, free, seen);
            collect_free_vars(then_branch, bound, top_level, builtins, free, seen);
            collect_free_vars(else_branch, bound, top_level, builtins, free, seen);
        }
        TermKind::IntLit(_) | TermKind::FloatLit(_) | TermKind::BoolLit(_) | TermKind::StringLit(_) => {}
    }
}

fn collect_free_vars_func(
    func: &FunctionName,
    bound: &HashSet<String>,
    top_level: &HashSet<String>,
    builtins: &HashSet<String>,
    free: &mut Vec<(String, Type<TypeName>)>,
    seen: &mut HashSet<String>,
) {
    match func {
        FunctionName::Term(t) => collect_free_vars(t, bound, top_level, builtins, free, seen),
        FunctionName::Var(_) | FunctionName::BinOp(_) | FunctionName::UnOp(_) => {}
    }
}

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
    /// Top-level definition names
    top_level: HashSet<String>,
    /// Built-in names (intrinsics, prelude functions)
    builtins: &'a HashSet<String>,
    /// New definitions created for lifted lambdas
    lifted_defs: Vec<Def>,
    /// Counter for generating unique lambda names
    lambda_counter: u32,
    /// Term ID generator
    term_ids: TermIdSource,
    /// Environment: variable name -> StaticVal
    env: HashMap<String, StaticVal>,
}

impl<'a> Defunctionalizer<'a> {
    /// Defunctionalize a program.
    pub fn defunctionalize(program: Program, builtins: &'a HashSet<String>) -> Program {
        let mut defunc = Self {
            top_level: program.defs.iter().map(|d| d.name.clone()).collect(),
            builtins,
            lifted_defs: vec![],
            lambda_counter: 0,
            term_ids: TermIdSource::new(),
            env: HashMap::new(),
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

        Program {
            defs: transformed_defs.into_iter().chain(defunc.lifted_defs).collect(),
            uniforms: program.uniforms,
            storage: program.storage,
        }
    }

    /// Defunctionalize but preserve outermost parameter lambdas (for entry points).
    fn defunc_preserving_params(&mut self, term: Term) -> Term {
        match term.kind {
            TermKind::Lam {
                param,
                param_ty,
                body,
            } => {
                // Keep this lambda, but recursively process its body
                // Mark param as Dynamic in env
                self.env.insert(param.clone(), StaticVal::Dynamic);
                let defunc_body = self.defunc_preserving_params(*body);
                self.env.remove(&param);

                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty,
                    span: term.span,
                    kind: TermKind::Lam {
                        param,
                        param_ty,
                        body: Box::new(defunc_body),
                    },
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
            TermKind::Var(ref name) => {
                // Look up static value from environment
                let sv = self.env.get(name).cloned().unwrap_or(StaticVal::Dynamic);
                DefuncResult { term, sv }
            }

            TermKind::Lam { .. } => self.defunc_lambda(term),

            TermKind::App { func, arg } => self.defunc_app(*func, *arg, ty, span),

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                // Defunc rhs and track its static value
                let rhs_result = self.defunc_term(*rhs);

                // Bind name -> StaticVal in environment
                self.env.insert(name.clone(), rhs_result.sv.clone());

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

            // Literals are dynamic
            TermKind::IntLit(_) | TermKind::FloatLit(_) | TermKind::BoolLit(_) | TermKind::StringLit(_) => {
                DefuncResult {
                    term,
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
            self.env.insert(p.clone(), StaticVal::Dynamic);
        }

        // Defunc the body
        let body_result = self.defunc_term(inner_body);

        // Pop params from env
        for (p, _) in &params {
            self.env.remove(p);
        }

        // Compute free variables (captures)
        let bound: HashSet<String> = params.iter().map(|(p, _)| p.clone()).collect();
        let captures = compute_free_vars(&body_result.term, &bound, &self.top_level, self.builtins);

        // Rebuild nested lambdas from inside out
        let rebuilt_lam = self.rebuild_nested_lam(&params, body_result.term, span);

        // Lift to top-level
        let lifted_name = self.fresh_name();

        if captures.is_empty() {
            // No captures: lift as-is
            self.lifted_defs.push(Def {
                name: lifted_name.clone(),
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
                    kind: TermKind::Var(lifted_name.clone()),
                },
                sv: StaticVal::Lambda {
                    lifted_name,
                    captures: vec![],
                },
            }
        } else {
            // Has captures: wrap with extra lambdas for captures
            let wrapped = self.wrap_lam_with_captures(rebuilt_lam, &captures, span);
            let arity = captures.len() + params.len();

            self.lifted_defs.push(Def {
                name: lifted_name.clone(),
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
                    kind: TermKind::Var(lifted_name.clone()),
                },
                sv: StaticVal::Lambda {
                    lifted_name,
                    captures,
                },
            }
        }
    }

    /// Handle application: collect spine, detect SOACs, flatten captures.
    fn defunc_app(
        &mut self,
        func: FunctionName,
        arg: Term,
        ty: Type<TypeName>,
        span: Span,
    ) -> DefuncResult {
        // Collect the application spine: f a1 a2 ... an
        let (base_func, args) = self.collect_spine(func, arg);

        // Defunctionalize all arguments
        let arg_results: Vec<DefuncResult> = args.into_iter().map(|a| self.defunc_term(a)).collect();

        // Check if this is a SOAC intrinsic
        if let FunctionName::Var(ref name) = base_func {
            if is_soac_intrinsic(name) {
                return self.defunc_soac(name.clone(), arg_results, ty, span);
            }
        }

        // Non-SOAC: handle based on the function type
        match base_func {
            FunctionName::BinOp(_) | FunctionName::UnOp(_) => {
                // Operators are preserved as-is - just rebuild the application
                let result_term =
                    self.rebuild_app_with_func_name(base_func, &arg_results, ty.clone(), span);
                DefuncResult {
                    term: result_term,
                    sv: StaticVal::Dynamic,
                }
            }

            FunctionName::Var(ref name) => {
                // Check if this variable has a Lambda StaticVal (captured function)
                let sv = self.env.get(name).cloned().unwrap_or(StaticVal::Dynamic);

                match sv {
                    StaticVal::Lambda {
                        lifted_name,
                        captures,
                    } if !captures.is_empty() => {
                        // Function has captures - prepend them to the argument list
                        let mut all_args = Vec::new();

                        // Add capture values first
                        for (cap_name, cap_ty) in &captures {
                            all_args.push(Term {
                                id: self.term_ids.next_id(),
                                ty: cap_ty.clone(),
                                span,
                                kind: TermKind::Var(cap_name.clone()),
                            });
                        }

                        // Add original arguments
                        for ar in &arg_results {
                            all_args.push(ar.term.clone());
                        }

                        // Build curried application to lifted function
                        let result_term = self.build_curried_app(&lifted_name, all_args, ty.clone(), span);
                        DefuncResult {
                            term: result_term,
                            sv: StaticVal::Dynamic,
                        }
                    }

                    _ => {
                        // No captures - rebuild as-is
                        let result_term = self.rebuild_app_with_func_name(
                            FunctionName::Var(name.clone()),
                            &arg_results,
                            ty.clone(),
                            span,
                        );
                        DefuncResult {
                            term: result_term,
                            sv: StaticVal::Dynamic,
                        }
                    }
                }
            }

            FunctionName::Term(t) => {
                // Defunc the term and check for Lambda StaticVal
                let func_result = self.defunc_term(*t);

                match &func_result.sv {
                    StaticVal::Lambda {
                        lifted_name,
                        captures,
                    } if !captures.is_empty() => {
                        // Function has captures - prepend them to the argument list
                        let mut all_args = Vec::new();

                        // Add capture values first
                        for (cap_name, cap_ty) in captures {
                            all_args.push(Term {
                                id: self.term_ids.next_id(),
                                ty: cap_ty.clone(),
                                span,
                                kind: TermKind::Var(cap_name.clone()),
                            });
                        }

                        // Add original arguments
                        for ar in &arg_results {
                            all_args.push(ar.term.clone());
                        }

                        // Build curried application to lifted function
                        let result_term = self.build_curried_app(lifted_name, all_args, ty.clone(), span);
                        DefuncResult {
                            term: result_term,
                            sv: StaticVal::Dynamic,
                        }
                    }

                    _ => {
                        // No captures - rebuild as-is
                        let result_term =
                            self.rebuild_app_chain(func_result.term, &arg_results, ty.clone(), span);
                        DefuncResult {
                            term: result_term,
                            sv: StaticVal::Dynamic,
                        }
                    }
                }
            }
        }
    }

    /// Special handling for SOAC applications.
    ///
    /// When the first argument is a Lambda StaticVal, we flatten the captures
    /// as additional arguments to the SOAC.
    fn defunc_soac(
        &mut self,
        soac_name: String,
        arg_results: Vec<DefuncResult>,
        ty: Type<TypeName>,
        span: Span,
    ) -> DefuncResult {
        if arg_results.is_empty() {
            // Shouldn't happen, but handle gracefully
            return DefuncResult {
                term: Term {
                    id: self.term_ids.next_id(),
                    ty,
                    span,
                    kind: TermKind::Var(soac_name),
                },
                sv: StaticVal::Dynamic,
            };
        }

        // Check if first argument has Lambda StaticVal
        let first_sv = &arg_results[0].sv;

        match first_sv {
            StaticVal::Lambda {
                lifted_name,
                captures,
            } => {
                // Flatten: soac (f cap1 cap2) arr  →  soac f arr cap1 cap2
                //
                // Build new argument list:
                // 1. Reference to lifted function (not the term which may have captures baked in)
                // 2. Original data arguments (args[1..])
                // 3. Capture values

                let mut new_args = Vec::new();

                // 1. Direct reference to lifted function
                let func_ref = Term {
                    id: self.term_ids.next_id(),
                    ty: arg_results[0].term.ty.clone(),
                    span,
                    kind: TermKind::Var(lifted_name.clone()),
                };
                new_args.push(func_ref);

                // 2. Data arguments (skip the function argument)
                for ar in arg_results.iter().skip(1) {
                    new_args.push(ar.term.clone());
                }

                // 3. Capture values
                for (cap_name, cap_ty) in captures {
                    new_args.push(Term {
                        id: self.term_ids.next_id(),
                        ty: cap_ty.clone(),
                        span,
                        kind: TermKind::Var(cap_name.clone()),
                    });
                }

                // Build curried application: soac f arr cap1 cap2 ...
                let result_term = self.build_curried_app(&soac_name, new_args, ty.clone(), span);

                DefuncResult {
                    term: result_term,
                    sv: StaticVal::Dynamic,
                }
            }

            StaticVal::Dynamic => {
                // Function is dynamic - just rebuild the application as-is
                // This handles cases like `map(f, arr)` where f is a parameter
                let result_term = self.rebuild_soac_app(&soac_name, &arg_results, ty.clone(), span);
                DefuncResult {
                    term: result_term,
                    sv: StaticVal::Dynamic,
                }
            }

            StaticVal::Record(_) => {
                // Shouldn't happen - records aren't functions
                panic!("BUG: Record passed as SOAC function argument");
            }
        }
    }
    /// Collect application spine: f a1 a2 ... -> (f, [a1, a2, ...])
    fn collect_spine(&self, func: FunctionName, arg: Term) -> (FunctionName, Vec<Term>) {
        let mut args = vec![arg];
        let mut current_func = func;

        // Walk up the spine
        while let FunctionName::Term(t) = current_func {
            if let TermKind::App {
                func: inner_func,
                arg: inner_arg,
            } = t.kind
            {
                args.push(*inner_arg);
                current_func = *inner_func;
            } else {
                // Hit a non-App term in function position
                current_func = FunctionName::Term(t);
                break;
            }
        }

        // Args were collected in reverse order
        args.reverse();
        (current_func, args)
    }

    /// Build a curried application: f a1 a2 a3 ...
    fn build_curried_app(
        &mut self,
        func_name: &str,
        args: Vec<Term>,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        if args.is_empty() {
            return Term {
                id: self.term_ids.next_id(),
                ty: result_ty,
                span,
                kind: TermKind::Var(func_name.to_string()),
            };
        }

        // Build from left to right: ((f a1) a2) a3 ...
        let mut current = Term {
            id: self.term_ids.next_id(),
            ty: Type::Variable(0), // Placeholder
            span,
            kind: TermKind::App {
                func: Box::new(FunctionName::Var(func_name.to_string())),
                arg: Box::new(args[0].clone()),
            },
        };

        for arg in args.into_iter().skip(1) {
            current = Term {
                id: self.term_ids.next_id(),
                ty: Type::Variable(0), // Placeholder
                span,
                kind: TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(current))),
                    arg: Box::new(arg),
                },
            };
        }

        // Fix up the final type
        Term {
            ty: result_ty,
            ..current
        }
    }

    /// Rebuild an application chain from defunc results.
    fn rebuild_app_chain(
        &mut self,
        func_term: Term,
        arg_results: &[DefuncResult],
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        if arg_results.is_empty() {
            return func_term;
        }

        let mut current = Term {
            id: self.term_ids.next_id(),
            ty: Type::Variable(0),
            span,
            kind: TermKind::App {
                func: Box::new(FunctionName::Term(Box::new(func_term))),
                arg: Box::new(arg_results[0].term.clone()),
            },
        };

        for ar in arg_results.iter().skip(1) {
            current = Term {
                id: self.term_ids.next_id(),
                ty: Type::Variable(0),
                span,
                kind: TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(current))),
                    arg: Box::new(ar.term.clone()),
                },
            };
        }

        Term {
            ty: result_ty,
            ..current
        }
    }

    /// Rebuild an application with a FunctionName (preserves BinOp/UnOp).
    fn rebuild_app_with_func_name(
        &mut self,
        func: FunctionName,
        arg_results: &[DefuncResult],
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        if arg_results.is_empty() {
            // No args - return the function as a term (for Var only)
            match func {
                FunctionName::Var(name) => {
                    return Term {
                        id: self.term_ids.next_id(),
                        ty: result_ty,
                        span,
                        kind: TermKind::Var(name),
                    };
                }
                _ => {
                    // Can't return BinOp/UnOp without args
                    panic!("BUG: BinOp/UnOp with no arguments");
                }
            }
        }

        // Build first application with the original FunctionName
        let mut current = Term {
            id: self.term_ids.next_id(),
            ty: Type::Variable(0),
            span,
            kind: TermKind::App {
                func: Box::new(func),
                arg: Box::new(arg_results[0].term.clone()),
            },
        };

        // Chain remaining arguments
        for ar in arg_results.iter().skip(1) {
            current = Term {
                id: self.term_ids.next_id(),
                ty: Type::Variable(0),
                span,
                kind: TermKind::App {
                    func: Box::new(FunctionName::Term(Box::new(current))),
                    arg: Box::new(ar.term.clone()),
                },
            };
        }

        Term {
            ty: result_ty,
            ..current
        }
    }

    /// Rebuild a SOAC application without flattening (for dynamic function args).
    fn rebuild_soac_app(
        &mut self,
        soac_name: &str,
        arg_results: &[DefuncResult],
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let args: Vec<Term> = arg_results.iter().map(|ar| ar.term.clone()).collect();
        self.build_curried_app(soac_name, args, result_ty, span)
    }

    /// Extract all nested lambda parameters from a term.
    fn extract_lambda_params(&self, term: Term) -> (Vec<(String, Type<TypeName>)>, Term) {
        let mut params = Vec::new();
        let mut current = term;

        while let TermKind::Lam {
            param,
            param_ty,
            body,
        } = current.kind
        {
            params.push((param, param_ty));
            current = *body;
        }

        (params, current)
    }

    /// Rebuild nested lambdas from params and body.
    fn rebuild_nested_lam(&mut self, params: &[(String, Type<TypeName>)], body: Term, span: Span) -> Term {
        params.iter().rev().fold(body, |acc, (param, param_ty)| {
            let lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), acc.ty.clone()]);
            Term {
                id: self.term_ids.next_id(),
                ty: lam_ty,
                span,
                kind: TermKind::Lam {
                    param: param.clone(),
                    param_ty: param_ty.clone(),
                    body: Box::new(acc),
                },
            }
        })
    }

    /// Wrap a lambda with extra lambdas for captures (captures first).
    fn wrap_lam_with_captures(
        &mut self,
        lam: Term,
        captures: &[(String, Type<TypeName>)],
        span: Span,
    ) -> Term {
        captures.iter().rev().fold(lam, |acc, (cap_name, cap_ty)| {
            let result_ty = Type::Constructed(TypeName::Arrow, vec![cap_ty.clone(), acc.ty.clone()]);
            Term {
                id: self.term_ids.next_id(),
                ty: result_ty,
                span,
                kind: TermKind::Lam {
                    param: cap_name.clone(),
                    param_ty: cap_ty.clone(),
                    body: Box::new(acc),
                },
            }
        })
    }

    fn fresh_name(&mut self) -> String {
        let name = format!("_lambda_{}", self.lambda_counter);
        self.lambda_counter += 1;
        name
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Defunctionalize a TLC program.
///
/// This combines lambda lifting with SOAC call site rewriting:
/// - All lambdas are lifted to top-level definitions
/// - Captures become extra parameters (prepended)
/// - SOAC function arguments have their captures flattened as trailing args
pub fn defunctionalize(program: Program, builtins: &HashSet<String>) -> Program {
    Defunctionalizer::defunctionalize(program, builtins)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_span() -> Span {
        Span::dummy()
    }

    fn i32_ty() -> Type<TypeName> {
        Type::Constructed(TypeName::Int(32), vec![])
    }

    fn array_ty(elem: Type<TypeName>) -> Type<TypeName> {
        Type::Constructed(TypeName::Array, vec![elem])
    }

    fn arrow(from: Type<TypeName>, to: Type<TypeName>) -> Type<TypeName> {
        Type::Constructed(TypeName::Arrow, vec![from, to])
    }

    #[test]
    fn test_defunc_simple_lambda_no_capture() {
        // def f = |x| x
        let mut ids = TermIdSource::new();

        let lam = Term {
            id: ids.next_id(),
            ty: arrow(i32_ty(), i32_ty()),
            span: dummy_span(),
            kind: TermKind::Lam {
                param: "x".to_string(),
                param_ty: i32_ty(),
                body: Box::new(Term {
                    id: ids.next_id(),
                    ty: i32_ty(),
                    span: dummy_span(),
                    kind: TermKind::Var("x".to_string()),
                }),
            },
        };

        let program = Program {
            defs: vec![Def {
                name: "f".to_string(),
                ty: lam.ty.clone(),
                body: lam,
                meta: DefMeta::Function,
                arity: 1,
            }],
            uniforms: vec![],
            storage: vec![],
        };

        let builtins = HashSet::new();
        let result = defunctionalize(program, &builtins);

        // Should preserve the parameter lambda (not lift it)
        assert_eq!(result.defs.len(), 1);
        assert!(matches!(result.defs[0].body.kind, TermKind::Lam { .. }));
    }

    #[test]
    fn test_defunc_lambda_with_capture() {
        // def f y = let g = |x| x + y in g
        // The lambda |x| x + y captures y
        let mut ids = TermIdSource::new();

        // Build: |x| x + y  (where y is free)
        let inner_lam = Term {
            id: ids.next_id(),
            ty: arrow(i32_ty(), i32_ty()),
            span: dummy_span(),
            kind: TermKind::Lam {
                param: "x".to_string(),
                param_ty: i32_ty(),
                body: Box::new(Term {
                    id: ids.next_id(),
                    ty: i32_ty(),
                    span: dummy_span(),
                    kind: TermKind::App {
                        func: Box::new(FunctionName::BinOp(crate::ast::BinaryOp { op: "+".to_string() })),
                        arg: Box::new(Term {
                            id: ids.next_id(),
                            ty: i32_ty(),
                            span: dummy_span(),
                            kind: TermKind::Var("y".to_string()),
                        }),
                    },
                }),
            },
        };

        // let g = inner_lam in g
        let let_expr = Term {
            id: ids.next_id(),
            ty: arrow(i32_ty(), i32_ty()),
            span: dummy_span(),
            kind: TermKind::Let {
                name: "g".to_string(),
                name_ty: arrow(i32_ty(), i32_ty()),
                rhs: Box::new(inner_lam),
                body: Box::new(Term {
                    id: ids.next_id(),
                    ty: arrow(i32_ty(), i32_ty()),
                    span: dummy_span(),
                    kind: TermKind::Var("g".to_string()),
                }),
            },
        };

        // |y| let_expr
        let outer_lam = Term {
            id: ids.next_id(),
            ty: arrow(i32_ty(), arrow(i32_ty(), i32_ty())),
            span: dummy_span(),
            kind: TermKind::Lam {
                param: "y".to_string(),
                param_ty: i32_ty(),
                body: Box::new(let_expr),
            },
        };

        let program = Program {
            defs: vec![Def {
                name: "f".to_string(),
                ty: outer_lam.ty.clone(),
                body: outer_lam,
                meta: DefMeta::Function,
                arity: 1,
            }],
            uniforms: vec![],
            storage: vec![],
        };

        let builtins = HashSet::new();
        let result = defunctionalize(program, &builtins);

        // Should have lifted the inner lambda
        assert!(result.defs.len() >= 2, "Expected lifted lambda def");

        // Find the lifted lambda
        let lifted = result.defs.iter().find(|d| d.name.starts_with("_lambda_"));
        assert!(lifted.is_some(), "Should have a _lambda_ definition");
    }
}
