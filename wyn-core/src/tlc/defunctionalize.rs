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
//!   Output: f _lambda_0 arr y
//!
//! The lifted lambda is: _lambda_0 = |x| |y| x + y  (captures at end)

use super::{Def, DefMeta, FunctionName, LoopKind, Program, Term, TermIdSource, TermKind};
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
// HOF Detection
// =============================================================================

/// Information about a higher-order function (HOF).
#[derive(Debug, Clone)]
struct HofInfo {
    /// Which parameter indices are function-typed
    func_param_indices: Vec<usize>,
    /// Original definition for cloning during specialization
    def: Def,
}

/// Check if a type is an arrow type (function type).
fn is_arrow_type(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::Arrow, _))
}

/// Extract parameter types from a nested arrow type.
/// e.g., (A -> B -> C) -> D -> E returns [(A -> B -> C), D]
fn extract_param_types(ty: &Type<TypeName>) -> Vec<Type<TypeName>> {
    let mut params = Vec::new();
    let mut current = ty;
    while let Type::Constructed(TypeName::Arrow, args) = current {
        if args.len() == 2 {
            params.push(args[0].clone());
            current = &args[1];
        } else {
            break;
        }
    }
    params
}

/// Detect which definitions are HOFs (have function-typed parameters).
fn detect_hofs(defs: &[Def]) -> HashMap<String, HofInfo> {
    let mut hof_info = HashMap::new();

    for def in defs {
        let param_types = extract_param_types(&def.ty);
        let func_param_indices: Vec<usize> = param_types
            .iter()
            .enumerate()
            .filter(|(_, ty)| is_arrow_type(ty))
            .map(|(i, _)| i)
            .collect();

        if !func_param_indices.is_empty() {
            hof_info.insert(
                def.name.clone(),
                HofInfo {
                    func_param_indices,
                    def: def.clone(),
                },
            );
        }
    }

    hof_info
}

// =============================================================================
// Type Substitution for HOF Specialization
// =============================================================================

type TypeSubst = HashMap<polytype::Variable, Type<TypeName>>;

/// Build a type substitution by unifying a polymorphic type with a concrete type.
/// This extracts mappings from type variables to concrete types.
fn build_type_subst(poly_ty: &Type<TypeName>, concrete_ty: &Type<TypeName>, subst: &mut TypeSubst) {
    match (poly_ty, concrete_ty) {
        (Type::Variable(id), concrete) => {
            // Map type variable to concrete type
            subst.insert(*id, concrete.clone());
        }
        (Type::Constructed(poly_name, poly_args), Type::Constructed(_, concrete_args)) => {
            // Recursively unify arguments
            // Skip name check - we assume types are already unified by the type checker
            for (p, c) in poly_args.iter().zip(concrete_args.iter()) {
                build_type_subst(p, c, subst);
            }
            // Also handle the case where poly_name itself contains a variable (for Size types)
            if let TypeName::Unsized = poly_name {
                // Unsized matches any size - extract size from concrete
                if let Type::Constructed(TypeName::Size(_n), _) = concrete_ty {
                    // Can't directly substitute Unsized, but the size is in the array type
                }
            }
        }
        _ => {}
    }
}

/// Apply a type substitution to a type.
fn apply_type_subst(ty: &Type<TypeName>, subst: &TypeSubst) -> Type<TypeName> {
    match ty {
        Type::Variable(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
        Type::Constructed(name, args) => {
            let new_args: Vec<_> = args.iter().map(|a| apply_type_subst(a, subst)).collect();
            Type::Constructed(name.clone(), new_args)
        }
    }
}

/// Apply a type substitution to a FunctionName.
fn apply_type_subst_to_funcname(func_name: &FunctionName, subst: &TypeSubst, term_ids: &mut TermIdSource) -> FunctionName {
    match func_name {
        FunctionName::Var(name) => FunctionName::Var(name.clone()),
        FunctionName::BinOp(op) => FunctionName::BinOp(op.clone()),
        FunctionName::UnOp(op) => FunctionName::UnOp(op.clone()),
        FunctionName::Term(term) => FunctionName::Term(Box::new(apply_type_subst_to_term(term, subst, term_ids))),
    }
}

/// Apply a type substitution to all types in a Term (recursively).
fn apply_type_subst_to_term(term: &Term, subst: &TypeSubst, term_ids: &mut TermIdSource) -> Term {
    let new_ty = apply_type_subst(&term.ty, subst);
    let new_kind = match &term.kind {
        TermKind::Var(name) => TermKind::Var(name.clone()),
        TermKind::IntLit(s) => TermKind::IntLit(s.clone()),
        TermKind::FloatLit(f) => TermKind::FloatLit(*f),
        TermKind::BoolLit(b) => TermKind::BoolLit(*b),
        TermKind::StringLit(s) => TermKind::StringLit(s.clone()),
        TermKind::App { func, arg } => TermKind::App {
            func: Box::new(apply_type_subst_to_funcname(func, subst, term_ids)),
            arg: Box::new(apply_type_subst_to_term(arg, subst, term_ids)),
        },
        TermKind::Lam { param, param_ty, body } => TermKind::Lam {
            param: param.clone(),
            param_ty: apply_type_subst(param_ty, subst),
            body: Box::new(apply_type_subst_to_term(body, subst, term_ids)),
        },
        TermKind::Let { name, name_ty, rhs, body } => TermKind::Let {
            name: name.clone(),
            name_ty: apply_type_subst(name_ty, subst),
            rhs: Box::new(apply_type_subst_to_term(rhs, subst, term_ids)),
            body: Box::new(apply_type_subst_to_term(body, subst, term_ids)),
        },
        TermKind::If { cond, then_branch, else_branch } => TermKind::If {
            cond: Box::new(apply_type_subst_to_term(cond, subst, term_ids)),
            then_branch: Box::new(apply_type_subst_to_term(then_branch, subst, term_ids)),
            else_branch: Box::new(apply_type_subst_to_term(else_branch, subst, term_ids)),
        },
        TermKind::Loop { loop_var, loop_var_ty, init, init_bindings, kind, body } => {
            let new_init_bindings = init_bindings
                .iter()
                .map(|(name, ty, expr)| {
                    (
                        name.clone(),
                        apply_type_subst(ty, subst),
                        apply_type_subst_to_term(expr, subst, term_ids),
                    )
                })
                .collect();
            let new_kind = match kind {
                LoopKind::For { var, var_ty, iter } => LoopKind::For {
                    var: var.clone(),
                    var_ty: apply_type_subst(var_ty, subst),
                    iter: Box::new(apply_type_subst_to_term(iter, subst, term_ids)),
                },
                LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                    var: var.clone(),
                    var_ty: apply_type_subst(var_ty, subst),
                    bound: Box::new(apply_type_subst_to_term(bound, subst, term_ids)),
                },
                LoopKind::While { cond } => LoopKind::While {
                    cond: Box::new(apply_type_subst_to_term(cond, subst, term_ids)),
                },
            };
            TermKind::Loop {
                loop_var: loop_var.clone(),
                loop_var_ty: apply_type_subst(loop_var_ty, subst),
                init: Box::new(apply_type_subst_to_term(init, subst, term_ids)),
                init_bindings: new_init_bindings,
                kind: new_kind,
                body: Box::new(apply_type_subst_to_term(body, subst, term_ids)),
            }
        }
    };
    Term {
        id: term_ids.next_id(),
        ty: new_ty,
        span: term.span,
        kind: new_kind,
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
        TermKind::Loop { loop_var, init, init_bindings, kind, body, .. } => {
            // init is evaluated outside the loop
            collect_free_vars(init, bound, top_level, builtins, free, seen);

            // Build inner bound set with loop_var and init_binding names
            let mut inner_bound = bound.clone();
            inner_bound.insert(loop_var.clone());
            for (name, _, _) in init_bindings {
                inner_bound.insert(name.clone());
            }

            // Add loop kind variable(s)
            match kind {
                LoopKind::For { var, iter, .. } => {
                    collect_free_vars(iter, bound, top_level, builtins, free, seen);
                    inner_bound.insert(var.clone());
                }
                LoopKind::ForRange { var, bound: bound_expr, .. } => {
                    collect_free_vars(bound_expr, bound, top_level, builtins, free, seen);
                    inner_bound.insert(var.clone());
                }
                LoopKind::While { cond } => {
                    // cond is evaluated inside the loop with loop_var in scope
                    collect_free_vars(cond, &inner_bound, top_level, builtins, free, seen);
                }
            }

            // init_bindings expressions reference loop_var
            for (_, _, expr) in init_bindings {
                collect_free_vars(expr, &inner_bound, top_level, builtins, free, seen);
            }

            // body has all bindings in scope
            collect_free_vars(body, &inner_bound, top_level, builtins, free, seen);
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
    /// HOF info from detection pass
    hof_info: HashMap<String, HofInfo>,
    /// Cache: (hof_name, lambda_name) -> specialized_name
    specialization_cache: HashMap<(String, String), String>,
    /// Counter for generating unique specialization names
    specialization_counter: usize,
}

impl<'a> Defunctionalizer<'a> {
    /// Defunctionalize a program.
    pub fn defunctionalize(program: Program, builtins: &'a HashSet<String>) -> Program {
        // Detect HOFs before defunctionalization
        let hof_info = detect_hofs(&program.defs);

        let mut defunc = Self {
            top_level: program.defs.iter().map(|d| d.name.clone()).collect(),
            builtins,
            lifted_defs: vec![],
            lambda_counter: 0,
            term_ids: TermIdSource::new(),
            env: HashMap::new(),
            hof_info,
            specialization_cache: HashMap::new(),
            specialization_counter: 0,
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
                let sv = if let Some(sv) = self.env.get(name) {
                    sv.clone()
                } else if self.top_level.contains(name) && is_arrow_type(&ty) {
                    // Top-level function reference - treat as a Lambda with no captures
                    // This allows HOF specialization to work for named function references
                    StaticVal::Lambda {
                        lifted_name: name.clone(),
                        captures: vec![],
                    }
                } else {
                    StaticVal::Dynamic
                };
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
                self.env.insert(loop_var.clone(), StaticVal::Dynamic);

                // Defunc init_bindings and add them to env
                let defunc_init_bindings: Vec<_> = init_bindings
                    .into_iter()
                    .map(|(name, binding_ty, expr)| {
                        let expr_result = self.defunc_term(expr);
                        self.env.insert(name.clone(), StaticVal::Dynamic);
                        (name, binding_ty, expr_result.term)
                    })
                    .collect();

                // Defunc kind (iter/bound/cond depending on variant)
                let defunc_kind = match kind {
                    LoopKind::For { var, var_ty, iter } => {
                        let iter_result = self.defunc_term(*iter);
                        self.env.insert(var.clone(), StaticVal::Dynamic);
                        LoopKind::For {
                            var,
                            var_ty,
                            iter: Box::new(iter_result.term),
                        }
                    }
                    LoopKind::ForRange { var, var_ty, bound } => {
                        let bound_result = self.defunc_term(*bound);
                        self.env.insert(var.clone(), StaticVal::Dynamic);
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
            // Has captures: append captures as additional parameters at the end
            // Build: |original_params...| |captures...| body
            let wrapped = self.append_capture_params(rebuilt_lam, &captures, span);
            let arity = params.len() + captures.len();

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

    /// Handle application: collect spine, flatten captures.
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

        // Handle based on the function type
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
                        // Function has captures - append them after the original arguments
                        let mut all_args = Vec::new();

                        // Add original arguments first
                        for ar in &arg_results {
                            all_args.push(ar.term.clone());
                        }

                        // Add capture values at the end
                        for (cap_name, cap_ty) in &captures {
                            all_args.push(Term {
                                id: self.term_ids.next_id(),
                                ty: cap_ty.clone(),
                                span,
                                kind: TermKind::Var(cap_name.clone()),
                            });
                        }

                        // Build curried application to lifted function
                        let result_term = self.build_curried_app(&lifted_name, all_args, ty.clone(), span);
                        DefuncResult {
                            term: result_term,
                            sv: StaticVal::Dynamic,
                        }
                    }

                    _ => {
                        // Check if this is a call to a user-defined HOF with a lambda argument
                        // We specialize for ALL lambda arguments (even non-capturing ones) because
                        // SPIRV doesn't support passing functions as values - they must be called directly
                        if let Some(hof_info) = self.hof_info.get(name).cloned() {
                            for &func_param_idx in &hof_info.func_param_indices {
                                if func_param_idx < arg_results.len() {
                                    if let StaticVal::Lambda { .. } = &arg_results[func_param_idx].sv {
                                        // Need to specialize this HOF (for any lambda, not just capturing ones)
                                        return self.specialize_hof_call(
                                            name,
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

                        // Check if any argument is a lambda with captures (for intrinsics/builtins)
                        // This handles SOACs like _w_intrinsic_map, _w_intrinsic_reduce, etc.
                        let mut has_lambda_with_captures = false;
                        let mut all_captures: Vec<(String, Type<TypeName>)> = Vec::new();
                        for ar in &arg_results {
                            if let StaticVal::Lambda { captures, .. } = &ar.sv {
                                if !captures.is_empty() {
                                    has_lambda_with_captures = true;
                                    // Collect captures (avoiding duplicates)
                                    for cap in captures {
                                        if !all_captures.iter().any(|(n, _)| n == &cap.0) {
                                            all_captures.push(cap.clone());
                                        }
                                    }
                                }
                            }
                        }

                        if has_lambda_with_captures {
                            // Append captures as trailing arguments
                            let mut all_args = Vec::new();
                            for ar in &arg_results {
                                all_args.push(ar.term.clone());
                            }
                            for (cap_name, cap_ty) in &all_captures {
                                all_args.push(Term {
                                    id: self.term_ids.next_id(),
                                    ty: cap_ty.clone(),
                                    span,
                                    kind: TermKind::Var(cap_name.clone()),
                                });
                            }
                            let result_term = self.build_curried_app(name, all_args, ty.clone(), span);
                            return DefuncResult {
                                term: result_term,
                                sv: StaticVal::Dynamic,
                            };
                        }

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
                        // Function has captures - append them after the original arguments
                        let mut all_args = Vec::new();

                        // Add original arguments first
                        for ar in &arg_results {
                            all_args.push(ar.term.clone());
                        }

                        // Add capture values at the end
                        for (cap_name, cap_ty) in captures {
                            all_args.push(Term {
                                id: self.term_ids.next_id(),
                                ty: cap_ty.clone(),
                                span,
                                kind: TermKind::Var(cap_name.clone()),
                            });
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

    /// Collect application spine: f a1 a2 ... -> (f, [a1, a2, ...])
    fn collect_spine(&self, func: FunctionName, arg: Term) -> (FunctionName, Vec<Term>) {
        let mut args = vec![arg];
        let mut current_func = func;

        // Walk up the spine
        while let FunctionName::Term(t) = current_func {
            match t.kind {
                TermKind::App {
                    func: inner_func,
                    arg: inner_arg,
                } => {
                    args.push(*inner_arg);
                    current_func = *inner_func;
                }
                TermKind::Var(name) => {
                    // Unwrap Term(Var(name)) to Var(name)
                    current_func = FunctionName::Var(name);
                    break;
                }
                _ => {
                    // Hit a non-App, non-Var term in function position
                    current_func = FunctionName::Term(t);
                    break;
                }
            }
        }

        // Args were collected in reverse order
        args.reverse();
        (current_func, args)
    }

    /// Build a curried application with a term as the function: t a1 a2 a3 ...
    fn build_curried_app_with_term(
        &mut self,
        func_term: Term,
        args: Vec<Term>,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        if args.is_empty() {
            return func_term;
        }

        // Build from left to right: ((t a1) a2) a3 ...
        let mut current = Term {
            id: self.term_ids.next_id(),
            ty: Type::Variable(0), // Placeholder
            span,
            kind: TermKind::App {
                func: Box::new(FunctionName::Term(Box::new(func_term))),
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

        // Fix the final type
        current.ty = result_ty;
        current
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

    /// Append capture parameters to a lambda (captures at end).
    /// Given `|x| |y| body` and captures [a, b], produces `|x| |y| |a| |b| body`.
    fn append_capture_params(
        &mut self,
        lam: Term,
        captures: &[(String, Type<TypeName>)],
        span: Span,
    ) -> Term {
        // Find the innermost body (unwrap all lambdas)
        fn extract_inner(term: Term) -> (Vec<(String, Type<TypeName>, Type<TypeName>)>, Term) {
            match term.kind {
                TermKind::Lam { param, param_ty, body } => {
                    let (mut params, inner) = extract_inner(*body);
                    params.insert(0, (param, param_ty, term.ty));
                    (params, inner)
                }
                _ => (vec![], term),
            }
        }

        let (orig_params, inner_body) = extract_inner(lam);

        // Wrap inner body with capture lambdas (innermost first, so reverse iterate)
        let with_captures = captures.iter().rev().fold(inner_body, |acc, (cap_name, cap_ty)| {
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
        });

        // Rebuild original params around it
        orig_params.into_iter().rev().fold(with_captures, |acc, (param, param_ty, _orig_ty)| {
            let result_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), acc.ty.clone()]);
            Term {
                id: self.term_ids.next_id(),
                ty: result_ty,
                span,
                kind: TermKind::Lam {
                    param,
                    param_ty,
                    body: Box::new(acc),
                },
            }
        })
    }

    // =========================================================================
    // HOF Specialization
    // =========================================================================

    /// Specialize a HOF call when a function argument is a lambda with captures.
    fn specialize_hof_call(
        &mut self,
        hof_name: &str,
        hof_info: &HofInfo,
        func_param_idx: usize,
        arg_results: &[DefuncResult],
        ty: Type<TypeName>,
        span: Span,
    ) -> DefuncResult {
        // 1. Extract lambda info from function argument
        let (lambda_name, captures) = match &arg_results[func_param_idx].sv {
            StaticVal::Lambda { lifted_name, captures } => (lifted_name.clone(), captures.clone()),
            _ => unreachable!("specialize_hof_call called without Lambda StaticVal"),
        };

        // 2. Check cache
        let cache_key = (hof_name.to_string(), lambda_name.clone());
        if let Some(specialized_name) = self.specialization_cache.get(&cache_key).cloned() {
            return self.build_specialized_call(&specialized_name, func_param_idx, arg_results, &captures, ty, span);
        }

        // 3. Build type substitution from polymorphic params to concrete types
        // This maps type variables in the HOF definition to concrete types at the call site
        let mut type_subst = TypeSubst::new();
        let poly_param_types = extract_param_types(&hof_info.def.ty);
        for (i, poly_ty) in poly_param_types.iter().enumerate() {
            if i < arg_results.len() {
                build_type_subst(poly_ty, &arg_results[i].term.ty, &mut type_subst);
            }
        }

        // 4. Generate specialized variant
        let specialized_name = format!("{}${}", hof_name, self.specialization_counter);
        self.specialization_counter += 1;

        // 5. Get the function parameter name from the HOF definition
        let func_param_name = self.get_func_param_name(&hof_info.def, func_param_idx);

        // 6. Clone HOF body and substitute f(args) -> lambda(args, captures)
        let specialized_body = self.specialize_body(
            &hof_info.def.body,
            &func_param_name,
            &lambda_name,
            &captures,
            span,
        );

        // 7. Apply type substitution to the specialized body
        let specialized_body = apply_type_subst_to_term(&specialized_body, &type_subst, &mut self.term_ids);

        // 8. Build new def with captures as trailing params
        let specialized_def = self.build_specialized_def(
            &specialized_name,
            &hof_info.def,
            specialized_body,
            func_param_idx,
            &captures,
        );

        // 9. Register and cache
        self.lifted_defs.push(specialized_def);
        self.specialization_cache.insert(cache_key, specialized_name.clone());

        // 10. Build call to specialized function
        self.build_specialized_call(&specialized_name, func_param_idx, arg_results, &captures, ty, span)
    }

    /// Get the parameter name at a given index from a function definition.
    fn get_func_param_name(&self, def: &Def, param_idx: usize) -> String {
        let mut body = &def.body;
        let mut idx = 0;
        while let TermKind::Lam { param, body: inner, .. } = &body.kind {
            if idx == param_idx {
                return param.clone();
            }
            idx += 1;
            body = inner;
        }
        format!("_param_{}", param_idx)
    }

    /// Recursively substitute calls to func_param with calls to lambda_name (with captures).
    fn specialize_body(
        &mut self,
        term: &Term,
        func_param_name: &str,
        lambda_name: &str,
        captures: &[(String, Type<TypeName>)],
        span: Span,
    ) -> Term {
        match &term.kind {
            TermKind::App { func, arg } => {
                // Collect spine and check if base is the function param
                let (base, args) = self.collect_spine_ref(func, arg);

                if let FunctionName::Var(name) = &base {
                    if name == func_param_name {
                        // Replace f(args...) with lambda(args..., captures...)
                        let new_args: Vec<Term> = args
                            .iter()
                            .map(|a| self.specialize_body(a, func_param_name, lambda_name, captures, span))
                            .collect();

                        // Add capture variables at the end
                        let mut all_args = new_args;
                        for (cap_name, cap_ty) in captures {
                            all_args.push(Term {
                                id: self.term_ids.next_id(),
                                ty: cap_ty.clone(),
                                span,
                                kind: TermKind::Var(cap_name.clone()),
                            });
                        }

                        return self.build_curried_app(lambda_name, all_args, term.ty.clone(), span);
                    }
                }

                // Not a direct call to f, but check if f is passed as an argument
                // If so, we need to append captures to THIS call
                // e.g., _w_intrinsic_map f xs -> _w_intrinsic_map _lambda_10 xs arr
                let args_contain_func_param = args.iter().any(|a| self.term_references_var(a, func_param_name));

                if args_contain_func_param {
                    // Transform args (replacing f with lambda_name)
                    let new_args: Vec<Term> = args
                        .iter()
                        .map(|a| self.specialize_body(a, func_param_name, lambda_name, captures, span))
                        .collect();

                    // Append captures to the call
                    let mut all_args = new_args;
                    for (cap_name, cap_ty) in captures {
                        all_args.push(Term {
                            id: self.term_ids.next_id(),
                            ty: cap_ty.clone(),
                            span,
                            kind: TermKind::Var(cap_name.clone()),
                        });
                    }

                    // Build the application with the transformed args
                    match &base {
                        FunctionName::Var(n) => {
                            return self.build_curried_app(n, all_args, term.ty.clone(), span);
                        }
                        FunctionName::Term(t) => {
                            // Recursively specialize the function term
                            let new_func_term = self.specialize_body(t, func_param_name, lambda_name, captures, span);
                            return self.build_curried_app_with_term(new_func_term, all_args, term.ty.clone(), span);
                        }
                        _ => {
                            // BinOp/UnOp - shouldn't have function params as args, but handle anyway
                            // Fall through to normal handling
                        }
                    }
                }

                // No function param reference, recurse normally
                let new_func = match func.as_ref() {
                    FunctionName::Var(_) | FunctionName::BinOp(_) | FunctionName::UnOp(_) => func.as_ref().clone(),
                    FunctionName::Term(t) => {
                        FunctionName::Term(Box::new(self.specialize_body(t, func_param_name, lambda_name, captures, span)))
                    }
                };
                let new_arg = self.specialize_body(arg, func_param_name, lambda_name, captures, span);
                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty.clone(),
                    span: term.span,
                    kind: TermKind::App {
                        func: Box::new(new_func),
                        arg: Box::new(new_arg),
                    },
                }
            }

            TermKind::Var(name) => {
                if name == func_param_name {
                    // Bare reference to function param -> reference to lambda
                    Term {
                        id: self.term_ids.next_id(),
                        ty: term.ty.clone(),
                        span: term.span,
                        kind: TermKind::Var(lambda_name.to_string()),
                    }
                } else {
                    term.clone()
                }
            }

            TermKind::Let { name, name_ty, rhs, body } => {
                let new_rhs = self.specialize_body(rhs, func_param_name, lambda_name, captures, span);
                let new_body = self.specialize_body(body, func_param_name, lambda_name, captures, span);
                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty.clone(),
                    span: term.span,
                    kind: TermKind::Let {
                        name: name.clone(),
                        name_ty: name_ty.clone(),
                        rhs: Box::new(new_rhs),
                        body: Box::new(new_body),
                    },
                }
            }

            TermKind::Lam { param, param_ty, body } => {
                let new_body = self.specialize_body(body, func_param_name, lambda_name, captures, span);
                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty.clone(),
                    span: term.span,
                    kind: TermKind::Lam {
                        param: param.clone(),
                        param_ty: param_ty.clone(),
                        body: Box::new(new_body),
                    },
                }
            }

            TermKind::If { cond, then_branch, else_branch } => {
                let new_cond = self.specialize_body(cond, func_param_name, lambda_name, captures, span);
                let new_then = self.specialize_body(then_branch, func_param_name, lambda_name, captures, span);
                let new_else = self.specialize_body(else_branch, func_param_name, lambda_name, captures, span);
                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty.clone(),
                    span: term.span,
                    kind: TermKind::If {
                        cond: Box::new(new_cond),
                        then_branch: Box::new(new_then),
                        else_branch: Box::new(new_else),
                    },
                }
            }

            TermKind::Loop { loop_var, loop_var_ty, init, init_bindings, kind, body } => {
                let new_init = self.specialize_body(init, func_param_name, lambda_name, captures, span);
                let new_init_bindings: Vec<_> = init_bindings
                    .iter()
                    .map(|(name, ty, expr)| {
                        (name.clone(), ty.clone(), self.specialize_body(expr, func_param_name, lambda_name, captures, span))
                    })
                    .collect();
                let new_kind = match kind {
                    LoopKind::For { var, var_ty, iter } => LoopKind::For {
                        var: var.clone(),
                        var_ty: var_ty.clone(),
                        iter: Box::new(self.specialize_body(iter, func_param_name, lambda_name, captures, span)),
                    },
                    LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                        var: var.clone(),
                        var_ty: var_ty.clone(),
                        bound: Box::new(self.specialize_body(bound, func_param_name, lambda_name, captures, span)),
                    },
                    LoopKind::While { cond } => LoopKind::While {
                        cond: Box::new(self.specialize_body(cond, func_param_name, lambda_name, captures, span)),
                    },
                };
                let new_body = self.specialize_body(body, func_param_name, lambda_name, captures, span);
                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty.clone(),
                    span: term.span,
                    kind: TermKind::Loop {
                        loop_var: loop_var.clone(),
                        loop_var_ty: loop_var_ty.clone(),
                        init: Box::new(new_init),
                        init_bindings: new_init_bindings,
                        kind: new_kind,
                        body: Box::new(new_body),
                    },
                }
            }

            // Literals are unchanged
            TermKind::IntLit(_) | TermKind::FloatLit(_) | TermKind::BoolLit(_) | TermKind::StringLit(_) => {
                term.clone()
            }
        }
    }

    /// Check if a term references a specific variable.
    fn term_references_var(&self, term: &Term, var_name: &str) -> bool {
        match &term.kind {
            TermKind::Var(name) => name == var_name,
            TermKind::App { func, arg } => {
                self.term_references_var(arg, var_name)
                    || match func.as_ref() {
                        FunctionName::Var(n) => n == var_name,
                        FunctionName::Term(t) => self.term_references_var(t, var_name),
                        _ => false,
                    }
            }
            TermKind::Let { rhs, body, .. } => {
                self.term_references_var(rhs, var_name) || self.term_references_var(body, var_name)
            }
            TermKind::Lam { body, .. } => self.term_references_var(body, var_name),
            TermKind::If { cond, then_branch, else_branch } => {
                self.term_references_var(cond, var_name)
                    || self.term_references_var(then_branch, var_name)
                    || self.term_references_var(else_branch, var_name)
            }
            TermKind::Loop { init, init_bindings, kind, body, .. } => {
                self.term_references_var(init, var_name)
                    || init_bindings.iter().any(|(_, _, expr)| self.term_references_var(expr, var_name))
                    || match kind {
                        LoopKind::For { iter, .. } => self.term_references_var(iter, var_name),
                        LoopKind::ForRange { bound, .. } => self.term_references_var(bound, var_name),
                        LoopKind::While { cond } => self.term_references_var(cond, var_name),
                    }
                    || self.term_references_var(body, var_name)
            }
            TermKind::IntLit(_) | TermKind::FloatLit(_) | TermKind::BoolLit(_) | TermKind::StringLit(_) => false,
        }
    }

    /// Collect application spine from references (used during specialization).
    fn collect_spine_ref<'b>(&self, func: &'b FunctionName, arg: &'b Term) -> (FunctionName, Vec<&'b Term>) {
        let mut args = vec![arg];
        let mut current_func = func;

        // Walk up the spine
        while let FunctionName::Term(t) = current_func {
            if let TermKind::App { func: inner_func, arg: inner_arg } = &t.kind {
                args.push(inner_arg.as_ref());
                current_func = inner_func.as_ref();
            } else {
                break;
            }
        }

        // Args were collected in reverse order
        args.reverse();
        (current_func.clone(), args)
    }

    /// Build a specialized definition with captures as trailing parameters.
    fn build_specialized_def(
        &mut self,
        specialized_name: &str,
        original_def: &Def,
        specialized_body: Term,
        func_param_idx: usize,
        captures: &[(String, Type<TypeName>)],
    ) -> Def {
        // Extract original params
        let (params, inner_body) = self.extract_lambda_params(specialized_body);

        // Build new param list: remove function param at func_param_idx, add captures at end
        let mut new_params: Vec<(String, Type<TypeName>)> = params
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i != func_param_idx)
            .map(|(_, p)| p)
            .collect();

        // Add captures as additional parameters
        for (cap_name, cap_ty) in captures {
            new_params.push((cap_name.clone(), cap_ty.clone()));
        }

        // Rebuild the nested lambdas
        let rebuilt = self.rebuild_nested_lam(&new_params, inner_body, original_def.body.span);

        Def {
            name: specialized_name.to_string(),
            ty: rebuilt.ty.clone(),
            body: rebuilt,
            meta: DefMeta::Function,
            arity: new_params.len(),
        }
    }

    /// Build a call to a specialized HOF.
    fn build_specialized_call(
        &mut self,
        specialized_name: &str,
        func_param_idx: usize,
        arg_results: &[DefuncResult],
        captures: &[(String, Type<TypeName>)],
        ty: Type<TypeName>,
        span: Span,
    ) -> DefuncResult {
        let mut call_args = Vec::new();

        // Add non-function arguments
        for (i, ar) in arg_results.iter().enumerate() {
            if i != func_param_idx {
                call_args.push(ar.term.clone());
            }
        }

        // Add captures at the end
        for (cap_name, cap_ty) in captures {
            call_args.push(Term {
                id: self.term_ids.next_id(),
                ty: cap_ty.clone(),
                span,
                kind: TermKind::Var(cap_name.clone()),
            });
        }

        let call_term = self.build_curried_app(specialized_name, call_args, ty, span);
        DefuncResult {
            term: call_term,
            sv: StaticVal::Dynamic,
        }
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
/// This combines lambda lifting with static value tracking:
/// - All lambdas are lifted to top-level definitions
/// - Captures become extra parameters (appended at end)
/// - All call sites have captures flattened as trailing arguments
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

    /// Helper to print TLC term structure for debugging
    fn print_term(term: &Term, indent: usize) -> String {
        let pad = "  ".repeat(indent);
        match &term.kind {
            TermKind::Var(name) => format!("{}Var({})", pad, name),
            TermKind::IntLit(n) => format!("{}Int({})", pad, n),
            TermKind::FloatLit(f) => format!("{}Float({})", pad, f),
            TermKind::BoolLit(b) => format!("{}Bool({})", pad, b),
            TermKind::StringLit(s) => format!("{}String({})", pad, s),
            TermKind::Lam { param, body, .. } => {
                format!("{}Lam({})\n{}", pad, param, print_term(body, indent + 1))
            }
            TermKind::App { func, arg } => {
                let func_str = match func.as_ref() {
                    FunctionName::Var(n) => format!("Var({})", n),
                    FunctionName::BinOp(op) => format!("BinOp({})", op.op),
                    FunctionName::UnOp(op) => format!("UnOp({})", op.op),
                    FunctionName::Term(t) => format!("Term:\n{}", print_term(t, indent + 2)),
                };
                format!("{}App\n{}  func: {}\n{}  arg:\n{}",
                    pad, pad, func_str, pad, print_term(arg, indent + 2))
            }
            TermKind::Let { name, rhs, body, .. } => {
                format!("{}Let {} =\n{}\n{}in\n{}",
                    pad, name, print_term(rhs, indent + 1), pad, print_term(body, indent + 1))
            }
            TermKind::If { cond, then_branch, else_branch } => {
                format!("{}If\n{}\n{}then\n{}\n{}else\n{}",
                    pad, print_term(cond, indent + 1),
                    pad, print_term(then_branch, indent + 1),
                    pad, print_term(else_branch, indent + 1))
            }
            TermKind::Loop { loop_var, init, kind, body, .. } => {
                let kind_str = match kind {
                    LoopKind::For { var, .. } => format!("for {} in ...", var),
                    LoopKind::ForRange { var, .. } => format!("for {} < ...", var),
                    LoopKind::While { .. } => "while ...".to_string(),
                };
                format!("{}Loop {} = {} ({})\n{}body:\n{}",
                    pad, loop_var, print_term(init, 0).trim(),
                    kind_str, pad, print_term(body, indent + 1))
            }
        }
    }

    /// Helper to print all defs in a program
    fn print_program(program: &Program) -> String {
        let mut out = String::new();
        for def in &program.defs {
            out.push_str(&format!("\n=== {} (arity {}) ===\n", def.name, def.arity));
            out.push_str(&print_term(&def.body, 0));
            out.push('\n');
        }
        out
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
