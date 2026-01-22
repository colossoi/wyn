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

use super::{Def, DefMeta, LoopKind, Program, Term, TermIdSource, TermKind};
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
    /// - Captured terms (already typed, ready to append at call sites)
    Lambda {
        lifted_name: String,
        captures: Vec<Term>,
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
    /// Original definition for cloning during specialization (None for intrinsics)
    def: Option<Def>,
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
        let func_param_indices: Vec<usize> =
            param_types.iter().enumerate().filter(|(_, ty)| is_arrow_type(ty)).map(|(i, _)| i).collect();

        if !func_param_indices.is_empty() {
            hof_info.insert(
                def.name.clone(),
                HofInfo {
                    func_param_indices,
                    def: Some(def.clone()),
                },
            );
        }
    }

    hof_info
}

/// Get HOF info for intrinsic SOAC functions.
/// These are higher-order but don't have definitions to specialize.
fn intrinsic_hof_info() -> HashMap<String, HofInfo> {
    let mut info = HashMap::new();

    // map : (a -> b) -> Array -> Array  -- func at index 0
    info.insert(
        "_w_intrinsic_map".to_string(),
        HofInfo {
            func_param_indices: vec![0],
            def: None,
        },
    );

    // inplace_map : (a -> a) -> Array -> Array  -- func at index 0
    info.insert(
        "_w_intrinsic_inplace_map".to_string(),
        HofInfo {
            func_param_indices: vec![0],
            def: None,
        },
    );

    // reduce : (a -> a -> a) -> a -> Array -> a  -- func at index 0
    info.insert(
        "_w_intrinsic_reduce".to_string(),
        HofInfo {
            func_param_indices: vec![0],
            def: None,
        },
    );

    // filter : (a -> bool) -> Array -> Array  -- func at index 0
    info.insert(
        "_w_intrinsic_filter".to_string(),
        HofInfo {
            func_param_indices: vec![0],
            def: None,
        },
    );

    // scan : (a -> a -> a) -> a -> Array -> Array  -- func at index 0
    info.insert(
        "_w_intrinsic_scan".to_string(),
        HofInfo {
            func_param_indices: vec![0],
            def: None,
        },
    );

    // hist_1d : Array -> (a -> a -> a) -> a -> Array -> Array -> Array  -- func at index 1
    info.insert(
        "_w_intrinsic_hist_1d".to_string(),
        HofInfo {
            func_param_indices: vec![1],
            def: None,
        },
    );

    info
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
            // Map type variable to concrete type, checking for consistency
            if let Some(existing) = subst.get(id) {
                assert_eq!(
                    existing, concrete,
                    "BUG: Inconsistent type substitution for variable {}: {:?} vs {:?}",
                    id, existing, concrete
                );
            } else {
                subst.insert(*id, concrete.clone());
            }
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

/// Format a type as a string key for cache lookups.
/// Uses a compact representation that uniquely identifies the type.
fn format_type_for_key(ty: &Type<TypeName>) -> String {
    match ty {
        Type::Variable(v) => format!("${}", v),
        Type::Constructed(name, args) => {
            if args.is_empty() {
                format!("{:?}", name)
            } else {
                let args_str: Vec<String> = args.iter().map(format_type_for_key).collect();
                format!("{:?}<{}>", name, args_str.join(","))
            }
        }
    }
}

/// Apply a type substitution to all types in a Term (recursively).
fn apply_type_subst_to_term(term: &Term, subst: &TypeSubst, term_ids: &mut TermIdSource) -> Term {
    let new_ty = apply_type_subst(&term.ty, subst);
    let new_kind = match &term.kind {
        TermKind::Var(name) => TermKind::Var(name.clone()),
        TermKind::BinOp(op) => TermKind::BinOp(op.clone()),
        TermKind::UnOp(op) => TermKind::UnOp(op.clone()),
        TermKind::IntLit(s) => TermKind::IntLit(s.clone()),
        TermKind::FloatLit(f) => TermKind::FloatLit(*f),
        TermKind::BoolLit(b) => TermKind::BoolLit(*b),
        TermKind::StringLit(s) => TermKind::StringLit(s.clone()),
        TermKind::App { func, arg } => TermKind::App {
            func: Box::new(apply_type_subst_to_term(func, subst, term_ids)),
            arg: Box::new(apply_type_subst_to_term(arg, subst, term_ids)),
        },
        TermKind::Lam {
            param,
            param_ty,
            body,
        } => TermKind::Lam {
            param: param.clone(),
            param_ty: apply_type_subst(param_ty, subst),
            body: Box::new(apply_type_subst_to_term(body, subst, term_ids)),
        },
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => TermKind::Let {
            name: name.clone(),
            name_ty: apply_type_subst(name_ty, subst),
            rhs: Box::new(apply_type_subst_to_term(rhs, subst, term_ids)),
            body: Box::new(apply_type_subst_to_term(body, subst, term_ids)),
        },
        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => TermKind::If {
            cond: Box::new(apply_type_subst_to_term(cond, subst, term_ids)),
            then_branch: Box::new(apply_type_subst_to_term(then_branch, subst, term_ids)),
            else_branch: Box::new(apply_type_subst_to_term(else_branch, subst, term_ids)),
        },
        TermKind::Loop {
            loop_var,
            loop_var_ty,
            init,
            init_bindings,
            kind,
            body,
        } => {
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
/// Returns the actual Term for each free variable (preserving its type and span).
fn compute_free_vars(
    term: &Term,
    bound: &HashSet<String>,
    top_level: &HashSet<String>,
    builtins: &HashSet<String>,
) -> Vec<Term> {
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
    free: &mut Vec<Term>,
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
                free.push(term.clone());
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
            collect_free_vars(func, bound, top_level, builtins, free, seen);
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
        TermKind::Loop {
            loop_var,
            init,
            init_bindings,
            kind,
            body,
            ..
        } => {
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
                LoopKind::ForRange {
                    var,
                    bound: bound_expr,
                    ..
                } => {
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
        TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::StringLit(_)
        | TermKind::BinOp(_)
        | TermKind::UnOp(_) => {}
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
    /// Cache: (hof_name, lambda_name, arg_types) -> specialized_name
    /// Includes arg types to handle same HOF+lambda at different type instantiations
    specialization_cache: HashMap<(String, String, Vec<String>), String>,
    /// Counter for generating unique specialization names
    specialization_counter: usize,
    /// Captures for lifted lambdas: lambda_name -> capture terms
    /// Used during HOF specialization to know what captures to append
    lifted_lambda_captures: HashMap<String, Vec<Term>>,
}

impl<'a> Defunctionalizer<'a> {
    /// Defunctionalize a program.
    pub fn defunctionalize(program: Program, builtins: &'a HashSet<String>) -> Program {
        // Detect HOFs before defunctionalization (user-defined + intrinsics)
        let mut hof_info = detect_hofs(&program.defs);
        hof_info.extend(intrinsic_hof_info());

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
                } else if let Some(captures) = self.lifted_lambda_captures.get(name) {
                    // Lifted lambda - we know its captures
                    StaticVal::Lambda {
                        lifted_name: name.clone(),
                        captures: captures.clone(),
                    }
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

            // Literals and operators are dynamic
            TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::StringLit(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_) => DefuncResult {
                term,
                sv: StaticVal::Dynamic,
            },
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

        // Record captures for this lifted lambda (used during HOF specialization)
        self.lifted_lambda_captures.insert(lifted_name.clone(), captures.clone());

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
    fn defunc_app(&mut self, func: Term, arg: Term, ty: Type<TypeName>, span: Span) -> DefuncResult {
        // Collect the application spine: f a1 a2 ... an
        let (base_func, args) = self.collect_spine(func, arg);

        // Defunctionalize all arguments
        let arg_results: Vec<DefuncResult> = args.into_iter().map(|a| self.defunc_term(a)).collect();
        let arg_terms: Vec<Term> = arg_results.iter().map(|ar| ar.term.clone()).collect();

        // Handle based on the function term kind
        match &base_func.kind {
            TermKind::BinOp(_) | TermKind::UnOp(_) => {
                // Operators are preserved as-is - just rebuild the application
                let result_term = self.build_curried_app_with_term(base_func, arg_terms, ty.clone(), span);
                DefuncResult {
                    term: result_term,
                    sv: StaticVal::Dynamic,
                }
            }

            TermKind::Var(ref name) => {
                // Get static value for the callee
                let callee_sv = self.env.get(name).cloned().unwrap_or_else(|| {
                    // Top-level function reference - treat as Lambda with no captures
                    if self.top_level.contains(name) && is_arrow_type(&base_func.ty) {
                        StaticVal::Lambda {
                            lifted_name: name.clone(),
                            captures: vec![],
                        }
                    } else {
                        StaticVal::Dynamic
                    }
                });

                // Check if this is a call to a HOF (user-defined or intrinsic) with a lambda argument
                // We specialize for ALL lambda arguments (even non-capturing ones) because
                // SPIRV doesn't support passing functions as values - they must be called directly
                if let Some(hof_info) = self.hof_info.get(name).cloned() {
                    for &func_param_idx in &hof_info.func_param_indices {
                        if func_param_idx < arg_results.len() {
                            if let StaticVal::Lambda { .. } = &arg_results[func_param_idx].sv {
                                return self.handle_hof_call(
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

    /// Collect application spine: f a1 a2 ... -> (f, [a1, a2, ...])
    fn collect_spine(&self, func: Term, arg: Term) -> (Term, Vec<Term>) {
        let mut args = vec![arg];
        let mut current = func;

        // Walk up the spine
        loop {
            match current.kind {
                TermKind::App {
                    func: inner_func,
                    arg: inner_arg,
                } => {
                    args.push(*inner_arg);
                    current = *inner_func;
                }
                _ => {
                    // Hit a non-App term - this is the base
                    break;
                }
            }
        }

        // Args were collected in reverse order
        args.reverse();
        (current, args)
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
            ty: Type::Constructed(TypeName::Ignored, vec![]),
            span,
            kind: TermKind::App {
                func: Box::new(func_term),
                arg: Box::new(args[0].clone()),
            },
        };

        for arg in args.into_iter().skip(1) {
            current = Term {
                id: self.term_ids.next_id(),
                ty: Type::Constructed(TypeName::Ignored, vec![]),
                span,
                kind: TermKind::App {
                    func: Box::new(current),
                    arg: Box::new(arg),
                },
            };
        }

        // Fix the final type
        current.ty = result_ty;
        current
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
                // Captures are already Terms - just clone them
                all_args.extend(captures.iter().cloned());
                self.build_curried_app(lifted_name, all_args, result_ty, span)
            }
            _ => {
                // No captures - call callee_term directly with args
                self.build_curried_app_with_term(callee_term, args, result_ty, span)
            }
        }
    }

    /// Build a curried application: f a1 a2 a3 ...
    fn build_curried_app(
        &mut self,
        func_name: &str,
        args: Vec<Term>,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let func_term = Term {
            id: self.term_ids.next_id(),
            ty: Type::Constructed(TypeName::Ignored, vec![]),
            span,
            kind: TermKind::Var(func_name.to_string()),
        };

        if args.is_empty() {
            return Term {
                ty: result_ty,
                ..func_term
            };
        }

        // Build from left to right: ((f a1) a2) a3 ...
        let mut current = Term {
            id: self.term_ids.next_id(),
            ty: Type::Constructed(TypeName::Ignored, vec![]),
            span,
            kind: TermKind::App {
                func: Box::new(func_term),
                arg: Box::new(args[0].clone()),
            },
        };

        for arg in args.into_iter().skip(1) {
            current = Term {
                id: self.term_ids.next_id(),
                ty: Type::Constructed(TypeName::Ignored, vec![]),
                span,
                kind: TermKind::App {
                    func: Box::new(current),
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
    fn append_capture_params(&mut self, lam: Term, captures: &[Term], span: Span) -> Term {
        // Find the innermost body (unwrap all lambdas)
        fn extract_inner(term: Term) -> (Vec<(String, Type<TypeName>, Type<TypeName>)>, Term) {
            match term.kind {
                TermKind::Lam {
                    param,
                    param_ty,
                    body,
                } => {
                    let (mut params, inner) = extract_inner(*body);
                    params.insert(0, (param, param_ty, term.ty));
                    (params, inner)
                }
                _ => (vec![], term),
            }
        }

        let (orig_params, inner_body) = extract_inner(lam);

        // Wrap inner body with capture lambdas (innermost first, so reverse iterate)
        let with_captures = captures.iter().rev().fold(inner_body, |acc, cap_term| {
            // Extract name from the capture term (must be a Var)
            let cap_name = match &cap_term.kind {
                TermKind::Var(name) => name.clone(),
                _ => panic!("BUG: capture term is not a Var: {:?}", cap_term.kind),
            };
            let cap_ty = &cap_term.ty;
            let result_ty = Type::Constructed(TypeName::Arrow, vec![cap_ty.clone(), acc.ty.clone()]);
            Term {
                id: self.term_ids.next_id(),
                ty: result_ty,
                span,
                kind: TermKind::Lam {
                    param: cap_name,
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
    // HOF Handling (User-defined and Intrinsic)
    // =========================================================================

    /// Handle a HOF call with a lambda argument.
    /// Dispatches between user-defined HOFs (which get specialized) and intrinsic HOFs
    /// (which just get captures appended).
    fn handle_hof_call(
        &mut self,
        hof_name: &str,
        hof_info: &HofInfo,
        func_param_idx: usize,
        arg_results: &[DefuncResult],
        ty: Type<TypeName>,
        span: Span,
    ) -> DefuncResult {
        // Extract lambda info from function argument
        let (lambda_name, captures) = match &arg_results[func_param_idx].sv {
            StaticVal::Lambda {
                lifted_name,
                captures,
            } => (lifted_name.clone(), captures.clone()),
            _ => unreachable!("handle_hof_call called without Lambda StaticVal"),
        };

        match &hof_info.def {
            Some(def) => {
                // User-defined HOF: specialize by cloning and substituting
                self.specialize_user_hof(
                    hof_name,
                    def,
                    func_param_idx,
                    &lambda_name,
                    &captures,
                    arg_results,
                    ty,
                    span,
                )
            }
            None => {
                // Intrinsic HOF: just append captures to the call
                self.build_intrinsic_hof_call(
                    hof_name,
                    func_param_idx,
                    &lambda_name,
                    &captures,
                    arg_results,
                    ty,
                    span,
                )
            }
        }
    }

    /// Handle intrinsic HOF call by appending captures.
    fn build_intrinsic_hof_call(
        &mut self,
        hof_name: &str,
        func_param_idx: usize,
        lambda_name: &str,
        captures: &[Term],
        arg_results: &[DefuncResult],
        ty: Type<TypeName>,
        span: Span,
    ) -> DefuncResult {
        // Build args: replace lambda arg with lifted name, keep others, append captures
        let mut call_args = Vec::new();

        for (i, ar) in arg_results.iter().enumerate() {
            if i == func_param_idx {
                // Replace lambda with reference to lifted function
                call_args.push(Term {
                    id: self.term_ids.next_id(),
                    ty: ar.term.ty.clone(),
                    span,
                    kind: TermKind::Var(lambda_name.to_string()),
                });
            } else {
                call_args.push(ar.term.clone());
            }
        }

        // Append captures at the end (they're already Terms)
        call_args.extend(captures.iter().cloned());

        let result_term = self.build_curried_app(hof_name, call_args, ty, span);
        DefuncResult {
            term: result_term,
            sv: StaticVal::Dynamic,
        }
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
        hof_name: &str,
        hof_def: &Def,
        func_param_idx: usize,
        lambda_name: &str,
        captures: &[Term],
        arg_results: &[DefuncResult],
        ty: Type<TypeName>,
        span: Span,
    ) -> DefuncResult {
        // Check cache (include arg types to handle different type instantiations)
        let arg_type_keys: Vec<String> =
            arg_results.iter().map(|ar| format_type_for_key(&ar.term.ty)).collect();
        let cache_key = (hof_name.to_string(), lambda_name.to_string(), arg_type_keys);
        if let Some(specialized_name) = self.specialization_cache.get(&cache_key).cloned() {
            return self.build_specialized_call(
                &specialized_name,
                func_param_idx,
                arg_results,
                captures,
                ty,
                span,
            );
        }

        // Build type substitution from polymorphic params to concrete types
        let mut type_subst = TypeSubst::new();
        let poly_param_types = extract_param_types(&hof_def.ty);
        for (i, poly_ty) in poly_param_types.iter().enumerate() {
            if i < arg_results.len() {
                build_type_subst(poly_ty, &arg_results[i].term.ty, &mut type_subst);
            }
        }

        // Generate specialized variant
        let specialized_name = format!("{}${}", hof_name, self.specialization_counter);
        self.specialization_counter += 1;

        // Cache early to prevent infinite recursion on mutually recursive HOFs
        self.specialization_cache.insert(cache_key, specialized_name.clone());

        // Get the function parameter name from the HOF definition
        let func_param_name = self.get_func_param_name(hof_def, func_param_idx);

        // Extract params FIRST, then substitute in the inner body
        // (Substitution must happen AFTER unwrapping lambdas because the func param
        // is bound by one of those lambdas, and substitute_var correctly avoids
        // substituting in shadowed scopes)
        let (params, inner_body) = self.extract_lambda_params(hof_def.body.clone());

        // Simple substitution in inner body: replace func_param_name with lambda_name
        let substituted_inner = self.substitute_var(&inner_body, &func_param_name, lambda_name);

        // Build new param list: remove function param at func_param_idx, add captures at end
        let mut new_params: Vec<(String, Type<TypeName>)> =
            params.into_iter().enumerate().filter(|(i, _)| *i != func_param_idx).map(|(_, p)| p).collect();

        // Add captures as trailing parameters (extract name from each capture Term)
        for cap_term in captures {
            let cap_name = match &cap_term.kind {
                TermKind::Var(name) => name.clone(),
                _ => panic!("BUG: capture term is not a Var: {:?}", cap_term.kind),
            };
            new_params.push((cap_name, cap_term.ty.clone()));
        }

        // Set up environment for defunc with new params as Dynamic
        let old_env = std::mem::take(&mut self.env);
        for (param_name, _) in &new_params {
            self.env.insert(param_name.clone(), StaticVal::Dynamic);
        }

        // Defunc the substituted inner body - this handles:
        // - Capture appending for calls to lambda_name (via apply_callable + lifted_lambda_captures)
        // - Nested HOF calls with lambda args (triggers recursive specialization)
        // - New lambdas (get lifted)
        let defunced_body = self.defunc_term(substituted_inner).term;

        // Restore env
        self.env = old_env;

        // Apply type substitution to the defunced body AND to the param types
        let defunced_body = apply_type_subst_to_term(&defunced_body, &type_subst, &mut self.term_ids);
        let new_params: Vec<(String, Type<TypeName>)> =
            new_params.into_iter().map(|(name, ty)| (name, apply_type_subst(&ty, &type_subst))).collect();

        // Rebuild nested lambdas with substituted params
        let rebuilt = self.rebuild_nested_lam(&new_params, defunced_body, hof_def.body.span);

        let specialized_def = Def {
            name: specialized_name.clone(),
            ty: rebuilt.ty.clone(),
            body: rebuilt,
            meta: DefMeta::Function,
            arity: new_params.len(),
        };

        // Register
        self.lifted_defs.push(specialized_def);

        // Build call to specialized function
        self.build_specialized_call(&specialized_name, func_param_idx, arg_results, captures, ty, span)
    }

    /// Simple variable substitution: replace all occurrences of old_name with new_name.
    fn substitute_var(&mut self, term: &Term, old_name: &str, new_name: &str) -> Term {
        match &term.kind {
            TermKind::Var(name) if name == old_name => Term {
                id: self.term_ids.next_id(),
                ty: term.ty.clone(),
                span: term.span,
                kind: TermKind::Var(new_name.to_string()),
            },

            TermKind::Var(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::StringLit(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_) => term.clone(),

            TermKind::App { func, arg } => Term {
                id: self.term_ids.next_id(),
                ty: term.ty.clone(),
                span: term.span,
                kind: TermKind::App {
                    func: Box::new(self.substitute_var(func, old_name, new_name)),
                    arg: Box::new(self.substitute_var(arg, old_name, new_name)),
                },
            },

            TermKind::Lam {
                param,
                param_ty,
                body,
            } => {
                // Don't substitute if the param shadows old_name
                if param == old_name {
                    term.clone()
                } else {
                    Term {
                        id: self.term_ids.next_id(),
                        ty: term.ty.clone(),
                        span: term.span,
                        kind: TermKind::Lam {
                            param: param.clone(),
                            param_ty: param_ty.clone(),
                            body: Box::new(self.substitute_var(body, old_name, new_name)),
                        },
                    }
                }
            }

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                let new_rhs = self.substitute_var(rhs, old_name, new_name);
                // Don't substitute in body if name shadows old_name
                let new_body = if name == old_name {
                    (**body).clone()
                } else {
                    self.substitute_var(body, old_name, new_name)
                };
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

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => Term {
                id: self.term_ids.next_id(),
                ty: term.ty.clone(),
                span: term.span,
                kind: TermKind::If {
                    cond: Box::new(self.substitute_var(cond, old_name, new_name)),
                    then_branch: Box::new(self.substitute_var(then_branch, old_name, new_name)),
                    else_branch: Box::new(self.substitute_var(else_branch, old_name, new_name)),
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
                // Check shadowing FIRST - loop_var and init_bindings are in scope for
                // init_bindings exprs, While cond, and body
                let shadows = loop_var == old_name || init_bindings.iter().any(|(n, _, _)| n == old_name);

                // init is evaluated outside the loop scope
                let new_init = self.substitute_var(init, old_name, new_name);

                // init_bindings exprs reference loop_var (inside loop scope)
                let new_init_bindings: Vec<_> = init_bindings
                    .iter()
                    .map(|(n, ty, e)| {
                        let new_e =
                            if shadows { e.clone() } else { self.substitute_var(e, old_name, new_name) };
                        (n.clone(), ty.clone(), new_e)
                    })
                    .collect();

                // iter/bound are outside loop scope, cond is inside
                let new_kind = match kind {
                    LoopKind::For { var, var_ty, iter } => LoopKind::For {
                        var: var.clone(),
                        var_ty: var_ty.clone(),
                        iter: Box::new(self.substitute_var(iter, old_name, new_name)),
                    },
                    LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                        var: var.clone(),
                        var_ty: var_ty.clone(),
                        bound: Box::new(self.substitute_var(bound, old_name, new_name)),
                    },
                    LoopKind::While { cond } => {
                        let new_cond = if shadows {
                            (**cond).clone()
                        } else {
                            self.substitute_var(cond, old_name, new_name)
                        };
                        LoopKind::While {
                            cond: Box::new(new_cond),
                        }
                    }
                };

                let new_body =
                    if shadows { (**body).clone() } else { self.substitute_var(body, old_name, new_name) };

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
        }
    }

    /// Get the parameter name at a given index from a function definition.
    fn get_func_param_name(&self, def: &Def, param_idx: usize) -> String {
        let mut body = &def.body;
        let mut idx = 0;
        while let TermKind::Lam {
            param, body: inner, ..
        } = &body.kind
        {
            if idx == param_idx {
                return param.clone();
            }
            idx += 1;
            body = inner;
        }
        format!("_param_{}", param_idx)
    }

    /// Build a call to a specialized HOF.
    fn build_specialized_call(
        &mut self,
        specialized_name: &str,
        func_param_idx: usize,
        arg_results: &[DefuncResult],
        captures: &[Term],
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

        // Add captures at the end (they're already Terms)
        call_args.extend(captures.iter().cloned());

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
