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

use super::{ArrayExpr, Def, DefMeta, Lambda, LoopKind, Place, Program, SoacOp, Term, TermIdSource, TermKind};
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
fn detect_hofs(defs: &[Def]) -> HashMap<SymbolId, HofInfo> {
    let mut hof_info = HashMap::new();

    for def in defs {
        let param_types = extract_param_types(&def.ty);
        let func_param_indices: Vec<usize> =
            param_types.iter().enumerate().filter(|(_, ty)| is_arrow_type(ty)).map(|(i, _)| i).collect();

        if !func_param_indices.is_empty() {
            hof_info.insert(
                def.name,
                HofInfo {
                    func_param_indices,
                    def: Some(def.clone()),
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
        (Type::Constructed(_, poly_args), Type::Constructed(_, concrete_args)) => {
            // Recursively unify arguments
            // Skip name check - we assume types are already unified by the type checker
            for (p, c) in poly_args.iter().zip(concrete_args.iter()) {
                build_type_subst(p, c, subst);
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
        TermKind::Extern(linkage) => TermKind::Extern(linkage.clone()),
        TermKind::App { func, arg } => TermKind::App {
            func: Box::new(apply_type_subst_to_term(func, subst, term_ids)),
            arg: Box::new(apply_type_subst_to_term(arg, subst, term_ids)),
        },
        TermKind::Lambda(Lambda { params, body, ret_ty, captures }) => TermKind::Lambda(Lambda {
            params: params.iter().map(|(p, ty)| (p.clone(), apply_type_subst(ty, subst))).collect(),
            body: Box::new(apply_type_subst_to_term(body, subst, term_ids)),
            ret_ty: apply_type_subst(ret_ty, subst),
            captures: captures.iter().map(|(s, ty, t)| (s.clone(), apply_type_subst(ty, subst), apply_type_subst_to_term(t, subst, term_ids))).collect(),
        }),
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
        TermKind::Soac(ref soac) => {
            TermKind::Soac(apply_type_subst_to_soac(soac, subst, term_ids))
        }
        TermKind::ArrayExpr(ref ae) => {
            TermKind::ArrayExpr(apply_type_subst_to_array_expr(ae, subst, term_ids))
        }
        TermKind::Force(ref inner) => {
            TermKind::Force(Box::new(apply_type_subst_to_term(inner, subst, term_ids)))
        }
        TermKind::Pack { .. } | TermKind::Unpack { .. } => {
            unreachable!("Pack/Unpack nodes not yet produced at this phase")
        }
    };
    Term {
        id: term_ids.next_id(),
        ty: new_ty,
        span: term.span,
        kind: new_kind,
    }
}

fn apply_type_subst_to_lambda(lam: &Lambda, subst: &TypeSubst, term_ids: &mut TermIdSource) -> Lambda {
    Lambda {
        params: lam.params.iter().map(|(p, ty)| (*p, apply_type_subst(ty, subst))).collect(),
        body: Box::new(apply_type_subst_to_term(&lam.body, subst, term_ids)),
        ret_ty: apply_type_subst(&lam.ret_ty, subst),
        captures: lam.captures.iter().map(|(s, ty, t)| (*s, apply_type_subst(ty, subst), apply_type_subst_to_term(t, subst, term_ids))).collect(),
    }
}

fn apply_type_subst_to_soac(soac: &SoacOp, subst: &TypeSubst, term_ids: &mut TermIdSource) -> SoacOp {
    match soac {
        SoacOp::Map { lam, inputs } => SoacOp::Map {
            lam: apply_type_subst_to_lambda(lam, subst, term_ids),
            inputs: inputs.iter().map(|ae| apply_type_subst_to_array_expr(ae, subst, term_ids)).collect(),
        },
        SoacOp::Reduce { op, ne, input, props } => SoacOp::Reduce {
            op: apply_type_subst_to_lambda(op, subst, term_ids),
            ne: Box::new(apply_type_subst_to_term(ne, subst, term_ids)),
            input: apply_type_subst_to_array_expr(input, subst, term_ids),
            props: props.clone(),
        },
        SoacOp::Scan { op, ne, input } => SoacOp::Scan {
            op: apply_type_subst_to_lambda(op, subst, term_ids),
            ne: Box::new(apply_type_subst_to_term(ne, subst, term_ids)),
            input: apply_type_subst_to_array_expr(input, subst, term_ids),
        },
        SoacOp::Filter { pred, input } => SoacOp::Filter {
            pred: apply_type_subst_to_lambda(pred, subst, term_ids),
            input: apply_type_subst_to_array_expr(input, subst, term_ids),
        },
        SoacOp::Scatter { dest, indices, values } => SoacOp::Scatter {
            dest: apply_type_subst_to_place(dest, subst, term_ids),
            indices: apply_type_subst_to_array_expr(indices, subst, term_ids),
            values: apply_type_subst_to_array_expr(values, subst, term_ids),
        },
        SoacOp::ReduceByIndex { dest, op, ne, indices, values, props } => SoacOp::ReduceByIndex {
            dest: apply_type_subst_to_place(dest, subst, term_ids),
            op: apply_type_subst_to_lambda(op, subst, term_ids),
            ne: Box::new(apply_type_subst_to_term(ne, subst, term_ids)),
            indices: apply_type_subst_to_array_expr(indices, subst, term_ids),
            values: apply_type_subst_to_array_expr(values, subst, term_ids),
            props: props.clone(),
        },
    }
}

fn apply_type_subst_to_array_expr(ae: &ArrayExpr, subst: &TypeSubst, term_ids: &mut TermIdSource) -> ArrayExpr {
    match ae {
        ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(apply_type_subst_to_term(t, subst, term_ids))),
        ArrayExpr::Zip(exprs) => ArrayExpr::Zip(exprs.iter().map(|e| apply_type_subst_to_array_expr(e, subst, term_ids)).collect()),
        ArrayExpr::Soac(op) => ArrayExpr::Soac(Box::new(apply_type_subst_to_soac(op, subst, term_ids))),
        ArrayExpr::Generate { shape, index_fn, elem_ty } => ArrayExpr::Generate {
            shape: shape.clone(),
            index_fn: apply_type_subst_to_lambda(index_fn, subst, term_ids),
            elem_ty: apply_type_subst(elem_ty, subst),
        },
        ArrayExpr::Literal(terms) => ArrayExpr::Literal(terms.iter().map(|t| apply_type_subst_to_term(t, subst, term_ids)).collect()),
        ArrayExpr::Range { start, len } => ArrayExpr::Range {
            start: Box::new(apply_type_subst_to_term(start, subst, term_ids)),
            len: Box::new(apply_type_subst_to_term(len, subst, term_ids)),
        },
    }
}

fn apply_type_subst_to_place(place: &Place, subst: &TypeSubst, _term_ids: &mut TermIdSource) -> Place {
    match place {
        Place::BufferSlice { base, offset, shape, elem_ty } => Place::BufferSlice {
            base: Box::new(apply_type_subst_to_term(base, subst, _term_ids)),
            offset: Box::new(apply_type_subst_to_term(offset, subst, _term_ids)),
            shape: shape.clone(),
            elem_ty: apply_type_subst(elem_ty, subst),
        },
        Place::LocalArray { id, shape, elem_ty } => Place::LocalArray {
            id: *id,
            shape: shape.clone(),
            elem_ty: apply_type_subst(elem_ty, subst),
        },
    }
}

// =============================================================================
// Free Variable Analysis (reused from lift.rs)
// =============================================================================

/// Compute free variables of a term, given explicit sets of bound names.
/// Returns the actual Term for each free variable (preserving its type and span).
fn compute_free_vars(
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

fn collect_free_vars(
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
            // Look up name to check if it's an intrinsic
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
        TermKind::App { func, arg } => {
            collect_free_vars(func, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(arg, bound, top_level, known_defs, symbols, free, seen);
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
            // init is evaluated outside the loop
            collect_free_vars(init, bound, top_level, known_defs, symbols, free, seen);

            // Build inner bound set with loop_var and init_binding names
            let mut inner_bound = bound.clone();
            inner_bound.insert(*loop_var);
            for (name, _, _) in init_bindings {
                inner_bound.insert(*name);
            }

            // Add loop kind variable(s)
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
                    // cond is evaluated inside the loop with loop_var in scope
                    collect_free_vars(cond, &inner_bound, top_level, known_defs, symbols, free, seen);
                }
            }

            // init_bindings expressions reference loop_var
            for (_, _, expr) in init_bindings {
                collect_free_vars(expr, &inner_bound, top_level, known_defs, symbols, free, seen);
            }

            // body has all bindings in scope
            collect_free_vars(body, &inner_bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::StringLit(_)
        | TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::Extern(_) => {}
        TermKind::Soac(ref soac) => {
            collect_free_vars_soac(soac, bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::ArrayExpr(ref ae) => {
            collect_free_vars_array_expr(ae, bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::Force(ref inner) => {
            collect_free_vars(inner, bound, top_level, known_defs, symbols, free, seen);
        }
        TermKind::Pack { .. } | TermKind::Unpack { .. } => {
            unreachable!("Pack/Unpack nodes not yet produced at this phase")
        }
    }
}

fn collect_free_vars_lambda(
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
    collect_free_vars(&lam.body, &inner_bound, top_level, known_defs, symbols, free, seen);
    for (_, _, cap_term) in &lam.captures {
        collect_free_vars(cap_term, bound, top_level, known_defs, symbols, free, seen);
    }
}

fn collect_free_vars_soac(
    soac: &SoacOp,
    bound: &HashSet<SymbolId>,
    top_level: &HashSet<SymbolId>,
    known_defs: &HashSet<String>,
    symbols: &SymbolTable,
    free: &mut Vec<Term>,
    seen: &mut HashSet<SymbolId>,
) {
    match soac {
        SoacOp::Map { lam, inputs } => {
            collect_free_vars_lambda(lam, bound, top_level, known_defs, symbols, free, seen);
            for input in inputs {
                collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
            }
        }
        SoacOp::Reduce { op, ne, input, .. } => {
            collect_free_vars_lambda(op, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(ne, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
        }
        SoacOp::Scan { op, ne, input } => {
            collect_free_vars_lambda(op, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(ne, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
        }
        SoacOp::Filter { pred, input } => {
            collect_free_vars_lambda(pred, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(input, bound, top_level, known_defs, symbols, free, seen);
        }
        SoacOp::Scatter { indices, values, .. } => {
            collect_free_vars_array_expr(indices, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(values, bound, top_level, known_defs, symbols, free, seen);
        }
        SoacOp::ReduceByIndex { op, ne, indices, values, .. } => {
            collect_free_vars_lambda(op, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars(ne, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(indices, bound, top_level, known_defs, symbols, free, seen);
            collect_free_vars_array_expr(values, bound, top_level, known_defs, symbols, free, seen);
        }
    }
}

fn collect_free_vars_array_expr(
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
        ArrayExpr::Soac(op) => collect_free_vars_soac(op, bound, top_level, known_defs, symbols, free, seen),
        ArrayExpr::Generate { index_fn, .. } => {
            collect_free_vars_lambda(index_fn, bound, top_level, known_defs, symbols, free, seen);
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
    pub fn defunctionalize(program: Program, known_defs: &'a HashSet<String>) -> Program {
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
        }
    }

    /// Consume the defunctionalizer and return the lifted definitions and symbol table.
    fn finish(self) -> (Vec<Def>, SymbolTable) {
        (self.lifted_defs, self.symbols)
    }

    /// Defunctionalize but preserve outermost parameter lambdas (for entry points).
    fn defunc_preserving_params(&mut self, term: Term) -> Term {
        match term.kind {
            TermKind::Lambda(Lambda { params, body, ret_ty, captures }) => {
                // Keep this lambda, but recursively process its body
                // These are single-param lambdas from build_lambda_chain
                let (param, param_ty) = params.into_iter().next().expect("BUG: Lambda with empty params");
                // Mark param as Dynamic in env
                self.env.insert(param, StaticVal::Dynamic);
                let defunc_body = self.defunc_preserving_params(*body);
                self.env.remove(&param);

                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty,
                    span: term.span,
                    kind: TermKind::Lambda(Lambda {
                        params: vec![(param, param_ty)],
                        body: Box::new(defunc_body),
                        ret_ty,
                        captures,
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
            TermKind::Var(sym) => {
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

            TermKind::Lambda(..) => self.defunc_lambda(term),

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
            | TermKind::StringLit(_)
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

            TermKind::Pack { .. } | TermKind::Unpack { .. } => {
                unreachable!("Pack/Unpack nodes not yet produced at this phase")
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
        let rebuilt_lam = self.rebuild_nested_lam(&params, body_result.term, span);

        // Lift to top-level
        let lifted_sym = self.fresh_symbol();

        // Record captures for this lifted lambda (used during HOF specialization)
        self.lifted_lambda_captures.insert(lifted_sym, captures.clone());

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
                    kind: TermKind::Var(lifted_sym),
                },
                sv: StaticVal::Lambda {
                    lifted_name: lifted_sym,
                    captures: vec![],
                },
            }
        } else {
            // Has captures: append captures as additional parameters at the end
            // Build: |original_params...| |captures...| body
            let wrapped = self.append_capture_params(rebuilt_lam, &captures, span);
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
                    kind: TermKind::Var(lifted_sym),
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
        let new_soac = match soac {
            SoacOp::Map { lam, inputs } => {
                let lam = self.defunc_lambda_in_soac(lam, span);
                let inputs = inputs.into_iter().map(|ae| self.defunc_array_expr(ae)).collect();
                SoacOp::Map { lam, inputs }
            }
            SoacOp::Reduce { op, ne, input, props } => {
                let op = self.defunc_lambda_in_soac(op, span);
                let ne = Box::new(self.defunc_term(*ne).term);
                let input = self.defunc_array_expr(input);
                SoacOp::Reduce { op, ne, input, props }
            }
            SoacOp::Scan { op, ne, input } => {
                let op = self.defunc_lambda_in_soac(op, span);
                let ne = Box::new(self.defunc_term(*ne).term);
                let input = self.defunc_array_expr(input);
                SoacOp::Scan { op, ne, input }
            }
            SoacOp::Filter { pred, input } => {
                let pred = self.defunc_lambda_in_soac(pred, span);
                let input = self.defunc_array_expr(input);
                SoacOp::Filter { pred, input }
            }
            SoacOp::Scatter { dest, indices, values } => {
                let indices = self.defunc_array_expr(indices);
                let values = self.defunc_array_expr(values);
                SoacOp::Scatter { dest, indices, values }
            }
            SoacOp::ReduceByIndex { dest, op, ne, indices, values, props } => {
                let op = self.defunc_lambda_in_soac(op, span);
                let ne = Box::new(self.defunc_term(*ne).term);
                let indices = self.defunc_array_expr(indices);
                let values = self.defunc_array_expr(values);
                SoacOp::ReduceByIndex { dest, op, ne, indices, values, props }
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
    /// and fills `Lambda.captures` with the resolved capture terms. The resulting Lambda
    /// has its body replaced with a reference to the lifted function and captures populated.
    fn defunc_lambda_in_soac(&mut self, lam: Lambda, span: Span) -> Lambda {
        // Build a term from the lambda so we can use defunc_lambda
        let lam_ty = if lam.params.len() == 1 {
            Type::Constructed(
                TypeName::Arrow,
                vec![lam.params[0].1.clone(), lam.ret_ty.clone()],
            )
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
            StaticVal::Lambda { lifted_name, captures } => (*lifted_name, captures.clone()),
            _ => {
                // This shouldn't happen for a lambda, but be safe
                match &result.term.kind {
                    TermKind::Var(sym) => (*sym, vec![]),
                    _ => panic!("BUG: defunc_lambda didn't produce a Var or Lambda StaticVal"),
                }
            }
        };

        // Build the captures list: (sym, ty, term) triples
        let captures: Vec<(SymbolId, Type<TypeName>, Term)> = capture_terms
            .into_iter()
            .map(|t| {
                let sym = match &t.kind {
                    TermKind::Var(s) => *s,
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
            kind: TermKind::Var(lifted_name),
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

        Lambda {
            params,
            body: Box::new(body),
            ret_ty,
            captures,
        }
    }

    /// Defunctionalize an ArrayExpr (returns transformed ArrayExpr).
    fn defunc_array_expr(&mut self, ae: ArrayExpr) -> ArrayExpr {
        match ae {
            ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(self.defunc_term(*t).term)),
            ArrayExpr::Zip(exprs) => ArrayExpr::Zip(exprs.into_iter().map(|e| self.defunc_array_expr(e)).collect()),
            ArrayExpr::Soac(op) => {
                // Create a dummy type/span for the recursive call
                let result = self.defunc_soac(*op, Type::Constructed(TypeName::Unit, vec![]), Span::new(0, 0, 0, 0));
                match result.term.kind {
                    TermKind::Soac(s) => ArrayExpr::Soac(Box::new(s)),
                    _ => unreachable!(),
                }
            }
            ArrayExpr::Generate { shape, index_fn, elem_ty } => {
                let index_fn = self.defunc_lambda_in_soac(index_fn, Span::new(0, 0, 0, 0));
                ArrayExpr::Generate { shape, index_fn, elem_ty }
            }
            ArrayExpr::Literal(terms) => ArrayExpr::Literal(terms.into_iter().map(|t| self.defunc_term(t).term).collect()),
            ArrayExpr::Range { start, len } => ArrayExpr::Range {
                start: Box::new(self.defunc_term(*start).term),
                len: Box::new(self.defunc_term(*len).term),
            },
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

            TermKind::Var(sym) => {
                let sym = *sym; // Copy out of the match to avoid borrow issues
                // Get static value for the callee
                let callee_sv = self.env.get(&sym).cloned().unwrap_or_else(|| {
                    // Top-level function reference - treat as Lambda with no captures
                    if self.top_level.contains(&sym) && is_arrow_type(&base_func.ty) {
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
                self.build_curried_app(*lifted_name, all_args, result_ty, span)
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
        func_sym: SymbolId,
        args: Vec<Term>,
        result_ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let func_term = Term {
            id: self.term_ids.next_id(),
            ty: Type::Constructed(TypeName::Ignored, vec![]),
            span,
            kind: TermKind::Var(func_sym),
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
    fn extract_lambda_params(&self, term: Term) -> (Vec<(SymbolId, Type<TypeName>)>, Term) {
        let mut params = Vec::new();
        let mut current = term;

        loop {
            match current.kind {
                TermKind::Lambda(Lambda { params: lam_params, body, .. }) => {
                    params.extend(lam_params);
                    current = *body;
                }
                _ => break,
            }
        }

        (params, current)
    }

    /// Rebuild nested lambdas from params and body.
    fn rebuild_nested_lam(
        &mut self,
        params: &[(SymbolId, Type<TypeName>)],
        body: Term,
        span: Span,
    ) -> Term {
        params.iter().rev().fold(body, |acc, (param, param_ty)| {
            let lam_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), acc.ty.clone()]);
            Term {
                id: self.term_ids.next_id(),
                ty: lam_ty,
                span,
                kind: TermKind::Lambda(Lambda {
                    params: vec![(*param, param_ty.clone())],
                    body: Box::new(acc.clone()),
                    ret_ty: acc.ty.clone(),
                    captures: vec![],
                }),
            }
        })
    }

    /// Append capture parameters to a lambda (captures at end).
    /// Given `|x| |y| body` and captures [a, b], produces `|x| |y| |a| |b| body`.
    fn append_capture_params(&mut self, lam: Term, captures: &[Term], span: Span) -> Term {
        // Find the innermost body (unwrap all lambdas)
        fn extract_inner(term: Term) -> (Vec<(SymbolId, Type<TypeName>, Type<TypeName>)>, Term) {
            match term.kind {
                TermKind::Lambda(Lambda { params, body, .. }) => {
                    let (mut rest_params, inner) = extract_inner(*body);
                    for (p, ty) in params.into_iter().rev() {
                        rest_params.insert(0, (p, ty, term.ty.clone()));
                    }
                    (rest_params, inner)
                }
                _ => (vec![], term),
            }
        }

        let (orig_params, inner_body) = extract_inner(lam);

        // Wrap inner body with capture lambdas (innermost first, so reverse iterate)
        let with_captures = captures.iter().rev().fold(inner_body, |acc, cap_term| {
            // Extract symbol from the capture term (must be a Var)
            let cap_sym = match &cap_term.kind {
                TermKind::Var(sym) => *sym,
                _ => panic!("BUG: capture term is not a Var: {:?}", cap_term.kind),
            };
            let cap_ty = &cap_term.ty;
            let result_ty = Type::Constructed(TypeName::Arrow, vec![cap_ty.clone(), acc.ty.clone()]);
            Term {
                id: self.term_ids.next_id(),
                ty: result_ty,
                span,
                kind: TermKind::Lambda(Lambda {
                    params: vec![(cap_sym, cap_ty.clone())],
                    body: Box::new(acc.clone()),
                    ret_ty: acc.ty.clone(),
                    captures: vec![],
                }),
            }
        });

        // Rebuild original params around it
        orig_params.into_iter().rev().fold(with_captures, |acc, (param, param_ty, _orig_ty)| {
            let result_ty = Type::Constructed(TypeName::Arrow, vec![param_ty.clone(), acc.ty.clone()]);
            Term {
                id: self.term_ids.next_id(),
                ty: result_ty,
                span,
                kind: TermKind::Lambda(Lambda {
                    params: vec![(param, param_ty)],
                    body: Box::new(acc.clone()),
                    ret_ty: acc.ty.clone(),
                    captures: vec![],
                }),
            }
        })
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
            return self.build_specialized_call(
                specialized_sym,
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

        // Generate specialized variant symbol
        let hof_name = self.symbols.get(hof_sym).expect("BUG: HOF symbol not in table");
        let specialized_name = format!("{}${}", hof_name, self.specialization_counter);
        self.specialization_counter += 1;
        let specialized_sym = self.symbols.alloc(specialized_name);

        // Cache early to prevent infinite recursion on mutually recursive HOFs
        self.specialization_cache.insert(cache_key, specialized_sym);

        // Get the function parameter symbol from the HOF definition
        let func_param_sym = self.get_func_param_sym(hof_def, func_param_idx);

        // Extract params FIRST, then substitute in the inner body
        // (Substitution must happen AFTER unwrapping lambdas because the func param
        // is bound by one of those lambdas, and substitute_var correctly avoids
        // substituting in shadowed scopes)
        let (params, inner_body) = self.extract_lambda_params(hof_def.body.clone());

        // Simple substitution in inner body: replace func_param_sym with lambda_sym
        let substituted_inner = self.substitute_var(&inner_body, func_param_sym, lambda_sym);

        // Build new param list: remove function param at func_param_idx, add captures at end
        let mut new_params: Vec<(SymbolId, Type<TypeName>)> =
            params.into_iter().enumerate().filter(|(i, _)| *i != func_param_idx).map(|(_, p)| p).collect();

        // Add captures as trailing parameters (extract symbol from each capture Term)
        for cap_term in captures {
            let cap_sym = match &cap_term.kind {
                TermKind::Var(sym) => *sym,
                _ => panic!("BUG: capture term is not a Var: {:?}", cap_term.kind),
            };
            new_params.push((cap_sym, cap_term.ty.clone()));
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
        let defunced_body = self.defunc_term(substituted_inner).term;

        // Restore env
        self.env = old_env;

        // Apply type substitution to the defunced body AND to the param types
        let defunced_body = apply_type_subst_to_term(&defunced_body, &type_subst, &mut self.term_ids);
        let new_params: Vec<(SymbolId, Type<TypeName>)> =
            new_params.into_iter().map(|(sym, ty)| (sym, apply_type_subst(&ty, &type_subst))).collect();

        // Rebuild nested lambdas with substituted params
        let rebuilt = self.rebuild_nested_lam(&new_params, defunced_body, hof_def.body.span);

        let specialized_def = Def {
            name: specialized_sym,
            ty: rebuilt.ty.clone(),
            body: rebuilt,
            meta: DefMeta::Function,
            arity: new_params.len(),
        };

        // Register
        self.lifted_defs.push(specialized_def);

        // Build call to specialized function
        self.build_specialized_call(specialized_sym, func_param_idx, arg_results, captures, ty, span)
    }

    /// Simple variable substitution: replace all occurrences of old_sym with new_sym.
    fn substitute_var(&mut self, term: &Term, old_sym: SymbolId, new_sym: SymbolId) -> Term {
        match &term.kind {
            TermKind::Var(sym) if *sym == old_sym => Term {
                id: self.term_ids.next_id(),
                ty: term.ty.clone(),
                span: term.span,
                kind: TermKind::Var(new_sym),
            },

            TermKind::Var(_)
            | TermKind::IntLit(_)
            | TermKind::FloatLit(_)
            | TermKind::BoolLit(_)
            | TermKind::StringLit(_)
            | TermKind::BinOp(_)
            | TermKind::UnOp(_)
            | TermKind::Extern(_) => term.clone(),

            TermKind::App { func, arg } => Term {
                id: self.term_ids.next_id(),
                ty: term.ty.clone(),
                span: term.span,
                kind: TermKind::App {
                    func: Box::new(self.substitute_var(func, old_sym, new_sym)),
                    arg: Box::new(self.substitute_var(arg, old_sym, new_sym)),
                },
            },

            TermKind::Lambda(Lambda { params, body, ret_ty, captures }) => {
                // Don't substitute if any param shadows old_sym
                if params.iter().any(|(p, _)| *p == old_sym) {
                    term.clone()
                } else {
                    Term {
                        id: self.term_ids.next_id(),
                        ty: term.ty.clone(),
                        span: term.span,
                        kind: TermKind::Lambda(Lambda {
                            params: params.clone(),
                            body: Box::new(self.substitute_var(body, old_sym, new_sym)),
                            ret_ty: ret_ty.clone(),
                            captures: captures.clone(),
                        }),
                    }
                }
            }

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                let new_rhs = self.substitute_var(rhs, old_sym, new_sym);
                // Don't substitute in body if name shadows old_sym
                let new_body = if *name == old_sym {
                    (**body).clone()
                } else {
                    self.substitute_var(body, old_sym, new_sym)
                };
                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty.clone(),
                    span: term.span,
                    kind: TermKind::Let {
                        name: *name,
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
                    cond: Box::new(self.substitute_var(cond, old_sym, new_sym)),
                    then_branch: Box::new(self.substitute_var(then_branch, old_sym, new_sym)),
                    else_branch: Box::new(self.substitute_var(else_branch, old_sym, new_sym)),
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
                let shadows = *loop_var == old_sym || init_bindings.iter().any(|(n, _, _)| *n == old_sym);

                // init is evaluated outside the loop scope
                let new_init = self.substitute_var(init, old_sym, new_sym);

                // init_bindings exprs reference loop_var (inside loop scope)
                let new_init_bindings: Vec<_> = init_bindings
                    .iter()
                    .map(|(n, ty, e)| {
                        let new_e =
                            if shadows { e.clone() } else { self.substitute_var(e, old_sym, new_sym) };
                        (*n, ty.clone(), new_e)
                    })
                    .collect();

                // iter/bound are outside loop scope, cond is inside
                let new_kind = match kind {
                    LoopKind::For { var, var_ty, iter } => LoopKind::For {
                        var: *var,
                        var_ty: var_ty.clone(),
                        iter: Box::new(self.substitute_var(iter, old_sym, new_sym)),
                    },
                    LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                        var: *var,
                        var_ty: var_ty.clone(),
                        bound: Box::new(self.substitute_var(bound, old_sym, new_sym)),
                    },
                    LoopKind::While { cond } => {
                        let new_cond = if shadows {
                            (**cond).clone()
                        } else {
                            self.substitute_var(cond, old_sym, new_sym)
                        };
                        LoopKind::While {
                            cond: Box::new(new_cond),
                        }
                    }
                };

                let new_body =
                    if shadows { (**body).clone() } else { self.substitute_var(body, old_sym, new_sym) };

                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty.clone(),
                    span: term.span,
                    kind: TermKind::Loop {
                        loop_var: *loop_var,
                        loop_var_ty: loop_var_ty.clone(),
                        init: Box::new(new_init),
                        init_bindings: new_init_bindings,
                        kind: new_kind,
                        body: Box::new(new_body),
                    },
                }
            }

            TermKind::Soac(ref soac) => {
                let new_soac = self.substitute_var_soac(soac, old_sym, new_sym);
                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty.clone(),
                    span: term.span,
                    kind: TermKind::Soac(new_soac),
                }
            }

            TermKind::ArrayExpr(ref ae) => {
                let new_ae = self.substitute_var_array_expr(ae, old_sym, new_sym);
                Term {
                    id: self.term_ids.next_id(),
                    ty: term.ty.clone(),
                    span: term.span,
                    kind: TermKind::ArrayExpr(new_ae),
                }
            }

            TermKind::Force(ref inner) => Term {
                id: self.term_ids.next_id(),
                ty: term.ty.clone(),
                span: term.span,
                kind: TermKind::Force(Box::new(self.substitute_var(inner, old_sym, new_sym))),
            },

            TermKind::Pack { .. } | TermKind::Unpack { .. } => {
                unreachable!("Pack/Unpack nodes not yet produced at this phase")
            }
        }
    }

    fn substitute_var_lambda(&mut self, lam: &Lambda, old_sym: SymbolId, new_sym: SymbolId) -> Lambda {
        if lam.params.iter().any(|(p, _)| *p == old_sym) {
            lam.clone()
        } else {
            Lambda {
                params: lam.params.clone(),
                body: Box::new(self.substitute_var(&lam.body, old_sym, new_sym)),
                ret_ty: lam.ret_ty.clone(),
                captures: lam.captures.iter().map(|(s, ty, t)| (*s, ty.clone(), self.substitute_var(t, old_sym, new_sym))).collect(),
            }
        }
    }

    fn substitute_var_soac(&mut self, soac: &SoacOp, old_sym: SymbolId, new_sym: SymbolId) -> SoacOp {
        match soac {
            SoacOp::Map { lam, inputs } => SoacOp::Map {
                lam: self.substitute_var_lambda(lam, old_sym, new_sym),
                inputs: inputs.iter().map(|ae| self.substitute_var_array_expr(ae, old_sym, new_sym)).collect(),
            },
            SoacOp::Reduce { op, ne, input, props } => SoacOp::Reduce {
                op: self.substitute_var_lambda(op, old_sym, new_sym),
                ne: Box::new(self.substitute_var(ne, old_sym, new_sym)),
                input: self.substitute_var_array_expr(input, old_sym, new_sym),
                props: props.clone(),
            },
            SoacOp::Scan { op, ne, input } => SoacOp::Scan {
                op: self.substitute_var_lambda(op, old_sym, new_sym),
                ne: Box::new(self.substitute_var(ne, old_sym, new_sym)),
                input: self.substitute_var_array_expr(input, old_sym, new_sym),
            },
            SoacOp::Filter { pred, input } => SoacOp::Filter {
                pred: self.substitute_var_lambda(pred, old_sym, new_sym),
                input: self.substitute_var_array_expr(input, old_sym, new_sym),
            },
            SoacOp::Scatter { dest, indices, values } => SoacOp::Scatter {
                dest: dest.clone(),
                indices: self.substitute_var_array_expr(indices, old_sym, new_sym),
                values: self.substitute_var_array_expr(values, old_sym, new_sym),
            },
            SoacOp::ReduceByIndex { dest, op, ne, indices, values, props } => SoacOp::ReduceByIndex {
                dest: dest.clone(),
                op: self.substitute_var_lambda(op, old_sym, new_sym),
                ne: Box::new(self.substitute_var(ne, old_sym, new_sym)),
                indices: self.substitute_var_array_expr(indices, old_sym, new_sym),
                values: self.substitute_var_array_expr(values, old_sym, new_sym),
                props: props.clone(),
            },
        }
    }

    fn substitute_var_array_expr(&mut self, ae: &ArrayExpr, old_sym: SymbolId, new_sym: SymbolId) -> ArrayExpr {
        match ae {
            ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(self.substitute_var(t, old_sym, new_sym))),
            ArrayExpr::Zip(exprs) => ArrayExpr::Zip(exprs.iter().map(|e| self.substitute_var_array_expr(e, old_sym, new_sym)).collect()),
            ArrayExpr::Soac(op) => ArrayExpr::Soac(Box::new(self.substitute_var_soac(op, old_sym, new_sym))),
            ArrayExpr::Generate { shape, index_fn, elem_ty } => ArrayExpr::Generate {
                shape: shape.clone(),
                index_fn: self.substitute_var_lambda(index_fn, old_sym, new_sym),
                elem_ty: elem_ty.clone(),
            },
            ArrayExpr::Literal(terms) => ArrayExpr::Literal(terms.iter().map(|t| self.substitute_var(t, old_sym, new_sym)).collect()),
            ArrayExpr::Range { start, len } => ArrayExpr::Range {
                start: Box::new(self.substitute_var(start, old_sym, new_sym)),
                len: Box::new(self.substitute_var(len, old_sym, new_sym)),
            },
        }
    }

    /// Get the parameter symbol at a given index from a function definition.
    fn get_func_param_sym(&self, def: &Def, param_idx: usize) -> SymbolId {
        let mut body = &def.body;
        let mut idx = 0;
        while let TermKind::Lambda(Lambda { params, body: inner, .. }) = &body.kind {
            for (param, _) in params {
                if idx == param_idx {
                    return *param;
                }
                idx += 1;
            }
            body = inner;
        }
        panic!(
            "BUG: param index {} out of bounds for function definition",
            param_idx
        )
    }

    /// Build a call to a specialized HOF.
    fn build_specialized_call(
        &mut self,
        specialized_sym: SymbolId,
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

        let call_term = self.build_curried_app(specialized_sym, call_args, ty, span);
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
pub fn defunctionalize(program: Program, known_defs: &HashSet<String>) -> Program {
    Defunctionalizer::defunctionalize(program, known_defs)
}
