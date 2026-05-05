//! HOF-specialization phase boundary (phase 2 of the defunctionalization split).
//!
//! Owns the post-condition verifier for HOF specialization: every
//! reachable top-level def has zero function-typed parameters. Wired in
//! post-`filter_reachable`, where dead unspecialized HOF definitions
//! have already been removed.
//!
//! The actual specialization logic still lives inside
//! `tlc::defunctionalize` (tightly coupled with lambda lifting and
//! StaticVal tracking). This module owns the architectural seam — the
//! verifier guarantees the existing pipeline produces well-shaped output
//! and gives a target shape for any future extraction of the
//! specialization phase.

use super::{ArrayExpr, Def, Lambda, LoopKind, Place, Program, SoacOp, Term, TermIdSource, TermKind};
use crate::SymbolId;
use crate::ast::TypeName;
use polytype::Type;
use std::collections::HashMap;

// =============================================================================
// Verifier
// =============================================================================

#[derive(Debug)]
pub enum HofSpecializeError {
    /// A top-level def has a function-typed parameter. Every HOF should
    /// have been specialized away into monomorphic copies by this point.
    FunctionTypedParam {
        def: SymbolId,
        param_index: usize,
    },
}

pub fn verify_hof_specialized(program: &Program) -> Result<(), HofSpecializeError> {
    for def in &program.defs {
        verify_def_no_arrow_params(def)?;
    }
    Ok(())
}

fn verify_def_no_arrow_params(def: &Def) -> Result<(), HofSpecializeError> {
    let mut current = &def.ty;
    let mut param_index = 0;
    while let Type::Constructed(TypeName::Arrow, args) = current {
        if args.len() != 2 {
            break;
        }
        if is_arrow_type(&args[0]) {
            return Err(HofSpecializeError::FunctionTypedParam {
                def: def.name,
                param_index,
            });
        }
        current = &args[1];
        param_index += 1;
    }
    Ok(())
}

// =============================================================================
// HOF detection
// =============================================================================

/// Information about a higher-order function (HOF).
#[derive(Debug, Clone)]
pub(super) struct HofInfo {
    /// Which parameter indices are function-typed.
    pub(super) func_param_indices: Vec<usize>,
    /// Original definition for cloning during specialization (None for intrinsics).
    pub(super) def: Option<Def>,
}

/// Check if a type is an arrow type (function type).
pub(super) fn is_arrow_type(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::Arrow, _))
}

/// Extract parameter types from a nested arrow type.
/// e.g., `(A -> B -> C) -> D -> E` returns `[(A -> B -> C), D]`.
pub(super) fn extract_param_types(ty: &Type<TypeName>) -> Vec<Type<TypeName>> {
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
pub(super) fn detect_hofs(defs: &[Def]) -> HashMap<SymbolId, HofInfo> {
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
// Type substitution
// =============================================================================
//
// Used by the specialization pass to clone a HOF body and rewrite it for
// the concrete types the call site supplies.

pub(super) type TypeSubst = HashMap<polytype::Variable, Type<TypeName>>;

/// Build a type substitution by unifying a polymorphic type with a concrete type.
pub(super) fn build_type_subst(
    poly_ty: &Type<TypeName>,
    concrete_ty: &Type<TypeName>,
    subst: &mut TypeSubst,
) {
    match (poly_ty, concrete_ty) {
        (Type::Variable(id), concrete) => {
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
            for (p, c) in poly_args.iter().zip(concrete_args.iter()) {
                build_type_subst(p, c, subst);
            }
        }
        _ => {}
    }
}

pub(super) fn apply_type_subst(ty: &Type<TypeName>, subst: &TypeSubst) -> Type<TypeName> {
    match ty {
        Type::Variable(id) => subst.get(id).cloned().unwrap_or_else(|| ty.clone()),
        Type::Constructed(name, args) => {
            let new_args: Vec<_> = args.iter().map(|a| apply_type_subst(a, subst)).collect();
            Type::Constructed(name.clone(), new_args)
        }
    }
}

/// Format a type as a string key for cache lookups.
pub(super) fn format_type_for_key(ty: &Type<TypeName>) -> String {
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

pub(super) fn apply_type_subst_to_term(
    term: &Term,
    subst: &TypeSubst,
    term_ids: &mut TermIdSource,
) -> Term {
    let new_ty = apply_type_subst(&term.ty, subst);
    let new_kind = match &term.kind {
        TermKind::Var(name) => TermKind::Var(*name),
        TermKind::BinOp(op) => TermKind::BinOp(op.clone()),
        TermKind::UnOp(op) => TermKind::UnOp(op.clone()),
        TermKind::IntLit(s) => TermKind::IntLit(s.clone()),
        TermKind::FloatLit(f) => TermKind::FloatLit(*f),
        TermKind::BoolLit(b) => TermKind::BoolLit(*b),
        TermKind::Extern(linkage) => TermKind::Extern(linkage.clone()),
        TermKind::App { func, args } => TermKind::App {
            func: Box::new(apply_type_subst_to_term(func, subst, term_ids)),
            args: args.iter().map(|a| apply_type_subst_to_term(a, subst, term_ids)).collect(),
        },
        TermKind::Lambda(Lambda {
            params,
            body,
            ret_ty,
            captures,
        }) => TermKind::Lambda(Lambda {
            params: params.iter().map(|(p, ty)| (*p, apply_type_subst(ty, subst))).collect(),
            body: Box::new(apply_type_subst_to_term(body, subst, term_ids)),
            ret_ty: apply_type_subst(ret_ty, subst),
            captures: captures
                .iter()
                .map(|(s, ty, t)| {
                    (
                        *s,
                        apply_type_subst(ty, subst),
                        apply_type_subst_to_term(t, subst, term_ids),
                    )
                })
                .collect(),
        }),
        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => TermKind::Let {
            name: *name,
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
                        *name,
                        apply_type_subst(ty, subst),
                        apply_type_subst_to_term(expr, subst, term_ids),
                    )
                })
                .collect();
            let new_kind = match kind {
                LoopKind::For { var, var_ty, iter } => LoopKind::For {
                    var: *var,
                    var_ty: apply_type_subst(var_ty, subst),
                    iter: Box::new(apply_type_subst_to_term(iter, subst, term_ids)),
                },
                LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                    var: *var,
                    var_ty: apply_type_subst(var_ty, subst),
                    bound: Box::new(apply_type_subst_to_term(bound, subst, term_ids)),
                },
                LoopKind::While { cond } => LoopKind::While {
                    cond: Box::new(apply_type_subst_to_term(cond, subst, term_ids)),
                },
            };
            TermKind::Loop {
                loop_var: *loop_var,
                loop_var_ty: apply_type_subst(loop_var_ty, subst),
                init: Box::new(apply_type_subst_to_term(init, subst, term_ids)),
                init_bindings: new_init_bindings,
                kind: new_kind,
                body: Box::new(apply_type_subst_to_term(body, subst, term_ids)),
            }
        }
        TermKind::Soac(soac) => TermKind::Soac(apply_type_subst_to_soac(soac, subst, term_ids)),
        TermKind::ArrayExpr(ae) => TermKind::ArrayExpr(apply_type_subst_to_array_expr(ae, subst, term_ids)),
        TermKind::Force(inner) => {
            TermKind::Force(Box::new(apply_type_subst_to_term(inner, subst, term_ids)))
        }
    };
    Term {
        id: term_ids.next_id(),
        ty: new_ty,
        span: term.span,
        kind: new_kind,
    }
}

pub(super) fn apply_type_subst_to_lambda(
    lam: &Lambda,
    subst: &TypeSubst,
    term_ids: &mut TermIdSource,
) -> Lambda {
    Lambda {
        params: lam.params.iter().map(|(p, ty)| (*p, apply_type_subst(ty, subst))).collect(),
        body: Box::new(apply_type_subst_to_term(&lam.body, subst, term_ids)),
        ret_ty: apply_type_subst(&lam.ret_ty, subst),
        captures: lam
            .captures
            .iter()
            .map(|(s, ty, t)| {
                (
                    *s,
                    apply_type_subst(ty, subst),
                    apply_type_subst_to_term(t, subst, term_ids),
                )
            })
            .collect(),
    }
}

pub(super) fn apply_type_subst_to_soac(
    soac: &SoacOp,
    subst: &TypeSubst,
    term_ids: &mut TermIdSource,
) -> SoacOp {
    match soac {
        SoacOp::Map {
            lam,
            inputs,
            consumes_input,
        } => SoacOp::Map {
            lam: apply_type_subst_to_lambda(lam, subst, term_ids),
            inputs: inputs.iter().map(|ae| apply_type_subst_to_array_expr(ae, subst, term_ids)).collect(),
            consumes_input: *consumes_input,
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
        SoacOp::Scatter {
            dest,
            indices,
            values,
        } => SoacOp::Scatter {
            dest: apply_type_subst_to_place(dest, subst, term_ids),
            indices: apply_type_subst_to_array_expr(indices, subst, term_ids),
            values: apply_type_subst_to_array_expr(values, subst, term_ids),
        },
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
            props,
        } => SoacOp::ReduceByIndex {
            dest: apply_type_subst_to_place(dest, subst, term_ids),
            op: apply_type_subst_to_lambda(op, subst, term_ids),
            ne: Box::new(apply_type_subst_to_term(ne, subst, term_ids)),
            indices: apply_type_subst_to_array_expr(indices, subst, term_ids),
            values: apply_type_subst_to_array_expr(values, subst, term_ids),
            props: props.clone(),
        },
        SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
            props,
        } => SoacOp::Redomap {
            op: apply_type_subst_to_lambda(op, subst, term_ids),
            reduce_op: apply_type_subst_to_lambda(reduce_op, subst, term_ids),
            ne: Box::new(apply_type_subst_to_term(ne, subst, term_ids)),
            inputs: inputs.iter().map(|ae| apply_type_subst_to_array_expr(ae, subst, term_ids)).collect(),
            props: props.clone(),
        },
    }
}

pub(super) fn apply_type_subst_to_array_expr(
    ae: &ArrayExpr,
    subst: &TypeSubst,
    term_ids: &mut TermIdSource,
) -> ArrayExpr {
    match ae {
        ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(apply_type_subst_to_term(t, subst, term_ids))),
        ArrayExpr::Zip(exprs) => ArrayExpr::Zip(
            exprs.iter().map(|e| apply_type_subst_to_array_expr(e, subst, term_ids)).collect(),
        ),
        ArrayExpr::Soac(op) => ArrayExpr::Soac(Box::new(apply_type_subst_to_soac(op, subst, term_ids))),
        ArrayExpr::Generate {
            shape,
            index_fn,
            elem_ty,
        } => ArrayExpr::Generate {
            shape: shape.clone(),
            index_fn: apply_type_subst_to_lambda(index_fn, subst, term_ids),
            elem_ty: apply_type_subst(elem_ty, subst),
        },
        ArrayExpr::Literal(terms) => {
            ArrayExpr::Literal(terms.iter().map(|t| apply_type_subst_to_term(t, subst, term_ids)).collect())
        }
        ArrayExpr::Range { start, len } => ArrayExpr::Range {
            start: Box::new(apply_type_subst_to_term(start, subst, term_ids)),
            len: Box::new(apply_type_subst_to_term(len, subst, term_ids)),
        },
        ArrayExpr::StorageBuffer { .. } => {
            unreachable!("StorageBuffer introduced after defunctionalization")
        }
    }
}

pub(super) fn apply_type_subst_to_place(
    place: &Place,
    subst: &TypeSubst,
    term_ids: &mut TermIdSource,
) -> Place {
    match place {
        Place::BufferSlice {
            base,
            offset,
            shape,
            elem_ty,
        } => Place::BufferSlice {
            base: Box::new(apply_type_subst_to_term(base, subst, term_ids)),
            offset: Box::new(apply_type_subst_to_term(offset, subst, term_ids)),
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
// Tests
// =============================================================================

#[cfg(test)]
#[path = "hof_specialize_tests.rs"]
mod hof_specialize_tests;
