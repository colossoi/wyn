//! HOF-specialization pass (phase 2 of the closure pipeline).
//!
//! Consumes the closure-converted program + `ClosureInfo` produced by
//! `closure_convert::run`. Specialises every higher-order function call
//! by cloning the HOF body and substituting the function-typed parameter
//! with the resolved callable symbol. After this pass, every top-level
//! def has zero function-typed parameters.
//!
//! Single dispatch point: for each `App(Var(hof_sym), args)` whose
//! `hof_sym` is a HOF, the func-arg-slot's symbol is resolved via
//! `closure_info.resolve_callable(sym)`. No `Lambda` discovery, no
//! let-alias tracking, no four-source extract — those concerns are
//! handled by `closure_convert` upstream.

use super::VarRef;
use super::closure_convert::{CallableValue, ClosureInfo};
use super::{
    ArrayExpr, Def, DefMeta, Lambda, LoopKind, Place, Program, SoacOp, Term, TermIdSource, TermKind,
};
use crate::ast::{Span, TypeName};
use crate::{SymbolId, SymbolTable};
use polytype::Type;
use std::collections::{HashMap, HashSet};

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
        TermKind::Var(v) => TermKind::Var(*v),
        TermKind::BinOp(op) => TermKind::BinOp(op.clone()),
        TermKind::UnOp(op) => TermKind::UnOp(op.clone()),
        TermKind::IntLit(s) => TermKind::IntLit(s.clone()),
        TermKind::FloatLit(f) => TermKind::FloatLit(*f),
        TermKind::BoolLit(b) => TermKind::BoolLit(*b),
        TermKind::UnitLit => TermKind::UnitLit,
        TermKind::Coerce { inner, target_ty } => TermKind::Coerce {
            inner: Box::new(apply_type_subst_to_term(inner, subst, term_ids)),
            target_ty: apply_type_subst(target_ty, subst),
        },
        TermKind::Extern(linkage) => TermKind::Extern(linkage.clone()),
        TermKind::App { func, args } => TermKind::App {
            func: Box::new(apply_type_subst_to_term(func, subst, term_ids)),
            args: args.iter().map(|a| apply_type_subst_to_term(a, subst, term_ids)).collect(),
        },
        TermKind::Lambda(Lambda { params, body, ret_ty }) => TermKind::Lambda(Lambda {
            params: params.iter().map(|(p, ty)| (*p, apply_type_subst(ty, subst))).collect(),
            body: Box::new(apply_type_subst_to_term(body, subst, term_ids)),
            ret_ty: apply_type_subst(ret_ty, subst),
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
        TermKind::Tuple(parts) => {
            TermKind::Tuple(parts.iter().map(|p| apply_type_subst_to_term(p, subst, term_ids)).collect())
        }
        TermKind::TupleProj { tuple, idx } => TermKind::TupleProj {
            tuple: Box::new(apply_type_subst_to_term(tuple, subst, term_ids)),
            idx: *idx,
        },
        TermKind::Index { array, index } => TermKind::Index {
            array: Box::new(apply_type_subst_to_term(array, subst, term_ids)),
            index: Box::new(apply_type_subst_to_term(index, subst, term_ids)),
        },
        TermKind::VecLit(parts) => {
            TermKind::VecLit(parts.iter().map(|p| apply_type_subst_to_term(p, subst, term_ids)).collect())
        }
        TermKind::OutputSlotStore { slot_index, value } => TermKind::OutputSlotStore {
            slot_index: *slot_index,
            value: Box::new(apply_type_subst_to_term(value, subst, term_ids)),
        },
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
    }
}

pub(super) fn apply_type_subst_to_soac_body(
    sb: &super::SoacBody,
    subst: &TypeSubst,
    term_ids: &mut TermIdSource,
) -> super::SoacBody {
    super::SoacBody {
        lam: apply_type_subst_to_lambda(&sb.lam, subst, term_ids),
        captures: sb
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
            destination,
        } => SoacOp::Map {
            lam: apply_type_subst_to_soac_body(lam, subst, term_ids),
            inputs: inputs.iter().map(|ae| apply_type_subst_to_array_expr(ae, subst, term_ids)).collect(),
            destination: *destination,
        },
        SoacOp::Reduce { op, ne, input } => SoacOp::Reduce {
            op: apply_type_subst_to_soac_body(op, subst, term_ids),
            ne: Box::new(apply_type_subst_to_term(ne, subst, term_ids)),
            input: apply_type_subst_to_array_expr(input, subst, term_ids),
        },
        SoacOp::Scan {
            op,
            reduce_op,
            ne,
            input,
            destination,
        } => SoacOp::Scan {
            op: apply_type_subst_to_soac_body(op, subst, term_ids),
            reduce_op: apply_type_subst_to_soac_body(reduce_op, subst, term_ids),
            ne: Box::new(apply_type_subst_to_term(ne, subst, term_ids)),
            input: apply_type_subst_to_array_expr(input, subst, term_ids),
            destination: *destination,
        },
        SoacOp::Filter {
            pred,
            input,
            destination,
        } => SoacOp::Filter {
            pred: apply_type_subst_to_soac_body(pred, subst, term_ids),
            input: apply_type_subst_to_array_expr(input, subst, term_ids),
            destination: *destination,
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
        } => SoacOp::ReduceByIndex {
            dest: apply_type_subst_to_place(dest, subst, term_ids),
            op: apply_type_subst_to_soac_body(op, subst, term_ids),
            ne: Box::new(apply_type_subst_to_term(ne, subst, term_ids)),
            indices: apply_type_subst_to_array_expr(indices, subst, term_ids),
            values: apply_type_subst_to_array_expr(values, subst, term_ids),
        },
        SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
        } => SoacOp::Redomap {
            op: apply_type_subst_to_soac_body(op, subst, term_ids),
            reduce_op: apply_type_subst_to_soac_body(reduce_op, subst, term_ids),
            ne: Box::new(apply_type_subst_to_term(ne, subst, term_ids)),
            inputs: inputs.iter().map(|ae| apply_type_subst_to_array_expr(ae, subst, term_ids)).collect(),
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
        ArrayExpr::Literal(terms) => {
            ArrayExpr::Literal(terms.iter().map(|t| apply_type_subst_to_term(t, subst, term_ids)).collect())
        }
        ArrayExpr::Range { start, len, step } => ArrayExpr::Range {
            start: Box::new(apply_type_subst_to_term(start, subst, term_ids)),
            len: Box::new(apply_type_subst_to_term(len, subst, term_ids)),
            step: step.as_ref().map(|s| Box::new(apply_type_subst_to_term(s, subst, term_ids))),
        },
        ArrayExpr::StorageView(sv) => ArrayExpr::StorageView(super::StorageView {
            binding: sv.binding,
            offset: Box::new(apply_type_subst_to_term(&sv.offset, subst, term_ids)),
            len: Box::new(apply_type_subst_to_term(&sv.len, subst, term_ids)),
            elem_ty: sv.elem_ty.clone(),
        }),
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
// Specialization-pass helpers
// =============================================================================
//
// Pure helpers used by the HOF specialization runtime in
// `defunctionalize::specialize_user_hof`. They take `&mut TermIdSource`
// for fresh term IDs and don't touch any other Defunctionalizer state.

/// Walk a function definition's leading-Lambda spine to find the
/// parameter symbol at `param_idx`. Panics if the index is out of
/// bounds.
pub(super) fn get_func_param_sym(def: &Def, param_idx: usize) -> SymbolId {
    let mut body = &def.body;
    let mut idx = 0;
    while let TermKind::Lambda(Lambda {
        params, body: inner, ..
    }) = &body.kind
    {
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

/// Build the call-side of a HOF specialization: `specialized(args, captures)`,
/// dropping the function-position arg at `func_param_idx` and appending
/// the resolved capture terms.
pub(super) fn build_specialized_call(
    specialized_sym: SymbolId,
    func_param_idx: usize,
    arg_terms: &[Term],
    captures: &[Term],
    ty: Type<TypeName>,
    span: Span,
    term_ids: &mut TermIdSource,
) -> Term {
    let mut call_args = Vec::new();
    for (i, t) in arg_terms.iter().enumerate() {
        if i != func_param_idx {
            call_args.push(t.clone());
        }
    }
    call_args.extend(captures.iter().cloned());
    super::closure_convert::build_app_call(specialized_sym, call_args, ty, span, term_ids)
}

/// Substitute every free occurrence of `old_sym` in `term` with
/// `new_sym`. Binder-aware (Lambda / Let / Loop respect shadowing).
pub(super) fn substitute_var(
    term: &Term,
    old_sym: SymbolId,
    new_sym: SymbolId,
    term_ids: &mut TermIdSource,
) -> Term {
    match &term.kind {
        TermKind::Var(VarRef::Symbol(sym)) if *sym == old_sym => Term {
            id: term_ids.next_id(),
            ty: term.ty.clone(),
            span: term.span,
            kind: TermKind::Var(VarRef::Symbol(new_sym)),
        },

        TermKind::Var(_)
        | TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::UnitLit
        | TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::Extern(_) => term.clone(),

        TermKind::Coerce { inner, target_ty } => Term {
            id: term_ids.next_id(),
            ty: term.ty.clone(),
            span: term.span,
            kind: TermKind::Coerce {
                inner: Box::new(substitute_var(inner, old_sym, new_sym, term_ids)),
                target_ty: target_ty.clone(),
            },
        },

        TermKind::App { func, args } => Term {
            id: term_ids.next_id(),
            ty: term.ty.clone(),
            span: term.span,
            kind: TermKind::App {
                func: Box::new(substitute_var(func, old_sym, new_sym, term_ids)),
                args: args.iter().map(|a| substitute_var(a, old_sym, new_sym, term_ids)).collect(),
            },
        },

        TermKind::Lambda(lam) => {
            if lam.params.iter().any(|(p, _)| *p == old_sym) {
                term.clone()
            } else {
                Term {
                    id: term_ids.next_id(),
                    ty: term.ty.clone(),
                    span: term.span,
                    kind: TermKind::Lambda(substitute_var_lambda(lam, old_sym, new_sym, term_ids)),
                }
            }
        }

        TermKind::Let {
            name,
            name_ty,
            rhs,
            body,
        } => {
            let new_rhs = substitute_var(rhs, old_sym, new_sym, term_ids);
            let new_body = if *name == old_sym {
                (**body).clone()
            } else {
                substitute_var(body, old_sym, new_sym, term_ids)
            };
            Term {
                id: term_ids.next_id(),
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
            id: term_ids.next_id(),
            ty: term.ty.clone(),
            span: term.span,
            kind: TermKind::If {
                cond: Box::new(substitute_var(cond, old_sym, new_sym, term_ids)),
                then_branch: Box::new(substitute_var(then_branch, old_sym, new_sym, term_ids)),
                else_branch: Box::new(substitute_var(else_branch, old_sym, new_sym, term_ids)),
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
            let shadows = *loop_var == old_sym || init_bindings.iter().any(|(n, _, _)| *n == old_sym);
            let new_init = substitute_var(init, old_sym, new_sym, term_ids);
            let new_init_bindings: Vec<_> = init_bindings
                .iter()
                .map(|(n, ty, e)| {
                    let new_e =
                        if shadows { e.clone() } else { substitute_var(e, old_sym, new_sym, term_ids) };
                    (*n, ty.clone(), new_e)
                })
                .collect();

            let new_kind = match kind {
                LoopKind::For { var, var_ty, iter } => LoopKind::For {
                    var: *var,
                    var_ty: var_ty.clone(),
                    iter: Box::new(substitute_var(iter, old_sym, new_sym, term_ids)),
                },
                LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                    var: *var,
                    var_ty: var_ty.clone(),
                    bound: Box::new(substitute_var(bound, old_sym, new_sym, term_ids)),
                },
                LoopKind::While { cond } => {
                    let new_cond = if shadows {
                        (**cond).clone()
                    } else {
                        substitute_var(cond, old_sym, new_sym, term_ids)
                    };
                    LoopKind::While {
                        cond: Box::new(new_cond),
                    }
                }
            };

            let new_body =
                if shadows { (**body).clone() } else { substitute_var(body, old_sym, new_sym, term_ids) };

            Term {
                id: term_ids.next_id(),
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

        TermKind::Soac(soac) => Term {
            id: term_ids.next_id(),
            ty: term.ty.clone(),
            span: term.span,
            kind: TermKind::Soac(substitute_var_soac(soac, old_sym, new_sym, term_ids)),
        },

        TermKind::ArrayExpr(ae) => Term {
            id: term_ids.next_id(),
            ty: term.ty.clone(),
            span: term.span,
            kind: TermKind::ArrayExpr(substitute_var_array_expr(ae, old_sym, new_sym, term_ids)),
        },

        TermKind::Tuple(parts) => Term {
            id: term_ids.next_id(),
            ty: term.ty.clone(),
            span: term.span,
            kind: TermKind::Tuple(
                parts.iter().map(|p| substitute_var(p, old_sym, new_sym, term_ids)).collect(),
            ),
        },
        TermKind::TupleProj { tuple, idx } => Term {
            id: term_ids.next_id(),
            ty: term.ty.clone(),
            span: term.span,
            kind: TermKind::TupleProj {
                tuple: Box::new(substitute_var(tuple, old_sym, new_sym, term_ids)),
                idx: *idx,
            },
        },
        TermKind::Index { array, index } => Term {
            id: term_ids.next_id(),
            ty: term.ty.clone(),
            span: term.span,
            kind: TermKind::Index {
                array: Box::new(substitute_var(array, old_sym, new_sym, term_ids)),
                index: Box::new(substitute_var(index, old_sym, new_sym, term_ids)),
            },
        },
        TermKind::VecLit(parts) => Term {
            id: term_ids.next_id(),
            ty: term.ty.clone(),
            span: term.span,
            kind: TermKind::VecLit(
                parts.iter().map(|p| substitute_var(p, old_sym, new_sym, term_ids)).collect(),
            ),
        },
        TermKind::OutputSlotStore { slot_index, value } => Term {
            id: term_ids.next_id(),
            ty: term.ty.clone(),
            span: term.span,
            kind: TermKind::OutputSlotStore {
                slot_index: *slot_index,
                value: Box::new(substitute_var(value, old_sym, new_sym, term_ids)),
            },
        },
    }
}

fn substitute_var_lambda(
    lam: &Lambda,
    old_sym: SymbolId,
    new_sym: SymbolId,
    term_ids: &mut TermIdSource,
) -> Lambda {
    if lam.params.iter().any(|(p, _)| *p == old_sym) {
        lam.clone()
    } else {
        Lambda {
            params: lam.params.clone(),
            body: Box::new(substitute_var(&lam.body, old_sym, new_sym, term_ids)),
            ret_ty: lam.ret_ty.clone(),
        }
    }
}

fn substitute_var_soac_body(
    sb: &super::SoacBody,
    old_sym: SymbolId,
    new_sym: SymbolId,
    term_ids: &mut TermIdSource,
) -> super::SoacBody {
    super::SoacBody {
        lam: substitute_var_lambda(&sb.lam, old_sym, new_sym, term_ids),
        captures: sb
            .captures
            .iter()
            .map(|(s, ty, t)| (*s, ty.clone(), substitute_var(t, old_sym, new_sym, term_ids)))
            .collect(),
    }
}

fn substitute_var_soac(
    soac: &SoacOp,
    old_sym: SymbolId,
    new_sym: SymbolId,
    term_ids: &mut TermIdSource,
) -> SoacOp {
    match soac {
        SoacOp::Map {
            lam,
            inputs,
            destination,
        } => SoacOp::Map {
            lam: substitute_var_soac_body(lam, old_sym, new_sym, term_ids),
            inputs: inputs
                .iter()
                .map(|ae| substitute_var_array_expr(ae, old_sym, new_sym, term_ids))
                .collect(),
            destination: *destination,
        },
        SoacOp::Reduce { op, ne, input } => SoacOp::Reduce {
            op: substitute_var_soac_body(op, old_sym, new_sym, term_ids),
            ne: Box::new(substitute_var(ne, old_sym, new_sym, term_ids)),
            input: substitute_var_array_expr(input, old_sym, new_sym, term_ids),
        },
        SoacOp::Scan {
            op,
            reduce_op,
            ne,
            input,
            destination,
        } => SoacOp::Scan {
            op: substitute_var_soac_body(op, old_sym, new_sym, term_ids),
            reduce_op: substitute_var_soac_body(reduce_op, old_sym, new_sym, term_ids),
            ne: Box::new(substitute_var(ne, old_sym, new_sym, term_ids)),
            input: substitute_var_array_expr(input, old_sym, new_sym, term_ids),
            destination: *destination,
        },
        SoacOp::Filter {
            pred,
            input,
            destination,
        } => SoacOp::Filter {
            pred: substitute_var_soac_body(pred, old_sym, new_sym, term_ids),
            input: substitute_var_array_expr(input, old_sym, new_sym, term_ids),
            destination: *destination,
        },
        SoacOp::Scatter {
            dest,
            indices,
            values,
        } => SoacOp::Scatter {
            dest: dest.clone(),
            indices: substitute_var_array_expr(indices, old_sym, new_sym, term_ids),
            values: substitute_var_array_expr(values, old_sym, new_sym, term_ids),
        },
        SoacOp::ReduceByIndex {
            dest,
            op,
            ne,
            indices,
            values,
        } => SoacOp::ReduceByIndex {
            dest: dest.clone(),
            op: substitute_var_soac_body(op, old_sym, new_sym, term_ids),
            ne: Box::new(substitute_var(ne, old_sym, new_sym, term_ids)),
            indices: substitute_var_array_expr(indices, old_sym, new_sym, term_ids),
            values: substitute_var_array_expr(values, old_sym, new_sym, term_ids),
        },
        SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
        } => SoacOp::Redomap {
            op: substitute_var_soac_body(op, old_sym, new_sym, term_ids),
            reduce_op: substitute_var_soac_body(reduce_op, old_sym, new_sym, term_ids),
            ne: Box::new(substitute_var(ne, old_sym, new_sym, term_ids)),
            inputs: inputs
                .iter()
                .map(|ae| substitute_var_array_expr(ae, old_sym, new_sym, term_ids))
                .collect(),
        },
    }
}

fn substitute_var_array_expr(
    ae: &ArrayExpr,
    old_sym: SymbolId,
    new_sym: SymbolId,
    term_ids: &mut TermIdSource,
) -> ArrayExpr {
    match ae {
        ArrayExpr::Ref(t) => ArrayExpr::Ref(Box::new(substitute_var(t, old_sym, new_sym, term_ids))),
        ArrayExpr::Zip(exprs) => ArrayExpr::Zip(
            exprs.iter().map(|e| substitute_var_array_expr(e, old_sym, new_sym, term_ids)).collect(),
        ),
        ArrayExpr::Soac(op) => {
            ArrayExpr::Soac(Box::new(substitute_var_soac(op, old_sym, new_sym, term_ids)))
        }
        ArrayExpr::Literal(terms) => ArrayExpr::Literal(
            terms.iter().map(|t| substitute_var(t, old_sym, new_sym, term_ids)).collect(),
        ),
        ArrayExpr::Range { start, len, step } => ArrayExpr::Range {
            start: Box::new(substitute_var(start, old_sym, new_sym, term_ids)),
            len: Box::new(substitute_var(len, old_sym, new_sym, term_ids)),
            step: step.as_ref().map(|s| Box::new(substitute_var(s, old_sym, new_sym, term_ids))),
        },
        ArrayExpr::StorageView(sv) => ArrayExpr::StorageView(super::StorageView {
            binding: sv.binding,
            offset: Box::new(substitute_var(&sv.offset, old_sym, new_sym, term_ids)),
            len: Box::new(substitute_var(&sv.len, old_sym, new_sym, term_ids)),
            elem_ty: sv.elem_ty.clone(),
        }),
    }
}

// =============================================================================
// HofSpecializer pass
// =============================================================================

struct HofSpecializer<'a> {
    symbols: SymbolTable,
    top_level: HashSet<SymbolId>,
    closure_info: &'a ClosureInfo,
    hof_info: HashMap<SymbolId, HofInfo>,
    specialized_defs: Vec<Def>,
    specialization_cache: HashMap<(SymbolId, SymbolId, Vec<String>), SymbolId>,
    /// Cache: `(lifted_def_sym, [callable_sym at each callable-cap slot])`
    /// → specialized lifted def sym. Keyed by the lifted def we're cloning
    /// plus the resolved callables that flow into its callable-typed
    /// captures, so a second SoacBody asking for the same specialization
    /// reuses the variant instead of minting another.
    closure_spec_cache: HashMap<(SymbolId, Vec<SymbolId>), SymbolId>,
    /// Lookup by sym for every def the cascade may need to clone — built
    /// once after the main loop from `transformed + specialized_defs`,
    /// then read-only during cascade. New cascade-specialized defs are
    /// added back in.
    defs_by_sym: HashMap<SymbolId, Def>,
    specialization_counter: usize,
    term_ids: TermIdSource,
}

impl<'a> HofSpecializer<'a> {
    fn run(program: Program, closure_info: &'a ClosureInfo) -> Program {
        let hof_info = detect_hofs(&program.defs);
        let top_level: HashSet<SymbolId> = program.defs.iter().map(|d| d.name).collect();

        let mut hs = Self {
            symbols: program.symbols,
            top_level,
            closure_info,
            hof_info,
            specialized_defs: vec![],
            specialization_cache: HashMap::new(),
            closure_spec_cache: HashMap::new(),
            defs_by_sym: HashMap::new(),
            specialization_counter: 0,
            term_ids: TermIdSource::new(),
        };

        let transformed: Vec<Def> = program
            .defs
            .into_iter()
            .map(|def| Def {
                body: hs.rewrite_def_body(def.body),
                ..def
            })
            .collect();

        // Cascade closure specialization. The main loop above eliminates
        // function-typed params from outer HOFs by cloning + substituting
        // the called code into the HOF body. But a HOF body that captures
        // its function-typed param into a lifted closure (e.g. via a SOAC
        // operand whose lambda is a separate top-level def) leaves the
        // callable flowing into the closure as a runtime arg — and the
        // closure itself still has a function-typed parameter for that
        // slot. Walk every def's body and, at each `SoacBody` whose
        // captures include `(_, arrow_ty, Var(known_callable))`, clone
        // the lifted def referenced by `lam.body`, substitute the
        // callable into its body, drop the callable param from its
        // signature, and rewrite the SoacBody to reference the
        // specialized variant with the callable capture removed.
        // Recurses through the cloned bodies, so nested closures cascade
        // until no callable captures remain.
        for d in transformed.iter().chain(hs.specialized_defs.iter()) {
            hs.defs_by_sym.insert(d.name, d.clone());
        }
        let transformed: Vec<Def> = transformed
            .into_iter()
            .map(|def| Def {
                body: hs.cascade_specialize_term(def.body),
                ..def
            })
            .collect();
        let main_loop_specialized: Vec<Def> = std::mem::take(&mut hs.specialized_defs)
            .into_iter()
            .map(|def| Def {
                body: hs.cascade_specialize_term(def.body),
                ..def
            })
            .collect();
        // Anything the cascade itself emitted while walking those bodies
        // is in `hs.specialized_defs` now; chain them in too.
        let cascade_specialized = std::mem::take(&mut hs.specialized_defs);

        Program {
            defs: transformed.into_iter().chain(main_loop_specialized).chain(cascade_specialized).collect(),
            symbols: hs.symbols,
            ..program
        }
    }

    /// Walk a def body, preserving the outer parameter-spine `Lambda`
    /// nodes (those carry the def's named parameters). Anything below
    /// the spine routes through `rewrite_term`, which scans App nodes
    /// for HOF calls.
    fn rewrite_def_body(&mut self, term: Term) -> Term {
        match term.kind {
            TermKind::Lambda(Lambda { params, body, ret_ty }) => {
                let new_body = self.rewrite_def_body(*body);
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
            _ => self.rewrite_term(term),
        }
    }

    /// Recursively walk a term. At every `App(Var(hof_sym), args)` whose
    /// `hof_sym` is a HOF and whose func-arg-slot resolves through
    /// `ClosureInfo` to a callable, dispatch to `specialize_call`.
    fn rewrite_term(&mut self, term: Term) -> Term {
        let ty = term.ty.clone();
        let span = term.span;
        match term.kind {
            TermKind::App { func, args } => {
                let new_func = self.rewrite_term(*func);
                let new_args: Vec<Term> = args.into_iter().map(|a| self.rewrite_term(a)).collect();

                if let TermKind::Var(VarRef::Symbol(sym)) = &new_func.kind {
                    let hof_sym = *sym;
                    if let Some(hof_info) = self.hof_info.get(&hof_sym).cloned() {
                        for &func_param_idx in &hof_info.func_param_indices {
                            if func_param_idx >= new_args.len() {
                                continue;
                            }
                            let arg_sym = match &new_args[func_param_idx].kind {
                                TermKind::Var(VarRef::Symbol(s)) => *s,
                                _ => continue,
                            };
                            let (code, captures) = match self.closure_info.resolve_callable(arg_sym) {
                                Some(CallableValue::Direct(code)) => (*code, Vec::new()),
                                Some(CallableValue::Closure { code, captures, .. }) => {
                                    (*code, captures.clone())
                                }
                                None => continue,
                            };
                            let hof_def =
                                hof_info.def.as_ref().expect("BUG: user HOF should have def in hof_info");
                            return self.specialize_call(
                                hof_sym,
                                hof_def,
                                func_param_idx,
                                code,
                                &captures,
                                &new_args,
                                ty,
                                span,
                            );
                        }
                    }
                }

                Term {
                    id: self.term_ids.next_id(),
                    ty,
                    span,
                    kind: TermKind::App {
                        func: Box::new(new_func),
                        args: new_args,
                    },
                }
            }

            TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Let {
                    name,
                    name_ty,
                    rhs: Box::new(self.rewrite_term(*rhs)),
                    body: Box::new(self.rewrite_term(*body)),
                },
            },

            TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::If {
                    cond: Box::new(self.rewrite_term(*cond)),
                    then_branch: Box::new(self.rewrite_term(*then_branch)),
                    else_branch: Box::new(self.rewrite_term(*else_branch)),
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
                let init = self.rewrite_term(*init);
                let init_bindings: Vec<_> =
                    init_bindings.into_iter().map(|(n, ty, e)| (n, ty, self.rewrite_term(e))).collect();
                let kind = match kind {
                    LoopKind::For { var, var_ty, iter } => LoopKind::For {
                        var,
                        var_ty,
                        iter: Box::new(self.rewrite_term(*iter)),
                    },
                    LoopKind::ForRange { var, var_ty, bound } => LoopKind::ForRange {
                        var,
                        var_ty,
                        bound: Box::new(self.rewrite_term(*bound)),
                    },
                    LoopKind::While { cond } => LoopKind::While {
                        cond: Box::new(self.rewrite_term(*cond)),
                    },
                };
                let body = self.rewrite_term(*body);
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

            // SOAC envelopes don't trigger HOF specialization — their
            // bodies are already lifted top-level defs (closure-convert
            // post-condition). Pass through.
            TermKind::Soac(_)
            | TermKind::ArrayExpr(_)
            | TermKind::Lambda(_)
            | TermKind::Var(_)
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
                    inner: Box::new(self.rewrite_term(*inner)),
                    target_ty,
                },
            },

            TermKind::Tuple(parts) => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Tuple(parts.into_iter().map(|p| self.rewrite_term(p)).collect()),
            },
            TermKind::TupleProj { tuple, idx } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::TupleProj {
                    tuple: Box::new(self.rewrite_term(*tuple)),
                    idx,
                },
            },
            TermKind::Index { array, index } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::Index {
                    array: Box::new(self.rewrite_term(*array)),
                    index: Box::new(self.rewrite_term(*index)),
                },
            },
            TermKind::VecLit(parts) => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::VecLit(parts.into_iter().map(|p| self.rewrite_term(p)).collect()),
            },
            TermKind::OutputSlotStore { slot_index, value } => Term {
                id: self.term_ids.next_id(),
                ty,
                span,
                kind: TermKind::OutputSlotStore {
                    slot_index,
                    value: Box::new(self.rewrite_term(*value)),
                },
            },
        }
    }

    /// Specialize one HOF call by cloning the HOF body, substituting
    /// the function-typed parameter with `code`, dropping the
    /// function-typed param from the spine, appending capture params,
    /// and recursively rewriting the cloned body so further nested
    /// HOF calls get specialized too.
    #[allow(clippy::too_many_arguments)]
    fn specialize_call(
        &mut self,
        hof_sym: SymbolId,
        hof_def: &Def,
        func_param_idx: usize,
        code: SymbolId,
        captures: &[Term],
        arg_terms: &[Term],
        ty: Type<TypeName>,
        span: Span,
    ) -> Term {
        let arg_type_keys: Vec<String> = arg_terms.iter().map(|a| format_type_for_key(&a.ty)).collect();
        let cache_key = (hof_sym, code, arg_type_keys);
        if let Some(specialized_sym) = self.specialization_cache.get(&cache_key).cloned() {
            return build_specialized_call(
                specialized_sym,
                func_param_idx,
                arg_terms,
                captures,
                ty,
                span,
                &mut self.term_ids,
            );
        }

        let mut type_subst = TypeSubst::new();
        let poly_param_types = extract_param_types(&hof_def.ty);
        for (i, poly_ty) in poly_param_types.iter().enumerate() {
            if i < arg_terms.len() {
                build_type_subst(poly_ty, &arg_terms[i].ty, &mut type_subst);
            }
        }

        let hof_name = self.symbols.get(hof_sym).expect("BUG: HOF symbol not in table");
        let specialized_name = format!("{}${}", hof_name, self.specialization_counter);
        self.specialization_counter += 1;
        let specialized_sym = self.symbols.alloc(specialized_name.clone());
        self.specialization_cache.insert(cache_key, specialized_sym);

        let func_param_sym = get_func_param_sym(hof_def, func_param_idx);
        let (params, inner_body) = super::extract_lambda_params(&hof_def.body);
        let substituted_inner = substitute_var(&inner_body, func_param_sym, code, &mut self.term_ids);

        let mut new_params: Vec<(SymbolId, Type<TypeName>)> =
            params.into_iter().enumerate().filter(|(i, _)| *i != func_param_idx).map(|(_, p)| p).collect();

        let mut capture_subst: Vec<(SymbolId, SymbolId)> = Vec::with_capacity(captures.len());
        for cap_term in captures {
            let outer_sym = match &cap_term.kind {
                TermKind::Var(VarRef::Symbol(sym)) => *sym,
                _ => panic!("BUG: capture term is not a Var: {:?}", cap_term.kind),
            };
            let outer_name = crate::symbol_name_or_bug(&self.symbols, outer_sym).to_string();
            let fresh_sym = self.symbols.alloc(format!("{}__cap_{}", specialized_name, outer_name));
            capture_subst.push((outer_sym, fresh_sym));
            new_params.push((fresh_sym, cap_term.ty.clone()));
        }

        // Recursively rewrite for any nested HOF calls in the cloned
        // body, then thread captures (using the same logic
        // `closure_calls_lower` applies globally). Capture threading
        // happens BEFORE the outer→fresh capture renaming below: the
        // threaded calls reference outer-scope capture syms, which the
        // renaming then rewrites to the specialization's fresh local
        // params. Threading is idempotent — when
        // `closure_calls_lower::run` later walks this body, it sees
        // `args.len() == param_count + captures.len()` and skips.
        let rewritten_inner = self.rewrite_term(substituted_inner);
        let threaded_inner = super::closure_calls_lower::thread_captures_in_term(
            rewritten_inner,
            self.closure_info,
            &mut self.term_ids,
        );
        let mut rewritten_inner = threaded_inner;
        for (outer_sym, fresh_sym) in &capture_subst {
            rewritten_inner = substitute_var(&rewritten_inner, *outer_sym, *fresh_sym, &mut self.term_ids);
        }

        let rewritten_inner = apply_type_subst_to_term(&rewritten_inner, &type_subst, &mut self.term_ids);
        let new_params: Vec<(SymbolId, Type<TypeName>)> =
            new_params.into_iter().map(|(sym, ty)| (sym, apply_type_subst(&ty, &type_subst))).collect();

        let rebuilt = super::closure_convert::rebuild_nested_lam(
            &new_params,
            rewritten_inner,
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

        self.top_level.insert(specialized_sym);
        self.specialized_defs.push(specialized_def);

        build_specialized_call(
            specialized_sym,
            func_param_idx,
            arg_terms,
            captures,
            ty,
            span,
            &mut self.term_ids,
        )
    }

    // =========================================================================
    // Cascade closure specialization
    //
    // Walks a term, recurses into every `SoacBody`, and for each captured
    // arrow-typed `Var` that resolves to a known callable, clones the
    // lifted def referenced by `lam.body` with the callable substituted in
    // and the callable param dropped. The SoacBody itself is rewritten to
    // reference the specialized variant and to omit the callable capture.
    // =========================================================================

    fn cascade_specialize_term(&mut self, term: Term) -> Term {
        let term = term.map_children(&mut |child| self.cascade_specialize_term(child));
        match term.kind {
            TermKind::Soac(soac) => Term {
                kind: TermKind::Soac(self.cascade_specialize_soac(soac)),
                ..term
            },
            TermKind::ArrayExpr(ae) => Term {
                kind: TermKind::ArrayExpr(self.cascade_specialize_array_expr(ae)),
                ..term
            },
            _ => term,
        }
    }

    fn cascade_specialize_array_expr(&mut self, ae: ArrayExpr) -> ArrayExpr {
        match ae {
            ArrayExpr::Soac(boxed) => ArrayExpr::Soac(Box::new(self.cascade_specialize_soac(*boxed))),
            other => other,
        }
    }

    fn cascade_specialize_soac(&mut self, soac: SoacOp) -> SoacOp {
        match soac {
            SoacOp::Map {
                lam,
                inputs,
                destination,
            } => SoacOp::Map {
                lam: self.cascade_specialize_soac_body(lam),
                inputs,
                destination,
            },
            SoacOp::Reduce { op, ne, input } => SoacOp::Reduce {
                op: self.cascade_specialize_soac_body(op),
                ne,
                input,
            },
            SoacOp::Scan {
                op,
                reduce_op,
                ne,
                input,
                destination,
            } => SoacOp::Scan {
                op: self.cascade_specialize_soac_body(op),
                reduce_op: self.cascade_specialize_soac_body(reduce_op),
                ne,
                input,
                destination,
            },
            SoacOp::Filter {
                pred,
                input,
                destination,
            } => SoacOp::Filter {
                pred: self.cascade_specialize_soac_body(pred),
                input,
                destination,
            },
            SoacOp::Scatter {
                dest,
                indices,
                values,
            } => SoacOp::Scatter {
                dest,
                indices,
                values,
            },
            SoacOp::ReduceByIndex {
                dest,
                op,
                ne,
                indices,
                values,
            } => SoacOp::ReduceByIndex {
                dest,
                op: self.cascade_specialize_soac_body(op),
                ne,
                indices,
                values,
            },
            SoacOp::Redomap {
                op,
                reduce_op,
                ne,
                inputs,
            } => SoacOp::Redomap {
                op: self.cascade_specialize_soac_body(op),
                reduce_op: self.cascade_specialize_soac_body(reduce_op),
                ne,
                inputs,
            },
        }
    }

    fn cascade_specialize_soac_body(&mut self, sb: super::SoacBody) -> super::SoacBody {
        let callable_indices: Vec<usize> = sb
            .captures
            .iter()
            .enumerate()
            .filter_map(|(i, (_, ty, t))| {
                if !is_arrow_type(ty) {
                    return None;
                }
                let s = match &t.kind {
                    TermKind::Var(VarRef::Symbol(s)) => *s,
                    _ => return None,
                };
                if self.closure_info.resolve_callable(s).is_some() { Some(i) } else { None }
            })
            .collect();

        if callable_indices.is_empty() {
            return sb;
        }

        let lifted_sym = match &sb.lam.body.kind {
            TermKind::Var(VarRef::Symbol(s)) => *s,
            _ => return sb,
        };
        // Only specialize lifted defs we have a body for. Intrinsic SOAC
        // operators (no def) fall through.
        if !self.defs_by_sym.contains_key(&lifted_sym) {
            return sb;
        }

        let callable_syms: Vec<SymbolId> = callable_indices
            .iter()
            .map(|&i| match &sb.captures[i].2.kind {
                TermKind::Var(VarRef::Symbol(s)) => *s,
                _ => unreachable!("filtered to Var above"),
            })
            .collect();

        let cache_key = (lifted_sym, callable_syms.clone());
        let specialized_sym = if let Some(&s) = self.closure_spec_cache.get(&cache_key) {
            s
        } else {
            self.specialize_closure(lifted_sym, &callable_indices, &callable_syms, &sb.captures)
        };

        let new_lifted_ty = self
            .defs_by_sym
            .get(&specialized_sym)
            .expect("just specialized def must be in defs_by_sym")
            .ty
            .clone();
        let new_lam_body = Term {
            id: self.term_ids.next_id(),
            ty: new_lifted_ty,
            span: sb.lam.body.span,
            kind: TermKind::Var(VarRef::Symbol(specialized_sym)),
        };

        let new_captures: Vec<_> = sb
            .captures
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !callable_indices.contains(i))
            .map(|(_, c)| c)
            .collect();

        super::SoacBody {
            lam: Lambda {
                params: sb.lam.params,
                body: Box::new(new_lam_body),
                ret_ty: sb.lam.ret_ty,
            },
            captures: new_captures,
        }
    }

    fn specialize_closure(
        &mut self,
        lifted_sym: SymbolId,
        callable_indices: &[usize],
        callable_syms: &[SymbolId],
        captures: &[(SymbolId, Type<TypeName>, Term)],
    ) -> SymbolId {
        let lifted_def = self
            .defs_by_sym
            .get(&lifted_sym)
            .expect("cascade_specialize_soac_body verified presence")
            .clone();
        let (params, inner_body) = super::extract_lambda_params(&lifted_def.body);

        let drop_set: HashSet<SymbolId> = callable_indices.iter().map(|&i| captures[i].0).collect();
        let mut new_body = inner_body.clone();
        for (k, &i) in callable_indices.iter().enumerate() {
            let local_sym = captures[i].0;
            new_body = substitute_var(&new_body, local_sym, callable_syms[k], &mut self.term_ids);
        }
        let new_params: Vec<(SymbolId, Type<TypeName>)> =
            params.into_iter().filter(|(s, _)| !drop_set.contains(s)).collect();

        // Cascade into the cloned body: nested closures with their own
        // callable captures get specialised too.
        let new_body = self.cascade_specialize_term(new_body);

        let rebuilt = super::closure_convert::rebuild_nested_lam(
            &new_params,
            new_body,
            lifted_def.body.span,
            &mut self.term_ids,
        );

        let lifted_name = crate::symbol_name_or_bug(&self.symbols, lifted_sym).to_string();
        let specialized_name = format!("{}${}", lifted_name, self.specialization_counter);
        self.specialization_counter += 1;
        let new_sym = self.symbols.alloc(specialized_name);
        self.top_level.insert(new_sym);

        let new_def = Def {
            name: new_sym,
            ty: rebuilt.ty.clone(),
            body: rebuilt,
            meta: lifted_def.meta,
            arity: new_params.len(),
        };

        let key = (lifted_sym, callable_syms.to_vec());
        self.closure_spec_cache.insert(key, new_sym);
        self.defs_by_sym.insert(new_sym, new_def.clone());
        self.specialized_defs.push(new_def);

        new_sym
    }
}

/// Run HOF specialization. Consumes the closure-converted program and
/// the closure-info side-table; returns a program in which every
/// reachable top-level def has zero function-typed parameters.
pub fn run(program: Program, closure_info: &ClosureInfo) -> Program {
    HofSpecializer::run(program, closure_info)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[path = "hof_specialize_tests.rs"]
mod hof_specialize_tests;
