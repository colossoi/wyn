//! Pin each compute entry's storage-param buffer region into its type.
//!
//! After type inference a view-array entry param `xs: []f32` has an
//! unresolved region *variable* in its type's region slot — the checker
//! cannot know which descriptor `xs` reads, because the source never says
//! so. The binding is decided separately, by `compute_entry_binding_layout`
//! (auto-allocated `set, 0..N` in declaration order) or by an explicit
//! `#[storage(set, binding)]` attribute. This pass takes that decided
//! binding and writes it into the type: it substitutes the param's region
//! variable → `Region(set, binding)` throughout the entry body.
//!
//! The entry is the one place a region is genuinely born (a buffer attaches
//! to a parameter); every other view — a slice of `xs`, `xs` passed into a
//! function, an element loaded from it — inherits its region by ordinary
//! unification, exactly as element types flow. So pinning entry params is
//! enough; `monomorphize` (which keys on the concrete region) then
//! specializes view functions per buffer.
//!
//! Runs first in the TLC pipeline, before any pass that observes regions.

#[cfg(test)]
#[path = "pin_entry_regions_tests.rs"]
mod pin_entry_regions_tests;

use super::{ArrayExpr, DefMeta, Lambda, LoopKind, Program, SoacOp, Term, TermKind};
use crate::ast::TypeName;
use crate::binding_layout::{compute_entry_binding_layout, extract_storage_binding};
use crate::interface::{EntryDecl, EntryParamBindingKind};
use crate::tlc::monomorphize::apply_subst;
use crate::types::{TypeExt, region_tag};
use crate::{BindingRef, SymbolId};
use polytype::Type;
use std::collections::HashMap;

/// Region-variable → concrete `Region(set, binding)` substitution.
type RegionSubst = HashMap<usize, Type<TypeName>>;

pub fn run(program: &mut Program) {
    for def in program.defs.iter_mut() {
        let DefMeta::EntryPoint(entry) = &def.meta else {
            continue;
        };
        let (params, _) = extract_params(&def.body);
        let mut subst = RegionSubst::new();
        collect_region_subst(&params, entry, &mut subst);
        if subst.is_empty() {
            continue;
        }
        def.ty = apply_subst(&def.ty, &subst);
        subst_term(&mut def.body, &subst);
    }
}

/// Flatten an entry's curried lambda chain into `(param_sym, param_ty)` pairs.
fn extract_params(term: &Term) -> (Vec<(SymbolId, Type<TypeName>)>, &Term) {
    match &term.kind {
        TermKind::Lambda(lam) => {
            let (mut inner, body) = extract_params(&lam.body);
            let mut params = lam.params.clone();
            params.append(&mut inner);
            (params, body)
        }
        _ => (vec![], term),
    }
}

/// Map each storage param's region variable to its concrete region. An
/// explicit `#[storage(set, binding)]` wins; otherwise the auto-allocated
/// layout decides. Non-view params contribute nothing (their type has no
/// region slot variable to pin).
fn collect_region_subst(params: &[(SymbolId, Type<TypeName>)], entry: &EntryDecl, subst: &mut RegionSubst) {
    let layout = compute_entry_binding_layout(params, entry, crate::egir::from_tlc::AUTO_STORAGE_SET);

    for (i, (_sym, ty)) in params.iter().enumerate() {
        let ty = TypeExt::strip_unique(ty);

        // Host-wired explicit binding: the region is the attribute's.
        if let Some(binding) = entry.params.get(i).and_then(extract_storage_binding) {
            pin_view_region(ty, binding, subst);
            continue;
        }

        match layout.get(i).and_then(|b| b.as_ref()).map(|b| &b.kind) {
            Some(EntryParamBindingKind::Single { binding, .. }) => {
                pin_view_region(ty, *binding, subst);
            }
            Some(EntryParamBindingKind::TupleOfViews(fields)) => {
                if let Type::Constructed(TypeName::Tuple(_), comps) = ty {
                    for (field, comp_ty) in fields.iter().zip(comps) {
                        pin_view_region(TypeExt::strip_unique(comp_ty), field.binding, subst);
                    }
                }
            }
            None => {}
        }
    }
}

/// If `view_ty`'s region slot is a free variable, record `var → Region(binding)`.
/// A region already concrete (or absent — a non-array) is left untouched.
fn pin_view_region(view_ty: &Type<TypeName>, binding: BindingRef, subst: &mut RegionSubst) {
    if let Some(Type::Variable(id)) = view_ty.array_region() {
        subst.insert(*id, region_tag(binding));
    }
}

// ---------------------------------------------------------------------------
// In-place type substitution over a term tree. Mirrors the type positions in
// `monomorphize`'s `apply_subst_term` family, but mutates every type in place
// (term ids unchanged — this only concretizes region variables).
// ---------------------------------------------------------------------------

fn subst_term(t: &mut Term, s: &RegionSubst) {
    t.ty = apply_subst(&t.ty, s);
    match &mut t.kind {
        TermKind::Var(_)
        | TermKind::IntLit(_)
        | TermKind::FloatLit(_)
        | TermKind::BoolLit(_)
        | TermKind::UnitLit
        | TermKind::BinOp(_)
        | TermKind::UnOp(_)
        | TermKind::Extern(_) => {}
        TermKind::Coerce { inner, target_ty } => {
            subst_term(inner, s);
            *target_ty = apply_subst(target_ty, s);
        }
        TermKind::Soac(soac) => subst_soac(soac, s),
        TermKind::ArrayExpr(ae) => subst_array_expr(ae, s),
        TermKind::Tuple(parts) | TermKind::VecLit(parts) => {
            parts.iter_mut().for_each(|p| subst_term(p, s));
        }
        TermKind::TupleProj { tuple, .. } => subst_term(tuple, s),
        TermKind::Index { array, index } => {
            subst_term(array, s);
            subst_term(index, s);
        }
        TermKind::OutputSlotStore { value, .. } => subst_term(value, s),
        TermKind::Lambda(lam) => subst_lambda(lam, s),
        TermKind::App { func, args } => {
            subst_term(func, s);
            args.iter_mut().for_each(|a| subst_term(a, s));
        }
        TermKind::Let {
            name_ty, rhs, body, ..
        } => {
            *name_ty = apply_subst(name_ty, s);
            subst_term(rhs, s);
            subst_term(body, s);
        }
        TermKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            subst_term(cond, s);
            subst_term(then_branch, s);
            subst_term(else_branch, s);
        }
        TermKind::Loop {
            loop_var_ty,
            init,
            init_bindings,
            kind,
            body,
            ..
        } => {
            *loop_var_ty = apply_subst(loop_var_ty, s);
            subst_term(init, s);
            for (_, ty, expr) in init_bindings.iter_mut() {
                *ty = apply_subst(ty, s);
                subst_term(expr, s);
            }
            match kind {
                LoopKind::For { var_ty, iter, .. } => {
                    *var_ty = apply_subst(var_ty, s);
                    subst_term(iter, s);
                }
                LoopKind::ForRange { var_ty, bound, .. } => {
                    *var_ty = apply_subst(var_ty, s);
                    subst_term(bound, s);
                }
                LoopKind::While { cond } => subst_term(cond, s),
            }
            subst_term(body, s);
        }
    }
}

fn subst_lambda(lam: &mut Lambda, s: &RegionSubst) {
    for (_, ty) in lam.params.iter_mut() {
        *ty = apply_subst(ty, s);
    }
    lam.ret_ty = apply_subst(&lam.ret_ty, s);
    subst_term(&mut lam.body, s);
}

fn subst_soac_body(sb: &mut super::SoacBody, s: &RegionSubst) {
    subst_lambda(&mut sb.lam, s);
    for (_, ty, t) in sb.captures.iter_mut() {
        *ty = apply_subst(ty, s);
        subst_term(t, s);
    }
}

fn subst_soac(soac: &mut SoacOp, s: &RegionSubst) {
    match soac {
        SoacOp::Map { lam, inputs, .. } => {
            subst_soac_body(lam, s);
            inputs.iter_mut().for_each(|ae| subst_array_expr(ae, s));
        }
        SoacOp::Reduce { op, ne, input } => {
            subst_soac_body(op, s);
            subst_term(ne, s);
            subst_array_expr(input, s);
        }
        SoacOp::Scan {
            op,
            reduce_op,
            ne,
            input,
            ..
        } => {
            subst_soac_body(op, s);
            subst_soac_body(reduce_op, s);
            subst_term(ne, s);
            subst_array_expr(input, s);
        }
        SoacOp::Filter { pred, input, .. } => {
            subst_soac_body(pred, s);
            subst_array_expr(input, s);
        }
        SoacOp::Scatter { indices, values, .. } => {
            subst_array_expr(indices, s);
            subst_array_expr(values, s);
        }
        SoacOp::ReduceByIndex {
            op,
            ne,
            indices,
            values,
            ..
        } => {
            subst_soac_body(op, s);
            subst_term(ne, s);
            subst_array_expr(indices, s);
            subst_array_expr(values, s);
        }
        SoacOp::Redomap {
            op,
            reduce_op,
            ne,
            inputs,
        } => {
            subst_soac_body(op, s);
            subst_soac_body(reduce_op, s);
            subst_term(ne, s);
            inputs.iter_mut().for_each(|ae| subst_array_expr(ae, s));
        }
    }
}

fn subst_array_expr(ae: &mut ArrayExpr, s: &RegionSubst) {
    match ae {
        ArrayExpr::Ref(t) => subst_term(t, s),
        ArrayExpr::Zip(exprs) => exprs.iter_mut().for_each(|e| subst_array_expr(e, s)),
        ArrayExpr::Soac(op) => subst_soac(op, s),
        ArrayExpr::Literal(terms) => terms.iter_mut().for_each(|t| subst_term(t, s)),
        ArrayExpr::Range { start, len, step } => {
            subst_term(start, s);
            subst_term(len, s);
            if let Some(step) = step {
                subst_term(step, s);
            }
        }
        ArrayExpr::StorageView(sv) => {
            subst_term(&mut sv.offset, s);
            subst_term(&mut sv.len, s);
            sv.elem_ty = apply_subst(&sv.elem_ty, s);
        }
    }
}
