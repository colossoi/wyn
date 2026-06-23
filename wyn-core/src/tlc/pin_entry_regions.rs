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

use super::{ArrayExpr, DefMeta, Lambda, LoopKind, Program, SoacOp, Term, TermKind, VarRef};
use crate::ast::{Span, TypeName};
use crate::binding_layout::{compute_entry_binding_layout, extract_storage_binding};
use crate::interface::{EntryDecl, EntryParamBindingKind};
use crate::tlc::monomorphize::apply_subst;
use crate::types::{region_tag, TypeExt};
use crate::{BindingRef, SymbolId};
use polytype::Type;
use std::collections::HashMap;

/// Region-variable → concrete `Region(set, binding)` substitution.
type RegionSubst = HashMap<usize, Type<TypeName>>;

pub fn run(program: &mut Program) -> crate::error::Result<()> {
    // Sole allocator of compute-entry auto-storage binding ids in the
    // module. Threaded across entries so two entries can't claim the
    // same `(set, binding)` for differently-typed buffers. The
    // computed layouts are cached on each `EntryDecl::param_bindings`;
    // downstream passes read that cache.
    let mut binding_ids: crate::IdSource<u32> = crate::IdSource::new();

    for def in program.defs.iter_mut() {
        if !matches!(&def.meta, DefMeta::EntryPoint(_)) {
            continue;
        }
        let span = def.body.span;
        let (params, _) = extract_params(&def.body);

        let mut subst = RegionSubst::new();
        let mut region_env = HashMap::new();
        if let DefMeta::EntryPoint(entry) = &mut def.meta {
            entry.param_bindings = compute_entry_binding_layout(
                &params,
                entry,
                crate::egir::from_tlc::AUTO_STORAGE_SET,
                &mut binding_ids,
            );
            collect_region_subst(&params, entry, &mut subst, &mut region_env, span)?;
        }

        while collect_view_slice_region_subst(&def.body, &mut subst, &region_env)? {}

        if !subst.is_empty() {
            def.ty = apply_subst(&def.ty, &subst);
            subst_term(&mut def.body, &subst);
        }
    }
    Ok(())
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
fn collect_region_subst(
    params: &[(SymbolId, Type<TypeName>)],
    entry: &EntryDecl,
    subst: &mut RegionSubst,
    region_env: &mut HashMap<SymbolId, Type<TypeName>>,
    span: Span,
) -> crate::error::Result<()> {
    let layout = &entry.param_bindings;

    for (i, (_sym, ty)) in params.iter().enumerate() {
        let ty = TypeExt::strip_unique(ty);

        // Host-wired explicit binding: the region is the attribute's.
        if let Some(binding) = entry.params.get(i).and_then(extract_storage_binding) {
            pin_view_region(ty, binding, subst, span)?;
            region_env.insert(params[i].0, region_tag(binding));
            continue;
        }

        match layout.get(i).and_then(|b| b.as_ref()).map(|b| &b.kind) {
            Some(EntryParamBindingKind::Single { binding, .. }) => {
                pin_view_region(ty, *binding, subst, span)?;
                region_env.insert(params[i].0, region_tag(*binding));
            }
            Some(EntryParamBindingKind::TupleOfViews(fields)) => {
                if let Type::Constructed(TypeName::Tuple(_), comps) = ty {
                    for (field, comp_ty) in fields.iter().zip(comps) {
                        pin_view_region(TypeExt::strip_unique(comp_ty), field.binding, subst, span)?;
                    }
                }
            }
            None => {}
        }
    }
    Ok(())
}

/// If `view_ty`'s region slot is a free variable, record `var → Region(binding)`.
/// A region already concrete (or absent — a non-array) is left untouched.
///
/// A region variable shared by two params (because type inference unified
/// them — e.g. `if c then xs else ys` forces both branches to one type) being
/// pinned to two *distinct* buffers is the merge-over-distinct-descriptors
/// case: there's no single static binding, so it's a type error rather than a
/// silent wrong-buffer read.
fn pin_view_region(
    view_ty: &Type<TypeName>,
    binding: BindingRef,
    subst: &mut RegionSubst,
    span: Span,
) -> crate::error::Result<()> {
    if let Some(Type::Variable(id)) = view_ty.array_region() {
        let region = region_tag(binding);
        if let Some(existing) = subst.get(id) {
            if *existing != region {
                let prev = match existing {
                    Type::Constructed(TypeName::Region(b), _) => *b,
                    _ => unreachable!("region subst only holds Region tags"),
                };
                return Err(crate::err_type_at!(
                    span,
                    "this view merges two distinct storage buffers — \
                     region(set={}, binding={}) and region(set={}, binding={}); \
                     a view's descriptor must be a single compile-time constant, \
                     so `if … then … else …` (or any merge) across different \
                     buffers cannot be lowered",
                    prev.set,
                    prev.binding,
                    binding.set,
                    binding.binding
                ));
            }
        }
        subst.insert(*id, region);
    }
    Ok(())
}

fn collect_view_slice_region_subst(
    term: &Term,
    subst: &mut RegionSubst,
    region_env: &HashMap<SymbolId, Type<TypeName>>,
) -> crate::error::Result<bool> {
    if let TermKind::Let {
        name,
        name_ty,
        rhs,
        body,
    } = &term.kind
    {
        let mut changed = collect_view_slice_region_subst(rhs, subst, region_env)?;
        let mut scoped_env = region_env.clone();
        if let Some(region) = view_region_for_term(rhs, subst, region_env) {
            if let Some(Type::Variable(id)) = apply_subst(name_ty, subst).array_region() {
                if !subst.contains_key(id) {
                    subst.insert(*id, region.clone());
                    changed = true;
                }
            }
            scoped_env.insert(*name, region);
        }
        changed |= collect_view_slice_region_subst(body, subst, &scoped_env)?;
        return Ok(changed);
    }

    let mut changed = false;
    let mut err = None;
    term.for_each_child(
        &mut |child| match collect_view_slice_region_subst(child, subst, region_env) {
            Ok(child_changed) => changed |= child_changed,
            Err(e) => err = Some(e),
        },
    );
    if let Some(e) = err {
        return Err(e);
    }

    if let Some(source_region) = view_region_for_term(term, subst, region_env) {
        let result_ty = apply_subst(&term.ty, subst);
        if result_ty.array_variant().map(crate::types::is_array_variant_view).unwrap_or(false) {
            if let Some(Type::Variable(id)) = result_ty.array_region() {
                if !subst.contains_key(id) {
                    subst.insert(*id, source_region);
                    changed = true;
                }
            }
        }
    }
    Ok(changed)
}

fn view_region_for_term(
    term: &Term,
    subst: &RegionSubst,
    region_env: &HashMap<SymbolId, Type<TypeName>>,
) -> Option<Type<TypeName>> {
    if let Some(region @ Type::Constructed(TypeName::Region(_), _)) =
        apply_subst(&term.ty, subst).array_region().cloned()
    {
        return Some(region);
    }

    match &term.kind {
        TermKind::Var(VarRef::Symbol(sym)) => region_env.get(sym).cloned(),
        TermKind::App { func, args } => {
            let TermKind::Var(VarRef::Builtin { id, .. }) = &func.kind else {
                return None;
            };
            if *id == crate::builtins::catalog().known().slice && args.len() == 3 {
                view_region_for_term(&args[0], subst, region_env)
            } else {
                None
            }
        }
        _ => None,
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
        SoacOp::Scatter { lam, inputs, .. } => {
            subst_soac_body(lam, s);
            for input in inputs {
                subst_array_expr(input, s);
            }
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
        SoacOp::Screma {
            map_lams,
            accumulators,
            inputs,
            ..
        } => {
            for map_lam in map_lams {
                subst_soac_body(map_lam, s);
            }
            for acc in accumulators {
                subst_soac_body(&mut acc.step_lam, s);
                subst_soac_body(&mut acc.reduce_op, s);
                subst_term(&mut acc.ne, s);
            }
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
