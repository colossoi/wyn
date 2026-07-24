//! Pin each compute entry's storage-param buffer region into its type.
//!
//! After type inference a view-array entry param `xs: []f32` has an
//! unresolved region *variable* in its type's buffer slot — the checker
//! cannot know which descriptor `xs` reads, because the source never says
//! so. The binding is decided separately, by `compute_entry_binding_layout`
//! (auto-allocated `set, 0..N` in declaration order) or by an explicit
//! `#[storage(set, binding)]` attribute. This pass takes that decided
//! binding and writes it into the type: it substitutes the param's region
//! variable → `Buffer(set, binding)` throughout the entry body.
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
#[path = "pin_entry_buffers_tests.rs"]
mod pin_entry_buffers_tests;

use super::data::Empty;
use super::run::Transformed;
use super::{apply_type_substitution, extract_lambda_params_ref, DefMeta, Program, Term, TermKind, VarRef};
use crate::ast::{Span, TypeName};
use crate::binding_layout::{
    compute_entry_binding_layout, extract_storage_binding, extract_storage_image_binding,
};
use crate::interface::{EntryDecl, EntryParamBindingKind};
use crate::types::{buffer_tag, TypeExt};
use crate::{BindingRef, LookupMap, SymbolId};
use polytype::Type;

/// TLC after entry-parameter buffer regions are pinned into their types.
#[derive(Debug, Clone, Copy, Default)]
pub struct BuffersPinned;

impl super::Stage for BuffersPinned {
    type Family = super::run::Polymorphic;
    type GlobalContext = super::context::RewriteGlobal;
}

/// Buffer-variable → concrete `Buffer(set, binding)` substitution.
type BufferSubst = LookupMap<usize, Type<TypeName>>;

pub fn run(mut program: Program<Transformed>) -> crate::error::Result<Program<BuffersPinned>> {
    let (defs, term_ids, global_context) = (
        &mut program.defs,
        &mut program.term_ids,
        &mut program.global_context,
    );
    for def in defs {
        if !matches!(&def.meta, DefMeta::EntryPoint(_)) {
            continue;
        }
        let span = def.body.span;
        let (_, params) = extract_lambda_params_ref(&def.body);

        let mut subst = BufferSubst::new();
        let mut buffer_env = LookupMap::new();
        if let DefMeta::EntryPoint(entry) = &mut def.meta {
            let param_bindings = compute_entry_binding_layout(
                &params,
                &entry.declaration,
                crate::egir::from_tlc::AUTO_STORAGE_SET,
                &mut global_context.auto_storage_binding_ids,
            );
            entry.declaration.param_bindings = param_bindings;
            collect_buffer_subst(&params, &entry.declaration, &mut subst, &mut buffer_env, span)?;
        }

        while collect_view_slice_buffer_subst(&def.body, &mut subst, &buffer_env)? {}

        if !subst.is_empty() {
            def.ty = apply_type_substitution(&def.ty, &subst);
            def.body.rewrite_types(term_ids, &mut |ty| apply_type_substitution(ty, &subst));
        }
    }
    Ok(
        program.map_global_context(|global| super::context::RewriteGlobal {
            known_defs: global.known_defs,
            auto_storage_binding_ids: global.auto_storage_binding_ids,
        }),
    )
}

/// Map each storage param's buffer variable to its concrete region. An
/// explicit `#[storage(set, binding)]` wins; otherwise the auto-allocated
/// layout decides. Non-view params contribute nothing (their type has no
/// buffer slot variable to pin).
fn collect_buffer_subst(
    params: &[(SymbolId, Type<TypeName>)],
    entry: &EntryDecl,
    subst: &mut BufferSubst,
    buffer_env: &mut LookupMap<SymbolId, Type<TypeName>>,
    span: Span,
) -> crate::error::Result<()> {
    let layout = &entry.param_bindings;

    for (i, (_sym, ty)) in params.iter().enumerate() {
        let ty = ty;

        if let Some((binding, ..)) = entry.params.get(i).and_then(extract_storage_image_binding) {
            pin_resource_buffer(ty, binding, subst, span)?;
            buffer_env.insert(params[i].0, buffer_tag(binding));
            continue;
        }

        // Host-wired explicit binding: the region is the attribute's.
        if let Some(binding) = entry.params.get(i).and_then(extract_storage_binding) {
            pin_resource_buffer(ty, binding, subst, span)?;
            buffer_env.insert(params[i].0, buffer_tag(binding));
            continue;
        }

        match layout.get(i).and_then(|b| b.as_ref()).map(|b| &b.kind) {
            Some(EntryParamBindingKind::Single { binding, .. }) => {
                pin_resource_buffer(ty, *binding, subst, span)?;
                buffer_env.insert(params[i].0, buffer_tag(*binding));
            }
            Some(EntryParamBindingKind::TupleOfViews(fields)) => {
                if let Type::Constructed(TypeName::Tuple(_), comps) = ty {
                    for (field, comp_ty) in fields.iter().zip(comps) {
                        pin_resource_buffer(comp_ty, field.binding, subst, span)?;
                    }
                }
            }
            None => {}
        }
    }
    Ok(())
}

/// If `view_ty`'s buffer slot is a free variable, record `var → Region(binding)`.
/// A region already concrete (or absent — a non-array) is left untouched.
///
/// A buffer variable shared by two params (because type inference unified
/// them — e.g. `if c then xs else ys` forces both branches to one type) being
/// pinned to two *distinct* buffers is the merge-over-distinct-descriptors
/// case: there's no single static binding, so it's a type error rather than a
/// silent wrong-buffer read.
fn pin_resource_buffer(
    view_ty: &Type<TypeName>,
    binding: BindingRef,
    subst: &mut BufferSubst,
    span: Span,
) -> crate::error::Result<()> {
    let buffer_slot = match view_ty {
        Type::Constructed(TypeName::StorageTexture, args) => args.first(),
        _ => view_ty.array_buffer(),
    };
    if let Some(Type::Variable(id)) = buffer_slot {
        let region = buffer_tag(binding);
        if let Some(existing) = subst.get(id) {
            if *existing != region {
                let prev = match existing {
                    Type::Constructed(TypeName::Buffer(b), _) => *b,
                    _ => unreachable!("region subst only holds Region tags"),
                };
                return Err(crate::err_type_at!(
                    span,
                    "this resource merges two distinct descriptor bindings — \
                     region(set={}, binding={}) and region(set={}, binding={}); \
                     a view's descriptor must be a single compile-time constant, \
                     so `if … then … else …` (or any merge) across different \
                     resources cannot be lowered",
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

fn collect_view_slice_buffer_subst(
    term: &Term<Empty, Empty>,
    subst: &mut BufferSubst,
    buffer_env: &LookupMap<SymbolId, Type<TypeName>>,
) -> crate::error::Result<bool> {
    if let TermKind::Let {
        name,
        name_ty,
        rhs,
        body,
    } = &term.kind
    {
        let mut changed = collect_view_slice_buffer_subst(rhs, subst, buffer_env)?;
        let mut scoped_env = buffer_env.clone();
        if let Some(region) = view_buffer_for_term(rhs, subst, buffer_env) {
            if let Some(Type::Variable(id)) = apply_type_substitution(name_ty, subst).array_buffer() {
                if !subst.contains_key(id) {
                    subst.insert(*id, region.clone());
                    changed = true;
                }
            }
            scoped_env.insert(*name, region);
        }
        changed |= collect_view_slice_buffer_subst(body, subst, &scoped_env)?;
        return Ok(changed);
    }

    let mut changed = false;
    let mut err = None;
    term.for_each_child(
        &mut |child| match collect_view_slice_buffer_subst(child, subst, buffer_env) {
            Ok(child_changed) => changed |= child_changed,
            Err(e) => err = Some(e),
        },
    );
    if let Some(e) = err {
        return Err(e);
    }

    if let Some(source_buffer) = view_buffer_for_term(term, subst, buffer_env) {
        let result_ty = apply_type_substitution(&term.ty, subst);
        if result_ty.array_variant().map(crate::types::is_array_variant_view).unwrap_or(false) {
            if let Some(Type::Variable(id)) = result_ty.array_buffer() {
                if !subst.contains_key(id) {
                    subst.insert(*id, source_buffer);
                    changed = true;
                }
            }
        }
    }
    Ok(changed)
}

fn view_buffer_for_term(
    term: &Term<Empty, Empty>,
    subst: &BufferSubst,
    buffer_env: &LookupMap<SymbolId, Type<TypeName>>,
) -> Option<Type<TypeName>> {
    if let Some(region @ Type::Constructed(TypeName::Buffer(_), _)) =
        apply_type_substitution(&term.ty, subst).array_buffer().cloned()
    {
        return Some(region);
    }

    match &term.kind {
        TermKind::Var(VarRef::Symbol(sym)) => buffer_env.get(sym).cloned(),
        TermKind::App { func, args } => {
            let TermKind::Var(VarRef::Builtin { id, .. }) = &func.kind else {
                return None;
            };
            if *id == crate::builtins::catalog().known().slice && args.len() == 3 {
                view_buffer_for_term(&args[0], subst, buffer_env)
            } else {
                None
            }
        }
        _ => None,
    }
}
