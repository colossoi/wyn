//! Compute-entry auto-storage binding allocation. Walks an entry's
//! params and decides which storage buffer each view-typed param
//! occupies — bindings 0..N in declaration order, one per view-array
//! param, N per tuple-of-views param (one per field).
//!
//! Producer side only; the layout is computed once at the start of
//! buffer specialization and cached on `EntryDecl::param_bindings`.
//! Consumers read the cached vector directly.

use polytype::Type;

use crate::SymbolId;
use crate::ast::{Pattern, PatternKind, TypeName};
use crate::interface::{Attribute, EntryBindingSlot, EntryDecl};
use crate::ssa::types::IoDecoration;
use crate::types::TypeExt;

/// Runtime-sized array: either an unresolved polytype variable in
/// size position (the normal type-inference chain) or a
/// `SizePlaceholder` (when `--fill-holes` substituted a hole there).
fn is_runtime_sized_array(ty: &Type<TypeName>) -> bool {
    ty.array_size()
        .map(|s| {
            matches!(
                s,
                Type::Variable(_) | Type::Constructed(TypeName::SizePlaceholder, _)
            )
        })
        .unwrap_or(false)
}

/// Walk a compute entry's params and produce the auto-storage binding
/// layout. Non-compute entries return an empty layout (graphics
/// pipeline params live in vertex attribs / fragment varyings, not
/// storage bindings).
///
/// Walk rules (compute entries only):
/// - Each view-array param (`[]T`, runtime size) gets one slot.
/// - Each tuple-of-views param gets one slot per field. Skipped if
///   the param is decorated `#[builtin(...)]` — builtins are not
///   user-supplied storage buffers.
/// - Other params (scalars, fixed-size arrays, structs) are skipped
///   and will be routed to push constants by the caller.
///
/// Binding numbers are assigned `set, 0..N` in declaration order.
pub fn compute_entry_binding_layout(
    body_params: &[(SymbolId, Type<TypeName>)],
    entry: &EntryDecl,
    set: u32,
) -> Vec<EntryBindingSlot> {
    if !matches!(entry.entry_type, Attribute::Compute) {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut binding_num: u32 = 0;

    for (i, (sym, ty)) in body_params.iter().enumerate() {
        let decoration = entry.params.get(i).and_then(extract_io_decoration);
        let has_builtin = matches!(decoration, Some(IoDecoration::BuiltIn(_)));

        // Tuple-of-views: one slot per field.
        if let Type::Constructed(TypeName::Tuple(_), field_tys) = ty {
            if !has_builtin && !field_tys.is_empty() && field_tys.iter().all(is_runtime_sized_array) {
                for (field_idx, field_ty) in field_tys.iter().enumerate() {
                    out.push(EntryBindingSlot {
                        set,
                        binding: binding_num,
                        param_sym: *sym,
                        tuple_field: Some(field_idx),
                        elem_ty: field_ty.elem_type().cloned().unwrap_or(Type::Variable(0)),
                    });
                    binding_num += 1;
                }
                continue;
            }
        }

        // Plain view-array. `has_builtin` is intentionally not gated:
        // a view-array param with a builtin decoration is malformed
        // (no builtin produces an array), but the allocator still
        // assigns a binding rather than silently routing to push
        // constants where the type wouldn't fit.
        if is_runtime_sized_array(ty) {
            out.push(EntryBindingSlot {
                set,
                binding: binding_num,
                param_sym: *sym,
                tuple_field: None,
                elem_ty: ty.elem_type().cloned().unwrap_or(Type::Variable(0)),
            });
            binding_num += 1;
        }
    }

    out
}

/// Extract a `#[builtin(...)]` or `#[location(N)]` decoration from a
/// param pattern. Recurses through `Attributed` / `Typed` wrappers.
pub fn extract_io_decoration(pattern: &Pattern) -> Option<IoDecoration> {
    match &pattern.kind {
        PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                match attr {
                    Attribute::BuiltIn(builtin) => {
                        return Some(IoDecoration::BuiltIn(*builtin));
                    }
                    Attribute::Location(loc) => {
                        return Some(IoDecoration::Location(*loc));
                    }
                    _ => {}
                }
            }
            extract_io_decoration(inner)
        }
        PatternKind::Typed(inner, _) => extract_io_decoration(inner),
        _ => None,
    }
}

/// Extract a `#[uniform(set, binding)]` from a param pattern.
pub fn extract_uniform_binding(pattern: &Pattern) -> Option<(u32, u32)> {
    match &pattern.kind {
        PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                if let Attribute::Uniform { set, binding } = attr {
                    return Some((*set, *binding));
                }
            }
            extract_uniform_binding(inner)
        }
        PatternKind::Typed(inner, _) => extract_uniform_binding(inner),
        _ => None,
    }
}

/// Extract a `#[storage(set, binding, ...)]` from a param pattern.
pub fn extract_storage_binding(pattern: &Pattern) -> Option<(u32, u32)> {
    match &pattern.kind {
        PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                if let Attribute::Storage { set, binding, .. } = attr {
                    return Some((*set, *binding));
                }
            }
            extract_storage_binding(inner)
        }
        PatternKind::Typed(inner, _) => extract_storage_binding(inner),
        _ => None,
    }
}
