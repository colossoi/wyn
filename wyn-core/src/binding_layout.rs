//! Compute-entry auto-storage binding allocation. Walks an entry's
//! params and decides which storage buffer each view-typed param
//! occupies — bindings 0..N in declaration order, one per view-array
//! param, N per tuple-of-views param (one per field).
//!
//! Producer side only; the layout is computed once at the start of
//! buffer specialization and cached on `EntryDecl::param_bindings` as
//! a `Vec<Option<...>>` aligned to the body params, so consumers can
//! iterate them in lockstep instead of joining by symbol.

use polytype::Type;

use crate::ast::{Pattern, PatternKind, TypeName};
use crate::interface::{Attribute, EntryDecl, EntryParamBinding, EntryParamBindingKind, TupleFieldBinding};
use crate::ssa::types::IoDecoration;
use crate::types::TypeExt;
use crate::{BindingRef, SymbolId};

/// Admits `ty` as a runtime-sized array entry-param shape and returns
/// the element type plus its static byte size. A runtime-sized array
/// has an unresolved polytype variable in size position (the normal
/// type-inference chain) or a `SizePlaceholder` (under `--fill-holes`);
/// sees through `Unique` (`*[]T` qualifies the same as `[]T`).
///
/// Admission additionally requires the element type to have a known
/// static byte size — i.e. anything except another runtime-sized
/// array. This rejects multi-rank runtime views (`[][]T`) at the
/// binding-allocation gate, matching the current spec (Stage 3
/// multi-dim is not yet shipped). Callers don't need a separate
/// `type_byte_size` retry — admission *is* the proof.
pub fn runtime_sized_array_elem(ty: &Type<TypeName>) -> Option<(&Type<TypeName>, u32)> {
    let ty = TypeExt::strip_unique(ty);
    let size = ty.array_size()?;
    if !matches!(
        size,
        Type::Variable(_) | Type::Constructed(TypeName::SizePlaceholder, _)
    ) {
        return None;
    }
    let elem = ty.elem_type()?;
    let elem_bytes = crate::ssa::layout::type_byte_size(elem)?;
    Some((elem, elem_bytes))
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
) -> Vec<Option<EntryParamBinding>> {
    if !matches!(entry.entry_type, Attribute::Compute) {
        return vec![None; body_params.len()];
    }

    let mut out: Vec<Option<EntryParamBinding>> = Vec::with_capacity(body_params.len());
    let mut binding_num: u32 = 0;

    for (i, (sym, ty)) in body_params.iter().enumerate() {
        let decoration = entry.params.get(i).and_then(extract_io_decoration);
        let has_builtin = matches!(decoration, Some(IoDecoration::BuiltIn(_)));

        // Explicit `#[storage(set, binding, access)]` on a `[]T` param
        // means the host wires it (e.g. the keyboard state). The auto-
        // allocator stays out of those slots so the binding number we
        // pick downstream agrees with the explicit one.
        if extract_storage_binding(&entry.params[i]).is_some() {
            out.push(None);
            continue;
        }

        // Uniqueness is an ownership marker; for binding allocation, `*[]T`
        // and `[]T` lower identically.
        let ty = TypeExt::strip_unique(ty);

        // Tuple-of-views: one slot per field. Each field's elem type +
        // byte size come from `runtime_sized_array_elem`, so the test
        // that admits the tuple is the same call that delivers both.
        if let Type::Constructed(TypeName::Tuple(_), field_tys) = ty {
            if !has_builtin && !field_tys.is_empty() {
                let field_elems: Option<Vec<(&Type<TypeName>, u32)>> =
                    field_tys.iter().map(runtime_sized_array_elem).collect();
                if let Some(field_elems) = field_elems {
                    let fields = field_elems
                        .into_iter()
                        .map(|(elem_ty, elem_bytes)| {
                            let slot = TupleFieldBinding {
                                binding: BindingRef::new(set, binding_num),
                                elem_ty: elem_ty.clone(),
                                elem_bytes,
                            };
                            binding_num += 1;
                            slot
                        })
                        .collect();
                    out.push(Some(EntryParamBinding {
                        param_sym: *sym,
                        kind: EntryParamBindingKind::TupleOfViews(fields),
                    }));
                    continue;
                }
            }
        }

        // Plain view-array. `has_builtin` is intentionally not gated:
        // a view-array param with a builtin decoration is malformed
        // (no builtin produces an array), but the allocator still
        // assigns a binding rather than silently routing to push
        // constants where the type wouldn't fit.
        if let Some((elem_ty, elem_bytes)) = runtime_sized_array_elem(ty) {
            out.push(Some(EntryParamBinding {
                param_sym: *sym,
                kind: EntryParamBindingKind::Single {
                    binding: BindingRef::new(set, binding_num),
                    elem_ty: elem_ty.clone(),
                    elem_bytes,
                },
            }));
            binding_num += 1;
        } else {
            out.push(None);
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
pub fn extract_uniform_binding(pattern: &Pattern) -> Option<BindingRef> {
    match &pattern.kind {
        PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                if let Attribute::Uniform { set, binding } = attr {
                    return Some(BindingRef::new(*set, *binding));
                }
            }
            extract_uniform_binding(inner)
        }
        PatternKind::Typed(inner, _) => extract_uniform_binding(inner),
        _ => None,
    }
}

/// Extract a `#[storage(set, binding, ...)]` from a param pattern.
pub fn extract_storage_binding(pattern: &Pattern) -> Option<BindingRef> {
    match &pattern.kind {
        PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                if let Attribute::Storage { set, binding, .. } = attr {
                    return Some(BindingRef::new(*set, *binding));
                }
            }
            extract_storage_binding(inner)
        }
        PatternKind::Typed(inner, _) => extract_storage_binding(inner),
        _ => None,
    }
}

/// Extract a `#[texture(set, binding)]` from a param pattern.
pub fn extract_texture_binding(pattern: &Pattern) -> Option<BindingRef> {
    match &pattern.kind {
        PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                if let Attribute::Texture { set, binding } = attr {
                    return Some(BindingRef::new(*set, *binding));
                }
            }
            extract_texture_binding(inner)
        }
        PatternKind::Typed(inner, _) => extract_texture_binding(inner),
        _ => None,
    }
}

/// Extract a `#[sampler(set, binding)]` from a param pattern.
pub fn extract_sampler_binding(pattern: &Pattern) -> Option<BindingRef> {
    match &pattern.kind {
        PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                if let Attribute::Sampler { set, binding } = attr {
                    return Some(BindingRef::new(*set, *binding));
                }
            }
            extract_sampler_binding(inner)
        }
        PatternKind::Typed(inner, _) => extract_sampler_binding(inner),
        _ => None,
    }
}

/// Extract a `#[storage_image(set, binding, format, access, size)]`
/// from a param pattern. Returns the binding ref plus the
/// format / access / size attributes — pinned at shader-compile time
/// and threaded into the descriptor + SPIR-V backend.
pub fn extract_storage_image_binding(
    pattern: &Pattern,
) -> Option<(
    BindingRef,
    crate::pipeline_descriptor::StorageImageFormat,
    crate::interface::StorageAccess,
    crate::pipeline_descriptor::StorageTextureSize,
)> {
    match &pattern.kind {
        PatternKind::Attributed(attrs, inner) => {
            for attr in attrs {
                if let Attribute::StorageImage {
                    set,
                    binding,
                    format,
                    access,
                    size,
                } = attr
                {
                    return Some((BindingRef::new(*set, *binding), *format, *access, *size));
                }
            }
            extract_storage_image_binding(inner)
        }
        PatternKind::Typed(inner, _) => extract_storage_image_binding(inner),
        _ => None,
    }
}
