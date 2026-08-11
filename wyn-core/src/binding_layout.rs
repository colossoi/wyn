//! Root-entry storage binding allocation. Walks an entry's
//! params and decides which storage buffer each view-typed param
//! occupies — bindings 0..N in declaration order, one per view-array
//! param, N per tuple-of-views param (one per field).
//!
//! Producer side only; the layout is computed once at the start of
//! buffer specialization and stored in the TLC entry payload as a
//! `Vec<Option<...>>` aligned to the body params, so consumers can
//! iterate them in lockstep instead of joining by symbol.

use polytype::Type;

use crate::ast::TypeName;
use crate::interface::{
    Attribute, EntryDecl, EntryParamBinding, EntryParamBindingKind, EntryParamDecl, IoDecoration,
    TupleFieldBinding,
};
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
    let ty = ty;
    let size = ty.array_size()?;
    if !matches!(
        size,
        Type::Variable(_) | Type::Constructed(TypeName::SizePlaceholder, _)
    ) {
        return None;
    }
    let elem = ty.elem_type()?;
    let elem_bytes = crate::ssa::layout::storage_elem_stride(elem)?;
    Some((elem, elem_bytes))
}

/// Arrays larger than the portable push-constant budget are represented by a
/// storage buffer even when every dimension is fixed. The outer element is
/// the buffer element, so multidimensional shape and row stride remain part
/// of the ordinary array type.
fn storage_array_elem(ty: &Type<TypeName>, consuming: bool) -> Option<(&Type<TypeName>, u32)> {
    if let Some(runtime) = runtime_sized_array_elem(ty) {
        return Some(runtime);
    }
    const PORTABLE_PUSH_CONSTANT_BYTES: u32 = 128;
    let elem = ty.elem_type()?;
    let total_bytes = crate::ssa::layout::type_byte_size(ty)?;
    (consuming || total_bytes > PORTABLE_PUSH_CONSTANT_BYTES)
        .then(|| crate::ssa::layout::storage_elem_stride(elem).map(|stride| (elem, stride)))?
}

/// Walk an entry's params and produce the auto-storage binding layout.
///
/// Runtime-sized arrays, fixed arrays larger than the portable push-constant
/// budget, and consuming array parameters can only be storage buffers. The
/// rules are the same whatever stage the entry runs at:
/// - Each admitted array param gets one slot.
/// - Each tuple whose fields are all admitted arrays gets one slot per field.
///   This is skipped for `#[builtin(...)]` params because builtins are not
///   user-supplied storage buffers.
/// - Other params (scalars, small non-consuming fixed arrays, structs) are
///   skipped and routed to push constants by the caller.
///
/// The binding a param receives is written back into its view type by
/// `pin_entry_buffers`, which is the only place a concrete
/// `Buffer(set, binding)` is born. A param left without one reaches the
/// backend as `Array[_, View, _, ?buffer]` and cannot be indexed.
///
/// Binding numbers come from `binding_ids` in declaration order.
pub fn compute_entry_binding_layout(
    body_params: &[(SymbolId, Type<TypeName>)],
    param_diets: &[crate::types::Diet],
    entry: &EntryDecl,
    set: u32,
    binding_ids: &mut crate::IdSource<u32>,
) -> Vec<Option<EntryParamBinding>> {
    let mut out: Vec<Option<EntryParamBinding>> = Vec::with_capacity(body_params.len());

    for (i, (sym, ty)) in body_params.iter().enumerate() {
        let consuming = param_diets.get(i).is_some_and(crate::types::Diet::is_consuming);
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
        let ty = ty;

        // Tuple-of-arrays: one slot per field. The admission test also
        // supplies the storage element type and stride.
        if let Type::Constructed(TypeName::Tuple(_), field_tys) = ty {
            if !has_builtin && !field_tys.is_empty() {
                let field_elems: Option<Vec<(&Type<TypeName>, u32)>> =
                    field_tys.iter().map(|field| storage_array_elem(field, consuming)).collect();
                if let Some(field_elems) = field_elems {
                    let fields = field_elems
                        .into_iter()
                        .map(|(elem_ty, elem_bytes)| TupleFieldBinding {
                            binding: BindingRef::new(set, binding_ids.next_id()),
                            elem_ty: elem_ty.clone(),
                            elem_bytes,
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

        // Plain admitted array. `has_builtin` is intentionally not gated:
        // an array param with a builtin decoration is malformed
        // (no builtin produces an array), but the allocator still
        // assigns a binding rather than silently routing to push
        // constants where the type wouldn't fit.
        if let Some((elem_ty, elem_bytes)) = storage_array_elem(ty, consuming) {
            out.push(Some(EntryParamBinding {
                param_sym: *sym,
                kind: EntryParamBindingKind::Single {
                    binding: BindingRef::new(set, binding_ids.next_id()),
                    elem_ty: elem_ty.clone(),
                    elem_bytes,
                },
            }));
        } else {
            out.push(None);
        }
    }

    out
}

/// Extract a `#[builtin(...)]`, `#[vertex_slot(N)]`, or `#[varying(N)]`
/// decoration from a param pattern. Recurses through `Attributed` / `Typed`
/// wrappers.
pub fn extract_io_decoration(param: &EntryParamDecl) -> Option<IoDecoration> {
    param.attributes.iter().find_map(|attribute| match attribute {
        Attribute::BuiltIn(builtin) => Some(IoDecoration::BuiltIn(*builtin)),
        Attribute::VertexSlot(n) | Attribute::Varying(n) => Some(IoDecoration::Location(*n)),
        _ => None,
    })
}

/// Extract a `#[uniform(set, binding)]` from a param pattern.
pub fn extract_uniform_binding(param: &EntryParamDecl) -> Option<BindingRef> {
    param.attributes.iter().find_map(|attribute| match attribute {
        Attribute::Uniform { set, binding } => Some(BindingRef::new(*set, *binding)),
        _ => None,
    })
}

/// Extract a `#[storage(set, binding, ...)]` from a param pattern.
pub fn extract_storage_binding(param: &EntryParamDecl) -> Option<BindingRef> {
    param.attributes.iter().find_map(|attribute| match attribute {
        Attribute::Storage { set, binding, .. } => Some(BindingRef::new(*set, *binding)),
        _ => None,
    })
}

/// Extract the declared `access` of a `#[storage(...)]` param (so the backend
/// knows whether the buffer is written in place, e.g. a `scatter` destination).
pub fn extract_storage_access(param: &EntryParamDecl) -> Option<crate::interface::StorageAccess> {
    param.attributes.iter().find_map(|attribute| match attribute {
        Attribute::Storage { access, .. } => Some(*access),
        _ => None,
    })
}

/// Extract a `#[texture(set, binding)]` from a param pattern.
pub fn extract_texture_binding(param: &EntryParamDecl) -> Option<BindingRef> {
    param.attributes.iter().find_map(|attribute| match attribute {
        Attribute::Texture { set, binding, .. } => Some(BindingRef::new(*set, *binding)),
        _ => None,
    })
}

/// Extract the backing storage-image binding of a texture param, if the
/// texture is a sampled view of a compiler-managed storage allocation
/// (a `resource`'s `sampled` view). `None` for host/external textures.
pub fn extract_texture_backing(param: &EntryParamDecl) -> Option<BindingRef> {
    param.attributes.iter().find_map(|attribute| match attribute {
        Attribute::Texture { backing, .. } => *backing,
        _ => None,
    })
}

/// The render-target `resource` name a `#[view(name, sampled)]` texture samples,
/// stamped by `resolve_resources`. `None` for a host texture or a storage view.
pub fn extract_texture_resource(param: &EntryParamDecl) -> Option<String> {
    param.attributes.iter().find_map(|attribute| match attribute {
        Attribute::Texture { resource, .. } => resource.clone(),
        _ => None,
    })
}

/// The `resource` name a `#[view(name, storage_read|storage_write)]` storage
/// image accesses, stamped by `resolve_resources`. `None` for a bare
/// `#[storage_image(...)]` param.
pub fn extract_storage_image_resource(param: &EntryParamDecl) -> Option<String> {
    param.attributes.iter().find_map(|attribute| match attribute {
        Attribute::StorageImage { resource, .. } => resource.clone(),
        _ => None,
    })
}

/// Extract a `#[sampler(set, binding)]` from a param pattern.
pub fn extract_sampler_binding(param: &EntryParamDecl) -> Option<BindingRef> {
    param.attributes.iter().find_map(|attribute| match attribute {
        Attribute::Sampler { set, binding } => Some(BindingRef::new(*set, *binding)),
        _ => None,
    })
}

/// Extract a `#[storage_image(set, binding, format, access, size)]`
/// from a param pattern. Returns the binding ref plus the
/// format / access / size attributes — pinned at shader-compile time
/// and threaded into the descriptor + SPIR-V backend.
pub fn extract_storage_image_binding(
    param: &EntryParamDecl,
) -> Option<(
    BindingRef,
    crate::pipeline_descriptor::StorageImageFormat,
    crate::interface::StorageAccess,
    crate::pipeline_descriptor::StorageTextureSize,
)> {
    param.attributes.iter().find_map(|attribute| match attribute {
        Attribute::StorageImage {
            set,
            binding,
            format,
            access,
            size,
            ..
        } => Some((BindingRef::new(*set, *binding), *format, *access, *size)),
        _ => None,
    })
}
