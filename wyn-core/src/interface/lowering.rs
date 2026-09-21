//! Lower source entry declarations into the shared shader interface.

use crate::ast::TypeName;
use crate::host::BufferLen;
use crate::interface::{
    Attribute, BindingExposure, EntryDecl, EntryInput, EntryInputKind, EntryOutput, EntryOutputDestination,
    EntryOutputKind, EntryParamDecl, IoDecoration, ResolvedAttribute,
};
use crate::ssa::layout::{storage_elem_stride, type_byte_size};
use crate::types::{array_size, canonical_storage_buffer_ty, strip_existentials, TypeExt};
use crate::BindingRef;
use polytype::Type;
use wyn_base::IdSource;

#[derive(Debug, thiserror::Error)]
#[error("internal compiler error: {0}")]
pub(crate) struct InterfaceError(pub(crate) String);

/// Descriptor set reserved for compiler-allocated storage. Compute
/// entry-input/output buffers (from SoA tuple splits), multi-stage SOAC
/// intermediates, and graphical-invariant prepass results all live on this
/// set. User-declared `#[uniform(...)]` and `#[storage(...)]` must use a
/// higher set (the parser enforces `set >= 1`). See SPECIFICATION.md
/// "Descriptor Set Layout" for the rationale.
pub const AUTO_STORAGE_SET: u32 = 0;

/// Extract a `#[size_hint(N)]` attribute from a lowered entry parameter.
pub fn extract_size_hint(param: &EntryParamDecl) -> Option<std::num::NonZeroU32> {
    param.attributes.iter().find_map(|attribute| match attribute {
        Attribute::SizeHint(n) => Some(*n),
        _ => None,
    })
}

/// Convert an AST attribute to an IO decoration.
fn convert_to_io_decoration(attr: &ResolvedAttribute) -> Option<IoDecoration> {
    match attr {
        Attribute::BuiltIn(b) => Some(IoDecoration::BuiltIn(*b)),
        Attribute::VertexSlot(n) | Attribute::Varying(n) => Some(IoDecoration::Location(*n)),
        _ => None,
    }
}

/// The render-target resource name of a `#[target(name)]` output attribute.
fn target_of(attr: Option<&ResolvedAttribute>) -> Option<String> {
    match attr {
        Some(Attribute::Target(name)) => Some(name.clone()),
        _ => None,
    }
}

/// Explicit storage binding requested by an output attribute.
fn storage_output_binding(attr: Option<&ResolvedAttribute>) -> Option<BindingRef> {
    match attr {
        Some(Attribute::Storage { set, binding, .. }) => Some(BindingRef::new(*set, *binding)),
        _ => None,
    }
}

pub(crate) fn is_storage_image_ty(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::StorageTexture, _))
}

pub(crate) fn entry_output_arity(entry: &EntryDecl, ret_type: &Type<TypeName>) -> usize {
    match strip_existentials(ret_type) {
        Type::Constructed(TypeName::Unit | TypeName::SideEffect | TypeName::StorageTexture, _) => 0,
        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => fields.len(),
        _ => usize::from(!entry.outputs.is_empty()),
    }
}

pub(crate) fn build_entry_outputs(
    entry: &EntryDecl,
    ret_type: &Type<TypeName>,
    slot_value_tys: &[Option<Type<TypeName>>],
    inputs: &[EntryInput],
    is_compute: bool,
    binding_ids: &mut IdSource<u32>,
) -> Result<Vec<EntryOutput>, InterfaceError> {
    let logical_ret_type = strip_existentials(ret_type);
    let output_arity = entry_output_arity(entry, ret_type);
    // Pick a `BufferLen` policy for the output binding, in order:
    //
    //   1. Output type carries a compile-time-known `Size(n)` literal
    //      → `Fixed { bytes: n * elem_bytes }`.
    //   2. Output's size variable matches one of the entry's storage
    //      inputs (the type checker has unified them) → `LikeInput`
    //      tracking that input.
    //   3. A runtime array output route is sized from the finalized semantic
    //      dispatch domain → `SameAsDispatch { elem_bytes }`.
    //   4. None — the host falls back to its default sizing or, if it
    //      tried to allocate this buffer, surfaces a clean error.
    //
    // The size info is already in the (post-monomorphize) type — we
    // just read it. No structural rewrites needed for `if/else`
    // branches whose result types have already been unified.
    let length_for =
        |binding: Option<BindingRef>, ty: &Type<TypeName>| -> Result<Option<BufferLen>, InterfaceError> {
            if binding.is_none() {
                return Ok(None);
            }
            let Some(elem_ty) = ty.elem_type() else {
                let Some(bytes) = type_byte_size(ty) else {
                    return Err(InterfaceError(format!(
                        "output has no static byte layout: {ty:?}"
                    )));
                };
                return Ok(Some(BufferLen::Fixed {
                    bytes: u64::from(bytes),
                }));
            };
            let Some(elem_bytes) = storage_elem_stride(elem_ty) else {
                return Err(InterfaceError(format!(
                    "output element has no static byte layout: {elem_ty:?}"
                )));
            };
            if let Some(out_size) = array_size(ty) {
                // Rule 1: compile-time size literal.
                if let Type::Constructed(TypeName::Size(n), _) = out_size {
                    return Ok(Some(BufferLen::Fixed {
                        bytes: (*n as u64) * elem_bytes as u64,
                    }));
                }
                // Rule 2: size variable shared with an entry input.
                for input in inputs {
                    let EntryInputKind::Storage {
                        exposure: BindingExposure::Host(in_binding),
                        ..
                    } = &input.kind
                    else {
                        continue;
                    };
                    let Some(in_size) = array_size(&input.ty) else {
                        continue;
                    };
                    if in_size == out_size {
                        let Some(in_elem_ty) = input.ty.elem_type() else {
                            continue;
                        };
                        let Some(src_elem_bytes) = storage_elem_stride(in_elem_ty) else {
                            return Err(InterfaceError(format!(
                                "input element has no static byte layout: {in_elem_ty:?}"
                            )));
                        };
                        return Ok(Some(BufferLen::LikeInput {
                            set: in_binding.set,
                            binding: in_binding.binding,
                            elem_bytes,
                            src_elem_bytes,
                        }));
                    }
                }
            }
            // Rule 3: dynamic arrays without a fixed or matching-input size
            // are sized from the finalized semantic dispatch domain.
            if ty.is_array() {
                return Ok(Some(BufferLen::SameAsDispatch { elem_bytes }));
            }
            Ok(None)
        };
    let mut storage_binding_for = |ty: &Type<TypeName>,
                                   is_compute: bool,
                                   attribute: Option<&ResolvedAttribute>|
     -> Option<BindingRef> {
        if is_compute && !matches!(ty, Type::Constructed(TypeName::Unit, _)) {
            storage_output_binding(attribute)
                .or_else(|| Some(BindingRef::new(AUTO_STORAGE_SET, binding_ids.next_id())))
        } else {
            None
        }
    };

    // Prefer the converted route value's representation-specialized type to
    // the parse-time output declaration. A source entry with no return value
    // has no logical output slot. Returning a synthetic Unit-typed
    // `EntryOutput` here would surface to the SPIR-V backend as an
    // `Output<void>` variable in the entry's interface — malformed and
    // rejected by naga / the Vulkan validation layer.
    if is_storage_image_ty(ret_type) || entry.outputs.iter().any(|output| is_storage_image_ty(&output.ty)) {
        return Ok(vec![]);
    }

    if entry.outputs.is_empty()
        && matches!(
            ret_type,
            Type::Constructed(TypeName::Unit | TypeName::SideEffect, _)
        )
    {
        return Ok(vec![]);
    }

    if entry.outputs.iter().all(|o| o.attribute.is_none()) && output_arity == 1 {
        if !matches!(ret_type, Type::Constructed(TypeName::Unit, _)) {
            let source_ty = slot_value_tys.first().and_then(Option::as_ref).unwrap_or(ret_type);
            let ty = canonical_storage_buffer_ty(source_ty);
            let attribute = entry.outputs.first().and_then(|output| output.attribute.as_ref());
            let storage_binding = storage_binding_for(&ty, is_compute, attribute);
            let length = length_for(storage_binding, &ty)?;
            Ok(vec![EntryOutput {
                ty,
                kind: entry_output_kind(storage_binding, length, None, None),
            }])
        } else {
            Ok(vec![])
        }
    } else if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), component_types) =
        logical_ret_type
    {
        component_types
            .iter()
            .enumerate()
            .map(|(slot, ty)| {
                let ty = slot_value_tys.get(slot).and_then(Option::as_ref).unwrap_or(ty);
                let ty = canonical_storage_buffer_ty(ty);
                let attribute = entry.outputs.get(slot).and_then(|output| output.attribute.as_ref());
                let storage_binding = storage_binding_for(&ty, is_compute, attribute);
                let length = length_for(storage_binding, &ty)?;
                Ok(EntryOutput {
                    ty,
                    kind: entry_output_kind(
                        storage_binding,
                        length,
                        attribute.and_then(convert_to_io_decoration),
                        target_of(attribute),
                    ),
                })
            })
            .collect()
    } else {
        let source_ty = slot_value_tys.first().and_then(Option::as_ref).unwrap_or(ret_type);
        let ty = canonical_storage_buffer_ty(source_ty);
        let first_attr = entry.outputs.first().and_then(|o| o.attribute.as_ref());
        let storage_binding = storage_binding_for(&ty, is_compute, first_attr);
        let length = length_for(storage_binding, &ty)?;
        Ok(vec![EntryOutput {
            ty,
            kind: entry_output_kind(
                storage_binding,
                length,
                first_attr.and_then(convert_to_io_decoration),
                target_of(first_attr),
            ),
        }])
    }
}

fn entry_output_kind(
    storage_binding: Option<BindingRef>,
    length: Option<BufferLen>,
    decoration: Option<IoDecoration>,
    target: Option<String>,
) -> EntryOutputKind {
    if let Some(binding) = storage_binding {
        return EntryOutputKind::Storage {
            exposure: BindingExposure::Host(binding),
            length,
        };
    }
    let destination = match (decoration, target) {
        (Some(IoDecoration::BuiltIn(builtin)), None) => EntryOutputDestination::BuiltIn(builtin),
        (Some(IoDecoration::Location(location)), None) => EntryOutputDestination::Location(location),
        (None, Some(target)) => EntryOutputDestination::Target(target),
        (None, None) => EntryOutputDestination::Plain,
        (Some(_), Some(_)) => unreachable!("entry output cannot have both a decoration and target"),
    };
    EntryOutputKind::Value { destination }
}
