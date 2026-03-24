//! Type size and alignment helpers for buffer layout calculations.

use crate::ast::{Type, TypeName};
use crate::types::TypeExt;

/// Calculate the byte size of a type for buffer layout purposes.
/// Returns None for types that don't have a fixed size (e.g., runtime arrays).
pub fn type_byte_size(ty: &Type) -> Option<u32> {
    match ty {
        Type::Constructed(TypeName::Int(bits), _) => Some((*bits / 8) as u32),
        Type::Constructed(TypeName::UInt(bits), _) => Some((*bits / 8) as u32),
        Type::Constructed(TypeName::Float(bits), _) => Some((*bits / 8) as u32),
        Type::Constructed(TypeName::Unit, _) => Some(0),
        _ if ty.is_vec() => {
            let size = ty.vec_size()? as u32;
            let elem_size = type_byte_size(ty.elem_type().expect("Vec has elem"))?;
            Some(size * elem_size)
        }
        _ if ty.is_mat() => {
            let cols = ty.mat_cols()? as u32;
            let rows = ty.mat_rows()? as u32;
            let elem_size = type_byte_size(ty.elem_type().expect("Mat has elem"))?;
            Some(cols * rows * elem_size)
        }
        _ if ty.is_array() => {
            let elem_size = type_byte_size(ty.elem_type().expect("Array has elem"))?;
            let size = match ty.array_size().expect("Array has size") {
                Type::Constructed(TypeName::Size(n), _) => *n as u32,
                _ => return None,
            };
            Some(size * elem_size)
        }
        Type::Constructed(TypeName::Tuple(arity), args) => {
            let mut total = 0u32;
            for i in 0..*arity {
                total += type_byte_size(&args[i])?;
            }
            Some(total)
        }
        _ => None,
    }
}

/// Calculate std430 alignment for a type.
/// Returns None for types that don't have a fixed alignment.
pub fn std430_alignment(ty: &Type) -> Option<u32> {
    match ty {
        Type::Constructed(TypeName::Int(bits), _) => Some((*bits / 8) as u32),
        Type::Constructed(TypeName::UInt(bits), _) => Some((*bits / 8) as u32),
        Type::Constructed(TypeName::Float(bits), _) => Some((*bits / 8) as u32),
        Type::Constructed(TypeName::Unit, _) => Some(1),
        _ if ty.is_vec() => {
            let size = ty.vec_size()? as u32;
            let elem_align = std430_alignment(ty.elem_type().expect("Vec has elem"))?;
            Some(if size == 2 { 2 * elem_align } else { 4 * elem_align })
        }
        _ if ty.is_mat() => {
            let rows = ty.mat_rows()? as u32;
            let elem_align = std430_alignment(ty.elem_type().expect("Mat has elem"))?;
            Some(if rows == 2 { 2 * elem_align } else { 4 * elem_align })
        }
        _ if ty.is_array() => std430_alignment(ty.elem_type().expect("Array has elem")),
        _ => None,
    }
}

/// Calculate std140 alignment for a type.
/// std140 has stricter alignment requirements than std430.
pub fn std140_alignment(ty: &Type) -> Option<u32> {
    match ty {
        Type::Constructed(TypeName::Int(bits), _) => Some((*bits / 8) as u32),
        Type::Constructed(TypeName::UInt(bits), _) => Some((*bits / 8) as u32),
        Type::Constructed(TypeName::Float(bits), _) => Some((*bits / 8) as u32),
        Type::Constructed(TypeName::Unit, _) => Some(1),
        _ if ty.is_vec() => {
            let size = ty.vec_size()? as u32;
            let elem_align = std140_alignment(ty.elem_type().expect("Vec has elem"))?;
            Some(if size == 2 { 2 * elem_align } else { 4 * elem_align })
        }
        _ if ty.is_mat() => {
            let elem_align = std140_alignment(ty.elem_type().expect("Mat has elem"))?;
            Some(4 * elem_align) // Always vec4 alignment in std140
        }
        _ if ty.is_array() => {
            let elem_align = std140_alignment(ty.elem_type().expect("Array has elem"))?;
            Some(((elem_align + 15) / 16) * 16)
        }
        _ => None,
    }
}

/// Compute the ArrayStride values needed for nested fixed-size arrays
/// when this type is used as a storage buffer element.
/// Returns one stride per array nesting level (outermost first).
/// E.g. `[8]u32` → [4], `[4][8]u32` → [32, 4], `u32` → [].
pub fn buffer_array_strides(ty: &Type) -> Vec<u32> {
    let mut strides = Vec::new();
    collect_array_strides(ty, &mut strides);
    strides
}

fn collect_array_strides(ty: &Type, out: &mut Vec<u32>) {
    if ty.is_array() {
        let variant = ty.array_variant().expect("Array has variant");
        let is_composite = matches!(variant, Type::Constructed(TypeName::ArrayVariantComposite, _));
        if is_composite {
            if let Type::Constructed(TypeName::Size(_), _) = ty.array_size().expect("Array has size") {
                let elem = ty.elem_type().expect("Array has elem");
                if let Some(elem_size) = type_byte_size(elem) {
                    out.push(elem_size);
                }
                collect_array_strides(elem, out);
            }
        }
    }
}
