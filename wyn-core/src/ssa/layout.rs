//! Type size and alignment helpers for buffer layout calculations.

use crate::ast::{Type, TypeName};
use crate::interface::StorageLayout;
use crate::types::TypeExt;
use wyn_pipeline_descriptor::VertexFormat;

#[cfg(test)]
#[path = "layout_tests.rs"]
mod layout_tests;

/// Map a Wyn type to its vertex-buffer attribute format, for
/// `#[vertex_slot(n)]` vertex-shader input parameters. Only 32-bit
/// float / signed / unsigned scalars and 2-4 wide vectors of them are
/// valid vertex formats; everything else (arrays, tuples, matrices,
/// non-32-bit scalars, `vec1`) returns `None`. This is the single
/// source of truth keeping the SPIR-V `Input` variable's type and the
/// pipeline descriptor's `VertexFormat` consistent.
pub fn vertex_format(ty: &Type) -> Option<VertexFormat> {
    use VertexFormat::*;

    // Component scalar class: 0 = f32, 1 = i32, 2 = u32. None for
    // anything else (f16/f64, i8/i16/i64, bool, ...).
    fn scalar_class(t: &Type) -> Option<u8> {
        match t {
            Type::Constructed(TypeName::Float(32), _) => Some(0),
            Type::Constructed(TypeName::Int(32), _) => Some(1),
            Type::Constructed(TypeName::UInt(32), _) => Some(2),
            _ => None,
        }
    }

    if let Some(class) = scalar_class(ty) {
        return Some(match class {
            0 => Float32,
            1 => Sint32,
            _ => Uint32,
        });
    }
    if ty.is_vec() {
        let n = ty.vec_size()?;
        let class = scalar_class(ty.elem_type()?)?;
        return match (class, n) {
            (0, 2) => Some(Float32x2),
            (0, 3) => Some(Float32x3),
            (0, 4) => Some(Float32x4),
            (1, 2) => Some(Sint32x2),
            (1, 3) => Some(Sint32x3),
            (1, 4) => Some(Sint32x4),
            (2, 2) => Some(Uint32x2),
            (2, 3) => Some(Uint32x3),
            (2, 4) => Some(Uint32x4),
            _ => None,
        };
    }
    None
}

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
        Type::Constructed(TypeName::Record(_), args) => {
            let mut total = 0u32;
            for arg in args {
                total += type_byte_size(arg)?;
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
        // A struct's std430 alignment is its largest member alignment.
        Type::Constructed(TypeName::Tuple(_), args) | Type::Constructed(TypeName::Record(_), args) => {
            let mut align = 1u32;
            for arg in args {
                align = align.max(std430_alignment(arg)?);
            }
            Some(align)
        }
        _ => None,
    }
}

/// Byte stride between adjacent columns of a column-major matrix in a
/// std430 interface block. SPIR-V requires this value to be published as a
/// `MatrixStride` member decoration whenever a block contains a matrix (or an
/// array of matrices).
pub fn std430_matrix_stride(ty: &Type) -> Option<u32> {
    if !ty.is_mat() {
        return None;
    }
    let rows = ty.mat_rows()? as u32;
    let elem = ty.elem_type()?;
    let elem_size = type_byte_size(elem)?;
    let column_size = rows * elem_size;
    let column_alignment =
        if rows == 2 { 2 * std430_alignment(elem)? } else { 4 * std430_alignment(elem)? };
    Some(column_size.div_ceil(column_alignment) * column_alignment)
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

/// Block layout rule set for interface blocks (uniform / storage).
/// For the member shapes `block_layout` supports (32-bit scalars and
/// vectors of them) the two sets produce identical member offsets;
/// they differ only in the final size rounding — std140 rounds the
/// block size up to 16.
/// Computed layout for an interface block value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockLayout {
    /// Total block size in bytes, rounded to the block alignment (and
    /// to 16 under std140).
    pub size: u32,
    /// The block's alignment: the largest member alignment.
    pub align: u32,
    /// Byte offset of each member, in declaration order. A bare
    /// scalar/vector is a single member at offset 0.
    pub member_offsets: Vec<u32>,
}

/// Layout for an interface-block value: a 32-bit `f32`/`i32`/`u32`
/// scalar, a vec2/3/4 of them, or a FLAT record/tuple of those. Under
/// Std430 (storage buffers) a member may also be a fixed-size array of
/// supported scalars/vectors (the SOAC passes synthesize tuple
/// elements like `(u32, [4]u32)`); std140's array rules (16-rounded
/// strides) are not implemented, so arrays stay unsupported for
/// uniforms. Returns `None` for anything else (bool, matrices, nested
/// aggregates, runtime arrays, non-32-bit scalars) — callers gate
/// support on this.
pub fn block_layout(ty: &Type, rules: StorageLayout) -> Option<BlockLayout> {
    // (size, alignment) of one supported member.
    fn member(ty: &Type, rules: StorageLayout) -> Option<(u32, u32)> {
        match ty {
            Type::Constructed(TypeName::Int(32), _)
            | Type::Constructed(TypeName::UInt(32), _)
            | Type::Constructed(TypeName::Float(32), _) => Some((4, 4)),
            _ if ty.is_vec() => {
                // Element must itself be a supported 32-bit scalar.
                member(ty.elem_type()?, rules)?;
                let n = ty.vec_size()? as u32;
                if !(2..=4).contains(&n) {
                    return None;
                }
                Some((4 * n, if n == 2 { 8 } else { 16 }))
            }
            _ if ty.is_array() && rules == StorageLayout::Std430 => {
                let (elem_size, elem_align) = member(ty.elem_type()?, rules)?;
                let n = match ty.array_size()? {
                    Type::Constructed(TypeName::Size(n), _) => *n as u32,
                    _ => return None,
                };
                let elem_stride = elem_size.div_ceil(elem_align) * elem_align;
                Some((n * elem_stride, elem_align))
            }
            _ => None,
        }
    }

    let fields: Vec<&Type> = match ty {
        Type::Constructed(TypeName::Tuple(_), args) | Type::Constructed(TypeName::Record(_), args) => {
            args.iter().collect()
        }
        other => vec![other],
    };
    if fields.is_empty() {
        return None;
    }

    let mut member_offsets = Vec::with_capacity(fields.len());
    let mut offset = 0u32;
    let mut align = 4u32;
    for field in fields {
        let (field_size, field_align) = member(field, rules)?;
        offset = offset.div_ceil(field_align) * field_align;
        member_offsets.push(offset);
        offset += field_size;
        align = align.max(field_align);
    }
    let mut size = offset.div_ceil(align) * align;
    if rules == StorageLayout::Std140 {
        size = size.div_ceil(16) * 16;
    }
    Some(BlockLayout {
        size,
        align,
        member_offsets,
    })
}

/// Host-facing byte stride of a storage-buffer ELEMENT: what hosts use
/// to size and index the buffer, so it must match the `ArrayStride`
/// the SPIR-V backend decorates. Struct elements use their aligned
/// std430 size; scalars and vectors keep their tight `type_byte_size`.
/// (vec3 is the known exception where the tight size 12 differs from
/// the decorated stride 16 — long-standing behavior, kept as is.)
pub fn storage_elem_stride(ty: &Type) -> Option<u32> {
    match ty {
        Type::Constructed(TypeName::Tuple(_), _) | Type::Constructed(TypeName::Record(_), _) => {
            block_layout(ty, StorageLayout::Std430).map(|l| l.size)
        }
        _ if ty.is_mat() => Some(ty.mat_cols()? as u32 * std430_matrix_stride(ty)?),
        _ => type_byte_size(ty),
    }
}
