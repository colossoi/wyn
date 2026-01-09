//! Type size and alignment helpers for buffer layout calculations.

use crate::ast::{Type, TypeName};

/// Calculate the byte size of a type for buffer layout purposes.
/// Returns None for types that don't have a fixed size (e.g., runtime arrays).
pub fn type_byte_size(ty: &Type) -> Option<u32> {
    match ty {
        Type::Constructed(TypeName::Int(bits), _) => Some((*bits / 8) as u32),
        Type::Constructed(TypeName::UInt(bits), _) => Some((*bits / 8) as u32),
        Type::Constructed(TypeName::Float(bits), _) => Some((*bits / 8) as u32),
        Type::Constructed(TypeName::Unit, _) => Some(0),
        Type::Constructed(TypeName::Vec, args) if args.len() >= 2 => {
            // Vec<size, elem_type>
            let size = match &args[0] {
                Type::Constructed(TypeName::Size(n), _) => *n as u32,
                _ => return None,
            };
            let elem_size = type_byte_size(&args[1])?;
            Some(size * elem_size)
        }
        Type::Constructed(TypeName::Mat, args) if args.len() >= 3 => {
            // Mat<cols, rows, elem_type>
            let cols = match &args[0] {
                Type::Constructed(TypeName::Size(n), _) => *n as u32,
                _ => return None,
            };
            let rows = match &args[1] {
                Type::Constructed(TypeName::Size(n), _) => *n as u32,
                _ => return None,
            };
            let elem_size = type_byte_size(&args[2])?;
            Some(cols * rows * elem_size)
        }
        Type::Constructed(TypeName::Array, args) => {
            assert!(args.len() == 3);
            // Array<size, elem_type>
            let size = match &args[0] {
                Type::Constructed(TypeName::Size(n), _) => *n as u32,
                _ => return None,
            };
            let elem_size = type_byte_size(&args[1])?;
            Some(size * elem_size)
        }
        Type::Constructed(TypeName::Tuple(arity), args) => {
            // Sum of field sizes (simplified - doesn't account for alignment padding)
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
        Type::Constructed(TypeName::Vec, args) if args.len() >= 2 => {
            // vec2 alignment = 2N, vec3/vec4 alignment = 4N
            let size = match &args[0] {
                Type::Constructed(TypeName::Size(n), _) => *n as u32,
                _ => return None,
            };
            let elem_align = std430_alignment(&args[1])?;
            Some(if size == 2 { 2 * elem_align } else { 4 * elem_align })
        }
        Type::Constructed(TypeName::Mat, args) if args.len() >= 3 => {
            // Matrix alignment = column vector alignment
            let rows = match &args[1] {
                Type::Constructed(TypeName::Size(n), _) => *n as u32,
                _ => return None,
            };
            let elem_align = std430_alignment(&args[2])?;
            Some(if rows == 2 { 2 * elem_align } else { 4 * elem_align })
        }
        Type::Constructed(TypeName::Array, args) if args.len() >= 2 => {
            // Array alignment = element alignment (std430 is tightly packed)
            std430_alignment(&args[1])
        }
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
        Type::Constructed(TypeName::Vec, args) if args.len() >= 2 => {
            // vec2 alignment = 2N, vec3/vec4 alignment = 4N
            let size = match &args[0] {
                Type::Constructed(TypeName::Size(n), _) => *n as u32,
                _ => return None,
            };
            let elem_align = std140_alignment(&args[1])?;
            Some(if size == 2 { 2 * elem_align } else { 4 * elem_align })
        }
        Type::Constructed(TypeName::Mat, args) if args.len() >= 3 => {
            // Matrix alignment = column vector alignment, rounded up to vec4
            let elem_align = std140_alignment(&args[2])?;
            Some(4 * elem_align) // Always vec4 alignment in std140
        }
        Type::Constructed(TypeName::Array, args) if args.len() >= 2 => {
            // In std140, arrays have alignment rounded up to vec4
            let elem_align = std140_alignment(&args[1])?;
            Some(((elem_align + 15) / 16) * 16)
        }
        _ => None,
    }
}
