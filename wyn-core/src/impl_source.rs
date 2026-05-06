// Field-lookup helper for vec types — used by the type checker for
// `vec3f32.x` style access. Lives here transitionally; will move to
// `types/mod.rs` alongside the swizzle helpers.

use crate::ast::{Type, TypeName};

/// Look up a vec component field type. `type_name` is a name like
/// `vec3f32`; `field_name` is one of `x`/`y`/`z`/`w` (validated against
/// the vec size). Returns the element type if the field is in range.
pub fn vec_field_type(type_name: &str, field_name: &str) -> Option<Type> {
    if !type_name.starts_with("vec") {
        return None;
    }
    let size = type_name.chars().nth(3)?.to_digit(10)? as usize;
    let valid = matches!(
        (size, field_name),
        (2, "x" | "y") | (3, "x" | "y" | "z") | (4, "x" | "y" | "z" | "w")
    );
    if !valid {
        return None;
    }
    let elem_type_str = &type_name[4..];
    let elem_type_name = match elem_type_str {
        "f32" => TypeName::Float(32),
        "f64" => TypeName::Float(64),
        "i32" => TypeName::Int(32),
        "i64" => TypeName::Int(64),
        "u32" => TypeName::UInt(32),
        "u64" => TypeName::UInt(64),
        "bool" => TypeName::Bool,
        other => TypeName::Named(other.to_string()),
    };
    Some(Type::Constructed(elem_type_name, vec![]))
}
