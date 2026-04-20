//! WGSL SSA-to-text lowering.
//!
//! The backend is built against WGSL directly (not as a GLSL dialect).
//! This file provides:
//! - [`wgsl_mangle`]: injective identifier mangler, aware of WGSL's
//!   reserved words.
//! - [`validate_wgsl_identifier`]: host-contract identifier validator
//!   (used when user-written names reach shader I/O sites — uniforms,
//!   bindings, vertex attributes).
//! - [`type_to_wgsl`]: Wyn polytype → WGSL type string.
//! - [`lower`]: entry point (currently returns an error until the
//!   per-instruction dispatch lands).

use std::collections::HashMap;

use polytype::Type as PolyType;

use crate::ast::TypeName;
use crate::error::Result;
use crate::ssa::types::Program;
use crate::types::TypeExt;

/// Lower an SSA program to a WGSL module. The module contains all entry
/// points (distinguished by `@vertex` / `@fragment` / `@compute`
/// attributes), module-scope types, bindings, and helper functions.
pub fn lower(_program: &Program) -> Result<String> {
    Err(crate::err_wgsl!("WGSL backend is not yet implemented"))
}

// -----------------------------------------------------------------------------
// Identifier mangling
// -----------------------------------------------------------------------------

/// Injective mangle from any Rust/Wyn identifier to a legal WGSL identifier.
///
/// Scheme:
/// - `w_` prefix for alnum-leading names; `w` prefix for non-alnum-leading
///   (so the escape's own leading underscore doesn't produce `w__`).
/// - `A-Za-z0-9` pass through.
/// - `_` → `_U`, `.` → `_D`, `$` → `_S`.
/// - Any other char → `_X<hex>_`.
/// - If the result hashes to a WGSL reserved word, suffix `_R`.
///
/// Identical to the GLSL scheme except for the reserved-word list. The
/// "never produce `__`" invariant is preserved.
pub fn wgsl_mangle(name: &str) -> String {
    use std::fmt::Write as _;
    let leads_with_alnum =
        name.chars().next().is_some_and(|c| matches!(c, 'A'..='Z' | 'a'..='z' | '0'..='9'));
    let mut out = String::with_capacity(name.len() + 2);
    out.push('w');
    if leads_with_alnum {
        out.push('_');
    }
    for c in name.chars() {
        match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' => out.push(c),
            '_' => out.push_str("_U"),
            '.' => out.push_str("_D"),
            '$' => out.push_str("_S"),
            _ => {
                let _ = write!(out, "_X{:x}_", c as u32);
            }
        }
    }
    // The scheme above cannot produce any of WGSL's reserved words (they're
    // all pure lowercase a-z without the `w_` prefix our output carries),
    // but a defensive check keeps the function robust against future
    // reserved-word additions.
    if wgsl_keyword_set().contains(out.as_str()) {
        out.push_str("_R");
    }
    out
}

/// Validate that a host-contract name (uniform / binding / attribute) is
/// a legal WGSL identifier. User-facing names that reach a shader I/O
/// site are preserved verbatim (the host side depends on the exact
/// spelling); we reject at compile time rather than let naga fail later.
pub fn validate_wgsl_identifier(name: &str) -> core::result::Result<(), String> {
    if name.is_empty() {
        return Err("identifier must not be empty".to_string());
    }
    let mut chars = name.chars();
    match chars.next().unwrap() {
        'A'..='Z' | 'a'..='z' | '_' => {}
        c => {
            return Err(format!(
                "identifier '{}' must start with a letter or underscore, got '{}'",
                name, c
            ));
        }
    }
    for c in chars {
        if !matches!(c, 'A'..='Z' | 'a'..='z' | '0'..='9' | '_') {
            return Err(format!(
                "identifier '{}' contains illegal character '{}'",
                name, c
            ));
        }
    }
    // WGSL reserves any identifier starting with `__` and all single-
    // underscore identifiers for the implementation.
    if name == "_" || name.starts_with("__") {
        return Err(format!("identifier '{}' is reserved by WGSL", name));
    }
    if wgsl_keyword_set().contains(name) {
        return Err(format!("identifier '{}' is a WGSL keyword", name));
    }
    Ok(())
}

/// WGSL keywords, reserved words, and built-in type names that user
/// identifiers must not collide with. Sourced from the W3C WGSL spec
/// §2.4 (keywords) and §2.5 (reserved words).
fn wgsl_keyword_set() -> &'static std::collections::HashSet<&'static str> {
    use std::sync::OnceLock;
    static SET: OnceLock<std::collections::HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        [
            // Keywords (§2.4)
            "alias",
            "break",
            "case",
            "const",
            "const_assert",
            "continue",
            "continuing",
            "default",
            "diagnostic",
            "discard",
            "else",
            "enable",
            "false",
            "fn",
            "for",
            "if",
            "let",
            "loop",
            "override",
            "requires",
            "return",
            "struct",
            "switch",
            "true",
            "var",
            "while",
            // Type predeclared names (§6)
            "array",
            "atomic",
            "bool",
            "f16",
            "f32",
            "i32",
            "mat2x2",
            "mat2x3",
            "mat2x4",
            "mat3x2",
            "mat3x3",
            "mat3x4",
            "mat4x2",
            "mat4x3",
            "mat4x4",
            "ptr",
            "sampler",
            "sampler_comparison",
            "texture_1d",
            "texture_2d",
            "texture_2d_array",
            "texture_3d",
            "texture_cube",
            "texture_cube_array",
            "texture_depth_2d",
            "texture_depth_2d_array",
            "texture_depth_cube",
            "texture_depth_cube_array",
            "texture_depth_multisampled_2d",
            "texture_external",
            "texture_multisampled_2d",
            "texture_storage_1d",
            "texture_storage_2d",
            "texture_storage_2d_array",
            "texture_storage_3d",
            "u32",
            "vec2",
            "vec3",
            "vec4",
            // Reserved words (§2.5) — subset that could realistically
            // collide; the full list includes dozens of future-reserved
            // tokens like `premultiplied`, `abstract`, `null`, etc.
            "NULL",
            "Self",
            "abstract",
            "active",
            "addressof",
            "as",
            "async",
            "attribute",
            "auto",
            "await",
            "become",
            "binding_array",
            "cast",
            "catch",
            "class",
            "co_await",
            "co_return",
            "co_yield",
            "coherent",
            "column_major",
            "common",
            "compile",
            "compile_fragment",
            "concept",
            "const_cast",
            "consteval",
            "constexpr",
            "constinit",
            "crate",
            "debugger",
            "decltype",
            "delete",
            "demote",
            "demote_to_helper",
            "do",
            "dynamic_cast",
            "enum",
            "explicit",
            "export",
            "extends",
            "extern",
            "external",
            "fallthrough",
            "filter",
            "final",
            "finally",
            "friend",
            "from",
            "fxgroup",
            "get",
            "goto",
            "groupshared",
            "highp",
            "impl",
            "implements",
            "import",
            "inline",
            "instanceof",
            "interface",
            "layout",
            "lowp",
            "macro",
            "macro_rules",
            "match",
            "mediump",
            "meta",
            "mod",
            "module",
            "move",
            "mut",
            "mutable",
            "namespace",
            "new",
            "nil",
            "noexcept",
            "noinline",
            "nointerpolation",
            "noperspective",
            "null",
            "nullptr",
            "of",
            "operator",
            "package",
            "packoffset",
            "partition",
            "pass",
            "patch",
            "pixelfragment",
            "precise",
            "precision",
            "premerge",
            "priv",
            "protected",
            "pub",
            "public",
            "readonly",
            "ref",
            "regardless",
            "register",
            "reinterpret_cast",
            "require",
            "resource",
            "restrict",
            "self",
            "set",
            "shared",
            "sizeof",
            "smooth",
            "snorm",
            "static",
            "static_assert",
            "static_cast",
            "std",
            "subroutine",
            "super",
            "target",
            "template",
            "this",
            "thread_local",
            "throw",
            "trait",
            "try",
            "type",
            "typedef",
            "typeid",
            "typename",
            "typeof",
            "union",
            "unless",
            "unorm",
            "unsafe",
            "unsized",
            "use",
            "using",
            "varying",
            "virtual",
            "volatile",
            "wgsl",
            "where",
            "with",
            "writeonly",
            "yield",
        ]
        .iter()
        .copied()
        .collect()
    })
}

// -----------------------------------------------------------------------------
// Type lowering
// -----------------------------------------------------------------------------

/// Lower a Wyn polytype to a WGSL type string. Caches generated tuple
/// structs so repeated uses of the same shape share a declaration.
///
/// `tuple_structs` is keyed by a canonical signature (comma-joined field
/// types) and maps to the generated struct name. Callers emit the full
/// struct declarations once at module scope using the accumulated map.
pub struct TypeEmitter {
    /// Cache: tuple field-type signature → generated struct name.
    tuple_type_cache: HashMap<String, String>,
    /// All tuple structs defined so far, keyed by struct name. The value
    /// is the ordered field-type list (WGSL type strings).
    pub tuple_structs: HashMap<String, Vec<String>>,
    tuple_counter: usize,
}

impl Default for TypeEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeEmitter {
    pub fn new() -> Self {
        Self {
            tuple_type_cache: HashMap::new(),
            tuple_structs: HashMap::new(),
            tuple_counter: 0,
        }
    }

    pub fn type_to_wgsl(&mut self, ty: &PolyType<TypeName>) -> Result<String> {
        match ty {
            PolyType::Constructed(name, args) => match name {
                TypeName::Float(32) => Ok("f32".to_string()),
                TypeName::Int(32) => Ok("i32".to_string()),
                TypeName::UInt(32) => Ok("u32".to_string()),
                TypeName::Bool => Ok("bool".to_string()),
                TypeName::Float(bits) | TypeName::Int(bits) | TypeName::UInt(bits) if *bits != 32 => Err(
                    crate::err_wgsl!("WGSL does not support {}-bit scalars (found {:?})", bits, ty),
                ),
                TypeName::Unit => Ok("/* unit */".to_string()),
                TypeName::Tuple(_) => {
                    let elem_types: Result<Vec<String>> =
                        args.iter().map(|a| self.type_to_wgsl(a)).collect();
                    let elem_types = elem_types?;
                    let sig = elem_types.join(",");
                    if let Some(name) = self.tuple_type_cache.get(&sig) {
                        return Ok(name.clone());
                    }
                    let struct_name = format!("T{}", self.tuple_counter);
                    self.tuple_counter += 1;
                    self.tuple_structs.insert(struct_name.clone(), elem_types);
                    self.tuple_type_cache.insert(sig, struct_name.clone());
                    Ok(struct_name)
                }
                TypeName::Vec => {
                    let elem = self.type_to_wgsl(
                        ty.elem_type()
                            .ok_or_else(|| crate::err_wgsl!("Vec type missing elem arg: {:?}", ty))?,
                    )?;
                    let n = ty
                        .vec_size()
                        .ok_or_else(|| crate::err_wgsl!("Vec type has non-concrete size: {:?}", ty))?;
                    if !(2..=4).contains(&n) {
                        return Err(crate::err_wgsl!("WGSL vector size must be 2/3/4, got {}", n));
                    }
                    Ok(format!("vec{}<{}>", n, elem))
                }
                TypeName::Mat => {
                    let elem = self.type_to_wgsl(
                        ty.elem_type()
                            .ok_or_else(|| crate::err_wgsl!("Mat type missing elem arg: {:?}", ty))?,
                    )?;
                    let cols = ty
                        .mat_cols()
                        .ok_or_else(|| crate::err_wgsl!("Mat type has non-concrete cols: {:?}", ty))?;
                    let rows = ty
                        .mat_rows()
                        .ok_or_else(|| crate::err_wgsl!("Mat type has non-concrete rows: {:?}", ty))?;
                    if elem != "f32" {
                        return Err(crate::err_wgsl!("WGSL matrices are f32-only (got {})", elem));
                    }
                    Ok(format!("mat{}x{}<{}>", cols, rows, elem))
                }
                TypeName::Array => {
                    let elem = self.type_to_wgsl(
                        ty.elem_type()
                            .ok_or_else(|| crate::err_wgsl!("Array type missing elem arg: {:?}", ty))?,
                    )?;
                    // WGSL fixed-size: `array<T, N>`. Runtime-sized is only
                    // legal in storage bindings; we handle those at the
                    // binding emission site, not here.
                    if let Some(PolyType::Constructed(TypeName::Size(n), _)) = ty.array_size() {
                        Ok(format!("array<{}, {}>", elem, n))
                    } else {
                        Err(crate::err_wgsl!(
                            "WGSL array must have concrete size for standalone use: {:?}",
                            ty
                        ))
                    }
                }
                TypeName::Record(fields) => Err(crate::err_wgsl!(
                    "Record type reached WGSL lowering (should have been lowered earlier): {:?}",
                    fields
                )),
                _ => Err(crate::err_wgsl!("unsupported type in WGSL lowering: {:?}", ty)),
            },
            _ => Err(crate::err_wgsl!("unsupported type in WGSL lowering: {:?}", ty)),
        }
    }
}

#[cfg(test)]
#[path = "ssa_lowering_tests.rs"]
mod ssa_lowering_tests;
