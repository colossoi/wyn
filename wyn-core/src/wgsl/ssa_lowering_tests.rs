//! WGSL backend unit tests.

use super::{TypeEmitter, validate_wgsl_identifier, wgsl_mangle};
use crate::ast::TypeName;
use polytype::Type as PolyType;

// ---------- wgsl_mangle: per-char cases ----------

#[test]
fn mangle_plain_passthrough() {
    assert_eq!(wgsl_mangle("foo"), "w_foo");
    assert_eq!(wgsl_mangle("a1b2c3"), "w_a1b2c3");
}

#[test]
fn mangle_dot() {
    assert_eq!(wgsl_mangle("a.b"), "w_a_Db");
    assert_eq!(
        wgsl_mangle("materials.pbrDistributionGGX"),
        "w_materials_DpbrDistributionGGX"
    );
}

#[test]
fn mangle_dollar() {
    assert_eq!(wgsl_mangle("foo$0"), "w_foo_S0");
}

#[test]
fn mangle_underscore_inside() {
    // `_` inside a name → `_U`.
    assert_eq!(wgsl_mangle("a_b"), "w_a_Ub");
}

#[test]
fn mangle_leading_underscore_contracts_prefix() {
    // Leading non-alnum char contracts the prefix from `w_` to `w` so we
    // never produce `__` at the prefix/body boundary.
    assert_eq!(wgsl_mangle("_foo"), "w_Ufoo");
    assert_eq!(wgsl_mangle("_w_intrinsic_foo"), "w_Uw_Uintrinsic_Ufoo");
}

#[test]
fn mangle_empty() {
    // Empty input: no leading char, so no prefix underscore. Prefix alone.
    assert_eq!(wgsl_mangle(""), "w");
}

#[test]
fn mangle_non_ascii_fallback() {
    // Non-ASCII char exercises the `_X<hex>_` fallback.
    assert_eq!(wgsl_mangle("a-b"), "w_a_X2d_b");
    assert_eq!(wgsl_mangle("-ab"), "w_X2d_ab");
}

#[test]
fn mangle_no_double_underscore() {
    // Invariant: no mangled output contains `__`. Spot-check a few edge
    // cases; the mangler's design prevents it structurally.
    for input in ["_w_lambda_13", "a__b", "foo_bar_baz", "_", "__", "a._b", "a.$b"] {
        let out = wgsl_mangle(input);
        assert!(
            !out.contains("__"),
            "mangle({:?}) produced {:?} which contains `__`",
            input,
            out
        );
    }
}

// ---------- validate_wgsl_identifier ----------

#[test]
fn validate_accepts_plain_ident() {
    assert!(validate_wgsl_identifier("iResolution").is_ok());
    assert!(validate_wgsl_identifier("my_buffer_0").is_ok());
    assert!(validate_wgsl_identifier("_leading_underscore_ok_for_host").is_ok());
}

#[test]
fn validate_rejects_empty() {
    assert!(validate_wgsl_identifier("").is_err());
}

#[test]
fn validate_rejects_digit_leading() {
    assert!(validate_wgsl_identifier("0invalid").is_err());
}

#[test]
fn validate_rejects_illegal_chars() {
    assert!(validate_wgsl_identifier("foo-bar").is_err());
    assert!(validate_wgsl_identifier("foo.bar").is_err());
}

#[test]
fn validate_rejects_reserved_keyword() {
    for kw in &["fn", "let", "var", "struct", "return", "loop"] {
        assert!(
            validate_wgsl_identifier(kw).is_err(),
            "keyword {} must be rejected",
            kw
        );
    }
}

#[test]
fn validate_rejects_reserved_type_name() {
    for ty in &["f32", "i32", "u32", "bool", "vec3", "mat4x4", "array"] {
        assert!(
            validate_wgsl_identifier(ty).is_err(),
            "type name {} must be rejected",
            ty
        );
    }
}

#[test]
fn validate_rejects_double_underscore_prefix() {
    // WGSL reserves `__...` for the implementation.
    assert!(validate_wgsl_identifier("__foo").is_err());
    assert!(validate_wgsl_identifier("_").is_err());
}

// ---------- type_to_wgsl ----------

fn scalar_ty(name: TypeName) -> PolyType<TypeName> {
    PolyType::Constructed(name, vec![])
}

#[test]
fn type_f32() {
    let mut e = TypeEmitter::new();
    assert_eq!(e.type_to_wgsl(&scalar_ty(TypeName::Float(32))).unwrap(), "f32");
}

#[test]
fn type_i32() {
    let mut e = TypeEmitter::new();
    assert_eq!(e.type_to_wgsl(&scalar_ty(TypeName::Int(32))).unwrap(), "i32");
}

#[test]
fn type_u32() {
    let mut e = TypeEmitter::new();
    assert_eq!(e.type_to_wgsl(&scalar_ty(TypeName::UInt(32))).unwrap(), "u32");
}

#[test]
fn type_bool() {
    let mut e = TypeEmitter::new();
    assert_eq!(e.type_to_wgsl(&scalar_ty(TypeName::Bool)).unwrap(), "bool");
}

#[test]
fn type_f64_rejected() {
    let mut e = TypeEmitter::new();
    let result = e.type_to_wgsl(&scalar_ty(TypeName::Float(64)));
    assert!(result.is_err(), "f64 must be rejected by WGSL type lowering");
}

#[test]
fn type_vec3f32() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Vec,
        vec![
            scalar_ty(TypeName::Float(32)),
            PolyType::Constructed(TypeName::Size(3), vec![]),
        ],
    );
    assert_eq!(e.type_to_wgsl(&ty).unwrap(), "vec3<f32>");
}

#[test]
fn type_mat4x4f32() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Mat,
        vec![
            scalar_ty(TypeName::Float(32)),
            PolyType::Constructed(TypeName::Size(4), vec![]),
            PolyType::Constructed(TypeName::Size(4), vec![]),
        ],
    );
    assert_eq!(e.type_to_wgsl(&ty).unwrap(), "mat4x4<f32>");
}

#[test]
fn type_array_sized() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Array,
        vec![
            scalar_ty(TypeName::Float(32)),
            PolyType::Constructed(TypeName::Size(8), vec![]),
            PolyType::Constructed(TypeName::ArrayVariantComposite, vec![]),
        ],
    );
    assert_eq!(e.type_to_wgsl(&ty).unwrap(), "array<f32, 8>");
}

#[test]
fn type_tuple_creates_struct() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Tuple(2),
        vec![scalar_ty(TypeName::Float(32)), scalar_ty(TypeName::Int(32))],
    );
    let name = e.type_to_wgsl(&ty).unwrap();
    assert_eq!(name, "T0");
    assert_eq!(
        e.tuple_structs.get("T0").unwrap(),
        &vec!["f32".to_string(), "i32".to_string()]
    );
}

#[test]
fn type_tuple_caches_by_signature() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Tuple(2),
        vec![scalar_ty(TypeName::Float(32)), scalar_ty(TypeName::Int(32))],
    );
    let n1 = e.type_to_wgsl(&ty).unwrap();
    let n2 = e.type_to_wgsl(&ty).unwrap();
    assert_eq!(n1, n2);
    assert_eq!(e.tuple_structs.len(), 1);
}

#[test]
fn type_tuple_distinct_shapes_distinct_structs() {
    let mut e = TypeEmitter::new();
    let ty_a = PolyType::Constructed(
        TypeName::Tuple(2),
        vec![scalar_ty(TypeName::Float(32)), scalar_ty(TypeName::Int(32))],
    );
    let ty_b = PolyType::Constructed(
        TypeName::Tuple(2),
        vec![scalar_ty(TypeName::Int(32)), scalar_ty(TypeName::Float(32))],
    );
    let na = e.type_to_wgsl(&ty_a).unwrap();
    let nb = e.type_to_wgsl(&ty_b).unwrap();
    assert_ne!(na, nb);
    assert_eq!(e.tuple_structs.len(), 2);
}

// ---------- end-to-end lowering (scaffold) ----------

#[test]
fn scaffold_lower_returns_error() {
    // Until the real lowering lands, `lower` must error on any program.
    let program = crate::ssa::types::Program {
        functions: Vec::new(),
        entry_points: Vec::new(),
        constants: Vec::new(),
        uniforms: Vec::new(),
        storage: Vec::new(),
    };
    assert!(super::lower(&program).is_err());
}
