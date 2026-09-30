use super::*;
use crate::builtins::names::{INTRINSIC_DOT, INTRINSIC_LENGTH};

#[test]
fn intrinsic_reuse_excludes_pointer_and_context_operations() {
    use lowering::PrimOp;
    // Modf/Frexp write through pointers; InterpolateAt* reads fragment context.
    for ext in [35, 51, 76, 77, 78, u32::MAX] {
        assert!(!BuiltinLowering::PrimOp(PrimOp::GlslExt(ext)).is_reusable());
    }
    for name in [
        "_w_intrinsic_uninit",
        "_w_intrinsic_storage_index",
        "f32.d_fdx",
        "f32.d_fdy",
        "f32.fwidth",
    ] {
        let builtin = catalog().lookup_by_any_name(name).unwrap();
        assert!(
            builtin
                .overloads()
                .iter()
                .all(|overload| builtin.raw.purity != Purity::Pure || !overload.lowering.is_reusable()),
            "{name}"
        );
    }
    assert!(!BuiltinLowering::LinkedSpirv("unknown").is_reusable());
}

#[test]
fn select_intrinsic_has_a_durable_typed_identity() {
    assert_eq!(intrinsic_arity(names::INTRINSIC_SELECT), Some(3));
    let def = by_id(catalog().known().select);
    assert_eq!(def.raw.purity, Purity::Pure);
    assert!(matches!(
        def.overloads()[0].lowering,
        BuiltinLowering::PrimOp(lowering::PrimOp::Select)
    ));
    assert!(def.overloads()[0].lowering.is_speculatable());
}

#[test]
fn intrinsic_arity_for_length_is_one() {
    // length: [n]A -> i32
    assert_eq!(intrinsic_arity(INTRINSIC_LENGTH), Some(1));
}

#[test]
fn intrinsic_arity_for_dot_is_two() {
    // dot: vecN A -> vecN A -> A
    assert_eq!(intrinsic_arity(INTRINSIC_DOT), Some(2));
}

#[test]
fn intrinsic_arity_for_unknown_is_none() {
    assert_eq!(intrinsic_arity("definitely_not_a_real_intrinsic"), None);
}
