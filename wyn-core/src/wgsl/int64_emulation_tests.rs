use super::*;

fn scalar(name: TypeName) -> Type<TypeName> {
    Type::Constructed(name, vec![])
}

#[test]
fn literals_split_into_little_endian_lanes() {
    assert_eq!(lower_literal("0").unwrap(), "vec2<u32>(0u, 0u)");
    assert_eq!(
        lower_literal("18446744073709551615").unwrap(),
        "vec2<u32>(4294967295u, 4294967295u)"
    );
    assert_eq!(
        lower_literal("81985529216486895").unwrap(),
        "vec2<u32>(2309737967u, 19088743u)"
    );
}

#[test]
fn negative_constant_fold_residual_preserves_u64_bits() {
    assert_eq!(
        lower_literal("-1").unwrap(),
        "vec2<u32>(4294967295u, 4294967295u)"
    );
    assert_eq!(
        lower_literal(i64::MIN.to_string().as_str()).unwrap(),
        "vec2<u32>(0u, 2147483648u)"
    );
}

#[test]
fn helpers_are_emitted_only_when_used() {
    let mut emulation = U64Emulation::default();
    let mut output = String::new();
    emulation.emit_helpers(&mut output);
    assert!(output.is_empty());

    assert_eq!(
        emulation.lower_binary(BinaryOperator::Add, "a", "b").unwrap(),
        "_wyn_u64_add(a, b)"
    );
    emulation.emit_helpers(&mut output);
    assert!(output.contains("fn _wyn_u64_add"));
    assert!(!output.contains("fn _wyn_u64_shl"));
}

#[test]
fn unsupported_arithmetic_reports_the_operator() {
    let error = U64Emulation::default().lower_binary(BinaryOperator::Multiply, "a", "b").unwrap_err();
    assert!(error.contains("u64 operator '*'"));
}

#[test]
fn constant_shifts_cover_lane_boundaries_without_helpers() {
    let emulation = U64Emulation::default();
    assert_eq!(
        emulation.lower_constant_shift(BinaryOperator::ShiftLeft, "x", 0),
        Some("x".to_string())
    );
    assert_eq!(
        emulation.lower_constant_shift(BinaryOperator::ShiftRight, "x", 32),
        Some("vec2<u32>((x).y >> 0u, 0u)".to_string())
    );
    assert_eq!(
        emulation.lower_constant_shift(BinaryOperator::ShiftLeft, "x", 63),
        Some("vec2<u32>(0u, (x).x << 31u)".to_string())
    );

    let mut helpers = String::new();
    emulation.emit_helpers(&mut helpers);
    assert!(helpers.is_empty());
}

#[test]
fn conversions_make_width_changes_explicit() {
    let u32_ty = scalar(TypeName::UInt(32));
    let i32_ty = scalar(TypeName::Int(32));
    let u64_ty = scalar(TypeName::UInt(64));

    assert_eq!(
        lower_conversion(&u32_ty, &u64_ty, "x").unwrap().unwrap(),
        "vec2<u32>(x, 0u)"
    );
    assert_eq!(lower_conversion(&u64_ty, &u32_ty, "x").unwrap().unwrap(), "(x).x");
    assert_eq!(
        lower_conversion(&i32_ty, &u64_ty, "x").unwrap().unwrap(),
        "vec2<u32>(bitcast<u32>(x), 0u)"
    );
    assert_eq!(
        lower_conversion(&u64_ty, &i32_ty, "x").unwrap().unwrap(),
        "bitcast<i32>((x).x)"
    );
}

#[test]
fn comparisons_reduce_vector_conditions_to_scalar_bool() {
    let mut emulation = U64Emulation::default();
    assert_eq!(
        emulation.lower_binary(BinaryOperator::Equal, "a", "b").unwrap(),
        "all(a == b)"
    );
    assert_eq!(
        emulation.lower_binary(BinaryOperator::NotEqual, "a", "b").unwrap(),
        "any(a != b)"
    );
    let less = emulation.lower_binary(BinaryOperator::Less, "a", "b").unwrap();
    assert!(less.contains("(a).y < (b).y"));
    assert!(less.contains("(a).x < (b).x"));
}
