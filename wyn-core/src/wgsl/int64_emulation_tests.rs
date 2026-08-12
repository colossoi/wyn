use super::*;

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
