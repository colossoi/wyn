//! Tests for algebraic folding at the TLC-to-EGIR construction boundary.

use std::collections::{HashMap, HashSet};

use polytype::Type;
use smallvec::smallvec;

use super::{ConversionArenas, Converter};
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::egir::types::{PureOp, ValueKind};
use crate::ssa::types::ConstantValue;
use crate::SymbolTable;

fn with_converter<T>(test: impl FnOnce(&mut Converter<'_, '_>) -> T) -> T {
    let top_level = HashMap::new();
    let symbols = SymbolTable::new();
    let pure_constants = HashSet::new();
    let callable_boundaries = HashMap::new();
    let mut binding_ids = crate::IdSource::<u32>::new();
    let mut effect_ids = crate::IdSource::new();
    let mut arenas = ConversionArenas::new();
    let mut converter = Converter::new(
        &top_level,
        &symbols,
        pure_constants,
        &callable_boundaries,
        &mut binding_ids,
        &mut effect_ids,
        &mut arenas,
    );
    test(&mut converter)
}

fn f32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn vec3f32_ty() -> Type<TypeName> {
    crate::types::vec(3, f32_ty())
}

fn intrinsic(name: &str) -> PureOp {
    let def = catalog().lookup_by_any_name(name).unwrap_or_else(|| panic!("missing test builtin {name}"));
    PureOp::Intrinsic {
        id: def.id,
        overload_idx: 0,
    }
}

#[test]
fn runtime_value_plus_signed_float_zero_folds_in_both_orders() {
    with_converter(|converter| {
        let value = converter.graph.add_test_value_parameter(0, f32_ty());
        let positive_zero = converter.intern_pure(PureOp::Float("0.0".into()), smallvec![], f32_ty());
        let negative_zero = converter.intern_pure(
            PureOp::UnaryOp(crate::op::UnaryOperator::Negate),
            smallvec![positive_zero],
            f32_ty(),
        );

        for zero in [positive_zero, negative_zero] {
            let value_plus_zero = converter.intern_pure(
                PureOp::BinOp(crate::op::BinaryOperator::Add),
                smallvec![value, zero],
                f32_ty(),
            );
            let zero_plus_value = converter.intern_pure(
                PureOp::BinOp(crate::op::BinaryOperator::Add),
                smallvec![zero, value],
                f32_ty(),
            );

            assert_eq!(value_plus_zero, value);
            assert_eq!(zero_plus_value, value);
        }
    });
}

#[test]
fn runtime_i32_plus_zero_folds_in_both_orders() {
    with_converter(|converter| {
        let value = converter.graph.add_test_value_parameter(0, i32_ty());
        let zero = converter.graph.intern_constant(ConstantValue::I32(0), i32_ty());

        let value_plus_zero = converter.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Add),
            smallvec![value, zero],
            i32_ty(),
        );
        let zero_plus_value = converter.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Add),
            smallvec![zero, value],
            i32_ty(),
        );

        assert_eq!(value_plus_zero, value);
        assert_eq!(zero_plus_value, value);
    });
}

#[test]
fn runtime_f32_div_constant_folds_to_reciprocal_multiply() {
    with_converter(|converter| {
        let value = converter.graph.add_test_value_parameter(0, f32_ty());
        let divisor = converter.graph.intern_constant(ConstantValue::from_f32(4.0), f32_ty());

        let result = converter.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Divide),
            smallvec![value, divisor],
            f32_ty(),
        );

        let ValueKind::Pure { op, operands } = &converter.graph.nodes[result].kind else {
            panic!("expected reciprocal multiply")
        };
        assert!(matches!(op, PureOp::BinOp(crate::op::BinaryOperator::Multiply)));
        assert_eq!(operands[0], value);
        assert!(matches!(
            converter.graph.nodes[operands[1]].kind,
            ValueKind::Constant(ConstantValue::F32(bits)) if f32::from_bits(bits) == 0.25
        ));
    });
}

#[test]
fn runtime_f32_vector_div_scalar_constant_folds_to_reciprocal_multiply() {
    with_converter(|converter| {
        let value = converter.graph.add_test_value_parameter(0, vec3f32_ty());
        let divisor = converter.graph.intern_constant(ConstantValue::from_f32(8.0), f32_ty());

        let result = converter.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Divide),
            smallvec![value, divisor],
            vec3f32_ty(),
        );

        let ValueKind::Pure { op, operands } = &converter.graph.nodes[result].kind else {
            panic!("expected vector/scalar reciprocal multiply")
        };
        assert!(matches!(op, PureOp::BinOp(crate::op::BinaryOperator::Multiply)));
        assert_eq!(operands[0], value);
        assert!(matches!(
            converter.graph.nodes[operands[1]].kind,
            ValueKind::Constant(ConstantValue::F32(bits)) if f32::from_bits(bits) == 0.125
        ));
        assert_eq!(converter.graph.nodes[operands[1]].ty, f32_ty());
    });
}

#[test]
fn f32_div_zero_does_not_rewrite_to_multiply() {
    with_converter(|converter| {
        let value = converter.graph.add_test_value_parameter(0, f32_ty());
        let zero = converter.graph.intern_constant(ConstantValue::from_f32(0.0), f32_ty());

        let result = converter.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Divide),
            smallvec![value, zero],
            f32_ty(),
        );

        assert!(matches!(
            &converter.graph.nodes[result].kind,
            ValueKind::Pure {
                op: PureOp::BinOp(crate::op::BinaryOperator::Divide),
                ..
            }
        ));
    });
}

#[test]
fn identity_bitcast_folds_to_operand() {
    with_converter(|converter| {
        let value = converter.graph.add_test_value_parameter(0, i32_ty());

        let result = converter.intern_pure(intrinsic("i32.i32"), smallvec![value], i32_ty());

        assert_eq!(result, value);
    });
}

#[test]
fn inverse_bitcasts_fold_to_original_operand() {
    with_converter(|converter| {
        let value = converter.graph.add_test_value_parameter(0, u32_ty());
        let as_i32 = converter.intern_pure(intrinsic("i32.u32"), smallvec![value], i32_ty());

        let round_trip = converter.intern_pure(intrinsic("u32.i32"), smallvec![as_i32], u32_ty());

        assert_eq!(round_trip, value);
    });
}

#[test]
fn required_bitcast_is_retained() {
    with_converter(|converter| {
        let value = converter.graph.add_test_value_parameter(0, u32_ty());

        let result = converter.intern_pure(intrinsic("i32.u32"), smallvec![value], i32_ty());

        assert!(matches!(
            &converter.graph.nodes[result].kind,
            ValueKind::Pure { op: PureOp::Intrinsic { .. }, operands } if operands.as_slice() == [value]
        ));
    });
}

#[test]
fn unary_neg_of_float_literal_folds_to_constant() {
    // `-(0.5)` must fold to the constant -0.5, not stay a runtime `OpFNegate`.
    // Otherwise an array element written `-0.5` keeps a non-constant operand
    // and the whole array lowers to `OpCompositeConstruct` (rebuilt per call)
    // instead of an `OpConstantComposite` that can hoist to a shared global.
    with_converter(|converter| {
        let half = converter.intern_pure(PureOp::Float("0.5".into()), smallvec![], f32_ty());
        let neg = converter.intern_pure(
            PureOp::UnaryOp(crate::op::UnaryOperator::Negate),
            smallvec![half],
            f32_ty(),
        );
        match &converter.graph.nodes[neg].kind {
            ValueKind::Constant(ConstantValue::F32(bits)) => assert_eq!(f32::from_bits(*bits), -0.5),
            _ => panic!("expected -(0.5) to fold to the constant -0.5"),
        }
    });
}

#[test]
fn unary_neg_of_int_literal_folds_to_constant() {
    with_converter(|converter| {
        let five = converter.intern_pure(PureOp::Int("5".into()), smallvec![], i32_ty());
        let neg = converter.intern_pure(
            PureOp::UnaryOp(crate::op::UnaryOperator::Negate),
            smallvec![five],
            i32_ty(),
        );
        match &converter.graph.nodes[neg].kind {
            ValueKind::Constant(ConstantValue::I32(v)) => assert_eq!(*v, -5),
            _ => panic!("expected -(5) to fold to the constant -5"),
        }
    });
}

#[test]
fn unary_neg_of_runtime_value_does_not_fold() {
    with_converter(|converter| {
        let value = converter.graph.add_test_value_parameter(0, f32_ty());
        let neg = converter.intern_pure(
            PureOp::UnaryOp(crate::op::UnaryOperator::Negate),
            smallvec![value],
            f32_ty(),
        );
        let ValueKind::Pure { op, operands } = &converter.graph.nodes[neg].kind else {
            panic!("expected a pure node")
        };
        assert!(matches!(op, PureOp::UnaryOp(crate::op::UnaryOperator::Negate)));
        assert_eq!(operands.as_slice(), &[value]);
    });
}
