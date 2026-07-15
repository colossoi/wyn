//! Tests for algebraic folds applied during `EGraph::intern_pure`.

use super::super::types::{EGraph as GenericEGraph, ENode, NodeId, PureOp};
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::ssa::types::ConstantValue;
use polytype::Type;
use smallvec::smallvec;

type EGraph = GenericEGraph;

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

/// Count chained multiplies starting at `nid`. Returns `Some(n)` if the
/// node is `x * x * … * x` (left-to-right) with `n` total mul ops over
/// the same leaf base, `None` otherwise.
fn chain_len_over_same_base(graph: &EGraph, nid: NodeId, base: NodeId) -> Option<usize> {
    let mut current = nid;
    let mut muls = 0usize;
    loop {
        if current == base {
            return Some(muls);
        }
        let ENode::Pure { op, operands } = &graph.nodes[current] else {
            return None;
        };
        match op {
            PureOp::BinOp(name) if name == "*" && operands.len() == 2 => {
                if operands[1] != base {
                    return None;
                }
                current = operands[0];
                muls += 1;
            }
            _ => return None,
        }
    }
}

#[test]
fn runtime_value_plus_signed_float_zero_folds_in_both_orders() {
    let mut g = EGraph::new();
    let value = g.add_func_param(0, f32_ty());
    let positive_zero = g.intern_pure(PureOp::Float("0.0".into()), smallvec![], f32_ty());
    let negative_zero = g.intern_pure(PureOp::UnaryOp("-".into()), smallvec![positive_zero], f32_ty());

    for zero in [positive_zero, negative_zero] {
        let value_plus_zero = g.intern_pure(PureOp::BinOp("+".into()), smallvec![value, zero], f32_ty());
        let zero_plus_value = g.intern_pure(PureOp::BinOp("+".into()), smallvec![zero, value], f32_ty());

        assert_eq!(value_plus_zero, value);
        assert_eq!(zero_plus_value, value);
    }
}

#[test]
fn runtime_i32_plus_zero_folds_in_both_orders() {
    let mut g = EGraph::new();
    let value = g.add_func_param(0, i32_ty());
    let zero = g.intern_constant(ConstantValue::I32(0), i32_ty());

    let value_plus_zero = g.intern_pure(PureOp::BinOp("+".into()), smallvec![value, zero], i32_ty());
    let zero_plus_value = g.intern_pure(PureOp::BinOp("+".into()), smallvec![zero, value], i32_ty());

    assert_eq!(value_plus_zero, value);
    assert_eq!(zero_plus_value, value);
}

#[test]
fn runtime_f32_div_constant_folds_to_reciprocal_multiply() {
    let mut g = EGraph::new();
    let value = g.add_func_param(0, f32_ty());
    let divisor = g.intern_constant(ConstantValue::from_f32(4.0), f32_ty());

    let result = g.intern_pure(PureOp::BinOp("/".into()), smallvec![value, divisor], f32_ty());

    let ENode::Pure { op, operands } = &g.nodes[result] else {
        panic!("expected reciprocal multiply")
    };
    assert!(matches!(op, PureOp::BinOp(name) if name == "*"));
    assert_eq!(operands[0], value);
    assert!(matches!(
        g.nodes[operands[1]],
        ENode::Constant(ConstantValue::F32(bits)) if f32::from_bits(bits) == 0.25
    ));
}

#[test]
fn runtime_f32_vector_div_scalar_constant_folds_to_reciprocal_multiply() {
    let mut g = EGraph::new();
    let value = g.add_func_param(0, vec3f32_ty());
    let divisor = g.intern_constant(ConstantValue::from_f32(8.0), f32_ty());

    let result = g.intern_pure(PureOp::BinOp("/".into()), smallvec![value, divisor], vec3f32_ty());

    let ENode::Pure { op, operands } = &g.nodes[result] else {
        panic!("expected vector/scalar reciprocal multiply")
    };
    assert!(matches!(op, PureOp::BinOp(name) if name == "*"));
    assert_eq!(operands[0], value);
    assert!(matches!(
        g.nodes[operands[1]],
        ENode::Constant(ConstantValue::F32(bits)) if f32::from_bits(bits) == 0.125
    ));
    assert_eq!(g.types[&operands[1]], f32_ty());
}

#[test]
fn f32_div_zero_does_not_rewrite_to_multiply() {
    let mut g = EGraph::new();
    let value = g.add_func_param(0, f32_ty());
    let zero = g.intern_constant(ConstantValue::from_f32(0.0), f32_ty());

    let result = g.intern_pure(PureOp::BinOp("/".into()), smallvec![value, zero], f32_ty());

    assert!(matches!(
        &g.nodes[result],
        ENode::Pure { op: PureOp::BinOp(name), .. } if name == "/"
    ));
}

#[test]
fn identity_bitcast_folds_to_operand() {
    let mut g = EGraph::new();
    let value = g.add_func_param(0, i32_ty());

    let result = g.intern_pure(intrinsic("i32.i32"), smallvec![value], i32_ty());

    assert_eq!(result, value);
}

#[test]
fn inverse_bitcasts_fold_to_original_operand() {
    let mut g = EGraph::new();
    let value = g.add_func_param(0, u32_ty());
    let as_i32 = g.intern_pure(intrinsic("i32.u32"), smallvec![value], i32_ty());

    let round_trip = g.intern_pure(intrinsic("u32.i32"), smallvec![as_i32], u32_ty());

    assert_eq!(round_trip, value);
}

#[test]
fn required_bitcast_is_retained() {
    let mut g = EGraph::new();
    let value = g.add_func_param(0, u32_ty());

    let result = g.intern_pure(intrinsic("i32.u32"), smallvec![value], i32_ty());

    assert!(matches!(
        &g.nodes[result],
        ENode::Pure { op: PureOp::Intrinsic { .. }, operands } if operands.as_slice() == [value]
    ));
}

#[test]
fn pow_const_2_folds_to_one_mul_f32() {
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let exp = g.intern_constant(ConstantValue::I32(2), i32_ty());
    let result = g.intern_pure(PureOp::BinOp("**".into()), smallvec![base, exp], f32_ty());
    assert_eq!(chain_len_over_same_base(&g, result, base), Some(1));
}

#[test]
fn pow_const_5_folds_to_four_muls_f32() {
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let exp = g.intern_constant(ConstantValue::I32(5), i32_ty());
    let result = g.intern_pure(PureOp::BinOp("**".into()), smallvec![base, exp], f32_ty());
    assert_eq!(chain_len_over_same_base(&g, result, base), Some(4));
}

#[test]
fn pow_const_7_folds_to_six_muls_i32() {
    let mut g = EGraph::new();
    let base = g.add_func_param(0, i32_ty());
    let exp = g.intern_constant(ConstantValue::I32(7), i32_ty());
    let result = g.intern_pure(PureOp::BinOp("**".into()), smallvec![base, exp], i32_ty());
    assert_eq!(chain_len_over_same_base(&g, result, base), Some(6));
}

#[test]
fn pow_const_8_does_not_fold() {
    // 8 is the exclusive upper bound; the chain (7 muls) stops being a
    // clear win and the backend's `**` lowering takes over.
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let exp = g.intern_constant(ConstantValue::I32(8), i32_ty());
    let result = g.intern_pure(PureOp::BinOp("**".into()), smallvec![base, exp], f32_ty());
    // Result is the raw `**` node, not a multiply chain.
    let ENode::Pure { op, .. } = &g.nodes[result] else {
        panic!("expected pure node")
    };
    assert!(matches!(op, PureOp::BinOp(n) if n == "**"));
}

#[test]
fn pow_const_1_does_not_fold() {
    // 1 is below the rule's 2..8 window. A separate identity rule could
    // strip it later; this one leaves it alone.
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let exp = g.intern_constant(ConstantValue::I32(1), i32_ty());
    let result = g.intern_pure(PureOp::BinOp("**".into()), smallvec![base, exp], f32_ty());
    let ENode::Pure { op, .. } = &g.nodes[result] else {
        panic!("expected pure node")
    };
    assert!(matches!(op, PureOp::BinOp(n) if n == "**"));
}

#[test]
fn pow_f32_exponent_folds_for_whole_float() {
    // Float `**` requires same-typed operands; the exponent arrives as a
    // f32 literal even when the user wrote `x ** 3`.
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let exp = g.intern_constant(ConstantValue::F32(3.0f32.to_bits()), f32_ty());
    let result = g.intern_pure(PureOp::BinOp("**".into()), smallvec![base, exp], f32_ty());
    assert_eq!(chain_len_over_same_base(&g, result, base), Some(2));
}

#[test]
fn pow_f32_fractional_exponent_does_not_fold() {
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let exp = g.intern_constant(ConstantValue::F32(2.5f32.to_bits()), f32_ty());
    let result = g.intern_pure(PureOp::BinOp("**".into()), smallvec![base, exp], f32_ty());
    let ENode::Pure { op, .. } = &g.nodes[result] else {
        panic!("expected pure node")
    };
    assert!(matches!(op, PureOp::BinOp(n) if n == "**"));
}

#[test]
fn pow_non_const_exponent_does_not_fold() {
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let exp = g.add_func_param(1, i32_ty()); // runtime
    let result = g.intern_pure(PureOp::BinOp("**".into()), smallvec![base, exp], f32_ty());
    let ENode::Pure { op, operands } = &g.nodes[result] else {
        panic!("expected pure node")
    };
    assert!(matches!(op, PureOp::BinOp(n) if n == "**"));
    assert_eq!(operands.as_slice(), &[base, exp]);
}

#[test]
fn unary_neg_of_float_literal_folds_to_constant() {
    // `-(0.5)` must fold to the constant -0.5, not stay a runtime `OpFNegate`.
    // Otherwise an array element written `-0.5` keeps a non-constant operand
    // and the whole array lowers to `OpCompositeConstruct` (rebuilt per call)
    // instead of an `OpConstantComposite` that can hoist to a shared global.
    let mut g = EGraph::new();
    let half = g.intern_pure(PureOp::Float("0.5".into()), smallvec![], f32_ty());
    let neg = g.intern_pure(PureOp::UnaryOp("-".into()), smallvec![half], f32_ty());
    match &g.nodes[neg] {
        ENode::Constant(ConstantValue::F32(bits)) => assert_eq!(f32::from_bits(*bits), -0.5),
        _ => panic!("expected -(0.5) to fold to the constant -0.5"),
    }
}

#[test]
fn unary_neg_of_int_literal_folds_to_constant() {
    let mut g = EGraph::new();
    let five = g.intern_pure(PureOp::Int("5".into()), smallvec![], i32_ty());
    let neg = g.intern_pure(PureOp::UnaryOp("-".into()), smallvec![five], i32_ty());
    match &g.nodes[neg] {
        ENode::Constant(ConstantValue::I32(v)) => assert_eq!(*v, -5),
        _ => panic!("expected -(5) to fold to the constant -5"),
    }
}

#[test]
fn unary_neg_of_runtime_value_does_not_fold() {
    let mut g = EGraph::new();
    let x = g.add_func_param(0, f32_ty()); // runtime
    let neg = g.intern_pure(PureOp::UnaryOp("-".into()), smallvec![x], f32_ty());
    let ENode::Pure { op, operands } = &g.nodes[neg] else {
        panic!("expected a pure node")
    };
    assert!(matches!(op, PureOp::UnaryOp(n) if n == "-"));
    assert_eq!(operands.as_slice(), &[x]);
}
