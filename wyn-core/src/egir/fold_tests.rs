//! Tests for algebraic folds applied during `EGraph::intern_pure`.

use super::super::types::{EGraph, ENode, NodeId, PureOp};
use crate::ast::TypeName;
use crate::ssa::types::ConstantValue;
use polytype::Type;
use smallvec::smallvec;

fn f32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
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
