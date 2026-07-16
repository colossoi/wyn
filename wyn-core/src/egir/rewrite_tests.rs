//! Tests for cost-arbitrated rewrite rules and their union resolution.

use super::super::extract;
use super::super::types::{EGraph as GenericEGraph, ENode, NodeId, PureOp};
use super::default_rewrites;
use crate::ast::TypeName;
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

fn vec3f32_ty() -> Type<TypeName> {
    crate::types::vec(3, f32_ty())
}

fn pow(g: &mut EGraph, base: NodeId, exp: ConstantValue, exp_ty: Type<TypeName>) -> NodeId {
    let result_ty = g.types[&base].clone();
    let exp = g.intern_constant(exp, exp_ty);
    g.intern_pure(PureOp::BinOp("**".into()), smallvec![base, exp], result_ty)
}

/// Apply the default rules to `node` (asserting one fired) and return
/// extraction's pick for it.
fn apply_and_extract(g: &mut EGraph, node: NodeId) -> NodeId {
    assert!(
        default_rewrites().apply_to_node(g, node),
        "expected a rewrite to fire"
    );
    assert!(
        matches!(g.nodes[node], ENode::Union { .. }),
        "expected the node to become a union in place"
    );
    let best = extract::extract(g);
    best.get(&node).copied().unwrap_or(node)
}

/// Count chained multiplies starting at `nid`. Returns `Some(n)` if the
/// node is `x * x * … * x` (left-to-right) with `n` total mul ops over
/// the same base, `None` otherwise.
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

fn is_pow(graph: &EGraph, nid: NodeId) -> bool {
    matches!(
        &graph.nodes[nid],
        ENode::Pure { op: PureOp::BinOp(name), .. } if name == "**"
    )
}

#[test]
fn pow_const_2_extracts_one_mul_f32() {
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let p = pow(&mut g, base, ConstantValue::I32(2), i32_ty());
    let winner = apply_and_extract(&mut g, p);
    assert_eq!(chain_len_over_same_base(&g, winner, base), Some(1));
}

#[test]
fn pow_const_5_extracts_four_muls_f32() {
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let p = pow(&mut g, base, ConstantValue::I32(5), i32_ty());
    let winner = apply_and_extract(&mut g, p);
    assert_eq!(chain_len_over_same_base(&g, winner, base), Some(4));
}

#[test]
fn pow_const_7_extracts_six_muls_i32() {
    let mut g = EGraph::new();
    let base = g.add_func_param(0, i32_ty());
    let p = pow(&mut g, base, ConstantValue::I32(7), i32_ty());
    let winner = apply_and_extract(&mut g, p);
    assert_eq!(chain_len_over_same_base(&g, winner, base), Some(6));
}

#[test]
fn pow_const_9_extracts_pow() {
    // 8 multiplies tie the modeled Pow cost; ties keep the original.
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let p = pow(&mut g, base, ConstantValue::I32(9), i32_ty());
    let winner = apply_and_extract(&mut g, p);
    assert!(is_pow(&g, winner));
}

#[test]
fn pow_const_17_does_not_rewrite() {
    // Beyond MAX_CHAIN no alternative is proposed at all.
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let p = pow(&mut g, base, ConstantValue::I32(17), i32_ty());
    assert!(!default_rewrites().apply_to_node(&mut g, p));
    assert!(is_pow(&g, p));
}

#[test]
fn pow_const_1_does_not_rewrite() {
    // 1 is below the rule's window. A separate identity rule could strip
    // it later; this one leaves it alone.
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let p = pow(&mut g, base, ConstantValue::I32(1), i32_ty());
    assert!(!default_rewrites().apply_to_node(&mut g, p));
    assert!(is_pow(&g, p));
}

#[test]
fn pow_f32_whole_exponent_extracts_chain() {
    // Float `**` requires same-typed operands; the exponent arrives as a
    // f32 literal even when the user wrote `x ** 3`.
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let p = pow(&mut g, base, ConstantValue::F32(3.0f32.to_bits()), f32_ty());
    let winner = apply_and_extract(&mut g, p);
    assert_eq!(chain_len_over_same_base(&g, winner, base), Some(2));
}

#[test]
fn pow_f32_fractional_exponent_does_not_rewrite() {
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let p = pow(&mut g, base, ConstantValue::F32(2.5f32.to_bits()), f32_ty());
    assert!(!default_rewrites().apply_to_node(&mut g, p));
    assert!(is_pow(&g, p));
}

#[test]
fn pow_non_const_exponent_does_not_rewrite() {
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let exp = g.add_func_param(1, i32_ty()); // runtime
    let p = g.intern_pure(PureOp::BinOp("**".into()), smallvec![base, exp], f32_ty());
    assert!(!default_rewrites().apply_to_node(&mut g, p));
    assert!(is_pow(&g, p));
}

#[test]
fn pow_vec_base_does_not_rewrite() {
    // No componentwise chain for `vec ** k`; the node must stay `**` so
    // the backend reports the missing lowering.
    let mut g = EGraph::new();
    let base = g.add_func_param(0, vec3f32_ty());
    let p = pow(&mut g, base, ConstantValue::I32(2), i32_ty());
    assert!(!default_rewrites().apply_to_node(&mut g, p));
    assert!(is_pow(&g, p));
}

#[test]
fn pow_of_expensive_shared_base_still_extracts_chain() {
    // The base is shared by both union sides, so its cost must cancel in
    // extraction's comparison. A plain subtree-sum DP would count the base
    // twice for `x * x` but once for `x ** 2` and wrongly keep the Pow.
    let mut g = EGraph::new();
    let mut base = g.add_func_param(0, f32_ty());
    for i in 0..10 {
        let c = g.intern_constant(ConstantValue::from_f32(1.5 + i as f32), f32_ty());
        base = g.intern_pure(PureOp::BinOp("+".into()), smallvec![base, c], f32_ty());
    }
    let p = pow(&mut g, base, ConstantValue::I32(2), i32_ty());
    let winner = apply_and_extract(&mut g, p);
    assert_eq!(chain_len_over_same_base(&g, winner, base), Some(1));
}

#[test]
fn consumers_see_the_rewrite_through_the_original_id() {
    let mut g = EGraph::new();
    let base = g.add_func_param(0, f32_ty());
    let p = pow(&mut g, base, ConstantValue::I32(2), i32_ty());
    let one = g.intern_constant(ConstantValue::from_f32(1.0), f32_ty());
    let consumer = g.intern_pure(PureOp::BinOp("+".into()), smallvec![p, one], f32_ty());

    let winner = apply_and_extract(&mut g, p);

    // The consumer still references the original id, which extraction now
    // resolves to the chain.
    let ENode::Pure { operands, .. } = &g.nodes[consumer] else {
        panic!("expected pure consumer")
    };
    assert_eq!(operands[0], p);
    assert_eq!(chain_len_over_same_base(&g, winner, base), Some(1));
}
