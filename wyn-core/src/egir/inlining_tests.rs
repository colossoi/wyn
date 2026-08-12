use super::*;

use crate::ast::{Span, TypeName};
use crate::egir::program::SemanticFunc;
use crate::egir::types::{EGraph, ENode, PureOp, Semantic, SkeletonTerminator};
use crate::ssa::types::ConstantValue;
use polytype::Type;
use smallvec::smallvec;

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

#[test]
fn inline_pure_call_clones_the_callee_dag_with_parameter_substitution() {
    let ty = u32_ty();
    let region = crate::FunctionId::from_index(0);
    let mut callee_graph = EGraph::<Semantic>::new();
    let x = callee_graph.add_func_param(0, ty.clone());
    let invariant = callee_graph.add_func_param(1, ty.clone());
    let square = callee_graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Multiply),
        smallvec![invariant, invariant],
        ty.clone(),
        None,
    );
    let result = callee_graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![x, square],
        ty.clone(),
        None,
    );
    callee_graph.skeleton.blocks[callee_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(result));
    let callee = SemanticFunc::new(
        region,
        "mixed".into(),
        Span::dummy(),
        None,
        vec![(ty.clone(), "x".into()), (ty.clone(), "invariant".into())],
        ty.clone(),
        callee_graph,
    );

    let mut caller = EGraph::<Semantic>::new();
    let actual_x = caller.add_func_param(0, ty.clone());
    let actual_invariant = caller.add_func_param(1, ty.clone());
    let call = caller.intern_pure(
        PureOp::Call(crate::FunctionId::from_index(0)),
        smallvec![actual_x, actual_invariant],
        ty,
        None,
    );

    let inlined = inline_pure_call(&mut caller, call, &callee).expect("pure call inlines");

    assert!(matches!(
        caller.nodes[call].kind,
        ENode::Union {
            left,
            right
        } if left == inlined && right == inlined
    ));
    let ENode::Pure { op, operands } = &caller.nodes[inlined].kind else {
        panic!("inlined root is not pure")
    };
    assert!(matches!(op, PureOp::BinOp(crate::op::BinaryOperator::Add)));
    assert!(operands.contains(&actual_x));
    let cloned_square = operands.iter().copied().find(|operand| *operand != actual_x).unwrap();
    assert!(matches!(
        &caller.nodes[cloned_square].kind,
        ENode::Pure {
            op: PureOp::BinOp(crate::op::BinaryOperator::Multiply),
            operands
        } if operands.as_slice() == [actual_invariant, actual_invariant]
    ));
    assert!(caller.verify_hash_cons().is_ok());
}

#[test]
fn inline_pure_call_folds_projection_of_substituted_aggregate() {
    let ty = u32_ty();
    let pair_ty = Type::Constructed(TypeName::Tuple(2), vec![ty.clone(), ty.clone()]);
    let region = crate::FunctionId::from_index(0);
    let mut callee_graph = EGraph::<Semantic>::new();
    let left = callee_graph.add_func_param(0, ty.clone());
    let right = callee_graph.add_func_param(1, ty.clone());
    let pair = callee_graph.intern_pure(PureOp::Tuple(2), smallvec![left, right], pair_ty, None);
    let selected =
        callee_graph.intern_pure(PureOp::Project { index: 1 }, smallvec![pair], ty.clone(), None);
    callee_graph.skeleton.blocks[callee_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(selected));
    let callee = SemanticFunc::new(
        region,
        "select_right".into(),
        Span::dummy(),
        None,
        vec![(ty.clone(), "left".into()), (ty.clone(), "right".into())],
        ty.clone(),
        callee_graph,
    );

    let mut caller = EGraph::<Semantic>::new();
    let two = caller.intern_constant(ConstantValue::U32(2), ty.clone());
    let seven = caller.intern_constant(ConstantValue::U32(7), ty.clone());
    let call = caller.intern_pure(PureOp::Call(region), smallvec![two, seven], ty, None);

    let inlined = inline_pure_call(&mut caller, call, &callee).expect("pure call inlines");

    assert_eq!(
        inlined, seven,
        "aggregate construction and projection fold during substitution"
    );
    assert!(matches!(
        caller.nodes[call].kind,
        ENode::Union { left, right } if left == seven && right == seven
    ));
}
