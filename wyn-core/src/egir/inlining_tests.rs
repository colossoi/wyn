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
    let mut callee_graph = EGraph::<Semantic>::new();
    let x = callee_graph.add_func_param(0, ty.clone());
    let invariant = callee_graph.add_func_param(1, ty.clone());
    let square = callee_graph.intern_pure(
        PureOp::BinOp("*".into()),
        smallvec![invariant, invariant],
        ty.clone(),
        None,
    );
    let result =
        callee_graph.intern_pure(PureOp::BinOp("+".into()), smallvec![x, square], ty.clone(), None);
    callee_graph.skeleton.blocks[callee_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(result));
    let callee = SemanticFunc::new(
        "mixed".into(),
        Span::dummy(),
        None,
        vec![(ty.clone(), "x".into()), (ty.clone(), "invariant".into())],
        ty.clone(),
        callee_graph,
        crate::LookupMap::new(),
    );

    let mut caller = EGraph::<Semantic>::new();
    let two = caller.intern_constant(ConstantValue::U32(2), ty.clone());
    let seven = caller.intern_constant(ConstantValue::U32(7), ty.clone());
    let call = caller.intern_pure(PureOp::Call("mixed".into()), smallvec![two, seven], ty, None);

    let inlined = inline_pure_call(&mut caller, call, &callee).expect("pure call inlines");

    assert!(matches!(
        caller.nodes[call],
        ENode::Union {
            left,
            right
        } if left == inlined && right == inlined
    ));
    let ENode::Pure { op, operands } = &caller.nodes[inlined] else {
        panic!("inlined root is not pure")
    };
    assert!(matches!(op, PureOp::BinOp(name) if name == "+"));
    assert!(operands.contains(&two));
    let cloned_square = operands.iter().copied().find(|operand| *operand != two).unwrap();
    assert!(matches!(
        &caller.nodes[cloned_square],
        ENode::Pure {
            op: PureOp::BinOp(name),
            operands
        } if name == "*" && operands.as_slice() == [seven, seven]
    ));
    assert!(caller.verify_hash_cons().is_ok());
}
