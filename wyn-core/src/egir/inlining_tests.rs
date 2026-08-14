use super::*;

use crate::ast::{Span, TypeName};
use crate::egir::graph_ops::bind_by_value_result;
use crate::egir::program::{SemanticFunc, SemanticResourceRef};
use crate::egir::types::{
    by_value_function_result, callable_parameter, CallEffects, EGraph, FuncParam, OperandRef,
    PureOp, ResultBinding, Semantic, SkeletonTerminator, ValueId, ValueKind,
    WynLanguage,
};
use crate::flow::ControlHeader;
use crate::ssa::types::ConstantValue;
use polytype::Type;
use smallvec::smallvec;

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn semantic_params(
    specs: impl IntoIterator<Item = (&'static str, Type<TypeName>)>,
) -> Vec<FuncParam<SemanticResourceRef, Type<TypeName>>> {
    specs
        .into_iter()
        .map(|(name, ty)| callable_parameter::<SemanticResourceRef, WynLanguage>(name.into(), ty))
        .collect()
}

fn add_call(
    graph: &mut EGraph<Semantic>,
    callee: crate::FunctionId,
    params: &[FuncParam<SemanticResourceRef, Type<TypeName>>],
    result_ty: Type<TypeName>,
    arguments: impl IntoIterator<Item = ValueId>,
) -> ResultBinding<Type<TypeName>> {
    graph
        .add_call(
            callee,
            params,
            &by_value_function_result::<WynLanguage>(result_ty),
            arguments.into_iter().map(OperandRef::Value),
            CallEffects::Pure,
            None,
        )
        .expect("complete test call")
        .1
}

#[test]
fn inline_pure_call_clones_the_callee_dag_with_parameter_substitution() {
    let ty = u32_ty();
    let region = crate::FunctionId::from_index(0);
    let mut callee_graph = EGraph::<Semantic>::new();
    let x = callee_graph.add_test_value_parameter(0, ty.clone());
    let invariant = callee_graph.add_test_value_parameter(1, ty.clone());
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
        SkeletonTerminator::Return(Some(callee_graph.value_result(result)));
    let params = semantic_params([("x", ty.clone()), ("invariant", ty.clone())]);
    let callee = SemanticFunc::new(
        region,
        "mixed".into(),
        Span::dummy(),
        None,
        params.clone(),
        by_value_function_result::<WynLanguage>(ty.clone()),
        CallEffects::Pure,
        callee_graph,
    );

    let mut caller = EGraph::<Semantic>::new();
    let actual_x = caller.add_test_value_parameter(0, ty.clone());
    let actual_invariant = caller.add_test_value_parameter(1, ty.clone());
    let call = add_call(
        &mut caller,
        region,
        &params,
        ty,
        [actual_x, actual_invariant],
    )
    .single_value()
    .unwrap();

    let inlined = inline_pure_call(&mut caller, call, &callee).expect("pure call inlines");

    assert!(matches!(
        caller.nodes[call].kind,
        ValueKind::Union {
            left,
            right
        } if left == inlined && right == inlined
    ));
    let ValueKind::Pure { op, operands } = &caller.nodes[inlined].kind else {
        panic!("inlined root is not pure")
    };
    assert!(matches!(op, PureOp::BinOp(crate::op::BinaryOperator::Add)));
    assert!(operands.contains(&actual_x));
    let cloned_square = operands.iter().copied().find(|operand| *operand != actual_x).unwrap();
    assert!(matches!(
        &caller.nodes[cloned_square].kind,
        ValueKind::Pure {
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
    let left = callee_graph.add_test_value_parameter(0, ty.clone());
    let right = callee_graph.add_test_value_parameter(1, ty.clone());
    let pair = callee_graph.intern_pure(PureOp::Tuple(2), smallvec![left, right], pair_ty, None);
    let selected =
        callee_graph.intern_pure(PureOp::Project { index: 1 }, smallvec![pair], ty.clone(), None);
    callee_graph.skeleton.blocks[callee_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(callee_graph.value_result(selected)));
    let params = semantic_params([("left", ty.clone()), ("right", ty.clone())]);
    let callee = SemanticFunc::new(
        region,
        "select_right".into(),
        Span::dummy(),
        None,
        params.clone(),
        by_value_function_result::<WynLanguage>(ty.clone()),
        CallEffects::Pure,
        callee_graph,
    );

    let mut caller = EGraph::<Semantic>::new();
    let two = caller.intern_constant(ConstantValue::U32(2), ty.clone());
    let seven = caller.intern_constant(ConstantValue::U32(7), ty.clone());
    let call = add_call(&mut caller, region, &params, ty, [two, seven])
        .single_value()
        .unwrap();

    let inlined = inline_pure_call(&mut caller, call, &callee).expect("pure call inlines");

    assert_eq!(
        inlined, seven,
        "aggregate construction and projection fold during substitution"
    );
    assert!(matches!(
        caller.nodes[call].kind,
        ValueKind::Union { left, right } if left == seven && right == seven
    ));
}

#[test]
fn inline_pure_call_replaces_every_leaf_of_a_product_result() {
    let ty = u32_ty();
    let pair_ty = Type::Constructed(TypeName::Tuple(2), vec![ty.clone(), ty.clone()]);
    let region = crate::FunctionId::from_index(0);
    let mut callee_graph = EGraph::<Semantic>::new();
    let left = callee_graph.add_test_value_parameter(0, ty.clone());
    let right = callee_graph.add_test_value_parameter(1, ty.clone());
    let pair = callee_graph.intern_pure(PureOp::Tuple(2), smallvec![left, right], pair_ty.clone(), None);
    let result_abi = by_value_function_result::<WynLanguage>(pair_ty.clone());
    let return_binding = bind_by_value_result(&mut callee_graph, &result_abi, pair);
    callee_graph.skeleton.blocks[callee_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(return_binding));
    let params = semantic_params([("left", ty.clone()), ("right", ty.clone())]);
    let callee = SemanticFunc::new(
        region,
        "make_pair".into(),
        Span::dummy(),
        None,
        params.clone(),
        result_abi,
        CallEffects::Pure,
        callee_graph,
    );

    let mut caller = EGraph::<Semantic>::new();
    let two = caller.intern_constant(ConstantValue::U32(2), ty.clone());
    let seven = caller.intern_constant(ConstantValue::U32(7), ty.clone());
    let call = add_call(&mut caller, region, &params, pair_ty, [two, seven]);
    let call_results = call.values();
    let left_result = call_results[0];
    let right_result = call_results[1];
    caller.skeleton.blocks[caller.skeleton.entry].term =
        SkeletonTerminator::Return(Some(caller.value_result(right_result)));

    inline_pure_call(&mut caller, left_result, &callee).expect("product call inlines");

    assert!(matches!(
        caller.nodes[right_result].kind,
        ValueKind::Union { left, right } if left == seven && right == seven
    ));
    assert!(caller.verify_hash_cons().is_ok());
}

#[test]
fn inline_call_at_block_splices_a_scalar_selection_cfg() {
    let ty = u32_ty();
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let region = crate::FunctionId::from_index(0);
    let mut callee_graph = EGraph::<Semantic>::new();
    let value = callee_graph.add_test_value_parameter(0, ty.clone());
    let choose_left = callee_graph.add_test_value_parameter(1, bool_ty.clone());
    let entry = callee_graph.skeleton.entry;
    let left = callee_graph.skeleton.create_block();
    let right = callee_graph.skeleton.create_block();
    let merge = callee_graph.skeleton.create_block();
    callee_graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond: choose_left,
        then_target: left,
        then_args: vec![],
        else_target: right,
        else_args: vec![],
    };
    callee_graph.skeleton.blocks[entry].control_header = Some(ControlHeader::Selection { merge });
    let one = callee_graph.intern_constant(ConstantValue::U32(1), ty.clone());
    let two = callee_graph.intern_constant(ConstantValue::U32(2), ty.clone());
    let left_value = callee_graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![value, one],
        ty.clone(),
        None,
    );
    let right_value = callee_graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![value, two],
        ty.clone(),
        None,
    );
    let left_args = callee_graph.admit_flow_values([left_value]);
    callee_graph.skeleton.blocks[left].term = SkeletonTerminator::Branch {
        target: merge,
        args: left_args,
    };
    let right_args = callee_graph.admit_flow_values([right_value]);
    callee_graph.skeleton.blocks[right].term = SkeletonTerminator::Branch {
        target: merge,
        args: right_args,
    };
    let selected = callee_graph.add_block_param(merge, ty.clone());
    callee_graph.skeleton.blocks[merge].term =
        SkeletonTerminator::Return(Some(callee_graph.value_result(selected)));
    let params = semantic_params([
        ("value", ty.clone()),
        ("choose_left", bool_ty.clone()),
    ]);
    let callee = SemanticFunc::new(
        region,
        "choose_offset".into(),
        Span::dummy(),
        None,
        params.clone(),
        by_value_function_result::<WynLanguage>(ty.clone()),
        CallEffects::Pure,
        callee_graph,
    );

    let mut caller = EGraph::<Semantic>::new();
    let actual = caller.add_test_value_parameter(0, ty.clone());
    let condition = caller.add_test_value_parameter(1, bool_ty);
    let call = add_call(&mut caller, region, &params, ty.clone(), [actual, condition])
        .single_value()
        .unwrap();
    let three = caller.intern_constant(ConstantValue::U32(3), ty.clone());
    let final_value = caller.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Multiply),
        smallvec![call, three],
        ty,
        None,
    );
    let caller_entry = caller.skeleton.entry;
    caller.skeleton.blocks[caller_entry].term =
        SkeletonTerminator::Return(Some(caller.value_result(final_value)));

    let inlined =
        inline_call_at_block(&mut caller, call, caller_entry, &callee).expect("selection CFG inlines");

    assert!(matches!(
        caller.nodes[call].kind,
        ValueKind::CallResult { .. }
    ));
    assert!(matches!(
        &caller.nodes[final_value].kind,
        ValueKind::Pure { operands, .. } if operands[0] == inlined
    ));
    assert!(caller
        .skeleton
        .blocks
        .values()
        .any(|block| matches!(block.control_header, Some(ControlHeader::Selection { .. }))));
    assert!(matches!(
        caller.skeleton.blocks[caller_entry].term,
        SkeletonTerminator::Branch { .. }
    ));
    assert_eq!(
        caller
            .skeleton
            .blocks
            .values()
            .filter(|block| matches!(block.term, SkeletonTerminator::Return(Some(_))))
            .count(),
        1
    );
    caller.skeleton.verify_branch_arities().expect("inlined CFG branch arities");
    assert!(caller.verify_hash_cons().is_ok());
}
