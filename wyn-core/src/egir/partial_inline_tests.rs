use super::*;
use crate::op;
use crate::types;
use crate::BindingRef;
use crate::FunctionId;
use crate::IdSource;

use crate::ast::{Span, TypeName};
use crate::egir::types::{
    by_value_function_result, callable_parameter, CallEffects, OperandRef, Parameters, PureOp,
    SkeletonTerminator, WynLanguage,
};
use crate::flow::ControlHeader;
use crate::ssa::types::ConstantValue;
use polytype::Type;
use smallvec::smallvec;

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn fixed_u32_array_ty(size: usize) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            u32_ty(),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Constructed(TypeName::Size(size), vec![]),
            types::no_buffer(),
        ],
    )
}

fn physical_params(
    specs: impl IntoIterator<Item = (&'static str, Type<TypeName>)>,
) -> Parameters<BindingRef, Type<TypeName>> {
    specs
        .into_iter()
        .map(|(name, ty)| callable_parameter::<BindingRef, WynLanguage>(name.into(), ty))
        .collect()
}

fn inline_test_body(
    graph: &mut EGraph<Physical>,
    callees: &LookupMap<FunctionId, Func<Physical>>,
) -> Result<InliningStats, String> {
    inline_body(graph, callees, &mut IdSource::new())
}

fn add_value_call(
    graph: &mut EGraph<Physical>,
    callee: FunctionId,
    params: &Parameters<BindingRef, Type<TypeName>>,
    result_ty: Type<TypeName>,
    arguments: impl IntoIterator<Item = ValueId>,
) -> ValueId {
    let result = by_value_function_result::<WynLanguage>(result_ty);
    graph
        .emit_call(
            graph.skeleton.entry,
            callee,
            params,
            &result,
            arguments.into_iter().map(OperandRef::Value),
            CallEffects::Pure,
            None,
            None,
        )
        .expect("complete test call")
        .1
        .single_value()
        .expect("scalar test call result")
}

fn mixed_callee() -> Func<Physical> {
    let ty = u32_ty();
    let region = FunctionId::from_index(0);
    let params = physical_params([("varying", ty.clone()), ("invariant", ty.clone())]);
    let parameter_ids = params.ids().collect::<Vec<_>>();
    let mut graph = EGraph::<Physical>::new();
    let varying = graph.add_test_value_parameter(parameter_ids[0], ty.clone());
    let invariant = graph.add_test_value_parameter(parameter_ids[1], ty.clone());
    let invariant_square = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Multiply),
        smallvec![invariant, invariant],
        ty.clone(),
        None,
    );
    let result = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![varying, invariant_square],
        ty.clone(),
        None,
    );
    graph.skeleton.blocks[graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(graph.value_result(result)));
    Func::<Physical>::new(
        region,
        "mixed".into(),
        Span::dummy(),
        None,
        params,
        by_value_function_result::<WynLanguage>(ty),
        CallEffects::Pure,
        graph,
    )
}

fn mixed_callee_without_invariant_subexpression() -> Func<Physical> {
    let ty = u32_ty();
    let region = FunctionId::from_index(0);
    let params = physical_params([("varying", ty.clone()), ("invariant", ty.clone())]);
    let parameter_ids = params.ids().collect::<Vec<_>>();
    let mut graph = EGraph::<Physical>::new();
    let varying = graph.add_test_value_parameter(parameter_ids[0], ty.clone());
    let invariant = graph.add_test_value_parameter(parameter_ids[1], ty.clone());
    let result = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![varying, invariant],
        ty.clone(),
        None,
    );
    graph.skeleton.blocks[graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(graph.value_result(result)));
    Func::<Physical>::new(
        region,
        "mixed".into(),
        Span::dummy(),
        None,
        params,
        by_value_function_result::<WynLanguage>(ty),
        CallEffects::Pure,
        graph,
    )
}

#[derive(Clone, Copy)]
enum CallArgs {
    Mixed,
    AllInvariant,
    AllVarying,
}

fn loop_caller(shape: CallArgs) -> (EGraph<Physical>, ValueId, ValueId) {
    let ty = u32_ty();
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let mut graph = EGraph::<Physical>::new();
    let caller_params = physical_params([("invariant", ty.clone())]);
    let invariant = graph.add_test_value_parameter(caller_params.ids().next().unwrap(), ty.clone());
    let entry = graph.skeleton.entry;
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();

    let initial = graph.intern_constant(ConstantValue::U32(0), ty.clone());
    let initial_args = graph.admit_flow_values([initial]);
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: header,
        args: initial_args,
    };
    let current = graph.add_block_param(header, ty.clone());
    let keep_going = graph.intern_constant(ConstantValue::Bool(true), bool_ty);
    let exit_args = graph.admit_flow_values([current]);
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond: keep_going,
        then_target: body,
        then_args: vec![],
        else_target: merge,
        else_args: exit_args,
    };
    let literal = graph.intern_constant(ConstantValue::U32(7), ty.clone());
    let operands = match shape {
        CallArgs::Mixed => vec![current, invariant],
        CallArgs::AllInvariant => vec![invariant, literal],
        CallArgs::AllVarying => vec![current, current],
    };
    let params = physical_params([("varying", ty.clone()), ("invariant", ty.clone())]);
    let call = add_value_call(
        &mut graph,
        FunctionId::from_index(0),
        &params,
        ty.clone(),
        operands,
    );
    let call_args = graph.admit_flow_values([call]);
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: header,
        args: call_args,
    };
    let result = graph.add_block_param(merge, ty);
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(Some(graph.value_result(result)));

    graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
        merge,
        continue_block: body,
    });
    (graph, call, invariant)
}

#[test]
fn inlines_a_profitable_mixed_variance_call_in_a_loop() {
    let callee = mixed_callee();
    let callees = [(callee.region, callee)].into_iter().collect();
    let (mut graph, call, invariant) = loop_caller(CallArgs::Mixed);

    let stats = inline_test_body(&mut graph, &callees).unwrap();

    assert_eq!(stats.calls_inlined, 1);
    assert_ne!(graph.canonical_value(call), call);
    assert!(graph.nodes.values().any(|node| matches!(
        &node.kind,
        ValueKind::Pure {
            op: PureOp::BinOp(op::BinaryOperator::Multiply),
            operands
        } if operands.as_slice() == [invariant, invariant]
    )));
}

#[test]
fn mixed_variance_alone_is_enough_for_the_bounded_policy() {
    let callee = mixed_callee_without_invariant_subexpression();
    let callees = [(callee.region, callee)].into_iter().collect();
    let (mut graph, call, _) = loop_caller(CallArgs::Mixed);

    let stats = inline_test_body(&mut graph, &callees).unwrap();

    assert_eq!(stats.calls_inlined, 1);
    assert_ne!(graph.canonical_value(call), call);
}

#[test]
fn leaves_whole_call_licm_and_fully_varying_calls_alone() {
    let callee = mixed_callee();
    let callees = [(callee.region, callee)].into_iter().collect();

    for shape in [CallArgs::AllInvariant, CallArgs::AllVarying] {
        let (mut graph, call, _) = loop_caller(shape);
        let stats = inline_test_body(&mut graph, &callees).unwrap();
        assert_eq!(stats.calls_inlined, 0);
        assert!(matches!(graph.nodes[call].kind, ValueKind::CallResult { .. }));
    }
}

#[test]
fn inlines_fixed_array_parameters_outside_loops() {
    let scalar = u32_ty();
    let array = fixed_u32_array_ty(4);
    let region = FunctionId::from_index(0);
    let params = physical_params([("values", array.clone())]);
    let mut callee_graph = EGraph::<Physical>::new();
    let values = callee_graph.add_test_value_parameter(params.ids().next().unwrap(), array.clone());
    let zero = callee_graph.intern_constant(ConstantValue::I32(0), i32_ty());
    let result = callee_graph.intern_pure(PureOp::Index, smallvec![values, zero], scalar.clone(), None);
    callee_graph.skeleton.blocks[callee_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(callee_graph.value_result(result)));
    let callee = Func::<Physical>::new(
        region,
        "fixed_array_element".into(),
        Span::dummy(),
        None,
        params.clone(),
        by_value_function_result::<WynLanguage>(scalar.clone()),
        CallEffects::Pure,
        callee_graph,
    );
    let callees = [(region, callee)].into_iter().collect();

    let mut caller = EGraph::<Physical>::new();
    let caller_params = physical_params([("values", array.clone())]);
    let values = caller.add_test_value_parameter(caller_params.ids().next().unwrap(), array);
    let call = add_value_call(&mut caller, region, &params, scalar, [values]);
    caller.skeleton.blocks[caller.skeleton.entry].term =
        SkeletonTerminator::Return(Some(caller.value_result(call)));

    let stats = inline_test_body(&mut caller, &callees).unwrap();

    assert_eq!(stats.calls_inlined, 1);
    assert_ne!(caller.canonical_value(call), call);
}

#[test]
fn inlines_fixed_array_parameters_through_a_selection_cfg() {
    let scalar = u32_ty();
    let index_ty = i32_ty();
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let array = fixed_u32_array_ty(4);
    let region = FunctionId::from_index(0);
    let params = physical_params([("index", index_ty.clone()), ("values", array.clone())]);
    let parameter_ids = params.ids().collect::<Vec<_>>();
    let mut callee_graph = EGraph::<Physical>::new();
    let index = callee_graph.add_test_value_parameter(parameter_ids[0], index_ty.clone());
    let values = callee_graph.add_test_value_parameter(parameter_ids[1], array.clone());
    let materialized =
        callee_graph.intern_pure(PureOp::Materialize, smallvec![values], array.clone(), None);
    let element = callee_graph.intern_pure(
        PureOp::DynamicExtract,
        smallvec![materialized, index],
        scalar.clone(),
        None,
    );
    let entry = callee_graph.skeleton.entry;
    let left = callee_graph.skeleton.create_block();
    let right = callee_graph.skeleton.create_block();
    let merge = callee_graph.skeleton.create_block();
    let zero = callee_graph.intern_constant(ConstantValue::I32(0), index_ty.clone());
    let condition = callee_graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Equal),
        smallvec![index, zero],
        bool_ty,
        None,
    );
    callee_graph.skeleton.blocks[entry].term = SkeletonTerminator::CondBranch {
        cond: condition,
        then_target: left,
        then_args: vec![],
        else_target: right,
        else_args: vec![],
    };
    callee_graph.skeleton.blocks[entry].control_header = Some(ControlHeader::Selection { merge });
    let left_args = callee_graph.admit_flow_values([element]);
    callee_graph.skeleton.blocks[left].term = SkeletonTerminator::Branch {
        target: merge,
        args: left_args,
    };
    let one = callee_graph.intern_constant(ConstantValue::U32(1), scalar.clone());
    let incremented = callee_graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![element, one],
        scalar.clone(),
        None,
    );
    let right_args = callee_graph.admit_flow_values([incremented]);
    callee_graph.skeleton.blocks[right].term = SkeletonTerminator::Branch {
        target: merge,
        args: right_args,
    };
    let selected = callee_graph.add_block_param(merge, scalar.clone());
    callee_graph.skeleton.blocks[merge].term =
        SkeletonTerminator::Return(Some(callee_graph.value_result(selected)));
    let callee = Func::<Physical>::new(
        region,
        "conditional_fixed_array_element".into(),
        Span::dummy(),
        None,
        params.clone(),
        by_value_function_result::<WynLanguage>(scalar.clone()),
        CallEffects::Pure,
        callee_graph,
    );
    let callees = [(region, callee)].into_iter().collect();

    let mut caller = EGraph::<Physical>::new();
    let caller_params = physical_params([("index", index_ty.clone()), ("values", array.clone())]);
    let caller_parameter_ids = caller_params.ids().collect::<Vec<_>>();
    let actual_index = caller.add_test_value_parameter(caller_parameter_ids[0], index_ty);
    let actual_values = caller.add_test_value_parameter(caller_parameter_ids[1], array);
    let call = add_value_call(
        &mut caller,
        region,
        &params,
        scalar,
        [actual_index, actual_values],
    );
    caller.skeleton.blocks[caller.skeleton.entry].term =
        SkeletonTerminator::Return(Some(caller.value_result(call)));

    let stats = inline_test_body(&mut caller, &callees).unwrap();

    assert_eq!(stats.calls_inlined, 1);
    assert_eq!(stats.block_budget, 5);
    assert!(matches!(caller.nodes[call].kind, ValueKind::CallResult { .. }));
    assert!(caller.skeleton.blocks.values().any(|block| {
        matches!(
            &block.term,
            SkeletonTerminator::Return(Some(result)) if result.single_value() != Some(call)
        )
    }));
    assert!(caller
        .skeleton
        .blocks
        .values()
        .any(|block| matches!(block.control_header, Some(ControlHeader::Selection { .. }))));
    caller.skeleton.verify_branch_arities().expect("structured fixed-array inline arities");
}
