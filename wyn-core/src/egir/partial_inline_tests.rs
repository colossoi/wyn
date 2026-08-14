use super::*;

use crate::ast::{Span, TypeName};
use crate::egir::types::SkeletonTerminator;
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
            crate::types::no_buffer(),
        ],
    )
}

fn mixed_callee() -> PhysicalFunc {
    let ty = u32_ty();
    let region = crate::FunctionId::from_index(0);
    let mut graph = EGraph::<Physical>::new();
    let varying = graph.add_func_param(0, ty.clone());
    let invariant = graph.add_func_param(1, ty.clone());
    let invariant_square = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Multiply),
        smallvec![invariant, invariant],
        ty.clone(),
        None,
    );
    let result = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![varying, invariant_square],
        ty.clone(),
        None,
    );
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
    PhysicalFunc::new(
        region,
        "mixed".into(),
        Span::dummy(),
        None,
        vec![(ty.clone(), "varying".into()), (ty.clone(), "invariant".into())],
        ty,
        graph,
    )
}

fn mixed_callee_without_invariant_subexpression() -> PhysicalFunc {
    let ty = u32_ty();
    let region = crate::FunctionId::from_index(0);
    let mut graph = EGraph::<Physical>::new();
    let varying = graph.add_func_param(0, ty.clone());
    let invariant = graph.add_func_param(1, ty.clone());
    let result = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![varying, invariant],
        ty.clone(),
        None,
    );
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
    PhysicalFunc::new(
        region,
        "mixed".into(),
        Span::dummy(),
        None,
        vec![(ty.clone(), "varying".into()), (ty.clone(), "invariant".into())],
        ty,
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
    let invariant = graph.add_func_param(0, ty.clone());
    let entry = graph.skeleton.entry;
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();

    let initial = graph.intern_constant(ConstantValue::U32(0), ty.clone());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![initial],
    };
    let current = graph.add_block_param(header, ty.clone());
    let keep_going = graph.intern_constant(ConstantValue::Bool(true), bool_ty);
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond: keep_going,
        then_target: body,
        then_args: vec![],
        else_target: merge,
        else_args: vec![current],
    };
    let literal = graph.intern_constant(ConstantValue::U32(7), ty.clone());
    let operands = match shape {
        CallArgs::Mixed => smallvec![current, invariant],
        CallArgs::AllInvariant => smallvec![invariant, literal],
        CallArgs::AllVarying => smallvec![current, current],
    };
    let call = graph.intern_pure(
        PureOp::Call(crate::FunctionId::from_index(0)),
        operands,
        ty.clone(),
        None,
    );
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![call],
    };
    let result = graph.add_block_param(merge, ty);
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(Some(result));

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

    let stats = inline_body(&mut graph, &callees).unwrap();

    assert_eq!(stats.calls_inlined, 1);
    assert!(matches!(graph.nodes[call].kind, ValueKind::Union { .. }));
    assert!(graph.nodes.values().any(|node| matches!(
        &node.kind,
        ValueKind::Pure {
            op: PureOp::BinOp(crate::op::BinaryOperator::Multiply),
            operands
        } if operands.as_slice() == [invariant, invariant]
    )));
}

#[test]
fn mixed_variance_alone_is_enough_for_the_bounded_policy() {
    let callee = mixed_callee_without_invariant_subexpression();
    let callees = [(callee.region, callee)].into_iter().collect();
    let (mut graph, call, _) = loop_caller(CallArgs::Mixed);

    let stats = inline_body(&mut graph, &callees).unwrap();

    assert_eq!(stats.calls_inlined, 1);
    assert!(matches!(graph.nodes[call].kind, ValueKind::Union { .. }));
}

#[test]
fn leaves_whole_call_licm_and_fully_varying_calls_alone() {
    let callee = mixed_callee();
    let callees = [(callee.region, callee)].into_iter().collect();

    for shape in [CallArgs::AllInvariant, CallArgs::AllVarying] {
        let (mut graph, call, _) = loop_caller(shape);
        let stats = inline_body(&mut graph, &callees).unwrap();
        assert_eq!(stats.calls_inlined, 0);
        assert!(matches!(
            graph.nodes[call].kind,
            ValueKind::Pure {
                op: PureOp::Call(_),
                ..
            }
        ));
    }
}

#[test]
fn inlines_fixed_array_parameters_outside_loops() {
    let scalar = u32_ty();
    let array = fixed_u32_array_ty(4);
    let region = crate::FunctionId::from_index(0);
    let mut callee_graph = EGraph::<Physical>::new();
    let values = callee_graph.add_func_param(0, array.clone());
    let zero = callee_graph.intern_constant(ConstantValue::I32(0), i32_ty());
    let result = callee_graph.intern_pure(PureOp::Index, smallvec![values, zero], scalar.clone(), None);
    callee_graph.skeleton.blocks[callee_graph.skeleton.entry].term =
        SkeletonTerminator::Return(Some(result));
    let callee = PhysicalFunc::new(
        region,
        "fixed_array_element".into(),
        Span::dummy(),
        None,
        vec![(array.clone(), "values".into())],
        scalar.clone(),
        callee_graph,
    );
    let callees = [(region, callee)].into_iter().collect();

    let mut caller = EGraph::<Physical>::new();
    let values = caller.add_func_param(0, array);
    let call = caller.intern_pure(PureOp::Call(region), smallvec![values], scalar, None);
    caller.skeleton.blocks[caller.skeleton.entry].term = SkeletonTerminator::Return(Some(call));

    let stats = inline_body(&mut caller, &callees).unwrap();

    assert_eq!(stats.calls_inlined, 1);
    assert!(matches!(caller.nodes[call].kind, ValueKind::Union { .. }));
}

#[test]
fn inlines_fixed_array_parameters_through_a_selection_cfg() {
    let scalar = u32_ty();
    let index_ty = i32_ty();
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let array = fixed_u32_array_ty(4);
    let region = crate::FunctionId::from_index(0);
    let mut callee_graph = EGraph::<Physical>::new();
    let index = callee_graph.add_func_param(0, index_ty.clone());
    let values = callee_graph.add_func_param(1, array.clone());
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
        PureOp::BinOp(crate::op::BinaryOperator::Equal),
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
    callee_graph.skeleton.blocks[left].term = SkeletonTerminator::Branch {
        target: merge,
        args: vec![element],
    };
    let one = callee_graph.intern_constant(ConstantValue::U32(1), scalar.clone());
    let incremented = callee_graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![element, one],
        scalar.clone(),
        None,
    );
    callee_graph.skeleton.blocks[right].term = SkeletonTerminator::Branch {
        target: merge,
        args: vec![incremented],
    };
    let selected = callee_graph.add_block_param(merge, scalar.clone());
    callee_graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(Some(selected));
    let callee = PhysicalFunc::new(
        region,
        "conditional_fixed_array_element".into(),
        Span::dummy(),
        None,
        vec![
            (index_ty.clone(), "index".into()),
            (array.clone(), "values".into()),
        ],
        scalar.clone(),
        callee_graph,
    );
    let callees = [(region, callee)].into_iter().collect();

    let mut caller = EGraph::<Physical>::new();
    let actual_index = caller.add_func_param(0, index_ty);
    let actual_values = caller.add_func_param(1, array);
    let call = caller.intern_pure(
        PureOp::Call(region),
        smallvec![actual_index, actual_values],
        scalar,
        None,
    );
    caller.skeleton.blocks[caller.skeleton.entry].term = SkeletonTerminator::Return(Some(call));

    let stats = inline_body(&mut caller, &callees).unwrap();

    assert_eq!(stats.calls_inlined, 1);
    assert_eq!(stats.block_budget, 5);
    assert!(matches!(
        caller.nodes[call].kind,
        ValueKind::Pure {
            op: PureOp::Call(_),
            ..
        }
    ));
    assert!(caller
        .skeleton
        .blocks
        .values()
        .any(|block| { matches!(block.term, SkeletonTerminator::Return(Some(result)) if result != call) }));
    assert!(caller
        .skeleton
        .blocks
        .values()
        .any(|block| matches!(block.control_header, Some(ControlHeader::Selection { .. }))));
    caller.skeleton.verify_branch_arities().expect("structured fixed-array inline arities");
}
