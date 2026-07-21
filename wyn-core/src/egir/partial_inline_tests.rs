use super::*;

use crate::ast::{Span, TypeName};
use crate::egir::types::SkeletonTerminator;
use crate::ssa::types::ConstantValue;
use polytype::Type;
use smallvec::smallvec;

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn mixed_callee() -> PhysicalFunc {
    let ty = u32_ty();
    let mut graph = EGraph::<Physical>::new();
    let varying = graph.add_func_param(0, ty.clone());
    let invariant = graph.add_func_param(1, ty.clone());
    let invariant_square = graph.intern_pure(
        PureOp::BinOp("*".into()),
        smallvec![invariant, invariant],
        ty.clone(),
        None,
    );
    let result = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![varying, invariant_square],
        ty.clone(),
        None,
    );
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
    PhysicalFunc::new(
        "mixed".into(),
        Span::dummy(),
        None,
        vec![(ty.clone(), "varying".into()), (ty.clone(), "invariant".into())],
        ty,
        graph,
        LookupMap::new(),
    )
}

fn mixed_callee_without_invariant_subexpression() -> PhysicalFunc {
    let ty = u32_ty();
    let mut graph = EGraph::<Physical>::new();
    let varying = graph.add_func_param(0, ty.clone());
    let invariant = graph.add_func_param(1, ty.clone());
    let result = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![varying, invariant],
        ty.clone(),
        None,
    );
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));
    PhysicalFunc::new(
        "mixed".into(),
        Span::dummy(),
        None,
        vec![(ty.clone(), "varying".into()), (ty.clone(), "invariant".into())],
        ty,
        graph,
        LookupMap::new(),
    )
}

#[derive(Clone, Copy)]
enum CallArgs {
    Mixed,
    AllInvariant,
    AllVarying,
}

fn loop_caller(
    shape: CallArgs,
) -> (
    EGraph<Physical>,
    LookupMap<BlockId, ControlHeader>,
    NodeId,
    NodeId,
) {
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
    let call = graph.intern_pure(PureOp::Call("mixed".into()), operands, ty.clone(), None);
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![call],
    };
    let result = graph.add_block_param(merge, ty);
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(Some(result));

    let headers = [(
        header,
        ControlHeader::Loop {
            merge,
            continue_block: body,
        },
    )]
    .into_iter()
    .collect();
    (graph, headers, call, invariant)
}

#[test]
fn inlines_a_profitable_mixed_variance_call_in_a_loop() {
    let callee = mixed_callee();
    let mut regions = RegionInterner::new();
    let callee_id = regions.intern(&callee.name);
    let callees = [(callee_id, callee)].into_iter().collect();
    let (mut graph, headers, call, invariant) = loop_caller(CallArgs::Mixed);

    let stats = inline_body(&mut graph, &headers, &regions, &callees).unwrap();

    assert_eq!(stats.calls_inlined, 1);
    assert!(matches!(graph.nodes[call], ENode::Union { .. }));
    assert!(graph.nodes.values().any(|node| matches!(
        node,
        ENode::Pure {
            op: PureOp::BinOp(name),
            operands
        } if name == "*" && operands.as_slice() == [invariant, invariant]
    )));
}

#[test]
fn mixed_variance_alone_is_enough_for_the_bounded_policy() {
    let callee = mixed_callee_without_invariant_subexpression();
    let mut regions = RegionInterner::new();
    let callee_id = regions.intern(&callee.name);
    let callees = [(callee_id, callee)].into_iter().collect();
    let (mut graph, headers, call, _) = loop_caller(CallArgs::Mixed);

    let stats = inline_body(&mut graph, &headers, &regions, &callees).unwrap();

    assert_eq!(stats.calls_inlined, 1);
    assert!(matches!(graph.nodes[call], ENode::Union { .. }));
}

#[test]
fn leaves_whole_call_licm_and_fully_varying_calls_alone() {
    let callee = mixed_callee();
    let mut regions = RegionInterner::new();
    let callee_id = regions.intern(&callee.name);
    let callees = [(callee_id, callee)].into_iter().collect();

    for shape in [CallArgs::AllInvariant, CallArgs::AllVarying] {
        let (mut graph, headers, call, _) = loop_caller(shape);
        let stats = inline_body(&mut graph, &headers, &regions, &callees).unwrap();
        assert_eq!(stats.calls_inlined, 0);
        assert!(matches!(
            graph.nodes[call],
            ENode::Pure {
                op: PureOp::Call(_),
                ..
            }
        ));
    }
}
