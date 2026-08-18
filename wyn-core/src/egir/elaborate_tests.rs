use super::*;
use crate::op;
use crate::ssa::types::{ConstantValue, Terminator};

#[test]
fn scalar_literals_stay_inline_in_ssa_operands() {
    let ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let mut graph = EGraph::<Physical>::new();
    let one = graph.intern_pure(PureOp::Uint("1".into()), SmallVec::new(), ty.clone(), None);
    let two = graph.intern_pure(PureOp::Uint("2".into()), SmallVec::new(), ty.clone(), None);
    let sum = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec::smallvec![one, two],
        ty.clone(),
        None,
    );
    let result = graph.value_result(sum);
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));

    let body = elaborate_one_body(graph, &Parameters::new(), ty);
    assert_eq!(
        body.num_insts(),
        1,
        "literal operands must not become SSA instructions"
    );
    let inst = body.inner.insts.values().next().expect("sum instruction");
    assert!(matches!(
        &inst.data,
        InstKind::Op {
            tag: OpTag::BinOp(op::BinaryOperator::Add),
            operands,
        } if operands == &vec![
            ValueRef::Const(ConstantValue::U32(1)),
            ValueRef::Const(ConstantValue::U32(2)),
        ]
    ));
}

#[test]
fn scalar_literals_stay_inline_in_ssa_terminators() {
    let ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let mut graph = EGraph::<Physical>::new();
    let seven = graph.intern_pure(PureOp::Uint("7".into()), SmallVec::new(), ty.clone(), None);
    let result = graph.value_result(seven);
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Return(Some(result));

    let body = elaborate_one_body(graph, &Parameters::new(), ty);
    assert_eq!(
        body.num_insts(),
        0,
        "literal return must not become an SSA instruction"
    );
    assert!(matches!(
        body.inner.blocks[body.inner.entry].term,
        Terminator::Return(Some(ValueRef::Const(ConstantValue::U32(7))))
    ));
}

#[test]
fn scalar_literals_stay_inline_in_ssa_block_arguments() {
    let ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let mut graph = EGraph::<Physical>::new();
    let target = graph.skeleton.create_block();
    let target_param = graph.add_block_param(target, ty.clone());
    let seven = graph.intern_pure(PureOp::Uint("7".into()), SmallVec::new(), ty.clone(), None);
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Branch {
        target,
        args: graph.admit_flow_values([seven]),
    };
    let result = graph.value_result(target_param);
    graph.skeleton.blocks[target].term = SkeletonTerminator::Return(Some(result));

    let body = elaborate_one_body(graph, &Parameters::new(), ty);
    assert_eq!(
        body.num_insts(),
        0,
        "literal block arguments must not become SSA instructions"
    );
    assert!(body.inner.blocks.values().any(|block| matches!(
        &block.term,
        Terminator::Branch { args, .. }
            if args == &[ValueRef::Const(ConstantValue::U32(7))]
    )));
}

#[test]
fn extraction_winners_follow_eliminated_block_parameter_aliases() {
    let ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let mut graph = EGraph::<Physical>::new();
    let merge = graph.skeleton.create_block();
    let selected = graph.add_block_param(merge, ty.clone());
    let replacement = graph.intern_constant(ConstantValue::U32(7), ty.clone());
    let call = graph.intern_pure(PureOp::Materialize, smallvec::smallvec![replacement], ty, None);
    graph.subsume_pure_in_place(call, selected);
    graph.nodes[selected].alias = Some(replacement);

    let mut best = extract::extract(&graph);
    close_extraction_over_aliases(&graph, &mut best);

    assert_eq!(best[&call], replacement);
    assert_eq!(best[&selected], replacement);
}
