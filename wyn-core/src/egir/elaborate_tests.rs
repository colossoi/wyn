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
    let call = graph.intern_pure(
        PureOp::Materialize,
        smallvec::smallvec![replacement],
        ty.clone(),
        None,
    );
    graph.subsume_pure_in_place(call, selected);
    graph.skeleton.blocks[graph.skeleton.entry].term = SkeletonTerminator::Branch {
        target: merge,
        args: graph.admit_flow_values([replacement]),
    };
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(Some(graph.value_result(call)));
    let aliases = super::super::skel_opt::run_one_body(&mut graph);
    graph.install_aliases(aliases);

    let mut best = extract::extract(&graph);
    close_extraction_over_aliases(&graph, &mut best);

    assert_eq!(best[&call], replacement);
    assert_eq!(best[&selected], replacement);
    assert_eq!(best[&replacement], replacement);

    let body = elaborate_one_body(graph, &Parameters::new(), ty);
    assert!(body.inner.blocks.values().any(|block| matches!(
        block.term,
        Terminator::Return(Some(ValueRef::Const(ConstantValue::U32(7))))
    )));
}

fn composition_graph<const N: usize>() -> (EGraph<Physical>, [ValueId; N]) {
    let mut graph = EGraph::new();
    let ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let nodes = std::array::from_fn(|_| graph.add_block_param(graph.skeleton.entry, ty.clone()));
    (graph, nodes)
}

#[test]
fn extraction_and_alias_edges_alternate_to_a_terminal() {
    let (mut graph, [a, b, c, d, terminal, unrelated]) = composition_graph();
    graph.install_aliases(LookupMap::from([(b, c), (d, terminal)]));
    let mut best = LookupMap::from([(a, b), (b, b), (c, d)]);

    close_extraction_over_aliases(&graph, &mut best);

    assert_eq!(best.len(), graph.nodes.len());
    for node in [a, b, c, d, terminal] {
        assert_eq!(best[&node], terminal);
    }
    assert_eq!(best[&unrelated], unrelated);
}

#[test]
fn extraction_winner_takes_precedence_over_an_alias() {
    let (mut graph, [source, winner, aliased]) = composition_graph();
    graph.install_aliases(LookupMap::from([(source, aliased)]));
    let mut best = LookupMap::from([(source, winner), (aliased, source)]);

    close_extraction_over_aliases(&graph, &mut best);

    assert_eq!(
        best,
        LookupMap::from([(source, winner), (winner, winner), (aliased, winner)])
    );
}

#[test]
fn composition_cycle_panics_before_updating_extraction() {
    let (mut graph, [source, selected]) = composition_graph();
    graph.install_aliases(LookupMap::from([(selected, source)]));
    let mut best = LookupMap::from([(source, selected), (selected, selected)]);
    let before = best.clone();

    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        close_extraction_over_aliases(&graph, &mut best);
    }))
    .expect_err("the composed cycle must fail");
    let message = error.downcast_ref::<String>().expect("replacement diagnostic");
    assert!(message.contains("invalid extraction/CFG alias composition"));
    assert!(message.contains("cycle"));
    assert!(message.contains(&format!("{source:?}")));
    assert!(message.contains(&format!("{selected:?}")));
    assert_eq!(best, before);
}
