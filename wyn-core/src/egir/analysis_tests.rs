use super::*;
use crate::ast::TypeName;
use crate::egir::graph_ops::value_producer_closure;
use crate::egir::loop_analysis::LoopInvariance;
use crate::egir::types::{
    EffectOp, OperandRef, PureOp, Semantic, SideEffect, SideEffectKind, SideEffectSite, SkeletonTerminator,
};
use crate::flow::ControlHeader;
use crate::ssa::types::ConstantValue;
use crate::LookupSet;
use polytype::Type;
use smallvec::smallvec;

#[test]
fn independent_selections_and_rebuilt_producers_follow_effect_rewrites() {
    let mut graph = EGraph::<Semantic>::new();
    let block = graph.skeleton.entry;
    let ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let zero = graph.intern_constant(ConstantValue::U32(0), ty.clone());
    let one = graph.intern_constant(ConstantValue::U32(1), ty.clone());
    let first = graph.alloc_side_effect_result(ty.clone());
    let second = graph.alloc_side_effect_result(ty);
    for (result, input) in [(first, zero), (second, one)] {
        let result = graph.value_result(result);
        graph.skeleton.blocks[block].side_effects.push(SideEffect {
            kind: SideEffectKind::Effect(EffectOp::Op {
                tag: PureOp::Materialize,
            }),
            operands: smallvec![OperandRef::Value(input)],
            result: Some(result),
            effects: None,
            span: None,
        });
    }
    {
        let facts = GraphAnalysis::new(&graph);
        let a = value_producer_closure(&facts, [first]);
        let b = value_producer_closure(&facts, [second]);
        assert_eq!(a.values(), &[first, zero].into_iter().collect());
        assert_eq!(b.values(), &[second, one].into_iter().collect());
        assert_eq!(
            facts.producers().site(first),
            Some(SideEffectSite { block, index: 0 })
        );
        assert_eq!(
            facts.producers().site(second),
            Some(SideEffectSite { block, index: 1 })
        );
    }
    graph.skeleton.blocks[block].side_effects.swap(0, 1);
    graph.skeleton.blocks[block].side_effects[1].operands = smallvec![OperandRef::Value(one)];
    let facts = GraphAnalysis::new(&graph);
    let a = value_producer_closure(&facts, [first]);
    assert_eq!(a.values(), &[first, one].into_iter().collect());
    assert_eq!(
        facts.producers().site(first),
        Some(SideEffectSite { block, index: 1 })
    );
    assert_eq!(
        a.operations(),
        &[SideEffectSite { block, index: 1 }].into_iter().collect()
    );
}

#[test]
fn nested_loop_membership_and_header_memos_share_one_snapshot() {
    let mut graph = EGraph::<Semantic>::new();
    let outer = graph.skeleton.entry;
    let inner = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let after_inner = graph.skeleton.create_block();
    let exit = graph.skeleton.create_block();
    let cond = graph.intern_constant(
        ConstantValue::Bool(true),
        Type::Constructed(TypeName::Bool, vec![]),
    );
    graph.skeleton.blocks[outer].control_header = Some(ControlHeader::Loop {
        merge: exit,
        continue_block: after_inner,
    });
    graph.skeleton.blocks[inner].control_header = Some(ControlHeader::Loop {
        merge: after_inner,
        continue_block: body,
    });
    for (header, then_target, else_target) in [(outer, inner, exit), (inner, body, after_inner)] {
        graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
            cond,
            then_target,
            then_args: vec![],
            else_target,
            else_args: vec![],
        };
    }
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: inner,
        args: vec![],
    };
    graph.skeleton.blocks[after_inner].term = SkeletonTerminator::Branch {
        target: outer,
        args: vec![],
    };
    let value = graph.alloc_side_effect_result(Type::Constructed(TypeName::UInt(32), vec![]));
    let result = graph.value_result(value);
    graph.skeleton.blocks[after_inner].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Op {
            tag: PureOp::Materialize,
        }),
        operands: smallvec![],
        result: Some(result),
        effects: None,
        span: None,
    });
    let facts = GraphAnalysis::new(&graph);
    for (block, expected) in [
        (outer, vec![outer]),
        (inner, vec![outer, inner]),
        (body, vec![outer, inner]),
        (after_inner, vec![outer]),
        (exit, vec![]),
    ] {
        assert_eq!(
            facts.loops().dependencies(block),
            Some(&expected.into_iter().collect::<LookupSet<_>>())
        );
    }
    assert!(!LoopInvariance::new(&facts, outer).is_invariant(value));
    assert!(LoopInvariance::new(&facts, inner).is_invariant(value));
    assert!(facts.loops().is_header(outer));
    assert!(facts.loops().is_header(inner));
    assert!(!facts.loops().is_header(body));
}
