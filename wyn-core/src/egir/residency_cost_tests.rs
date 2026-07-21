use super::*;
use crate::ast::TypeName;
use crate::egir::graph_projector::GraphProjector;
use crate::egir::types::{EffectToken, PureOp, SideEffectSite};
use polytype::Type;
use smallvec::smallvec;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

#[test]
fn profitability_includes_launch_loads_and_margin() {
    assert!(!materialization_is_profitable(1, 128));
    assert!(materialization_is_profitable(256, 64));

    let cost = 20;
    let invocations = 64;
    let recompute = cost * invocations;
    let handoff = SINGLETON_LAUNCH_COST + cost + STORAGE_LOAD_COST * invocations;
    assert_eq!(
        materialization_is_profitable(cost, invocations),
        4 * recompute >= 5 * handoff
    );
}

#[test]
fn structured_storage_prefix_requires_materialization() {
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let continuation = graph.skeleton.create_block();
    let zero = graph.intern_constant(ConstantValue::U32(0), u32_ty());
    let view = graph_ops::intern_resource_view(&mut graph, crate::ResourceId::for_test(1), i32_ty(), None);
    let place = graph.intern_pure(PureOp::ViewIndex, smallvec![view, zero], i32_ty(), None);
    let loaded = graph.alloc_side_effect_result(i32_ty());
    graph.skeleton.blocks[entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::Load),
        operand_nodes: smallvec![place],
        result: Some(loaded),
        effects: Some((EffectToken::from(0), EffectToken::from(1))),
        span: None,
    });
    let result = graph.add_block_param(continuation, i32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: continuation,
        args: vec![loaded],
    };
    graph.skeleton.blocks[continuation].term = SkeletonTerminator::Return(None);

    let recipe = GraphProjector::new(&graph, &LookupMap::new())
        .captured_value_recipe(
            result,
            SideEffectSite {
                block: continuation,
                index: 0,
            },
        )
        .expect("structured storage recipe");
    assert_eq!(
        prelude_materialization_policy(&recipe),
        PreludeMaterializationPolicy::Required
    );
}

#[test]
fn canonical_fixed_range_loop_recovers_trip_count() {
    let mut graph = EGraph::new();
    let entry = graph.skeleton.entry;
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();
    let zero = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty(), None);
    let one = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty(), None);
    let bound = graph.intern_pure(PureOp::Int("32".into()), smallvec![], i32_ty(), None);
    let index = graph.add_block_param(header, i32_ty());
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![zero],
    };
    let cond = graph.intern_pure(
        PureOp::BinOp("<".into()),
        smallvec![index, bound],
        Type::Constructed(TypeName::Bool, vec![]),
        None,
    );
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: body,
        then_args: vec![],
        else_target: merge,
        else_args: vec![],
    };
    let next = graph.intern_pure(PureOp::BinOp("+".into()), smallvec![index, one], i32_ty(), None);
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![next],
    };
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Return(None);

    assert_eq!(fixed_loop_trip_count(&graph, header, body, merge), Some(32));
}
