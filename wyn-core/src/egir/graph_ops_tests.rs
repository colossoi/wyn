use super::*;
use crate::ast::TypeName;
use crate::egir::program::SemanticOpId;
use crate::egir::types::{EffectToken, SkeletonTerminator};
use crate::ssa::types::{ConstantValue, InstKind};
use polytype::Type;

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

#[test]
fn value_producer_closure_crosses_effects_block_params_and_loop_cycles() {
    let mut graph = EGraph::<Semantic>::new();
    let entry = graph.skeleton.entry;
    let header = graph.skeleton.create_block();
    let exit = graph.skeleton.create_block();
    let ty = u32_ty();
    let place = graph.intern_constant(ConstantValue::U32(0), ty.clone());
    let produced = graph.alloc_side_effect_result(ty.clone());
    graph.skeleton.blocks[entry].side_effects.push(SideEffect {
        semantic_id: Some(SemanticOpId(0)),
        kind: SideEffectKind::Inst(InstKind::Load {
            place: Default::default(),
        }),
        operand_nodes: smallvec![place],
        result: Some(produced),
        effects: Some((EffectToken(0), EffectToken(1))),
        span: None,
    });
    graph.skeleton.blocks[entry].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![produced],
    };

    let current = graph.add_block_param(header, 0, ty.clone());
    graph.skeleton.blocks[header].params.push(current);
    let one = graph.intern_constant(ConstantValue::U32(1), ty.clone());
    let next = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![current, one],
        ty.clone(),
        None,
    );
    let cond = graph.intern_constant(
        ConstantValue::Bool(true),
        Type::Constructed(TypeName::Bool, vec![]),
    );

    let merged = graph.add_block_param(exit, 0, ty.clone());
    graph.skeleton.blocks[exit].params.push(merged);
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: header,
        then_args: vec![next],
        else_target: exit,
        else_args: vec![current],
    };
    graph.skeleton.blocks[exit].term = SkeletonTerminator::Return(Some(merged));
    let tail = graph.intern_pure(PureOp::BinOp("+".into()), smallvec![merged, one], ty, None);

    let closure = value_producer_closure(&graph, [tail]);

    assert_eq!(
        closure.effects,
        HashSet::from([SideEffectSite {
            block: entry,
            index: 0,
        }])
    );
    for expected in [tail, merged, current, next, one, cond, produced, place] {
        assert!(
            closure.nodes.contains(&expected),
            "producer closure omitted {expected:?}"
        );
    }
}
