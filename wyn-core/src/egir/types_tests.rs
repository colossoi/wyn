use super::*;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn u32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::UInt(32), vec![])
}

fn effect(result: NodeId) -> SideEffect {
    SideEffect {
        semantic_id: None,
        kind: SideEffectKind::Inst(InstKind::ControlBarrier),
        operand_nodes: SmallVec::new(),
        result: Some(result),
        effects: None,
        span: None,
    }
}

#[test]
fn indexes_results_across_skeleton_blocks() {
    let mut graph = EGraph::new();
    let unit = Type::Constructed(TypeName::Unit, vec![]);
    let first = graph.alloc_side_effect_result(unit.clone());
    let second = graph.alloc_side_effect_result(unit);
    let entry = graph.skeleton.entry;
    let other = graph.skeleton.create_block();
    graph.skeleton.blocks[entry].side_effects.push(effect(first));
    graph.skeleton.blocks[other].side_effects.push(effect(second));

    let index = graph.side_effect_index();
    assert_eq!(
        index.site(first),
        Some(SideEffectSite {
            block: entry,
            index: 0
        })
    );
    assert_eq!(
        index.site(second),
        Some(SideEffectSite {
            block: other,
            index: 0
        })
    );
    assert_eq!(
        index.effect(&graph, second).and_then(|effect| effect.result),
        Some(second)
    );
}

#[test]
fn replace_all_references_does_not_leave_stale_hash_cons_key() {
    let mut graph = EGraph::new();
    let int = i32_ty();
    let a = graph.intern_pure(PureOp::Int("1".into()), smallvec::smallvec![], int.clone());
    let b = graph.intern_pure(PureOp::Int("2".into()), smallvec::smallvec![], int.clone());
    let old_call = graph.intern_pure(
        PureOp::Call("__test_hash_cons".into()),
        smallvec::smallvec![a, b],
        int.clone(),
    );

    crate::egir::graph_ops::replace_all_references(&mut graph, b, a);

    let reinterned_old_call = graph.intern_pure(
        PureOp::Call("__test_hash_cons".into()),
        smallvec::smallvec![a, b],
        int,
    );

    assert_ne!(old_call, reinterned_old_call);
    assert!(graph.verify_hash_cons().is_ok());
}

#[test]
fn retype_node_does_not_leave_stale_hash_cons_key() {
    let mut graph = EGraph::<Semantic>::new();
    let int = i32_ty();
    let uint = u32_ty();
    let arg = graph.intern_pure(PureOp::Int("1".into()), smallvec::smallvec![], int.clone());
    let old_call = graph.intern_pure(
        PureOp::Call("__test_hash_cons_retype".into()),
        smallvec::smallvec![arg],
        int.clone(),
    );

    graph.retype_node(old_call, uint);

    let reinterned_old_call = graph.intern_pure(
        PureOp::Call("__test_hash_cons_retype".into()),
        smallvec::smallvec![arg],
        int,
    );

    assert_ne!(old_call, reinterned_old_call);
    assert!(graph.verify_hash_cons().is_ok());
}
