use super::*;

fn effect(result: NodeId) -> SideEffect {
    SideEffect {
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
