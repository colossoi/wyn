use super::*;
use crate::egir::program::SemanticOpId;
use crate::egir::types::EffectOp;
use smallvec::SmallVec;

fn raw_map() -> SideEffect<Raw> {
    SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(
            (),
            Soac::Screma(screma::Op {
                inputs: vec![],
                form: screma::ScremaForm {
                    pre: screma::Lambda::identity(vec![]),
                    scans: vec![],
                    reductions: vec![],
                    post: screma::Lambda::identity(vec![]),
                },
                result_state: vec![],
                state: screma::RawState,
            }),
        )),
        operands: SmallVec::new(),
        result: None,
        effects: None,
        span: None,
    }
}

fn facts() -> Facts {
    Facts {
        space: SegSpace::new(SegExtent::Fixed(1)),
        placement: screma::Placement::LaneLocal,
        output_slots: vec![],
        resources: vec![],
        entry: false,
    }
}

#[test]
fn phase_boundary_assigns_ids_to_soacs_but_not_instructions() {
    let mut graph = EGraph::<Raw>::new();
    let block = graph.skeleton.entry;
    graph.skeleton.blocks[block].side_effects.push(raw_map());
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Effect(EffectOp::ControlBarrier),
        operands: SmallVec::new(),
        result: None,
        effects: None,
        span: None,
    });
    graph.skeleton.blocks[block].side_effects.push(raw_map());

    let mut semantic_ids = SemanticOpIdSource::default();
    for _ in 0..7 {
        semantic_ids.next_id();
    }
    let (graph, _) = map_graph(
        graph,
        HashMap::from([((block, 0), facts()), ((block, 2), facts())]),
        &mut semantic_ids,
    );
    let ids: Vec<_> = graph.skeleton.blocks[graph.skeleton.entry]
        .side_effects
        .iter()
        .map(|effect| effect.kind.soac_id().copied())
        .collect();

    assert_eq!(
        ids,
        vec![
            Some(SemanticOpId::for_test(7)),
            None,
            Some(SemanticOpId::for_test(8))
        ]
    );
    assert_eq!(semantic_ids.next_id(), SemanticOpId::for_test(9));
}
