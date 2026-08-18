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
        output_slots: vec![],
        resources: vec![],
        entry: false,
    }
}

fn reified_source(source: &str) -> Segmented {
    let program = crate::compile_thru_tlc(source).expect("compile through TLC");
    let program = crate::tlc::infer_input_slice_bounds(program);
    let program = crate::to_egraph(program).expect("convert to raw EGIR");
    reify_soacs(program)
}

#[test]
fn canonical_resource_verifier_covers_screma_and_filter() {
    let mut screma = reified_source("entry main(xs: []i32) []i32 = map(|x: i32| x + 1, xs)");
    assert!(verify_canonical_resource_accesses(&screma).is_ok());
    let resources = screma.entry_points[0]
        .graph
        .skeleton
        .blocks
        .values_mut()
        .find_map(|block| {
            block.side_effects.iter_mut().find_map(|effect| {
                let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &mut effect.kind else {
                    return None;
                };
                let screma::SemanticState::Segmented { resources, .. } = op.semantic_state_mut() else {
                    return None;
                };
                Some(resources)
            })
        })
        .expect("segmented Screma resources");
    resources.push(resources[0]);
    assert!(verify_canonical_resource_accesses(&screma).unwrap_err().contains("Screma"));

    let mut filter = reified_source("entry main(xs: []i32) []i32 = filter(|x: i32| x % 2 == 0, xs)");
    assert!(verify_canonical_resource_accesses(&filter).is_ok());
    let resources = filter.entry_points[0]
        .graph
        .skeleton
        .blocks
        .values_mut()
        .find_map(|block| {
            block.side_effects.iter_mut().find_map(|effect| {
                let SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) = &mut effect.kind else {
                    return None;
                };
                Some(&mut op.state.resources)
            })
        })
        .expect("Filter resources");
    resources.push(resources[0]);
    assert!(verify_canonical_resource_accesses(&filter).unwrap_err().contains("Filter"));
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
