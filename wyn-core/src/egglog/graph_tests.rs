use super::fusion::analysis::{emit, Egglog};
use super::{from_tlc, fuse, ExprId, Program, SoacBody};
use crate::compile_thru_tlc;
use crate::egglog::data::{
    Array, ExprData, ExprKind, ExternData, Ir, OperationData, OperationId, OperationKind, RegionId,
};
use crate::egglog::dependencies::analyze;
use crate::egglog::{parse_program, Fused, Imported, OptimizeError};
use crate::tlc::infer_input_slice_bounds;
use egglog_engine::EGraph;

const CHAIN: &str = "entry chain(xs: [4]i32) [4]i32 =
    let a = map(|x: i32| x + 1, xs) in map(|x: i32| x * 2, a)";

fn fuse_updated(mut input: Program<Imported>) -> Result<Program<Fused>, OptimizeError> {
    let mut sink = Egglog::new();
    emit(&input.ir, &mut sink)?;
    input.state.facts = parse_program("test-fusion.egg", &sink.text)?;
    fuse(input)
}

fn imported(source: &str) -> Program<Imported> {
    from_tlc(&infer_input_slice_bounds(compile_thru_tlc(source).unwrap())).unwrap()
}

fn entry(data: &Ir) -> RegionId {
    data.definitions[data.entries.values().next().unwrap().definition].body
}

fn schedule(data: &Ir) -> Vec<OperationId> {
    analyze(data).schedules(data).unwrap().remove(&entry(data)).unwrap_or_default()
}

fn parameter(data: &Ir) -> ExprId {
    let region = entry(data);
    data.expressions
        .iter()
        .find_map(|(&id, expr)| {
            matches!(expr.kind, ExprKind::Parameter(p) if data.parameters[p].region == region).then_some(id)
        })
        .unwrap()
}

#[test]
fn fusion_follows_dependencies_across_an_independent_reduction() {
    // The opening of EGIR's reduction-bearing-producer case: the independent
    // reduction used to break adjacency between the two maps.
    let input = imported(
        "entry redomap_then_map(xs: [4]i32) (i32, [4]i32) =
        let mapped = map(|x: i32| x + 1, xs) in
        let total = reduce(|a: i32, b: i32| a + b, 0, xs) in
        let consumed = map(|x: i32| x * 2, mapped) in (total, consumed)",
    );
    let before = schedule(&input.ir);
    assert_eq!(before.len(), 3);
    let result = fuse_updated(input).unwrap();
    assert_eq!(schedule(&result.ir), before[2..]);
    // The reduction joins the maps horizontally; only the surviving execution
    // remains a member, while old arena records remain available as provenance.
    assert_eq!(result.ir.regions[entry(&result.ir)].members.len(), 1);
    assert_eq!(result.ir.operations.len(), 3);
    let OperationKind::Screma { form, .. } = &result.ir.operations[before[2]].kind else {
        panic!("map")
    };
    assert!(matches!(form.pre, SoacBody::Compose { .. }));
    let mut graph = EGraph::default();
    let mut sink = Egglog::new();
    emit(&result.ir, &mut sink).unwrap();
    graph.parse_and_run_program(None, &sink.text).unwrap();
    graph
        .parse_and_run_program(
            None,
            &format!(
                "(check (Current g p) (= (Scope p) {}) (= (Owner p) {})) (fail (check (Current g p) (= (Scope p) {}) (= (Owner p) {})))",
                entry(&result.ir).egglog(),
                before[2].egglog(),
                entry(&result.ir).egglog(),
                before[0].egglog(),
            ),
        )
        .unwrap();
}

#[test]
fn schedule_is_topological_even_when_membership_ids_and_source_positions_disagree() {
    let mut input = imported(CHAIN);
    let region = entry(&input.ir);
    let before = schedule(&input.ir);
    let old_producer = before[0];
    let consumer = before[1];
    // Give the producer a larger identity than its consumer, and reverse source
    // positions too. Neither ordering may determine the schedule of pure work.
    let producer = input.ir.operations.alloc(input.ir.operations[old_producer].clone());
    input.ir.regions[region].members.remove(&old_producer);
    input.ir.regions[region].members.insert(producer);
    let ids: Vec<_> = input.ir.expressions.ids().collect();
    for id in ids {
        if matches!(input.ir.expressions[id].kind, ExprKind::OperationResult(op) if op == old_producer) {
            input.ir.expressions[id].kind = ExprKind::OperationResult(producer);
        }
    }
    input.ir.operations[producer].source_position = 10;
    input.ir.operations[consumer].source_position = 0;
    let OperationKind::Screma { reuse_inputs, .. } = &mut input.ir.operations[consumer].kind else {
        panic!("map")
    };
    reuse_inputs[0] = None;
    assert_eq!(schedule(&input.ir), [producer, consumer]);
    let mut graph = fusion_dependencies(&input.ir);
    graph
        .parse_and_run_program(
            None,
            &format!(
                "(check (GroupEdge (Group {}) (Group {}))) (fail (check (GroupBefore (Group {}) (Group {}))))",
                producer.egglog(),
                consumer.egglog(),
                consumer.egglog(),
                producer.egglog(),
            ),
        )
        .unwrap();
}

#[test]
fn backward_traversal_does_not_root_unused_members_or_metadata() {
    let mut input = imported(CHAIN);
    let region = entry(&input.ir);
    // The second map may reuse the first map's temporary. That permission is
    // not an externally observable effect requiring either unused map to run.
    input.ir.regions[region].results = vec![parameter(&input.ir)];
    assert!(schedule(&input.ir).is_empty());
    let result = fuse_updated(input).unwrap();
    assert!(schedule(&result.ir).is_empty());
    assert_eq!(result.ir.regions[region].members.len(), 2);
}

#[test]
fn effect_roots_keep_their_value_dependencies_but_not_unrelated_ordered_work() {
    let mut input = imported(CHAIN);
    let region = entry(&input.ir);
    let before = schedule(&input.ir);
    let producer = before[0];
    let consumer = before[1];
    let OperationKind::Screma { inputs, .. } = &input.ir.operations[consumer].kind else {
        panic!("map")
    };
    let Array::Value(producer_array) = inputs[0] else {
        panic!("array")
    };
    let extern_id = input.ir.externs.alloc(ExternData {
        linkage_name: "observe".into(),
    });
    let function = input.ir.expressions.alloc(ExprData {
        ty: input.ir.definitions[input.ir.entries.values().next().unwrap().definition].ty,
        kind: ExprKind::Extern(extern_id),
    });
    let effect = input.ir.operations.alloc(OperationData {
        kind: OperationKind::Call {
            function,
            args: vec![producer_array],
        },
        source_position: 2,
        ..input.ir.operations[producer].clone()
    });
    input.ir.regions[region].members.insert(effect);
    input.ir.regions[region].results = vec![parameter(&input.ir)];
    assert_eq!(schedule(&input.ir), [producer, effect]);
    // The dead consumer supplies neither demand nor a fusion constraint.
    let mut graph = fusion_dependencies(&input.ir);
    graph
        .parse_and_run_program(
            None,
            &format!(
                "(fail (check (GroupBefore (Group {}) (Group {})))) (check (Current g p) (= (Scope p) {}) (= (Owner p) {}))",
                consumer.egglog(),
                effect.egglog(),
                region.egglog(),
                effect.egglog(),
            ),
        )
        .unwrap();
    assert_eq!(schedule(&fuse_updated(input).unwrap().ir), [producer, effect]);
}

#[test]
fn inactive_nested_regions_do_not_keep_enclosing_pure_work_alive() {
    let mut input = imported(
        "entry branch(xs: [4]i32, flag: bool) [4]i32 =
        let a = map(|x: i32| x + 1, xs) in
        if flag then map(|x: i32| x * 2, a) else a",
    );
    let region = entry(&input.ir);
    let members: Vec<_> = input.ir.regions[region].members.iter().copied().collect();
    let control = *members.last().unwrap();
    // Make the branch an unused lambda value instead of an invoked Choose.
    // Its arena records and result expressions still refer to the outer map.
    let OperationKind::If { then_region, .. } = input.ir.operations[control].kind else {
        panic!("branch")
    };
    input.ir.expressions.alloc(ExprData {
        ty: input.ir.operations[control].ty,
        kind: ExprKind::Lambda(then_region),
    });
    input.ir.regions[region].members.remove(&control);
    input.ir.regions[region].results = vec![parameter(&input.ir)];
    assert!(schedule(&input.ir).is_empty());
    let schedules = analyze(&input.ir).schedules(&input.ir).unwrap();
    assert!(!schedules.contains_key(&then_region));
}

#[test]
fn cyclic_execution_graphs_are_rejected() {
    let mut input = imported(CHAIN);
    let before = schedule(&input.ir);
    let consumer_array = input.ir.regions[entry(&input.ir)].results[0];
    let OperationKind::Screma { inputs, .. } = &mut input.ir.operations[before[0]].kind else {
        panic!("map")
    };
    inputs[0] = Array::Value(consumer_array);
    let error = analyze(&input.ir).schedules(&input.ir).unwrap_err();
    assert!(error.to_string().contains("cycle"));
}

/// Load only the fusion summary and its dependency rules for relation assertions.
pub(super) fn fusion_dependencies(data: &Ir) -> EGraph {
    let mut sink = Egglog::new();
    emit(data, &mut sink).unwrap();
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, &sink.text).unwrap();
    graph.parse_and_run_program(None, include_str!("fusion/fusion.egg")).unwrap();
    graph.parse_and_run_program(None, "(run-schedule (saturate (run fusion-dependencies)))").unwrap();
    graph
}
