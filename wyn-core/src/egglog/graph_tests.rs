use super::optimize::analyze;
use super::{
    convert_program, optimize, snapshot, Array, AssociatedData, Converted, ExprData, ExprKind, ExternData,
    OperationData, OperationId, OperationKind, RegionId,
};
use crate::{compile_thru_tlc, tlc, types};
use egglog_engine::EGraph;

const CHAIN: &str = "entry chain(xs: [4]i32) [4]i32 =
    let a = map(|x: i32| x + 1, xs) in map(|x: i32| x * 2, a)";

fn imported(source: &str) -> Converted {
    convert_program(&tlc::infer_input_slice_bounds(compile_thru_tlc(source).unwrap())).unwrap()
}

fn entry(data: &AssociatedData) -> RegionId {
    data.definitions[data.entries.values().next().unwrap().definition].body
}

fn schedule(data: &AssociatedData) -> Vec<OperationId> {
    snapshot::analyze(data).schedules(data).unwrap().remove(&entry(data)).unwrap_or_default()
}

fn parameter(data: &AssociatedData) -> super::ExprId {
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
    let before = schedule(&input.data);
    assert_eq!(before.len(), 3);
    let result = optimize(input).unwrap();
    assert_eq!(schedule(&result.data), before[2..]);
    // The reduction joins the maps horizontally; only the surviving execution
    // remains a member, while old arena records remain available as provenance.
    assert_eq!(result.data.regions[entry(&result.data)].members.len(), 1);
    assert_eq!(result.data.operations.len(), 3);
    let OperationKind::Screma { form, .. } = &result.data.operations[before[2]].kind else {
        panic!("map")
    };
    assert!(matches!(form.pre, super::SoacBody::Compose { .. }));
    let mut graph = EGraph::default();
    graph.run_program(result.program).unwrap();
    graph
        .parse_and_run_program(
            None,
            &format!(
                "(check (Operation {} {})) (fail (check (Operation {} {})))",
                entry(&result.data).egglog(),
                before[2].egglog(),
                entry(&result.data).egglog(),
                before[0].egglog(),
            ),
        )
        .unwrap();
}

#[test]
fn schedule_is_topological_even_when_membership_ids_and_source_positions_disagree() {
    let mut input = imported(CHAIN);
    let region = entry(&input.data);
    let before = schedule(&input.data);
    let old_producer = before[0];
    let consumer = before[1];
    // Give the producer a larger identity than its consumer, and reverse source
    // positions too. Neither ordering may determine the schedule of pure work.
    let producer = input.data.operations.alloc(input.data.operations[old_producer].clone());
    input.data.regions[region].members.remove(&old_producer);
    input.data.regions[region].members.insert(producer);
    let ids: Vec<_> = input.data.expressions.ids().collect();
    for id in ids {
        if matches!(input.data.expressions[id].kind, ExprKind::OperationResult(op) if op == old_producer) {
            input.data.expressions[id].kind = ExprKind::OperationResult(producer);
        }
    }
    input.data.operations[producer].source_position = 10;
    input.data.operations[consumer].source_position = 0;
    let OperationKind::Screma { ownership, .. } = &mut input.data.operations[consumer].kind else {
        panic!("map")
    };
    ownership[0] = types::SoacOwnership::Fresh;
    assert_eq!(schedule(&input.data), [producer, consumer]);
    let mut graph = analyze(&input.data).unwrap();
    graph
        .parse_and_run_program(
            None,
            &format!(
                "(check (DependsOn {} {})) (fail (check (EffectBefore {} {})))",
                consumer.egglog(),
                producer.egglog(),
                consumer.egglog(),
                producer.egglog(),
            ),
        )
        .unwrap();
}

#[test]
fn backward_traversal_does_not_root_unused_members_or_metadata() {
    let mut input = imported(CHAIN);
    let region = entry(&input.data);
    // The second map may reuse the first map's temporary. That permission is
    // not an externally observable effect requiring either unused map to run.
    input.data.regions[region].results = vec![parameter(&input.data)];
    assert!(schedule(&input.data).is_empty());
    let result = optimize(input).unwrap();
    assert!(schedule(&result.data).is_empty());
    assert_eq!(result.data.regions[region].members.len(), 2);
}

#[test]
fn effect_roots_keep_their_value_dependencies_but_not_unrelated_ordered_work() {
    let mut input = imported(CHAIN);
    let region = entry(&input.data);
    let before = schedule(&input.data);
    let producer = before[0];
    let consumer = before[1];
    let OperationKind::Screma { inputs, .. } = &input.data.operations[consumer].kind else {
        panic!("map")
    };
    let Array::Value(producer_array) = inputs[0] else {
        panic!("array")
    };
    let extern_id = input.data.externs.alloc(ExternData {
        linkage_name: "observe".into(),
    });
    let function = input.data.expressions.alloc(ExprData {
        ty: input.data.definitions[input.data.entries.values().next().unwrap().definition].ty,
        kind: ExprKind::Extern(extern_id),
    });
    let effect = input.data.operations.alloc(OperationData {
        kind: OperationKind::Call {
            function,
            args: vec![producer_array],
        },
        source_position: 2,
        ..input.data.operations[producer].clone()
    });
    input.data.regions[region].members.insert(effect);
    input.data.regions[region].results = vec![parameter(&input.data)];
    assert_eq!(schedule(&input.data), [producer, effect]);
    // The dead consumer supplies neither demand nor a fusion constraint.
    let mut graph = analyze(&input.data).unwrap();
    graph
        .parse_and_run_program(
            None,
            &format!(
                "(fail (check (EffectBefore {} {}))) (check (Operation {} {}))",
                consumer.egglog(),
                effect.egglog(),
                region.egglog(),
                effect.egglog(),
            ),
        )
        .unwrap();
    assert_eq!(schedule(&optimize(input).unwrap().data), [producer, effect]);
}

#[test]
fn inactive_nested_regions_do_not_keep_enclosing_pure_work_alive() {
    let mut input = imported(
        "entry branch(xs: [4]i32, flag: bool) [4]i32 =
        let a = map(|x: i32| x + 1, xs) in
        if flag then map(|x: i32| x * 2, a) else a",
    );
    let region = entry(&input.data);
    let members: Vec<_> = input.data.regions[region].members.iter().copied().collect();
    let control = *members.last().unwrap();
    // Make the branch an unused lambda value instead of an invoked Choose.
    // Its arena records and result expressions still refer to the outer map.
    let OperationKind::If { then_region, .. } = input.data.operations[control].kind else {
        panic!("branch")
    };
    input.data.expressions.alloc(ExprData {
        ty: input.data.operations[control].ty,
        kind: ExprKind::Lambda(then_region),
    });
    input.data.regions[region].members.remove(&control);
    input.data.regions[region].results = vec![parameter(&input.data)];
    assert!(schedule(&input.data).is_empty());
    let schedules = snapshot::analyze(&input.data).schedules(&input.data).unwrap();
    assert!(!schedules.contains_key(&then_region));
}

#[test]
fn cyclic_execution_graphs_are_rejected() {
    let mut input = imported(CHAIN);
    let before = schedule(&input.data);
    let consumer_array = input.data.regions[entry(&input.data)].results[0];
    let OperationKind::Screma { inputs, .. } = &mut input.data.operations[before[0]].kind else {
        panic!("map")
    };
    inputs[0] = Array::Value(consumer_array);
    let error = snapshot::analyze(&input.data).schedules(&input.data).unwrap_err();
    assert!(error.to_string().contains("cycle"));
}
