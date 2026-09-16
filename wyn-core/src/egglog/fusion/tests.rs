use super::super::{
    convert_program, fuse, Array, AssociatedData, Converted, ExprData, ExprKind, ExternData, OperationData,
    OperationId, OperationKind, RegionId, ScremaForm, SoacBody,
};
use crate::{ast::TypeName, compile_thru_tlc, tlc, types};
use egglog_engine::EGraph;

fn imported(source: &str) -> Converted {
    convert_program(&tlc::infer_input_slice_bounds(compile_thru_tlc(source).unwrap())).unwrap()
}

#[test]
fn selection_uses_the_complete_lexicographic_priority() {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, crate::egglog::SCHEMA).unwrap();
    graph.parse_and_run_program(None, include_str!("fusion.egg")).unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (PlanningRound 0)
        (Current (Group (OperationId 0)) (Source (OperationId 0)))
        (set (Owner (Source (OperationId 0))) (OperationId 0))
        (set (Scope (Source (OperationId 0))) (RegionId 2))
        (Current (Group (OperationId 3)) (Source (OperationId 3)))
        (set (Owner (Source (OperationId 3))) (OperationId 3))
        (set (Scope (Source (OperationId 3))) (RegionId 1))
        (Current (Group (OperationId 4)) (Source (OperationId 4)))
        (set (Owner (Source (OperationId 4))) (OperationId 4))
        (set (Scope (Source (OperationId 4))) (RegionId 1))
        (Current (Group (OperationId 5)) (Source (OperationId 5)))
        (set (Owner (Source (OperationId 5))) (OperationId 5))
        (Current (Group (OperationId 6)) (Source (OperationId 6)))
        (set (Owner (Source (OperationId 6))) (OperationId 6))
        (PlanCandidate 0 1 (Group (OperationId 0)) (Group (OperationId 0)))
        (PlanCandidate 0 0 (Group (OperationId 0)) (Group (OperationId 0)))
        (PlanCandidate 0 0 (Group (OperationId 4)) (Group (OperationId 0)))
        (PlanCandidate 0 0 (Group (OperationId 3)) (Group (OperationId 6)))
        (PlanCandidate 0 0 (Group (OperationId 3)) (Group (OperationId 5)))
        (run-schedule (saturate fusion-select) fusion-choose)
        (check (Chosen 0 0 (Group (OperationId 3)) (Group (OperationId 5))))
    "#,
        )
        .unwrap();
    assert_eq!(
        graph.function_to_dag("Chosen", usize::MAX, false).unwrap().0.len(),
        1
    );
}

#[test]
fn empty_planning_schedule_terminates_without_advancing() {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, crate::egglog::SCHEMA).unwrap();
    graph.parse_and_run_program(None, include_str!("fusion.egg")).unwrap();
    graph.parse_and_run_program(None, include_str!("schedule.egg")).unwrap();
    graph.parse_and_run_program(None, "(check (PlanningRound 0))").unwrap();
    assert!(graph.function_to_dag("FusionStep", usize::MAX, false).unwrap().0.is_empty());
}

#[test]
fn source_import_emits_linear_facts_for_a_map_chain() {
    let commands = |n: usize| {
        // Build the execution graph directly, isolating fact import from TLC's
        // recursive processing of deeply nested source lets.
        let mut data = imported("entry chain(xs:[4]i32) [4]i32=map(|x:i32|x+1,xs)").data;
        let region = entry(&data);
        let template = data.operations[*data.regions[region].members.first().unwrap()].clone();
        let OperationKind::Screma { inputs, .. } = &template.kind else {
            unreachable!()
        };
        let Array::Value(mut previous) = inputs[0] else {
            unreachable!()
        };
        let array_ty = data.expressions[previous].ty;
        data.regions[region].members.clear();
        for source_position in 0..n {
            let mut operation = template.clone();
            operation.source_position = source_position;
            let OperationKind::Screma { inputs, .. } = &mut operation.kind else {
                unreachable!()
            };
            inputs[0] = Array::Value(previous);
            let id = data.operations.alloc(operation);
            data.regions[region].members.insert(id);
            let result = data.expressions.alloc(ExprData {
                ty: template.ty,
                kind: ExprKind::OperationResult(id),
            });
            previous = data.expressions.alloc(ExprData {
                ty: array_ty,
                kind: ExprKind::Project {
                    tuple: result,
                    index: 0,
                },
            });
        }
        data.regions[region].results = vec![previous];
        let mut sink = super::analysis::Egglog::new();
        super::analysis::emit(&data, &mut sink).unwrap();
        crate::egglog::term::parse("scaling.egg", &sink.text).unwrap().len()
    };
    let small = commands(32);
    let large = commands(128);
    assert!(
        large <= 4 * small,
        "import grew faster than its source: {small} -> {large}"
    );
}

fn optimized(input: Converted) -> Converted {
    // Some tests modify the imported graph to introduce a precise effect/use,
    // just as the corresponding EGIR tests modify its semantic graph.
    let result = fuse(input).unwrap();
    EGraph::default().run_program(result.program.clone()).expect("extracted program must execute");
    result
}

fn entry(data: &AssociatedData) -> RegionId {
    data.definitions[data.entries.values().next().unwrap().definition].body
}

fn entry_ops(data: &AssociatedData) -> Vec<OperationId> {
    super::super::dependencies::analyze(data)
        .schedules(data)
        .unwrap()
        .remove(&entry(data))
        .unwrap_or_default()
}

fn form(data: &AssociatedData, id: OperationId) -> &ScremaForm {
    let OperationKind::Screma { form, .. } = &data.operations[id].kind else {
        panic!("expected Screma")
    };
    form
}

fn functions<'a>(body: &'a SoacBody, out: &mut Vec<&'a SoacBody>) {
    match body {
        SoacBody::Compose { first, then } => {
            functions(first, out);
            functions(then, out);
        }
        SoacBody::Parallel { left, right } => {
            functions(left, out);
            functions(right, out);
        }
        SoacBody::Apply { .. } => out.push(body),
        SoacBody::Identity(_) | SoacBody::Route { .. } => {}
    }
}

const CHAIN: &str = "entry chain(xs: [4]i32) [4]i32 =
    let a = map(|x: i32| x + 1, xs) in map(|x: i32| x * 2, a)";

#[test]
fn vertical_fusion_preserves_body_order_captures_and_consumer_identity() {
    let input = imported(
        "entry chain(xs: [4]i32, offset: i32, scale: i32) [4]i32 =
        let a = map(|x: i32| x + offset, xs) in map(|x: i32| x * scale, a)",
    );
    let [producer, consumer] = entry_ops(&input.data)[..] else {
        panic!("two maps")
    };
    let before = [
        form(&input.data, producer).pre.clone(),
        form(&input.data, consumer).pre.clone(),
    ];
    let result = optimized(input);
    assert_eq!(entry_ops(&result.data), &[consumer]);
    let mut after = Vec::new();
    functions(&form(&result.data, consumer).pre, &mut after);
    assert_eq!(format!("{after:?}"), format!("{before:?}"));
    for body in after {
        let SoacBody::Apply { captures, .. } = body else {
            unreachable!()
        };
        assert_eq!(captures.len(), 1);
    }
}

#[test]
fn saturation_fuses_a_chain_without_creating_a_dependency_cycle() {
    // EGIR planner_tests::cycle_candidates_are_skipped_without_changing_the_graph.
    // Adjacent contractions can consume this chain; merging first and last alone
    // would create a cycle through the middle operation.
    let input = imported(
        "entry chain(xs: [4]i32) [4]i32 =
        let a = map(|x: i32| x + 1, xs) in
        let b = map(|x: i32| x * 2, a) in map(|x: i32| x - 3, b)",
    );
    let last = *entry_ops(&input.data).last().unwrap();
    let result = optimized(input);
    assert_eq!(entry_ops(&result.data), &[last]);
    let mut bodies = Vec::new();
    functions(&form(&result.data, last).pre, &mut bodies);
    assert_eq!(bodies.len(), 3);
    let OperationKind::Screma { inputs, .. } = &result.data.operations[last].kind else {
        unreachable!()
    };
    let [Array::Value(input)] = inputs.as_slice() else {
        panic!("one original input")
    };
    assert!(matches!(
        result.data.expressions[*input].kind,
        ExprKind::Parameter(_)
    ));
    let again = fuse(result.clone()).unwrap();
    assert_eq!(
        format!("{:?}", result.data),
        format!("{:?}", again.data),
        "fixed point"
    );
}

#[test]
fn fused_scan_allocates_output_when_its_unique_input_is_absorbed() {
    // Port of EGIR fusion::mod_tests with its original gather/scan fixture.
    let input = imported(include_str!("../../../../testfiles/gather_scan_chain.wyn"));
    let original = entry_ops(&input.data).to_vec();
    let scan = original.iter().copied().find(|&id| {
        matches!(&input.data.operations[id].kind, OperationKind::Screma { form, .. } if !form.scans.is_empty())
    }).unwrap();
    let OperationKind::Screma { ownership, .. } = &input.data.operations[scan].kind else {
        unreachable!()
    };
    assert_eq!(ownership, &[types::SoacOwnership::UniqueInput]);
    let result = optimized(input);
    assert_eq!(entry_ops(&result.data).len(), original.len() - 1);
    let OperationKind::Screma { ownership, form, .. } = &result.data.operations[scan].kind else {
        unreachable!()
    };
    assert_eq!(ownership, &[types::SoacOwnership::Fresh]);
    assert_eq!(form.scans.len(), 1);
    assert_eq!(form.scans[0].neutral.len(), 1);
    // The later gather still observes a materialized scan result.
    assert!(entry_ops(&result.data).contains(&scan));
    assert_eq!(entry_ops(&result.data).last(), original.last());
}

#[test]
fn conditional_tuple_elements_keep_their_logical_boundaries() {
    // Conditional result handling from EGIR fusion::projection_tests, combined
    // with tuple-sensitive routing: the pair is one element, never two arrays.
    let input = imported(
        "entry pair_sum(xs: [4]i32) (i32, i32) =
        let pairs = map(|x: i32| (if x < 0 then 0 - x else x, x * 2), xs) in
        reduce(|(a,b): (i32,i32), (c,d): (i32,i32)| (a+c,b+d), (0,0), pairs)",
    );
    let result = optimized(input);
    assert_eq!(entry_ops(&result.data).len(), 1);
    let fused = form(&result.data, entry_ops(&result.data)[0]);
    let results = super::super::data::body_signature(&fused.pre).1;
    assert_eq!(results.len(), 1);
    assert!(matches!(
        &result.data.types[results[0]].ty,
        types::Type::Constructed(TypeName::Tuple(2), _)
    ));
    assert_eq!(fused.reductions.len(), 1);
    assert_eq!(fused.reductions[0].neutral.len(), 1);
    assert!(!fused.reductions[0].commutative);
    let SoacBody::Apply {
        parameters, results, ..
    } = &fused.reductions[0].operator
    else {
        panic!("reducer")
    };
    assert_eq!(parameters.len(), 2);
    assert_eq!(results.len(), 1);
}

#[test]
fn opaque_barriers_prevent_fusion_without_effect_tokens() {
    // Port of EGIR snapshot_preserves_opaque_barriers_without_effect_tokens.
    let mut input = imported(CHAIN);
    let region = entry(&input.data);
    let consumer = entry_ops(&input.data)[1];
    let definition = &input.data.definitions[input.data.entries.values().next().unwrap().definition];
    let ty = definition.ty;
    let extern_id = input.data.externs.alloc(ExternData {
        linkage_name: "opaque_barrier".into(),
    });
    let function = input.data.expressions.alloc(ExprData {
        ty,
        kind: ExprKind::Extern(extern_id),
    });
    let parameter = input
        .data
        .expressions
        .iter()
        .find_map(|(&id, value)| {
            matches!(value.kind, ExprKind::Parameter(p) if input.data.parameters[p].region == region)
                .then_some(id)
        })
        .unwrap();
    let template = input.data.operations[entry_ops(&input.data)[0]].clone();
    let barrier = input.data.operations.alloc(OperationData {
        kind: OperationKind::Call {
            function,
            args: vec![parameter],
        },
        ..template
    });
    input.data.regions[region].members.insert(barrier);
    input.data.operations[consumer].source_position = 2;
    input.data.operations[barrier].source_position = 1;
    let before = entry_ops(&input.data).to_vec();
    let result = optimized(input);
    assert_eq!(entry_ops(&result.data), before);
}

#[test]
fn cross_region_uses_keep_the_producer_materialized() {
    // Region equivalent of EGIR snapshot_keeps_cross_block_uses_as_external_observers.
    let input = imported(
        "entry branch(xs: [4]i32, flag: bool) [4]i32 =
        let a = map(|x: i32| x + 1, xs) in
        if flag then map(|x: i32| x * 2, a) else a",
    );
    let before = entry_ops(&input.data).to_vec();
    let result = optimized(input);
    assert_eq!(entry_ops(&result.data), before);
    assert_eq!(
        super::super::dependencies::analyze(&result.data)
            .schedules(&result.data)
            .unwrap()
            .values()
            .map(Vec::len)
            .sum::<usize>(),
        3
    );
}

#[test]
fn separate_bodies_and_loop_parameters_do_not_alias() {
    // EGIR snapshot_qualifies_values_by_body_and_accepts_loop_parameters,
    // with a fusible pair in each scope to exercise the rewrite as well.
    let input = imported(
        "entry first(xs: [4]i32) [4]i32 =
          map(|x: i32| x * 2, map(|x: i32| x + 1, xs))
        entry second(xs: [4]i32) [4]i32 =
          loop values = xs for i < 3 do map(|x: i32| x * 2, map(|x: i32| x + 1, values))",
    );
    let result = optimized(input);
    let mut parameters = Vec::new();
    for (region_id, ops) in
        super::super::dependencies::analyze(&result.data).schedules(&result.data).unwrap()
    {
        let region = &result.data.regions[region_id];
        for op in ops {
            if let OperationKind::Screma { inputs, .. } = &result.data.operations[op].kind {
                let [Array::Value(value)] = inputs.as_slice() else {
                    panic!("one input")
                };
                let ExprKind::Parameter(param) = result.data.expressions[*value].kind else {
                    panic!("scoped input")
                };
                assert_eq!(
                    result.data.regions[result.data.parameters[param].region].definition,
                    region.definition
                );
                parameters.push(param);
            }
        }
    }
    assert_eq!(parameters.len(), 2);
    assert_ne!(parameters[0], parameters[1]);
}

#[test]
fn capture_dependencies_prevent_absorption() {
    for (source, expected) in [
        (
            "entry captured(xs: [4]i32) [4]i32 =
          let a = map(|x: i32| x + 1, xs) in map(|x: i32| x + a[0], a)",
            2,
        ),
        (
            "entry cycle(xs: [4]i32) [4]i32 =
          let a = map(|x: i32| x + 1, xs) in
          let b = map(|x: i32| x * 2, a) in map(|x: i32| x + b[0], a)",
            2,
        ),
    ] {
        let input = imported(source);
        let before = entry_ops(&input.data).to_vec();
        let result = optimized(input);
        // A and B may combine while retaining A, but the final consumer must
        // still run separately because its capture needs the completed array.
        let after = entry_ops(&result.data);
        assert_eq!(after.len(), expected, "{source}");
        assert_eq!(after.last(), before.last(), "{source}");
    }
}

#[test]
fn input_write_hazards_and_unknown_body_calls_prevent_fusion() {
    // The ownership-side equivalent of EGIR's input resource write hazard.
    let mut input = imported(CHAIN);
    let producer = entry_ops(&input.data)[0];
    let OperationKind::Screma { ownership, .. } = &mut input.data.operations[producer].kind else {
        unreachable!()
    };
    ownership[0] = types::SoacOwnership::UniqueInput;
    let before = entry_ops(&input.data).to_vec();
    assert_eq!(entry_ops(&optimized(input).data), before);

    // A source body with an array read is conservatively kept as a separate
    // operation; scalar expression interning does not prove storage-read safety.
    let input = imported(
        "entry reads(xs: [4]i32, ys: [4]i32) [4]i32 =
        let a = map(|x: i32| ys[x], xs) in map(|x: i32| x + 1, a)",
    );
    let before = entry_ops(&input.data).to_vec();
    assert_eq!(entry_ops(&optimized(input).data), before);
}

#[test]
fn unreferenced_metadata_and_expressions_do_not_block_fusion() {
    let mut input = imported(CHAIN);
    let producer = entry_ops(&input.data)[0];
    let consumer = entry_ops(&input.data)[1];
    let result = input
        .data
        .expressions
        .iter()
        .find_map(|(&id, expr)| {
            matches!(expr.kind, ExprKind::OperationResult(op) if op == producer).then_some(id)
        })
        .unwrap();
    input.data.expressions.alloc(ExprData {
        ty: input.data.expressions[result].ty,
        kind: ExprKind::Tuple(vec![result, result]),
    });
    assert_eq!(entry_ops(&optimized(input).data), &[consumer]);
}

#[test]
fn scalar_builtin_values_fuse_but_execution_dependent_builtins_do_not() {
    for (builtin, expected) in [("abs", 1), ("d_fdx", 2), ("d_fdy", 2), ("fwidth", 2)] {
        let input = imported(&format!(
            "open f32\nentry values(xs: [4]f32) [4]f32 =
              let a = map(|x: f32| {builtin}(x), xs) in map(|x: f32| x + 1.0, a)"
        ));
        assert_eq!(entry_ops(&optimized(input).data).len(), expected, "{builtin}");
    }
}
