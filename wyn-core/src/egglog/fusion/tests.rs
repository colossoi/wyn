use super::super::data::body_signature;
use super::super::dependencies::analyze;
use super::super::{from_tlc, fuse, Program};
use super::analysis::import;
use crate::ast::TypeName;
use crate::compile_thru_tlc;
use crate::egglog::data::{
    Array, ExprData, ExprKind, ExternData, Ir, OperationData, OperationId, OperationKind, RegionId,
    ScremaForm, SoacBody,
};
use crate::egglog::{Fused, Imported, SCHEMA};
use crate::tlc::infer_input_slice_bounds;
use crate::types::Type;
use egglog_engine::EGraph;

fn imported(source: &str) -> Program<Imported> {
    from_tlc(&infer_input_slice_bounds(compile_thru_tlc(source).unwrap())).unwrap()
}

#[test]
fn selection_uses_the_complete_lexicographic_priority() {
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, SCHEMA).unwrap();
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
    graph.parse_and_run_program(None, SCHEMA).unwrap();
    graph.parse_and_run_program(None, include_str!("fusion.egg")).unwrap();
    graph.parse_and_run_program(None, include_str!("schedule.egg")).unwrap();
    graph.parse_and_run_program(None, "(check (PlanningRound 0))").unwrap();
    assert!(graph.function_to_dag("FusionStep", usize::MAX, false).unwrap().0.is_empty());
}

#[test]
fn source_import_emits_linear_facts_for_a_map_chain() {
    let tuples = |n: usize| {
        // Build the execution graph directly, isolating fact import from TLC's
        // recursive processing of deeply nested source lets.
        let mut data = imported("entry chain(xs:[4]i32) [4]i32=map(|x:i32|x+1,xs)").ir;
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
        import(&data).unwrap().num_tuples()
    };
    let small = tuples(32);
    let large = tuples(128);
    assert!(
        large <= 4 * small,
        "import grew faster than its source: {small} -> {large}"
    );
}

fn optimized(mut input: Program<Imported>) -> Program<Fused> {
    // Some tests modify the imported graph to introduce a precise effect/use,
    // before importing the modified dependency graph.
    input.state.graph = import(&input.ir).unwrap();
    fuse(input).unwrap()
}

fn entry(data: &Ir) -> RegionId {
    data.definitions[data.entries.values().next().unwrap().definition].body
}

fn entry_ops(data: &Ir) -> Vec<OperationId> {
    analyze(data).schedules(data).unwrap().remove(&entry(data)).unwrap_or_default()
}

#[test]
fn filter_post_map_rejects_effectful_callbacks() {
    for source in [
        "entry main(dest:*[3]i32,xs:[3]i32) []i32 =
         map(|x:i32|let updated=scatter(dest,[0],[x]) in updated[0],filter(|x:i32|x>0,xs))",
        "def read(other:[3]i32,x:i32) i32 = other[x]
         entry main(xs:[3]i32,other:[3]i32) []i32 = map(|x:i32|read(other,x),filter(|x:i32|x>0,xs))",
    ] {
        let program = optimized(imported(source));
        assert_eq!(entry_ops(&program.ir).len(), 2, "{source}");
    }
}

#[test]
fn filter_post_map_captures_array_predicate_and_changes_record_type() {
    let program = imported(include_str!("../../../../testfiles/rust_host_filter_post.wyn"));
    let program = optimized(program);
    assert_eq!(entry_ops(&program.ir).len(), 2);
    let filter = entry_ops(&program.ir)
        .into_iter()
        .find(|op| matches!(program.ir.operations[*op].kind, OperationKind::Filter { .. }))
        .unwrap();
    assert!(!analyze(&program.ir).movable.contains(&filter));
}

fn form(data: &Ir, id: OperationId) -> &ScremaForm {
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
    let [producer, consumer] = entry_ops(&input.ir)[..] else {
        panic!("two maps")
    };
    let before = [
        form(&input.ir, producer).pre.clone(),
        form(&input.ir, consumer).pre.clone(),
    ];
    let result = optimized(input);
    assert_eq!(entry_ops(&result.ir), &[consumer]);
    let mut after = Vec::new();
    functions(&form(&result.ir, consumer).pre, &mut after);
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
    // Cyclic candidates must be skipped without changing the graph.
    // Adjacent contractions can consume this chain; merging first and last alone
    // would create a cycle through the middle operation.
    let input = imported(
        "entry chain(xs: [4]i32) [4]i32 =
        let a = map(|x: i32| x + 1, xs) in
        let b = map(|x: i32| x * 2, a) in map(|x: i32| x - 3, b)",
    );
    let last = *entry_ops(&input.ir).last().unwrap();
    let result = optimized(input);
    assert_eq!(entry_ops(&result.ir), &[last]);
    let mut bodies = Vec::new();
    functions(&form(&result.ir, last).pre, &mut bodies);
    assert_eq!(bodies.len(), 3);
    let OperationKind::Screma { inputs, .. } = &result.ir.operations[last].kind else {
        unreachable!()
    };
    let [Array::Value(input)] = inputs.as_slice() else {
        panic!("one original input")
    };
    assert!(matches!(
        result.ir.expressions[*input].kind,
        ExprKind::Parameter(_)
    ));
    let again = optimized(Program {
        ir: result.ir.clone(),
        state: Imported {
            graph: EGraph::default(),
        },
    });
    assert_eq!(
        format!("{:?}", result.ir),
        format!("{:?}", again.ir),
        "fixed point"
    );
}

#[test]
fn fused_scan_allocates_output_when_its_unique_input_is_absorbed() {
    // Gather/scan regression fixture.
    let input = imported(include_str!("../../../../testfiles/gather_scan_chain.wyn"));
    let original = entry_ops(&input.ir).to_vec();
    let scan = original.iter().copied().find(|&id| {
        matches!(&input.ir.operations[id].kind, OperationKind::Screma { form, .. } if !form.scans.is_empty())
    }).unwrap();
    let OperationKind::Screma { reuse_inputs, .. } = &input.ir.operations[scan].kind else {
        unreachable!()
    };
    assert_eq!(reuse_inputs, &[Some(0)]);
    let result = optimized(input);
    assert_eq!(entry_ops(&result.ir).len(), original.len() - 1);
    let OperationKind::Screma {
        reuse_inputs, form, ..
    } = &result.ir.operations[scan].kind
    else {
        unreachable!()
    };
    assert_eq!(reuse_inputs, &[None]);
    assert_eq!(form.scans.len(), 1);
    assert_eq!(form.scans[0].neutral.len(), 1);
    // The later gather still observes a materialized scan result.
    assert!(entry_ops(&result.ir).contains(&scan));
    assert_eq!(entry_ops(&result.ir).last(), original.last());
}

#[test]
fn conditional_tuple_elements_keep_their_logical_boundaries() {
    // Conditional result handling combined
    // with tuple-sensitive routing: the pair is one element, never two arrays.
    let input = imported(
        "entry pair_sum(xs: [4]i32) (i32, i32) =
        let pairs = map(|x: i32| (if x < 0 then 0 - x else x, x * 2), xs) in
        reduce(|(a,b): (i32,i32), (c,d): (i32,i32)| (a+c,b+d), (0,0), pairs)",
    );
    let result = optimized(input);
    assert_eq!(entry_ops(&result.ir).len(), 1);
    let fused = form(&result.ir, entry_ops(&result.ir)[0]);
    let results = body_signature(&fused.pre).1;
    assert_eq!(results.len(), 1);
    assert!(matches!(
        &result.ir.types[results[0]].ty,
        Type::Constructed(TypeName::Tuple(2), _)
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
fn opaque_barriers_prevent_stream_and_indexed_fusion_without_effect_tokens() {
    // Opaque barriers must survive without explicit effect tokens.
    for source in [
        CHAIN,
        "entry indexed(xs: [4]i32, i: i32) i32 = let a=map(|x:i32|x+1,xs) in a[i]",
        "entry filtered(xs:[4]i32) []i32 = map(|x:i32|x*2,filter(|x:i32|x>0,xs))",
    ] {
        let mut input = imported(source);
        let region = entry(&input.ir);
        let consumer = entry_ops(&input.ir)[1];
        let definition = &input.ir.definitions[input.ir.entries.values().next().unwrap().definition];
        let ty = definition.ty;
        let extern_id = input.ir.externs.alloc(ExternData {
            linkage_name: "opaque_barrier".into(),
        });
        let function = input.ir.expressions.alloc(ExprData {
            ty,
            kind: ExprKind::Extern(extern_id),
        });
        let parameter = input
            .ir
            .expressions
            .iter()
            .find_map(|(&id, value)| {
                matches!(value.kind, ExprKind::Parameter(p) if input.ir.parameters[p].region == region)
                    .then_some(id)
            })
            .unwrap();
        let template = input.ir.operations[entry_ops(&input.ir)[0]].clone();
        let barrier = input.ir.operations.alloc(OperationData {
            kind: OperationKind::Call {
                function,
                args: vec![parameter],
            },
            ..template
        });
        input.ir.regions[region].members.insert(barrier);
        input.ir.operations[consumer].source_position = 2;
        input.ir.operations[barrier].source_position = 1;
        let before = entry_ops(&input.ir).to_vec();
        let result = optimized(input);
        assert_eq!(entry_ops(&result.ir), before);
    }
}

#[test]
fn cross_region_uses_keep_the_producer_materialized() {
    // Cross-region uses remain external observers.
    let input = imported(
        "entry branch(xs: [4]i32, flag: bool) [4]i32 =
        let a = map(|x: i32| x + 1, xs) in
        if flag then map(|x: i32| x * 2, a) else a",
    );
    let before = entry_ops(&input.ir).to_vec();
    let result = optimized(input);
    assert_eq!(entry_ops(&result.ir), before);
    assert_eq!(
        analyze(&result.ir).schedules(&result.ir).unwrap().values().map(Vec::len).sum::<usize>(),
        3
    );
}

#[test]
fn separate_bodies_and_loop_parameters_do_not_alias() {
    // Values must be qualified by body, including loop parameters,
    // with a fusible pair in each scope to exercise the rewrite as well.
    let input = imported(
        "entry first(xs: [4]i32) [4]i32 =
          map(|x: i32| x * 2, map(|x: i32| x + 1, xs))
        entry second(xs: [4]i32) [4]i32 =
          loop values = xs for i < 3 do map(|x: i32| x * 2, map(|x: i32| x + 1, values))",
    );
    let result = optimized(input);
    let mut parameters = Vec::new();
    for (region_id, ops) in analyze(&result.ir).schedules(&result.ir).unwrap() {
        let region = &result.ir.regions[region_id];
        for op in ops {
            if let OperationKind::Screma { inputs, .. } = &result.ir.operations[op].kind {
                let [Array::Value(value)] = inputs.as_slice() else {
                    panic!("one input")
                };
                let ExprKind::Parameter(param) = result.ir.expressions[*value].kind else {
                    panic!("scoped input")
                };
                assert_eq!(
                    result.ir.regions[result.ir.parameters[param].region].definition,
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
            "entry captured(xs:[4]i32) []i32 =
             let a=filter(|x:i32|x>0,xs) in map(|x:i32|x+length(a),a)",
            2,
        ),
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
        let before = entry_ops(&input.ir).to_vec();
        let result = optimized(input);
        // A and B may combine while retaining A, but the final consumer must
        // still run separately because its capture needs the completed array.
        let after = entry_ops(&result.ir);
        assert_eq!(after.len(), expected, "{source}");
        assert_eq!(after.last(), before.last(), "{source}");
    }
}

#[test]
fn reuse_permission_allows_fusion_but_unknown_body_reads_do_not() {
    let mut input = imported(CHAIN);
    let producer = entry_ops(&input.ir)[0];
    let OperationKind::Screma { reuse_inputs, .. } = &mut input.ir.operations[producer].kind else {
        unreachable!()
    };
    reuse_inputs[0] = Some(0);
    assert_eq!(entry_ops(&optimized(input).ir).len(), 1);

    // A source body with an array read is conservatively kept as a separate
    // operation; scalar expression interning does not prove storage-read safety.
    let input = imported(
        "entry reads(xs: [4]i32, ys: [4]i32) [4]i32 =
        let a = map(|x: i32| ys[x], xs) in map(|x: i32| x + 1, a)",
    );
    let before = entry_ops(&input.ir).to_vec();
    assert_eq!(entry_ops(&optimized(input).ir), before);
}

#[test]
fn unreferenced_metadata_and_expressions_do_not_block_fusion() {
    let mut input = imported(CHAIN);
    let producer = entry_ops(&input.ir)[0];
    let consumer = entry_ops(&input.ir)[1];
    let result = input
        .ir
        .expressions
        .iter()
        .find_map(|(&id, expr)| {
            matches!(expr.kind, ExprKind::OperationResult(op) if op == producer).then_some(id)
        })
        .unwrap();
    input.ir.expressions.alloc(ExprData {
        ty: input.ir.expressions[result].ty,
        kind: ExprKind::Tuple(vec![result, result]),
    });
    assert_eq!(entry_ops(&optimized(input).ir), &[consumer]);
}

#[test]
fn scalar_builtin_values_fuse_but_execution_dependent_builtins_do_not() {
    for (builtin, expected) in [("abs", 1), ("d_fdx", 2), ("d_fdy", 2), ("fwidth", 2)] {
        let input = imported(&format!(
            "open f32\nentry values(xs: [4]f32) [4]f32 =
              let a = map(|x: f32| {builtin}(x), xs) in map(|x: f32| x + 1.0, a)"
        ));
        assert_eq!(entry_ops(&optimized(input).ir).len(), expected, "{builtin}");
    }
}
