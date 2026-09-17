use super::insert_expressions;
use crate::compile_thru_tlc;
use crate::egglog::dependencies::analyze;
use crate::egglog::graph_tests::fusion_dependencies;
use crate::egglog::{from_tlc, fuse, ExprKind, Expressions, OperationKind, Program};
use crate::tlc::infer_input_slice_bounds;
use egglog_engine::EGraph;

fn compile(source: &str) -> Program<Expressions> {
    let tlc = infer_input_slice_bounds(compile_thru_tlc(source).unwrap());
    insert_expressions(fuse(from_tlc(&tlc).unwrap()).unwrap()).unwrap()
}

fn graph(result: &Program<Expressions>) -> EGraph {
    let mut graph = result.state.graph.clone();
    graph.parse_and_run_program(None, super::RUN).unwrap();
    graph
}

#[test]
fn inserts_scalar_only_entry_and_derives_its_dependencies() {
    let result = compile("entry main(x: i32) i32 = x + 1");
    let mut graph = graph(&result);
    graph
        .parse_and_run_program(
            None,
            "(check (RegionResult r 0 e) (ExprParameter e p) (RegionParameter r 0 p t) (RegionExpr r e))",
        )
        .unwrap();
    assert!(graph.function_to_dag("Current", usize::MAX, false).is_err());
}

#[test]
fn shares_syntax_across_uses_without_equating_function_parameters() {
    let result = compile(
        "entry first(x: i32) (i32, i32) = (x + 1, x + 1)
        entry second(x: i32) i32 = x + 1",
    );
    let mut graph = graph(&result);
    graph
        .parse_and_run_program(
            None,
            r#"
        (check (EntryRegion a ra) (RegionResult ra 0 (Typed t (Tuple (vec-of e e))))
               (ExprParameter e p) (RegionParameter ra 0 p pt))
        (check (EntryRegion a ra) (EntryRegion b rb) (!= ra rb)
               (RegionExpr ra (Typed t (Int "1"))) (RegionExpr rb (Typed t (Int "1"))))
        (fail (check (EntryRegion a ra) (EntryRegion b rb) (!= ra rb)
                     (RegionParameter ra 0 p ta) (RegionParameter rb 0 p tb)))
    "#,
        )
        .unwrap();
    let apps: Vec<_> =
        result.ir.expressions.values().filter(|e| matches!(e.kind, ExprKind::PureApp { .. })).collect();
    assert_eq!(apps.len(), 2);
}

#[test]
fn inserts_selected_fused_bodies_without_reviving_the_producer() {
    let tlc = infer_input_slice_bounds(
        compile_thru_tlc(
            "entry main(xs: []i32, bias: i32) []i32 = map(|x: i32| x * 2, map(|x: i32| x + bias, xs))",
        )
        .unwrap(),
    );
    let selected = fuse(from_tlc(&tlc).unwrap()).unwrap();
    let live = analyze(&selected.ir).live;
    let dead: Vec<_> =
        selected.ir.operations.iter().filter_map(|(&id, _)| (!live.contains(&id)).then_some(id)).collect();
    assert_eq!(dead.len(), 1);
    let result = insert_expressions(selected).unwrap();
    let mut graph = graph(&result);
    graph
        .parse_and_run_program(
            None,
            &format!(
                "(fail (check (Execution r {} t))) (fail (check (ExprOperation e {})))",
                dead[0].egglog(),
                dead[0].egglog()
            ),
        )
        .unwrap();
    graph
        .parse_and_run_program(
            None,
            "
        (check (OperationBody op (Pre) body) (BodyRegion body ra) (BodyRegion body rb) (!= ra rb))
        (check (EntryRegion id r) (RegionParameter r 1 bias t) (OperationParameter op bias))
    ",
        )
        .unwrap();
    assert_eq!(
        graph.function_to_dag("Execution", usize::MAX, false).unwrap().0.len(),
        1
    );
    // Fusion still uses its original summary and has no expression declarations.
    let fusion = fusion_dependencies(&result.ir);
    assert!(fusion.function_to_dag("Typed", 1, false).is_err());
}

#[test]
fn loop_carried_parameters_are_dependencies_inside_the_loop_only() {
    let result = compile(
        "entry main(xs: [4]i32, n: i32) [4]i32 =
        loop acc = xs for i < n do map(|x: i32| x + i, acc)",
    );
    let (loop_id, header, body) = result
        .ir
        .operations
        .iter()
        .find_map(|(&id, op)| {
            if let OperationKind::Loop { header, body, .. } = op.kind {
                Some((id, header, body))
            } else {
                None
            }
        })
        .unwrap();
    let mut graph = graph(&result);
    graph
        .parse_and_run_program(
            None,
            &format!(
                "
        (check (LoopRegions {} {} {})
               (RegionParameter {} 0 acc t) (Execution {} child ct) (OperationParameter child acc))
        (fail (check (RegionParameter {} i p t) (OperationParameter {} p)))
        (fail (check (Execution {} child t) (OperationDependency {} child)))
        (check (EntryRegion id outer) (RegionParameter outer 1 n t) (OperationParameter {} n))
        ",
                loop_id.egglog(),
                header.egglog(),
                body.egglog(),
                header.egglog(),
                body.egglog(),
                header.egglog(),
                loop_id.egglog(),
                body.egglog(),
                loop_id.egglog(),
                loop_id.egglog()
            ),
        )
        .unwrap();
}

#[test]
fn expression_dependencies_retain_read_occurrences_and_effect_order() {
    let result = compile("entry main(xs: []i32, i: i32) i32 = xs[i] + xs[i]");
    let mut graph = graph(&result);
    graph
        .parse_and_run_program(
            None,
            "
        (check (IndexOperands a xs i) (IndexOperands b xs i) (!= a b) (ExecutionOrder a b)
               (RegionResult r 0 result) (ExprOperation result a) (ExprOperation result b))
    ",
        )
        .unwrap();
    assert_eq!(
        graph.function_to_dag("IndexOperands", usize::MAX, false).unwrap().0.len(),
        2
    );
}

#[test]
fn expression_export_starts_at_entries_and_ignores_dead_arena_records() {
    let result = compile(
        "def unused(x: i32) i32 = x + 909
        entry main(xs: []i32) i32 = let _ = map(|x: i32| x + 808, xs) in 7",
    );
    let mut graph = graph(&result);
    graph
        .parse_and_run_program(
            None,
            r#"
        (check (EntryRegion id r) (RegionResult r 0 (Typed t (Int "7"))))
        (fail (check (= e (Typed t (Int "909")))))
        (fail (check (= e (Typed t (Int "808")))))
        (fail (check (Execution r op t)))
    "#,
        )
        .unwrap();
}

#[test]
fn lambda_values_depend_on_captures_without_invoking_their_bodies() {
    // A lexical lambda captures an outer parameter and read result. Its own
    // argument and read are local to invocation, not dependencies of creation.
    let mut graph = EGraph::default();
    graph.parse_and_run_program(None, include_str!("ids.egg")).unwrap();
    graph.parse_and_run_program(None, include_str!("expressions.egg")).unwrap();
    graph
        .parse_and_run_program(
            None,
            "
        (Region (RegionId 0)) (Region (RegionId 1))
        (RegionParent (RegionId 1) (RegionId 0))
        (RegionParameter (RegionId 0) 0 (ParameterId 0) (TypeId 0))
        (RegionParameter (RegionId 1) 0 (ParameterId 1) (TypeId 0))
        (Execution (RegionId 0) (OperationId 0) (TypeId 0))
        (Execution (RegionId 1) (OperationId 1) (TypeId 0))
        (let outer (Typed (TypeId 0) (Parameter (ParameterId 0))))
        (let inner (Typed (TypeId 0) (Parameter (ParameterId 1))))
        (RegionResult (RegionId 1) 0 (Typed (TypeId 1) (Tuple (vec-of outer inner
            (Typed (TypeId 0) (OperationResult (OperationId 0)))
            (Typed (TypeId 0) (OperationResult (OperationId 1)))))))
        (let lambda (Typed (TypeId 2) (Lambda (RegionId 1))))
        (RegionResult (RegionId 0) 0 lambda)
        (run-schedule (saturate (run expressions)))
        (check (ExprParameter lambda (ParameterId 0)) (ExprOperation lambda (OperationId 0)))
        (fail (check (ExprParameter lambda (ParameterId 1))))
        (fail (check (ExprOperation lambda (OperationId 1))))
        (fail (check (Invokes op r)))
    ",
        )
        .unwrap();
}
