use crate::egglog::dataflow::{
    edit,
    ir::{self, Bounds, Code, Name},
};
use crate::egglog::graph::Formula;
use crate::egglog::graph::{ArgumentSite, Graph};
use crate::egglog::payload::Payload;
use crate::egglog::query::Query;
use crate::egglog::{analysis, planning, test_graph, OptimizeError};
use crate::op::BinaryOperator;
use crate::types;
use egglog_engine::{RawValues, Write};

#[test]
fn copy_forwarding_preserves_captures_loop_entries_and_backedges() -> Result<(), OptimizeError> {
    let mut graph =
        test_graph::import("entry main(a:i32,b:i32,n:i32) i32 = loop t = a for i < n do t - b")?;
    let query = Query(&graph);
    let root = test_graph::root(&graph);
    let owner = query.first("CountedLoopOperation")?.unwrap()[0];
    let inner = query.0.control_region(owner, 0_i64)?;
    let subtract = query
        .operations(inner)?
        .into_iter()
        .find(|&op| query.0.code(op).unwrap() == Code::Scalar(Payload::Binary(BinaryOperator::Subtract)))
        .expect("loop body has a subtraction");
    let initial = query.0.argument(owner, 0_i64, 0_i64)?;
    let operands = query.0.inputs(subtract);
    let backedge = query.0.yield_value(inner, 0_i64)?;
    let ty = query.required("ValueType", query.name(initial)?)?;
    let sources = [
        query.name(initial)?,
        query.name(operands[0])?,
        query.name(backedge)?,
        query.name(operands[1])?,
    ];
    graph.update(|mut sink| {
        Ok((|| -> Result<(), OptimizeError> {
            for (i, (region, order, source)) in [
                (root, -1_i64, sources[0]),
                (inner, -1, sources[1]),
                (inner, 100, sources[2]),
                (root, -2, sources[3]),
            ]
            .into_iter()
            .enumerate()
            {
                let id = 1000 + i as i64;
                let raw = sink.add("ValueId", id)?;
                let name = Name::Value(raw);
                let copy = ir::operation(&mut sink, id, Code::IndexValue, &[])?;
                let index = sink.add("IndexValue", source)?;
                sink.set_formula(copy.0, Formula::Index, index)?;
                ir::outputs(&mut sink, copy, &[name])?;
                sink.set("ValueType", raw, ty)?;
                sink.set_region(copy.0, region);
                sink.set_order(copy.0, order);
            }
            let mut alias = |id| {
                let raw = sink.add("ValueId", id)?;
                sink.add("ValueNameRef", raw)
            };
            let (start, left, end) = (alias(1000_i64)?, alias(1001_i64)?, alias(1002_i64)?);
            let capture = alias(1003_i64)?;
            sink.set_argument(ArgumentSite(owner, 0, 0), start)?;
            sink.set_input(subtract, 0_i64, left)?;
            sink.set_input(subtract, 1_i64, capture)?;
            sink.set_yield(inner, 0_i64, end)?;
            Ok(())
        })())
    })??;
    analysis::run(&mut graph)?;
    graph.parse_and_run_program(None, "(run-schedule (saturate scalar-analysis))")?;
    assert!(super::forward::run(&mut graph)?);
    analysis::ensure(&mut graph)?;
    let query = Query(&graph);
    assert_eq!(query.0.argument(owner, 0_i64, 0_i64)?, initial);
    assert_eq!(query.0.inputs(subtract), operands);
    assert!(query.contains("RegionCapture", (inner, operands[1]))?);
    assert_eq!(query.0.yield_value(inner, 0_i64)?, backedge);
    for id in 1000_i64..1004 {
        let copy = query.constructor("OperationId", id)?.unwrap();
        assert!(!query.0.has_operation(copy));
    }
    Ok(())
}

#[test]
fn inlining_does_not_run_scalar_candidate_discovery() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(x:i32) i32=x+x")?;
    assert!(Query(&graph).first("ScalarPayload")?.is_some());
    super::inline(&mut graph, true)?;
    assert!(Query(&graph).first("ForwardCandidate")?.is_none());
    Ok(())
}

#[test]
fn inlining_preserves_dependent_calls_and_shape_arguments() -> Result<(), OptimizeError> {
    for reverse in [false, true] {
        let mut calls = vec![(1, Code::Call), (2, Code::Call), (3, Code::Call)];
        if reverse {
            calls.reverse();
        }
        calls.push((4, Code::Range));
        calls.push((5, Code::Scratch));
        let mut graph = test_graph::fixture(&calls, include_str!("tests/inline_arguments.egg"))?;
        super::inline(&mut graph, true)?;
        assert!(
            Query(&graph).first("FlowLive")?.is_none(),
            "inlining must not run liveness"
        );
        analysis::run(&mut graph)?;
        analysis::validate(&graph)?;
        graph.parse_and_run_program(
            None,
            r#"
            (fail (check (LiveOperation call) (CallOperation call)))
            (check (= (wyn-index-expression (OperationId 4)) (IndexValue (ValueId 0))))
            (check (= (wyn-operation-extents (OperationId 5)) (vec-of (IndexValue (ValueId 0)))))
            (check (= (wyn-yield (RegionId 0) 0) (ValueNameRef (ValueId 3))))
            (check (= (ValueType (ValueId 0)) (ScalarInfo (ScalarElement (I32)))))
        "#,
        )?;
    }
    Ok(())
}

#[test]
fn inlining_exposes_parallel_work_through_scalar_helpers_before_copying_them() -> Result<(), OptimizeError>
{
    let mut graph = test_graph::fixture(
        &[
            (1, Code::Call),
            (2, Code::Call),
            (3, Code::Call),
            (4, Code::Range),
            (5, Code::Scratch),
            (6, Code::Parallel),
        ],
        include_str!("tests/inline_arguments.egg"),
    )?;
    test_graph::facts_with_positions(
        &mut graph,
        r#"
        (set (FixtureRegion (OperationId 6)) (RegionId 2))
        (set (FixtureOrder (OperationId 6)) 0)

        "#,
    )?;
    super::inline(&mut graph, false)?;
    analysis::structure(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (fail (check (LiveOperation call) (CallOperation call)))
        (let prepared (ClonedOperation (OperationId 3) (OperationId 6)))
        (check (ParallelOperation (ClonedOperation (OperationId 1) prepared)))
        (check (ParallelOperation (ClonedOperation (OperationId 2) prepared)))
        "#,
    )?;
    Ok(())
}

#[test]
fn nested_calls_mark_only_the_reachable_cycle_recursive() -> Result<(), OptimizeError> {
    let mut graph = test_graph::fixture(
        &[
            (1, Code::Call),
            (2, Code::Call),
            (3, Code::Call),
            (4, Code::Range),
            (5, Code::Scratch),
            (6, Code::Call),
        ],
        include_str!("tests/inline_arguments.egg"),
    )?;
    test_graph::facts_with_positions(
        &mut graph,
        r#"
        (set (FixtureResultReducer (OperationId 5) 0) (RegionId 10))
        (set (FixtureRegion (OperationId 6)) (RegionId 10))
        (set (FixtureCallTarget (OperationId 6)) (FunctionId 2 "identity"))
        (set (FixtureAccumulatorCombiner (OperationId 6) 0) (RegionId 11))
        (delete (FixtureRegion (OperationId 2)))
        (set (FixtureRegion (OperationId 2)) (RegionId 11))
        (delete (FixtureRegion (OperationId 1)))
        (set (FixtureRegion (OperationId 1)) (RegionId 2))
        (set (FixtureFunctionRoot (FunctionId 3 "unused")) (RegionId 12))
    "#,
    )?;
    let query = Query(&graph);
    let main = query.constructor("FunctionId", (0_i64, "main"))?.unwrap();
    let reachability = analysis::calls::Reachability::read(&graph)?;
    let sites = &reachability.calls.calls[&main];
    for (operation, function) in [(6_i64, (2_i64, "identity")), (2_i64, (1_i64, "forward"))] {
        let operation = query.constructor("OperationId", operation)?.unwrap();
        let callee = query.constructor("FunctionId", function)?.unwrap();
        assert!(sites.iter().any(|site| site.operation == operation && site.callee == callee));
    }
    analysis::structure(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (check (= (RecursiveFunction (FunctionId 0 "main")) false))
        (check (= (RecursiveFunction (FunctionId 1 "forward")) true))
        (check (= (RecursiveFunction (FunctionId 2 "identity")) true))
        (fail (check (LiveFunction (FunctionId 3 "unused"))))
        (set (RetainAllFunctions) true)
    "#,
    )?;
    analysis::structure(&mut graph)?;
    graph.parse_and_run_program(None, r#"(check (LiveFunction (FunctionId 3 "unused")))"#)?;
    Ok(())
}

#[test]
fn inline_all_leaves_recursive_calls_intact() -> Result<(), OptimizeError> {
    let mut graph = test_graph::fixture(
        &[
            (1, Code::Call),
            (2, Code::Call),
            (3, Code::Call),
            (4, Code::Range),
            (5, Code::Scratch),
        ],
        include_str!("tests/inline_arguments.egg"),
    )?;
    test_graph::facts_with_positions(
        &mut graph,
        r#"
        (set (FixtureCallTarget (OperationId 3)) (FunctionId 1 "forward"))
        "#,
    )?;
    let revision = edit::revision(&graph)?;
    super::inline(&mut graph, true)?;
    assert_eq!(edit::revision(&graph)?, revision);
    Ok(())
}

#[test]
fn structural_cloning_renames_storage_shapes_without_analysis() -> Result<(), OptimizeError> {
    let mut graph = test_graph::fixture(
        &[
            (1, Code::Call),
            (2, Code::Call),
            (3, Code::Call),
            (4, Code::Range),
            (5, Code::Scratch),
        ],
        include_str!("tests/inline_arguments.egg"),
    )?;
    let query = Query(&graph);
    assert!(query.first("FlowScopeAncestor")?.is_none());
    let root = query.constructor("RegionId", 0_i64)?.unwrap();
    let context = query.constructor("OperationId", 1_i64)?.unwrap();
    let storage = query.constructor("OperationId", 5_i64)?.unwrap();
    let arguments = query.0.parameters(root);
    edit::clone_region(&mut graph, context, root, &arguments)?;
    let query = Query(&graph);
    let cloned = query.constructor("ClonedOperation", (context, storage))?.unwrap();
    let shape = query.vector(query.0.formula(cloned, Formula::Extents)?)?;
    let index = query.fields("IndexValue", shape[0])?[0];
    let expected = query
        .constructor(
            "ClonedValue",
            (context, query.constructor("ValueId", 2_i64)?.unwrap()),
        )?
        .unwrap();
    assert_eq!(index, expected);
    assert!(query.first("FlowLive")?.is_none());
    Ok(())
}

#[test]
fn structural_cloning_follows_nested_regions_without_analysis() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        "entry main(xs:[]i32,n:i32,m:i32,b:bool) i32=if b then reduce((+),0,xs[0..n]) else m",
    )?;
    let query = Query(&graph);
    assert!(query.first("FlowScopeAncestor")?.is_none());
    let root = test_graph::root(&graph);
    let branch = query.first("BranchOperation")?.unwrap()[0];
    let view = query.first("ViewOperation")?.unwrap()[0];
    let slices = query.0.formula(view, Formula::Slices)?;
    let returns = query.0.yield_values(root);
    let mut arguments = query.0.parameters(root);
    arguments[1] = arguments[2];
    let cloned_root = edit::clone_region(&mut graph, branch, root, &arguments)?.region;
    let query = Query(&graph);
    let copied_branch = query.constructor("ClonedOperation", (branch, branch))?.unwrap();
    let original_arm = query.0.control_region(branch, 0_i64)?;
    let copied_arm = query.0.control_region(copied_branch, 0_i64)?;
    assert_eq!(
        copied_arm,
        query.constructor("ClonedRegion", (branch, original_arm))?.unwrap()
    );
    let copied_view = query.constructor("ClonedOperation", (branch, view))?.unwrap();
    assert_eq!(query.0.operation_region(copied_view)?, copied_arm);
    assert_ne!(query.0.formula(copied_view, Formula::Slices)?, slices);
    assert!(query.first("FlowLive")?.is_none());
    let results = query.0.yield_values(cloned_root);
    let operations = query.operations(cloned_root)?;
    edit::replace(&mut graph, &returns.into_iter().zip(results).collect::<Vec<_>>())?;
    edit::splice(&mut graph, branch, &operations)?;
    analysis::run(&mut graph)?;
    Ok(())
}

#[test]
fn keyed_positions_keep_multi_result_occurrences_distinct() -> Result<(), OptimizeError> {
    let graph = test_graph::fixture(
        &[
            (
                0,
                Code::Constant {
                    ty: types::i32(),
                    bits: 7,
                },
            ),
            (
                1,
                Code::Constant {
                    ty: types::i32(),
                    bits: 7,
                },
            ),
        ],
        r#"

        (set (FixtureOutput (OperationId 0) 0) (ValueNameRef (ValueId 9000)))
        (set (FixtureOutput (OperationId 0) 1) (ValueNameRef (ValueId 9001)))

        (set (FixtureOutput (OperationId 1) 0) (ValueNameRef (ValueId 9002)))
    "#,
    )?;
    let query = Query(&graph);
    let a = query.constructor("OperationId", 0_i64)?.unwrap();
    let b = query.constructor("OperationId", 1_i64)?.unwrap();
    assert_ne!(a, b);
    assert_eq!(graph.code(a)?, graph.code(b)?);
    assert_eq!(query.0.outputs(a).len(), 2);
    assert_eq!(query.0.outputs(b).len(), 1);
    assert_ne!(query.0.output(a, 0_i64)?, query.0.output(a, 1_i64)?);
    assert!(query.0.output_at(a, -1_i64).is_none());
    assert!(query.0.output_at(a, 2_i64).is_none());
    Ok(())
}

#[test]
fn unused_values_follow_complete_uses_and_the_current_program() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(x:i32) i32=x+7")?;
    super::validate(&mut graph)?;
    let function = test_graph::entry(&graph);
    let root = test_graph::root(&graph);
    let outputs = Query(&graph).0.yield_values(root);
    super::simplify(&mut graph, false)?;
    assert_eq!(Query(&graph).0.yield_values(root), outputs);
    assert!(!Query(&graph).operations(root)?.is_empty());
    graph.update(|mut sink| {
        sink.set_yields(root, Vec::new())?;
        sink.remove("RegionReturnType", (root, 0_i64))?;
        sink.remove("FunctionResultCount", function)?;
        sink.set("FunctionResultCount", function, 0_i64)?;
        sink.remove("FunctionResultType", (function, 0_i64))?;
        Ok(())
    })?;
    super::simplify(&mut graph, false)?;
    assert!(Query(&graph).operations(root)?.is_empty());
    Ok(())
}

#[test]
fn substitution_renames_accumulator_shapes_without_analysis() -> Result<(), OptimizeError> {
    let mut graph = test_graph::fixture(
        &[],
        r#"
        (let original (ValueNameRef (ValueId 0)))
        (let replacement (ValueNameRef (ValueId 1)))
        (let info (AccumulatorInfo (vec-of (IndexValue (ValueId 0))) (vec-of (ScalarElement (I32)))))
        (set (AccumulatorType (AccumulatorId 0)) info)
        (set (RegionReturnType (RegionId 0) 0) (ReturnAccumulator info))
        (set (FunctionResultType (FunctionId 0 "main") 0) (ReturnAccumulator info))
        (set (CallReturnType (OperationId 0) 0) (ReturnAccumulator info))
        "#,
    )?;
    let query = Query(&graph);
    let old = query.constructor("ValueNameRef", query.constructor("ValueId", 0_i64)?.unwrap())?.unwrap();
    let new = Query(&graph)
        .constructor(
            "ValueNameRef",
            Query(&graph).constructor("ValueId", 1_i64)?.unwrap(),
        )?
        .unwrap();
    edit::replace(&mut graph, &[(old, new)])?;
    graph.parse_and_run_program(
        None,
        r#"
        (let expected (AccumulatorInfo (vec-of (IndexValue (ValueId 1))) (vec-of (ScalarElement (I32)))))
        (check (= (AccumulatorType (AccumulatorId 0)) expected))
        (check (= (RegionReturnType (RegionId 0) 0) (ReturnAccumulator expected)))
        (check (= (FunctionResultType (FunctionId 0 "main") 0) (ReturnAccumulator expected)))
        (check (= (CallReturnType (OperationId 0) 0) (ReturnAccumulator expected)))
        "#,
    )?;
    assert!(Query(&graph).first("FlowLive")?.is_none());
    Ok(())
}

#[test]
fn substitution_preserves_static_types_and_renames_nested_indices() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(xs:[][4]i32,start:i32,other:i32) [][4]i32=xs[start..]")?;
    analysis::run(&mut graph)?;
    let root = test_graph::root(&graph);
    let query = Query(&graph);
    let start = query.0.parameter(root, 1_i64)?;
    let other = query.0.parameter(root, 2_i64)?;
    let view = query.first("ViewOperation")?.unwrap()[0];
    let output = query.name(query.0.output(view, 0_i64)?)?;
    let info = query.required("ValueType", output)?;
    edit::replace(&mut graph, &[(start, other)])?;
    analysis::run(&mut graph)?;
    let query = Query(&graph);
    assert_eq!(query.required("ValueType", output)?, info);
    assert!(query.contains("DirectOperationUse", (view, other))?);
    assert!(!query.contains("DirectOperationUse", (view, start))?);
    Ok(())
}

#[test]
fn applied_rewrites_do_not_replay_when_another_pass_revisits_the_same_body() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(x:i32,y:i32,z:i32) i32=x+1")?;
    analysis::run(&mut graph)?;
    let root = test_graph::root(&graph);
    let query = Query(&graph);
    let x = query.0.parameter(root, 0_i64)?;
    let y = query.0.parameter(root, 1_i64)?;
    let z = query.0.parameter(root, 2_i64)?;
    edit::replace(&mut graph, &[(x, y)])?;
    analysis::run(&mut graph)?;
    edit::replace(&mut graph, &[(y, z)])?;
    analysis::run(&mut graph)?;
    let revision = edit::revision(&graph)?;
    graph.parse_and_run_program(None, "(run-schedule (saturate graph-candidates graph-apply))")?;
    assert_eq!(edit::revision(&graph)?, revision);
    let operation = graph.codes().next().unwrap().0;
    assert!(!Query(&graph).contains("DirectOperationUse", (operation, x))?);
    assert!(!Query(&graph).contains("DirectOperationUse", (operation, y))?);
    Ok(())
}

#[test]
fn reduction_combiner_cannot_capture_a_lane_index() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(xs:[4]i32) i32=reduce((+),0,xs)")?;
    analysis::run(&mut graph)?;
    let query = Query(&graph);
    let parallel = query.first("ParallelOperation")?.unwrap()[0];
    let domain = query.0.operation_domain(parallel)?;
    let index = query.0.domain_axis(domain, 0_i64)?;
    let reducer = query.0.reducer(parallel, 0_i64)?;
    graph.update(|mut sink| {
        let value = sink.add("ValueNameRef", index)?;
        sink.set_yield(reducer, 0_i64, value)
    })?;
    assert!(super::validate(&mut graph).is_err());
    Ok(())
}

fn fusion_case(pointwise: bool) -> Result<Graph, OptimizeError> {
    let consumer = if pointwise { "map(|x|x+2,a)" } else { "map(|x|a[0],xs)" };
    let source =
        format!("entry main(xs:[4]i32) ([4]i32,[4]i32)=let a=map(|x|x+1,xs) in let b={consumer} in (a,b)");
    test_graph::import(&source)
}

#[test]
fn fusion_forwards_pointwise_reads_and_retains_both_results() -> Result<(), OptimizeError> {
    let mut graph = fusion_case(true)?;
    super::simplify(&mut graph, false)?;
    crate::egglog::fusion::run(&mut graph)?;
    super::validate(&mut graph)?;
    let query = Query(&graph);
    let mut results = 0;
    let mut parallels = 0;
    query.for_each("LiveOperation", |row| {
        if query.contains("ParallelOperation", row[0])? {
            parallels += 1;
            results += query.0.output_count(row[0]);
        }
        Ok(())
    })?;
    assert_eq!(results, 2);
    assert_eq!(parallels, 1);
    let mut reads = 0;
    query.for_each("LiveOperation", |row| {
        if query.contains("IndexOperation", row[0])? {
            let source = query.0.input(row[0], 0_i64)?;
            query.for_each("ParallelOperation", |row| {
                reads += usize::from(query.0.outputs(row[0]).contains(&source));
                Ok(())
            })?;
        }
        Ok(())
    })?;
    assert_eq!(reads, 0, "pointwise loads of unfinished arrays are redirected");
    Ok(())
}

#[test]
fn fusion_keeps_the_reducer_with_its_result() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        "entry main(xs:[8]i32) ([8]i32,i32) =
            let mapped = map(|x| x + 1, xs) in (mapped, reduce((+), 0, mapped))",
    )?;
    let root = test_graph::root(&graph);
    let initial = Query(&graph);
    let parallels: Vec<_> = initial
        .operations(root)?
        .into_iter()
        .filter(|&op| initial.contains("ParallelOperation", op).unwrap())
        .collect();
    let [producer, consumer] = parallels.as_slice() else {
        panic!("expected a separate producer and reduction before fusion");
    };
    let reducer = initial.0.reducer(*consumer, 0_i64)?;
    crate::egglog::fusion::run(&mut graph)?;
    super::validate(&mut graph)?;
    let query = Query(&graph);
    assert!(query.contains("LiveOperation", *producer)?);
    assert!(!query.contains("LiveOperation", *consumer)?);
    assert_eq!(query.0.reducer(*producer, 1_i64)?, reducer);
    assert_eq!(query.0.output_count(*producer), 2);
    Ok(())
}

#[test]
fn fusion_preserves_nonpointwise_dependencies() -> Result<(), OptimizeError> {
    let mut graph = fusion_case(false)?;
    super::simplify(&mut graph, false)?;
    crate::egglog::fusion::run(&mut graph)?;
    let query = Query(&graph);
    let mut parallels = 0;
    query.for_each("LiveOperation", |row| {
        parallels += usize::from(query.contains("ParallelOperation", row[0])?);
        Ok(())
    })?;
    assert_eq!(parallels, 2);
    Ok(())
}

#[test]
fn source_import_builds_closed_scalar_control_and_parallel_bodies() -> Result<(), OptimizeError> {
    for text in [
        "entry main(x:i32) i32 = x + 1",
        "entry main(x:i32) i32 = if x > 0 then x + 1 else x - 1",
        "entry main(xs:[64]i32) [64]i32 = map(|x|x+1,xs)",
        "entry main(xs:[64]i32) i32 = reduce((+),0i32,map(|x|x+1,xs))",
    ] {
        let source = crate::compile_thru_tlc(text).unwrap();
        let (mut graph, _) = super::super::from_tlc::import(&source)?;
        super::validate(&mut graph)?;
        super::super::fusion::run(&mut graph)?;
        super::validate(&mut graph)?;
        assert!(Query(&graph).first("InvalidDataflow")?.is_none());
    }
    Ok(())
}

#[test]
fn logical_import_preserves_storage_products_and_rectangular_rows() -> Result<(), OptimizeError> {
    for text in [
        "entry main(xs:*[4](i32,i32)) (i32,i32) = xs[1]",
        "entry main(xs:*[4](i32,i32)) [4]i32 = map(|p|p.0+p.1,xs)",
        "entry main(xs:*[4](i32,i32)) [4](i32,i32) = xs with [1] = (3,4)",
        "entry main(xs:*[2][3](i32,i32)) (i32,i32) = xs[1][2]",
        "entry main() [2][2]i32 = [[1,2],[3,4]]",
    ] {
        let source = crate::compile_thru_tlc(text).unwrap();
        let (mut graph, _) = super::super::from_tlc::import(&source)
            .map_err(|error| crate::egglog::output_error(format!("{text}: {error}")))?;
        super::validate(&mut graph)?;
        assert!(Query(&graph).any("ValueType", |row| Query(&graph)
            .enode("LogicalValue", *row.last().unwrap())
            .unwrap()
            .is_some())?);
    }
    Ok(())
}

#[test]
fn execution_planning_preserves_logical_array_types() -> Result<(), OptimizeError> {
    for text in [
        "#[linked(\"helper_impl\")] extern helper(xs:[4]i32) [4]i32\nentry main(xs:[4]i32) [4]i32 = helper(xs)",
        "entry main(xs:[64]i32) [64]i32 = map(|x|x*2,map(|x|x+1,xs))",
        "entry main(xs:[4]i32,flag:bool) []i32 = if flag then xs[1..3] else xs[0..2]",
        "entry main(xs:[4]i32,n:i32) [4]i32 = loop a = xs for i < n do a with [0] = i",
        "entry main(xs:[64]i32) []i32 = filter(|x|x>0,xs)",
        "entry main(dest:[3]i32,xs:[4]i32) [3]i32 = reduce_by_index(dest,(+),0i32,[0,1,1,2],xs)",
    ] {
        let source = crate::compile_thru_tlc(text).unwrap();
        let (mut graph, _) = super::super::from_tlc::import(&source)?;
        super::validate(&mut graph)?;
        super::super::fusion::run(&mut graph)?;
        crate::egglog::planning::place(&mut graph, crate::PipelineTopologyPolicy::AllowGenerated, crate::egglog::cost::CostModel::default())?;
        crate::egglog::planning::schedule(&mut graph)?;
        assert!(Query(&graph).any("ValueType",|row| Query(&graph).enode("LogicalValue",*row.last().unwrap()).unwrap().is_some())?);
    }
    Ok(())
}

#[test]
fn different_parameter_names_do_not_prove_disjoint_memory() -> Result<(), OptimizeError> {
    let mut graph = test_graph::fixture(
        &[],
        r#"
        (let array (LogicalValue (ArrayType (StorageArray) (vec-of (FixedExtent 4)) (ElementValue (ScalarElement (I32))))))
        (set (ValueType (ValueId 0)) array) (set (ValueType (ValueId 1)) array)
        (set (FixtureParameter (RegionId 0) 0) (ValueNameRef (ValueId 0)))
        (set (FixtureParameter (RegionId 0) 1) (ValueNameRef (ValueId 1)))
        (set (FixtureInput (OperationId 0) 0) (ValueNameRef (ValueId 0))) (WritesInput (OperationId 0) 0)
        (set (FixtureInput (OperationId 1) 0) (ValueNameRef (ValueId 1))) (ReadsInput (OperationId 1) 0)
        (EffectCompare (OperationId 0) (OperationId 1))
        (run-schedule (saturate dataflow-resources) (saturate dataflow-effects))
        (check (OperationsConflict (OperationId 0) (OperationId 1)))
    "#,
    )?;
    graph.parse_and_run_program(None,"(check (ResourceRoot (ValueNameRef (ValueId 0)) (ExternalMemory))) (check (ResourceRoot (ValueNameRef (ValueId 1)) (ExternalMemory)))")?;
    Ok(())
}

#[test]
fn consumption_follows_product_components_and_excludes_shape_witnesses() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        r#"
        #[linked("consume_pair_impl")] extern consume_pair(p: (*[4]i32, [4]i32)) i32
        entry main(x:*[4]i32,y:*[4]i32) i32=consume_pair((x,y))
    "#,
    )?;
    super::validate(&mut graph)?;
    let query = Query(&graph);
    let [call] = query.first("CallOperation")?.unwrap().try_into().unwrap();
    let parameters = graph.external_parameters(graph.call_target(call)?)?;
    assert_eq!(parameters.len(), 1);
    assert_eq!(parameters[0].diet, query.required("InputDiet", (call, 0_i64))?);
    graph.parse_and_run_program(
        None,
        r#"
        (check (CallOperation operation) (= (wyn-input-count operation) 1)
            (= (InputDiet operation 0) (ProductDiet (vec-of (Consume) (Observe)))))
    "#,
    )?;
    Ok(())
}

#[test]
fn read_only_indexing_keeps_its_guard_without_a_domain_bounds_proof() -> Result<(), OptimizeError> {
    for text in [
        "entry main(xs:[]i32,i:i32,b:bool) i32=if b then xs[i] else 0",
        "entry main(xs:[]i32,i:i32,n:i32) []i32=map(|j|xs[i]+j,iota(n))",
    ] {
        let source = crate::compile_thru_tlc(text).unwrap();
        let program = crate::egglog::fuse(crate::egglog::from_tlc(&source)?)?;
        let query = Query(&program.graph);
        let root = test_graph::root(&program.graph);
        let mut guarded = 0;
        query.for_each("LiveOperation", |row| {
            if query.contains("IndexOperation", row[0])? && query.contains("OperationTrap", row[0])? {
                assert_ne!(query.0.operation_region(row[0])?, root);
                guarded += 1;
            }
            Ok(())
        })?;
        assert!(guarded > 0);
    }
    Ok(())
}

#[test]
fn allocation_bounds_do_not_assume_integer_sums_cannot_overflow() -> Result<(), OptimizeError> {
    for limit in [1, 2147483647] {
        let text = format!("entry main(b:bool) i32=reduce((+),0,map(|i|if b then {limit} else 0,iota(8)))");
        let mut graph = test_graph::import(&text)?;
        super::simplify(&mut graph, false)?;
        crate::egglog::fusion::run(&mut graph)?;
        planning::analyze(&mut graph)?;
        let query = Query(&graph);
        let output = query.name(query.0.yield_value(test_graph::root(&graph), 0_i64)?)?;
        let index = graph.update(|mut sink| sink.add("IndexValue", output))?;
        let bound =
            Query(&graph).lookup("IndexUpper", index)?.map(|value| graph.value_to_base::<i64>(value));
        assert_eq!(bound, if limit == 1 { Some(8) } else { None });
    }
    Ok(())
}

#[test]
fn range_lengths_clamp_negative_counts_before_shape_planning() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main() i32=length(iota(-3))")?;
    super::simplify(&mut graph, false)?;
    let query = Query(&graph);
    let result = query.0.yield_value(test_graph::root(&graph), 0_i64)?;
    assert_eq!(
        graph.value_to_base::<i64>(query.required("ConstantValue", result)?),
        0
    );

    let mut graph = test_graph::import("entry main(n:i32) i32=length(iota(n))")?;
    super::simplify(&mut graph, false)?;
    let query = Query(&graph);
    let calculation = query.first("IndexCalculationOperation")?.unwrap()[0];
    let extent = query.0.formula(calculation, Formula::Index)?;
    let max = query.fields("IndexMax", extent)?;
    assert_eq!(
        graph.value_to_base::<i64>(query.fields("IndexConstant", max[0])?[0]),
        0
    );
    Ok(())
}

#[test]
fn consumed_tokens_cannot_be_reused() -> Result<(), OptimizeError> {
    let mut graph = test_graph::fixture(
        &[
            (
                0,
                Code::Accumulate {
                    bounds: Bounds::Proven,
                },
            ),
            (
                1,
                Code::Accumulate {
                    bounds: Bounds::Proven,
                },
            ),
        ],
        r#"
        (let info (AccumulatorInfo (vec-of (IndexConstant 4)) (vec-of (ScalarElement (I32)))))
        (set (AccumulatorType (AccumulatorId 0)) info) (set (AccumulatorType (AccumulatorId 1)) info) (set (AccumulatorType (AccumulatorId 2)) info)
         (FlowScope (RegionId 0))
        (set (FixtureParameter (RegionId 0) 0) (AccNameRef (AccumulatorId 0)))
        (set (ParameterDiet (RegionId 0) 0) (Consume))
        (set (FixtureInput (OperationId 0) 0) (AccNameRef (AccumulatorId 0)))
        (set (FixtureOutput (OperationId 0) 0) (AccNameRef (AccumulatorId 1)))
        (set (FixtureInput (OperationId 1) 0) (AccNameRef (AccumulatorId 0)))
        (set (FixtureOutput (OperationId 1) 0) (AccNameRef (AccumulatorId 2)))
        (set (FixtureRegion (OperationId 0)) (RegionId 0)) (set (FixtureOrder (OperationId 0)) 0)
        (set (FixtureRegion (OperationId 1)) (RegionId 0)) (set (FixtureOrder (OperationId 1)) 1)
         (set (FixtureFunctionRoot (FunctionId 0 "invalid")) (RegionId 0))
        (set (FunctionResultCount (FunctionId 0 "invalid")) 0) (set (FixtureEntryOrder (FunctionId 0 "invalid")) 0)
    "#,
    )?;
    let error = super::validate(&mut graph).expect_err("consumed capability reuse");
    assert!(error.to_string().contains("consumed"), "{error}");
    Ok(())
}

#[test]
fn validation_checks_live_functions_and_ignores_retired_programs() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(x:i32) i32=x")?;
    let root = test_graph::root(&graph);
    let function = test_graph::entry(&graph);
    graph.update(|mut sink| {
        let missing = sink.add("ValueId", 999_i64)?;
        let name = sink.add("ValueNameRef", missing)?;
        let dead = sink.add("FunctionId", (999_i64, "dead"))?;
        let dead_root = sink.add("RegionId", 999_i64)?;
        sink.set_function_root(dead, dead_root);
        sink.set_yields(dead_root, vec![name])?;
        Ok(())
    })?;
    super::validate(&mut graph)?;
    let original = Query(&graph).0.parameter(root, 0_i64)?;
    graph.update(|mut sink| {
        sink.set_parameters(root, vec![original, original])?;
        let observe = sink.add("Observe", RawValues(Vec::new()))?;
        sink.set("ParameterDiet", (root, 1_i64), observe)?;
        Ok(())
    })?;
    let error = super::validate(&mut graph).expect_err("duplicate binders");
    assert!(error.to_string().contains("more than one definition"), "{error}");
    graph.update(|mut sink| {
        sink.remove("ParameterDiet", (root, 1_i64))?;
        sink.set_parameters(root, vec![original])?;
        sink.remove_entry(function);
        Ok(())
    })?;
    super::validate(&mut graph)?;
    Ok(())
}

#[test]
fn accumulator_name_filter_preserves_only_capabilities() -> Result<(), OptimizeError> {
    let graph = test_graph::fixture(
        &[],
        r#"
        (function FixtureName (i64) Name :no-merge)
        (set (FixtureName 0) (ValueNameRef (ValueId 0))) (set (FixtureName 1) (AccNameRef (AccumulatorId 2)))
        (set (FixtureName 2) (ValueNameRef (ValueId 3))) (set (FixtureName 3) (AccNameRef (AccumulatorId 4)))
    "#,
    )?;
    let query = Query(&graph);
    let mut tokens = 0;
    query.for_function("FixtureName", |_, name| {
        tokens += usize::from(query.value_name(name)?.is_none());
        Ok(())
    })?;
    assert_eq!(tokens, 2);
    Ok(())
}

#[test]
fn invariant_auditing_is_enabled_only_in_debug_builds() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(x:i32) i32=x")?;
    let root = test_graph::root(&graph);
    graph.update(|mut sink| {
        let value = sink.add("ValueId", 999_i64)?;
        let name = sink.add("ValueNameRef", value)?;
        sink.set_yield(root, 0_i64, name)
    })?;
    assert_eq!(super::validate(&mut graph).is_err(), cfg!(debug_assertions));
    Ok(())
}

#[test]
fn borrowed_vector_indexing_preserves_bounds_and_sort_overloads() -> Result<(), OptimizeError> {
    let mut graph = test_graph::fixture(
        &[],
        r#"
        (sort FixtureNames (Vec ValueName))
        (let numbers (vec-of 7 19)) (let names (vec-of (ValueId 3) (ValueId 8)))
        (check (= (vec-get numbers 0) 7)) (check (= (vec-get numbers 1) 19))
        (check (= (vec-get names 1) (ValueId 8)))
        (fail (check (= (vec-get numbers -1) 7))) (fail (check (= (vec-get numbers 2) 7)))
        (let empty (vec-pop (vec-of 7))) (fail (check (= (vec-get empty 0) 7)))
        (relation IndexedNumber (i64)) (ruleset vector-read)
        (rule ((= value (vec-get numbers 1))) ((IndexedNumber value)) :ruleset vector-read)
        (run-schedule vector-read) (check (IndexedNumber 19))
    "#,
    )?;
    graph.parse_and_run_program(None, "(check (IndexedNumber 19))")?;
    Ok(())
}

#[test]
fn compact_bindings_preserve_clause_offsets_and_liveness() -> Result<(), OptimizeError> {
    let graph = test_graph::fixture(&[], include_str!("tests/compact_bindings.egg"))?;
    let query = Query(&graph);
    let first = query.constructor("OperationId", 0_i64)?.unwrap();
    let second = query.constructor("OperationId", 1_i64)?.unwrap();
    let kept = [query.0.output(first, 0_i64)?, query.0.output(second, 0_i64)?];
    assert_eq!(
        super::demand::dead_clause(query, |name| kept.contains(&name))?,
        Some((first, 1))
    );
    Ok(())
}

#[test]
fn collection_extents_preserve_domains_and_clause_offsets() -> Result<(), OptimizeError> {
    let mut graph = test_graph::fixture(&[], include_str!("tests/collection_extents.egg"))?;
    graph.parse_and_run_program(None,"(run-schedule (saturate graph-sizes)) (check (= (LogicalOuterCount (ValueId 11)) (IndexConstant 5))) (check (= (LogicalOuterCount (ValueId 12)) (IndexConstant 5))) (check (= (LogicalOuterCount (ValueId 13)) (IndexConstant 7))) (fail (check (= count (LogicalOuterCount (ValueId 10))))) (fail (check (= count (LogicalOuterCount (ValueId 14)))))")?;
    Ok(())
}

#[test]
fn local_rewrites_batch_functions_without_combining_competing_edits() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        "entry first(x:i32) i32=x+1 entry second(y:i32) i32=y+2 entry untouched(z:i32) i32=z",
    )?;
    analysis::run(&mut graph)?;
    let query = Query(&graph);
    let mut operations = Vec::new();
    query.for_each("LiveOperation", |row| {
        if query.lookup("ScalarPayload", row[0])?.is_some() {
            operations.push(row[0]);
        }
        Ok(())
    })?;
    assert_eq!(operations.len(), 2);
    for (operation, bits) in operations.into_iter().zip([11, 22]) {
        edit::set_code(
            &mut graph,
            operation,
            Code::Constant {
                ty: types::i32(),
                bits,
            },
        )?;
        edit::inputs(&mut graph, operation, &[])?;
    }
    super::simplify(&mut graph, false)?;
    let query = Query(&graph);
    let mut entries = 0;
    crate::egglog::test_graph::for_entries(query, |_, _| {
        entries += 1;
        Ok(())
    })?;
    assert_eq!(entries, 3);
    Ok(())
}
