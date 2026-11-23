use super::{ensure, invalidate, liveness, run};
use crate::egglog::cost::CostModel;
use crate::egglog::dataflow::edit;
use crate::egglog::dataflow::ir::{operation, outputs, parameters, region, yields, Code, Name, Parameter};
use crate::egglog::execution::Execution;
use crate::egglog::fusion;
use crate::egglog::query::Query;
use crate::egglog::{planning, test_graph, OptimizeError};
use crate::types;
use egglog_engine::{RawValues, Write};

#[test]
fn planning_metadata_survives_candidate_resets_and_tracks_edits() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(x:i32) i32=x+7")?;
    let operation = Query(&graph).first("ConstantOperation")?.unwrap()[0];
    // Introduce the identity after import so frontend folding cannot erase the candidate.
    edit::set_code(
        &mut graph,
        operation,
        Code::Constant {
            ty: types::i32(),
            bits: 0,
        },
    )?;
    planning::analyze(&mut graph)?;
    let query = Query(&graph);
    let output = query.0.output(operation, 0_i64)?;
    let candidate = query.first("ForwardCandidate")?.unwrap();
    let revision = query.required("PlanningRevision", RawValues(Vec::new()))?;
    let constant = query.required("ConstantValue", output)?;

    edit::clear(&mut graph)?;
    assert!(Query(&graph).first("ForwardCandidate")?.is_none());
    assert_eq!(Query(&graph).required("ConstantValue", output)?, constant);
    planning::analyze(&mut graph)?;
    assert_eq!(Query(&graph).first("ForwardCandidate")?, Some(candidate));
    assert_eq!(
        Query(&graph).required("PlanningRevision", RawValues(Vec::new()))?,
        revision
    );

    let epoch = Query(&graph).required("AnalysisRevision", RawValues(Vec::new()))?;
    graph.update(|mut sink| sink.add("IntegerBoundsDemand", output))?;
    planning::analyze(&mut graph)?;
    assert_eq!(
        Query(&graph).required("AnalysisRevision", RawValues(Vec::new()))?,
        epoch
    );
    assert_eq!(Query(&graph).required("IntegerUpper", output)?, constant);

    edit::set_code(
        &mut graph,
        operation,
        Code::Constant {
            ty: types::i32(),
            bits: 7,
        },
    )?;
    planning::analyze(&mut graph)?;
    assert_ne!(
        Query(&graph).required("PlanningRevision", RawValues(Vec::new()))?,
        revision
    );
    assert_eq!(
        graph.value_to_base::<i64>(Query(&graph).required("ConstantValue", output)?),
        7
    );
    assert!(Query(&graph).first("ForwardCandidate")?.is_none());

    let revision = edit::revision(&graph)?;
    invalidate(&mut graph)?;
    assert_eq!(edit::revision(&graph)?, revision);
    assert!(Query(&graph).lookup("PlanningRevision", RawValues(Vec::new()))?.is_none());
    planning::analyze(&mut graph)?;
    assert_eq!(
        graph.value_to_base::<i64>(Query(&graph).required("ConstantValue", output)?),
        7
    );
    Ok(())
}

#[test]
fn dataflow_requests_reuse_the_same_graph_revision() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(x:i32,y:i32) i32=x+y")?;
    let root = test_graph::root(&graph);
    let query = Query(&graph);
    let operation = query.operations(root)?[0];
    let inputs = query.0.inputs(operation);
    ensure(&mut graph)?;
    let revision = Query(&graph).required("AnalysisRevision", RawValues(Vec::new()))?;
    ensure(&mut graph)?;
    assert_eq!(
        Query(&graph).required("AnalysisRevision", RawValues(Vec::new()))?,
        revision
    );
    assert!(Query(&graph).contains("OperationUse", (operation, inputs[0]))?);
    edit::inputs(&mut graph, operation, &[inputs[1], inputs[1]])?;
    ensure(&mut graph)?;
    assert!(!Query(&graph).contains("OperationUse", (operation, inputs[0]))?);
    assert!(Query(&graph).contains("OperationUse", (operation, inputs[1]))?);
    Ok(())
}

#[test]
fn region_order_refreshes_the_changed_body() -> Result<(), OptimizeError> {
    let mut graph =
        test_graph::import("entry main(x:i32,y:i32) (i32,i32)=(x+y,x-y) entry other(x:i32) i32=x+1")?;
    ensure(&mut graph)?;
    let query = Query(&graph);
    let mut roots = Vec::new();
    query.for_function("EntryOrder", |keys, order| {
        roots.push((graph.value_to_base::<i64>(order), query.0.function_root(keys[0])?));
        Ok(())
    })?;
    roots.sort_by_key(|&(order, _)| order);
    let main = roots[0].1;
    let other = roots[1].1;
    let original = query.0.region_order(main)?.to_vec();
    let untouched = query.0.region_order(other)?.to_vec();
    edit::move_before(&mut graph, original[1], original[0])?;
    ensure(&mut graph)?;
    let query = Query(&graph);
    assert_eq!(query.0.region_order(other)?, untouched);
    assert!(query.contains("FirstOperation", (main, original[1]))?);
    assert!(query.contains("LastOperation", (main, *original.last().unwrap()))?);
    assert!(query.contains("NextOperation", (original[1], original[0]))?);
    assert!(!query.contains("NextOperation", (original[0], original[1]))?);
    Ok(())
}

#[test]
fn rewriting_defers_live_after_and_host_classification() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(x:i32) i32=(x+1)*x entry other(x:i32) i32=(x+2)*x")?;
    run(&mut graph)?;
    planning::analyze(&mut graph)?;
    for table in ["FlowLive", "NameLiveAfter", "LaneDependentFunction", "HostRegion"] {
        assert!(
            Query(&graph).first(table)?.is_none(),
            "{table} was calculated before its consumer"
        );
    }
    let root = test_graph::root(&graph);
    liveness(&mut graph, &[root])?;
    let query = Query(&graph);
    query.for_function("EntryOrder", |keys, _| {
        let candidate = query.0.function_root(keys[0])?;
        assert_eq!(query.contains("LivenessRegion", candidate)?, candidate == root);
        Ok(())
    })?;
    assert!(query.first("NameLiveAfter")?.is_some());
    liveness(&mut graph, &[root])?;
    assert_eq!(graph.get_size("LivenessRequest"), 1);
    planning::place(
        &mut graph,
        crate::PipelineTopologyPolicy::AuthoredOnly,
        CostModel::default(),
    )?;
    assert!(Query(&graph).first("LaneDependentFunction")?.is_some());
    assert!(Query(&graph).first("HostRegion")?.is_some());
    Ok(())
}

#[test]
fn loop_and_reducer_metadata_is_computed_by_its_consumers() -> Result<(), OptimizeError> {
    let mut graph =
        test_graph::import("entry main(xs:[]i32,n:i32) i32=loop x=reduce((+),0i32,xs) for i<n do x+i")?;
    run(&mut graph)?;
    planning::analyze(&mut graph)?;
    for table in ["LoopStateBytes", "ReducerState", "NativeAddition"] {
        assert!(
            Query(&graph).first(table)?.is_none(),
            "{table} was computed before use"
        );
    }
    planning::place(
        &mut graph,
        crate::PipelineTopologyPolicy::AuthoredOnly,
        CostModel::default(),
    )?;
    assert!(Query(&graph).first("LoopStateBytes")?.is_some());
    for table in ["ReducerState", "NativeAddition"] {
        assert!(
            Query(&graph).first(table)?.is_none(),
            "{table} was computed during placement"
        );
    }
    planning::schedule(&mut graph)?;
    planning::epilogues(&mut graph)?;
    assert!(Query(&graph).first("ReducerState")?.is_none());
    Execution::select(&mut graph)?;
    assert!(Query(&graph).first("ReducerState")?.is_some());
    assert!(Query(&graph).first("NativeAddition")?.is_some());
    Ok(())
}

#[test]
fn constant_index_arithmetic_does_not_prove_wrapping_counts() {
    let mut graph = fusion::new_graph().unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (let difference (IndexSub (IndexConstant 3) (IndexConstant 1)))
        (let sum (IndexAdd (IndexConstant 2147483647) (IndexConstant 1)))
        (let product (IndexMul (IndexConstant 65536) (IndexConstant 65536)))
        (run-schedule (saturate graph-sizes))
        (check (= (StaticIndex difference) 2))
        (fail (check (= count (StaticIndex sum))))
        (fail (check (= count (StaticIndex product))))
    "#,
        )
        .unwrap();
}

#[test]
fn immutable_term_facts_survive_an_analysis_refresh() -> Result<(), OptimizeError> {
    let mut graph = fusion::new_graph()?;
    test_graph::facts_with_positions(
        &mut graph,
        r#"
        (let n (ValueId 1))
        (let index (IndexAdd (IndexValue n) (IndexConstant 2)))
        (let element (ScalarElement (I32)))
        (set (FixtureOperationShape (OperationId 1)) (vec-of index))
        (set (FixtureOperationSlices (OperationId 1)) (vec-of (At index)))
        (let shape (FixtureOperationShape (OperationId 1)))
        (let slices (FixtureOperationSlices (OperationId 1)))
        (run-schedule (saturate graph-indices dataflow-dependencies type-sizes))
        "#,
    )?;
    invalidate(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (check (IndexUse index n))
        (check (= (ShapeAt shape 0) index) (ShapeUse shape n))
        (check (= (SliceAt slices 0) (At index)) (SliceUse slices n))
        (check (= (ElementBytes element) (bigint 4)))
        "#,
    )?;
    let key = Query(&graph).constructor("OperationId", 1_i64)?.unwrap();
    graph.update(|mut sink| {
        sink.remove_formula(key, crate::egglog::graph::Formula::Slices);
        Ok(())
    })?;
    test_graph::facts_with_positions(
        &mut graph,
        r#"
        (let m (ValueId 2))
        (set (FixtureOperationShape (OperationId 1)) (vec-of (IndexValue m)))
        (let revised (FixtureOperationShape (OperationId 1)))
        (run-schedule (saturate graph-indices dataflow-dependencies))
        (check (ShapeUse revised m) (DirectOperationUse (OperationId 1) (ValueNameRef m)))
        (fail (check (DirectOperationUse (OperationId 1) (ValueNameRef n))))
        "#,
    )?;
    Ok(())
}

#[test]
fn logical_values_and_storage_reads_have_distinct_effects() {
    let mut graph = fusion::new_graph().unwrap();
    super::super::test_graph::facts_with_positions(&mut graph,r#"
        (let value (ValueId 0)) (let storage (ValueId 1))
        (set (ValueType value) (LogicalValue (ArrayType (ValueArray) (vec-of (FixedExtent 4)) (ElementValue (ScalarElement (I32))))))
        (set (ValueType storage) (LogicalValue (ArrayType (StorageArray) (vec-of (FixedExtent 4)) (ElementValue (ScalarElement (I32))))))
        (set (FixtureParameter (RegionId 0) 0) (ValueNameRef value))
        (set (FixtureParameter (RegionId 0) 1) (ValueNameRef storage))
        (set (FixtureInput (OperationId 0) 0) (ValueNameRef value)) (ReadsInput (OperationId 0) 0)
        (set (FixtureInput (OperationId 1) 0) (ValueNameRef storage)) (ReadsInput (OperationId 1) 0)
        (OrderedOperation (OperationId 2)) (OrderedOperation (OperationId 3))
        (set (FixtureInput (OperationId 4) 0) (ValueNameRef storage)) (WritesInput (OperationId 4) 0)
        (EffectCompare (OperationId 2) (OperationId 1))
        (EffectCompare (OperationId 2) (OperationId 4))
        (EffectCompare (OperationId 2) (OperationId 3))
        (run-schedule (saturate dataflow-resources) (saturate dataflow-total) (saturate dataflow-effects))
        (fail (check (OperationRead (OperationId 0) root)))
        (check (OperationRead (OperationId 1) (ExternalMemory)))
        (fail (check (OperationsConflict (OperationId 2) (OperationId 1))))
        (check (OperationsConflict (OperationId 2) (OperationId 4)))
        (check (OperationsConflict (OperationId 2) (OperationId 3)))
    "#).unwrap();
}

#[test]
fn reducer_width_requires_agreement_and_a_length_preserving_combiner() -> Result<(), OptimizeError> {
    for (item_width, preserves, expected) in [(4, true, true), (3, true, false), (4, false, false)] {
        let result_bound = if preserves { "(IndexValue length)" } else { "(IndexConstant 3)" };
        let mut graph = fusion::new_graph()?;
        super::super::test_graph::facts_with_positions(
            &mut graph,
            &format!(
                r#"
            (let fold (OperationId 0)) (let reducer (RegionId 0))
            (let neutral (ValueId 0)) (let item (ValueId 1))
            (let left (ValueId 2)) (let right (ValueId 3))
            (let result (ValueId 4)) (let length (ValueId 5)) (let measure (OperationId 1))
            (set (ResultKindOf fold 0) (Folded (Total)))
            (set (FixtureResultNeutral fold 0 0) (ValueNameRef neutral))
            (set (FixtureResultValue fold 0 0) (ValueNameRef item))
            (set (FixtureResultReducer fold 0) reducer)

            (set (FixtureParameter reducer 0) (ValueNameRef left))
            (set (FixtureParameter reducer 1) (ValueNameRef right))
            (set (FixtureYield reducer 0) (ValueNameRef result))
            (LengthOperation measure 0)
            (set (FixtureInput measure 0) (ValueNameRef left))
            (set (FixtureOutput measure 0) (ValueNameRef length))
            (ArrayOuterBound neutral (IndexConstant 4))
            (ArrayOuterBound item (IndexConstant {item_width}))
            (ArrayOuterBound result {result_bound})
            (run-schedule (saturate dataflow-dependencies))
        "#
            ),
        )?;
        let check = "(check (ArrayOuterBound right (IndexConstant 4)))";
        let check = if expected { check.to_string() } else { format!("(fail {check})") };
        graph.parse_and_run_program(None, &check)?;
    }
    Ok(())
}

#[test]
fn counted_shift_requires_a_proven_in_range_amount() -> Result<(), OptimizeError> {
    for (limit, total) in [("16", true), ("17", false), ("n", false)] {
        let source = format!(
            "entry main(x:i32,n:i32,b:bool) i32 =
             let count = if b then {limit} else 0 in
             loop acc=0 for i<count do acc + (x >> (i*2+1))"
        );
        let mut graph = test_graph::import(&source)?;
        run(&mut graph)?;
        let query = Query(&graph);
        let mut shifts = 0;
        query.for_each("LiveOperation", |row| {
            let operation = row[0];
            if let Some(payload) = query.lookup("ScalarPayload", operation)? {
                if query.contains("ShiftPayload", payload)? {
                    shifts += 1;
                    assert_eq!(!query.contains("OperationTrap", operation)?, total);
                }
            }
            Ok(())
        })?;
        assert_eq!(shifts, 1);
    }
    Ok(())
}

#[test]
fn bounded_work_counts_every_loop_iteration() {
    let mut graph = fusion::new_graph().unwrap();
    graph
        .update(|mut sink| {
            Ok((|| -> Result<(), OptimizeError> {
                let scalar = sink.add("I32", RawValues(Vec::new()))?;
                let element = sink.add("ScalarElement", scalar)?;
                let info = sink.add("ScalarInfo", element)?;
                let diet = sink.add("Observe", RawValues(Vec::new()))?;
                let value = sink.add("ValueId", 0_i64)?;
                sink.set("ValueType", value, info)?;
                let index = sink.add("ValueId", 1_i64)?;
                let literal = operation(
                    &mut sink,
                    0,
                    Code::Constant {
                        ty: types::i32(),
                        bits: 1,
                    },
                    &[],
                )?;
                outputs(&mut sink, literal, &[Name::Value(value)])?;
                let body = region(&mut sink, 0, &[literal])?;
                parameters(
                    &mut sink,
                    body,
                    &[Parameter {
                        name: Name::Value(index),
                        info,
                        diet,
                    }],
                )?;
                yields(&mut sink, body, &[])?;
                let loop_ = operation(&mut sink, 1, Code::CountedLoop, &[])?;
                outputs(&mut sink, loop_, &[])?;
                sink.set_control(loop_.0, 0_i64, body.0);
                let count = sink.add("IndexConstant", 40_i64)?;
                sink.set_formula(loop_.0, crate::egglog::graph::Formula::LoopCount, count)?;
                let root = region(&mut sink, 1, &[loop_])?;
                parameters(&mut sink, root, &[])?;
                yields(&mut sink, root, &[])?;
                let function = sink.add("FunctionId", (0_i64, "work"))?;
                sink.add("Function", function)?;
                sink.set_function_root(function, root.0);
                sink.set("FunctionResultCount", function, 0_i64)?;
                sink.set("EntryOrder", function, 0_i64)?;
                Ok(())
            })())
        })
        .unwrap()
        .unwrap();
    run(&mut graph).unwrap();
    planning::analyze(&mut graph).unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
            (fail (check (= wanted (OperationWork operation))))
            (OperationWorkDemand (OperationId 1))
            (run-schedule (saturate graph-work))
            (check (= (OperationWork (OperationId 1)) (bigint 40)))
            (check (= (RegionWork (RegionId 0)) (bigint 1)))
            (fail (check (= unused (RegionWork (RegionId 1)))))
        "#,
        )
        .unwrap();
}
