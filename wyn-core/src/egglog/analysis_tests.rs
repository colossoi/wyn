use super::{liveness, run};
use crate::egglog::cost::CostModel;
use crate::egglog::dataflow::ir::{operation, outputs, parameters, region, yields, Code, Name, Parameter};
use crate::egglog::fusion;
use crate::egglog::query::Query;
use crate::egglog::{planning, test_graph, OptimizeError};
use crate::types;
use egglog_engine::{RawValues, Write};

#[test]
fn rewriting_defers_live_after_and_host_classification() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import("entry main(x:i32) i32=(x+1)*x")?;
    run(&mut graph)?;
    planning::analyze(&mut graph)?;
    for table in ["FlowLive", "NameLiveAfter", "LaneDependentFunction", "HostRegion"] {
        assert!(
            Query(&graph).first(table)?.is_none(),
            "{table} was calculated before its consumer"
        );
    }
    liveness(&mut graph)?;
    assert!(Query(&graph).first("NameLiveAfter")?.is_some());
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
fn logical_values_and_storage_reads_have_distinct_effects() {
    let mut graph = fusion::new_graph().unwrap();
    graph.parse_and_run_program(None,r#"
        (let value (ValueId 0)) (let storage (ValueId 1))
        (set (ValueType value) (LogicalValue (ArrayType (ValueArray) (vec-of (FixedExtent 4)) (ElementValue (ScalarElement (I32))))))
        (set (ValueType storage) (LogicalValue (ArrayType (StorageArray) (vec-of (FixedExtent 4)) (ElementValue (ScalarElement (I32))))))
        (set (RegionParameter (RegionId 0) 0) (ValueNameRef value))
        (set (RegionParameter (RegionId 0) 1) (ValueNameRef storage))
        (set (Input (OperationId 0) 0) (ValueNameRef value)) (ReadsInput (OperationId 0) 0)
        (set (Input (OperationId 1) 0) (ValueNameRef storage)) (ReadsInput (OperationId 1) 0)
        (OrderedOperation (OperationId 2)) (OrderedOperation (OperationId 3))
        (set (Input (OperationId 4) 0) (ValueNameRef storage)) (WritesInput (OperationId 4) 0)
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
                sink.set("ControlRegion", (loop_.0, 0_i64), body.0)?;
                let role = sink.add("LoopRegion", RawValues(Vec::new()))?;
                sink.set("ControlRole", (loop_.0, 0_i64), role)?;
                sink.set("LoopIndex", loop_.0, index)?;
                for (table, n) in [("LoopStart", 0_i64), ("LoopCount", 40_i64), ("LoopStep", 1_i64)] {
                    let value = sink.add("IndexConstant", n)?;
                    sink.set(table, loop_.0, value)?;
                }
                let root = region(&mut sink, 1, &[loop_])?;
                parameters(&mut sink, root, &[])?;
                yields(&mut sink, root, &[])?;
                let function = sink.add("FunctionId", (0_i64, "work"))?;
                sink.add("Function", function)?;
                sink.set("FunctionRoot", function, root.0)?;
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
        .parse_and_run_program(None, "(check (= (OperationWork (OperationId 1)) (bigint 40)))")
        .unwrap();
}
