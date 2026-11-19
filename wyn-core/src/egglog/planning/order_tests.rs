use super::{place, schedule, CostModel};
use crate::egglog::dataflow::edit;
use crate::egglog::dataflow::ir::{self, Code};
use crate::egglog::query::Query;
use crate::egglog::{analysis, test_graph};
use crate::PipelineTopologyPolicy;

#[test]
fn placement_preserves_scopes_and_allocation_follows_scheduling() {
    let mut graph =
        test_graph::import("entry main(xs:[4]i32) [4]i32=map(|x|reduce((+),0i32,map(|y|x+y,xs)),xs)")
            .unwrap();
    let root = test_graph::root(&graph);
    place(
        &mut graph,
        PipelineTopologyPolicy::AllowGenerated,
        CostModel::default(),
    )
    .unwrap();
    assert_eq!(test_graph::root(&graph), root);
    schedule(&mut graph).unwrap();
    let query = Query(&graph);
    let mut device = false;
    let mut local = false;
    query
        .for_each("LiveOperation", |row| {
            if query.contains("ParallelOperation", row[0])? {
                let execution = query.required("OperationExecution", row[0])?;
                device |= query.enode("DeviceKernel", execution)?.is_some();
                local |= query.enode("InvocationLocal", execution)?.is_some();
            }
            Ok(())
        })
        .unwrap();
    assert!(device && local);
    assert!(!graph.get_function_names().iter().any(|name| name == "Alloc"));
}

#[test]
fn authored_only_keeps_parallel_work_inside_its_invocation() {
    let mut graph = test_graph::import("entry main(xs:[4]i32) [4]i32=map(|x|x+1,xs)").unwrap();
    place(
        &mut graph,
        PipelineTopologyPolicy::AuthoredOnly,
        CostModel::default(),
    )
    .unwrap();
    schedule(&mut graph).unwrap();
    let query = Query(&graph);
    let mut count = 0;
    query
        .for_each("LiveOperation", |row| {
            if query.contains("ParallelOperation", row[0])? {
                assert!(query
                    .enode("InvocationLocal", query.required("OperationExecution", row[0])?)?
                    .is_some());
                count += 1;
            }
            Ok(())
        })
        .unwrap();
    assert!(count > 0);
}

#[test]
fn forward_dependencies_are_rejected_before_scheduling() {
    let mut graph = test_graph::import("entry main(x:i32) i32=let y=x+1 in y*y").unwrap();
    analysis::run(&mut graph).unwrap();
    let query = Query(&graph);
    let mut add = None;
    let mut multiply = None;
    query
        .for_each("LiveOperation", |row| {
            match ir::code(&graph, query.required("OperationPayload", row[0])?) {
                Code::Scalar(crate::egglog::payload::Payload::Binary(crate::op::BinaryOperator::Add)) => {
                    add = Some(row[0])
                }
                Code::Scalar(crate::egglog::payload::Payload::Binary(
                    crate::op::BinaryOperator::Multiply,
                )) => multiply = Some(row[0]),
                _ => {}
            }
            Ok(())
        })
        .unwrap();
    let (add, multiply) = (add.unwrap(), multiply.unwrap());
    edit::move_before(&mut graph, multiply, add).unwrap();
    assert!(place(
        &mut graph,
        PipelineTopologyPolicy::AllowGenerated,
        CostModel::default()
    )
    .is_err());
}
