use super::Execution;
use crate::egglog::dataflow::ir::{self, Code};
use crate::egglog::memory;
use crate::egglog::query::Query;
use egglog_engine::EGraph;

fn graph(grid: &str, rank: i64) -> EGraph {
    let mut graph = EGraph::default();
    memory::load(&mut graph).unwrap();
    graph.update(|mut sink| Ok(ir::operation(&mut sink, 0, Code::Parallel, &[]))).unwrap().unwrap();
    graph
        .parse_and_run_program(
            None,
            &format!(
                r#"
        (let name (FunctionId 0 "main"))
        (set (FunctionRoot name) (RegionId 0)) (set (EntryOrder name) 0)
        (set (OperationRegion (OperationId 0)) (RegionId 0)) (set (OperationOrder (OperationId 0)) 0)
        (set (OperationExecution (OperationId 0)) (DeviceKernel {grid}))
        (set (OperationWidth (OperationId 0)) 128)
        (set (OperationDomain (OperationId 0)) (DomainId 0)) (set (DomainRank (DomainId 0)) {rank})
    "#
            ),
        )
        .unwrap();
    graph
}

#[test]
fn emitted_dispatches_reference_the_authoritative_statements() {
    let mut graph = graph("(AutomaticGrid)", 1);
    let execution = Execution::select(&mut graph).unwrap();
    let query = Query(&graph);
    let stages = execution.dispatches.keys().copied().collect::<Vec<_>>();
    assert_eq!(stages.len(), 1);
    let selected = query.required("OperationExecution", stages[0]).unwrap();
    let kernel = query.fields("DeviceKernel", selected).unwrap();
    assert!(query.enode("AutomaticGrid", kernel[0]).unwrap().is_some());
    let function = query.first("EntryOrder").unwrap().unwrap()[0];
    let root = query.required("FunctionRoot", function).unwrap();
    assert_eq!(query.operations(root).unwrap(), stages);
    let dispatch = execution.dispatch(stages[0]).unwrap();
    assert_eq!(dispatch.local_size, (128, 1, 1));
    assert!(
        dispatch.groups.is_none(),
        "derive group counts during SSA emission"
    );
    assert_eq!(
        dispatch.width.get(),
        dispatch.local_size.0 * dispatch.local_size.1 * dispatch.local_size.2
    );
}

#[test]
fn malformed_grids_are_rejected_during_interface_readout() {
    for (grid,expected) in [
        ("(AutomaticGrid)","automatic dispatch requires an iteration axis"),
        ("(Grid3D (vec-of (IndexConstant 1) (IndexConstant 1) (IndexConstant 1)) (vec-of (IndexConstant 0) (IndexConstant 1) (IndexConstant 1)))","invalid workgroup size"),
        ("(Grid3D (vec-of (IndexConstant 1) (IndexConstant 1) (IndexConstant 1)) (vec-of (IndexConstant 32)))","three dimensions"),
    ] {
        let mut graph=graph(grid,0);
        let Err(error)=Execution::select(&mut graph) else {panic!("accepted {grid}");};
        assert!(error.to_string().contains(expected),"{error}");
    }
}
