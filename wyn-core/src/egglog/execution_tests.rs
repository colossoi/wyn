use super::Execution;
use crate::egglog::query::Query;

#[test]
fn emitted_dispatches_reference_the_authoritative_statements() {
    let mut graph = egglog_engine::EGraph::default();
    graph
        .parse_and_run_program(
            None,
            concat!(
                include_str!("memory/schema.egg"),
                "\n(relation CurrentProgram (Program))"
            ),
        )
        .unwrap();
    graph
        .parse_and_run_program(
            None,
            r#"
        (let name (FunctionId 0 "main"))
        (let statement (Let (StatementTag 0) (vec-of) (vec-of)
            (Parallel (DeviceKernel (AutomaticGrid))
                (IterationSpace (ValueId 0) (vec-of (DenseAxis (ValueId 1) (IndexConstant 64))))
                (ParallelBlock (vec-of) (vec-of)))))
        (let function (DefineFunction name (vec-of) (vec-of)
            (Block (vec-of statement) (vec-of))))
        (CurrentProgram (MemoryProgram (vec-of function) (vec-of) (vec-of (EntryPoint name 0))))
    "#,
        )
        .unwrap();
    let query = Query(&graph);
    let execution = Execution::select(&graph).unwrap();
    assert_eq!(execution.entries.len(), 1);
    let stages = execution.entries.values().next().unwrap();
    assert_eq!(stages.len(), 1);
    let statement = query.enode("Let", stages[0]).unwrap().unwrap();
    let parallel = query.fields("Parallel", statement[3]).unwrap();
    let kernel = query.fields("DeviceKernel", parallel[0]).unwrap();
    assert!(query.enode("AutomaticGrid", kernel[0]).unwrap().is_some());
    let current = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    let current = query.fields("MemoryProgram", current).unwrap();
    let function = query.vector(current[0]).unwrap()[0];
    let function = query.fields("DefineFunction", function).unwrap();
    let block = query.fields("Block", function[3]).unwrap();
    assert_eq!(query.vector(block[0]).unwrap(), *stages);
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
    for (grid, expected) in [
        ("(AutomaticGrid)", "automatic dispatch requires an iteration axis"),
        ("(Grid3D (vec-of (IndexConstant 1) (IndexConstant 1) (IndexConstant 1)) (vec-of (IndexConstant 0) (IndexConstant 1) (IndexConstant 1)))", "invalid workgroup size"),
        ("(Grid3D (vec-of (IndexConstant 1) (IndexConstant 1) (IndexConstant 1)) (vec-of (IndexConstant 32)))", "three local dimensions"),
    ] {
        let mut graph = egglog_engine::EGraph::default();
        graph.parse_and_run_program(None, concat!(include_str!("memory/schema.egg"), "\n(relation CurrentProgram (Program))")).unwrap();
        graph.parse_and_run_program(None, &format!(r#"
          (let name (FunctionId 0 "main"))
          (let function (DefineFunction name (vec-of) (vec-of) (Block
            (vec-of (Let (StatementTag 0) (vec-of) (vec-of)
              (Parallel (DeviceKernel {grid}) (IterationSpace (ValueId 0) (vec-of))
                (ParallelBlock (vec-of) (vec-of))))) (vec-of))))
          (CurrentProgram (MemoryProgram (vec-of function) (vec-of) (vec-of (EntryPoint name 0))))
        "#)).unwrap();
        let Err(error) = Execution::select(&graph) else { panic!("accepted {grid}"); };
        assert!(error.to_string().contains(expected), "{error}");
    }
}
