use super::Execution;
use crate::egglog::query::Query;
use crate::egglog::{from_tlc, fuse, place, schedule};
use crate::{compile_thru_tlc, PipelineTopologyPolicy};

#[test]
fn emitted_dispatches_reference_the_authoritative_statements() {
    let source = compile_thru_tlc("entry main(xs:[64]i32) [64]i32=map(|x|x+1,xs)").unwrap();
    let program = schedule(
        place(
            fuse(from_tlc(&source).unwrap()).unwrap(),
            PipelineTopologyPolicy::AllowGenerated,
        )
        .unwrap(),
    )
    .unwrap();
    let query = Query(&program.graph);
    let execution = Execution::select(&program.graph).unwrap();
    assert_eq!(execution.entries.len(), 1);
    let stages = execution.entries.values().next().unwrap();
    assert_eq!(stages.len(), 1);
    let statement = query.enode("Let", stages[0]).unwrap().unwrap();
    let launch = query.fields("Launch", statement[3]).unwrap();
    assert!(query.enode("Grid3D", launch[0]).unwrap().is_some());
    let current = query.row("CurrentProgram", |_| true).unwrap().unwrap()[0];
    let root = query.constructor("ProgramRoot", (current,)).unwrap().unwrap();
    assert!(query.contains("ContainsStm", (root, stages[0])).unwrap());
    let grid = query.fields("Grid3D", launch[0]).unwrap();
    let local = query.vector(grid[1]).unwrap().into_iter().map(|dimension| {
        let constant = query.fields("IndexConstant", dimension).unwrap();
        program.graph.value_to_base::<i64>(constant[0]) as u32
    }).collect::<Vec<_>>();
    assert_eq!(local, [64, 1, 1], "64 elements use one full workgroup");
    assert_eq!(execution.dispatch(stages[0]).unwrap().width.get(), local.iter().product::<u32>());
}

#[test]
fn malformed_grids_are_rejected_during_interface_readout() {
    for (grid, expected) in [
        ("(AutomaticGrid)", "unresolved dispatch grid"),
        ("(Grid3D (vec-of (IndexConstant 1) (IndexConstant 1) (IndexConstant 1)) (vec-of (IndexConstant 0) (IndexConstant 1) (IndexConstant 1)))", "invalid workgroup size"),
        ("(Grid3D (vec-of (IndexConstant 1) (IndexConstant 1) (IndexConstant 1)) (vec-of (IndexConstant 32)))", "three local dimensions"),
    ] {
        let mut graph = egglog_engine::EGraph::default();
        crate::egglog::memory::load(&mut graph).unwrap();
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
