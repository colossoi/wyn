use super::*;
use crate::compile_thru_tlc;

#[test]
fn placement_preserves_scopes_and_allocation_follows_scheduling() -> Result<(), OptimizeError> {
    let source =
        compile_thru_tlc("entry main(xs:[4]i32) [4]i32=map(|x|reduce((+),0i32,map(|y|x+y,xs)),xs)")
            .unwrap();
    let (mut graph, _) = crate::egglog::from_tlc::import(&source)?;
    crate::egglog::fusion::run(&mut graph)?;
    place(
        &mut graph,
        PipelineTopologyPolicy::AllowGenerated,
        CostModel::default(),
    )?;
    graph.parse_and_run_program(
        None,
        r#"
        (check (CurrentProgram program)
          (ContainsExecution (ProgramRoot program) (DeviceKernel (AutomaticGrid))))
        (check (CurrentProgram program)
          (ContainsExecution (ProgramRoot program) (InvocationLocal)))
    "#,
    )?;
    schedule(&mut graph, CostModel::default())?;
    graph.parse_and_run_program(
        None,
        r#"
        (check (CurrentProgram program) (ContainsExp (ProgramRoot program) (Alloc space size)))
    "#,
    )?;
    Ok(())
}

#[test]
fn authored_only_keeps_parallel_work_inside_its_invocation() -> Result<(), OptimizeError> {
    let source = compile_thru_tlc("entry main(xs:[4]i32) [4]i32=map(|x|x+1,xs)").unwrap();
    let (mut graph, _) = crate::egglog::from_tlc::import(&source)?;
    place(
        &mut graph,
        PipelineTopologyPolicy::AuthoredOnly,
        CostModel::default(),
    )?;
    schedule(&mut graph, CostModel::default())?;
    let query = Query(&graph);
    query.for_each("CurrentProgram", |program| {
        let root = query.constructor("ProgramRoot", (program[0],))?.unwrap();
        query.for_each("ContainsExecution", |execution| {
            if execution[0] == root {
                assert!(query.enode("InvocationLocal", execution[1])?.is_some());
            }
            Ok(())
        })
    })
}

#[test]
fn forward_dependencies_are_rejected_before_scheduling() -> Result<(), OptimizeError> {
    let mut graph = EGraph::default();
    memory::load(&mut graph)?;
    graph.parse_and_run_program(None, r#"
        (let scalar (ScalarInfo (ScalarElement (I32))))
        (let function (DefineFunction (FunctionId 0 "bad") (vec-of) (vec-of)
          (Block (vec-of
            (Let (StatementTag 0) (vec-of (BindValue (ValueId 0) scalar)) (vec-of) (Use (ValueOperand (Var (ValueId 1)))))
            (Let (StatementTag 1) (vec-of (BindValue (ValueId 1) scalar)) (vec-of) (Use (ValueOperand (LiteralBits (I32) 1)))))
            (vec-of))))
        (CurrentProgram (MemoryProgram (vec-of function) (vec-of) (vec-of)))
    "#)?;
    assert!(place(
        &mut graph,
        PipelineTopologyPolicy::AllowGenerated,
        CostModel::default()
    )
    .is_err());
    Ok(())
}
