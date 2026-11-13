use super::super::{from_tlc, memory, query::Query, OptimizeError, ScalarOptimization};
use egglog_engine::EGraph;

fn optimized(source: &str) -> Result<EGraph, OptimizeError> {
    let source = crate::compile_thru_tlc(source).unwrap();
    let (mut graph, _) = from_tlc::import(&source)?;
    super::run(&mut graph, ScalarOptimization::Full)?;
    memory::validate(&mut graph)?;
    Ok(graph)
}

#[test]
fn dominating_total_expressions_share_one_evaluation() -> Result<(), OptimizeError> {
    let graph = optimized("entry main(x:i32) i32=(x+1)*(x+1)")?;
    let query = Query(&graph);
    let program = query.row("CurrentProgram", |_| true)?.unwrap()[0];
    let root = query.constructor("ProgramRoot", (program,))?.unwrap();
    let mut additions = 0;
    query.for_each("ContainsStm", |row| {
        if row[0] != root {
            return Ok(());
        }
        let statement = query.fields("Let", row[1])?;
        if let Some(operation) = query.enode("PrimOp", statement[3])? {
            if graph.value_to_base::<egglog_engine::sort::S>(operation[0]).as_str() == "+" {
                additions += 1;
            }
        }
        Ok(())
    })?;
    assert_eq!(additions, 1);
    Ok(())
}

#[test]
fn partial_and_effectful_operands_keep_their_evaluation_guards() -> Result<(), OptimizeError> {
    let mut graph = optimized(
        r#"
        #[linked("effect_impl")] extern effect(x:i32) i32
        entry divide(b:bool,x:i32,y:i32) i32=if b then x/y else 0
        entry invoke(b:bool,x:i32) i32=if b then effect(x)+1 else 0
    "#,
    )?;
    // Discard the result in this IR so TLC cannot erase the test scenario first.
    graph.parse_and_run_program(
        None,
        r#"
        (check (CurrentProgram p) (ContainsExp (ProgramRoot p) (Call name args returns)))
        (ruleset discard-result)
        (rule ((CurrentProgram p) (ContainsStm (ProgramRoot p) statement)
            (= statement (Let id pattern certs (PrimOp "+" args))))
          ((ScalarReplacement p statement (LiteralBits (I32) 0))) :ruleset discard-result)
        (run-schedule discard-result)
    "#,
    )?;
    super::run(&mut graph, ScalarOptimization::Full)?;
    graph.parse_and_run_program(
        None,
        r#"
        (check (CurrentProgram p) (ContainsExp (ProgramRoot p) (PrimOp "/" args)))
        (check (CurrentProgram p) (ContainsExp (ProgramRoot p) (Call name args returns)))
        (fail (check (CurrentProgram p) (ProgramFunction p f)
            (= f (DefineFunction name params returns (Block statements results)))
            (WalkStms statements i) (< i (vec-size statements))
            (= (vec-at statements i) (Let id pattern certs (PrimOp "/" args)))))
        (fail (check (CurrentProgram p) (ProgramFunction p f)
            (= f (DefineFunction name params returns (Block statements results)))
            (WalkStms statements i) (< i (vec-size statements))
            (= (vec-at statements i) (Let id pattern certs (Call target args signature)))))
    "#,
    )?;
    Ok(())
}

#[test]
fn shared_helpers_keep_separate_entry_and_loop_parameters() -> Result<(), OptimizeError> {
    let mut graph = optimized(
        r#"
        def walk(x:f32,n:i32) f32=loop sum=0.0f32 for k<n do sum+x*x+f32(k)
        entry first(xs:[]f32,n:i32) []f32=map(|x|walk(x,n)+x*2.0f32,xs)
        entry second(xs:[]f32,n:i32) []f32=map(|x|walk(x+1.0f32,n)+x*3.0f32,xs)
    "#,
    )?;
    graph.parse_and_run_program(
        None,
        r#"
        (check (CurrentProgram (MemoryProgram functions externals entries)) (= (vec-size entries) 2) (= (vec-size functions) 2))
        (fail (check (CurrentProgram p) (ContainsExp (ProgramRoot p) (Call name args returns))))
    "#,
    )?;
    Ok(())
}

#[test]
fn unrelated_syntax_does_not_add_evaluation_dependencies() -> Result<(), OptimizeError> {
    let mut graph = optimized("entry main(x:i32) i32=x")?;
    let original = Query(&graph).row("CurrentProgram", |_| true)?.unwrap()[0];
    graph.parse_and_run_program(
        None,
        r#"
        (Let (StatementTag 999) (vec-of (BindValue (ValueId 999) (ScalarInfo (ScalarElement (I32)))))
            (vec-of) (PrimOp "/" (vec-of (Var (ValueId 999)) (LiteralBits (I32) 0))))
    "#,
    )?;
    super::run(&mut graph, ScalarOptimization::Full)?;
    assert_eq!(
        Query(&graph).row("CurrentProgram", |_| true)?.unwrap()[0],
        original
    );
    Ok(())
}

#[test]
fn function_pruning_keeps_called_helpers_and_removes_substituted_callbacks() -> Result<(), OptimizeError> {
    let mut graph = EGraph::default();
    memory::load(&mut graph)?;
    graph.parse_and_run_program(
        None,
        r#"
        (let i32 (ScalarInfo (ScalarElement (I32))))
        (let returns (vec-of (ReturnScalar (ScalarElement (I32)))))
        (let helper (FunctionId 0 "helper"))
        (let entry (FunctionId 1 "main"))
        (let dead (FunctionId 2 "substituted_callback"))
        (let helper-body (Block (vec-of)
            (vec-of (Yield (vec-of) (ValueOperand (Var (ValueId 0)))))))
        (let params (vec-of (Parameter (BindValue (ValueId 0) i32) (Observe))))
        (let main-params (vec-of (Parameter (BindValue (ValueId 1) i32) (Observe))))
        (let main-body (Block (vec-of
            (Let (StatementTag 0) (vec-of (BindValue (ValueId 2) i32)) (vec-of)
                (Call helper (vec-of (Pass (ValueOperand (Var (ValueId 1))) (Observe))) returns)))
            (vec-of (Yield (vec-of) (ValueOperand (Var (ValueId 2)))))))
        (CurrentProgram (MemoryProgram (vec-of
            (DefineFunction helper params returns helper-body)
            (DefineFunction entry main-params returns main-body)
            (DefineFunction dead (vec-of) returns
                (Block (vec-of) (vec-of (Yield (vec-of) (ValueOperand (LiteralBits (I32) 1)))))))
            (vec-of) (vec-of (EntryPoint entry 0))))
    "#,
    )?;
    super::run(&mut graph, ScalarOptimization::Basic)?;
    graph.parse_and_run_program(
        None,
        r#"
        (check (CurrentProgram (MemoryProgram functions externals entries)) (= (vec-size functions) 2))
        (check (CurrentProgram p) (ContainsExp (ProgramRoot p) (Call name arguments returns)))
    "#,
    )?;
    super::run(&mut graph, ScalarOptimization::Full)?;
    graph.parse_and_run_program(
        None,
        r#"
        (check (CurrentProgram (MemoryProgram functions externals entries)) (= (vec-size functions) 1))
        (fail (check (CurrentProgram p) (ContainsExp (ProgramRoot p) (Call name arguments returns))))
    "#,
    )?;
    Ok(())
}
