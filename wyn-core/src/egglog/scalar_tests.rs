use super::super::{from_tlc, memory, query::Query, OptimizeError, ScalarOptimization};
use crate::egglog::dataflow::{
    edit,
    ir::{self, Code, Name},
};
use crate::egglog::graph::Graph;
use crate::egglog::payload::Payload;
use crate::op::BinaryOperator;
use crate::types;
use egglog_engine::Write;

fn optimized(source: &str) -> Result<Graph, OptimizeError> {
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
    let mut additions = 0;
    query.for_each("LiveOperation", |row| {
        additions +=
            usize::from(query.0.code(row[0])? == Code::Scalar(Payload::Binary(BinaryOperator::Add)));
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
    let mut additions = Vec::new();
    let query = Query(&graph);
    query.for_each("LiveOperation", |row| {
        if query.0.code(row[0])? == Code::Scalar(Payload::Binary(BinaryOperator::Add)) {
            additions.push(row[0]);
        }
        Ok(())
    })?;
    for operation in additions {
        edit::set_code(
            &mut graph,
            operation,
            Code::Constant {
                ty: types::i32(),
                bits: 0,
            },
        )?;
        edit::inputs(&mut graph, operation, &[])?;
    }
    super::run(&mut graph, ScalarOptimization::Full)?;
    let query = Query(&graph);
    let mut divisions = 0;
    let mut calls = 0;
    query.for_each("LiveOperation", |row| {
        let code = query.0.code(row[0])?;
        let division = code == Code::Scalar(Payload::Binary(BinaryOperator::Divide));
        let call = matches!(code, Code::Call);
        if division || call {
            let region = query.required("OperationRegion", row[0])?;
            assert!(
                query.row("RegionOwnedBy", |row| row[0] == region)?.is_some(),
                "partial evaluation escaped its branch"
            );
        }
        divisions += usize::from(division);
        calls += usize::from(call);
        Ok(())
    })?;
    assert_eq!((divisions, calls), (1, 1));
    Ok(())
}

#[test]
fn shared_helpers_keep_separate_entry_and_loop_parameters() -> Result<(), OptimizeError> {
    let graph = optimized(
        r#"
        def walk(x:f32,n:i32) f32=loop sum=0.0f32 for k<n do sum+x*x+f32(k)
        entry first(xs:[]f32,n:i32) []f32=map(|x|walk(x,n)+x*2.0f32,xs)
        entry second(xs:[]f32,n:i32) []f32=map(|x|walk(x+1.0f32,n)+x*3.0f32,xs)
    "#,
    )?;
    let query = Query(&graph);
    let mut functions = 0;
    let mut entries = 0;
    let mut calls = 0;
    query.for_each("LiveFunction", |_| {
        functions += 1;
        Ok(())
    })?;
    query.for_function("EntryOrder", |_, _| {
        entries += 1;
        Ok(())
    })?;
    query.for_each("LiveOperation", |row| {
        calls += usize::from(query.contains("CallOperation", row[0])?);
        Ok(())
    })?;
    assert_eq!((functions, entries, calls), (2, 2, 0));
    Ok(())
}

#[test]
fn unrelated_syntax_does_not_add_evaluation_dependencies() -> Result<(), OptimizeError> {
    let mut graph = optimized("entry main(x:i32) i32=x")?;
    let entry = Query(&graph).first("EntryOrder")?.unwrap()[0];
    let root = Query(&graph).required("FunctionRoot", entry)?;
    let before = Query(&graph).sequence("RegionYield", root)?;
    graph.update(|mut sink| {
        Ok((|| -> Result<(), OptimizeError> {
            let value = sink.add("ValueId", 999_i64)?;
            ir::operation(
                &mut sink,
                999,
                Code::Scalar(Payload::Binary(BinaryOperator::Divide)),
                &[Name::Value(value), Name::Value(value)],
            )?;
            Ok(())
        })())
    })??;
    super::run(&mut graph, ScalarOptimization::Full)?;
    assert_eq!(Query(&graph).sequence("RegionYield", root)?, before);
    assert!(Query(&graph).operations(root)?.is_empty());
    Ok(())
}

#[test]
fn function_pruning_keeps_called_helpers_and_removes_substituted_callbacks() -> Result<(), OptimizeError> {
    let tlc = crate::compile_thru_tlc("def helper(x:i32) i32=x+1 entry main(x:i32) i32=helper(x)").unwrap();
    let (mut graph, _) = from_tlc::import(&tlc)?;
    // Keep an uncalled function in the graph, independent of TLC's own pruning.
    let helper = Query(&graph).first("Function")?.unwrap()[0];
    let root = Query(&graph).required("FunctionRoot", helper)?;
    graph.update(|mut sink| {
        let dead = sink.add("FunctionId", (999_i64, "substituted_callback"))?;
        sink.add("Function", dead)?;
        sink.set("FunctionRoot", dead, root)?;
        sink.set("FunctionResultCount", dead, 1_i64)?;
        Ok(())
    })?;
    for (policy, expected) in [(ScalarOptimization::Basic, 2), (ScalarOptimization::Full, 1)] {
        super::run(&mut graph, policy)?;
        let query = Query(&graph);
        let mut functions = 0;
        let mut calls = 0;
        query.for_each("LiveFunction", |_| {
            functions += 1;
            Ok(())
        })?;
        query.for_each("LiveOperation", |row| {
            calls += usize::from(query.contains("CallOperation", row[0])?);
            Ok(())
        })?;
        assert_eq!(functions, expected);
        assert_eq!(calls, usize::from(policy == ScalarOptimization::Basic));
    }
    Ok(())
}
