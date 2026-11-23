use super::BodyGraph;
use crate::egglog::dataflow::ir::Code;
use crate::egglog::{analysis, query::Query, test_graph, OptimizeError};
use crate::memory_ir::{Definition, Function, ValueRef};

#[test]
fn nested_candidate_keeps_captures_without_importing_its_callee() -> Result<(), OptimizeError> {
    let mut graph = test_graph::import(
        "def helper(x:i32) i32 = x*x+1
         entry main(x:i32, choose:bool) i32 = if choose then helper(x) else 0",
    )?;
    analysis::run(&mut graph)?;
    let query = Query(&graph);
    let root = test_graph::root(&graph);
    let branch = graph.region_order(root)?[0];
    assert_eq!(graph.code(branch)?, Code::Branch);
    let candidate = graph.control_region(branch, 0)?;
    let call = graph.region_order(candidate)?[0];
    let helper_body = graph.function_root(graph.call_target(call)?)?;
    assert!(query.contains("LiveRegion", helper_body)?);

    let body = BodyGraph::read(query, [candidate])?;
    let Function::Defined(outer) = body.ir.functions[body.ir.entries[0]] else {
        panic!("the enclosing entry body must be imported");
    };
    assert_eq!(body.regions[outer], root);
    let call = &body.ir.operations[body.operation_ids[&call]];
    let Definition::Call(helper) = call.definition else {
        panic!("expected the candidate's call to retain a target");
    };
    assert_eq!(call.inputs, [ValueRef::Parameter(outer, 0)]);
    assert_eq!(body.ir.functions[helper], Function::Unloaded);
    assert!(!body.regions.values().any(|&region| region == helper_body));
    Ok(())
}
