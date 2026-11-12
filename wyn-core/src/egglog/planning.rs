//! Logical planning policies. Concrete execution is emitted directly into SSA.
use super::{
    cost::{self, CostModel},
    memory, output_error,
    query::Query,
    OptimizeError,
};
use crate::PipelineTopologyPolicy;
use egglog_engine::{EGraph, RawValues, Write};

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(
        Some("logical execution choices".into()),
        include_str!("planning/logical.egg"),
    )?;
    Ok(())
}

pub(super) fn place(
    graph: &mut EGraph,
    topology: PipelineTopologyPolicy,
    costs: CostModel,
) -> Result<(), OptimizeError> {
    require_logical(graph, false)?;
    cost::register(graph, costs, false);
    cost::register_width(graph, costs);
    graph.parse_and_run_program(
        None,
        "(rule
        ((ContainsExp (FunctionRoot function) expression)
         (= expression (Loop merges form returns body))
         (= bytes (LoopStateBytes function merges (vec-length merges)))
         (= work (BodyWork function body)) (= conditional (WhileLoop form))
         (= (wyn-local-loop work bytes conditional) true))
        ((set (LocalLoop function expression) true)) :ruleset schedule-cost)",
    )?;
    graph.parse_and_run_program(
        None,
        "(rule
        ((NeutralPreparation function expression) (= work (BoundedWork function expression))
         (= (wyn-local-loop work (bigint 0) false) true))
        ((set (KernelPreparation function expression) true)) :ruleset schedule-cost)",
    )?;
    graph.parse_and_run_program(None, "(rule
        ((DistributionCandidate program function outer inner prefix available)
         (= (DistributionSafe program outer) true)
         (= outer (Let id pattern certs (Parallel execution (IterationSpace flat axes) outer-body)))
         (= inner (Let iid ipattern proofs (Parallel nested (IterationSpace iflat inner-axes) (ParallelBlock body results))))
         (= returned (ForwardedResults (ParallelClauses outer-body)))
         (ReturnedPattern ipattern returned (vec-length ipattern))
         (= coordinates (EstimatedDomain function axes (vec-length axes)))
         (= elements (EstimatedDomain function inner-axes (vec-length inner-axes)))
         (= fold (HasFold results))
         (= (wyn-distribute coordinates (bigint (vec-length prefix)) (bigint (vec-length body)) elements fold) true))
        ((DistributeNest program function outer inner prefix)) :ruleset distribution-select)")?;
    graph.update(|mut sink| {
        sink.set(
            "GeneratedDispatches",
            RawValues(Vec::new()),
            topology == PipelineTopologyPolicy::AllowGenerated,
        )
    })?;
    memory::validate(graph)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule
        (saturate schedule-cost)
        (saturate placement-context)
        (saturate (seq
            (saturate fusion-index memory-names)
            (saturate memory-scopes)
            (saturate schedule-cost memory-origin simplify-analysis)
            (saturate memory-effects)
            (saturate placement-context distribution-analysis)
            (saturate distribution-select)
            (saturate memory-rewrite-distribution memory-rename))))",
    )?;
    memory::validate(graph)?;
    require_logical(graph, false)
}

pub(super) fn schedule(graph: &mut EGraph, costs: CostModel) -> Result<(), OptimizeError> {
    cost::register_width(graph, costs);
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate (seq
        (saturate fusion-index memory-names)
        (saturate memory-scopes)
        (saturate schedule-cost memory-origin memory-effects simplify-analysis)
        (saturate placement-context)
        (saturate placement-select) (saturate memory-rewrite-placement memory-rename))))",
    )?;
    memory::validate(graph)?;
    require_logical(graph, true)
}

/// Strategy selection may annotate the logical program but cannot materialize storage.
fn require_logical(graph: &EGraph, planned: bool) -> Result<(), OptimizeError> {
    let query = Query(graph);
    query.for_each("CurrentProgram", |row| {
        let Some(root) = query.constructor("ProgramRoot", (row[0],))? else {
            return Err(output_error("logical program has not been indexed"));
        };
        if planned {
            query.for_each("ContainsExecution", |execution| {
                if execution[0] == root && query.enode("Unscheduled", execution[1])?.is_some() {
                    return Err(output_error(
                        "logical planning left an execution strategy unselected",
                    ));
                }
                Ok(())
            })?;
        }
        Ok(())
    })
}

#[cfg(test)]
#[path = "planning_tests.rs"]
mod tests;
