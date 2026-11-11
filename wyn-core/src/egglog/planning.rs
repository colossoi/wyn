//! Placement and scheduling rewrite the shared statement tree in egglog.
use super::{
    cost::{self, CostModel},
    fusion, memory, output_error,
    query::Query,
    OptimizeError,
};
use crate::PipelineTopologyPolicy;
use egglog_engine::{EGraph, RawValues, Write};

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(Some("schedule cost analysis".into()), include_str!("cost.egg"))?;
    graph.parse_and_run_program(
        Some("statement scheduling".into()),
        include_str!("planning/schedule.egg"),
    )?;
    graph.parse_and_run_program(
        Some("collective decomposition".into()),
        include_str!("planning/collectives.egg"),
    )?;
    graph.parse_and_run_program(
        Some("local storage placement".into()),
        include_str!("planning/sequential.egg"),
    )?;
    let mut rules = String::new();
    for (constructor, operator, guard) in [
        ("IndexAdd", "+", ""),
        ("IndexSub", "-", ""),
        ("IndexMul", "*", ""),
        ("IndexDiv", "/", "(!= b 0)"),
        ("IndexMod", "%", "(!= b 0)"),
        ("IndexMin", "min", ""),
        ("IndexMax", "max", ""),
    ] {
        rules.push_str(&format!("(rule ((ContainsIndexExp (FunctionRoot function) index) (= index ({constructor} left right)) (= a (StaticIndex function left)) (= b (StaticIndex function right)) {guard}) ((set (StaticIndex function index) ({operator} a b))) :ruleset schedule-cost)\n"));
    }
    for expression in [
        "(Use operand)",
        "(IndexOp index)",
        "(PrimOp name arguments)",
        "(Load name indices)",
        "(View name slices)",
        "(Reshape name shape)",
        "(Permute name permutation)",
        "(Accumulate mode token indices values)",
        "(Reserve token indices value)",
        "(Store mode array slices value)",
    ] {
        rules.push_str(&format!("(rule ((ContainsExp (FunctionRoot function) expression) (= expression {expression})) ((set (BoundedWork function expression) (bigint 1))) :ruleset schedule-cost)\n"));
    }
    graph.parse_and_run_program(Some("schedule arithmetic".into()), &rules)?;
    Ok(())
}

pub(super) fn place(
    graph: &mut EGraph,
    topology: PipelineTopologyPolicy,
    costs: CostModel,
) -> Result<(), OptimizeError> {
    memory::validate(graph)?;
    cost::register(graph, costs, false);
    graph.parse_and_run_program(None, "(rule ((ContainsExp (FunctionRoot function) expression) (= expression (Loop merges form returns body))
        (= bytes (LoopStateBytes function merges (vec-length merges))) (= work (BodyWork function body))
        (= conditional (WhileLoop form))
        (= (wyn-local-loop work (bigint bytes) conditional) true))
        ((set (LocalLoop function expression) true)) :ruleset schedule-cost)")?;
    graph.update(|mut sink| {
        sink.set(
            "GeneratedDispatches",
            RawValues(Vec::new()),
            topology == PipelineTopologyPolicy::AllowGenerated,
        )
    })?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate (seq
            (saturate fusion-index memory-names)
            (saturate memory-scopes)
            (saturate memory-allocation schedule-cost memory-origin)
            (saturate memory-effects)
            (saturate accumulator-atomic schedule-cost)
            (saturate placement-context)
            (saturate placement-scopes)
            (saturate (seq (saturate dispatch-shapes)
                (saturate memory-names) (saturate local-storage-analysis)))
            (saturate schedule-cost dispatch-shapes allocation-bounds)
            (saturate allocation-sizes) memory-allocation-current
            (saturate fusion-forward memory-reuse)
            (saturate memory-reuse-propose) (saturate memory-reuse-select)
            (saturate memory-allocation memory-origin)
            (saturate loop-buffer-provenance)
            (saturate loop-buffer-analysis)
            (saturate scalar-analysis collective-memory-analysis)
            (saturate collective-memory-build)
            (saturate local-storage-status)
            (saturate local-storage-rewrite)
            (saturate clone-index memory-names)
            (saturate clone-prepare inline-prepare inline-select memory-rename)
            (saturate local-storage-rewrite)
            (saturate memory-rewrite memory-rename))))
         (run-schedule (saturate (seq
            (saturate fusion-index memory-names)
            (saturate memory-scopes)
            (saturate memory-allocation schedule-cost memory-origin) (saturate memory-effects)
            (saturate accumulator-atomic schedule-cost)
            (saturate placement-context)
            (saturate placement-scopes)
            (saturate (seq (saturate dispatch-shapes)
                (saturate memory-names) (saturate local-storage-analysis)))
            (saturate local-storage-status)
            (saturate placement-select)
            (saturate memory-rewrite memory-rename))))",
    )?;
    require_scheduled(graph, false)
}

/// Choose legal domain contractions before lowering them to physical kernels.
pub(super) fn distribute(graph: &mut EGraph, costs: CostModel) -> Result<(), OptimizeError> {
    cost::register_width(graph, costs);
    cost::register(graph, costs, true);
    graph.parse_and_run_program(None, "(rule ((DistributionCandidate program function outer inner prefix available)
        (= (DistributionSafe program outer) true)
        (= outer (Let id pattern certs (Parallel execution (IterationSpace flat axes) (ParallelBlock statements clauses))))
        (= inner (Let iid ipattern proofs (Parallel nested (IterationSpace iflat iaxes) (ParallelBlock body results))))
        (= returned (ForwardedResults clauses)) (ReturnedPattern ipattern returned (vec-length ipattern))
        (= coordinates (EstimatedDomain function axes (vec-length axes))) (= elements (EstimatedDomain function iaxes (vec-length iaxes)))
        (= serial (SequenceWork function statements (- (vec-length statements) 1)))
        (= parallel (SequenceWork function body (vec-length body)))
        (= fold (HasFold results))
        (= (wyn-distribute coordinates serial parallel elements fold) true))
        ((DistributeNest program function outer inner prefix)) :ruleset distribution-select)")?;
    graph.parse_and_run_program(None, "(rule ((ReindexCandidate program function producer consumer read slot root count)
        (= producer (Let pid outputs certs (Parallel execution (IterationSpace flat axes) (ParallelBlock code clauses))))
        (= read (Let rid bindings proofs (Load array indices)))
        (ReindexCoordinates producer read (vec-length indices) parts) (CoordinateCoverage parts (set-empty) count)
        (= (Commutes (EffectsStm function producer) (EffectsStm function consumer)) true)
        (= coordinates (DomainWork function axes (vec-length axes))) (> coordinates (bigint 0))
        (= serial (SequenceWork function code (vec-length code)))
        (= (wyn-distribute coordinates serial (bigint 0) (/ (bigint count) coordinates) false) true))
        ((ReindexProof program function producer consumer read slot)) :ruleset coordinate-select)")?;
    graph.parse_and_run_program(
        None,
        &format!("(run-schedule (saturate (seq
        {fusion}
        (saturate fusion-index memory-names)
        (saturate memory-scopes)
        (saturate memory-allocation schedule-cost memory-origin)
        (saturate memory-effects simplify-analysis scalar-analysis schedule-cost)
        (saturate coordinate-analysis memory-effects scalar-analysis)
        (saturate dispatch-shapes local-storage-analysis distribution-analysis)
        (saturate coordinate-select memory-names)
        (saturate clone-index memory-names)
        (saturate clone-prepare inline-prepare inline-select memory-rename)
        (saturate coordinate-select memory-rename memory-names)
        (saturate scalar-copies simplify-copies simplify-removals scalar-candidates)
        (saturate simplify-select scalar-select memory-edits)
        (saturate memory-rewrite memory-rename)
        (saturate fusion-index memory-names memory-scopes)
        (saturate memory-allocation schedule-cost memory-origin) (saturate memory-effects)
        (saturate fusion-forward dispatch-shapes local-storage-analysis distribution-analysis fusion-domains memory-rename memory-effects)
        (saturate conditional-select distribution-select)
        (saturate memory-rewrite memory-rename))))", fusion = fusion::SCHEDULE),
    )?;
    Ok(())
}

pub(super) fn schedule(graph: &mut EGraph, costs: CostModel) -> Result<(), OptimizeError> {
    distribute(graph, costs)?;
    graph.parse_and_run_program(None, "(rule ((ContainsStm (FunctionRoot function) statement)
        (= statement (Let id pattern certs (Parallel (DeviceKernel (AutomaticGrid)) domain
            (ParallelBlock statements results)))) (WalkParallelResults results i) (< i (vec-length results))
        (= clause (vec-get results i)) (= clause (Fold kind (Reduction order neutral combine) values))
        (= bytes (FoldStateBytes function clause (vec-length neutral))) (= scan (ScanWork kind))
        (= elements (CollectiveElements function statement)) (= width (SelectedWidth function domain results)))
        ((set (CollectiveGroups function statement)
            (wyn-collective-groups elements (bigint bytes) (bigint 0)
                (bigint bytes) (bigint (+ 1 (vec-length statements)))
                (bigint (+ 1 (vec-length neutral))) scan width))) :ruleset collective-discover)")?;
    graph.parse_and_run_program(None, "(rule ((CurrentProgram program) (ProgramFunction program function)
        (ContainsExp (FunctionRoot function) (Parallel execution domain (ParallelBlock statements results)))
        (= domain (IterationSpace flat axes)) (= elements (DomainWork function axes (vec-length axes)))
        (= work (SequenceWork function statements (vec-length statements))) (= (HasFold results) false))
        ((set (SelectedWidth function domain results) (wyn-workgroup-width elements work))) :ruleset dispatch-shapes)")?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate (seq
            (saturate fusion-index memory-names)
            (saturate memory-scopes)
            (saturate schedule-cost dispatch-shapes memory-allocation)
            (saturate loop-buffer-analysis scalar-analysis collective-memory-analysis)
            (saturate local-storage-analysis distribution-analysis collective-memory-build kernel-memory-build)
            (saturate collective-discover)
            (saturate clone-index memory-names)
            (saturate clone-prepare inline-prepare inline-select memory-rename)
            (saturate collective-build kernel-memory-build)
            (saturate memory-rewrite memory-rename))))
         (run-schedule (saturate (seq
            (saturate fusion-index memory-names)
            (saturate memory-scopes schedule-cost)
            (saturate dispatch-shapes)
            (saturate dispatch-select)
            (saturate memory-rewrite memory-rename))))",
    )?;
    graph.parse_and_run_program(None,
        "(run-schedule (saturate (seq
            (saturate fusion-index memory-names)
            (saturate memory-scopes schedule-cost memory-origin)
            (saturate memory-effects)
            (saturate placement-context) (saturate placement-scopes)
            (saturate (seq (saturate dispatch-shapes) (saturate memory-names) (saturate local-storage-analysis)))
            (saturate schedule-cost dispatch-shapes allocation-bounds)
            (saturate allocation-sizes) memory-allocation-current
            (saturate fusion-forward memory-reuse)
            (saturate memory-reuse-propose) (saturate memory-reuse-select)
            (saturate memory-allocation)
            (saturate memory-expand-select memory-rename)
            (saturate memory-rewrite memory-rename))))")?;
    graph.parse_and_run_program(None,
        "(run-schedule (saturate (seq
            (saturate fusion-index memory-names)
            (saturate memory-scopes schedule-cost memory-origin)
            (saturate memory-effects)
            (saturate placement-context) (saturate placement-scopes)
            (saturate (seq (saturate dispatch-shapes) (saturate memory-names) (saturate local-storage-analysis)))
            (saturate schedule-cost dispatch-shapes allocation-bounds)
            (saturate allocation-sizes) memory-allocation-current
            (saturate fusion-forward memory-reuse)
            (saturate memory-reuse-propose) (saturate memory-reuse-select)
            (saturate memory-allocation)
            (saturate loop-buffer-analysis scalar-analysis collective-memory-analysis)
            (saturate collective-memory-build kernel-memory-native)
            (saturate kernel-memory-select)
            (saturate (seq
                (saturate memory-names memory-allocation local-storage-analysis)
                (saturate kernel-memory-build collective-build local-storage-rewrite)
                (saturate clone-index memory-names)
                (saturate clone-prepare inline-prepare inline-select memory-rename)))
            (saturate memory-rewrite memory-rename))))")?;
    graph.parse_and_run_program(None,
        "(run-schedule (saturate (seq
            (saturate fusion-index memory-names)
            (saturate memory-scopes)
            (saturate placement-context)
            (saturate schedule-cost) (saturate placement-scopes)
            (saturate (seq (saturate dispatch-shapes) (saturate memory-names) (saturate local-storage-analysis)))
            (saturate schedule-cost dispatch-shapes allocation-bounds)
            (saturate allocation-sizes) memory-allocation-current
            (saturate fusion-forward memory-reuse)
            (saturate memory-reuse-propose) (saturate memory-reuse-select)
            (saturate memory-allocation memory-origin)
            (saturate loop-buffer-provenance)
            (saturate loop-buffer-analysis)
            (saturate loop-buffer-rewrite)
            (saturate memory-rewrite memory-rename))))")?;
    require_scheduled(graph, true)?;
    memory::allocate(graph)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate (seq
            (saturate fusion-index memory-names)
            (saturate memory-scopes memory-origin)
            (saturate placement-context) (saturate placement-scopes)
            (saturate memory-effects simplify-analysis)
            (saturate kernel-prelude-analysis memory-effects memory-rename)
            (saturate kernel-prelude-select)
            (saturate memory-rewrite memory-rename))))",
    )?;
    memory::validate(graph)
}

fn require_scheduled(graph: &mut EGraph, grid_required: bool) -> Result<(), OptimizeError> {
    memory::validate(graph)?;
    let query = Query(graph);
    query.for_each("CurrentProgram", |row| {
        let Some(root) = query.constructor("ProgramRoot", (row[0],))? else {
            return Err(output_error("scheduled program was not indexed"));
        };
        query.for_each("ContainsExecution", |execution| {
            if execution[0] != root {
                return Ok(());
            }
            if query.enode("Unscheduled", execution[1])?.is_some() {
                return Err(output_error("placement left an unscheduled parallel body"));
            }
            if grid_required {
                if let Some(fields) = query.enode("DeviceKernel", execution[1])? {
                    if query.enode("AutomaticGrid", fields[0])?.is_some() {
                        return Err(output_error("scheduling left an unresolved dispatch grid"));
                    }
                }
            }
            Ok(())
        })?;
        if !grid_required {
            return Ok(());
        }
        query.for_each("ProgramFunction", |function| {
            if function[0] != row[0] {
                return Ok(());
            }
            let Some(root) = query.constructor("FunctionRoot", (function[1],))? else {
                return Err(output_error("scheduled function was not indexed"));
            };
            query.for_each("ContainsExp", |expression| {
                if expression[0] != root {
                    return Ok(());
                }
                let Some(parallel) = query.enode("Parallel", expression[1])? else {
                    return Ok(());
                };
                if query.enode("DeviceKernel", parallel[0])?.is_none() {
                    return Ok(());
                }
                let body = query.fields("ParallelBlock", parallel[2])?;
                let has_fold = query.required("HasFold", (body[1],))?;
                if !graph.value_to_base::<bool>(has_fold) {
                    return Ok(());
                }
                let native = query.required("NativeKernel", (function[1], body[1]))?;
                if !graph.value_to_base::<bool>(native) {
                    return Err(output_error(
                        "scheduling left a portable fold without memory lowering",
                    ));
                }
                Ok(())
            })
        })
    })
}

#[cfg(test)]
#[path = "planning/order_tests.rs"]
mod order_tests;
#[cfg(test)]
#[path = "planning_tests.rs"]
mod tests;
