//! Plan execution using the shared graph and the existing portable cost policies.
use super::cost::{self, CostModel};
use super::query::Query;
use super::{analysis, output_error, OptimizeError};
use crate::PipelineTopologyPolicy;
use egglog_engine::{EGraph, RawValues, Write};

mod conditional;
mod distribution;

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(
        Some("graph planning".into()),
        concat!(
            include_str!("planning/graph.egg"),
            "\n",
            include_str!("planning/distribution.egg"),
            "\n",
            include_str!("planning/conditional.egg"),
            "\n",
            include_str!("planning/shapes.egg"),
            "\n",
            include_str!("planning/reuse.egg"),
            "\n",
            include_str!("planning/epilogue.egg"),
            "\n",
            include_str!("planning/compaction.egg"),
            "\n",
            include_str!("planning/legality.egg"),
            "\n",
            include_str!("analysis/regularity.egg")
        ),
    )?;
    Ok(())
}

/// Recompute metadata after graph analysis has established live regions and edges.
pub(super) fn analyze(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = super::timing::span("egglog graph metadata");
    for table in [
        "StaticIndex",
        "CollectionRank",
        "CollectionExtent",
        "CollectionElement",
        "CompactPath",
        "CompactPair",
        "CompactSafe",
        "CompactMaskPrefix",
        "CompactCandidate",
        "CompactObserved",
        "CompactCountCandidate",
        "CompactLengthUse",
        "IndexUpper",
        "CountLength",
        "LengthWitness",
        "ProductField",
        "ProductComponent",
        "ProjectionPath",
        "LoopValueCarry",
        "LoopArrayComponent",
        "LoopPreservesComponentLength",
        "ArrayElementRange",
        "LiteralRangePrefix",
        "IntegerLower",
        "IntegerUpper",
        "ConsumedScalarArray",
        "ReuseCandidate",
        "SnapshotCopy",
        "SnapshotUpdate",
        "SnapshotUseSafe",
        "SnapshotCopySafe",
        "CanSnapshotCopy",
        "ReuseClauses",
        "ReuseReadSafe",
        "ReuseLegal",
        "ReuseSnapshot",
        "EpiloguePair",
        "EpiloguePath",
        "EpilogueReuse",
        "PointwisePair",
        "PointwiseSource",
        "EpilogueDomainPrefix",
        "EpilogueClauses",
        "EpilogueHasScan",
        "PointwiseLoadPrefix",
        "PointwiseLoad",
        "PointwiseSafeOperation",
        "PointwiseRegionPrefix",
        "PointwiseSafeRegion",
        "PointwiseHandledUse",
        "PointwiseLegal",
        "LogicalOuterCount",
        "OuterCapacity",
        "OuterShapeSource",
        "LoopPreservesOuterLength",
        "ReducerPreservesOuterLength",
        "FreshArray",
        "LocalLoop",
        "LoopStateBytes",
        "OperationWork",
        "RegionWorkPrefix",
        "RegionWork",
        "DomainWork",
        "LaneDependentFunction",
        "HostRegion",
        "KernelPreparation",
        "SelectedExecution",
        "FirstOperation",
        "LastOperation",
        "EmptyRegion",
        "RegionOperationCount",
        "DistributionPair",
        "DistributionReturnPrefix",
        "DistributionPrefixAt",
        "DistributionPrefixComplete",
        "DistributionPrefixOperation",
        "TransformRequirement",
        "TransformAvailable",
        "DistributionLaneDependent",
        "DistributionSafe",
        "DistributionHasFold",
        "KnownIndex",
        "EstimatedDomain",
        "DistributeNest",
        "ConditionalPair",
        "ConditionalDomainPrefix",
        "ConditionalReturnPrefix",
        "ConditionalTypePrefix",
        "ConditionalSafe",
        "FactorConditional",
        "ScanEpilogue",
        "BinaryReducer",
        "NativeBinaryReducer",
        "NativeAddition",
        "RequiresSnapshot",
        "CanReuse",
        "ForwardLength",
        "LogicalBytes",
        "ExtentProduct",
        "ProductBytes",
        "ElementBytes",
        "ElementProductBytes",
        "IndexArithmetic",
        "IndexArithmeticOperands",
        "SelectOutputPrefix",
        "CanSpeculate",
        "RetrySafe",
        "AtomicAccumulatorArray",
        "ReducerState",
        "LoopComponentPreserved",
        "LoopHasArrayComponents",
    ] {
        graph.clear_function(table)?;
    }
    analysis::advance(graph)?;
    let mut boundaries = Vec::new();
    Query(graph).for_each("LiveRegion", |row| {
        let operations = Query(graph).operations(row[0])?;
        boundaries.push((
            row[0],
            operations.len() as i64,
            operations.first().copied(),
            operations.last().copied(),
        ));
        Ok(())
    })?;
    graph.update(|mut sink| {
        for (region, count, first, last) in boundaries {
            sink.set("RegionOperationCount", region, count)?;
            if let (Some(first), Some(last)) = (first, last) {
                sink.add("FirstOperation", (region, first))?;
                sink.add("LastOperation", (region, last))?;
            } else {
                sink.add("EmptyRegion", region)?;
            }
        }
        Ok(())
    })?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate scalar-analysis graph-sizes graph-work))",
    )?;
    Ok(())
}

pub(super) fn place(
    graph: &mut EGraph,
    topology: PipelineTopologyPolicy,
    costs: CostModel,
) -> Result<(), OptimizeError> {
    cost::register(graph, costs, false);
    cost::register_width(graph, costs);
    graph.parse_and_run_program(
        Some("execution cost choices".into()),
        include_str!("planning/choices.egg"),
    )?;
    graph.update(|mut sink| {
        sink.set(
            "GeneratedDispatches",
            RawValues(Vec::new()),
            topology == PipelineTopologyPolicy::AllowGenerated,
        )
    })?;
    loop {
        analysis::run(graph)?;
        analyze(graph)?;
        if distribution::canonicalize(graph)? {
            continue;
        }
        graph.parse_and_run_program(
            None,
            "(run-schedule
            (saturate distribution-index)
            (saturate transform-availability dataflow-scopes)
            (saturate dataflow-effects)
            (saturate distribution-prefix)
            (saturate distribution-legality)
            (saturate distribution-select))",
        )?;
        let Some(candidate) = Query(graph).first("DistributeNest")? else {
            break;
        };
        distribution::apply(graph, candidate[0], candidate[1])?;
    }
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate graph-placement) (saturate graph-loop-select) (saturate graph-host))",
    )?;
    Ok(())
}

pub(super) fn schedule(graph: &mut EGraph) -> Result<(), OptimizeError> {
    // Host-region reachability is complete before false facts select local execution.
    graph.parse_and_run_program(None, "(run-schedule (saturate graph-placement-select))")?;
    let query = Query(graph);
    let mut choices = Vec::new();
    query.for_each("LiveOperation", |row| {
        if query.contains("ParallelOperation", row[0])? {
            let execution = query.required("SelectedExecution", row[0])?;
            choices.push((row[0], execution));
        }
        Ok(())
    })?;
    graph.update(|mut sink| {
        for (operation, execution) in choices {
            sink.remove("OperationExecution", operation)?;
            sink.set("OperationExecution", operation, execution)?;
        }
        Ok(())
    })?;
    Ok(())
}

pub(super) fn load_execution(graph: &mut EGraph) -> Result<(), OptimizeError> {
    // Physical policies are loaded after the final structural analysis.
    graph.parse_and_run_program(
        Some("physical execution policies".into()),
        include_str!("planning/execution.egg"),
    )?;
    Ok(())
}

pub(super) fn epilogues(graph: &mut EGraph) -> Result<(), OptimizeError> {
    analyze(graph)?;
    load_execution(graph)?;
    graph.parse_and_run_program(None, "(run-schedule (saturate physical-legality) (saturate execution-features) (saturate execution-atomics))")?;
    graph.parse_and_run_program(None, "(run-schedule (saturate reuse-reads epilogue-index) (saturate reuse-legality epilogue-legality) (saturate reuse-select epilogue-select))")?;
    let query = Query(graph);
    query.for_each("LiveOperation", |row| {
        if query.contains("ParallelOperation", row[0])? {
            let selected = query.required("OperationExecution", row[0])?;
            if query.enode("Unscheduled", selected)?.is_some() {
                return Err(output_error("parallel operation has no selected execution"));
            }
        }
        Ok(())
    })
}

#[cfg(test)]
#[path = "planning_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "planning/order_tests.rs"]
mod order_tests;

pub(super) fn factor_conditionals(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    graph.parse_and_run_program(
        None,
        "(run-schedule
        (saturate conditional-index)
        (saturate transform-availability dataflow-scopes)
        (saturate conditional-legality)
        (saturate conditional-select))",
    )?;
    let Some(candidate) = Query(graph).first("FactorConditional")? else {
        return Ok(false);
    };
    conditional::apply(graph, candidate[0], candidate[1], candidate[2])?;
    Ok(true)
}
