//! Plan execution using the shared graph and the existing portable cost policies.
use super::cost::{self, CostModel};
use super::dataflow::edit;
use super::query::Query;
use super::{analysis, memory, output_error, OptimizeError};
use crate::egglog::graph::Graph;
use crate::{LookupMap, LookupSet, PipelineTopologyPolicy};
use egglog_engine::sort::{Boxed, Z};
use egglog_engine::{RawValues, Value, Write};

mod conditional;
mod distribution;
mod layout;

pub(super) fn load(graph: &mut Graph) -> Result<(), OptimizeError> {
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
            include_str!("planning/layout.egg"),
            "\n",
            include_str!("planning/reuse.egg"),
            "\n",
            include_str!("planning/legality.egg"),
            "\n",
            include_str!("planning/epilogue.egg"),
            "\n",
            include_str!("analysis/regularity.egg")
        ),
    )?;
    Ok(())
}

/// Retain graph metadata across candidate resets, while solving newly added demands.
pub(super) fn analyze(graph: &mut Graph) -> Result<(), OptimizeError> {
    analysis::ensure(graph)?;
    let _timing = super::timing::span("egglog graph metadata");
    let revision = edit::revision(graph)?;
    let current = Query(graph).lookup("PlanningRevision", RawValues(Vec::new()))?
        == Some(graph.base_to_value(revision));
    if !current {
        let _invalidation = super::timing::span("egglog metadata invalidation");
        for table in [
            "StaticIndex",
            "CollectionRank",
            "CollectionExtent",
            "CollectionElement",
            "LoopCapturedRow",
            "LoopInitialRow",
            "RankedLoopCollection",
            "LoopReadsRowWidth",
            "IndexUpper",
            "CountLength",
            "LengthWitness",
            "ProductField",
            "ProductComponent",
            "ComponentLengthWitness",
            "ProjectionPath",
            "LoopValueCarry",
            "LoopArrayComponent",
            "LoopPreservesComponentLength",
            "ArrayElementRange",
            "LiteralRangePrefix",
            "LogicalOuterCount",
            "OuterCapacity",
            "OuterShapeSource",
            "LoopPreservesOuterLength",
            "ReducerPreservesOuterLength",
            "FreshArray",
            "OperationWorkDemand",
            "RegionWorkDemand",
            "RegionWorkUser",
            "OperationWork",
            "RegionWorkPrefix",
            "RegionWork",
            "DomainWork",
            "KnownIndex",
            "BinaryReducer",
            "NativeBinaryReducer",
            "ForwardLength",
            "IndexArithmetic",
            "IndexArithmeticOperands",
        ] {
            graph.clear_function(table)?;
        }
        let key = || RawValues(Vec::new());
        let revision = analysis::count(Query(graph), "ShapeRevision", key())?;
        graph.update(|mut sink| {
            sink.remove("ShapeRevision", key())?;
            sink.set("ShapeRevision", key(), revision + 1)
        })?;
        clear_candidates(graph)?;
    }
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate scalar-analysis graph-sizes integer-ranges type-sizes))",
    )?;
    if !current {
        graph.update(|mut sink| {
            sink.remove("PlanningRevision", RawValues(Vec::new()))?;
            sink.set("PlanningRevision", RawValues(Vec::new()), revision)
        })?;
    }
    Ok(())
}

/// Selection depends on phase policy as well as the executable graph.
fn clear_candidates(graph: &mut Graph) -> Result<(), OptimizeError> {
    for table in [
        "ConsumedScalarArray",
        "ReuseCandidate",
        "ReuseDomainInput",
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
        "EpilogueConsumer",
        "EpilogueScratch",
        "EpilogueOverwrite",
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
        "EpilogueRequiresCompletedTotal",
        "PointwiseLegal",
        "LocalLoop",
        "LaneDependentFunction",
        "HostRegion",
        "KernelPreparationCandidate",
        "KernelPreparation",
        "SelectedExecution",
        "ScanEpilogue",
        "NativeAddition",
        "RequiresSnapshot",
        "CanReuse",
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
    clear_transform_candidates(graph)?;
    analysis::advance(graph)
}

pub(super) fn clear_transform_candidates(graph: &mut Graph) -> Result<(), OptimizeError> {
    for table in [
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
        "EstimatedDomain",
        "DistributeNest",
        "ReadyDistribution",
        "RectangularRows",
        "RowWidthLoop",
        "RowWidthLength",
        "RowWidthPreparation",
        "RowWidthInput",
        "RowWidthInputReady",
        "RowWidthLegal",
        "LiftUniformRowLength",
        "ConditionalPair",
        "ConditionalPrefix",
        "ConditionalDomainPrefix",
        "ConditionalReturnPrefix",
        "ConditionalTypePrefix",
        "ConditionalSafe",
        "FactorConditional",
    ] {
        graph.clear_function(table)?;
    }
    let key = || RawValues(Vec::new());
    let revision = analysis::count(Query(graph), "TransformRevision", key())?;
    graph.update(|mut sink| {
        sink.remove("TransformRevision", key())?;
        sink.set("TransformRevision", key(), revision + 1)
    })?;
    Ok(())
}

fn resolve_transform_availability(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut transforms = Vec::new();
    query.for_each("TransformRequirement", |row| {
        transforms.push((row[0], row[1], analysis::available_before(query, row[0], row[1])?));
        Ok(())
    })?;
    let mut rows = Vec::new();
    query.for_each("RowWidthInput", |row| {
        if analysis::available_before(query, row[0], row[2])? {
            rows.push((row[0], row[1], row[2]));
        }
        Ok(())
    })?;
    graph.update(|mut sink| {
        for (operation, name, available) in transforms {
            sink.set("TransformAvailable", (operation, name), available)?;
        }
        for key in rows {
            sink.set("RowWidthInputReady", key, true)?;
        }
        Ok(())
    })?;
    Ok(())
}

pub(super) fn repair_sibling_order(
    graph: &mut Graph,
    updated: &[(Value, Vec<Value>)],
) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut removed = Vec::new();
    let mut prefixes = Vec::new();
    for (region, body) in updated {
        if !query.contains("RegionWorkDemand", *region)? {
            continue;
        }
        let mut prefix: Option<Z> = Some(Boxed(0.into()));
        for &operation in body {
            let key = (*region, operation);
            removed.push(key);
            if let Some(current) = prefix.take() {
                let next = query.lookup("OperationWork", operation)?;
                prefix = next.map(|work| Boxed(&*current + &*graph.value_to_base::<Z>(work)));
                prefixes.push((key, current));
            }
        }
    }
    graph.update(|mut sink| {
        for key in removed {
            sink.remove("RegionWorkPrefix", key)?;
        }
        for (key, work) in prefixes {
            sink.set("RegionWorkPrefix", key, work)?;
        }
        Ok(())
    })?;
    reset_order_candidates(graph)
}

fn reset_order_candidates(graph: &mut Graph) -> Result<(), OptimizeError> {
    // A reorder can change adjacency and speculative scope for pending transforms.
    if graph.get_size("DistributionPair") != 0 || graph.get_size("ConditionalPair") != 0 {
        clear_transform_candidates(graph)?;
    }
    Ok(())
}

pub(super) struct ChangedRegionWork {
    regions: LookupSet<Value>,
    operations: LookupSet<Value>,
}

/// Retract affected work prefixes before the old occurrence order is replaced.
pub(super) fn invalidate_region_work(
    graph: &mut Graph,
    previous: &[(Value, Vec<Value>)],
) -> Result<ChangedRegionWork, OptimizeError> {
    let query = Query(graph);
    let mut work = ChangedRegionWork {
        regions: LookupSet::new(),
        operations: LookupSet::new(),
    };
    let mut pending = Vec::new();
    for &(region, _) in previous {
        if query.contains("RegionWorkDemand", region)? {
            pending.push(region);
        }
    }
    if pending.is_empty() {
        return Ok(work);
    }
    let mut consumers = LookupMap::<Value, Vec<Value>>::new();
    query.for_each("RegionWorkUser", |row| {
        consumers.entry(row[0]).or_default().push(row[1]);
        Ok(())
    })?;
    let mut prefixes = Vec::new();
    while let Some(region) = pending.pop() {
        if !work.regions.insert(region) {
            continue;
        }
        prefixes.extend(query.0.region_order(region)?.iter().map(|&op| (region, op)));
        for &op in consumers.get(&region).into_iter().flatten() {
            if work.operations.insert(op) {
                let owner = query.0.operation_region(op)?;
                if query.contains("RegionWorkDemand", owner)? {
                    pending.push(owner);
                }
            }
        }
    }
    graph.update(|mut sink| {
        for key in prefixes {
            sink.remove("RegionWorkPrefix", key)?;
        }
        for &region in &work.regions {
            sink.remove("RegionWork", region)?;
            sink.remove("RegionWorkDemand", region)?;
        }
        for &op in &work.operations {
            sink.remove("OperationWork", op)?;
            sink.remove("OperationWorkDemand", op)?;
        }
        Ok(())
    })?;
    Ok(work)
}

pub(super) fn restore_region_work(graph: &mut Graph, work: ChangedRegionWork) -> Result<(), OptimizeError> {
    if !work.regions.is_empty() {
        graph.update(|mut sink| {
            for region in work.regions {
                sink.add("RegionWorkDemand", region)?;
            }
            for op in work.operations {
                sink.add("OperationWorkDemand", op)?;
            }
            Ok(())
        })?;
        graph.parse_and_run_program(None, "(run-schedule (saturate graph-work))")?;
    }
    reset_order_candidates(graph)
}

pub(super) fn place(
    graph: &mut Graph,
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
    clear_candidates(graph)?;
    loop {
        analysis::ensure(graph)?;
        analyze(graph)?;
        if distribution::canonicalize(graph)? {
            continue;
        }
        graph.parse_and_run_program(
            None,
            "(run-schedule (saturate distribution-index) (saturate transform-availability))",
        )?;
        resolve_transform_availability(graph)?;
        graph.parse_and_run_program(
            None,
            "(run-schedule
            (saturate dataflow-effects)
            (saturate distribution-prefix)
            (run row-width-legality)
            (run row-width-select)
            (saturate distribution-legality)
            (saturate distribution-select))",
        )?;
        if distribution::lift_row_length(graph)? {
            memory::simplify(graph, true)?;
            continue;
        }
        let Some(candidate) = Query(graph).first("DistributeNest")? else {
            if layout::fuse(graph)? {
                continue;
            }
            let revision = edit::revision(graph)?;
            memory::simplify(graph, true)?;
            if edit::revision(graph)? == revision {
                break;
            }
            continue;
        };
        distribution::apply(graph, candidate[0], candidate[1])?;
    }
    loop_layout(graph)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate entry-invocations graph-placement) (saturate graph-loop-select graph-work) (saturate graph-host))",
    )?;
    Ok(())
}

fn loop_layout(graph: &mut Graph) -> Result<(), OptimizeError> {
    for table in [
        "LoopStateBytes",
        "LoopFieldType",
        "LoopFieldBytes",
        "LoopProductBytes",
    ] {
        graph.clear_function(table)?;
    }
    graph.parse_and_run_program(None, "(run-schedule (saturate graph-loop-layout))")?;
    Ok(())
}

pub(super) fn schedule(graph: &mut Graph) -> Result<(), OptimizeError> {
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

pub(super) fn load_execution(graph: &mut Graph) -> Result<(), OptimizeError> {
    // Physical policies are loaded after the final structural analysis.
    graph.parse_and_run_program(
        Some("physical execution policies".into()),
        concat!(
            include_str!("planning/execution.egg"),
            "\n",
            include_str!("planning/column-scan.egg")
        ),
    )?;
    Ok(())
}

pub(super) fn epilogues(graph: &mut Graph) -> Result<(), OptimizeError> {
    analyze(graph)?;
    clear_candidates(graph)?;
    loop_layout(graph)?;
    load_execution(graph)?;
    graph.parse_and_run_program(None, "(run-schedule (saturate entry-invocations physical-legality graph-work) (saturate execution-features) (saturate execution-atomics))")?;
    graph.parse_and_run_program(None, "(run-schedule (saturate reuse-reads epilogue-index) (saturate dataflow-effects) (saturate reuse-legality epilogue-legality) (saturate reuse-select epilogue-select))")?;
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

pub(super) fn factor_conditionals(graph: &mut Graph) -> Result<bool, OptimizeError> {
    graph.parse_and_run_program(None, "(run-schedule (saturate conditional-index))")?;
    resolve_transform_availability(graph)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule
        (saturate dataflow-effects)
        (saturate conditional-legality)
        (saturate conditional-select))",
    )?;
    let Some(candidate) = Query(graph).first("FactorConditional")? else {
        return Ok(false);
    };
    conditional::apply(graph, candidate[0], candidate[1], candidate[2])?;
    Ok(true)
}
