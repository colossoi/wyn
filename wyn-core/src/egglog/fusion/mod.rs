//! Compose selected dataflow regions and preserve their occurrence order.
use super::dataflow::edit;
use super::query::Query;
use super::{analysis, memory, OptimizeError};
use crate::LookupSet;
use egglog_engine::sort::Z;
use egglog_engine::{EGraph, Value, Write};

mod compaction;

#[cfg(test)]
mod tests;

pub(super) fn new_graph() -> Result<EGraph, OptimizeError> {
    let mut graph = EGraph::default();
    memory::load(&mut graph)?;
    Ok(graph)
}

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(Some("graph fusion".into()), include_str!("graph.egg"))?;
    Ok(())
}

fn clear(graph: &mut EGraph) -> Result<(), OptimizeError> {
    for table in [
        "ParallelPair",
        "ParallelPath",
        "FusionInputAvailable",
        "FusionChildRegion",
        "SameDomainPrefix",
        "FusionLoadPrefix",
        "FusionForwardLoad",
        "FusionUnsafe",
        "FusionHandledUse",
        "FuseCandidate",
        "FuseSelected",
        "FusionClauseBase",
        "FusionOutputBase",
        "FusionOperationBase",
        "MotionCandidate",
        "FusionPreparation",
        "FusionBoundary",
        "FusionNested",
        "FusionCollective",
        "FusionHasForwardedLoad",
        "FusionPrivateWrites",
        "SinkProducer",
        "SinkPath",
        "AccumulatorPair",
        "AccumulatorFuseSelected",
        "FusionGroupBase",
    ] {
        graph.clear_function(table)?;
    }
    Ok(())
}

pub(super) fn run(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = super::timing::span("egglog graph fusion");
    memory::simplify(graph, false)?;
    loop {
        clear(graph)?;
        analysis::run(graph)?;
        crate::egglog::planning::analyze(graph)?;
        graph.parse_and_run_program(None, "(run-schedule (saturate graph-fusion dataflow-scopes) (saturate dataflow-effects) (saturate graph-fusion-legality) (saturate graph-fusion-select))")?;
        let query = Query(graph);
        let mut selected = None;
        query.for_each("FuseCandidate", |row| {
            if selected.is_none()
                && (!query.contains("OperationsConflict", (row[0], row[1]))?
                    || query
                        .lookup("FusionPrivateWrites", (row[0], row[1]))?
                        .is_some_and(|safe| query.0.value_to_base::<bool>(safe)))
            {
                selected = Some((row[0], row[1]));
            }
            Ok(())
        })?;
        if let Some((first, second)) = selected {
            fuse(graph, first, second)?;
        } else {
            query.for_each("AccumulatorPair", |row| {
                if selected.is_none()
                    && !query.contains("OperationsConflict", (row[0], row[1]))?
                    && !query.contains("OperationDependency", (row[1], row[0]))?
                {
                    selected = Some((row[0], row[1]));
                }
                Ok(())
            })?;
            let Some((first, second)) = selected else {
                if compaction::run(graph)? {
                    memory::simplify(graph, false)?;
                    continue;
                }
                if sink_producer(graph)? {
                    memory::simplify(graph, false)?;
                    continue;
                }
                if motion(graph, false)? {
                    memory::simplify(graph, false)?;
                    continue;
                }
                return Ok(());
            };
            fuse_accumulators(graph, first, second)?;
        }
        memory::simplify(graph, false)?;
    }
}

fn sink_producer(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let mut selected = None;
    query.for_each("SinkProducer", |row| {
        if selected.is_some() {
            return Ok(());
        }
        let producer = row[0];
        let scope = row[1];
        let body = row[2];
        let mut outputs = query.sequence("Output", producer)?.into_iter().collect::<LookupSet<_>>();
        let mut moved = vec![producer];
        let parent = query.required("OperationRegion", producer)?;
        for operation in query.operations(parent)? {
            if operation == scope {
                break;
            }
            if operation == producer {
                continue;
            }
            let mut used = false;
            query.for_each("DirectOperationUse", |row| {
                used |= row[0] == operation && outputs.contains(&row[1]);
                Ok(())
            })?;
            if used
                && query
                    .lookup("PureOperation", operation)?
                    .is_some_and(|pure| query.0.value_to_base::<bool>(pure))
                && query.contains("DiscardableOperation", operation)?
                && query.contains("RepeatableOperation", operation)?
            {
                moved.push(operation);
                outputs.extend(query.sequence("Output", operation)?);
            }
        }
        let mut safe = true;
        query.for_each("OperationWrite", |write| {
            safe &= write[0] != producer;
            Ok(())
        })?;
        query.for_each("OperationUse", |use_| {
            if outputs.contains(&use_[1]) && use_[0] != scope && !moved.contains(&use_[0]) {
                let region = query.required("OperationRegion", use_[0])?;
                safe &= query.contains("FlowScopeAncestor", (body, region))?;
            }
            Ok(())
        })?;
        query.for_each("EndDemand", |use_| {
            if outputs.contains(&use_[1]) && query.contains("LiveRegion", use_[0])? {
                safe &= query.contains("FlowScopeAncestor", (body, use_[0]))?;
            }
            Ok(())
        })?;
        query.for_function("AccumulatorArray", |keys, value| {
            if keys[0] == scope {
                if let Some(name) = query.constructor("ValueNameRef", value)? {
                    safe &= !outputs.contains(&name);
                }
            }
            Ok(())
        })?;
        if safe {
            if let Some(&first) = query.operations(body)?.first() {
                selected = Some((moved, first));
            }
        }
        Ok(())
    })?;
    let Some((moved, first)) = selected else {
        return Ok(false);
    };
    for operation in moved {
        edit::move_before(graph, operation, first)?;
    }
    Ok(true)
}

fn fuse_accumulators(graph: &mut EGraph, first: Value, second: Value) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let region = query.required("ControlRegion", (first, 0_i64))?;
    let groups = analysis::count(query, "AccumulatorGroupCount", first)?;
    let outputs = analysis::count(query, "OutputCount", first)?;
    let mut order = 0_i64;
    for operation in query.operations(region)? {
        order = order.max(query.0.value_to_base::<i64>(query.required("OperationOrder", operation)?) + 1);
    }
    graph.update(|mut sink| {
        sink.add("AccumulatorFuseSelected", (first, second))?;
        sink.set("FusionGroupBase", (first, second), groups)?;
        sink.set("FusionOutputBase", (first, second), outputs)?;
        sink.set("FusionOperationBase", (first, second), order)?;
        Ok(())
    })?;
    graph.parse_and_run_program(None, "(run-schedule (run graph-fusion-apply))")?;
    edit::remove(graph, &[second])
}

fn fuse(graph: &mut EGraph, first: Value, second: Value) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let a = query.required("ControlRegion", (first, 0_i64))?;
    let b = query.required("ControlRegion", (second, 0_i64))?;
    let parameters = query.sequence("RegionParameter", b)?;
    let arguments = query.sequence("RegionParameter", a)?;
    let mut replacements: Vec<_> = parameters.into_iter().zip(arguments).collect();
    let mut loads = Vec::new();
    query.for_each("FusionForwardLoad", |row| {
        if row[0] == first && row[1] == second {
            replacements.push((query.required("Output", (row[2], 0_i64))?, row[3]));
            loads.push(row[2]);
        }
        Ok(())
    })?;
    let clauses = analysis::count(query, "ResultCount", first)?;
    let outputs = analysis::count(query, "OutputCount", first)?;
    let mut offset = 0;
    for operation in query.operations(a)? {
        let order = query.0.value_to_base::<i64>(query.required("OperationOrder", operation)?);
        offset = offset.max(order + 1);
    }
    edit::replace(graph, &replacements)?;
    graph.update(|mut sink| {
        sink.add("FuseSelected", (first, second))?;
        sink.set("FusionClauseBase", (first, second), clauses)?;
        sink.set("FusionOutputBase", (first, second), outputs)?;
        sink.set("FusionOperationBase", (first, second), offset)?;
        Ok(())
    })?;
    graph.parse_and_run_program(None, "(run-schedule (run graph-fusion-apply))")?;
    loads.push(second);
    edit::remove(graph, &loads)
}

pub(super) fn motion(graph: &mut EGraph, hoist: bool) -> Result<bool, OptimizeError> {
    clear(graph)?;
    analysis::advance(graph)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate graph-fusion) (saturate dataflow-scopes dataflow-effects))",
    )?;
    let query = Query(graph);
    let mut selected = Vec::new();
    let mut operations = LookupSet::default();
    query.for_each("MotionCandidate", |row| {
        let crosses_scope =
            query.required("OperationRegion", row[0])? != query.required("OperationRegion", row[1])?;
        if operations.contains(&row[0])
            || (!hoist && crosses_scope)
            // Accumulator scopes execute once; hoisting only undoes producer sinking.
            || (crosses_scope && query.contains("AccumulatorOperation", row[1])?)
            || query.contains("InvocationDependent", row[0])?
            || query.contains("OperationTrap", row[0])?
            || query.contains("OperationsConflict", (row[0], row[1]))?
        {
            return Ok(());
        }
        if query.contains("BranchOperation", row[1])? {
            let Some(work) = query.lookup("OperationWork", row[0])? else {
                return Ok(());
            };
            if u64::try_from(&*query.0.value_to_base::<Z>(work)).map_or(true, |work| work > 4) {
                return Ok(());
            }
        }
        let mut available = true;
        query.for_each("OperationUse", |use_| {
            if use_[0] == row[0] {
                available &= query.contains("AvailableAt", (row[1], use_[1]))?;
            }
            Ok(())
        })?;
        if available {
            operations.insert(row[0]);
            selected.push((row[0], row[1]));
        }
        Ok(())
    })?;
    // Keep every destination fixed until the next analysis round.
    selected.retain(|&(_, owner)| !operations.contains(&owner));
    let changed = !selected.is_empty();
    for (operation, owner) in selected {
        edit::move_before(graph, operation, owner)?;
    }
    Ok(changed)
}

#[cfg(test)]
#[path = "reachability_tests.rs"]
mod reachability_tests;
