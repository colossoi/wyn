//! Compose selected dataflow regions and preserve their occurrence order.
use super::dataflow::{
    edit,
    ir::{self, Code, Name, Parameter, RegionId},
};
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
        "RepeatedArrayMotion",
        "ArrayMotionReadOnly",
        "FusionPreparation",
        "FusionBoundary",
        "FusionNested",
        "FusionCollective",
        "FusionHasForwardedLoad",
        "FusionPrivateWrites",
        "SinkProducer",
        "PlacementReadOnly",
        "PlacementMember",
        "PlacementValue",
        "PlacementInput",
        "PlacementPath",
        "PlacementDemand",
        "PlacementHasDemand",
        "PlacementDead",
        "PlacementSite",
        "PlacementUnder",
        "PlacementGuarded",
        "PlacementAvailable",
        "PlacementUnsafe",
        "BranchPair",
        "BranchUnsafe",
        "BranchFuseSelected",
        "BranchOperationBase",
        "AccumulatorPair",
        "AccumulatorFuseSelected",
        "FusionGroupBase",
    ] {
        graph.clear_function(table)?;
    }
    analysis::advance(graph)
}

pub(super) fn run(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = super::timing::span("egglog graph fusion");
    memory::simplify(graph, false)?;
    loop {
        clear(graph)?;
        analysis::ensure(graph)?;
        crate::egglog::planning::analyze(graph)?;
        graph.parse_and_run_program(None, "(run-schedule (saturate placement-index) (saturate placement-demand dataflow-scopes) (saturate placement-legality graph-work) (saturate placement-select) (saturate graph-fusion dataflow-scopes) (saturate dataflow-effects) (saturate graph-fusion-legality) (saturate graph-fusion-select))")?;
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
                let branch = query.first("BranchPair")?;
                if fuse_ranked_bucket(graph)? {
                    memory::simplify(graph, false)?;
                    continue;
                }
                if let Some(pair) = branch {
                    graph.update(|mut sink| sink.add("BranchFuseSelected", (pair[0], pair[1])))?;
                    graph.parse_and_run_program(None, "(run-schedule (run graph-fusion-apply))")?;
                    edit::remove(graph, &[pair[1]])?;
                    memory::simplify(graph, false)?;
                    continue;
                }
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

struct MapLevel {
    operation: Value,
    region: Value,
    index: Value,
    source: Value,
    result: Value,
}

fn collected_map(query: Query<'_>, operation: Value) -> Result<Option<MapLevel>, OptimizeError> {
    if !query.contains("ParallelOperation", operation)?
        || analysis::count(query, "ResultCount", operation)? != 1
        || query.enode("Collected", query.required("ResultKindOf", (operation, 0_i64))?)?.is_none()
        || analysis::count(query, "ResultValueCount", (operation, 0_i64))? != 1
    {
        return Ok(None);
    }
    let region = query.required("ControlRegion", (operation, 0_i64))?;
    let domain = query.required("OperationDomain", operation)?;
    if analysis::count(query, "DomainRank", domain)? != 1 {
        return Ok(None);
    }
    let axis = query.required("DomainAxis", (domain, 0_i64))?;
    let mut found = None;
    for candidate in query.operations(region)? {
        if query.contains("IndexOperation", candidate)?
            && analysis::count(query, "InputCount", candidate)? == 2
            && query.required("Input", (candidate, 1_i64))?
                == query
                    .constructor("ValueNameRef", axis)?
                    .ok_or_else(|| super::output_error("axis name missing"))?
        {
            found = Some(candidate);
            break;
        }
    }
    let Some(index) = found else { return Ok(None) };
    Ok(Some(MapLevel {
        operation,
        region,
        index,
        source: query.required("Input", (index, 0_i64))?,
        result: query.required("ResultValue", (operation, 0_i64, 0_i64))?,
    }))
}

fn output_producer(query: Query<'_>, name: Value) -> Result<Option<Value>, OptimizeError> {
    let mut producer = None;
    query.for_function("Output", |keys, value| {
        if value == name {
            producer = Some(keys[0]);
        }
        Ok(())
    })?;
    Ok(producer)
}

fn fuse_ranked_bucket(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let mut candidate = None;
    query.for_each("LiveOperation", |row| {
        if candidate.is_some() {
            return Ok(());
        }
        let bucket = row[0];
        let Code::BucketScatter { coordinates, depth } =
            ir::code(graph, query.required("OperationPayload", bucket)?)
        else {
            return Ok(());
        };
        let expected = vec![(0..depth).map(|axis| axis as i64).collect::<Vec<_>>()];
        if depth == 0 || coordinates != expected || analysis::count(query, "InputCount", bucket)? != 2 {
            return Ok(());
        }
        let items = query.required("Input", (bucket, 1_i64))?;
        let mut levels = Vec::new();
        let mut output = items;
        for _ in 0..depth {
            let Some(operation) = output_producer(query, output)? else {
                return Ok(());
            };
            let Some(level) = collected_map(query, operation)? else {
                return Ok(());
            };
            output = level.result;
            levels.push(level);
        }
        let mut exclusive = true;
        query.for_each("DirectOperationUse", |use_| {
            if use_[1] == items && use_[0] != bucket {
                exclusive = false;
            }
            Ok(())
        })?;
        query.for_each("EndDemand", |demand| {
            if demand[1] == items {
                exclusive = false;
            }
            Ok(())
        })?;
        let mut moved = Vec::new();
        for level in &levels {
            for operation in query.operations(level.region)? {
                if operation != level.index && !levels.iter().any(|nested| nested.operation == operation) {
                    moved.push(operation);
                    exclusive &= query
                        .lookup("PureOperation", operation)?
                        .is_some_and(|value| graph.value_to_base::<bool>(value));
                }
            }
        }
        let mut hoisted = Vec::new();
        for pair in levels.windows(2) {
            if let Some(source) = output_producer(query, pair[1].source)? {
                for operation in query.operations(pair[0].region)? {
                    if operation == source || query.contains("OperationDependency", (source, operation))? {
                        moved.retain(|&item| item != operation);
                        exclusive &= query
                            .lookup("MovableOperation", operation)?
                            .is_some_and(|value| graph.value_to_base::<bool>(value));
                        hoisted.push(operation);
                    }
                }
            }
        }
        if !exclusive {
            return Ok(());
        }
        // Scatter inputs must exist before the traversal, including inputs to hoisted producers.
        let anchor = levels[0].operation;
        let mut hoisted_outputs = LookupSet::default();
        for &operation in &hoisted {
            if query.contains("InvocationDependent", operation)? {
                return Ok(());
            }
            let mut available = true;
            query.for_each("OperationUse", |use_| {
                if use_[0] == operation && !hoisted_outputs.contains(&use_[1]) {
                    available &= query.contains("AvailableAt", (anchor, use_[1]))?;
                }
                Ok(())
            })?;
            if !available {
                return Ok(());
            }
            hoisted_outputs.extend(query.sequence("Output", operation)?);
        }
        for level in &levels {
            if !hoisted_outputs.contains(&level.source)
                && !query.contains("AvailableAt", (anchor, level.source))?
            {
                return Ok(());
            }
        }
        candidate = Some((bucket, levels, hoisted, moved));
        Ok(())
    })?;
    let Some((bucket, levels, hoisted, moved)) = candidate else {
        return Ok(false);
    };
    let outer = &levels[0];
    let inner = &levels[levels.len() - 1];
    let query = Query(graph);
    let callback = query.required("ControlRegion", (bucket, 0_i64))?;
    let old_parameter = query.required("RegionParameter", (callback, 0_i64))?;
    edit::replace(graph, &[(old_parameter, inner.result)])?;
    for operation in hoisted {
        edit::move_before(graph, operation, outer.operation)?;
    }
    let first = Query(graph).operations(callback)?.first().copied();
    for operation in moved {
        edit::move_into(graph, operation, callback, first)?;
    }
    let query = Query(graph);
    let destination = query.required("Input", (bucket, 0_i64))?;
    let inputs =
        std::iter::once(destination).chain(levels.iter().map(|level| level.source)).collect::<Vec<_>>();
    edit::inputs(graph, bucket, &inputs)?;
    let query = Query(graph);
    let parameters = levels
        .iter()
        .map(|level| -> Result<Parameter, OptimizeError> {
            let name = query.required("Output", (level.index, 0_i64))?;
            let value = query.name(name)?;
            Ok(Parameter {
                name: Name::Value(value),
                info: query.required("ValueType", value)?,
                diet: query.required("ParameterDiet", (level.region, 0_i64))?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let old_count = analysis::count(query, "RegionParameterCount", callback)?;
    graph.update(|mut sink| {
        Ok((|| -> Result<(), OptimizeError> {
            for slot in 0..old_count {
                sink.remove("RegionParameter", (callback, slot))?;
                sink.remove("ParameterDiet", (callback, slot))?;
            }
            sink.remove("RegionParameterCount", callback)?;
            ir::parameters(&mut sink, RegionId(callback), &parameters)?;
            sink.remove("OperationPayload", bucket)?;
            ir::properties(
                &mut sink,
                bucket,
                Code::BucketScatter {
                    coordinates: (0..levels.len()).map(|axis| vec![axis as i64]).collect(),
                    depth: levels.len(),
                },
            )
        })())
    })??;
    let removed = levels.iter().flat_map(|level| [level.index, level.operation]).collect::<Vec<_>>();
    edit::remove(graph, &removed)?;
    Ok(true)
}

fn sink_producer(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let mut dead = Vec::new();
    query.for_each("PlacementDead", |row| {
        dead.push(row[0]);
        Ok(())
    })?;
    if !dead.is_empty() {
        edit::remove(graph, &dead)?;
        return Ok(true);
    }
    let Some(site) = query.first("SinkProducer")? else {
        return Ok(false);
    };
    let (producer, scope, body) = (site[0], site[1], site[2]);
    let first = query.operations(body)?.first().copied();
    let mut moved = Vec::new();
    let parent = query.required("OperationRegion", producer)?;
    for operation in query.operations(parent)? {
        if query
            .lookup("PlacementMember", (producer, operation))?
            .is_some_and(|v| graph.value_to_base::<bool>(v))
        {
            moved.push((operation, query.required("OperationOrder", operation)?));
        }
    }
    let copy = query.contains("BranchOperation", scope)?;
    let edge = i64::from(copy && query.required("ControlRegion", (scope, 1_i64))? == body);
    for &(operation, _) in &moved {
        edit::move_into(graph, operation, body, first)?;
    }
    if copy {
        let cloned = edit::clone_region(graph, scope, body, &[])?;
        graph.update(|mut sink| {
            sink.remove("ControlRegion", (scope, edge))?;
            sink.set("ControlRegion", (scope, edge), cloned)?;
            for (operation, order) in moved {
                sink.remove("OperationRegion", operation)?;
                sink.set("OperationRegion", operation, parent)?;
                sink.remove("OperationOrder", operation)?;
                sink.set("OperationOrder", operation, order)?;
            }
            Ok(())
        })?;
        edit::changed(graph)?;
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
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate graph-fusion motion-work graph-work) (saturate dataflow-scopes dataflow-effects))",
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
