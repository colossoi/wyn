//! Compose selected dataflow regions and preserve their occurrence order.
use super::dataflow::{
    edit,
    ir::{self, Code, Name, Parameter, RegionId},
};
use super::query::Query;
use super::{analysis, memory, OptimizeError};
use crate::LookupSet;
use egglog_engine::{EGraph, Value, Write};

mod compaction;
mod parallel;

#[cfg(test)]
mod tests;

pub(super) fn new_graph() -> Result<EGraph, OptimizeError> {
    let mut graph = EGraph::default();
    memory::load(&mut graph)?;
    Ok(graph)
}

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(Some("graph fusion".into()), include_str!("parallel.egg"))?;
    Ok(())
}

pub(super) fn run(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = super::timing::span("egglog graph fusion");
    memory::simplify(graph, false)?;
    loop {
        analysis::ensure(graph)?;
        crate::egglog::planning::analyze(graph)?;
        let fused = parallel::run(graph)?;
        if fused {
            // Ranked scatter and the remaining legacy readers consume the published tables.
            crate::egglog::planning::analyze(graph)?;
        }
        if (fused || !parallel::accumulators(graph, false)?)
            && !fuse_ranked_bucket(graph)?
            && !parallel::branches(graph)?
            && !compaction::run(graph)?
            && !parallel::accumulators(graph, true)?
            && !parallel::move_siblings(graph)?
        {
            return Ok(());
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
                    if operation == source || analysis::uses_output_of(query, source, operation)? {
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

pub(super) fn hoist(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    if parallel::hoist(graph)? {
        return Ok(true);
    }
    parallel::move_siblings(graph)
}

#[cfg(test)]
#[path = "reachability_tests.rs"]
mod reachability_tests;
