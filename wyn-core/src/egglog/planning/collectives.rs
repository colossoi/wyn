//! Decode legal collective recipes, rank launches, and publish one choice.
use super::super::{
    cost::{CollectiveWork, CostModel},
    query::Query,
};
use super::OptimizeError;
use crate::{LookupMap, LookupSet};
use egglog_engine::{EGraph, Value, Write};

fn summarize(graph: &EGraph, work: &mut LookupMap<Value, CollectiveWork>) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let integer = |value| graph.value_to_base::<i64>(value).max(0) as u64;
    query.for_function("CollectiveStatePrefix", |keys, value| {
        if let Some(work) = work.get_mut(&keys[0]) {
            work.state_bytes = work.state_bytes.max(integer(value));
        }
        Ok(())
    })?;
    let mut inputs = LookupSet::default();
    query.for_each("PlannedOperation", |row| {
        let Some(work) = work.get_mut(&row[1]) else {
            return Ok(());
        };
        if let Some(region) = query.lookup("SourceOperatorBody", (row[0],))? {
            let cost = query.lookup("SourceRegionWork", (region,))?.map(integer).unwrap_or(65).min(65);
            if query.flag("PlanMapping", (row[0],))? {
                work.map_ops = work.map_ops.saturating_add(cost);
            } else {
                work.combine_ops = work.combine_ops.saturating_add(cost);
            }
        }
        if let Some(count) = query.lookup("SourceInputCount", (row[0],))? {
            for index in 0..integer(count) as i64 {
                let input = query.required("SourceInput", (row[0], index))?;
                // Internal fusion edges do not cause global-memory reads.
                if !query.flag("PlanInternal", (row[2], input))? {
                    inputs.insert((row[1], input));
                }
            }
        }
        Ok(())
    })?;
    let mut elements = LookupMap::default();
    query.for_each("SourceArrayElement", |row| {
        if elements.insert(row[0], row[1]).is_some_and(|old| old != row[1]) {
            return Err(OptimizeError::Output("ambiguous SourceArrayElement".into()));
        }
        Ok(())
    })?;
    for (owner, input) in inputs {
        let ty = query.required("SourceType", (input,))?;
        let element = elements.get(&ty).copied().unwrap_or(ty);
        let work = work.get_mut(&owner).expect("input belongs to a selected collective");
        work.input_bytes = work
            .input_bytes
            .saturating_add(query.lookup("SourceStorageStride", (element,))?.map(integer).unwrap_or(4));
    }
    query.for_each("ArrayResult", |row| {
        if let Some(work) = work.get_mut(&row[0]) {
            work.output_bytes = work
                .output_bytes
                .saturating_add(query.lookup("SourceStorageStride", (row[2],))?.map(integer).unwrap_or(4));
        }
        Ok(())
    })?;
    for work in work.values_mut() {
        work.state_bytes = work.state_bytes.max(4);
        work.combine_ops = work.combine_ops.max(1);
    }
    Ok(())
}

pub(super) fn select(graph: &mut EGraph, costs: CostModel) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut choices = Vec::new();
    let mut workloads = LookupMap::default();
    query.for_each("ParallelCollective", |row| {
        let op = row[0];
        let Some(grid) = query.lookup("OperationGrid", (op,))? else {
            return Ok(());
        };
        let mut width = None;
        if query.enode("AutomaticGrid", grid)?.is_some() {
            let extent = query.required("InputDomain", (op,))?;
            if let Some(elements) = query.lookup("StaticCapacityBound", (extent,))? {
                width =
                    Some(graph.value_to_base::<i64>(query.required("CollectiveChunkWidth", (op,))?) as u32);
                let recipe = query.required("Plan", (op,))?;
                workloads.insert(
                    op,
                    CollectiveWork {
                        elements: graph.value_to_base::<i64>(elements).max(0) as u64,
                        scan: query.enode("Prefixes", recipe)?.is_some(),
                        ..CollectiveWork::default()
                    },
                );
            }
        }
        choices.push((op, width));
        Ok(())
    })?;
    summarize(graph, &mut workloads)?;
    let choices = choices
        .into_iter()
        .map(|(op, width)| {
            let Some(width) = width else {
                return Ok((op, 256, 256));
            };
            let work = workloads[&op];
            let launch =
                costs.select(work, width, 256).map_err(|message| OptimizeError::Output(message.into()))?;
            if std::env::var_os("WYN_SCHEDULE_REPORT").is_some() {
                eprintln!(
                    "schedule {op:?}: {work:?} {launch:?} {:?}",
                    costs.estimate(work, launch)
                );
            }
            Ok((op, i64::from(launch.groups), i64::from(launch.carry_width)))
        })
        .collect::<Result<Vec<_>, OptimizeError>>()?;
    graph.update(|mut sink| {
        for (op, groups, combine) in choices {
            sink.set("SelectedCollectiveGroups", (op,), groups)?;
            sink.set("CollectiveCombineWidth", (op,), combine)?;
        }
        Ok::<_, egglog_engine::Error>(())
    })?;
    Ok(())
}
