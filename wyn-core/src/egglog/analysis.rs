//! Finite structural summaries and native adapters for the retained planners.
use super::{parse_program, timing, OptimizeError};
use egglog_engine::{EGraph, Value, Write};
use std::collections::{BTreeMap, HashMap, HashSet};

pub(super) fn run(graph: &mut EGraph) -> Result<(), OptimizeError> {
    timing::time("egglog analysis / summaries", || {
        graph
            .run_program(parse_program(
                "source summary",
                "(run-schedule (saturate source-summary))",
            )?)
            .map_err(OptimizeError::from)
    })?;
    timing::time("egglog analysis / dependencies", || {
        graph
            .run_program(parse_program(
                "source dependencies",
                "(run-schedule (saturate source-dependencies))",
            )?)
            .map_err(OptimizeError::from)
    })?;
    timing::time("egglog analysis / execution facts", || execution_facts(graph))?;
    graph.parse_and_run_program(Some("fusion import".into()), include_str!("fusion/import.egg"))?;
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum WorkNode {
    Value(Value),
    Region(Value),
}

/// Preserve the existing bounded work estimate. Shared source values count once;
/// crossing a stored result uses its value instead of repeating its execution.
fn execution_facts(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let mut children: HashMap<WorkNode, Vec<WorkNode>> = HashMap::new();
    let mut work = HashMap::new();
    graph.function_entries("SourceWork", |row| {
        work.insert(
            WorkNode::Value(row.inputs[0]),
            graph.value_to_base::<i64>(row.output),
        );
    })?;
    graph.function_entries("SourceRegionWork", |row| {
        work.insert(
            WorkNode::Region(row.inputs[0]),
            graph.value_to_base::<i64>(row.output),
        );
    })?;
    graph.constructor_enodes("SourceSummaryEnters", |row| {
        children
            .entry(WorkNode::Value(row.children[0]))
            .or_default()
            .push(WorkNode::Region(row.children[1]));
    })?;
    graph.constructor_enodes("SourceRegionEnters", |row| {
        children
            .entry(WorkNode::Region(row.children[0]))
            .or_default()
            .push(WorkNode::Region(row.children[1]));
    })?;
    let mut positions = HashMap::new();
    graph.constructor_enodes("ImportedPosition", |row| {
        positions.insert(row.children[0], graph.value_to_base::<i64>(row.children[1]));
    })?;
    let mut readonly = HashMap::new();
    graph.function_entries("SourceReadOnly", |row| {
        readonly.insert(row.inputs[0], graph.value_to_base::<bool>(row.output));
    })?;
    let mut operations = HashMap::new();
    graph.constructor_enodes("SourceOperationValue", |row| {
        operations.insert(row.children[0], row.children[1]);
    })?;
    let mut scopes: HashMap<Value, BTreeMap<i64, (Value, Value)>> = HashMap::new();
    graph.constructor_enodes("SourceOperation", |row| {
        if let Some(&value) = operations.get(&row.children[0]) {
            if let Some(&position) = positions.get(&row.children[0]) {
                scopes.entry(row.children[1]).or_default().insert(position, (row.children[0], value));
            }
        }
    })?;
    let mut costs = Vec::new();
    for (&operation, &root) in &operations {
        let mut pending = vec![(WorkNode::Value(root), false)];
        let mut seen = HashSet::new();
        let mut active = HashSet::new();
        let mut count = 0;
        while let Some((value, leaving)) = pending.pop() {
            if leaving {
                active.remove(&value);
                continue;
            }
            if active.contains(&value) {
                count = 65;
                break;
            }
            if !seen.insert(value) {
                continue;
            }
            count += work.get(&value).copied().unwrap_or(65);
            if count > 64 {
                break;
            }
            active.insert(value);
            pending.push((value, true));
            pending.extend(children.get(&value).into_iter().flatten().map(|&child| (child, false)));
        }
        costs.push((operation, count <= 64));
    }
    graph.update(|mut sink| {
        for (op, cheap) in costs {
            sink.add("SourceOperationCheap", (op, cheap))?;
        }
        for operations in scopes.values() {
            let mut barrier = None;
            let mut readers = Vec::new();
            for (&position, &(op, value)) in operations {
                sink.add("ImportedPosition", (op, position))?;
                if let Some(before) = barrier {
                    sink.add("ImportedEffectOrder", (before, op))?;
                }
                if readonly.get(&value) == Some(&true) {
                    readers.push(op);
                } else {
                    for before in readers.drain(..) {
                        sink.add("ImportedEffectOrder", (before, op))?;
                    }
                    barrier = Some(op);
                }
            }
        }
        Ok(())
    })?;
    Ok(())
}
