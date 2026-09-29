//! Run execution placement and structural dispatch planning on native facts.
use super::{timing, OptimizeError};
use crate::PipelineTopologyPolicy;
use egglog_engine::{EGraph, RawValues, Value, Write};
use std::collections::{BTreeMap, HashMap};
use wyn_graph::topo_sort_by_dependencies;

pub(super) const RULES: &str = concat!(
    include_str!("planning.egg"),
    "\n",
    include_str!("execution.egg"),
    "\n",
    include_str!("schedule.egg"),
    "\n",
    include_str!("residency.egg"),
    "\n",
    include_str!("allocation.egg"),
    "\n",
    include_str!("reuse.egg"),
    "\n",
    include_str!("dispatch.egg"),
    "\n",
    include_str!("epilogues.egg"),
    "\n",
);
pub(super) const KEYS: &str = "(datatype ExprKey (ExprId i64))\n(datatype TypeKey (TypeId i64))\n";

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(Some("planning keys".into()), KEYS)?;
    graph.parse_and_run_program(Some("structural planning rules".into()), RULES)?;
    Ok(())
}

pub(super) fn place(graph: &mut EGraph, topology: PipelineTopologyPolicy) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(
        Some("planning bridge".into()),
        include_str!("planning/import.egg"),
    )?;
    let policy = match topology {
        PipelineTopologyPolicy::AllowGenerated => "AllowGeneratedTopology",
        PipelineTopologyPolicy::AuthoredOnly => "AuthoredTopology",
    };
    graph.update(|mut sink| sink.add(policy, RawValues(Vec::new())))?;
    timing::time("egglog placement / import", || {
        graph.parse_and_run_program(None, "(run-schedule (saturate planning-import))")
    })?;
    timing::time("egglog placement / order", || order(graph))?;
    graph.parse_and_run_program(
        Some("execution placement".into()),
        include_str!("planning/place.egg"),
    )?;
    Ok(())
}

/// Contracted operations must be ordered again before forming scalar dispatches.
/// Dependencies in other scopes remain invocation inputs, not local order edges.
fn order(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let mut sites = HashMap::new();
    let mut source_order = BTreeMap::new();
    graph.constructor_enodes("PlannedSourcePosition", |row| {
        let op = row.children[0];
        let region = row.children[1];
        let position = graph.value_to_base::<i64>(row.children[2]);
        sites.insert(op, region);
        source_order.entry(position).or_insert_with(Vec::new).push(op);
    })?;
    let mut dependencies: HashMap<Value, Vec<Value>> = HashMap::new();
    graph.constructor_enodes("SourceDependency", |row| {
        let after = row.children[0];
        let before = row.children[1];
        if sites.get(&after) == sites.get(&before) {
            dependencies.entry(after).or_default().push(before);
        }
    })?;
    let mut gates: HashMap<Value, Vec<Value>> = HashMap::new();
    graph.constructor_enodes("EffectInput", |row| {
        gates.entry(row.children[0]).or_default().push(row.children[1]);
    })?;
    graph.constructor_enodes("EffectWait", |row| {
        let after = row.children[0];
        for &before in gates.get(&row.children[1]).into_iter().flatten() {
            if sites.get(&after) == sites.get(&before) {
                dependencies.entry(after).or_default().push(before);
            }
        }
    })?;
    let ordered = topo_sort_by_dependencies(source_order.into_values().flatten(), |op, out| {
        out.extend(dependencies.get(&op).into_iter().flatten().copied());
    })
    .map_err(|_| OptimizeError::Extraction("cycle in the fused execution graph".into()))?;
    let mut previous = HashMap::new();
    graph.update(|mut sink| {
        for (position, op) in ordered.into_iter().enumerate() {
            if let Some(&region) = sites.get(&op) {
                sink.add("SourcePosition", (op, region, position as i64))?;
                if let Some(before) = previous.insert(region, op) {
                    sink.add("Consecutive", (before, op))?;
                }
            }
        }
        Ok(())
    })?;
    Ok(())
}

pub(super) fn schedule(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(
        Some("structural scheduling".into()),
        include_str!("planning/run.egg"),
    )?;
    if std::env::var_os("WYN_EGGLOG_PROFILE").is_some() {
        for output in graph.parse_and_run_program(None, "(print-stats) (print-size)")? {
            eprintln!("{output}");
        }
    }
    Ok(())
}
