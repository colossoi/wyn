//! Demand-driven scalar admission followed by an egglog optimization fixed point.
//! The structural graph remains authoritative for effects and dispatch placement.
use super::{parse_program, source, timing, OptimizeError};
use crate::{LookupMap, LookupSet};
use egglog_engine::{EGraph, TermDag, TermId, Value, Write};
use std::collections::BTreeMap;
use std::time::Instant;

mod extract;
mod fold;
mod import;
mod placement;
mod scopes;
use import::Importer;

/// Egglog's own extracted DAG, retained unchanged for direct SSA emission.
pub(super) struct Selected {
    pub dag: TermDag,
    pub values: Vec<Value>,
    pub roots: LookupMap<(Value, Value), TermId>,
}

const RULES: &str = concat!(
    include_str!("scalar/schema.egg"),
    "\n",
    include_str!("scalar/analysis.egg"),
    "\n",
    include_str!("scalar/inline.egg"),
    "\n",
    include_str!("scalar/micro.egg"),
);

#[derive(Default)]
struct Facts {
    types: LookupMap<Value, Value>,
    parameters: LookupMap<Value, (Value, i64)>,
    owners: LookupMap<Value, Value>,
    callable: LookupMap<Value, Value>,
    results: LookupMap<Value, Value>,
    operations: LookupSet<Value>,
    duplicable: LookupSet<Value>,
    eligible: LookupSet<Value>,
    branches: LookupMap<Value, (Value, Value)>,
    roots: LookupMap<Value, LookupSet<Value>>,
    boundaries: LookupSet<Value>,
    dispatched: LookupSet<Value>,
    dispatch_operations: LookupMap<Value, Value>,
    dispatch_roots: LookupMap<Value, (Value, LookupSet<Value>)>,
    dispatch_scopes: LookupMap<Value, Value>,
    invocations: Vec<(Value, Value, Value)>,
}

impl Facts {
    fn read(graph: &EGraph) -> Result<Self, OptimizeError> {
        let mut facts = Self::default();
        graph.function_entries("SourceType", |row| {
            facts.types.insert(row.inputs[0], row.output);
        })?;
        graph.constructor_enodes("SourceParameter", |row| {
            facts.parameters.insert(
                row.children[2],
                (row.children[0], graph.value_to_base::<i64>(row.children[1])),
            );
        })?;
        graph.constructor_enodes("SourceFormal", |row| {
            facts.owners.insert(row.eclass, row.children[0]);
        })?;
        graph.constructor_enodes("SourceTerm", |row| {
            facts.owners.insert(row.eclass, row.children[0]);
        })?;
        graph.constructor_enodes("SourceCallable", |row| {
            facts.callable.insert(row.children[0], row.children[1]);
            facts.boundaries.insert(row.children[1]);
        })?;
        graph.constructor_enodes("SourceOperatorBody", |row| {
            facts.boundaries.insert(row.children[1]);
        })?;
        graph.constructor_enodes("SourceResult", |row| {
            facts.results.insert(row.children[0], row.children[1]);
            facts.roots.entry(row.children[0]).or_default().insert(row.children[1]);
        })?;
        let mut operation_values = LookupMap::default();
        graph.constructor_enodes("SourceOperationValue", |row| {
            operation_values.insert(row.children[0], row.children[1]);
            facts.operations.insert(row.children[1]);
        })?;
        graph.function_entries("SourceDuplicable", |row| {
            if graph.value_to_base::<bool>(row.output) {
                facts.duplicable.insert(row.inputs[0]);
            }
        })?;
        graph.constructor_enodes("SourceBranch", |row| {
            facts.branches.insert(row.children[0], (row.children[2], row.children[3]));
        })?;
        let mut planned = LookupMap::default();
        graph.constructor_enodes("PlannedOperation", |row| {
            planned.insert(row.children[0], (row.children[2], row.children[3]));
            if let Some(&value) = operation_values.get(&row.children[0]) {
                facts.roots.entry(row.children[3]).or_default().insert(value);
            }
        })?;
        graph.constructor_enodes("SourceOperand", |row| {
            if let Some(&(_, region)) = planned.get(&row.children[0]) {
                facts.roots.entry(region).or_default().insert(row.children[1]);
            }
        })?;
        graph.constructor_enodes("SourceOperatorBody", |row| {
            if let Some(&(plan, _)) = planned.get(&row.children[0]) {
                facts.invocations.push((plan, row.children[0], row.children[1]));
            }
        })?;
        let mut scalar_values = LookupSet::default();
        graph.constructor_enodes("ScalarValue", |row| {
            scalar_values.insert(row.children[0]);
        })?;
        let mut dispatches = LookupMap::default();
        graph.constructor_enodes("ScalarGroup", |row| {
            if scalar_values.contains(&row.children[0]) {
                dispatches.insert(row.children[0], row.children[1]);
            }
        })?;
        for (&op, &leader) in &dispatches {
            if let (Some(&source), Some(&(_, scope))) = (operation_values.get(&op), planned.get(&op)) {
                facts.dispatched.insert(source);
                facts.dispatch_operations.insert(source, leader);
                facts
                    .dispatch_roots
                    .entry(leader)
                    .or_insert_with(|| (scope, LookupSet::default()))
                    .1
                    .insert(source);
                if let Some(roots) = facts.roots.get_mut(&scope) {
                    roots.remove(&source);
                }
            }
        }
        graph.constructor_enodes("Enters", |row| {
            if !facts.boundaries.contains(&row.children[1]) {
                if let Some(&leader) = dispatches.get(&row.children[0]) {
                    facts.dispatch_scopes.insert(row.children[1], leader);
                }
            }
        })?;
        graph.constructor_enodes("SourceOperand", |row| {
            if let Some(leader) = dispatches.get(&row.children[0]) {
                if let Some((scope, roots)) = facts.dispatch_roots.get_mut(leader) {
                    roots.insert(row.children[1]);
                    if let Some(roots) = facts.roots.get_mut(scope) {
                        roots.remove(&row.children[1]);
                    }
                }
            }
        })?;
        for (&region, &source) in &facts.results {
            facts.roots.entry(region).or_default().insert(source);
        }
        let mut work = LookupMap::default();
        let mut duplicable = LookupSet::default();
        let mut calls: LookupMap<Value, Vec<Value>> = LookupMap::default();
        graph.function_entries("SourceRegionWork", |row| {
            work.insert(row.inputs[0], graph.value_to_base::<i64>(row.output));
        })?;
        graph.function_entries("SourceRegionDuplicable", |row| {
            if graph.value_to_base::<bool>(row.output) {
                duplicable.insert(row.inputs[0]);
            }
        })?;
        graph.constructor_enodes("SourceRegionEnters", |row| {
            calls.entry(row.children[0]).or_default().push(row.children[1]);
        })?;
        // Reproducible evaluation permits call-site substitution, including
        // partial arithmetic. ScalarTotal separately authorizes speculation.
        // The existing work estimate is an inlining profitability policy. A
        // cycle or unknown region cannot authorize recursive specialization.
        for &region in facts.callable.values() {
            if inline_candidate(
                region,
                &work,
                &calls,
                &duplicable,
                &mut LookupSet::default(),
                &mut 64,
            ) {
                facts.eligible.insert(region);
            }
        }
        Ok(facts)
    }
}

fn inline_candidate(
    region: Value,
    work: &LookupMap<Value, i64>,
    calls: &LookupMap<Value, Vec<Value>>,
    duplicable: &LookupSet<Value>,
    active: &mut LookupSet<Value>,
    remaining: &mut i64,
) -> bool {
    if !duplicable.contains(&region) || !active.insert(region) {
        return false;
    }
    *remaining -= work.get(&region).copied().unwrap_or(65);
    if *remaining < 0 {
        return false;
    }
    for &child in calls.get(&region).into_iter().flatten() {
        if !inline_candidate(child, work, calls, duplicable, active, remaining) {
            return false;
        }
    }
    active.remove(&region);
    true
}

pub(super) fn run(
    graph: &mut EGraph,
    identities: &source::Identities<'_>,
) -> Result<Selected, OptimizeError> {
    let _timing = timing::span("egglog scalar optimization");
    let facts = timing::time("egglog scalar / regions", || Facts::read(graph))?;
    fold::register(graph);
    graph.parse_and_run_program(Some("scalar rules".into()), RULES)?;
    let schedule = parse_program("scalar fixed point", include_str!("scalar/schedule.egg"))?;
    let mut regions = Vec::new();
    graph.constructor_enodes("RegionId", |row| {
        if facts.roots.contains_key(&row.eclass) {
            regions.push((graph.value_to_base::<i64>(row.children[0]), row.eclass));
        }
    })?;
    regions.sort_by_key(|&(id, _)| id);
    let mut cache = LookupMap::default();
    let mut templates = LookupMap::default();
    let mut groups: BTreeMap<i64, (Value, Vec<Value>, Vec<(Value, Value, bool)>)> = BTreeMap::new();
    let mut callback_contexts = LookupMap::default();
    let mut scope_contexts = LookupMap::default();
    let mut group_ids = LookupMap::default();
    graph.update(|mut sink| {
        for &(plan, _, region) in &facts.invocations {
            callback_contexts.insert(region, sink.add("ScalarKernel", plan)?);
        }
        for &region in identities.scopes.keys() {
            let owner = home(region, identities, &facts);
            let mut ancestor = region;
            let mut dispatch = None;
            while ancestor != owner {
                if let Some(&leader) = facts.dispatch_scopes.get(&ancestor) {
                    dispatch = Some(leader);
                    break;
                }
                let Some(&(Some(parent), _)) = identities.scopes.get(&ancestor) else {
                    break;
                };
                ancestor = parent;
            }
            let context = if let Some(leader) = dispatch {
                sink.add("ScalarDispatch", leader)?
            } else if let Some(&context) = callback_contexts.get(&owner) {
                context
            } else {
                sink.add("ScalarFunction", owner)?
            };
            scope_contexts.insert(region, context);
            sink.add("ScalarContextScope", (context, region))?;
        }
        for &(id, region) in &regions {
            let Some(&context) = scope_contexts.get(&region) else {
                return Err(egglog_engine::Error::ExtractError(
                    "missing scalar scope owner".into(),
                ));
            };
            let group_id = *group_ids.entry(context).or_insert(id);
            groups.entry(group_id).or_insert_with(|| (context, Vec::new(), Vec::new())).1.push(region);
        }
        let mut next = regions.last().map_or(0, |&(id, _)| id + 1);
        for (&leader, (scope, roots)) in &facts.dispatch_roots {
            let context = sink.add("ScalarDispatch", leader)?;
            let group_id = *group_ids.entry(context).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            });
            groups.entry(group_id).or_insert_with(|| (context, Vec::new(), Vec::new())).2.extend(
                roots.iter().map(|&source| {
                    (
                        *scope,
                        source,
                        facts.dispatch_operations.get(&source) == Some(&leader),
                    )
                }),
            );
            sink.add("ScalarContextScope", (context, *scope))?;
        }
        Ok(())
    })?;
    let mut largest = 0;
    let profile = std::env::var_os("WYN_SCALAR_PROFILE").is_some();
    for (&id, (context, members, roots)) in &groups {
        let start = Instant::now();
        let before = cache.len();
        graph.update(|mut sink| {
            sink.add("ScalarActive", *context)?;
            let mut importer = Importer::new(sink, identities, &facts, &mut cache, &mut templates);
            Ok(
                members.iter().try_for_each(|&region| importer.region(*context, region)).and_then(|()| {
                    roots.iter().try_for_each(|&(region, source, expand)| {
                        importer.root(*context, region, source, expand)
                    })
                }),
            )
        })??;
        largest = largest.max(cache.len() - before);
        // Each region admits only its demanded DAG and eligible helper bodies.
        // Egglog's incremental rules retain sharing with earlier regions.
        let imported = start.elapsed();
        graph.run_program(schedule.clone())?;
        graph.update(|mut sink| sink.remove("ScalarActive", *context))?;
        if profile {
            eprintln!(
                "scalar region {id}: {} admitted, import {:.3} ms, optimize {:.3} ms",
                cache.len() - before,
                imported.as_secs_f64() * 1000.0,
                (start.elapsed() - imported).as_secs_f64() * 1000.0
            );
        }
    }
    graph.update(|mut sink| {
        for &(plan, op, region) in &facts.invocations {
            if let Some(&source) = facts.results.get(&region) {
                let Some(&context) = callback_contexts.get(&region) else {
                    return Err(egglog_engine::Error::ExtractError(
                        "missing callback context".into(),
                    ));
                };
                if let Some(&root) = cache.get(&(context, source, true)) {
                    sink.add("ScalarInvocation", (plan, op, region, root))?;
                }
            }
        }
        Ok(())
    })?;
    let selected = placement::run(graph, identities, &facts)?;
    if timing::enabled() {
        let mut placements = 0;
        graph.constructor_enodes("ScalarSelectedPlacement", |_| placements += 1)?;
        eprintln!(
            "egglog scalar scopes: {}, optimization regions: {}, admitted values: {}, largest admission: {}, selected placements: {}",
            regions.len(),
            groups.len(),
            cache.len(),
            largest,
            placements
        );
    }
    if profile {
        for output in graph.parse_and_run_program(None, "(print-stats)")? {
            eprintln!("{output}");
        }
    }
    Ok(selected)
}

fn home(mut region: Value, identities: &source::Identities<'_>, facts: &Facts) -> Value {
    while !facts.boundaries.contains(&region) {
        let Some(&(Some(parent), _)) = identities.scopes.get(&region) else {
            break;
        };
        region = parent;
    }
    region
}
