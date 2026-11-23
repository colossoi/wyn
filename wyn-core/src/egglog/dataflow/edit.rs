//! Apply selected edits to keyed graph cells without reconstructing a program.
use super::cse;
use super::ir::{self, Code};
use super::properties;
use crate::egglog::analysis::scopes::Changes;
use crate::egglog::graph::{Formula, Graph};
use crate::egglog::query::Query;
use crate::egglog::{analysis, output_error, OptimizeError};
use crate::LookupSet;
use egglog_engine::{RawValues, Value, Write};
use wyn_graph::ReplacementForest;

pub(in crate::egglog) fn revision(graph: &Graph) -> Result<i64, OptimizeError> {
    analysis::count(Query(graph), "GraphEditRevision", RawValues(Vec::new()))
}

pub(in crate::egglog) fn changed(graph: &mut Graph) -> Result<(), OptimizeError> {
    let Some(revision) = revision(graph)?.checked_add(1) else {
        return Err(output_error("graph edit revision exhausted"));
    };
    graph.update(|mut sink| {
        sink.remove("GraphEditRevision", RawValues(Vec::new()))?;
        sink.set("GraphEditRevision", RawValues(Vec::new()), revision)
    })?;
    Ok(())
}

/// Advance the graph revision after an edit has repaired its affected facts.
pub(in crate::egglog) fn preserve_revisions(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let previous = graph.base_to_value(revision(graph)?);
    let key = || RawValues(Vec::new());
    let mut current = Vec::new();
    for table in ["DataflowRevision", "PlanningRevision"] {
        if query.lookup(table, key())? == Some(previous) {
            current.push(table);
        }
    }
    let repaired = current.contains(&"DataflowRevision");
    if repaired {
        analysis::scopes::repair_transforms(graph, [])?;
    }
    changed(graph)?;
    let revision = revision(graph)?;
    graph.update(|mut sink| {
        for table in current {
            sink.remove(table, key())?;
            sink.set(table, key(), revision)?;
        }
        Ok(())
    })?;
    if repaired {
        take_changes(graph);
    }
    Ok(())
}

/// Invalidate the lexical reuse cache before consuming the executable edits.
pub(in crate::egglog) fn take_changes(graph: &mut Graph) -> Changes {
    cse::invalidate(graph);
    std::mem::take(&mut graph.proof_changes)
}

/// Reorder existing siblings while retaining proofs about definitions and effects.
pub(in crate::egglog) fn reorder_regions(
    graph: &mut Graph,
    updated: Vec<(Value, Vec<Value>)>,
) -> Result<(), OptimizeError> {
    let previous = updated
        .iter()
        .map(|&(region, _)| Ok((region, graph.region_order(region)?.to_vec())))
        .collect::<Result<_, OptimizeError>>()?;
    graph.update(|mut sink| {
        for (_, order) in &updated {
            for (slot, &operation) in order.iter().enumerate() {
                sink.set_order(operation, slot as i64);
            }
        }
        Ok(())
    })?;
    analysis::replace_order(graph, previous, updated)?;
    preserve_revisions(graph)
}

pub(in crate::egglog) fn load(graph: &mut Graph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(Some("graph edits".into()), include_str!("edits.egg"))?;
    Ok(())
}

fn clear_substitution(graph: &mut Graph) -> Result<(), OptimizeError> {
    for table in [
        "ValueReplacement",
        "RewrittenAccumulatorInfo",
        "RewrittenIndex",
        "IndexReplacement",
        "ChangedShape",
        "ChangedSlices",
        "SliceReplacement",
        "RewrittenShape",
        "RewrittenShapePrefix",
        "RewrittenSlice",
        "RewrittenSlices",
        "RewrittenSlicesPrefix",
    ] {
        graph.clear_function(table)?;
    }
    Ok(())
}

pub(in crate::egglog) fn remove(graph: &mut Graph, operations: &[Value]) -> Result<(), OptimizeError> {
    remove_without_revision(graph, operations)?;
    changed(graph)
}

pub(in crate::egglog) fn remove_without_revision(
    graph: &mut Graph,
    operations: &[Value],
) -> Result<(), OptimizeError> {
    let keys: LookupSet<_> = operations.iter().copied().collect();
    let mut removed = Vec::new();
    for operation in keys {
        if graph.has_operation(operation) {
            removed.push((
                operation,
                graph.code(operation)?,
                graph.input_count(operation),
                graph.output_count(operation),
            ));
        }
    }
    graph.update(|mut sink| {
        for (operation, code, inputs, outputs) in removed {
            properties::remove(&mut sink, operation, &code, inputs, outputs)?;
            sink.remove_operation(operation)?;
        }
        Ok(())
    })?;
    Ok(())
}

pub(in crate::egglog) fn replace(
    graph: &mut Graph,
    replacements: &[(Value, Value)],
) -> Result<(), OptimizeError> {
    let replacements = resolve_replacements(replacements)?;
    if replacements.is_empty() {
        return Ok(());
    }
    let mut arguments = Vec::new();
    let mut operands = Vec::new();
    let mut yields = Vec::new();
    {
        let graph = graph.structure().read();
        for &(old, new) in &replacements {
            if let Some(sites) = graph.argument_uses.get(&old) {
                arguments.extend(sites.iter().map(|&site| (site, new)));
            }
            if let Some(sites) = graph.inputs.uses.get(&old) {
                operands.extend(sites.iter().map(|&(op, slot)| (op, slot, new)));
            }
            if let Some(sites) = graph.yields.uses.get(&old) {
                yields.extend(sites.iter().map(|&(region, slot)| (region, slot, new)));
            }
        }
    }
    seed_replacements(graph, &replacements)?;
    graph.update(|mut sink| {
        for (region, slot, value) in yields {
            sink.set_yield(region, slot, value)?;
        }
        for (op, slot, value) in operands {
            sink.set_input(op, slot, value)?;
        }
        for (site, value) in arguments {
            sink.set_argument(site, value)?;
        }
        Ok(())
    })?;
    replace_attributes_with_seeded_values(graph)
}

pub(in crate::egglog) fn replace_attributes(
    graph: &mut Graph,
    replacements: &[(Value, Value)],
) -> Result<(), OptimizeError> {
    let replacements = resolve_replacements(replacements)?;
    if replacements.is_empty() {
        return Ok(());
    }
    seed_replacements(graph, &replacements)?;
    replace_attributes_with_seeded_values(graph)
}

pub(in crate::egglog) fn resolve_replacements(
    replacements: &[(Value, Value)],
) -> Result<Vec<(Value, Value)>, OptimizeError> {
    let mut forest = ReplacementForest::new();
    for &(old, new) in replacements {
        if old != new {
            forest.replace(old, new).map_err(|error| output_error(error.to_string()))?;
        }
    }
    Ok(forest.into_map().into_iter().collect())
}

fn seed_replacements(graph: &mut Graph, replacements: &[(Value, Value)]) -> Result<(), OptimizeError> {
    let mut collective = Vec::new();
    let mut certificates = Vec::new();
    let mut indices = Vec::new();
    {
        let query = Query(graph);
        let structure = graph.structure().read();
        for &(old, new) in replacements {
            if let Some(sites) = structure.collective.uses.get(&old) {
                collective.extend(
                    sites
                        .iter()
                        .filter(|(site, _)| !site.kind.uses_raw_name())
                        .map(|&(site, field)| (site, field, new)),
                );
            }
            let Some(raw_old) = query.value_name(old)? else {
                continue;
            };
            if query.constructor("IndexValue", raw_old)?.is_some() {
                indices.push((old, new));
            }
            let raw_inputs = structure.collective.uses.get(&raw_old);
            let guards = structure.certificates.uses.get(&raw_old);
            if raw_inputs.is_none() && guards.is_none() {
                continue;
            }
            let Some(raw_new) = query.value_name(new)? else {
                continue;
            };
            if let Some(sites) = raw_inputs {
                collective.extend(
                    sites
                        .iter()
                        .filter(|(site, _)| site.kind.uses_raw_name())
                        .map(|&(site, field)| (site, field, raw_new)),
                );
            }
            if let Some(sites) = guards {
                certificates.extend(sites.iter().map(|&site| (site, raw_old, raw_new)));
            }
        }
    }
    graph.update(|mut sink| {
        for (site, old, replacement) in certificates {
            sink.redirect_certificate(site, old, replacement);
        }
        for (site, field, value) in collective {
            sink.redirect_collective_input(site, field, value)?;
        }
        for (old, new) in indices {
            sink.set("ValueReplacement", old, new)?;
        }
        Ok(())
    })?;
    Ok(())
}

fn replace_attributes_with_seeded_values(graph: &mut Graph) -> Result<(), OptimizeError> {
    if graph.get_size("ValueReplacement") == 0 {
        return changed(graph);
    }
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate graph-indices graph-substitution-index) (saturate graph-substitution))",
    )?;
    let query = Query(graph);
    let mut formulas = Vec::new();
    let mut extents = Vec::new();
    {
        let structure = graph.structure().read();
        for (table, kinds) in [
            ("RewrittenIndex", &[Formula::Index, Formula::LoopCount][..]),
            ("RewrittenShape", &[Formula::Extents][..]),
            ("RewrittenSlices", &[Formula::Slices][..]),
        ] {
            let domain_extents = table == "RewrittenIndex" && !structure.domains.extent_uses.is_empty();
            if !domain_extents
                && !kinds
                    .iter()
                    .any(|kind| structure.formulas.by_kind.get(kind).is_some_and(|terms| !terms.is_empty()))
            {
                continue;
            }
            query.for_function(table, |keys, replacement| {
                if keys[0] != replacement {
                    if table == "RewrittenIndex" {
                        if let Some(sites) = structure.domains.extent_uses.get(&keys[0]) {
                            extents.extend(sites.iter().map(|&(domain, slot)| (domain, slot, replacement)));
                        }
                    }
                    for &kind in kinds {
                        if let Some(sites) = structure.formulas.sites.get(&(kind, keys[0])) {
                            formulas.extend(sites.iter().map(|&op| (op, kind, replacement)));
                        }
                    }
                }
                Ok(())
            })?;
        }
    }
    if !formulas.is_empty() || !extents.is_empty() {
        graph.update(|mut sink| {
            for (domain, slot, extent) in extents {
                sink.set_domain_extent(domain, slot, extent)?;
            }
            for (op, kind, replacement) in formulas {
                sink.set_formula(op, kind, replacement)?;
            }
            Ok(())
        })?;
    }
    graph.parse_and_run_program(None, "(run-schedule (saturate graph-apply))")?;
    clear_substitution(graph)?;
    changed(graph)?;
    Ok(())
}

pub(in crate::egglog) fn inputs(
    graph: &mut Graph,
    operation: Value,
    names: &[Value],
) -> Result<(), OptimizeError> {
    let count = graph.input_count(operation);
    graph.update(|mut sink| {
        for slot in 0..count {
            sink.remove("InputDiet", (operation, slot))?;
        }
        sink.set_inputs(operation, names.to_vec())
    })?;
    changed(graph)?;
    Ok(())
}

/// Insert operations at an execution point, retaining the relative order of both groups.
pub(in crate::egglog) fn splice(
    graph: &mut Graph,
    at: Value,
    inserted: &[Value],
) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let region = query.0.operation_region(at)?;
    let mut order = query.operations(region)?;
    let Some(position) = order.iter().position(|&operation| operation == at) else {
        return Err(output_error("splice point is outside its region"));
    };
    order.splice(position..=position, inserted.iter().copied());
    let query = Query(graph);
    let mut edits = Vec::new();
    for (index, operation) in order.into_iter().enumerate() {
        let index = index as i64;
        let move_region = query.0.operation_region(operation)? != region;
        let set_order = query.0.operation_order(operation)? != index;
        if move_region || set_order {
            edits.push((operation, index, move_region, set_order));
        }
    }
    graph.update(|mut sink| {
        for (operation, index, move_region, set_order) in edits {
            if move_region {
                sink.set_region(operation, region);
            }
            if set_order {
                sink.set_order(operation, index);
            }
        }
        Ok(())
    })?;
    remove(graph, &[at])
}

pub(in crate::egglog) fn move_before(
    graph: &mut Graph,
    operation: Value,
    before: Value,
) -> Result<(), OptimizeError> {
    let destination = graph.operation_region(before)?;
    move_into(graph, operation, destination, Some(before))
}

/// Move into a region, appending when no anchor is supplied.
pub(in crate::egglog) fn move_into(
    graph: &mut Graph,
    operation: Value,
    destination: Value,
    before: Option<Value>,
) -> Result<(), OptimizeError> {
    let mut order = Query(graph).operations(destination)?;
    order.retain(|&item| item != operation);
    let position = if let Some(before) = before {
        let Some(position) = order.iter().position(|&item| item == before) else {
            return Err(output_error("motion destination is outside its execution region"));
        };
        position
    } else {
        order.len()
    };
    order.insert(position, operation);
    graph.update(|mut sink| {
        for (slot, item) in order.into_iter().enumerate() {
            sink.set_region(item, destination);
            sink.set_order(item, slot as i64);
        }
        Ok(())
    })?;
    changed(graph)?;
    Ok(())
}

pub(in crate::egglog) fn set_code(
    graph: &mut Graph,
    operation: Value,
    code: Code,
) -> Result<(), OptimizeError> {
    let previous = graph.code(operation)?;
    let (inputs, outputs) = (graph.input_count(operation), graph.output_count(operation));
    graph.update(|mut sink| {
        properties::remove(&mut sink, operation, &previous, inputs, outputs)?;
        sink.clear_collective_inputs(operation);
        sink.clear_collective_layouts(operation);
        sink.clear_formulas(operation, matches!(code, Code::IndexValue));
        sink.clear_operation_domain(operation);
        sink.clear_bodies(operation);
        sink.remove_call_target(operation);
        Ok(())
    })?;
    graph.update(|mut sink| Ok(ir::properties(&mut sink, operation, code)))??;
    changed(graph)?;
    Ok(())
}

pub(in crate::egglog) use super::clone::clone_region;

/// Remove a result clause and compact only its keyed slot numbers.
pub(in crate::egglog) fn remove_clause(
    graph: &mut Graph,
    operation: Value,
    clause: i64,
) -> Result<(), OptimizeError> {
    let mut results = graph.parallel_results(operation)?;
    let Ok(index) = usize::try_from(clause) else {
        return Err(output_error("negative collective clause"));
    };
    if index >= results.kinds.len() {
        return Err(output_error("missing collective clause"));
    }
    results.kinds.remove(index);
    let outputs = graph.outputs(operation);
    let mut names = Vec::with_capacity(outputs.len());
    let mut bindings = Vec::with_capacity(outputs.len());
    for (name, mut binding) in outputs.into_iter().zip(results.bindings) {
        let Some(owner) = binding.clause else {
            return Err(output_error("collective output has no clause"));
        };
        if owner != clause {
            binding.clause = Some(if owner > clause { owner - 1 } else { owner });
            names.push(name);
            bindings.push(binding);
        }
    }
    results.bindings = bindings;
    let reducers = graph
        .reducers(operation)
        .into_iter()
        .filter_map(|(slot, region)| {
            (slot != clause).then_some((if slot > clause { slot - 1 } else { slot }, region))
        })
        .collect();
    let retained_edges = graph.collective_groups(operation);
    graph.update(|mut sink| {
        sink.clear_collective_inputs(operation);
        for ((group, kind), values) in retained_edges {
            if group != clause {
                let next = if group > clause { group - 1 } else { group };
                sink.set_collective_inputs(operation, next, kind, values)?;
            }
        }
        sink.set_reducers(operation, reducers);
        sink.set_parallel_results(operation, results);
        sink.set_outputs(operation, names)?;
        Ok(())
    })?;
    changed(graph)?;
    Ok(())
}
