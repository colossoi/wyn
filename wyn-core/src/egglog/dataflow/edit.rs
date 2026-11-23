//! Apply selected edits to keyed graph cells without reconstructing a program.
use super::ir::{self, Code};
use crate::egglog::graph::{ArgumentSite, Graph};
use crate::egglog::graph::{CollectiveInput, Formula, ParallelResults};
use crate::egglog::query::Query;
use crate::egglog::{analysis, output_error, OptimizeError};
use crate::{LookupMap, LookupSet, StableMap};
use egglog_engine::{RawValues, Value, Write};
use wyn_graph::ReplacementForest;

pub(in crate::egglog) const OPERATION_CELLS: &[&str] = &[
    "InstructionWork",
    "LiteralBits",
    "InputDiet",
    "OperationWidth",
    "OperationExecution",
    "CallReturnType",
    "ScalarPayload",
];

pub(in crate::egglog) const OPERATION_RELATIONS: &[&str] = &[
    "OperationCertificate",
    "MaterializedCopyOperation",
    "ViewOperation",
    "ArrayLiteralOperation",
    "RangeOperation",
    "PreservesInputShape",
    "ConstantOperation",
    "ProductOperation",
    "ProjectionOperation",
    "LengthOperation",
    "IndexOperation",
    "IndexCalculationOperation",
    "CallOperation",
    "BranchOperation",
    "CountedLoopOperation",
    "WhileLoopOperation",
    "ParallelOperation",
    "AccumulatorOperation",
    "DiscardableOperation",
    "RepeatableOperation",
    "OrderedOperation",
    "LaneDependentOperation",
    "ReadsInput",
    "WritesInput",
    "ConsumesInput",
    "AliasesInput",
    "FreshOutput",
    "ScratchAllocation",
    "ProvenTotalOperation",
];

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
    changed(graph)?;
    let revision = revision(graph)?;
    graph.update(|mut sink| {
        for table in current {
            sink.remove(table, key())?;
            sink.set(table, key(), revision)?;
        }
        Ok(())
    })?;
    Ok(())
}

pub(in crate::egglog) fn load(graph: &mut Graph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(Some("graph edits".into()), include_str!("edits.egg"))?;
    graph.parse_and_run_program(Some("graph cloning".into()), include_str!("clone.egg"))?;
    Ok(())
}

pub(in crate::egglog) fn clear(graph: &mut Graph) -> Result<(), OptimizeError> {
    for table in [
        "ForwardCandidate",
        "ValueReplacement",
        "FoldedConstant",
        "BranchCandidate",
        "RewrittenAccumulatorInfo",
        "RewrittenIndex",
        "IndexReplacement",
        "RewrittenShape",
        "RewrittenShapePrefix",
        "RewrittenSlice",
        "RewrittenSlices",
        "RewrittenSlicesPrefix",
    ] {
        graph.clear_function(table)?;
    }
    analysis::advance(graph)
}

pub(in crate::egglog) fn remove(graph: &mut Graph, operations: &[Value]) -> Result<(), OptimizeError> {
    remove_without_revision(graph, operations)?;
    changed(graph)
}

pub(in crate::egglog) fn remove_without_revision(
    graph: &mut Graph,
    operations: &[Value],
) -> Result<(), OptimizeError> {
    let operations: LookupSet<_> = operations.iter().copied().collect();
    let query = Query(graph);
    let mut rows = Vec::new();
    for &table in OPERATION_CELLS {
        query.for_function(table, |keys, _| {
            if operations.contains(&keys[0]) {
                rows.push((table, keys.to_vec()));
            }
            Ok(())
        })?;
    }
    for &table in OPERATION_RELATIONS {
        query.for_each(table, |keys| {
            if operations.contains(&keys[0]) {
                rows.push((table, keys.to_vec()));
            }
            Ok(())
        })?;
    }
    graph.update(|mut sink| {
        for (table, keys) in rows {
            sink.remove(table, RawValues(keys))?;
        }
        for operation in operations {
            sink.remove_operation(operation);
        }
        Ok(())
    })?;
    Ok(())
}

pub(in crate::egglog) fn replace(
    graph: &mut Graph,
    replacements: &[(Value, Value)],
) -> Result<(), OptimizeError> {
    let mut resolved = ReplacementForest::new();
    for &(old, new) in replacements {
        if old != new {
            resolved.replace(old, new).map_err(|error| output_error(error.to_string()))?;
        }
    }
    let mut arguments = Vec::new();
    let mut operands = Vec::new();
    let mut yields = Vec::new();
    {
        let graph = graph.structure().read();
        for (old, new) in resolved.into_map() {
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
    seed_replacements(graph, replacements)?;
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
    seed_replacements(graph, replacements)?;
    replace_attributes_with_seeded_values(graph)
}

fn seed_replacements(graph: &mut Graph, replacements: &[(Value, Value)]) -> Result<(), OptimizeError> {
    clear(graph)?;
    let mut forest = ReplacementForest::new();
    for &(old, new) in replacements {
        if old != new {
            forest.replace(old, new).map_err(|error| output_error(error.to_string()))?;
        }
    }
    let mut collective = Vec::new();
    {
        let query = Query(graph);
        let structure = graph.structure().read();
        for (old, new) in forest.into_map() {
            if let Some(sites) = structure.collective.uses.get(&old) {
                collective.extend(
                    sites
                        .iter()
                        .filter(|(site, _)| !site.kind.uses_raw_name())
                        .map(|&(site, field)| (site, field, new)),
                );
            }
            if let Some(raw_old) = query.value_name(old)? {
                if let Some(sites) = structure.collective.uses.get(&raw_old) {
                    if let Some(raw_new) = query.value_name(new)? {
                        collective.extend(
                            sites
                                .iter()
                                .filter(|(site, _)| site.kind.uses_raw_name())
                                .map(|&(site, field)| (site, field, raw_new)),
                        );
                    }
                }
            }
        }
    }
    graph.update(|mut sink| {
        for (site, field, value) in collective {
            sink.redirect_collective_input(site, field, value)?;
        }
        for &(old, new) in replacements {
            sink.set("ValueReplacement", old, new)?;
        }
        Ok(())
    })?;
    Ok(())
}

fn replace_attributes_with_seeded_values(graph: &mut Graph) -> Result<(), OptimizeError> {
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
    // Operation classes and access properties describe the payload, so replace them together.
    let mut rows = Vec::new();
    for &table in &OPERATION_RELATIONS[1..] {
        Query(graph).for_each(table, |row| {
            if row[0] == operation {
                rows.push((table, row.to_vec()));
            }
            Ok(())
        })?;
    }
    let mut obsolete = Vec::new();
    for &table in OPERATION_CELLS {
        if ["InstructionWork", "LiteralBits", "ScalarPayload"].contains(&table) {
            continue;
        }
        Query(graph).for_function(table, |keys, _| {
            if keys[0] == operation {
                obsolete.push((table, keys.to_vec()));
            }
            Ok(())
        })?;
    }
    graph.update(|mut sink| {
        sink.clear_collective_inputs(operation);
        sink.clear_collective_layouts(operation);
        sink.clear_formulas(operation, matches!(code, Code::IndexValue));
        sink.clear_operation_domain(operation);
        for (table, row) in rows {
            sink.remove(table, RawValues(row))?;
        }
        for (table, keys) in obsolete {
            sink.remove(table, RawValues(keys))?;
        }
        sink.clear_bodies(operation);
        sink.remove_call_target(operation);
        sink.remove("ScalarPayload", operation)?;
        sink.remove("InstructionWork", operation)?;
        sink.remove("LiteralBits", operation)?;

        Ok(())
    })?;
    graph.update(|mut sink| Ok(ir::properties(&mut sink, operation, code)))??;
    changed(graph)?;
    Ok(())
}

fn collective_name(query: Query<'_>, kind: CollectiveInput, value: Value) -> Result<Value, OptimizeError> {
    if !kind.uses_raw_name() {
        return Ok(value);
    }
    name_reference(query, value)
}

fn name_reference(query: Query<'_>, value: Value) -> Result<Value, OptimizeError> {
    query
        .constructor("ValueNameRef", value)?
        .ok_or_else(|| output_error("operand has no value identity"))
}

struct ClonedOperation {
    source: Value,
    key: Value,
    region: Value,
    order: i64,
    target: Option<Value>,
    controls: Vec<(i64, Value, Vec<Value>)>,
    inputs: Vec<Value>,
    outputs: Vec<Value>,
    reducers: Vec<(i64, Value)>,
    combiners: Vec<(i64, Value)>,
    collective: StableMap<(i64, CollectiveInput), Vec<Value>>,
    formulas: Vec<(Formula, Value)>,
    results: Option<ParallelResults>,
    policies: Vec<Option<Value>>,
    domain: Option<Value>,
}

/// Copy a selected region using fresh graph identities, substituting its entry edge.
pub(in crate::egglog) fn clone_region(
    graph: &mut Graph,
    context: Value,
    region: Value,
    arguments: &[Value],
) -> Result<Value, OptimizeError> {
    for table in [
        "CloneMember",
        "CloneOperation",
        "CloneLocal",
        "CloneNameDemand",
        "CloneSubstitute",
        "CloneIsLocal",
        "CloneIsSubstituted",
        "CloneName",
        "CloneIndexDemand",
        "CloneIndex",
        "CloneShapeDemand",
        "CloneShapeSlot",
        "CloneShapePrefix",
        "CloneShape",
        "CloneSliceDemand",
        "CloneSliceSlot",
        "CloneSlicePrefix",
        "CloneSlices",
        "CloneDimDemand",
        "CloneDim",
        "CloneAccInfo",
        "CloneAccInfoDemand",
        "CloneReturnDemand",
        "CloneReturn",
    ] {
        graph.clear_function(table)?;
    }
    let parameters = Query(graph).0.parameters(region);
    if parameters.len() != arguments.len() {
        return Err(output_error("region entry edge has the wrong argument count"));
    }
    let query = Query(graph);
    let mut pending = vec![region];
    let mut members = LookupSet::new();
    let mut operations = Vec::new();
    let mut argument_names = LookupSet::new();
    let mut result_names = LookupSet::new();
    let mut formula_demands = LookupSet::new();
    let mut domains = LookupMap::new();
    while let Some(region) = pending.pop() {
        if members.insert(region) {
            result_names.extend(graph.parameters(region));
            argument_names.extend(graph.yield_values(region));
            for operation in query.operations(region)? {
                pending.extend(graph.attached_regions(operation));
                argument_names.extend(graph.inputs(operation));
                for ((_, kind), values) in graph.collective_groups(operation) {
                    for value in values {
                        argument_names.insert(collective_name(query, kind, value)?);
                    }
                }
                result_names.extend(graph.outputs(operation));
                for (slot, _) in graph.controls(operation) {
                    argument_names.extend(graph.arguments(operation, slot));
                }
                if let Some(domain) = graph.operation_domain_at(operation) {
                    if !domains.contains_key(&domain) {
                        let layout = graph.iteration_domain(domain)?;
                        for name in layout
                            .ordinal
                            .into_iter()
                            .chain(layout.dimensions.iter().filter_map(|dimension| dimension.coordinate))
                        {
                            argument_names.insert(name_reference(query, name)?);
                        }
                        for extent in layout.dimensions.iter().filter_map(|dimension| dimension.extent) {
                            formula_demands.insert(("CloneIndexDemand", extent));
                        }
                        domains.insert(domain, layout);
                    }
                }
                for (kind, value) in graph.formulas(operation) {
                    let demand = match kind {
                        Formula::Index | Formula::LoopCount => "CloneIndexDemand",
                        Formula::Extents => "CloneShapeDemand",
                        Formula::Slices => "CloneSliceDemand",
                        Formula::Layout => continue,
                    };
                    formula_demands.insert((demand, value));
                }
                operations.push(operation);
            }
        }
    }
    graph.update(|mut sink| {
        for &domain in domains.keys() {
            sink.add("ClonedDomain", (context, domain))?;
        }
        for (demand, value) in formula_demands {
            sink.add(demand, (context, value))?;
        }
        for &region in &members {
            sink.add("CloneMember", (context, region))?;
        }
        for operation in operations {
            sink.add("ClonedOperation", (context, operation))?;

            sink.add("CloneOperation", (context, operation))?;
        }
        for (parameter, &argument) in parameters.into_iter().zip(arguments) {
            sink.set("CloneSubstitute", (context, parameter), argument)?;
        }
        for argument in argument_names {
            sink.add("CloneNameDemand", (context, argument))?;
        }
        for result in result_names {
            sink.add("CloneNameDemand", (context, result))?;
            sink.add("CloneLocal", (context, result))?;
        }
        Ok(())
    })?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate clone-index) (saturate clone-names) (saturate clone-edges))",
    )?;
    let query = Query(graph);
    let mut clone_domains = Vec::new();
    let clone_name = |name: Value| -> Result<Value, OptimizeError> {
        let name = name_reference(query, name)?;
        query.name(query.required("CloneName", (context, name))?)
    };
    for (domain, mut layout) in domains {
        let cloned = query
            .constructor("ClonedDomain", (context, domain))?
            .ok_or_else(|| output_error("cloned domain has no identity"))?;
        layout.ordinal = layout.ordinal.map(clone_name).transpose()?;
        for dimension in &mut layout.dimensions {
            dimension.coordinate = dimension.coordinate.map(clone_name).transpose()?;
            dimension.extent = dimension
                .extent
                .map(|extent| query.required("CloneIndex", (context, extent)))
                .transpose()?;
        }
        clone_domains.push((cloned, layout));
    }
    let mut cloned_yields = Vec::new();
    for region in members {
        let new_region = query
            .constructor("ClonedRegion", (context, region))?
            .ok_or_else(|| output_error("cloned region has no identity"))?;
        let values = graph
            .yield_values(region)
            .into_iter()
            .map(|name| query.required("CloneName", (context, name)))
            .collect::<Result<Vec<_>, _>>()?;
        let parameters = graph
            .parameters(region)
            .into_iter()
            .map(|name| query.required("CloneName", (context, name)))
            .collect::<Result<Vec<_>, _>>()?;
        cloned_yields.push((new_region, parameters, values));
    }
    let mut operations = Vec::new();
    query.for_each("CloneOperation", |row| {
        if row[0] == context {
            let Some(copy) = query.constructor("ClonedOperation", (context, row[1]))? else {
                return Err(output_error("cloned operation has no identity"));
            };
            let region = query.0.operation_region(row[1])?;
            let Some(region) = query.constructor("ClonedRegion", (context, region))? else {
                return Err(output_error("cloned operation has no region"));
            };
            let target = if matches!(graph.code(row[1])?, Code::Call) {
                Some(graph.call_target(row[1])?)
            } else {
                None
            };
            let controls = graph
                .controls(row[1])
                .map(|(slot, child)| {
                    let clone = query
                        .constructor("ClonedRegion", (context, child))?
                        .ok_or_else(|| output_error("cloned control has no region"))?;
                    let arguments = graph
                        .arguments(row[1], slot)
                        .into_iter()
                        .map(|name| query.required("CloneName", (context, name)))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok((slot, clone, arguments))
                })
                .collect::<Result<Vec<_>, OptimizeError>>()?;
            let clone_callbacks = |callbacks: Vec<(i64, Value)>| -> Result<Vec<_>, OptimizeError> {
                callbacks
                    .into_iter()
                    .map(|(slot, child)| {
                        let Some(region) = query.constructor("ClonedRegion", (context, child))? else {
                            return Err(output_error("cloned callback has no region"));
                        };
                        Ok((slot, region))
                    })
                    .collect()
            };
            let mut collective = graph.collective_groups(row[1]);
            for (&(_, kind), values) in &mut collective {
                for value in values {
                    let name = collective_name(query, kind, *value)?;
                    let clone = query.required("CloneName", (context, name))?;
                    *value = if kind.uses_raw_name() { query.name(clone)? } else { clone };
                }
            }
            let mut formulas = graph.formulas(row[1]);
            for (kind, value) in &mut formulas {
                let clone = match kind {
                    Formula::Index | Formula::LoopCount => "CloneIndex",
                    Formula::Extents => "CloneShape",
                    Formula::Slices => "CloneSlices",
                    Formula::Layout => continue,
                };
                *value = query.required(clone, (context, *value))?;
            }
            operations.push(ClonedOperation {
                source: row[1],
                key: copy,
                region,
                order: query.0.operation_order(row[1])?,
                target,
                controls,
                inputs: graph
                    .inputs(row[1])
                    .into_iter()
                    .map(|name| query.required("CloneName", (context, name)))
                    .collect::<Result<Vec<_>, _>>()?,
                outputs: graph
                    .outputs(row[1])
                    .into_iter()
                    .map(|name| query.required("CloneName", (context, name)))
                    .collect::<Result<Vec<_>, _>>()?,
                reducers: clone_callbacks(graph.reducers(row[1]))?,
                combiners: clone_callbacks(graph.combiners(row[1]))?,
                collective,
                formulas,
                results: graph.parallel_results_at(row[1]),
                policies: graph.accumulator_policies(row[1]),
                domain: graph
                    .operation_domain_at(row[1])
                    .map(|domain| {
                        query
                            .constructor("ClonedDomain", (context, domain))?
                            .ok_or_else(|| output_error("cloned operation has no domain"))
                    })
                    .transpose()?,
            });
        }
        Ok(())
    })?;
    for operation in &operations {
        graph.copy_occurrence(operation.source, operation.key)?;
    }
    graph.update(|mut sink| {
        for (id, layout) in clone_domains {
            sink.set_domain(id, layout);
        }
        for (region, parameters, values) in cloned_yields {
            sink.set_parameters(region, parameters)?;
            sink.set_yields(region, values)?;
        }
        for operation in operations {
            let key = operation.key;
            if let Some(domain) = operation.domain {
                sink.set_operation_domain(key, domain);
            }
            if let Some(results) = operation.results {
                sink.set_parallel_results(key, results);
            }
            if !operation.policies.is_empty() {
                sink.set_accumulator_policies(key, operation.policies);
            }
            for (kind, value) in operation.formulas {
                sink.set_formula(key, kind, value)?;
            }
            for ((group, kind), values) in operation.collective {
                sink.set_collective_inputs(key, group, kind, values)?;
            }
            sink.set_reducers(key, operation.reducers);
            sink.set_combiners(key, operation.combiners);
            sink.set_inputs(key, operation.inputs)?;
            sink.set_outputs(key, operation.outputs)?;
            for (slot, child, arguments) in operation.controls {
                sink.set_control(key, slot, child);
                for (index, name) in arguments.into_iter().enumerate() {
                    sink.set_argument(ArgumentSite(key, slot, index), name)?;
                }
            }
            sink.set_region(key, operation.region);
            sink.set_order(key, operation.order);
            if let Some(target) = operation.target {
                sink.set_call_target(key, target);
            }
        }
        Ok(())
    })?;
    let Some(region) = Query(graph).constructor("ClonedRegion", (context, region))? else {
        return Err(output_error("selected region was not cloned"));
    };
    Ok(region)
}

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
