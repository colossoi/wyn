//! Apply selected edits to keyed graph cells without reconstructing a program.
use super::ir::{self, Code};
use crate::egglog::graph::{ArgumentSite, Graph};
use crate::egglog::query::Query;
use crate::egglog::{analysis, output_error, OptimizeError};
use crate::{LookupMap, LookupSet};
use egglog_engine::{RawValues, Value, Write};
use wyn_graph::ReplacementForest;

pub(in crate::egglog) const OPERATION_CELLS: &[&str] = &[
    "InstructionWork",
    "LiteralBits",
    "InputDiet",
    "LoopIndex",
    "LoopStart",
    "LoopCount",
    "LoopStep",
    "LoopCondition",
    "OperationIndex",
    "OperationSlices",
    "OperationLayout",
    "OperationShape",
    "OperationDomain",
    "OperationWidth",
    "OperationExecution",
    "ResultKindOf",
    "ResultCount",
    "ResultValueCount",
    "ResultValue",
    "ResultPredicate",
    "ResultNeutral",
    "ResultNeutralCount",
    "ResultCommutative",
    "OutputClause",
    "OutputField",
    "OutputRole",
    "AccumulatorPolicy",
    "AccumulatorGroupCount",
    "AccumulatorArrayCount",
    "AccumulatorArray",
    "CallReturnType",
    "ScalarPayload",
];

pub(in crate::egglog) const OPERATION_RELATIONS: &[&str] = &[
    "Operation",
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
    graph.update(|mut sink| {
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
        "(run-schedule (saturate graph-indices graph-substitution-index) (saturate graph-substitution) (saturate graph-apply))",
    )?;
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
    for &table in &OPERATION_RELATIONS[2..] {
        Query(graph).for_each(table, |row| {
            if row[0] == operation {
                rows.push((table, row.to_vec()));
            }
            Ok(())
        })?;
    }
    let mut obsolete = Vec::new();
    for &table in OPERATION_CELLS {
        if ["InstructionWork", "LiteralBits", "ScalarPayload"].contains(&table)
            || (table == "OperationIndex" && matches!(code, Code::IndexValue))
        {
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
        "CloneDomain",
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
    while let Some(region) = pending.pop() {
        if members.insert(region) {
            result_names.extend(graph.parameters(region));
            argument_names.extend(graph.yield_values(region));
            for operation in query.operations(region)? {
                pending.extend(graph.attached_regions(operation));
                argument_names.extend(graph.inputs(operation));
                result_names.extend(graph.outputs(operation));
                for (slot, _) in graph.controls(operation) {
                    argument_names.extend(graph.arguments(operation, slot));
                }
                operations.push(operation);
            }
        }
    }
    graph.update(|mut sink| {
        for &region in &members {
            sink.add("CloneMember", (context, region))?;
        }
        for operation in operations {
            let cloned = sink.add("ClonedOperation", (context, operation))?;
            sink.add("Operation", cloned)?;
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
            operations.push((
                row[1],
                copy,
                region,
                query.0.operation_order(row[1])?,
                target,
                controls,
                graph
                    .inputs(row[1])
                    .into_iter()
                    .map(|name| query.required("CloneName", (context, name)))
                    .collect::<Result<Vec<_>, _>>()?,
                graph
                    .outputs(row[1])
                    .into_iter()
                    .map(|name| query.required("CloneName", (context, name)))
                    .collect::<Result<Vec<_>, _>>()?,
                clone_callbacks(graph.reducers(row[1]))?,
                clone_callbacks(graph.combiners(row[1]))?,
            ));
        }
        Ok(())
    })?;
    for &(original, copy, ..) in &operations {
        graph.copy_occurrence(original, copy)?;
    }
    graph.update(|mut sink| {
        for (region, parameters, values) in cloned_yields {
            sink.set_parameters(region, parameters)?;
            sink.set_yields(region, values)?;
        }
        for (_, operation, region, order, target, controls, inputs, outputs, reducers, combiners) in
            operations
        {
            sink.set_reducers(operation, reducers);
            sink.set_combiners(operation, combiners);
            sink.set_inputs(operation, inputs)?;
            sink.set_outputs(operation, outputs)?;
            for (slot, child, arguments) in controls {
                sink.set_control(operation, slot, child);
                for (index, name) in arguments.into_iter().enumerate() {
                    sink.set_argument(ArgumentSite(operation, slot, index), name)?;
                }
            }
            sink.set_region(operation, region);
            sink.set_order(operation, order);
            if let Some(target) = target {
                sink.set_call_target(operation, target);
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
    let query = Query(graph);
    let count = analysis::count(query, "ResultCount", operation)?;
    let outputs = query.0.outputs(operation);
    let mut names = Vec::with_capacity(outputs.len());
    let mut slots = LookupMap::default();
    let mut retained = 0_i64;
    for (slot, name) in outputs.into_iter().enumerate() {
        let slot = slot as i64;
        let owner = query.0.value_to_base::<i64>(query.required("OutputClause", (operation, slot))?);
        if owner != clause {
            names.push(name);
            slots.insert(slot, retained);
            retained += 1;
        }
    }
    let mut old = Vec::new();
    let mut next = Vec::new();
    for &table in OPERATION_CELLS {
        let results = table.starts_with("Result") && table != "ResultCount";
        let output = table.starts_with("Output");
        if !results && !output {
            continue;
        }
        query.for_function(table, |keys, value| {
            if keys[0] != operation {
                return Ok(());
            }
            old.push((table, keys.to_vec()));
            let slot = query.0.value_to_base::<i64>(keys[1]);
            let destination = if results {
                if slot == clause {
                    return Ok(());
                }
                if slot > clause {
                    slot - 1
                } else {
                    slot
                }
            } else {
                let Some(&slot) = slots.get(&slot) else {
                    return Ok(());
                };
                slot
            };
            let value = if table == "OutputClause" {
                let owner = query.0.value_to_base::<i64>(value);
                query.0.base_to_value(if owner > clause { owner - 1 } else { owner })
            } else {
                value
            };
            let mut keys = keys.to_vec();
            keys[1] = query.0.base_to_value(destination);
            next.push((table, keys, value));
            Ok(())
        })?;
    }
    let reducers = graph
        .reducers(operation)
        .into_iter()
        .filter_map(|(slot, region)| {
            (slot != clause).then_some((if slot > clause { slot - 1 } else { slot }, region))
        })
        .collect();
    graph.update(|mut sink| {
        sink.set_reducers(operation, reducers);
        for (table, keys) in old {
            sink.remove(table, RawValues(keys))?;
        }
        for (table, keys, value) in next {
            sink.set(table, RawValues(keys), value)?;
        }
        sink.remove("ResultCount", operation)?;
        sink.set("ResultCount", operation, count - 1)?;
        sink.set_outputs(operation, names)?;
        Ok(())
    })?;
    changed(graph)?;
    Ok(())
}
