//! Apply selected edits to keyed graph cells without reconstructing a program.
use super::ir::{self, Code};
use crate::egglog::graph::Graph;
use crate::egglog::query::Query;
use crate::egglog::{analysis, output_error, OptimizeError};
use crate::{LookupMap, LookupSet};
use egglog_engine::{RawValues, Value, Write};

pub(in crate::egglog) const OPERATION_CELLS: &[&str] = &[
    "InstructionWork",
    "LiteralBits",
    "Input",
    "InputCount",
    "InputDiet",
    "Output",
    "OutputCount",
    "ControlRegion",
    "ControlRole",
    "EdgeArgument",
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
    "ResultReducer",
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
    "AccumulatorCombiner",
    "CallTarget",
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
    seed_replacements(graph, replacements)?;
    graph.parse_and_run_program(None, "(run-schedule (saturate graph-operands))")?;
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
    let count = analysis::count(Query(graph), "InputCount", operation)?;
    graph.update(|mut sink| {
        for slot in 0..count {
            sink.remove("Input", (operation, slot))?;
            sink.remove("InputDiet", (operation, slot))?;
        }
        sink.remove("InputCount", operation)?;
        sink.set("InputCount", operation, names.len() as i64)?;
        for (slot, &name) in names.iter().enumerate() {
            sink.set("Input", (operation, slot as i64), name)?;
        }
        Ok(())
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
        if [
            "InstructionWork",
            "LiteralBits",
            "ScalarPayload",
            "Input",
            "InputCount",
            "Output",
            "OutputCount",
        ]
        .contains(&table)
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
    let parameters = Query(graph).sequence("RegionParameter", region)?;
    if parameters.len() != arguments.len() {
        return Err(output_error("region entry edge has the wrong argument count"));
    }
    let query = Query(graph);
    let mut pending = vec![region];
    let mut members = LookupSet::new();
    let mut operations = Vec::new();
    while let Some(region) = pending.pop() {
        if members.insert(region) {
            for operation in query.operations(region)? {
                pending.extend(query.attached_regions(operation)?);
                operations.push(operation);
            }
        }
    }
    graph.update(|mut sink| {
        for region in members {
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
        Ok(())
    })?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate clone-index) (saturate clone-names) (saturate clone-edges))",
    )?;
    let query = Query(graph);
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
            operations.push((row[1], copy, region, query.0.operation_order(row[1])?));
        }
        Ok(())
    })?;
    for &(original, copy, _, _) in &operations {
        graph.copy_occurrence(original, copy)?;
    }
    graph.update(|mut sink| {
        for (_, operation, region, order) in operations {
            sink.set_region(operation, region);
            sink.set_order(operation, order);
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
    let outputs = analysis::count(query, "OutputCount", operation)?;
    let mut slots = LookupMap::default();
    let mut retained = 0_i64;
    for slot in 0..outputs {
        let owner = query.0.value_to_base::<i64>(query.required("OutputClause", (operation, slot))?);
        if owner != clause {
            slots.insert(slot, retained);
            retained += 1;
        }
    }
    let mut old = Vec::new();
    let mut next = Vec::new();
    for &table in OPERATION_CELLS {
        let results = table.starts_with("Result") && table != "ResultCount";
        let output = table.starts_with("Output") && table != "OutputCount";
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
    graph.update(|mut sink| {
        for (table, keys) in old {
            sink.remove(table, RawValues(keys))?;
        }
        for (table, keys, value) in next {
            sink.set(table, RawValues(keys), value)?;
        }
        sink.remove("ResultCount", operation)?;
        sink.set("ResultCount", operation, count - 1)?;
        sink.remove("OutputCount", operation)?;
        sink.set("OutputCount", operation, retained)?;
        Ok(())
    })?;
    changed(graph)?;
    Ok(())
}
