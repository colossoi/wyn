//! Derived facts are disposable; executable edges remain in the flat graph.
use super::dataflow::edit;
use super::query::Query;
use super::{output_error, timing, OptimizeError};
use crate::egglog::graph::{CollectiveInput, Graph};
use crate::{LookupMap, StableMap};
use egglog_engine::sort::S;
use egglog_engine::{IntoValues, RawValues, Value, Write};

mod calls;
mod formulas;

pub(in crate::egglog) use formulas::formula_names;
mod signatures;

pub(super) fn load(graph: &mut Graph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(
        Some("dataflow analysis".into()),
        concat!(
            include_str!("analysis/schema.egg"),
            "\n",
            include_str!("analysis/dataflow.egg"),
            "\n",
            include_str!("analysis/resources.egg"),
            "\n",
            include_str!("analysis/integer-ranges.egg")
        ),
    )?;
    Ok(())
}

/// Retire body-dependent proofs before solving a new graph revision.
pub(super) fn invalidate(graph: &mut Graph) -> Result<(), OptimizeError> {
    let _timing = timing::span("egglog full analysis invalidation");
    edit::clear(graph)?;
    for table in [
        "DataflowRevision",
        "PlanningRevision",
        "ValuePrimitive",
        "ConstantValue",
        "LivenessRequest",
        "LivenessRegion",
        "LivenessOperation",
        "LivenessName",
        "InlineCall",
        "RecursiveFunction",
        "NameOrdinal",
        "NameLiveAfter",
        "FlowNext",
        "FlowLive",
        "FlowLiveOut",
        "FlowExecution",
        "FlowLiveAfter",
        "FlowUse",
        "FlowKilled",
        "FlowEdge",
        "FlowParameter",
        "FlowArgument",
        "Defines",
        "ResultOrigin",
        "FormalOrigin",
        "ReducerBody",
        "CollectiveScopeUse",
        "CollectiveLaneUse",
        "FoldSeedUse",
        "PackedClause",
        "CollectedClause",
        "CompactedClause",
        "ContributedClause",
        "FoldClause",
        "AccumulatorBackingUse",
        "SingleAccumulatorBacking",
        "ReducerWidthInputs",
        "DefinitionRegion",
        "RootFunction",
        "RegionOwnedBy",
        "LiveFunction",
        "LiveRegion",
        "LiveOperation",
        "AvailableIn",
        "AvailableAt",
        "AvailableEnd",
        "AvailabilityDemand",
        "EntryDemand",
        "EndDemand",
        "BranchResult",
        "BinaryOperands",
        "IndexOperands",
        "LengthMeasure",
        "RegionAttributeUse",
        "RegionCapture",
        "DirectOperationUse",
        "OperationAttributeUse",
        "OperationUse",
        "InvalidDataflow",
        "RegularLength",
        "RowDependent",
        "VaryingShape",
        "RowShapePair",
        "SameRowShape",
        "RowExtent",
        "IrregularResult",
        "ShapeDemand",
        "SliceDemand",
        "TotalIndex",
        "ArrayOuterBound",
        "ArrayRowWidth",
        "LoopMatchesOuterBound",
        "ConstantBound",
        "StorageValue",
        "ArrayValue",
        "ResourceDemand",
        "ResourceParent",
        "ResourceAnchor",
        "ResourceRoot",
        "TokenArray",
        "TokenCapability",
        "ConsumedToken",
        "TokenUnavailableAt",
        "OperationRead",
        "OperationWrite",
        "OperationTrap",
        "RegionRead",
        "RegionWrite",
        "RegionTrap",
        "OperationsConflict",
        "EffectCompare",
        "OperationTotal",
        "IntegerBoundsDemand",
        "TrackedI32",
        "IntegerLower",
        "IntegerUpper",
        "DiscardableRead",
        "PureOperation",
        "MovableOperation",
        "InvocationDependent",
        "InvocationRegion",
        "LoopInduction",
        "LoopValueCarry",
        "AccumulatorParameter",
        "FlowScopeParent",
        "FlowScopeAncestor",
        "SourceAlias",
        "SourceField",
        "SourceSelect",
        "SourceProjection",
        "SourceForwardField",
        "SourceSelectResolved",
        "FlowOperand",
        "FlowFrom",
        "FlowRead",
        "FlowScopeRequirement",
        "FlowInputAt",
        "FlowStructuralInput",
        "FlowEvaluationRead",
        "FlowEffect",
        "FlowReach",
        "FlowDemand",
    ] {
        graph.clear_function(table)?;
    }
    advance(graph)
}

/// Reseed rules after clearing derived tables while keeping valid graph facts.
pub(super) fn advance(graph: &mut Graph) -> Result<(), OptimizeError> {
    let revision = count(Query(graph), "AnalysisRevision", RawValues(Vec::new()))?;
    graph.update(|mut sink| {
        sink.remove("AnalysisRevision", RawValues(Vec::new()))?;
        sink.set("AnalysisRevision", RawValues(Vec::new()), revision + 1)
    })?;
    Ok(())
}

/// Refresh lexical ownership, reachable calls, and recursion without solving dataflow.
pub(super) fn structure(graph: &mut Graph, for_inlining: bool) -> Result<(), OptimizeError> {
    let _timing = timing::span("egglog graph structure");
    invalidate(graph)?;
    calls::publish(graph, for_inlining)?;
    if for_inlining {
        formulas::publish(graph, true)?;
    } else {
        publish_proof_operands(graph)?;
    }
    graph.parse_and_run_program(None, "(run-schedule (saturate graph-structure flow-scopes))")?;
    Ok(())
}

// Correlate the operands of each proof before the solver joins their types and ranges.
pub(super) fn publish_proof_operands(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut definitions = Vec::new();
    {
        let structure = graph.structure().read();
        query.for_each("LiveRegion", |row| {
            let region = row[0];
            for (slot, &name) in structure.parameters.get(&region).into_iter().flatten().enumerate() {
                definitions.push(("ParameterSite", region, slot as i64, name, region));
            }
            Ok(())
        })?;
        query.for_each("LiveOperation", |row| {
            let operation = row[0];
            let Some(region) = structure.operations.get(&operation).and_then(|position| position.region)
            else {
                return Ok(());
            };
            for (slot, &name) in structure.outputs.get(&operation).into_iter().flatten().enumerate() {
                definitions.push(("OperationSite", operation, slot as i64, name, region));
            }
            Ok(())
        })?;
    }
    let mut reducers = Vec::new();
    query.for_each("LiveOperation", |row| {
        reducers.extend(graph.reducers(row[0]).into_iter().map(|(slot, body)| (row[0], slot, body)));
        Ok(())
    })?;
    let mut induction = Vec::new();
    query.for_each("CountedLoopOperation", |row| {
        let operation = row[0];
        let Some(region) = graph.control_at(operation, 0) else {
            return Ok(());
        };
        let position = graph.parameter_count(region) - 1;
        if let Some(parameter) = graph.parameter_at(region, position) {
            if let Some(index) = query.value_name(parameter)? {
                induction.push((operation, index));
            }
        }
        Ok(())
    })?;
    let mut branches = Vec::new();
    query.for_each("BranchOperation", |row| {
        let op = row[0];
        let Some(yes) = graph.control_at(op, 0) else {
            return Ok(());
        };
        let Some(no) = graph.control_at(op, 1) else {
            return Ok(());
        };
        let results = graph.outputs(op);
        let arms = graph.yield_values(yes).into_iter().zip(graph.yield_values(no));
        for (slot, (result, (yes, no))) in results.into_iter().zip(arms).enumerate() {
            branches.push((op, slot as i64, result, yes, no));
        }
        Ok(())
    })?;
    let mut direct_uses = Vec::new();
    let mut end_demands = Vec::new();
    let mut binary = Vec::new();
    let mut indices = Vec::new();
    let mut lengths = Vec::new();
    {
        let structure = graph.structure().read();
        for (&operation, inputs) in &structure.inputs.values {
            direct_uses.extend(inputs.iter().map(|&input| (operation, input)));
        }
        for (&region, yields) in &structure.yields.values {
            end_demands.extend(yields.iter().map(|&value| (region, value)));
        }
        query.for_function("ScalarPayload", |keys, _| {
            if let Some([left, right]) = structure.inputs.values.get(&keys[0]).map(Vec::as_slice) {
                binary.push((keys[0], *left, *right));
            }
            Ok(())
        })?;
        query.for_each("IndexOperation", |row| {
            if let Some([array, index]) = structure.inputs.values.get(&row[0]).map(Vec::as_slice) {
                if let (Some(array), Some(index)) = (query.value_name(*array)?, query.value_name(*index)?) {
                    indices.push((row[0], array, index));
                }
            }
            Ok(())
        })?;
        query.for_each("LengthOperation", |row| {
            let Some(array) = structure.inputs.values.get(&row[0]).and_then(|inputs| inputs.first()) else {
                return Ok(());
            };
            let Some(length) = query.0.output_at(row[0], 0_i64) else {
                return Ok(());
            };
            if let (Some(array), Some(length)) = (query.value_name(*array)?, query.value_name(length)?) {
                lengths.push((row[0], row[1], array, length));
            }
            Ok(())
        })?;
    }
    graph.update(|mut sink| {
        let mut sites = LookupMap::new();
        for (kind, key, slot, name, region) in definitions {
            let site = sink.add(kind, (key, slot))?;
            if sites.insert(name, site).is_some_and(|previous| site != previous) {
                sink.add("InvalidDataflow", "value has more than one definition")?;
            }
            sink.set("Defines", name, site)?;
            let origin = if kind == "ParameterSite" { "FormalOrigin" } else { "ResultOrigin" };
            sink.add(origin, (name, key, slot))?;
            sink.add("DefinitionRegion", (name, region))?;
            if kind == "ParameterSite" {
                sink.add("AvailableIn", (region, name))?;
            }
        }
        for site in direct_uses {
            sink.add("DirectOperationUse", site)?;
        }
        for site in end_demands {
            sink.add("EndDemand", site)?;
        }
        for reducer in reducers {
            sink.add("ReducerBody", reducer)?;
        }
        for index in induction {
            sink.add("LoopInduction", index)?;
        }
        for branch in branches {
            sink.add("BranchResult", branch)?;
        }
        for operands in binary {
            sink.add("BinaryOperands", operands)?;
        }
        for operands in indices {
            sink.add("IndexOperands", operands)?;
        }
        for length in lengths {
            sink.add("LengthMeasure", length)?;
        }
        Ok(())
    })?;
    formulas::publish(graph, false)?;
    publish_collective_uses(graph)
}

/// Publish the value sets needed by the scope, resource and execution proofs.
pub(super) fn publish_collective_uses(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut classes = Vec::new();
    let mut collective_uses = Vec::new();
    {
        let structure = graph.structure().read();
        for (&operation, results) in &structure.parallels {
            for (clause, &kind) in results.kinds.iter().enumerate() {
                let Some(kind) = kind else { continue };
                let (class, arguments) =
                    query.application(kind, &["Collected", "Compacted", "Contributed", "Folded"])?;
                classes.push((operation, clause as i64, class, arguments.first().copied()));
            }
        }
        for (&operation, groups) in &structure.collective.groups {
            let scope = structure.controls.get(&(operation, 0)).map(|edge| edge.region);
            for (&(group, kind), names) in groups {
                collective_uses.push((operation, scope, group, kind, names.clone()));
            }
        }
    }
    graph.update(|mut sink| {
        for (operation, clause, class, fold) in classes {
            match class {
                "Collected" => {
                    sink.add("CollectedClause", (operation, clause))?;
                }
                "Compacted" => {
                    sink.add("CompactedClause", (operation, clause))?;
                }
                "Contributed" => {
                    sink.add("ContributedClause", (operation, clause))?;
                }
                _ => {
                    if let Some(fold) = fold {
                        sink.add("FoldClause", (operation, clause, fold))?;
                    }
                }
            }
        }
        for (owner, scope, group, kind, names) in collective_uses {
            match kind {
                CollectiveInput::Contribution => {
                    for name in names {
                        sink.add("CollectiveLaneUse", (owner, group, name))?;
                        if let Some(scope) = scope {
                            sink.add("CollectiveScopeUse", (scope, name))?;
                        }
                    }
                }
                CollectiveInput::Predicate => {
                    if let Some(&predicate) = names.first() {
                        sink.add("PackedClause", (owner, group))?;
                        if let Some(scope) = scope {
                            let name = sink.add("ValueNameRef", predicate)?;
                            sink.add("CollectiveScopeUse", (scope, name))?;
                        }
                    }
                }
                CollectiveInput::Neutral => {
                    for name in names {
                        sink.add("FoldSeedUse", (owner, name))?;
                    }
                }
                CollectiveInput::Backing => {
                    if let [array] = names.as_slice() {
                        sink.add("SingleAccumulatorBacking", (owner, group, *array))?;
                    }
                    for (slot, name) in names.into_iter().enumerate() {
                        sink.add("AccumulatorBackingUse", (owner, group, slot as i64, name))?;
                    }
                }
            }
        }
        Ok(())
    })?;
    Ok(())
}

pub(super) fn run(graph: &mut Graph) -> Result<(), OptimizeError> {
    let _timing = timing::span("egglog graph analysis");
    structure(graph, false)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule
          (saturate graph-indices dataflow-scopes flow-scopes)
          (saturate graph-indices dataflow-dependencies dataflow-scopes flow-scopes)
          (saturate flow-forward dataflow-dependencies dataflow-scopes)
          (saturate flow-projections source-summary))",
    )?;
    graph.parse_and_run_program(
        None,
        "(run-schedule
        (saturate dataflow-resources)
        (saturate dataflow-total integer-ranges)
        (saturate dataflow-effects)
        (saturate dataflow-motion dataflow-purity))",
    )?;
    publish_order(graph)?;
    validate(graph)?;
    let revision = edit::revision(graph)?;
    graph.update(|mut sink| sink.set("DataflowRevision", RawValues(Vec::new()), revision))?;
    Ok(())
}

/// Reuse body-dependent proofs until the canonical graph is edited.
pub(super) fn ensure(graph: &mut Graph) -> Result<(), OptimizeError> {
    let revision = edit::revision(graph)?;
    if Query(graph).lookup("DataflowRevision", RawValues(Vec::new()))?
        == Some(graph.base_to_value(revision))
    {
        return Ok(());
    }
    run(graph)
}

fn publish_order(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut operations = StableMap::<Value, Vec<(i64, Value)>>::default();
    query.for_each("LiveRegion", |row| {
        operations.entry(row[0]).or_default();
        Ok(())
    })?;
    query.for_each("LiveOperation", |row| {
        let operation = row[0];
        let region = query.0.operation_region(operation)?;
        let order = query.0.operation_order(operation)?;
        operations.entry(region).or_default().push((order, operation));
        Ok(())
    })?;
    // Region order depends only on the occurrences in that region.
    let mut previous = graph.region_orders.clone();
    let mut removed = Vec::new();
    let mut updated = Vec::new();
    for (region, mut body) in operations {
        body.sort_by_key(|&(order, _)| order);
        let body = body.into_iter().map(|(_, operation)| operation).collect::<Vec<_>>();
        let old = previous.swap_remove(&region);
        if old.as_ref() == Some(&body) {
            continue;
        }
        if let Some(old) = old {
            removed.push((region, old));
        }
        updated.push((region, body));
    }
    removed.extend(previous);
    replace_order(graph, removed, updated)
}

pub(super) fn replace_order(
    graph: &mut Graph,
    removed: Vec<(Value, Vec<Value>)>,
    updated: Vec<(Value, Vec<Value>)>,
) -> Result<(), OptimizeError> {
    if removed.is_empty() && updated.is_empty() {
        return Ok(());
    }
    graph.update(|mut sink| {
        for (region, body) in &removed {
            let region = *region;
            sink.remove("RegionOperationCount", region)?;
            if let (Some(&first), Some(&last)) = (body.first(), body.last()) {
                sink.remove("FirstOperation", (region, first))?;
                sink.remove("LastOperation", (region, last))?;
            } else {
                sink.remove("EmptyRegion", region)?;
            }
            for pair in body.windows(2) {
                sink.remove("NextOperation", (pair[0], pair[1]))?;
            }
        }
        Ok(())
    })?;
    graph.update(|mut sink| {
        for (region, body) in &updated {
            let region = *region;
            sink.set("RegionOperationCount", region, body.len() as i64)?;
            if let (Some(&first), Some(&last)) = (body.first(), body.last()) {
                sink.add("FirstOperation", (region, first))?;
                sink.add("LastOperation", (region, last))?;
            } else {
                sink.add("EmptyRegion", region)?;
            }
            for pair in body.windows(2) {
                sink.add("NextOperation", (pair[0], pair[1]))?;
            }
        }
        Ok(())
    })?;
    for (region, _) in removed {
        graph.region_orders.swap_remove(&region);
    }
    graph.region_orders.extend(updated);
    Ok(())
}

/// Test whether an operation uses any of the producer's results.
pub(super) fn uses_output_of(
    query: Query<'_>,
    operation: Value,
    producer: Value,
) -> Result<bool, OptimizeError> {
    for slot in 0..query.0.output_count(producer) {
        let output = query.0.output(producer, slot)?;
        if query.contains("OperationUse", (operation, output))? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Test a prospective use against the current lexical order.
pub(super) fn available_before(
    query: Query<'_>,
    mut operation: Value,
    name: Value,
) -> Result<bool, OptimizeError> {
    let Some(site) = query.lookup("Defines", name)? else {
        return Ok(false);
    };
    let (kind, fields) = query.application(site, &["OperationSite", "ParameterSite"])?;
    let producer = (kind == "OperationSite").then_some(fields[0]);
    let definition =
        if let Some(producer) = producer { query.0.operation_region(producer)? } else { fields[0] };
    loop {
        let region = query.0.operation_region(operation)?;
        if region == definition {
            return if let Some(producer) = producer {
                Ok(query.0.operation_order(producer)? < query.0.operation_order(operation)?)
            } else {
                Ok(true)
            };
        }
        let Some(owner) = query.lookup("RegionOwnedBy", region)? else {
            return Ok(false);
        };
        operation = owner;
    }
}

/// Compute live-after values for the requested function roots.
pub(super) fn liveness(graph: &mut Graph, roots: &[Value]) -> Result<(), OptimizeError> {
    ensure(graph)?;
    let query = Query(graph);
    let mut pending = Vec::new();
    for &root in roots {
        if !query.contains("LivenessRequest", root)? {
            pending.push(root);
        }
    }
    if pending.is_empty() {
        return Ok(());
    }
    let _timing = timing::span("egglog execution liveness");
    graph.update(|mut sink| {
        for root in pending {
            sink.add("LivenessRequest", root)?;
        }
        Ok(())
    })?;
    graph.parse_and_run_program(None, "(run-schedule (saturate dataflow-live-index))")?;
    let query = Query(graph);
    let mut next = 0_i64;
    query.for_function("NameOrdinal", |_, ordinal| {
        next = next.max(graph.value_to_base::<i64>(ordinal) + 1);
        Ok(())
    })?;
    let mut names = Vec::new();
    query.for_each("LivenessName", |row| {
        if query.lookup("NameOrdinal", row[0])?.is_none() {
            names.push(row[0]);
        }
        Ok(())
    })?;
    graph.update(|mut sink| {
        for name in names {
            sink.set("NameOrdinal", name, next)?;
            next += 1;
        }
        Ok(())
    })?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate dataflow-live-seeds) (saturate flow-live dataflow-live) (saturate dataflow-resources))",
    )?;
    Ok(())
}

/// Check the selected graph, including captures and simultaneous loop edges.
pub(super) fn validate(graph: &Graph) -> Result<(), OptimizeError> {
    if !cfg!(debug_assertions) {
        return Ok(());
    }
    let query = Query(graph);
    query.for_each("InvalidDataflow", |row| {
        Err(output_error(graph.value_to_base::<S>(row[0]).to_string()))
    })?;
    query.for_each("LiveFunction", |row| signatures::function(query, row[0]))?;
    query.for_each("LiveOperation", |row| {
        let operation = row[0];
        graph.code(operation)?;
        query.0.operation_region(operation)?;
        query.0.operation_order(operation)?;
        for value in graph.inputs(operation).into_iter().chain(query.0.outputs(operation)) {
            typed(query, value)?;
        }
        if query.contains("CallOperation", operation)? {
            let target = query.0.call_target(operation)?;
            let expected = if query.contains("External", target)? {
                count(query, "ExternalParameterCount", target)?
            } else {
                query.0.parameter_count(query.0.function_root(target)?)
            };
            if query.0.input_count(operation) != expected {
                return Err(output_error("call argument count differs from its signature"));
            }
        }
        signatures::operation(query, operation)?;
        if query.contains("CountedLoopOperation", operation)?
            || query.contains("WhileLoopOperation", operation)?
        {
            let body = query.0.control_region(operation, 0_i64)?;
            let results = query.0.output_count(operation);
            let induction = i64::from(query.contains("CountedLoopOperation", operation)?);
            if query.0.parameter_count(body) != results + induction || query.0.yield_count(body) != results
            {
                return Err(output_error("loop carry signature is inconsistent"));
            }
            for slot in 0..results {
                let initial = query.0.argument(operation, 0_i64, slot)?;
                let parameter = query.0.parameter(body, slot)?;
                let backedge = query.0.yield_value(body, slot)?;
                let result = query.0.output(operation, slot)?;
                let ty = typed(query, parameter)?;
                for name in [initial, backedge, result] {
                    if !signatures::compatible(query, typed(query, name)?, ty)? {
                        return Err(output_error("loop carry types differ across an edge"));
                    }
                }
            }
        }
        Ok(())
    })?;
    query.for_each("OperationUse", |row| {
        if query.contains("LiveOperation", row[0])? && !query.contains("AvailableAt", (row[0], row[1]))? {
            return Err(output_error(format!(
                "operand {:?} is unavailable at {:?} ({:?}) in {:?}",
                row[1],
                row[0],
                query.0.code(row[0])?,
                query.0.operation_region(row[0])?
            )));
        }
        Ok(())
    })?;
    query.for_each("LiveRegion", |row| {
        let region = row[0];
        signatures::region(query, region)?;
        for slot in 0..query.0.parameter_count(region) {
            typed(query, query.0.parameter(region, slot)?)?;
            query.required("ParameterDiet", (region, slot))?;
        }
        for slot in 0..query.0.yield_count(region) {
            let name = query.0.yield_value(region, slot)?;
            typed(query, name)?;
            if !query.contains("AvailableEnd", (region, name))? {
                return Err(output_error("region yields a value outside its execution scope"));
            }
        }
        Ok(())
    })?;
    query.for_each("EndDemand", |row| {
        if query.contains("LiveRegion", row[0])? && !query.contains("AvailableEnd", (row[0], row[1]))? {
            return Err(output_error(
                "collective result is unavailable in its lane region",
            ));
        }
        Ok(())
    })
}

pub(super) fn count(query: Query<'_>, table: &str, owner: impl IntoValues) -> Result<i64, OptimizeError> {
    let count = query.0.value_to_base::<i64>(query.required(table, owner)?);
    if count < 0 {
        return Err(output_error(format!("negative {table}")));
    }
    Ok(count)
}

pub(super) fn typed(query: Query<'_>, name: Value) -> Result<Value, OptimizeError> {
    let (kind, fields) = query.application(name, &["ValueNameRef", "AccNameRef"])?;
    let info = query.lookup(
        if kind == "ValueNameRef" { "ValueType" } else { "AccumulatorType" },
        fields[0],
    )?;
    let Some(info) = info else {
        return Err(output_error(
            "operand outside its scope or missing a declared type",
        ));
    };
    Ok(info)
}

#[cfg(test)]
#[path = "analysis_tests.rs"]
mod tests;
