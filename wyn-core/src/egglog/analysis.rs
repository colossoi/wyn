//! Derived facts are disposable; executable edges remain in the flat graph.
use super::dataflow::{edit, ir};
use super::query::Query;
use super::{output_error, timing, OptimizeError};
use crate::{LookupMap, StableMap};
use egglog_engine::sort::VecContainer;
use egglog_engine::sort::S;
use egglog_engine::{Core, EGraph, IntoValues, RawValues, Value, Write};

mod signatures;

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
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
pub(super) fn invalidate(graph: &mut EGraph) -> Result<(), OptimizeError> {
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
        "CallReach",
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
        "RegionCapture",
        "DirectOperationUse",
        "OperationUse",
        "OperationDependency",
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
        "LoopInitial",
        "LoopBackedge",
        "AccumulatorParameter",
        "FlowScopeParent",
        "FlowScopeAncestor",
        "FlowScopeContains",
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
pub(super) fn advance(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let revision = count(Query(graph), "AnalysisRevision", RawValues(Vec::new()))?;
    graph.update(|mut sink| {
        sink.remove("AnalysisRevision", RawValues(Vec::new()))?;
        sink.set("AnalysisRevision", RawValues(Vec::new()), revision + 1)
    })?;
    Ok(())
}

/// Refresh lexical ownership, reachable calls, and recursion without solving dataflow.
pub(super) fn structure(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = timing::span("egglog graph structure");
    invalidate(graph)?;
    graph.parse_and_run_program(None, "(run-schedule (saturate graph-structure flow-scopes))")?;
    Ok(())
}

pub(super) fn run(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = timing::span("egglog graph analysis");
    structure(graph)?;
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
pub(super) fn ensure(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let revision = edit::revision(graph)?;
    if Query(graph).lookup("DataflowRevision", RawValues(Vec::new()))?
        == Some(graph.base_to_value(revision))
    {
        return Ok(());
    }
    run(graph)
}

fn publish_order(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut operations = StableMap::<Value, Vec<(i64, Value)>>::default();
    query.for_each("LiveRegion", |row| {
        operations.entry(row[0]).or_default();
        Ok(())
    })?;
    query.for_each("LiveOperation", |row| {
        let operation = row[0];
        let region = query.required("OperationRegion", operation)?;
        let order = query.required("OperationOrder", operation)?;
        operations.entry(region).or_default().push((graph.value_to_base(order), operation));
        Ok(())
    })?;
    // Region order depends only on the occurrences in that region.
    let mut previous = LookupMap::default();
    query.for_function("RegionOperations", |keys, old| {
        previous.insert(keys[0], query.vector(old)?);
        Ok(())
    })?;
    let mut removed = Vec::new();
    let mut updated = Vec::new();
    for (region, mut body) in operations {
        body.sort_by_key(|&(order, _)| order);
        let body = body.into_iter().map(|(_, operation)| operation).collect::<Vec<_>>();
        let old = previous.remove(&region);
        if old.as_ref() == Some(&body) {
            continue;
        }
        if let Some(old) = old {
            removed.push((region, old));
        }
        updated.push((region, body));
    }
    removed.extend(previous);
    if removed.is_empty() && updated.is_empty() {
        return Ok(());
    }
    graph.update(|mut sink| {
        for (region, body) in removed {
            sink.remove("RegionOperations", region)?;
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
        for (region, body) in updated {
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
            let value = sink.container_to_value(VecContainer {
                data: body,
                do_rebuild: true,
            });
            sink.set("RegionOperations", region, value)?;
        }
        Ok(())
    })?;
    Ok(())
}

/// Compute live-after values for the requested function roots.
pub(super) fn liveness(graph: &mut EGraph, roots: &[Value]) -> Result<(), OptimizeError> {
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
pub(super) fn validate(graph: &EGraph) -> Result<(), OptimizeError> {
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
        query.required("OperationPayload", operation)?;
        query.required("OperationRegion", operation)?;
        query.required("OperationOrder", operation)?;
        for table in ["Input", "Output"] {
            let count = count(query, &format!("{table}Count"), operation)?;
            for slot in 0..count {
                typed(query, query.required(table, (operation, slot))?)?;
            }
        }
        if query.contains("CallOperation", operation)? {
            let target = query.required("CallTarget", operation)?;
            let expected = if query.contains("External", target)? {
                count(query, "ExternalParameterCount", target)?
            } else {
                count(
                    query,
                    "RegionParameterCount",
                    query.required("FunctionRoot", target)?,
                )?
            };
            if count(query, "InputCount", operation)? != expected {
                return Err(output_error("call argument count differs from its signature"));
            }
        }
        signatures::operation(query, operation)?;
        if query.contains("CountedLoopOperation", operation)?
            || query.contains("WhileLoopOperation", operation)?
        {
            let body = query.required("ControlRegion", (operation, 0_i64))?;
            let results = count(query, "OutputCount", operation)?;
            let induction = i64::from(query.contains("CountedLoopOperation", operation)?);
            if count(query, "RegionParameterCount", body)? != results + induction
                || count(query, "RegionYieldCount", body)? != results
            {
                return Err(output_error("loop carry signature is inconsistent"));
            }
            for slot in 0..results {
                let initial = query.required("EdgeArgument", (operation, 0_i64, slot))?;
                let parameter = query.required("RegionParameter", (body, slot))?;
                let backedge = query.required("RegionYield", (body, slot))?;
                let result = query.required("Output", (operation, slot))?;
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
                ir::code(graph, query.required("OperationPayload", row[0])?),
                query.required("OperationRegion", row[0])?
            )));
        }
        Ok(())
    })?;
    query.for_each("LiveRegion", |row| {
        let region = row[0];
        signatures::region(query, region)?;
        for slot in 0..count(query, "RegionParameterCount", region)? {
            typed(query, query.required("RegionParameter", (region, slot))?)?;
            query.required("ParameterDiet", (region, slot))?;
        }
        for slot in 0..count(query, "RegionYieldCount", region)? {
            let name = query.required("RegionYield", (region, slot))?;
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

fn typed(query: Query<'_>, name: Value) -> Result<Value, OptimizeError> {
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
