//! Derived facts are disposable; executable edges remain in the flat graph.
use super::dataflow::ir;
use super::query::Query;
use super::{output_error, OptimizeError};
use egglog_engine::sort::S;
use egglog_engine::{EGraph, IntoValues, RawValues, Value, Write};

mod live;
mod signatures;

pub(super) fn load(graph: &mut EGraph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(
        Some("dataflow analysis".into()),
        concat!(
            include_str!("analysis/schema.egg"),
            "\n",
            include_str!("analysis/dataflow.egg"),
            "\n",
            include_str!("analysis/resources.egg")
        ),
    )?;
    Ok(())
}

/// Clear proofs before editing edges. No irreversible value unions are used here.
pub(super) fn invalidate(graph: &mut EGraph) -> Result<(), OptimizeError> {
    for table in [
        "NextOperation",
        "CallReach",
        "RecursiveFunction",
        "NameOrdinal",
        "NameLiveAfter",
        "ResourceLiveAfter",
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
        "IndexUse",
        "ShapeDemand",
        "ShapeWalk",
        "ShapeAt",
        "ShapeUse",
        "SliceDemand",
        "SliceWalk",
        "SliceAt",
        "SliceUse",
        "TotalIndex",
        "ArrayOuterBound",
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

pub(super) fn run(graph: &mut EGraph) -> Result<(), OptimizeError> {
    let _timing = super::timing::span("egglog graph analysis");
    invalidate(graph)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule
          (saturate dataflow-scopes flow-scopes)
          (saturate dataflow-dependencies dataflow-scopes flow-scopes)
          (saturate flow-forward dataflow-dependencies dataflow-scopes)
          (saturate flow-projections source-summary))",
    )?;
    live::publish(graph)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule
        (saturate flow-live dataflow-live)
        (saturate dataflow-resources)
        (saturate dataflow-total)
        (saturate dataflow-effects dataflow-live)
        (saturate dataflow-motion dataflow-purity))",
    )?;
    let mut edges = Vec::new();
    Query(graph).for_each("LiveRegion", |row| {
        for pair in Query(graph).operations(row[0])?.windows(2) {
            edges.push((pair[0], pair[1]));
        }
        Ok(())
    })?;
    graph.update(|mut sink| {
        for edge in edges {
            sink.add("NextOperation", edge)?;
        }
        Ok(())
    })?;
    validate(graph)
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
