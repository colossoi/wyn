//! Derived facts are disposable; executable edges remain in the flat graph.
use super::dataflow::edit;
use super::query::Query;
use super::{output_error, timing, OptimizeError};
use crate::egglog::graph::{callback_at, CollectiveInput, Graph, Origin};
use crate::LookupSet;
use egglog_engine::sort::S;
use egglog_engine::{IntoValues, RawValues, Value, Write};

pub(super) mod calls;
mod formulas;
mod liveness;
mod projections;

pub(super) use liveness::{liveness, Liveness};

pub(in crate::egglog) use formulas::{formula_names, seed_substitution};
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
        "ConstantValue",
        "RecursiveFunction",
        "ResultOrigin",
        "FormalOrigin",
        "ReducerBody",
        "CollectiveScopeUse",
        "CollectiveLaneUse",
        "FoldSeedUse",
        "PackedClause",
        "IterationAxis",
        "OperationGuardRequirement",
        "YieldGuardRequirement",
        "LaneAxisExtent",
        "ReductionBoundOperands",
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
        "EndDemand",
        "BranchResult",
        "BinaryOperands",
        "ProjectionAlias",
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
pub(super) fn structure(graph: &mut Graph) -> Result<(), OptimizeError> {
    let _timing = timing::span("egglog graph structure");
    invalidate(graph)?;
    calls::publish(graph)?;
    publish_proof_operands(graph)?;
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
                definitions.push((name, Origin::Parameter(region, slot), region));
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
                definitions.push((name, Origin::Result(operation, slot), region));
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
    let mut operation_guards = Vec::new();
    let mut yield_guards = Vec::new();
    let mut end_demands = Vec::new();
    let mut binary = Vec::new();
    let mut indices = Vec::new();
    let mut lengths = Vec::new();
    {
        let structure = graph.structure().read();
        for (&operation, names) in &structure.certificates.operations {
            let parallel = matches!(
                graph.code(operation)?,
                crate::egglog::dataflow::ir::Code::Parallel
            );
            operation_guards.extend(names.iter().map(|&name| (operation, name, parallel)));
        }
        for (&region, values) in &structure.certificates.yields {
            yield_guards.extend(values.iter().map(|&(_, value)| (region, value)));
        }
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
        let mut defined = LookupSet::new();
        for (name, origin, region) in definitions {
            if !defined.insert(name) {
                sink.add("InvalidDataflow", "value has more than one definition")?;
            }
            sink.add("DefinitionRegion", (name, region))?;
            match origin {
                Origin::Result(operation, slot) => {
                    sink.add("ResultOrigin", (name, operation, slot as i64))?;
                }
                Origin::Parameter(region, slot) => {
                    sink.add("FormalOrigin", (name, region, slot as i64))?;
                }
            }
        }
        for (operation, certificate, parallel) in operation_guards {
            let name = sink.add("ValueNameRef", certificate)?;
            sink.add("DirectOperationUse", (operation, name))?;
            if parallel {
                sink.add("OperationGuardRequirement", (operation, certificate))?;
            }
        }
        for (region, certificate) in yield_guards {
            let name = sink.add("ValueNameRef", certificate)?;
            sink.add("EndDemand", (region, name))?;
            sink.add("RegionAttributeUse", (region, name))?;
            sink.add("YieldGuardRequirement", (region, certificate))?;
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
    let mut reduction_bounds = Vec::new();
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
        for (&operation, results) in &structure.parallels {
            let domain = structure
                .domains
                .operations
                .get(&operation)
                .and_then(|id| structure.domains.values.get(id));
            let Some([axis]) = domain.map(|domain| domain.dimensions.as_slice()) else {
                continue;
            };
            let Some(extent) = axis.extent else { continue };
            for (&output, binding) in
                structure.outputs.get(&operation).into_iter().flatten().zip(&results.bindings)
            {
                let (Some(clause), Some(role)) = (binding.clause, binding.role) else {
                    continue;
                };
                if query.enode("TotalOutput", role)?.is_none() {
                    continue;
                }
                let Some(reducer) = callback_at(&structure.reducers, operation, clause) else {
                    continue;
                };
                let [neutral] = structure.collective.get(operation, clause, CollectiveInput::Neutral)
                else {
                    continue;
                };
                let Some(&value) =
                    structure.collective.get(operation, clause, CollectiveInput::Contribution).first()
                else {
                    continue;
                };
                reduction_bounds.push((output, reducer, extent, *neutral, value));
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
        for bound in reduction_bounds {
            sink.add("ReductionBoundOperands", bound)?;
        }
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
    structure(graph)?;
    projections::publish(graph)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate graph-indices dataflow-dependencies))",
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

pub(super) fn current(graph: &Graph) -> Result<bool, OptimizeError> {
    let revision = edit::revision(graph)?;
    Ok(Query(graph).lookup("DataflowRevision", RawValues(Vec::new()))?
        == Some(graph.base_to_value(revision)))
}

/// Reuse body-dependent proofs until the executable graph is edited.
pub(super) fn ensure(graph: &mut Graph) -> Result<(), OptimizeError> {
    if current(graph)? {
        return Ok(());
    }
    run(graph)
}

fn publish_order(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut live = LookupSet::new();
    let mut removed = Vec::new();
    let mut updated = Vec::new();
    query.for_each("LiveRegion", |row| {
        let region = row[0];
        live.insert(region);
        let old = graph.region_orders.get(&region);
        if old.is_some() && !graph.dirty_orders.contains(&region) {
            return Ok(());
        }
        let current = query.operations(region)?;
        if old != Some(&current) {
            if let Some(old) = old {
                removed.push((region, old.clone()));
            }
            updated.push((region, current));
        }
        Ok(())
    })?;
    for (&region, body) in &graph.region_orders {
        if !live.contains(&region) {
            removed.push((region, body.clone()));
        }
    }
    replace_order(graph, removed, updated)?;
    graph.dirty_orders.clear();
    Ok(())
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
        graph.dirty_orders.remove(&region);
    }
    for (region, body) in updated {
        graph.region_orders.insert(region, body);
        graph.dirty_orders.remove(&region);
    }
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
    let Some(origin) = query.live_origin(name)? else {
        return Ok(false);
    };
    let (definition, producer) = match origin {
        Origin::Result(operation, _) => (query.0.operation_region(operation)?, Some(operation)),
        Origin::Parameter(region, _) => (region, None),
    };
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

/// A local formal is available at both boundaries. A local result is
/// available at exit. A captured definition must dominate the owning operation.
fn available_at_boundary(
    query: Query<'_>,
    region: Value,
    name: Value,
    exit: bool,
) -> Result<bool, OptimizeError> {
    let Some(origin) = query.live_origin(name)? else {
        return Ok(false);
    };
    match origin {
        Origin::Parameter(owner, _) if owner == region => return Ok(true),
        Origin::Result(operation, _) if query.0.operation_region(operation)? == region => {
            return Ok(exit);
        }
        _ => {}
    }
    let Some(owner) = query.lookup("RegionOwnedBy", region)? else {
        return Ok(false);
    };
    available_before(query, owner, name)
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
        if query.contains("LiveOperation", row[0])? && !available_before(query, row[0], row[1])? {
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
            if !available_at_boundary(query, region, name, true)? {
                return Err(output_error("region yields a value outside its execution scope"));
            }
        }
        Ok(())
    })?;
    query.for_each("EndDemand", |row| {
        if query.contains("LiveRegion", row[0])? && !available_at_boundary(query, row[0], row[1], true)? {
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
