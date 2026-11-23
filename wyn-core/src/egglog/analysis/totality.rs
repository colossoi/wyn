//! Prove literal shift bounds from the definitions of the two actual operands.
use super::{Graph, OptimizeError, Origin, Query};
use crate::egglog::dataflow::ir::Code;
use crate::egglog::payload::Payload;
use crate::op::BinaryOperator;
use crate::{LookupMap, LookupSet};
use egglog_engine::{RawValues, Value, Write};

pub(super) fn publish_literal_shifts(graph: &mut Graph, operations: &[Value]) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut supported = Vec::new();
    for kind in ["I32", "U32"] {
        let Some(primitive) = query.constructor(kind, RawValues(Vec::new()))? else {
            continue;
        };
        let Some(element) = query.constructor("ScalarElement", primitive)? else {
            continue;
        };
        supported.extend(query.constructor("ScalarInfo", element)?);
    }
    let mut total = Vec::new();
    for &operation in operations {
        if !matches!(
            graph.code(operation)?,
            Code::Scalar(Payload::Binary(
                BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight
            ))
        ) || !query.contains("LiveOperation", operation)?
        {
            continue;
        }
        let inputs = graph.inputs(operation);
        let [value, amount] = inputs.as_slice() else {
            continue;
        };
        let (Some(value_name), Some(amount_name)) = (query.value_name(*value)?, query.value_name(*amount)?)
        else {
            continue;
        };
        let Some(ty) = query.lookup("ValueType", value_name)? else {
            continue;
        };
        if !supported.contains(&ty) || query.lookup("ValueType", amount_name)? != Some(ty) {
            continue;
        }
        let Some(Origin::Result(literal, 0)) = query.live_origin(*amount)? else {
            continue;
        };
        if let Some(bits) = query.lookup("LiteralBits", literal)? {
            if (0..32).contains(&graph.value_to_base::<i64>(bits)) {
                total.push(operation);
            }
        }
    }
    graph.update(|mut sink| {
        for operation in total {
            sink.set("OperationTotal", operation, true)?;
        }
        Ok(())
    })?;
    Ok(())
}

/// Strengthen bound proofs after a leaf edit, then retract only the traps that
/// lost their last source through control ownership or a retained call.
pub(in crate::egglog) fn refine(graph: &mut Graph) -> Result<(), OptimizeError> {
    if graph.get_size("OperationTrap") == 0 {
        return Ok(());
    }
    let operations: Vec<_> = graph
        .codes()
        .filter(|(_, code)| {
            matches!(
                code,
                Code::Scalar(Payload::Binary(
                    BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight
                ))
            )
        })
        .map(|(op, _)| op)
        .collect();
    publish_literal_shifts(graph, &operations)?;
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate graph-indices dataflow-dependencies) (saturate dataflow-total integer-ranges))",
    )?;
    let query = Query(graph);
    let mut pending = Vec::new();
    query.for_each("OperationTrap", |row| {
        let op = row[0];
        if query.contains("LiveOperation", op)?
            && query.lookup("OperationTotal", op)? == Some(graph.base_to_value(true))
        {
            pending.push(op);
        }
        Ok(())
    })?;
    if pending.is_empty() {
        return Ok(());
    }
    let mut callers = LookupMap::<Value, Vec<Value>>::new();
    query.for_each("CallBody", |row| {
        callers.entry(row[1]).or_default().push(row[0]);
        Ok(())
    })?;
    let mut operations = LookupSet::new();
    let mut regions = LookupSet::new();
    while let Some(op) = pending.pop() {
        if operations.contains(&op)
            || !query.contains("OperationTrap", op)?
            || still_traps(query, op, &regions)?
        {
            continue;
        }
        operations.insert(op);
        let region = graph.operation_region(op)?;
        let mut remaining = false;
        for &member in graph.region_order(region)? {
            if !operations.contains(&member) && query.contains("OperationTrap", member)? {
                remaining = true;
                break;
            }
        }
        if !remaining && regions.insert(region) {
            pending.extend(graph.region_owner(region));
            pending.extend(callers.get(&region).into_iter().flatten().copied());
        }
    }
    let reads = effect_sites(query, &operations, "OperationRead")?;
    let writes = effect_sites(query, &operations, "OperationWrite")?;
    let consumed = effect_sites(query, &operations, "ConsumedToken")?;
    let mut changed = Vec::new();
    for &op in &operations {
        changed.push((
            op,
            !reads.contains(&op) && !writes.contains(&op),
            !writes.contains(&op)
                && !consumed.contains(&op)
                && !query.contains("MaterializedCopyOperation", op)?,
        ));
    }
    let mut conflicts = Vec::new();
    query.for_each("OperationsConflict", |row| {
        if operations.contains(&row[0]) || operations.contains(&row[1]) {
            conflicts.push((row[0], row[1]));
        }
        Ok(())
    })?;
    graph.update(|mut sink| {
        for (op, pure, movable) in changed {
            sink.remove("OperationTrap", op)?;
            sink.remove("PureOperation", op)?;
            sink.remove("MovableOperation", op)?;
            sink.set("PureOperation", op, pure)?;
            sink.set("MovableOperation", op, movable)?;
        }
        for region in regions {
            sink.remove("RegionTrap", region)?;
        }
        for &pair in &conflicts {
            sink.remove("OperationsConflict", pair)?;
            sink.remove("EffectCompare", pair)?;
        }
        Ok(())
    })?;
    if !conflicts.is_empty() {
        graph.update(|mut sink| {
            for pair in conflicts {
                sink.add("EffectCompare", pair)?;
            }
            Ok(())
        })?;
        graph.parse_and_run_program(None, "(run-schedule (saturate dataflow-effects))")?;
    }
    super::scopes::repair_transforms(graph, operations)
}

fn still_traps(
    query: Query<'_>,
    operation: Value,
    cleared: &LookupSet<Value>,
) -> Result<bool, OptimizeError> {
    let graph = query.0;
    if query.contains("OrderedOperation", operation)?
        && query.lookup("OperationTotal", operation)? != Some(graph.base_to_value(true))
    {
        return Ok(true);
    }
    let mut bodies = graph.attached_regions(operation);
    if graph.code(operation)? == Code::Call {
        let target = graph.call_target(operation)?;
        if graph.is_external(target)
            || query.lookup("RecursiveFunction", target)? == Some(graph.base_to_value(true))
        {
            return Ok(true);
        }
        bodies.extend(graph.root_of(target));
    }
    for body in bodies {
        if !cleared.contains(&body) && query.contains("RegionTrap", body)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn effect_sites(
    query: Query<'_>,
    changed: &LookupSet<Value>,
    table: &str,
) -> Result<LookupSet<Value>, OptimizeError> {
    let mut sites = LookupSet::new();
    query.for_each(table, |row| {
        if changed.contains(&row[0]) {
            sites.insert(row[0]);
        }
        Ok(())
    })?;
    Ok(sites)
}
