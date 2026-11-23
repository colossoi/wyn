//! Delete unused scalar leaves without rebuilding proofs about surviving values.
use super::{analysis, body::BodyGraph, demand::Needed, edit, scalar, timing, OptimizeError, Query};
use crate::egglog::dataflow::ir::Code;
use crate::egglog::graph::Graph;
use crate::egglog::planning;
use crate::memory_ir::{Definition, OperationId, ValueRef};
use crate::LookupSet;
use egglog_engine::Write;

pub(super) fn apply(
    graph: &mut Graph,
    body: &mut BodyGraph,
    needed: &Needed,
    dead: &[OperationId],
) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let keys = dead.iter().map(|&op| body.operations[op]).collect::<Vec<_>>();
    for &op in dead {
        let operation = &body.ir.operations[op];
        let key = body.operations[op];
        if !matches!(
            operation.definition,
            Definition::Other(Code::Scalar(_) | Code::Constant { .. })
        ) || operation.result_types.len() != 1
            || query.lookup("ValuePrimitive", body.names[&ValueRef::Result(op, 0)])?.is_none()
            || !query.flag("PureOperation", key)?
            || query.contains("LaneDependentOperation", key)?
        {
            return Ok(false);
        }
    }

    let _timing = timing::span("Rust unused leaf publication");
    let removed: LookupSet<_> = dead.iter().copied().collect();
    let mut previous = Vec::new();
    for (region, block) in &body.ir.regions {
        if block.operations.iter().any(|op| removed.contains(op)) {
            previous.push((
                body.regions[region],
                block.operations.iter().map(|&op| body.operations[op]).collect(),
            ));
        }
    }
    let work = planning::invalidate_region_work(graph, &previous)?;
    let mut updated = Vec::new();
    for (region, block) in &mut body.ir.regions {
        let count = block.operations.len();
        block.operations.retain(|op| !removed.contains(op));
        if block.operations.len() != count {
            updated.push((
                body.regions[region],
                block.operations.iter().map(|&op| body.operations[op]).collect(),
            ));
        }
    }
    // Dead leaves can use one another. Keep all definitions until their last
    // lexical use has been retracted.
    for &op in dead {
        for value in needed.use_sites(op) {
            scalar::remove_use(graph, body.operations[op], body.names[&value])?;
        }
    }
    for &op in dead {
        let operation = body.operations[op];
        let region = Query(graph).required("OperationRegion", operation)?;
        let output = body.names[&ValueRef::Result(op, 0)];
        let names = needed.use_sites(op).map(|value| body.names[&value]);
        graph.update(|mut sink| {
            for name in names {
                sink.remove("DirectOperationUse", (operation, name))?;
                sink.remove("OperationAttributeUse", (operation, name))?;
            }
            for table in [
                "LiveOperation",
                "OperationTotal",
                "PureOperation",
                "MovableOperation",
                "RegularLength",
                "OperationWork",
                "OperationWorkDemand",
            ] {
                sink.remove(table, operation)?;
            }
            for table in [
                "Defines",
                "ConstantValue",
                "IntegerLower",
                "IntegerUpper",
                "TrackedI32",
                "IntegerBoundsDemand",
            ] {
                sink.remove(table, output)?;
            }
            sink.remove("DefinitionRegion", (output, region))
        })?;
    }
    for &op in dead {
        let result = ValueRef::Result(op, 0);
        body.operation_ids.remove(&body.operations[op]);
        body.refs.remove(&body.names[&result]);
        body.operations.remove(op);
        body.names.remove(&result);
        body.ir.operations.remove(op);
    }
    edit::remove_without_revision(graph, &keys)?;
    analysis::replace_order(graph, previous, updated)?;
    planning::restore_region_work(graph, work)?;
    edit::preserve_revisions(graph)?;
    Ok(true)
}
