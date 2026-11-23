//! Delete unused scalar leaves without rebuilding proofs about surviving values.
use super::{analysis, edit, scalar, timing, OptimizeError, Query};
use crate::egglog::dataflow::ir::Code;
use crate::egglog::graph::Graph;
use crate::egglog::planning;
use crate::LookupSet;
use egglog_engine::{Value, Write};

pub(super) fn apply(
    graph: &mut Graph,
    dead: &[Value],
    use_sites: impl Fn(Query<'_>, Value) -> Result<Vec<Value>, OptimizeError>,
) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let mut regions = LookupSet::new();
    let mut uses = Vec::new();
    for &key in dead {
        if !matches!(graph.code(key)?, Code::Scalar(_) | Code::Constant { .. })
            || graph.output_count(key) != 1
            || query.lookup("ValuePrimitive", graph.output(key, 0)?)?.is_none()
            || !query.flag("PureOperation", key)?
            || query.contains("LaneDependentOperation", key)?
        {
            return Ok(false);
        }
        regions.insert(graph.operation_region(key)?);
        uses.push((key, use_sites(query, key)?));
    }
    let _timing = timing::span("Rust unused leaf publication");
    let removed: LookupSet<_> = dead.iter().copied().collect();
    let mut previous = Vec::new();
    let mut updated = Vec::new();
    for region in regions {
        let operations = graph.region_order(region)?;
        previous.push((region, operations.to_vec()));
        updated.push((
            region,
            operations.iter().copied().filter(|op| !removed.contains(op)).collect(),
        ));
    }
    let work = planning::invalidate_region_work(graph, &previous)?;
    // Dead leaves can use one another. Keep all definitions until their last
    // lexical use has been retracted.
    for (operation, names) in &uses {
        for &value in names {
            scalar::remove_use(graph, *operation, value)?;
        }
    }
    for (operation, names) in uses {
        let region = graph.operation_region(operation)?;
        let output = graph.output(operation, 0)?;
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
                "ConstantValue",
                "IntegerLower",
                "IntegerUpper",
                "TrackedI32",
                "IntegerBoundsDemand",
            ] {
                sink.remove(table, output)?;
            }
            sink.remove("ResultOrigin", (output, operation, 0_i64))?;
            sink.remove("DefinitionRegion", (output, region))
        })?;
    }
    edit::remove_without_revision(graph, dead)?;
    analysis::replace_order(graph, previous, updated)?;
    planning::restore_region_work(graph, work)?;
    edit::preserve_revisions(graph)?;
    Ok(true)
}
