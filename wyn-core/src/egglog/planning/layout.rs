use crate::egglog::{dataflow::edit, query::Query, scalar, OptimizeError};
use egglog_engine::EGraph;

pub(super) fn fuse(graph: &mut EGraph) -> Result<bool, OptimizeError> {
    if graph.get_size("OperationLayout") == 0 {
        return Ok(false);
    }
    for table in [
        "LayoutRead",
        "ForwardLayoutExtent",
        "LayoutProducer",
        "LayoutLegal",
        "LayoutOrdinal",
        "FuseLayout",
    ] {
        graph.clear_function(table)?;
    }
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate layout-fusion) (run layout-fusion-select))",
    )?;
    let query = Query(graph);
    let mut extents = Vec::new();
    query.for_each("ForwardLayoutExtent", |row| {
        extents.push((row[0], row[1]));
        Ok(())
    })?;
    if !extents.is_empty() {
        for (operation, index) in extents {
            scalar::forward_length(graph, operation, index)?;
        }
        return Ok(true);
    }
    let Some(candidate) = query.first("FuseLayout")? else {
        return Ok(false);
    };
    let (producer, read) = (candidate[0], candidate[2]);
    let lane = query.required("ControlRegion", (producer, 0_i64))?;
    let result = query.required("ResultValue", (producer, 0_i64, 0_i64))?;
    let replaced = query.required("Output", (read, 0_i64))?;
    let row = query.required("Input", (read, 1_i64))?;
    let column = query.required("Input", (read, 2_i64))?;
    let cloned = edit::clone_region(graph, read, lane, &[row, row, column])?;
    let query = Query(graph);
    let operations = query.operations(cloned)?;
    let replacement = query.required("CloneName", (read, result))?;
    edit::replace(graph, &[(replaced, replacement)])?;
    edit::splice(graph, read, &operations)?;
    edit::remove(graph, &[producer, read])?;
    Ok(true)
}
