use crate::egglog::graph::Graph;
use crate::egglog::{dataflow::edit, query::Query, scalar, OptimizeError};

pub(super) fn fuse(graph: &mut Graph) -> Result<bool, OptimizeError> {
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
    if scalar::forward_lengths(graph, "ForwardLayoutExtent")? {
        return Ok(true);
    }
    let query = Query(graph);
    let Some(candidate) = query.first("FuseLayout")? else {
        return Ok(false);
    };
    let (producer, read) = (candidate[0], candidate[2]);
    let lane = query.0.control_region(producer, 0_i64)?;
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
