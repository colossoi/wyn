use crate::egglog::graph::CollectiveInput;
use crate::egglog::graph::Formula;
use crate::egglog::graph::Graph;
use crate::egglog::{dataflow::edit, query::Query, scalar, OptimizeError};

pub(super) fn fuse(graph: &mut Graph) -> Result<bool, OptimizeError> {
    if !graph.has_formulas(Formula::Layout) {
        return Ok(false);
    }
    graph.parse_and_run_program(
        None,
        "(run-schedule (saturate transform-inputs layout-fusion) (run layout-fusion-select))",
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
    let result = query.0.collective_input(producer, 0_i64, CollectiveInput::Contribution, 0_i64)?;
    let replaced = query.0.output(read, 0_i64)?;
    let row = query.0.input(read, 1_i64)?;
    let column = query.0.input(read, 2_i64)?;
    let cloned = edit::clone_region(graph, read, lane, &[row, row, column])?;
    let query = Query(graph);
    let operations = query.operations(cloned.region)?;
    let replacement = cloned.name(result);
    edit::replace(graph, &[(replaced, replacement)])?;
    edit::splice(graph, read, &operations)?;
    edit::remove(graph, &[producer, read])?;
    Ok(true)
}
