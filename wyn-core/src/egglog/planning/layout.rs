use crate::egglog::graph::CollectiveInput;
use crate::egglog::graph::Formula;
use crate::egglog::graph::Graph;
use crate::egglog::{dataflow::edit, output_error, query::Query, scalar, OptimizeError};
use egglog_engine::{Value, Write};

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
    if write_in_layout(graph)? {
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
    let inputs = query.0.inputs(read);
    let [_, first, ..] = inputs.as_slice() else {
        return Err(output_error("layout read has no coordinates"));
    };
    // The legality check excludes uses of the producer ordinal.
    let arguments = [*first].into_iter().chain(inputs.into_iter().skip(1)).collect::<Vec<_>>();
    let cloned = edit::clone_region(graph, read, lane, &arguments)?;
    let query = Query(graph);
    let operations = query.operations(cloned.region)?;
    let replacement = cloned.name(result);
    edit::replace(graph, &[(replaced, replacement)])?;
    edit::splice(graph, read, &operations)?;
    edit::remove(graph, &[producer, read])?;
    Ok(true)
}

fn write_in_layout(graph: &mut Graph) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let mut selected = None;
    query.for_each("WriteLayout", |row| {
        let (layout, producer, slot) = (row[0], row[1], graph.value_to_base::<i64>(row[2]));
        if selected.is_some() || !query.flag("WriteLayoutSafe", layout)? {
            return Ok(());
        }
        let output_domain = graph.operation_domain(layout)?;
        let domain = graph.operation_domain(producer)?;
        let (Some(size), Some(tiles), Some(lanes)) = (
            static_extent(query, output_domain, 0)?,
            static_extent(query, domain, 0)?,
            static_extent(query, domain, 1)?,
        ) else {
            return Ok(());
        };
        let Some(rows) = tiles.checked_mul(lanes).filter(|&rows| rows > 0) else {
            return Ok(());
        };
        if size <= 0 || size > i64::from(i32::MAX) || size % rows != 0 {
            return Ok(());
        }
        let rank = graph.value_to_base::<i64>(row[3]);
        // Zero records a lane value stored as-is. A row contributes one value per item.
        if rank == 3 || size == rows {
            let items = if rank == 3 { size / rows } else { 0 };
            selected = Some((layout, producer, slot, lanes, items, size));
        }
        Ok(())
    })?;
    let Some((layout, producer, slot, lanes, items, size)) = selected else {
        return Ok(false);
    };
    let output = graph.output(layout, 0_i64)?;
    edit::remove_without_revision(graph, &[layout])?;
    graph.update(|mut sink| {
        let role = sink.add("FlatOutput", (lanes, items, size))?;
        sink.set_output_role(producer, slot, role)?;
        sink.set_output(producer, slot, output)
    })?;
    edit::changed(graph)?;
    Ok(true)
}

fn static_extent(query: Query<'_>, domain: Value, axis: i64) -> Result<Option<i64>, OptimizeError> {
    let extent = query.0.domain_extent(domain, axis)?;
    Ok(query.lookup("StaticIndex", extent)?.map(|value| query.0.value_to_base::<i64>(value)))
}
