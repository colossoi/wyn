//! Resolve projections of constructed products through native definitions.
use super::{Graph, OptimizeError, Origin, Query};
use crate::egglog::dataflow::ir::Code;
use crate::LookupMap;
use egglog_engine::Write;
use wyn_graph::dfs_postorder;

pub(super) fn publish(graph: &mut Graph) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut selected = Vec::new();
    query.for_each("ProjectionOperation", |row| {
        if query.contains("LiveOperation", row[0])? {
            selected.extend(graph.output_at(row[0], 0));
        }
        Ok(())
    })?;

    // Products can have projections as fields. Visit their inputs first so
    // following a field always reaches its existing definition.
    let mut result: Result<(), OptimizeError> = Ok(());
    let mut projections = LookupMap::new();
    let order = dfs_postorder(
        selected,
        |_| false,
        |name, children| {
            if result.is_err() {
                return;
            }
            result = (|| {
                let Some(Origin::Result(operation, 0)) = query.live_origin(name)? else {
                    return Ok(());
                };
                match graph.code(operation)? {
                    Code::Product => children.extend(graph.inputs(operation)),
                    Code::Project { index } => {
                        if let Some(input) = graph.input_at(operation, 0) {
                            children.push(input);
                            projections.insert(name, (input, index));
                        }
                    }
                    _ => {}
                }
                Ok(())
            })();
        },
    );
    result?;

    let mut aliases = LookupMap::new();
    let mut facts = Vec::new();
    for name in order {
        let Some((input, index)) = projections.remove(&name) else {
            continue;
        };
        let parent = aliases.get(&input).copied().unwrap_or(input);
        let Some(Origin::Result(operation, 0)) = query.live_origin(parent)? else {
            continue;
        };
        if graph.code(operation)? != Code::Product {
            continue;
        }
        let Some(field) = graph.input_at(operation, index as i64) else {
            continue;
        };
        if query.value_name(name)?.is_some() && query.value_name(field)?.is_some() && name != field {
            aliases.insert(name, aliases.get(&field).copied().unwrap_or(field));
            facts.push((name, field));
        }
    }
    graph.update(|mut sink| {
        for (name, source) in facts {
            sink.set("ProjectionAlias", name, source)?;
        }
        Ok(())
    })?;
    Ok(())
}
