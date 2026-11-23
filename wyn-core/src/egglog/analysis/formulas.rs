//! Publish the type-independent dependencies of executed index expressions.
use crate::egglog::graph::{Formula, Graph};
use crate::egglog::{query::Query, OptimizeError};
use crate::LookupSet;
use egglog_engine::{Value, Write};

pub(super) fn publish(graph: &mut Graph, for_inlining: bool) -> Result<(), OptimizeError> {
    let query = Query(graph);
    let mut demands = LookupSet::new();
    let mut uses = Vec::new();
    let mut dimensions = Vec::new();
    let mut lane_axes = Vec::new();
    {
        let structure = graph.structure().read();
        if !for_inlining {
            for (&operation, domain) in &structure.domains.operations {
                if let Some(domain) = structure.domains.values.get(domain) {
                    for dimension in &domain.dimensions {
                        if let Some(extent) = dimension.extent {
                            if let (Some(lane), Some(axis)) =
                                (structure.controls.get(&(operation, 0)), dimension.coordinate)
                            {
                                lane_axes.push((lane.region, axis, extent));
                            }
                            uses.extend(
                                formula_names(query, Formula::Index, extent)?
                                    .into_iter()
                                    .map(|name| (operation, name)),
                            );
                        }
                    }
                }
            }
            for (&domain, layout) in &structure.domains.values {
                dimensions.extend((0..layout.dimensions.len()).map(|slot| (domain, slot as i64)));
            }
        }
        for (&kind, formulas) in &structure.formulas.by_kind {
            for (&operation, &formula) in formulas {
                match kind {
                    Formula::Extents => {
                        demands.insert(("ShapeDemand", formula));
                    }
                    Formula::Slices => {
                        demands.insert(("SliceDemand", formula));
                    }
                    _ => {}
                }
                if !for_inlining {
                    uses.extend(
                        formula_names(query, kind, formula)?.into_iter().map(|name| (operation, name)),
                    );
                }
            }
        }
    }
    if demands.is_empty() && uses.is_empty() && dimensions.is_empty() {
        return Ok(());
    }
    graph.update(|mut sink| {
        for axis in lane_axes {
            sink.add("LaneAxisExtent", axis)?;
        }
        for axis in dimensions {
            sink.add("IterationAxis", axis)?;
        }
        for (demand, term) in demands {
            sink.add(demand, term)?;
        }
        for (operation, name) in uses {
            let name = sink.add("ValueNameRef", name)?;
            sink.add("OperationAttributeUse", (operation, name))?;
        }
        Ok(())
    })?;
    Ok(())
}

pub(in crate::egglog) fn formula_names(
    query: Query<'_>,
    kind: Formula,
    term: Value,
) -> Result<LookupSet<Value>, OptimizeError> {
    let mut pending = match kind {
        Formula::Index | Formula::LoopCount => vec![term],
        Formula::Extents => query.vector(term)?,
        Formula::Slices => {
            let mut indices = Vec::new();
            for slice in query.vector(term)? {
                let (_, fields) = query.application(slice, &["At", "Slice"])?;
                indices.extend(fields);
            }
            indices
        }
        Formula::Layout => Vec::new(),
    };
    let mut seen = LookupSet::new();
    let mut names = LookupSet::new();
    while let Some(index) = pending.pop() {
        if !seen.insert(index) {
            continue;
        }
        let (kind, fields) = query.application(
            index,
            &[
                "IndexValue",
                "IndexConstant",
                "IndexAdd",
                "IndexSub",
                "IndexMul",
                "IndexDiv",
                "IndexMod",
                "IndexMin",
                "IndexMax",
            ],
        )?;
        match kind {
            "IndexValue" => {
                names.insert(fields[0]);
            }
            "IndexConstant" => {}
            _ => pending.extend(fields),
        }
    }
    Ok(names)
}
