//! Replace proven leaf values and repair only the operand edges they change.
use crate::egglog::dataflow::{
    edit,
    ir::{self, Code},
};
use crate::egglog::{output_error, query::Query, timing, OptimizeError};
use crate::{types, LookupSet};
use egglog_engine::sort::VecContainer;
use egglog_engine::{Core, EGraph, Value, Write};

pub(in crate::egglog) fn forward_lengths(
    graph: &mut EGraph,
    candidates_table: &str,
) -> Result<bool, OptimizeError> {
    let mut candidates = Vec::new();
    Query(graph).for_each(candidates_table, |row| {
        candidates.push((row[0], row[1]));
        Ok(())
    })?;
    if candidates.is_empty() {
        return Ok(false);
    }
    let _timing = timing::span("Rust length forwarding");
    let mut changed = false;
    for (operation, index) in candidates {
        let query = Query(graph);
        let old = ir::code(graph, query.required("OperationPayload", operation)?);
        if !matches!(old, Code::Length { .. }) {
            continue;
        }
        let (new, extent) = if let Some(bits) = query.lookup("StaticIndex", index)? {
            (
                Code::Constant {
                    ty: types::i32(),
                    bits: graph.value_to_base(bits),
                },
                None,
            )
        } else {
            (Code::IndexValue, Some(index))
        };
        if rewrite(graph, operation, old, new, extent)? {
            graph.update(|mut sink| sink.remove(candidates_table, (operation, index)))?;
            changed = true;
        }
    }
    if changed {
        edit::preserve_revisions(graph)?;
    }
    Ok(changed)
}

pub(super) fn fold(
    graph: &mut EGraph,
    operation: Value,
    bits: Value,
    constant: Code,
) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let old = ir::code(graph, query.required("OperationPayload", operation)?);
    if !matches!(old, Code::Scalar(_) | Code::IndexValue)
        || !query.contains("DiscardableOperation", operation)?
        || query.contains("OrderedOperation", operation)?
        || query.contains("LaneDependentOperation", operation)?
        || !rewrite(graph, operation, old, constant, None)?
    {
        return Ok(false);
    }
    graph.update(|mut sink| sink.remove("FoldedConstant", (operation, bits)))?;
    Ok(true)
}

pub(in crate::egglog) fn materialize_scratch(
    graph: &mut EGraph,
    candidates: Vec<(Value, Value)>,
) -> Result<(), OptimizeError> {
    let mut changed = false;
    for (operation, count) in candidates {
        changed |= rewrite(graph, operation, Code::Scratch, Code::Uninitialized, Some(count))?;
    }
    if changed {
        edit::preserve_revisions(graph)?;
    }
    Ok(())
}

fn rewrite(
    graph: &mut EGraph,
    operation: Value,
    old: Code,
    new: Code,
    extent: Option<Value>,
) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let inputs = query.sequence("Input", operation)?;
    let old_index = query.lookup("OperationIndex", (operation, 0_i64))?;
    let old_attributes = index_operands(query, old_index)?;
    let new_attributes = index_operands(query, extent)?;
    let mut removed = LookupSet::new();
    removed.extend(inputs.iter().copied());
    removed.extend(old_attributes.iter().copied());
    let mut changes = Vec::new();
    for name in removed.difference(&new_attributes).copied() {
        if !query.contains("OperationCertificate", (operation, query.name(name)?))? {
            let Some(path) = UsePath::read(query, operation, name)? else {
                return Ok(false);
            };
            changes.push((name, path, false));
        }
    }
    for &name in &new_attributes {
        let Some(path) = UsePath::read(query, operation, name)? else {
            return Ok(false);
        };
        changes.push((name, path, true));
    }
    graph.update(|mut sink| {
        Ok((|| -> Result<(), OptimizeError> {
            match old {
                Code::Length { axis } => sink.remove("LengthOperation", (operation, axis as i64))?,
                Code::Scalar(_) => sink.remove("ScalarPayload", operation)?,
                Code::IndexValue => {
                    sink.remove("OperationIndex", (operation, 0_i64))?;
                    sink.remove("IndexCalculationOperation", operation)?;
                }
                Code::Scratch => {}
                _ => {
                    unreachable!("equivalent rewrites only change length, scalar, index and scratch leaves")
                }
            }
            sink.remove("OperationPayload", operation)?;
            for slot in 0..inputs.len() as i64 {
                sink.remove("Input", (operation, slot))?;
                sink.remove("InputDiet", (operation, slot))?;
            }
            sink.remove("InputCount", operation)?;
            sink.set("InputCount", operation, 0_i64)?;
            for &name in old_attributes.difference(&new_attributes) {
                sink.remove("OperationAttributeUse", (operation, name))?;
            }
            for (name, _, used) in &changes {
                if *used {
                    sink.add("OperationAttributeUse", (operation, *name))?;
                    sink.add("DirectOperationUse", (operation, *name))?;
                } else {
                    sink.remove("DirectOperationUse", (operation, *name))?;
                }
            }
            if let Some(index) = extent {
                if matches!(new, Code::Uninitialized) {
                    let shape = sink.container_to_value(VecContainer {
                        data: vec![index],
                        do_rebuild: true,
                    });
                    sink.set("OperationShape", operation, shape)?;
                    sink.add("ShapeDemand", shape)?;
                    sink.set("ShapeAt", (shape, 0_i64), index)?;
                } else {
                    sink.set("OperationIndex", (operation, 0_i64), index)?;
                }
            }
            ir::properties(&mut sink, operation, new)
        })())
    })??;
    for (name, path, used) in changes {
        repair_use(graph, path, name, used)?;
    }
    Ok(true)
}

fn index_operands(query: Query<'_>, index: Option<Value>) -> Result<LookupSet<Value>, OptimizeError> {
    let Some(index) = index else {
        return Ok(LookupSet::new());
    };
    index_uses(query, index)?
        .into_iter()
        .map(|value| {
            let Some(name) = query.constructor("ValueNameRef", value)? else {
                return Err(output_error("index operand has no value name"));
            };
            Ok(name)
        })
        .collect()
}

fn index_uses(query: Query<'_>, index: Value) -> Result<LookupSet<Value>, OptimizeError> {
    let mut pending = vec![index];
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

struct UsePath {
    producer: Option<Value>,
    scopes: Vec<(Value, Value)>,
}

impl UsePath {
    fn read(query: Query<'_>, mut op: Value, name: Value) -> Result<Option<Self>, OptimizeError> {
        let Some(site) = query.lookup("Defines", name)? else {
            return Ok(None);
        };
        let (kind, fields) = query.application(site, &["OperationSite", "ParameterSite"])?;
        let producer = (kind == "OperationSite").then_some(fields[0]);
        let definition = if let Some(producer) = producer {
            query.required("OperationRegion", producer)?
        } else {
            fields[0]
        };
        let mut scopes = Vec::new();
        loop {
            let region = query.required("OperationRegion", op)?;
            scopes.push((region, op));
            if region == definition {
                if let Some(producer) = producer {
                    let before = query.0.value_to_base::<i64>(query.required("OperationOrder", producer)?);
                    let after = query.0.value_to_base::<i64>(query.required("OperationOrder", op)?);
                    if before >= after {
                        return Ok(None);
                    }
                }
                return Ok(Some(Self { producer, scopes }));
            }
            let Some(parent) = query.lookup("RegionOwnedBy", region)? else {
                return Ok(None);
            };
            op = parent;
        }
    }
}

fn repair_use(graph: &mut EGraph, path: UsePath, name: Value, mut used: bool) -> Result<(), OptimizeError> {
    let UsePath { producer, scopes } = path;
    let mut scopes = scopes.into_iter().peekable();
    while let Some((region, operation)) = scopes.next() {
        let query = Query(graph);
        if query.contains("OperationUse", (operation, name))? == used {
            break;
        }
        let enclosing = scopes.peek().map(|&(_, op)| op);
        let preserve_dependency = if let Some(producer) = producer {
            let mut remaining = false;
            for output in query.sequence("Output", producer)? {
                remaining |= output != name && query.contains("OperationUse", (operation, output))?;
            }
            remaining
        } else {
            false
        };
        graph.update(|mut sink| {
            if used {
                sink.add("OperationUse", (operation, name))?;
                sink.add("AvailabilityDemand", (operation, name))?;
                sink.add("AvailableAt", (operation, name))?;
                if enclosing.is_some() {
                    sink.add("AvailableIn", (region, name))?;
                }
            } else {
                sink.remove("OperationUse", (operation, name))?;
            }
            if let Some(producer) = producer {
                if used {
                    sink.add("OperationDependency", (operation, producer))?;
                } else if !preserve_dependency {
                    sink.remove("OperationDependency", (operation, producer))?;
                }
            }
            Ok(())
        })?;
        let Some(parent) = enclosing else { break };

        let query = Query(graph);
        let mut captured = used
            || query.contains("EndDemand", (region, name))?
            || query.contains("RegionAttributeUse", (region, name))?;
        if !captured {
            for op in query.vector(query.required("RegionOperations", region)?)? {
                captured |= query.contains("OperationUse", (op, name))?;
                if captured {
                    break;
                }
            }
        }
        if query.contains("RegionCapture", (region, name))? == captured {
            return Ok(());
        }
        graph.update(|mut sink| {
            if captured {
                sink.add("RegionCapture", (region, name))?;
            } else {
                sink.remove("RegionCapture", (region, name))?;
            }
            Ok(())
        })?;
        let query = Query(graph);
        used = captured
            || query.contains("DirectOperationUse", (parent, name))?
            || child_captures(query, parent, name)?;
    }
    Ok(())
}

fn child_captures(query: Query<'_>, parent: Value, name: Value) -> Result<bool, OptimizeError> {
    let mut slot = 0_i64;
    while let Some(region) = query.lookup("ControlRegion", (parent, slot))? {
        if query.contains("RegionCapture", (region, name))? {
            return Ok(true);
        }
        slot += 1;
    }
    for (table, count) in [
        ("ResultReducer", "ResultCount"),
        ("AccumulatorCombiner", "AccumulatorGroupCount"),
    ] {
        let Some(count) = query.lookup(count, parent)? else {
            continue;
        };
        for slot in 0..query.0.value_to_base::<i64>(count) {
            if let Some(region) = query.lookup(table, (parent, slot))? {
                if query.contains("RegionCapture", (region, name))? {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}
