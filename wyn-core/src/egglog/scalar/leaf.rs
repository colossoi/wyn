//! Replace proven leaf values and repair only the operand edges they change.
use crate::egglog::dataflow::{
    edit,
    ir::{self, Code},
};
use crate::egglog::graph::Formula;
use crate::egglog::graph::{Graph, Origin};
use crate::egglog::{output_error, query::Query, timing, OptimizeError};
use crate::{types, LookupSet};
use egglog_engine::sort::VecContainer;
use egglog_engine::{Core, Value, Write};

pub(in crate::egglog) fn forward_lengths(
    graph: &mut Graph,
    candidates_table: &str,
) -> Result<bool, OptimizeError> {
    let mut candidates = Vec::new();
    Query(graph).for_each(candidates_table, |row| {
        if Query(graph).contains("LiveOperation", row[0])? {
            candidates.push((row[0], row[1]));
        }
        Ok(())
    })?;
    if candidates.is_empty() {
        return Ok(false);
    }
    let _timing = timing::span("Rust length forwarding");
    let mut changed = false;
    for (operation, index) in candidates {
        let query = Query(graph);
        let old = query.0.code(operation)?;
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
    graph: &mut Graph,
    operation: Value,
    bits: Value,
    constant: Code,
) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let old = query.0.code(operation)?;
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
    graph: &mut Graph,
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
    graph: &mut Graph,
    operation: Value,
    old: Code,
    new: Code,
    extent: Option<Value>,
) -> Result<bool, OptimizeError> {
    let query = Query(graph);
    let inputs = query.0.inputs(operation);
    let old_index = query.0.formula_at(operation, Formula::Index);
    let old_attributes = index_operands(query, old_index)?;
    let new_attributes = index_operands(query, extent)?;
    let mut removed = LookupSet::new();
    removed.extend(inputs.iter().copied());
    removed.extend(old_attributes.iter().copied());
    let mut changes = Vec::new();
    for name in removed.difference(&new_attributes).copied() {
        if !graph.has_certificate(operation, query.name(name)?) {
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
                    sink.remove_formula(operation, Formula::Index);
                    sink.remove("IndexCalculationOperation", operation)?;
                }
                Code::Scratch => {}
                _ => {
                    unreachable!("equivalent rewrites only change length, scalar, index and scratch leaves")
                }
            }
            for slot in 0..inputs.len() as i64 {
                sink.remove("InputDiet", (operation, slot))?;
            }
            sink.set_inputs(operation, Vec::new())?;
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
                    sink.set_formula(operation, Formula::Extents, shape)?;
                    sink.set("ShapeAt", (shape, 0_i64), index)?;
                } else {
                    sink.set_formula(operation, Formula::Index, index)?;
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
    crate::egglog::analysis::formula_names(query, Formula::Index, index)?
        .into_iter()
        .map(|value| {
            let Some(name) = query.constructor("ValueNameRef", value)? else {
                return Err(output_error("index operand has no value name"));
            };
            Ok(name)
        })
        .collect()
}

/// A replacement of ordinary scalar operands whose lexical uses can be repaired in place.
pub(in crate::egglog) struct ScalarInputs {
    operation: Value,
    after: Vec<Value>,
    uses: Vec<(Value, UsePath, bool)>,
}

impl ScalarInputs {
    pub fn read(
        query: Query<'_>,
        operation: Value,
        updates: Vec<(usize, Value)>,
    ) -> Result<Option<Self>, OptimizeError> {
        let before = query.0.inputs(operation);
        let mut after = before.clone();
        for (slot, value) in updates {
            let Some(input) = after.get_mut(slot) else {
                return Ok(None);
            };
            *input = value;
        }
        let before_names: LookupSet<Value> = before.iter().copied().collect();
        let after_names: LookupSet<Value> = after.iter().copied().collect();
        let mut uses = Vec::new();
        for (a, b, used) in [
            (&before_names, &after_names, false),
            (&after_names, &before_names, true),
        ] {
            for &name in a.difference(b) {
                if !used && query.contains("OperationAttributeUse", (operation, name))? {
                    return Ok(None);
                }
                let Some(path) = UsePath::read(query, operation, name)? else {
                    return Ok(None);
                };
                uses.push((name, path, used));
            }
        }
        Ok(Some(Self {
            operation,
            after,
            uses,
        }))
    }

    pub fn apply(self, graph: &mut Graph) -> Result<(), OptimizeError> {
        let Self {
            operation,
            after,
            uses,
        } = self;
        graph.update(|mut sink| {
            sink.set_inputs(operation, after)?;
            for &(name, _, used) in &uses {
                if used {
                    sink.add("DirectOperationUse", (operation, name))?;
                } else {
                    sink.remove("DirectOperationUse", (operation, name))?;
                }
            }
            Ok(())
        })?;
        for (name, path, used) in uses {
            repair_use(graph, path, name, used)?;
        }
        Ok(())
    }
}

struct UsePath {
    scopes: Vec<(Value, Value)>,
}

impl UsePath {
    fn read(query: Query<'_>, mut op: Value, name: Value) -> Result<Option<Self>, OptimizeError> {
        let Some(origin) = query.live_origin(name)? else {
            return Ok(None);
        };
        let (definition, producer) = match origin {
            Origin::Result(operation, _) => (query.0.operation_region(operation)?, Some(operation)),
            Origin::Parameter(region, _) => (region, None),
        };
        let mut scopes = Vec::new();
        loop {
            let region = query.0.operation_region(op)?;
            scopes.push((region, op));
            if region == definition {
                if let Some(producer) = producer {
                    let before = query.0.operation_order(producer)?;
                    let after = query.0.operation_order(op)?;
                    if before >= after {
                        return Ok(None);
                    }
                }
                return Ok(Some(Self { scopes }));
            }
            let Some(parent) = query.lookup("RegionOwnedBy", region)? else {
                return Ok(None);
            };
            op = parent;
        }
    }
}

fn repair_use(graph: &mut Graph, path: UsePath, name: Value, mut used: bool) -> Result<(), OptimizeError> {
    let UsePath { scopes } = path;
    let mut scopes = scopes.into_iter().peekable();
    while let Some((region, operation)) = scopes.next() {
        let query = Query(graph);
        if query.contains("OperationUse", (operation, name))? == used {
            break;
        }
        let enclosing = scopes.peek().map(|&(_, op)| op);
        graph.update(|mut sink| {
            if used {
                sink.add("OperationUse", (operation, name)).map(|_| ())
            } else {
                sink.remove("OperationUse", (operation, name))
            }
        })?;
        let Some(parent) = enclosing else { break };

        let query = Query(graph);
        let mut captured = used
            || query.contains("EndDemand", (region, name))?
            || query.contains("RegionAttributeUse", (region, name))?;
        if !captured {
            for &op in query.0.region_order(region)? {
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
                sink.add("RegionCapture", (region, name)).map(|_| ())
            } else {
                sink.remove("RegionCapture", (region, name))
            }
        })?;
        let query = Query(graph);
        used = captured
            || query.contains("DirectOperationUse", (parent, name))?
            || child_captures(query, parent, name)?;
    }
    Ok(())
}

pub(in crate::egglog) fn remove_use(
    graph: &mut Graph,
    operation: Value,
    name: Value,
) -> Result<(), OptimizeError> {
    let Some(path) = UsePath::read(Query(graph), operation, name)? else {
        return Err(output_error("deleted leaf has an unavailable operand"));
    };
    repair_use(graph, path, name, false)
}

fn child_captures(query: Query<'_>, parent: Value, name: Value) -> Result<bool, OptimizeError> {
    for child in query.0.attached_regions(parent) {
        if query.contains("RegionCapture", (child, name))? {
            return Ok(true);
        }
    }
    Ok(false)
}
