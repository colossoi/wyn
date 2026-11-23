//! Follow the dependencies of observable operations and returned values.
use super::super::dataflow::body::BodyGraph;
use super::super::query::Query;
use super::super::{analysis, output_error, OptimizeError};
use crate::memory_ir::{Definition, Function, OperationId, ValueRef};
use crate::LookupSet;
use egglog_engine::Value;
use slotmap::SecondaryMap;

#[derive(Default)]
pub(super) struct Needed {
    names: LookupSet<ValueRef>,
    uses: SecondaryMap<OperationId, Vec<ValueRef>>,
    pub operations: LookupSet<OperationId>,
}

impl Needed {
    pub fn read(query: Query<'_>, body: &BodyGraph) -> Result<Self, OptimizeError> {
        let mut needed = Self::default();
        let mut parents = SecondaryMap::new();
        for (owner, operation) in &body.ir.operations {
            for edge in &operation.bodies {
                for &child in &body.ir.regions[edge.region].operations {
                    parents.insert(child, owner);
                }
            }
        }

        query.for_each("OperationUse", |row| {
            if let Some(&op) = body.operation_ids.get(&row[0]) {
                let value = value_ref(body, row[1])?;
                if !needed.uses.contains_key(op) {
                    needed.uses.insert(op, Vec::new());
                }
                needed.uses[op].push(value);
            }
            Ok(())
        })?;

        let mut pending = Vec::new();
        for table in ["OperationTrap", "OperationWrite", "OperationRead"] {
            query.for_each(table, |row| {
                let key = row[0];
                let Some(&op) = body.operation_ids.get(&key) else {
                    return Ok(());
                };
                if table != "OperationRead"
                    || matches!(body.ir.operations[op].definition, Definition::Call(_))
                    || query.lookup("DiscardableRead", key)? == Some(query.0.base_to_value(false))
                {
                    pending.push(op);
                }
                Ok(())
            })?;
        }
        let mut names = Vec::new();
        for function in body.ir.functions.values() {
            if let Function::Defined(root) = *function {
                names.extend(&body.ir.regions[root].yields);
            }
        }
        while !pending.is_empty() || !names.is_empty() {
            for name in names.drain(..) {
                if needed.names.insert(name) {
                    if let ValueRef::Result(op, _) = name {
                        pending.push(op);
                    }
                }
            }
            let Some(op) = pending.pop() else { break };
            if !needed.operations.insert(op) {
                continue;
            }
            pending.extend(parents.get(op).copied());
            names.extend(needed.uses.get(op).into_iter().flatten().copied());
            let operation = &body.ir.operations[op];
            for edge in &operation.bodies {
                names.extend(&body.ir.regions[edge.region].yields);
            }
            let key = body.operations[op];
            if let Some(count) = query.lookup("ResultCount", key)? {
                for clause in 0..query.0.value_to_base::<i64>(count) {
                    let fields = analysis::count(query, "ResultValueCount", (key, clause))?;
                    for field in 0..fields {
                        names.push(value_ref(
                            body,
                            query.required("ResultValue", (key, clause, field))?,
                        )?);
                    }
                    if let Some(predicate) = query.lookup("ResultPredicate", (key, clause))? {
                        let name = query
                            .constructor("ValueNameRef", predicate)?
                            .ok_or_else(|| output_error("collective predicate has no value name"))?;
                        names.push(value_ref(body, name)?);
                    }
                }
            }
        }
        Ok(needed)
    }

    pub fn use_sites(&self, op: OperationId) -> impl Iterator<Item = ValueRef> + '_ {
        self.uses.get(op).into_iter().flatten().copied()
    }

    pub fn uses(&self, body: &BodyGraph, name: Value) -> bool {
        body.refs.get(&name).is_some_and(|value| self.names.contains(value))
    }
}

fn value_ref(body: &BodyGraph, name: Value) -> Result<ValueRef, OptimizeError> {
    let Some(&value) = body.refs.get(&name) else {
        return Err(output_error("demand has no live memory definition"));
    };
    Ok(value)
}

pub(super) fn dead_clause(
    query: Query<'_>,
    needed: impl Fn(Value) -> bool,
) -> Result<Option<(Value, i64)>, OptimizeError> {
    let mut dead = None;
    query.for_function("ResultKindOf", |keys, kind| {
        let operation = keys[0];
        let clause = query.0.value_to_base(keys[1]);
        if dead.is_some()
            || !query.contains("LiveOperation", operation)?
            || analysis::count(query, "ResultValueCount", (operation, clause))? != 1
            || (query.enode("Collected", kind)?.is_none() && query.enode("Compacted", kind)?.is_none())
        {
            return Ok(());
        }
        for (slot, output) in query.0.outputs(operation).into_iter().enumerate() {
            if needed(output) && analysis::count(query, "OutputClause", (operation, slot as i64))? == clause
            {
                return Ok(());
            }
        }
        dead = Some((operation, clause));
        Ok(())
    })?;
    Ok(dead)
}
