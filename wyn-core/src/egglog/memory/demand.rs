//! Follow effect and yield dependencies on the authoritative Rust graph.
use super::super::dataflow::ir::Code;
use super::super::query::Query;
use super::super::{output_error, OptimizeError};
use crate::egglog::graph::CollectiveInput;
use crate::{LookupMap, LookupSet};
use egglog_engine::Value;

#[derive(Default)]
pub(super) struct Needed {
    pub names: LookupSet<Value>,
    uses: LookupMap<Value, Vec<Value>>,
    pub operations: LookupSet<Value>,
}

impl Needed {
    pub fn read(query: Query<'_>) -> Result<Self, OptimizeError> {
        let mut needed = Self::default();
        query.for_each("OperationUse", |row| {
            if query.contains("LiveOperation", row[0])? {
                needed.uses.entry(row[0]).or_default().push(row[1]);
            }
            Ok(())
        })?;
        let mut pending = Vec::new();
        for table in ["OperationTrap", "OperationWrite", "OperationRead"] {
            query.for_each(table, |row| {
                let op = row[0];
                if query.contains("LiveOperation", op)?
                    && (table != "OperationRead"
                        || matches!(query.0.code(op)?, Code::Call)
                        || query.lookup("DiscardableRead", op)? == Some(query.0.base_to_value(false)))
                {
                    pending.push(op);
                }
                Ok(())
            })?;
        }
        let mut names = Vec::new();
        for root in query.live_roots()? {
            names.extend(query.0.yield_values(root));
        }
        while !pending.is_empty() || !names.is_empty() {
            for name in names.drain(..) {
                if needed.names.insert(name) {
                    let Some(origin) = query.live_origin(name)? else {
                        return Err(output_error("demand has no live memory definition"));
                    };
                    if let crate::egglog::graph::Origin::Result(op, _) = origin {
                        pending.push(op);
                    }
                }
            }
            let Some(op) = pending.pop() else { break };
            if !needed.operations.insert(op) {
                continue;
            }
            let region = query.0.operation_region(op)?;
            pending.extend(query.0.region_owner(region));
            names.extend(needed.use_sites(op));
            for child in query.0.attached_regions(op) {
                names.extend(query.0.yield_values(child));
            }
            for clause in 0..query.0.result_count(op) {
                names.extend(query.0.collective_inputs(op, clause, CollectiveInput::Contribution));
                if let Some(predicate) =
                    query.0.collective_input_at(op, clause, CollectiveInput::Predicate, 0)
                {
                    let name = query
                        .constructor("ValueNameRef", predicate)?
                        .ok_or_else(|| output_error("collective predicate has no value name"))?;
                    names.push(name);
                }
            }
        }
        Ok(needed)
    }

    pub fn use_sites(&self, operation: Value) -> impl Iterator<Item = Value> + '_ {
        self.uses.get(&operation).into_iter().flatten().copied()
    }
}

pub(super) fn dead_clause(
    query: Query<'_>,
    needed: impl Fn(Value) -> bool,
) -> Result<Option<(Value, i64)>, OptimizeError> {
    let structure = query.0.structure().read();
    for (&operation, results) in &structure.parallels {
        if !query.contains("LiveOperation", operation)? {
            continue;
        }
        for (clause, kind) in results.kinds.iter().enumerate() {
            let Some(kind) = kind else { continue };
            let clause = clause as i64;
            if structure.collective.get(operation, clause, CollectiveInput::Contribution).len() != 1
                || (query.enode("Collected", *kind)?.is_none()
                    && query.enode("Compacted", *kind)?.is_none())
            {
                continue;
            }
            let mut live = false;
            for (&output, binding) in
                structure.outputs.get(&operation).into_iter().flatten().zip(&results.bindings)
            {
                if binding.clause == Some(clause) && needed(output) {
                    live = true;
                    break;
                }
            }
            if !live {
                return Ok(Some((operation, clause)));
            }
        }
    }
    Ok(None)
}
