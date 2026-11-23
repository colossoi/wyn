//! Retain proof subjects by function and propagate invalidation through actual calls.
use super::calls::Reachability;
use super::{Graph, OptimizeError, Query};
use crate::{LookupMap, LookupSet};
use egglog_engine::Value;
use wyn_graph::{for_each_reachable, WalkOrder};

#[derive(Clone, Default)]
pub(in crate::egglog) struct Changes {
    pub operations: LookupSet<Value>,
    pub regions: LookupSet<Value>,
    pub domains: LookupSet<Value>,
}

#[derive(Clone, Default)]
pub(in crate::egglog) struct Subjects {
    pub function: Option<Value>,
    pub operations: Vec<Value>,
    pub regions: Vec<Value>,
    pub names: LookupSet<Value>,
    pub values: LookupSet<Value>,
    pub tokens: LookupSet<Value>,
    pub domains: LookupSet<Value>,
}

impl Subjects {
    fn read(graph: &Graph, reach: &Reachability, function: Value) -> Result<Self, OptimizeError> {
        let query = Query(graph);
        let mut subjects = Self {
            function: Some(function),
            ..Self::default()
        };
        {
            let positions = graph.structure().read();
            for &region in reach.bodies.get(&function).into_iter().flatten() {
                subjects.regions.push(region);
                subjects.names.extend(positions.parameters.get(&region).into_iter().flatten().copied());
                for &operation in positions.by_region.get(&region).into_iter().flatten() {
                    if graph.has_operation(operation) {
                        subjects.operations.push(operation);
                        subjects
                            .names
                            .extend(positions.outputs.get(&operation).into_iter().flatten().copied());
                        subjects.domains.extend(positions.domains.operations.get(&operation).copied());
                    }
                }
            }
        }
        for &name in &subjects.names {
            if let Some(value) = query.value_name(name)? {
                subjects.values.insert(value);
            } else if let Some(token) = query.enode("AccNameRef", name)? {
                subjects.tokens.insert(token[0]);
            }
        }
        Ok(subjects)
    }
}

#[derive(Clone, Default)]
pub(in crate::egglog) struct Scopes {
    members: LookupMap<Value, Subjects>,
    owners: LookupMap<Value, Value>,
    callers: LookupMap<Value, LookupSet<Value>>,
}

pub(super) struct Refresh {
    pub retire: Vec<Subjects>,
    pub solve: Vec<Subjects>,
}

impl Scopes {
    pub(super) fn refresh(
        &mut self,
        graph: &Graph,
        reach: &Reachability,
        changes: Changes,
    ) -> Result<Refresh, OptimizeError> {
        let mut owners = LookupMap::new();
        let active: LookupSet<_> = reach.live.iter().copied().collect();
        for &function in &reach.live {
            for &region in reach.bodies.get(&function).into_iter().flatten() {
                owners.insert(region, function);
            }
        }
        let mut dirty: LookupSet<_> =
            self.members.keys().filter(|id| !active.contains(id)).copied().collect();
        dirty.extend(active.iter().filter(|id| !self.members.contains_key(id)).copied());
        let mut regions = changes.regions;
        regions.extend(graph.dirty_orders.iter().copied());
        let mut operations = changes.operations;
        if !changes.domains.is_empty() {
            let structure = graph.structure().read();
            operations.extend(
                structure
                    .domains
                    .operations
                    .iter()
                    .filter(|(_, domain)| changes.domains.contains(domain))
                    .map(|(&operation, _)| operation),
            );
        }
        for op in operations {
            if graph.has_operation(op) {
                regions.extend(graph.operation_region(op).ok());
            }
        }
        for region in regions {
            dirty.extend(self.owners.get(&region).copied());
            dirty.extend(owners.get(&region).copied());
        }

        let mut callers = LookupMap::<Value, LookupSet<Value>>::new();
        for &function in &reach.live {
            for site in reach.calls.calls.get(&function).into_iter().flatten() {
                callers.entry(site.callee).or_default().insert(function);
            }
        }
        let mut dependent = Vec::new();
        for_each_reachable(
            dirty,
            WalkOrder::DepthFirst,
            |function, next| {
                next.extend(self.callers.get(&function).into_iter().flatten().copied());
                next.extend(callers.get(&function).into_iter().flatten().copied());
            },
            |function| dependent.push(function),
        );

        self.owners = owners;
        self.callers = callers;
        let mut retire = Vec::new();
        let mut solve = Vec::new();
        for function in dependent {
            if let Some(previous) = self.members.remove(&function) {
                retire.push(previous);
            }
            if active.contains(&function) {
                let current = Subjects::read(graph, reach, function)?;
                self.members.insert(function, current.clone());
                solve.push(current);
            }
        }
        Ok(Refresh { retire, solve })
    }
}

/// Install the analysis and planning retirement rules after their tables exist.
pub(in crate::egglog) fn load(graph: &mut Graph) -> Result<(), OptimizeError> {
    graph.parse_and_run_program(Some("proof retirement".into()), include_str!("retire.egg"))?;
    Ok(())
}

impl Subjects {
    fn retired_keys(&self) -> impl Iterator<Item = (&'static str, Value)> + '_ {
        self.operations
            .iter()
            .copied()
            .map(|key| ("RetiredOperation", key))
            .chain(self.regions.iter().copied().map(|key| ("RetiredRegion", key)))
            .chain(self.names.iter().copied().map(|key| ("RetiredName", key)))
            .chain(self.values.iter().copied().map(|key| ("RetiredValue", key)))
            .chain(self.tokens.iter().copied().map(|key| ("RetiredToken", key)))
            .chain(self.domains.iter().copied().map(|key| ("RetiredDomain", key)))
            .chain(self.function.map(|key| ("RetiredFunction", key)))
    }
}

pub(super) fn prepare(graph: &mut Graph, reach: &Reachability) -> Result<Refresh, OptimizeError> {
    use egglog_engine::{RawValues, Write};
    let _timing = super::timing::span("egglog dirty scope proof repair");
    let changes = std::mem::take(&mut graph.proof_changes);
    let mut scopes = std::mem::take(&mut graph.proof_scopes);
    let refresh = scopes.refresh(graph, reach, changes)?;
    graph.proof_scopes = scopes;

    graph.update(|mut sink| {
        sink.remove("DataflowRevision", RawValues(vec![]))?;
        sink.remove("PlanningRevision", RawValues(vec![]))?;
        Ok(())
    })?;
    graph.clear_function("InvalidDataflow")?;
    retire(graph, &refresh.retire)?;
    if wyn_base::timing::enabled() {
        let operations = refresh.solve.iter().map(|scope| scope.operations.len()).sum::<usize>();
        eprintln!(
            "egglog scope proofs: {} updated of {} functions, {operations} operations",
            refresh.solve.len(),
            reach.live.len()
        );
    }
    Ok(refresh)
}

pub(in crate::egglog) fn retire(graph: &mut Graph, scopes: &[Subjects]) -> Result<(), OptimizeError> {
    use egglog_engine::Write;
    if scopes.is_empty() {
        return Ok(());
    }
    graph.update(|mut sink| {
        for (table, subject) in scopes.iter().flat_map(Subjects::retired_keys) {
            sink.add(table, subject)?;
        }
        Ok(())
    })?;
    graph.parse_and_run_program(
            None,
            "(run-schedule (saturate graph-indices) (run retire-indices) (run retire-analysis) (run retire-shape))",
        )?;
    for table in [
        "RetiredOperation",
        "RetiredRegion",
        "RetiredName",
        "RetiredValue",
        "RetiredToken",
        "RetiredFunction",
        "RetiredDomain",
        "RetiredIndex",
    ] {
        graph.clear_function(table)?;
    }
    Ok(())
}

#[cfg(test)]
pub(in crate::egglog) fn seed_fixture(graph: &mut Graph) -> Result<Subjects, OptimizeError> {
    let query = Query(graph);
    let mut subjects = Subjects::default();
    let mut operations = LookupSet::new();
    let mut regions = LookupSet::new();
    {
        let structure = graph.structure().read();
        operations.extend(structure.operations.keys().copied());
        operations.extend(structure.inputs.values.keys().copied());
        operations.extend(structure.outputs.keys().copied());
        operations.extend(structure.parallels.keys().copied());
        operations.extend(structure.domains.operations.keys().copied());
        operations.extend(structure.collective.groups.keys().copied());
        for terms in structure.formulas.by_kind.values() {
            operations.extend(terms.keys().copied());
        }
        regions.extend(structure.parameters.keys().copied());
        regions.extend(structure.yields.values.keys().copied());
        subjects.domains.extend(structure.domains.values.keys().copied());
    }
    query.for_each("OperationId", |row| {
        operations.extend(query.constructor("OperationId", row[0])?);
        Ok(())
    })?;
    query.for_each("LiveOperation", |row| {
        operations.insert(row[0]);
        Ok(())
    })?;
    query.for_each("LiveRegion", |row| {
        regions.insert(row[0]);
        Ok(())
    })?;
    subjects.operations.extend(operations);
    subjects.regions.extend(regions);
    for &operation in &subjects.operations {
        subjects.names.extend(graph.outputs(operation));
    }
    for &region in &subjects.regions {
        subjects.names.extend(graph.parameters(region));
    }
    for &name in &subjects.names {
        if let Some(raw) = query.value_name(name)? {
            subjects.values.insert(raw);
        } else if let Some(token) = query.enode("AccNameRef", name)? {
            subjects.tokens.insert(token[0]);
        }
    }
    query.for_function("ValueType", |key, _| {
        subjects.values.insert(key[0]);
        subjects.names.extend(query.constructor("ValueNameRef", key[0])?);
        Ok(())
    })?;
    query.for_function("AccumulatorType", |key, _| {
        subjects.tokens.insert(key[0]);
        subjects.names.extend(query.constructor("AccNameRef", key[0])?);
        Ok(())
    })?;
    Ok(subjects)
}
