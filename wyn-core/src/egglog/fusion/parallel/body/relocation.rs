use super::Body;
use crate::egglog::dataflow::edit;
use crate::egglog::graph::Graph;
use crate::egglog::query::Query;
use crate::egglog::{analysis, planning, timing, OptimizeError};
use crate::memory_ir::{OperationId, RegionId, ValueRef};
use crate::{LookupMap, LookupSet};
use egglog_engine::{FullState, Value, Write};

impl Body {
    pub(in crate::egglog::fusion::parallel) fn publish_hoists(
        self,
        graph: &mut Graph,
    ) -> Result<(), OptimizeError> {
        let _timing = timing::span("Rust hoist publication");
        let query = Query(graph);
        let (previous, updated) = self.orders(query)?;
        let old_regions = self
            .relocated
            .iter()
            .map(|&op| {
                let region = query.required("OperationRegion", self.graph.operations[op])?;
                Ok((op, self.regions[&region]))
            })
            .collect::<Result<LookupMap<_, _>, OptimizeError>>()?;
        self.repair_scopes(graph, &old_regions)?;
        self.repair_reads(graph, &old_regions)?;
        let work = planning::invalidate_region_work(graph, &previous)?;
        graph.update(|mut sink| self.publish_order(&mut sink))?;
        analysis::replace_order(graph, previous, updated)?;
        planning::restore_region_work(graph, work)?;
        edit::preserve_revisions(graph)
    }

    fn repair_scopes(
        &self,
        graph: &mut Graph,
        old_regions: &LookupMap<OperationId, RegionId>,
    ) -> Result<(), OptimizeError> {
        let mut names = LookupSet::new();
        let mut definitions = Vec::new();
        let mut children = Vec::new();
        for (&op, &old) in old_regions {
            let region = self.positions[op].0;
            names.extend(self.tree_uses(op).into_iter().map(|(_, name)| name));
            for slot in 0..self.graph.ir.operations[op].result_types.len() {
                let name = self.graph.names[&ValueRef::Result(op, slot)];
                names.insert(name);
                definitions.push((name, self.graph.regions[old], self.graph.regions[region]));
            }
            children.extend(self.subregions(op).into_iter().map(|child| (child, old, region)));
        }
        let mut lost_ancestors = LookupSet::new();
        for &(child, old, destination) in &children {
            let mut skipped = Vec::new();
            let mut ancestor = old;
            while ancestor != destination {
                skipped.push(ancestor);
                let owner = self.parents[ancestor];
                ancestor = old_regions.get(&owner).copied().unwrap_or(self.positions[owner].0);
            }
            let mut descendants = vec![child];
            while let Some(region) = descendants.pop() {
                lost_ancestors.extend(skipped.iter().map(|&ancestor| (ancestor, region)));
                for &op in &self.graph.ir.regions[region].operations {
                    descendants.extend(self.subregions(op));
                }
            }
        }

        let mut uses = Vec::new();
        let mut captures = Vec::new();
        let unchanged = LookupMap::new();
        for name in names {
            let mut before = Uses::default();
            let mut after = Uses::default();
            for site in self.users(name) {
                let operation = self.site_operation(site);
                let region = self.demand_region(site)?;
                let prior = operation.and_then(|op| old_regions.get(&op).copied()).unwrap_or(region);
                let value = self.graph.refs[&name];
                let definition = self.definition_region(value);
                let old_definition = match value {
                    ValueRef::Result(op, _) => old_regions.get(&op).copied().unwrap_or(definition),
                    ValueRef::Parameter(..) => definition,
                };
                before.record(self, operation, prior, old_definition, old_regions);
                after.record(self, operation, region, definition, &unchanged);
            }
            for (set, other, used) in [(&before, &after, false), (&after, &before, true)] {
                for &op in set.operations.difference(&other.operations) {
                    let key = self.graph.operations[op];
                    uses.push(((key, name), used));
                }
                for &region in set.captures.difference(&other.captures) {
                    captures.push((self.graph.regions[region], name, used));
                }
            }
        }
        graph.update(|mut sink| {
            for (name, old, region) in definitions {
                sink.remove("DefinitionRegion", (name, old))?;
                sink.add("DefinitionRegion", (name, region))?;
            }
            for (child, old, region) in children {
                let child = self.graph.regions[child];
                sink.remove("FlowScopeParent", (child, self.graph.regions[old]))?;
                sink.add("FlowScopeParent", (child, self.graph.regions[region]))?;
            }
            for (ancestor, region) in lost_ancestors {
                let key = (self.graph.regions[ancestor], self.graph.regions[region]);
                sink.remove("FlowScopeAncestor", key)?;
            }
            for (key, used) in uses {
                for table in ["OperationUse", "AvailabilityDemand", "AvailableAt"] {
                    update(&mut sink, table, key, used)?;
                }
            }
            for (region, name, used) in captures {
                for table in ["RegionCapture", "EntryDemand", "AvailableIn"] {
                    update(&mut sink, table, (region, name), used)?;
                }
            }
            Ok(())
        })?;
        Ok(())
    }

    fn repair_reads(
        &self,
        graph: &mut Graph,
        old_regions: &LookupMap<OperationId, RegionId>,
    ) -> Result<(), OptimizeError> {
        let mut regions = LookupSet::new();
        for (&op, &old) in old_regions {
            if let Some(effect) = self.effects.get(op) {
                for &read in &effect.reads {
                    regions.insert((old, read));
                    regions.insert((self.positions[op].0, read));
                }
            }
        }
        for (op, before) in &self.reads_before_lift {
            for &read in before.difference(&self.effects[op].reads) {
                let region = self.positions[op].0;
                regions.insert((old_regions.get(&op).copied().unwrap_or(region), read));
            }
        }

        let mut conflicts = Vec::new();
        if !self.reads_before_lift.is_empty() {
            Query(graph).for_each("OperationsConflict", |row| {
                let a = self.graph.operation_ids[&row[0]];
                let b = self.graph.operation_ids[&row[1]];
                let left = &self.effects[a];
                let right = &self.effects[b];
                if (self.reads_before_lift.contains_key(a) || self.reads_before_lift.contains_key(b))
                    && left.reads.is_disjoint(&right.writes)
                    && left.writes.is_disjoint(&right.reads)
                    && left.writes.is_disjoint(&right.writes)
                    && !(left.traps && (right.traps || !right.writes.is_empty()))
                    && (!right.traps || left.writes.is_empty())
                {
                    conflicts.push((row[0], row[1]));
                }
                Ok(())
            })?;
        }

        graph.update(|mut sink| {
            for (op, before) in &self.reads_before_lift {
                let key = self.graph.operations[op];
                let effect = &self.effects[op];
                for &read in before.difference(&effect.reads) {
                    sink.remove("OperationRead", (key, read))?;
                }
                if effect.reads.is_empty() && effect.writes.is_empty() && !effect.traps {
                    sink.remove("PureOperation", key)?;
                    sink.set("PureOperation", key, true)?;
                }
            }
            for (region, read) in regions {
                let still_read = self.graph.ir.regions[region]
                    .operations
                    .iter()
                    .any(|&op| self.effects.get(op).is_some_and(|effect| effect.reads.contains(&read)));
                update(
                    &mut sink,
                    "RegionRead",
                    (self.graph.regions[region], read),
                    still_read,
                )?;
            }
            for key in conflicts {
                sink.remove("OperationsConflict", key)?;
            }
            Ok(())
        })?;
        Ok(())
    }
}

#[derive(Default)]
struct Uses {
    operations: LookupSet<OperationId>,
    captures: LookupSet<RegionId>,
}

impl Uses {
    fn record(
        &mut self,
        body: &Body,
        operation: Option<OperationId>,
        mut region: RegionId,
        definition: RegionId,
        moved: &LookupMap<OperationId, RegionId>,
    ) {
        self.operations.extend(operation);
        while region != definition && self.captures.insert(region) {
            let Some(owner) = body.parent(region) else { break };
            self.operations.insert(owner);
            region = moved.get(&owner).copied().unwrap_or(body.positions[owner].0);
        }
    }
}

fn update(
    sink: &mut FullState<'_, '_>,
    relation: &str,
    key: (Value, Value),
    exists: bool,
) -> Result<(), egglog_engine::Error> {
    if exists {
        sink.add(relation, key)?;
    } else {
        sink.remove(relation, key)?;
    }
    Ok(())
}
