use super::{Body, Effects, Site};
use crate::egglog::graph::CollectiveInput;
use crate::egglog::graph::Graph;
use crate::egglog::{output_error, OptimizeError};
use crate::memory_ir::{BodyRole, OperationId, RegionId, ValueRef};
use crate::LookupSet;
use egglog_engine::{RawValues, Write};
use std::mem::take;

impl Body {
    pub(in crate::egglog::fusion::parallel) fn join_accumulators(
        &mut self,
        first: OperationId,
        second: OperationId,
    ) -> Result<RegionId, OptimizeError> {
        let destination = self.lane(first)?;
        let source = self.lane(second)?;
        let offset = self.graph.ir.regions[destination].parameters.len();
        let parameters = take(&mut self.graph.ir.regions[source].parameters);
        for slot in 0..parameters.len() {
            self.move_value(
                ValueRef::Parameter(source, slot),
                ValueRef::Parameter(destination, offset + slot),
            );
        }
        self.graph.ir.regions[destination].parameters.extend(parameters);
        self.append_yields(destination, source)?;
        let combiners = self.graph.ir.operations[second]
            .bodies
            .iter()
            .filter_map(|edge| {
                let BodyRole::Combiner(group) = edge.role else {
                    return None;
                };
                let mut edge = edge.clone();
                edge.role = BodyRole::Combiner(offset + group);
                Some(edge)
            })
            .collect::<Vec<_>>();
        for edge in &combiners {
            self.parents.insert(edge.region, first);
        }
        self.graph.ir.operations[first].bodies.extend(combiners);
        let attributes = self.uses.at(Site::Attribute(second)).map(|(_, name)| name).collect::<Vec<_>>();
        for name in attributes {
            self.uses.add(Site::Attribute(first), name);
        }
        self.append_results(first, second);
        self.append_operations(destination, source);
        self.absorb_effects(first, second);
        let origins = self.accumulator_origins.remove(second).unwrap_or_else(|| vec![second]);
        let Some(entry) = self.accumulator_origins.entry(first) else {
            return Err(output_error("accumulator has no live occurrence key"));
        };
        entry.or_insert_with(|| vec![first]).extend(origins);
        self.drop_op(second);
        Ok(destination)
    }

    pub(in crate::egglog::fusion::parallel) fn sink_producers(
        &mut self,
        scope: OperationId,
        destination: RegionId,
        members: &[OperationId],
    ) {
        let source = self.positions[scope].0;
        let moved: LookupSet<_> = members.iter().copied().collect();
        self.graph.ir.regions[source].operations.retain(|op| !moved.contains(op));
        self.graph.ir.regions[destination].operations.splice(0..0, members.iter().copied());
        self.relocated.extend(members);
        for region in [source, destination] {
            for (index, &operation) in self.graph.ir.regions[region].operations.iter().enumerate() {
                self.positions[operation] = (region, index);
            }
            self.moved.insert(region);
            self.dirty(region);
        }
        let mut effects = self.effects.remove(scope).unwrap_or_default();
        for &member in members {
            if let Some(Effects { reads, traps, .. }) = self.effects.get(member) {
                effects.reads.extend(reads);
                effects.traps |= traps;
            }
        }
        self.effects.insert(scope, effects);
    }

    pub(super) fn publish_accumulators(&self, graph: &mut Graph) -> Result<(), OptimizeError> {
        if self.accumulator_origins.is_empty() {
            return Ok(());
        }
        let index = |slot: usize| graph.base_to_value(slot as i64);
        let mut rows = Vec::new();
        let mut parameters = Vec::new();
        let mut callbacks = Vec::new();
        let mut backings = Vec::new();
        let mut policies = Vec::new();
        for (op, origins) in &self.accumulator_origins {
            let key = self.graph.operations[op];
            let region = self.lane(op)?;
            let wire_region = self.graph.regions[region];
            let body = &self.graph.ir.regions[region];
            let mut base = 0;
            let mut combiners = Vec::new();
            let mut combined_policies = Vec::new();
            for &origin in origins {
                let origin = self.graph.operations[origin];
                let groups = graph.accumulator_policies(origin);
                let size = groups.len();
                for (slot, policy) in groups.into_iter().enumerate() {
                    let slot = slot as i64;
                    if let Some(combine) = graph.combiner_at(origin, slot) {
                        combiners.push((base as i64 + slot, combine));
                    }
                    combined_policies.push(policy);
                    let arrays = graph.collective_inputs(origin, slot, CollectiveInput::Backing);
                    backings.push((key, base as i64 + slot, arrays));
                }
                base += size;
            }
            let mut names = Vec::with_capacity(body.parameters.len());
            for (slot, parameter) in body.parameters.iter().enumerate() {
                rows.push(("ParameterDiet", vec![wire_region, index(slot)], parameter.diet));
                names.push(self.graph.names[&ValueRef::Parameter(region, slot)]);
            }
            parameters.push((wire_region, names));
            callbacks.push((key, combiners));
            policies.push((key, combined_policies));
        }
        graph.update(|mut sink| {
            for (key, policies) in policies {
                sink.set_accumulator_policies(key, policies);
            }
            for (key, group, arrays) in backings {
                sink.set_collective_inputs(key, group, CollectiveInput::Backing, arrays)?;
            }
            for (op, combiners) in callbacks {
                sink.set_combiners(op, combiners);
            }
            for (region, names) in parameters {
                sink.set_parameters(region, names)?;
            }
            for (table, keys, value) in rows {
                sink.remove(table, RawValues(keys.clone()))?;
                sink.set(table, RawValues(keys), value)?;
            }
            Ok(())
        })?;
        Ok(())
    }
}
