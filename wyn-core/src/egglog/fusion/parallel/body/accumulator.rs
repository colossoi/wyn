use super::{Body, Effects, Site};
use crate::egglog::{analysis::count, output_error, query::Query, OptimizeError};
use crate::memory_ir::{BodyRole, OperationId, RegionId, UseSite, ValueRef};
use crate::{LookupMap, LookupSet};
use egglog_engine::{EGraph, RawValues, Write};
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
        let yields = take(&mut self.graph.ir.regions[source].yields);
        let start = self.graph.ir.regions[destination].yields.len();
        for (slot, value) in yields.iter().enumerate() {
            self.forget(Site::Operand(UseSite::Yield(source, slot)));
            self.uses.add(
                Site::Operand(UseSite::Yield(destination, start + slot)),
                self.graph.names[value],
            );
        }
        self.graph.ir.regions[destination].yields.extend(yields);
        let region_uses = self.uses.at(Site::Region(source)).map(|(_, name)| name).collect::<Vec<_>>();
        for name in region_uses {
            self.uses.add(Site::Region(destination), name);
        }
        self.forget(Site::Region(source));
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

    pub(super) fn publish_accumulators(&self, graph: &mut EGraph) -> Result<(), OptimizeError> {
        if self.accumulator_origins.is_empty() {
            return Ok(());
        }
        let query = Query(graph);
        let index = |slot: usize| graph.base_to_value(slot as i64);
        let mut rows = Vec::new();
        let mut certificates = Vec::new();
        let mut destinations = LookupMap::new();
        for (op, origins) in &self.accumulator_origins {
            let key = self.graph.operations[op];
            let region = self.lane(op)?;
            let wire_region = self.graph.regions[region];
            let body = &self.graph.ir.regions[region];
            let mut base = 0;
            for &origin in origins {
                let origin = self.graph.operations[origin];
                let incoming = query.required("ControlRegion", (origin, 0_i64))?;
                destinations.insert(incoming, (wire_region, base));
                let size = count(query, "AccumulatorGroupCount", origin)?;
                for slot in 0..size {
                    let group = index(base + slot as usize);
                    for table in [
                        "AccumulatorPolicy",
                        "AccumulatorArrayCount",
                        "AccumulatorCombiner",
                    ] {
                        if let Some(value) = query.lookup(table, (origin, slot))? {
                            rows.push((table, vec![key, group], value));
                        }
                    }
                    for (field, array) in query
                        .indexed(
                            "AccumulatorArray",
                            &[origin, graph.base_to_value(slot)],
                            count(query, "AccumulatorArrayCount", (origin, slot))?,
                        )?
                        .into_iter()
                        .enumerate()
                    {
                        rows.push(("AccumulatorArray", vec![key, group, index(field)], array));
                    }
                    if let Some(ty) = query.lookup("RegionReturnType", (incoming, slot))? {
                        rows.push(("RegionReturnType", vec![wire_region, group], ty));
                    }
                }
                base += size as usize;
            }
            for (slot, parameter) in body.parameters.iter().enumerate() {
                let args = vec![wire_region, index(slot)];
                rows.push(("ParameterDiet", args.clone(), parameter.diet));
                rows.push((
                    "RegionParameter",
                    args,
                    self.graph.names[&ValueRef::Parameter(region, slot)],
                ));
            }
            for (slot, value) in body.yields.iter().enumerate() {
                rows.push((
                    "RegionYield",
                    vec![wire_region, index(slot)],
                    self.graph.names[value],
                ));
            }
            let outputs = self.graph.ir.operations[op].result_types.len();
            for slot in 0..outputs {
                rows.push((
                    "Output",
                    vec![key, index(slot)],
                    self.graph.names[&ValueRef::Result(op, slot)],
                ));
            }
            rows.extend([
                ("AccumulatorGroupCount", vec![key], index(base)),
                ("OutputCount", vec![key], index(outputs)),
                (
                    "RegionParameterCount",
                    vec![wire_region],
                    index(body.parameters.len()),
                ),
                ("RegionYieldCount", vec![wire_region], index(body.yields.len())),
            ]);
            for (_, name) in self.uses.at(Site::Certificate(op)) {
                certificates.push((key, query.name(name)?));
            }
        }
        let mut yields = Vec::new();
        query.for_each("YieldCertificate", |row| {
            if let Some(&(destination, base)) = destinations.get(&row[0]) {
                let slot = graph.value_to_base::<i64>(row[1]) + base as i64;
                yields.push((destination, slot, row[2]));
            }
            Ok(())
        })?;
        graph.update(|mut sink| {
            for (table, keys, value) in rows {
                sink.remove(table, RawValues(keys.clone()))?;
                sink.set(table, RawValues(keys), value)?;
            }
            for certificate in certificates {
                sink.add("OperationCertificate", certificate)?;
            }
            for certificate in yields {
                sink.add("YieldCertificate", certificate)?;
            }
            Ok(())
        })?;
        Ok(())
    }
}
