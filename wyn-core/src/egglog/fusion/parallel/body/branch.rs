use super::{Body, Site};
use crate::egglog::{analysis::count, output_error, query::Query, OptimizeError};
use crate::memory_ir::{OperationId, RegionId, UseSite, ValueRef};
use crate::LookupMap;
use egglog_engine::{EGraph, RawValues, Write};
use std::mem::take;

impl Body {
    pub(in crate::egglog::fusion::parallel) fn join_branches(
        &mut self,
        first: OperationId,
        second: OperationId,
    ) -> Result<[RegionId; 2], OptimizeError> {
        let arms = [self.control_region(first, 0)?, self.control_region(first, 1)?];
        for (slot, destination) in arms.into_iter().enumerate() {
            let source = self.control_region(second, slot)?;
            self.append_yields(destination, source)?;
            self.append_operations(destination, source);
        }
        self.append_results(first, second);
        self.absorb_effects(first, second);
        self.joined_branches.remove(&second);
        self.joined_branches.insert(first);
        self.drop_op(second);
        Ok(arms)
    }

    pub(super) fn append_yields(
        &mut self,
        destination: RegionId,
        source: RegionId,
    ) -> Result<(), OptimizeError> {
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
        let origins = self.joined_yields.remove(source).unwrap_or_else(|| vec![source]);
        let Some(entry) = self.joined_yields.entry(destination) else {
            return Err(output_error("joined region has no live identity"));
        };
        entry.or_insert_with(|| vec![destination]).extend(origins);
        Ok(())
    }

    pub(super) fn publish_joins(&self, graph: &mut EGraph) -> Result<(), OptimizeError> {
        if self.joined_yields.is_empty() {
            return Ok(());
        }
        let query = Query(graph);
        let index = |slot: usize| graph.base_to_value(slot as i64);
        let mut rows = Vec::new();
        let mut destinations = LookupMap::new();
        for (region, origins) in &self.joined_yields {
            let target = self.graph.regions[region];
            let yields = &self.graph.ir.regions[region].yields;
            let mut base = 0;
            for &origin in origins {
                let origin = self.graph.regions[origin];
                destinations.insert(origin, (target, base));
                let size = count(query, "RegionYieldCount", origin)?;
                for slot in 0..size {
                    if let Some(ty) = query.lookup("RegionReturnType", (origin, slot))? {
                        rows.push(("RegionReturnType", vec![target, index(base + slot as usize)], ty));
                    }
                }
                base += size as usize;
            }
            for (slot, value) in yields.iter().enumerate() {
                rows.push(("RegionYield", vec![target, index(slot)], self.graph.names[value]));
            }
            rows.push(("RegionYieldCount", vec![target], index(yields.len())));
        }
        let mut certificates = Vec::new();
        for op in self.joined_branches.iter().copied().chain(self.accumulator_origins.keys()) {
            let key = self.graph.operations[op];
            let size = self.graph.ir.operations[op].result_types.len();
            for slot in 0..size {
                rows.push((
                    "Output",
                    vec![key, index(slot)],
                    self.graph.names[&ValueRef::Result(op, slot)],
                ));
            }
            rows.push(("OutputCount", vec![key], index(size)));
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
