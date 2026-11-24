use super::Body;
use crate::egglog::analysis::{self, scopes};
use crate::egglog::dataflow::edit;
use crate::egglog::graph::Graph;
use crate::egglog::query::Query;
use crate::egglog::{planning, timing, OptimizeError};
use crate::LookupSet;
use egglog_engine::Write;

impl Body {
    pub(in crate::egglog::fusion::parallel) fn repair_empty_joins(
        &self,
        graph: &mut Graph,
    ) -> Result<bool, OptimizeError> {
        let query = Query(graph);
        if !self.relocated.is_empty() || !analysis::current(query.0)? {
            return Ok(false);
        }
        let mut removed = scopes::Subjects::default();
        let mut changed = scopes::Subjects::default();
        for (destination, sources) in &self.joined_yields {
            let destination = self.graph.regions[destination];
            changed.regions.push(destination);
            for &source in sources {
                let source = self.graph.regions[source];
                if source != destination {
                    if !query.0.region_order(source)?.is_empty() {
                        return Ok(false);
                    }
                    removed.regions.push(source);
                }
            }
        }
        for &op in &self.removed {
            let operation = self.graph.operations[op];
            if !query.flag("PureOperation", operation)? || !query.flag("MovableOperation", operation)? {
                return Ok(false);
            }
            removed.operations.push(operation);
        }
        changed.operations.extend(self.joined_branches.iter().map(|&op| self.graph.operations[op]));
        let _timing = timing::span("Rust empty-arm join proof repair");
        let mut order_regions: LookupSet<_> =
            self.moved.iter().map(|&region| self.graph.regions[region]).collect();
        order_regions.extend(removed.regions.iter().copied());
        for &operation in &removed.operations {
            order_regions.insert(graph.operation_region(operation)?);
        }
        let previous = order_regions
            .iter()
            .map(|&region| Ok((region, graph.region_order(region)?.to_vec())))
            .collect::<Result<Vec<_>, OptimizeError>>()?;
        let mut old_origins = Vec::new();
        for &operation in &removed.operations {
            old_origins.extend(
                graph
                    .outputs(operation)
                    .into_iter()
                    .enumerate()
                    .map(|(slot, name)| (name, operation, slot as i64)),
            );
        }
        let removed_regions: LookupSet<_> = removed.regions.iter().copied().collect();
        let mut dead_ancestors = Vec::new();
        query.for_each("FlowScopeAncestor", |row| {
            if removed_regions.contains(&row[1]) {
                dead_ancestors.push((row[0], row[1]));
            }
            Ok(())
        })?;
        planning::invalidate_region_work(graph, &previous)?;
        self.publish_joins(graph)?;
        graph.update(|mut sink| self.publish_order(&mut sink))?;
        edit::remove_without_revision(graph, &removed.operations)?;
        scopes::retire(graph, std::slice::from_ref(&removed))?;
        graph.update(|mut sink| {
            for &region in &removed.regions {
                sink.remove("RegionOwnedBy", region)?;
            }
            for row in old_origins {
                sink.remove("ResultOrigin", row)?;
            }
            for row in dead_ancestors {
                sink.remove("FlowScopeAncestor", row)?;
            }
            Ok(())
        })?;
        graph.dirty_owners.retain(|region| !removed_regions.contains(region));
        graph.proof_scopes.forget_empty_arms(&removed.regions, &removed.operations);
        let query = Query(graph);
        let updated = order_regions
            .into_iter()
            .filter(|region| !removed_regions.contains(region))
            .map(|region| Ok((region, query.operations(region)?)))
            .collect::<Result<_, OptimizeError>>()?;
        analysis::replace_order(graph, previous, updated)?;
        // Each result keeps its name, guard, and incoming values, so value proofs remain valid.
        analysis::publish_proof_operands(graph, std::slice::from_ref(&changed))?;
        graph.parse_and_run_program(None, "(run-schedule (saturate dataflow-dependencies))")?;
        edit::preserve_revisions(graph)?;
        Ok(true)
    }
}
