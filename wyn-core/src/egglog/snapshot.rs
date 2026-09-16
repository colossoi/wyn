//! Execution dependencies, liveness and safety shared by compiler passes.
use super::data::{AssociatedData, ExprId, ExprKind, OperationId, OperationKind, RegionId, SoacBody};
use super::optimize::OptimizeError;
use super::visit::Operand;
use crate::{types, LookupMap};
use std::collections::{BTreeMap, BTreeSet};
use wyn_base::persistent_sets::{Set, Sets, EMPTY};
mod fusion;
mod order;
mod worklists;
pub(super) use fusion::{analyze as fusion, Role};

/// Stop at operation results. Nested regions are visited separately so dead
/// work inside a lambda or branch cannot keep an enclosing producer alive.
#[derive(Clone, Copy, Default)]
struct References {
    operations: Set,
    regions: Set,
}
impl References {
    fn extend(&mut self, other: &Self, sets: &mut Sets) {
        self.operations = sets.union(self.operations, other.operations);
        self.regions = sets.union(self.regions, other.regions);
    }
    fn dependencies(&self, external: &BTreeMap<RegionId, Set>, sets: &mut Sets) -> Set {
        let mut result = self.operations;
        let regions: Vec<_> = sets.iter(self.regions).map(RegionId::from).collect();
        for region in regions {
            result = sets.union(result, external[&region]);
        }
        result
    }
}
pub(super) struct Snapshot {
    pub live: BTreeSet<OperationId>,
    data_predecessors: BTreeMap<OperationId, Vec<OperationId>>,
    pub effects: order::Effects,
    pub movable: BTreeSet<OperationId>,
    pub discardable: BTreeSet<OperationId>,
    pub safe_regions: BTreeSet<RegionId>,
}

pub(super) fn analyze(data: &AssociatedData) -> Snapshot {
    Analysis::new(data).snapshot
}

// Retain the expression cache while deriving optional fusion facts. Core callers
// keep only the execution snapshot; they never materialize fusion use tables.
struct Analysis<'a> {
    snapshot: Snapshot,
    visitor: Visitor<'a>,
    external: BTreeMap<RegionId, Set>,
    results: BTreeMap<RegionId, References>,
    active: BTreeSet<RegionId>,
}
impl<'a> Analysis<'a> {
    fn new(data: &'a AssociatedData) -> Self {
        let mut visitor = Visitor {
            data,
            expressions: LookupMap::new(),
            sets: Sets::default(),
        };
        let members: BTreeSet<_> = data.regions.values().flat_map(|r| r.members.iter().copied()).collect();
        let operands: BTreeMap<_, _> =
            members.iter().map(|&id| (id, visitor.operation(&data.operations[id].kind))).collect();
        let results: BTreeMap<_, _> =
            data.regions.iter().map(|(&id, r)| (id, visitor.expressions(&r.results))).collect();
        let (safe_regions, movable, discardable) = worklists::safety(data, &members);
        let external = worklists::external(data, &members, &operands, &results, &mut visitor.sets);
        let (active, live) =
            worklists::live(data, &members, &discardable, &operands, &results, &visitor.sets);
        let mut data_predecessors = BTreeMap::new();
        for &consumer in &live {
            let deps = operands[&consumer].dependencies(&external, &mut visitor.sets);
            data_predecessors.insert(consumer, visitor.sets.iter(deps).map(OperationId::from).collect());
        }
        let effects = order::effects(data, &live, &movable);
        let snapshot = Snapshot {
            live,
            data_predecessors,
            effects,
            movable,
            discardable,
            safe_regions,
        };
        Self {
            snapshot,
            visitor,
            external,
            results,
            active,
        }
    }
}
impl Snapshot {
    /// Consumer/producer pairs, derived from the canonical adjacency index.
    pub(super) fn dependencies(&self) -> impl Iterator<Item = (OperationId, OperationId)> + '_ {
        self.data_predecessors.iter().flat_map(|(&after, before)| before.iter().map(move |&b| (after, b)))
    }

    pub(super) fn schedules(
        &self,
        data: &AssociatedData,
    ) -> Result<BTreeMap<RegionId, Vec<OperationId>>, OptimizeError> {
        order::schedules(self, data)
    }

    /// Visit the root and its transitive dependencies in the same lexical region.
    /// Shared effect gates are ordinary graph vertices, visited only once.
    pub(super) fn walk_dependencies(
        &self,
        data: &AssociatedData,
        root: OperationId,
        mut visit: impl FnMut(OperationId),
    ) {
        wyn_graph::for_each_reachable(
            [order::Node::Operation(root)],
            wyn_graph::WalkOrder::DepthFirst,
            |node, out| order::predecessors(self, data, node, out),
            |node| {
                if let order::Node::Operation(op) = node {
                    visit(op);
                }
            },
        );
    }
}

pub(super) fn safe_body(body: &SoacBody, regions: &BTreeSet<RegionId>) -> bool {
    let mut safe = true;
    body.for_each_apply(&mut |b| {
        if let SoacBody::Apply { region, .. } = b {
            safe &= regions.contains(region);
        }
    });
    safe
}

struct Visitor<'a> {
    data: &'a AssociatedData,
    expressions: LookupMap<ExprId, References>,
    sets: Sets,
}
impl Visitor<'_> {
    fn expressions(&mut self, ids: &[ExprId]) -> References {
        let mut refs = References::default();
        for &id in ids {
            let value = self.expression(id);
            refs.extend(&value, &mut self.sets);
        }
        refs
    }
    fn expression(&mut self, id: ExprId) -> References {
        let order = wyn_graph::dag_postorder(
            [id],
            |e| self.expressions.contains_key(&e),
            |e, out| out.extend(self.data.expressions[e].kind.children()),
        );
        for e in order {
            let mut refs = References::default();
            match self.data.expressions[e].kind {
                ExprKind::OperationResult(op) => refs.operations = self.sets.singleton(op.as_u32()),
                ExprKind::Lambda(r) => refs.regions = self.sets.singleton(r.as_u32()),
                _ => {
                    for child in self.data.expressions[e].kind.children() {
                        refs.extend(&self.expressions[&child], &mut self.sets);
                    }
                }
            }
            self.expressions.insert(e, refs);
        }
        self.expressions[&id]
    }
    fn operation(&mut self, kind: &OperationKind) -> References {
        let mut refs = References::default();
        kind.for_each_operand(&mut |operand| match operand {
            Operand::Value(_, e) => {
                let value = self.expression(e);
                refs.extend(&value, &mut self.sets);
            }
            Operand::Region(r) => refs.regions = self.sets.insert(refs.regions, r.as_u32()),
        });
        refs
    }
}
