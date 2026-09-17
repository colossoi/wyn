use super::{Dependencies, OptimizeError};
use crate::egglog::data::{Ir, OperationId, RegionId};
use std::collections::{BTreeMap, BTreeSet};

/// A gate waits for a group of operations. Consumers depend on the gate rather
/// than on every group member, so even tied source positions need O(n) edges.
#[derive(Default)]
pub(in crate::egglog) struct Effects {
    groups: Vec<Vec<OperationId>>,
    waits: BTreeMap<OperationId, Vec<usize>>,
}
impl Effects {
    /// Preserve shared gates when exporting order constraints to other analyses.
    pub(in crate::egglog) fn gates(&self) -> impl Iterator<Item = (usize, &[OperationId])> {
        self.groups.iter().enumerate().map(|(i, ops)| (i, ops.as_slice()))
    }

    pub(in crate::egglog) fn waits(&self) -> impl Iterator<Item = (OperationId, usize)> + '_ {
        self.waits.iter().flat_map(|(&op, gates)| gates.iter().map(move |&gate| (op, gate)))
    }

    fn gate(&mut self, before: Vec<OperationId>) -> usize {
        let id = self.groups.len();
        self.groups.push(before);
        id
    }
    pub fn predecessors(&self, after: OperationId) -> impl Iterator<Item = OperationId> + '_ {
        self.waits.get(&after).into_iter().flatten().flat_map(|&g| self.groups[g].iter().copied())
    }
    // Expand pairs only at the boundary to the current egglog relation schema.
    // Rust dependency analysis and topological sorting retain the compact gates.
    pub fn pairs<'a>(
        &'a self,
        included: &'a BTreeSet<OperationId>,
    ) -> impl Iterator<Item = (OperationId, OperationId)> + 'a {
        self.waits.keys().filter(|op| included.contains(op)).flat_map(|&after| {
            self.predecessors(after).filter(|op| included.contains(op)).map(move |before| (before, after))
        })
    }
}

// Connect consecutive effect barriers and the movable runs between them.
// Equal source positions remain unordered, including after cloning/fusion.
pub(super) fn effects(data: &Ir, live: &BTreeSet<OperationId>, movable: &BTreeSet<OperationId>) -> Effects {
    let mut effects = Effects::default();
    for region in data.regions.values() {
        let mut groups: BTreeMap<usize, Vec<OperationId>> = BTreeMap::new();
        for &op in region.members.iter().filter(|op| live.contains(op)) {
            groups.entry(data.operations[op].source_position).or_default().push(op);
        }
        let mut barrier = None;
        let mut pending = vec![];
        for group in groups.values() {
            if let Some(gate) = barrier {
                for &after in group {
                    effects.waits.entry(after).or_default().push(gate);
                }
            }
            let next: Vec<_> = group.iter().copied().filter(|op| !movable.contains(op)).collect();
            if !next.is_empty() {
                if !pending.is_empty() {
                    let gate = effects.gate(std::mem::take(&mut pending));
                    for &after in &next {
                        effects.waits.entry(after).or_default().push(gate);
                    }
                }
                barrier = Some(effects.gate(next));
            }
            pending.extend(group.iter().copied().filter(|op| movable.contains(op)));
        }
    }
    effects
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Node {
    Operation(OperationId),
    Gate(usize),
}

/// Keep gates explicit: expanding their fan-in/fan-out to operation pairs can
/// require quadratic space. Cross-region data edges are invocation dependencies,
/// not ordering constraints within either region.
pub(super) fn predecessors(dependencies: &Dependencies, data: &Ir, node: Node, out: &mut Vec<Node>) {
    match node {
        Node::Operation(op) => {
            out.extend(
                dependencies
                    .data_predecessors
                    .get(&op)
                    .into_iter()
                    .flatten()
                    .filter(|&&p| data.operations[p].region == data.operations[op].region)
                    .copied()
                    .map(Node::Operation),
            );
            out.extend(dependencies.effects.waits.get(&op).into_iter().flatten().copied().map(Node::Gate));
        }
        Node::Gate(g) => out.extend(dependencies.effects.groups[g].iter().copied().map(Node::Operation)),
    }
}

pub(super) fn schedules(
    dependencies: &Dependencies,
    data: &Ir,
) -> Result<BTreeMap<RegionId, Vec<OperationId>>, OptimizeError> {
    // Ready gates precede operations, matching immediate release of their
    // waiters. Ready operations retain their deterministic OperationId order.
    let nodes = (0..dependencies.effects.groups.len())
        .map(Node::Gate)
        .chain(dependencies.live.iter().copied().map(Node::Operation));
    let ordered = wyn_graph::topo_sort_by_dependencies(nodes, |node, out| {
        predecessors(dependencies, data, node, out)
    })
    .map_err(|_| OptimizeError::Extraction("cycle in the selected execution graph".into()))?;
    let mut regions = BTreeMap::<RegionId, Vec<OperationId>>::new();
    for node in ordered {
        if let Node::Operation(op) = node {
            regions.entry(data.operations[op].region).or_default().push(op);
        }
    }
    Ok(regions)
}

#[cfg(test)]
#[path = "order_tests.rs"]
mod tests;
