use super::*;

/// A gate waits for a group of operations. Consumers depend on the gate rather
/// than on every group member, so even tied source positions need O(n) edges.
#[derive(Default)]
pub(in crate::egglog) struct Effects {
    groups: Vec<Vec<OperationId>>,
    waits: BTreeMap<OperationId, Vec<usize>>,
}
impl Effects {
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
pub(super) fn effects(
    data: &AssociatedData,
    live: &BTreeSet<OperationId>,
    movable: &BTreeSet<OperationId>,
) -> Effects {
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
pub(super) fn predecessors(snapshot: &Snapshot, data: &AssociatedData, node: Node, out: &mut Vec<Node>) {
    match node {
        Node::Operation(op) => {
            out.extend(
                snapshot
                    .data_predecessors
                    .get(&op)
                    .into_iter()
                    .flatten()
                    .filter(|&&p| data.operations[p].region == data.operations[op].region)
                    .copied()
                    .map(Node::Operation),
            );
            out.extend(snapshot.effects.waits.get(&op).into_iter().flatten().copied().map(Node::Gate));
        }
        Node::Gate(g) => out.extend(snapshot.effects.groups[g].iter().copied().map(Node::Operation)),
    }
}

pub(super) fn schedules(
    snapshot: &Snapshot,
    data: &AssociatedData,
) -> Result<BTreeMap<RegionId, Vec<OperationId>>, OptimizeError> {
    // Ready gates precede operations, matching immediate release of their
    // waiters. Ready operations retain their deterministic OperationId order.
    let nodes = (0..snapshot.effects.groups.len())
        .map(Node::Gate)
        .chain(snapshot.live.iter().copied().map(Node::Operation));
    let ordered =
        wyn_graph::topo_sort_by_dependencies(nodes, |node, out| predecessors(snapshot, data, node, out))
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
mod tests {
    use super::*;
    fn graph(n: usize) -> (AssociatedData, Vec<OperationId>) {
        let tlc = crate::compile_thru_tlc("entry main(xs:[4]i32) [4]i32=map(|x:i32|x+1,xs)").unwrap();
        let mut data =
            crate::egglog::convert_program(&crate::tlc::infer_input_slice_bounds(tlc)).unwrap().data;
        let entry = data.definitions[data.entries.values().next().unwrap().definition].body;
        let template = data.operations[*data.regions[entry].members.first().unwrap()].clone();
        data.regions[entry].members.clear();
        let ids: Vec<_> = (0..n)
            .map(|position| {
                data.operations.alloc(super::super::super::data::OperationData {
                    source_position: position,
                    ..template.clone()
                })
            })
            .collect();
        data.regions[entry].members.extend(&ids);
        (data, ids)
    }
    fn closure(
        ids: &[OperationId],
        edges: &BTreeSet<(OperationId, OperationId)>,
    ) -> BTreeSet<(OperationId, OperationId)> {
        let mut result = BTreeSet::new();
        for &start in ids {
            let mut pending = vec![start];
            while let Some(a) = pending.pop() {
                for &(_, b) in edges.iter().filter(|&&(a0, _)| a0 == a) {
                    if result.insert((start, b)) {
                        pending.push(b);
                    }
                }
            }
        }
        result
    }
    #[test]
    fn sparse_effect_frontiers_preserve_all_orders_including_tied_positions() {
        let (mut data, ids) = graph(6);
        let live = ids.iter().copied().collect();
        for positions in [[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 2, 3]] {
            for (&op, pos) in ids.iter().zip(positions) {
                data.operations[op].source_position = pos;
            }
            for mask in 0..64 {
                let movable = ids
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &op)| (mask & (1 << i) != 0).then_some(op))
                    .collect::<BTreeSet<_>>();
                let mut dense = BTreeSet::new();
                for &a in &ids {
                    for &b in &ids {
                        if data.operations[a].source_position < data.operations[b].source_position
                            && (!movable.contains(&a) || !movable.contains(&b))
                        {
                            dense.insert((a, b));
                        }
                    }
                }
                assert_eq!(
                    closure(&ids, &effects(&data, &live, &movable).pairs(&live).collect()),
                    closure(&ids, &dense)
                );
            }
        }
    }
    #[test]
    fn distinct_source_positions_need_only_linear_effect_edges() {
        let (data, ids) = graph(4096);
        let live = ids.iter().copied().collect();
        assert_eq!(
            effects(&data, &live, &BTreeSet::new()).pairs(&live).count(),
            ids.len() - 1
        );
        assert_eq!(effects(&data, &live, &live).pairs(&live).count(), 0);
        let movable = ids.iter().enumerate().filter_map(|(i, &op)| (i % 3 != 0).then_some(op)).collect();
        assert!(effects(&data, &live, &movable).pairs(&live).count() <= 3 * ids.len());
    }

    #[test]
    fn large_tied_groups_sort_through_linear_sized_gates() {
        let (mut data, ids) = graph(8192);
        for (i, &op) in ids.iter().enumerate() {
            data.operations[op].source_position = usize::from(i >= ids.len() / 2);
        }
        let live = ids.iter().copied().collect();
        let effects = effects(&data, &live, &BTreeSet::new());
        let storage = effects.groups.len()
            + effects.groups.iter().map(Vec::len).sum::<usize>()
            + effects.waits.values().map(Vec::len).sum::<usize>();
        assert!(
            storage <= 4 * ids.len(),
            "no quadratic operation pairs inside the sorter"
        );
        let snapshot = Snapshot {
            live,
            effects,
            data_predecessors: BTreeMap::new(),
            movable: BTreeSet::new(),
            discardable: BTreeSet::new(),
            safe_regions: BTreeSet::new(),
        };
        let sorted = snapshot.schedules(&data).unwrap();
        assert_eq!(sorted[&data.operations[ids[0]].region], ids);
        let mut reached = Vec::new();
        snapshot.walk_dependencies(&data, *ids.last().unwrap(), |op| reached.push(op));
        assert_eq!(reached.len(), ids.len() / 2 + 1);
        let mut edges = 0;
        wyn_graph::for_each_reachable(
            ids.iter().copied().map(Node::Operation),
            wyn_graph::WalkOrder::DepthFirst,
            |node, out| {
                predecessors(&snapshot, &data, node, out);
                edges += out.len();
            },
            |_| {},
        );
        assert!(edges <= 2 * ids.len(), "shared gates are expanded once per walk");
    }

    #[test]
    fn ready_gates_release_lower_id_operations_immediately_and_detect_cycles() {
        let (data, ids) = graph(3);
        let mut snapshot = Snapshot {
            live: ids.iter().copied().collect(),
            data_predecessors: BTreeMap::new(),
            effects: Effects {
                groups: vec![vec![ids[1]]],
                waits: [(ids[0], vec![0])].into_iter().collect(),
            },
            movable: BTreeSet::new(),
            discardable: BTreeSet::new(),
            safe_regions: BTreeSet::new(),
        };
        assert_eq!(
            snapshot.schedules(&data).unwrap()[&data.operations[ids[0]].region],
            [ids[1], ids[0], ids[2]]
        );
        snapshot.effects.groups[0] = vec![ids[0]];
        assert!(snapshot.schedules(&data).is_err());
    }
}
