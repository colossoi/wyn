use super::super::super::data::OperationData;
use super::{effects, predecessors, Dependencies, Effects, Node};
use crate::compile_thru_tlc;
use crate::egglog::data::{Ir, OperationId};
use crate::egglog::from_tlc;
use crate::tlc::infer_input_slice_bounds;
use std::collections::{BTreeMap, BTreeSet};

fn graph(n: usize) -> (Ir, Vec<OperationId>) {
    let tlc = compile_thru_tlc("entry main(xs:[4]i32) [4]i32=map(|x:i32|x+1,xs)").unwrap();
    let mut data = from_tlc(&infer_input_slice_bounds(tlc)).unwrap().ir;
    let entry = data.definitions[data.entries.values().next().unwrap().definition].body;
    let template = data.operations[*data.regions[entry].members.first().unwrap()].clone();
    data.regions[entry].members.clear();
    let ids: Vec<_> = (0..n)
        .map(|position| {
            data.operations.alloc(OperationData {
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
    let dependencies = Dependencies {
        live,
        effects,
        data_predecessors: BTreeMap::new(),
        movable: BTreeSet::new(),
        discardable: BTreeSet::new(),
        safe_regions: BTreeSet::new(),
    };
    let sorted = dependencies.schedules(&data).unwrap();
    assert_eq!(sorted[&data.operations[ids[0]].region], ids);
    let mut reached = Vec::new();
    dependencies.walk_dependencies(&data, *ids.last().unwrap(), |op| reached.push(op));
    assert_eq!(reached.len(), ids.len() / 2 + 1);
    let mut edges = 0;
    wyn_graph::for_each_reachable(
        ids.iter().copied().map(Node::Operation),
        wyn_graph::WalkOrder::DepthFirst,
        |node, out| {
            predecessors(&dependencies, &data, node, out);
            edges += out.len();
        },
        |_| {},
    );
    assert!(edges <= 2 * ids.len(), "shared gates are expanded once per walk");
}

#[test]
fn ready_gates_release_lower_id_operations_immediately_and_detect_cycles() {
    let (data, ids) = graph(3);
    let mut dependencies = Dependencies {
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
        dependencies.schedules(&data).unwrap()[&data.operations[ids[0]].region],
        [ids[1], ids[0], ids[2]]
    );
    dependencies.effects.groups[0] = vec![ids[0]];
    assert!(dependencies.schedules(&data).is_err());
}
