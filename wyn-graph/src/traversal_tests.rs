use super::{dag_postorder, forest_intervals, reachable_targets, DfsInterval};
use crate::{reachable_set, WalkOrder};
use std::collections::HashSet;

#[test]
fn target_propagation_matches_individual_walks_through_cycles_and_shared_paths() {
    let targets: HashSet<_> = [2, 7, 10, 13].into_iter().collect();
    for seed in 0..64 {
        let edges: Vec<_> = (0..12)
            .flat_map(|from| {
                (0..12).filter_map(move |to| {
                    ((from * 37 + to * 19 + seed * 11) % 29 < 4).then_some((from, to))
                })
            })
            .collect();
        let actual = reachable_targets(edges.iter().copied(), [2, 7, 10, 13, 2]);
        for root in 0..14 {
            let reached = reachable_set([root], WalkOrder::DepthFirst, |node, out| {
                out.extend(edges.iter().filter_map(|&(from, to)| (from == node).then_some(to)));
            });
            let expected: HashSet<_> = reached.intersection(&targets).copied().collect();
            assert_eq!(
                actual.get(&root).cloned().unwrap_or_default(),
                expected,
                "seed {seed}, root {root}"
            );
        }
    }
}

#[test]
fn shared_dag_is_bottom_up_and_cached_subgraphs_are_not_revisited() {
    let edges = [vec![], vec![0], vec![0, 1], vec![1, 2, 1], vec![3, 2]];
    let mut complete = HashSet::new();
    let first = dag_postorder(
        [3, 2, 3],
        |n| complete.contains(&n),
        |n, out| out.extend(&edges[n]),
    );
    assert_eq!(first, [0, 1, 2, 3]);
    complete.extend(first);
    let mut expanded = Vec::new();
    let next = dag_postorder(
        [4, 3],
        |n| complete.contains(&n),
        |n, out| {
            expanded.push(n);
            out.extend(&edges[n]);
        },
    );
    assert_eq!(next, [4]);
    assert_eq!(expanded, [4]);
}

#[test]
fn forest_intervals_distinguish_siblings_and_separate_roots() {
    let edges = [vec![1, 3], vec![2], vec![], vec![], vec![5], vec![]];
    let bounds = forest_intervals([0, 4], |n, out| out.extend(&edges[n]));
    assert_eq!(bounds[&0], DfsInterval { start: 0, end: 4 });
    assert_eq!(bounds[&1], DfsInterval { start: 1, end: 3 });
    assert!(bounds[&1].contains(bounds[&2].start));
    assert_eq!(bounds[&1].intersect(bounds[&3]), None);
    assert_eq!(bounds[&0].intersect(bounds[&4]), None);
    assert_eq!(DfsInterval::ANY.intersect(bounds[&4]), Some(bounds[&4]));
}

#[test]
fn long_chains_use_an_explicit_stack() {
    let children = |n, out: &mut Vec<usize>| {
        if n > 0 {
            out.push(n - 1)
        }
    };
    let order = dag_postorder([100_000], |_| false, children);
    assert!(order.iter().copied().eq(0..=100_000));
    let bounds = forest_intervals([100_000], children);
    assert_eq!(
        bounds[&100_000],
        DfsInterval {
            start: 0,
            end: 100_001
        }
    );
}
