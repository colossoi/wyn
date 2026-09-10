use super::*;
use Definition::*;

fn graph() -> Graph<u32, u32, u32> {
    Graph::new(
        [
            (0, Input),
            (1, Produced(10)),
            (2, Produced(10)),
            (3, Pure(vec![2])),
            (4, Pure(vec![1])),
            (5, Flow(vec![1, 6])),
            (6, Pure(vec![5])),
        ],
        [(10, vec![0])],
        [(20, vec![3])],
    )
}

#[test]
fn sibling_results_are_outputs_only_when_observed() {
    let graph = graph();
    let slice = graph.select([1], [], &HashSet::from([0]), |_| true, |_| true, &[]).unwrap();
    assert_eq!(slice.operations(), &HashSet::from([10]));
    assert_eq!(slice.inputs(), &HashSet::from([0]));
    assert_eq!(slice.live_outs(), &[2]);
    assert!(!slice.values().contains(&3));
    assert!(!slice.values().contains(&4));
    let slice = graph.select([1], [], &HashSet::from([0]), |_| true, |_| false, &[]).unwrap();
    assert!(slice.live_outs().is_empty());
    assert_eq!(slice.outputs().copied().collect::<Vec<_>>(), vec![1]);
}

#[test]
fn cycles_close_and_explicit_inputs_recompute_the_closure() {
    let graph = graph();
    let slice = graph.select([5], [], &HashSet::from([0]), |_| true, |_| false, &[]).unwrap();
    assert!(slice.values().is_superset(&HashSet::from([0, 1, 5, 6])));
    let cut = graph.select([6], [], &HashSet::from([5]), |_| true, |_| false, &[]).unwrap();
    assert_eq!(cut.values(), &HashSet::from([5, 6]));
    assert_eq!(cut.inputs(), &HashSet::from([5]));
    assert!(cut.operations().is_empty());
}

#[test]
fn forbidden_dependencies_and_missing_inputs_fail_construction() {
    let graph = graph();
    assert!(matches!(
        graph.select([1], [], &HashSet::new(), |_| true, |_| false, &[]),
        Err(Error::Unsupplied(0))
    ));
    assert!(matches!(
        graph.select(
            [1],
            [],
            &HashSet::from([0]),
            |node| node != Node::Operation(10),
            |_| false,
            &[]
        ),
        Err(Error::OutsideRegion(Node::Operation(10)))
    ));
    assert!(graph.select([1], [], &HashSet::from([1]), |_| false, |_| false, &[]).is_ok());
}

#[test]
fn external_roots_and_retained_operations_observe_results() {
    let graph = Graph::<_, _, u32>::new(
        [
            (0, Pure(vec![])),
            (1, Produced(10)),
            (2, Produced(10)),
            (3, Pure(vec![2])),
        ],
        [(10, vec![0]), (11, vec![3])],
        [],
    );
    let slice = graph.select([1], [], &HashSet::new(), |_| true, |_| true, &[]).unwrap();
    assert_eq!(slice.live_outs(), &[2]);
    let slice = graph
        .select(
            [1],
            [Node::Operation(11)],
            &HashSet::new(),
            |_| true,
            |_| false,
            &[],
        )
        .unwrap();
    assert!(slice.live_outs().is_empty());
    let slice = graph
        .select(
            [1],
            [Node::Operation(11)],
            &HashSet::new(),
            |_| true,
            |_| false,
            &[3],
        )
        .unwrap();
    assert_eq!(slice.live_outs(), &[2]);
}
