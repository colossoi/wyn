use super::{summarize, Evaluation, Node, Summary, OVER_BUDGET};
use crate::egglog::data::ExprId;
use std::collections::BTreeMap;

fn node(id: u32) -> Node {
    Node::Expr(ExprId::from(id))
}

fn evaluation(work: u8, children: &[u32]) -> Evaluation {
    Evaluation {
        children: children.iter().copied().map(node).collect(),
        backings: vec![],
        reads: vec![],
        local: Summary {
            work,
            ..Summary::default()
        },
        count_children: true,
    }
}

#[test]
fn duplication_counts_each_use_and_saturates_above_eight() {
    let evaluations = BTreeMap::from([
        (node(0), evaluation(4, &[])),
        (node(1), evaluation(0, &[0, 0])),
        (node(2), evaluation(1, &[1])),
        (node(3), evaluation(1, &[2, 2])),
    ]);
    let result = summarize(&evaluations);
    assert_eq!(result[&node(1)].work, 8);
    assert_eq!(result[&node(2)].work, OVER_BUDGET);
    assert_eq!(result[&node(3)].work, OVER_BUDGET);
}

#[test]
fn stored_results_and_metadata_keep_availability_without_repeating_producer_work() {
    let mut producer = evaluation(OVER_BUDGET, &[]);
    producer.local.device = true;
    let mut stored = evaluation(0, &[0]);
    stored.count_children = false;
    let mut length = evaluation(1, &[0]);
    length.count_children = false;
    let evaluations = BTreeMap::from([(node(0), producer), (node(1), stored), (node(2), length)]);
    let result = summarize(&evaluations);
    assert_eq!(result[&node(1)].work, 0);
    assert_eq!(result[&node(2)].work, 1);
    assert!(result[&node(1)].device);
    assert!(result[&node(2)].device);
}

#[test]
fn a_mutable_read_through_forwarded_aliases_exhausts_the_budget() {
    let mut source = evaluation(0, &[]);
    source.local.mutable = true;
    let mut alias = evaluation(0, &[2]);
    alias.backings.push(node(2));
    let mut read = evaluation(1, &[1]);
    read.reads.push(node(1));
    // The reader is visited before its aliases; changed facts must reach it.
    let evaluations = BTreeMap::from([(node(0), read), (node(1), alias), (node(2), source)]);
    assert_eq!(summarize(&evaluations)[&node(0)].work, OVER_BUDGET);
}

#[test]
fn recursive_evaluations_reach_a_bounded_fixed_point() {
    let mut recursive = evaluation(1, &[1]);
    recursive.local.device = true;
    let evaluations = BTreeMap::from([
        (node(0), recursive),
        (node(1), evaluation(0, &[0])),
        (node(2), evaluation(0, &[2])),
    ]);
    let result = summarize(&evaluations);
    assert_eq!(result[&node(0)].work, OVER_BUDGET);
    assert_eq!(result[&node(1)].work, OVER_BUDGET);
    assert!(result[&node(1)].device);
    assert_eq!(result[&node(2)].work, 0);
    assert!(!result[&node(2)].device);
}
