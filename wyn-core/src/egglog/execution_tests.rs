use super::{summarize, Evaluation, Node, Summary};
use crate::egglog::data::ExprId;
use std::collections::BTreeMap;

fn node(id: u32) -> Node {
    Node::Expr(ExprId::from(id))
}

fn evaluation(children: &[u32]) -> Evaluation {
    Evaluation {
        children: children.iter().copied().map(node).collect(),
        backings: vec![],
        reads: vec![],
        local: Summary::default(),
        count_children: true,
    }
}

#[test]
fn stored_results_and_metadata_keep_availability_without_repeating_producer_work() {
    let mut producer = evaluation(&[]);
    producer.local.device = true;
    producer.local.duplication_blocked = true;
    let mut stored = evaluation(&[0]);
    stored.count_children = false;
    let mut length = evaluation(&[0]);
    length.count_children = false;
    let evaluations = BTreeMap::from([(node(0), producer), (node(1), stored), (node(2), length)]);
    let result = summarize(&evaluations);
    assert!(!result[&node(1)].duplication_blocked);
    assert!(!result[&node(2)].duplication_blocked);
    assert!(result[&node(1)].device);
    assert!(result[&node(2)].device);
}

#[test]
fn a_mutable_read_through_forwarded_aliases_blocks_duplication_independently_of_cost() {
    let mut source = evaluation(&[]);
    source.local.mutable = true;
    let mut alias = evaluation(&[2]);
    alias.backings.push(node(2));
    let mut read = evaluation(&[1]);
    read.reads.push(node(1));
    // The reader is visited before its aliases; changed facts must reach it.
    let evaluations = BTreeMap::from([(node(0), read), (node(1), alias), (node(2), source)]);
    let result = summarize(&evaluations);
    assert!(result[&node(0)].duplication_blocked);
}

#[test]
fn duplication_restrictions_propagate_until_a_stored_boundary() {
    let mut restricted = evaluation(&[]);
    restricted.local.duplication_blocked = true;
    let mut stored = evaluation(&[0]);
    stored.count_children = false;
    let evaluations = BTreeMap::from([
        (node(0), restricted),
        (node(1), evaluation(&[0])),
        (node(2), evaluation(&[])),
        (node(3), stored),
    ]);
    let result = summarize(&evaluations);
    assert!(result[&node(1)].duplication_blocked);
    assert!(!result[&node(2)].duplication_blocked);
    assert!(!result[&node(3)].duplication_blocked);
}

#[test]
fn recursive_evaluations_reach_a_bounded_fixed_point() {
    let mut recursive = evaluation(&[1]);
    recursive.local.device = true;
    let evaluations = BTreeMap::from([
        (node(0), recursive),
        (node(1), evaluation(&[0])),
        (node(2), evaluation(&[2])),
    ]);
    let result = summarize(&evaluations);
    assert!(result[&node(1)].device);
    assert!(!result[&node(2)].device);
}
