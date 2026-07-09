use super::*;
use crate::egir::types::NodeId;
use slotmap::SlotMap;

fn op(scope: &str, result: NodeId) -> SemanticOpId {
    SemanticOpId {
        scope: scope.into(),
        result,
    }
}

fn dep(
    producer: &SemanticOpId,
    consumer: &SemanticOpId,
    kind: SemanticDependencyKind,
) -> SemanticDependency {
    SemanticDependency {
        producer: producer.clone(),
        consumer: consumer.clone(),
        kind,
    }
}

#[test]
fn oracle_distinguishes_conflict_flow_and_value_edges() {
    let mut sm: SlotMap<NodeId, ()> = SlotMap::with_key();
    let (r1, r2, r3) = (sm.insert(()), sm.insert(()), sm.insert(()));
    let a = op("e", r1);
    let b = op("e", r2);
    let c = op("e", r3);

    // a --Resource--> b, b --Effect--> c, a --Value--> c
    let deps = vec![
        dep(&a, &b, SemanticDependencyKind::Resource),
        dep(&b, &c, SemanticDependencyKind::Effect),
        dep(&a, &c, SemanticDependencyKind::Value),
    ];
    let g = SemanticGraph::new(&deps);

    // Resource and effect edges are reordering conflicts (both directions);
    // value edges are represented separately.
    assert!(g.conflicts(&a, &b), "resource edge is a conflict");
    assert!(g.conflicts(&b, &a));
    assert!(g.conflicts(&b, &c), "effect ordering is a reordering conflict");
    assert!(
        !g.conflicts(&a, &c),
        "a value edge alone is fusable, not a conflict"
    );

    // Reachability follows value edges only, not effect ordering or resource.
    assert!(g.reachable_between(&a, &c), "a --Value--> c");
    assert!(
        !g.reachable_between(&b, &c),
        "b --Effect--> c is ordering, not a value chain"
    );
    assert!(
        !g.reachable_between(&a, &b),
        "a--b is resource-only, not a value edge"
    );
    assert!(!g.reachable_between(&c, &a), "no back edge");
    assert_eq!(g.value_consumer_count(&a), 1);
    assert_eq!(g.value_consumer_count(&b), 0);
}

#[test]
fn unknown_ops_have_no_edges() {
    let mut sm: SlotMap<NodeId, ()> = SlotMap::with_key();
    let (r1, r2) = (sm.insert(()), sm.insert(()));
    let a = op("e", r1);
    let lonely = op("e", r2);
    let g = SemanticGraph::new(&[dep(&a, &a, SemanticDependencyKind::Value)]);
    assert!(!g.conflicts(&lonely, &a));
    assert!(!g.reachable_between(&lonely, &a));
}
