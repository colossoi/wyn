use crate::{ReplacementError, ReplacementForest};
use std::collections::HashMap;

#[test]
fn unknown_ids_are_survivors_and_are_not_exported() {
    let mut forest = ReplacementForest::new();
    assert_eq!(forest.resolve(7), 7);
    assert_eq!(forest.resolve(8), 8);
    assert!(forest.into_map().is_empty());
}

#[test]
fn ids_need_neither_default_nor_debug() {
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    struct Id(u8);

    let mut forest = ReplacementForest::default();
    assert!(forest.replace(Id(0), Id(1)).is_ok());
    assert!(forest.resolve(Id(0)) == Id(1));
}

#[test]
fn chains_flatten_in_either_insertion_order() {
    for edges in [[(0, 1), (1, 2), (2, 3)], [(2, 3), (1, 2), (0, 1)]] {
        let mut forest = ReplacementForest::new();
        for (old, survivor) in edges {
            assert_eq!(forest.replace(old, survivor), Ok(true));
        }
        assert_eq!(forest.into_map(), HashMap::from([(0, 3), (1, 3), (2, 3)]));
    }
}

#[test]
fn replacing_a_shared_survivor_forwards_all_sources() {
    let mut forest = ReplacementForest::new();
    for old in 0..3 {
        assert_eq!(forest.replace(old, 3), Ok(true));
    }
    assert_eq!(forest.resolve(0), 3);
    assert_eq!(forest.replace(3, 4), Ok(true));
    for old in 0..4 {
        assert_eq!(forest.resolve(old), 4);
    }
    assert_eq!(forest.into_map(), (0..4).map(|old| (old, 4)).collect());
}

#[test]
fn canonically_identical_assignments_are_unchanged() {
    let mut forest = ReplacementForest::new();
    for (old, survivor) in [(0, 1), (1, 2), (3, 2)] {
        assert_eq!(forest.replace(old, survivor), Ok(true));
    }
    for survivor in [1, 2, 3] {
        assert_eq!(forest.replace(0, survivor), Ok(false));
    }
    assert_eq!(forest.into_map(), HashMap::from([(0, 2), (1, 2), (3, 2)]));
}

#[test]
fn conflicts_preserve_both_chains_without_redirecting_survivors() {
    let mut forest = ReplacementForest::new();
    for (old, survivor) in [(0, 1), (1, 2), (3, 4)] {
        assert_eq!(forest.replace(old, survivor), Ok(true));
    }
    assert_eq!(
        forest.replace(0, 3),
        Err(ReplacementError::Conflict {
            old: 0,
            survivor: 3,
            existing: 2
        })
    );
    for (id, survivor) in [(0, 2), (1, 2), (2, 2), (3, 4), (4, 4)] {
        assert_eq!(forest.resolve(id), survivor);
    }
    assert_eq!(forest.into_map(), HashMap::from([(0, 2), (1, 2), (3, 4)]));
}

#[test]
fn direct_cycles_are_rejected_even_for_an_assigned_source() {
    let mut forest = ReplacementForest::new();
    assert_eq!(
        forest.replace(0, 0),
        Err(ReplacementError::Cycle { old: 0, survivor: 0 })
    );
    assert_eq!(forest.resolve(0), 0);
    assert_eq!(forest.replace(0, 1), Ok(true));
    assert_eq!(
        forest.replace(0, 0),
        Err(ReplacementError::Cycle { old: 0, survivor: 0 })
    );
    assert_eq!(forest.into_map(), HashMap::from([(0, 1)]));
}

#[test]
fn indirect_cycles_are_rejected_before_and_after_compression() {
    for compress in [false, true] {
        let mut forest = ReplacementForest::new();
        assert_eq!(forest.replace(0, 1), Ok(true));
        assert_eq!(forest.replace(1, 2), Ok(true));
        if compress {
            assert_eq!(forest.resolve(0), 2);
        }
        assert_eq!(
            forest.replace(2, 0),
            Err(ReplacementError::Cycle { old: 2, survivor: 0 })
        );
        assert_eq!(forest.resolve(2), 2);
        assert_eq!(forest.into_map(), HashMap::from([(0, 2), (1, 2)]));
    }
}

#[test]
fn long_chains_resolve_and_export_iteratively() {
    let mut forest = ReplacementForest::new();
    let terminal = 100_000;
    for old in 0..terminal {
        assert_eq!(forest.replace(old, old + 1), Ok(true));
    }
    assert_eq!(forest.resolve(0), terminal);
    let exported = forest.into_map();
    assert_eq!(exported.len(), terminal);
    for old in 0..terminal {
        assert_eq!(exported[&old], terminal);
    }
    assert!(!exported.contains_key(&terminal));
}

#[test]
fn export_preserves_disjoint_survivors_without_intermediate_targets() {
    let mut forest = ReplacementForest::new();
    for (old, survivor) in [(0, 1), (1, 2), (3, 4), (4, 5), (6, 4)] {
        assert_eq!(forest.replace(old, survivor), Ok(true));
    }
    assert_eq!(forest.resolve(7), 7);
    let exported = forest.into_map();
    assert_eq!(exported, HashMap::from([(0, 2), (1, 2), (3, 5), (4, 5), (6, 5)]));
    assert!(exported.values().all(|target| !exported.contains_key(target)));
}
