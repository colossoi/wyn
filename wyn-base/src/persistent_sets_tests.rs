use super::{Sets, EMPTY};
use std::collections::BTreeSet;

#[test]
fn persistent_set_algebra_and_long_prefix_sharing() {
    let mut s = Sets::default();
    let mut a = EMPTY;
    let mut b = EMPTY;
    for i in 0..4096 {
        let before = s.nodes.len();
        a = s.insert(a, i);
        if i % 3 == 0 {
            b = s.insert(b, i);
        }
        assert!(
            s.nodes.len() - before <= 56,
            "two insertions allocate at most a leaf and 26 radix branches each"
        );
    }
    let common = s.intersection(a, b);
    let rest = s.difference(a, b);
    assert_eq!(common, b);
    assert_eq!(s.union(common, rest), a);
    assert_eq!(
        s.iter(rest).collect::<Vec<_>>(),
        (0..4096).filter(|i| i % 3 != 0).collect::<Vec<_>>()
    );
    let high = s.insert(a, u32::MAX);
    assert_eq!(s.iter(high).last(), Some(u32::MAX));
    assert_eq!(s.iter(a).count(), 4096, "old versions stay unchanged");
}

#[test]
fn compressed_sets_match_sorted_sets_for_dense_and_sparse_ids() {
    let mut state = 17u32;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut sets = Sets::default();
    for case in 0..128 {
        let values: Vec<BTreeSet<u32>> = (0..2)
            .map(|_| {
                (0..case % 64)
                    .map(|_| {
                        let r = random();
                        match case % 4 {
                            0 => r % 128,
                            1 => r % 4096,
                            2 => r | 0xffff_0000,
                            _ => r,
                        }
                    })
                    .collect()
            })
            .collect();
        let a = values[0].iter().fold(EMPTY, |s, &v| sets.insert(s, v));
        let b = values[1].iter().rev().fold(EMPTY, |s, &v| sets.insert(s, v));
        let reordered = values[0].iter().rev().fold(EMPTY, |s, &v| sets.insert(s, v));
        assert_eq!(a, reordered, "insertion order cannot change identity");
        let union = sets.union(a, b);
        let intersection = sets.intersection(a, b);
        let difference = sets.difference(a, b);
        assert_eq!(
            sets.iter(union).collect::<BTreeSet<_>>(),
            values[0].union(&values[1]).copied().collect()
        );
        assert_eq!(
            sets.iter(intersection).collect::<BTreeSet<_>>(),
            values[0].intersection(&values[1]).copied().collect()
        );
        assert_eq!(
            sets.iter(difference).collect::<BTreeSet<_>>(),
            values[0].difference(&values[1]).copied().collect()
        );
    }
}
