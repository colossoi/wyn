//! Iterative traversal helpers for acyclic graphs and forests.
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Visit a DAG in dependency-first order, preserving root and child order.
///
/// `complete` prunes nodes whose summaries the caller already has, including
/// their descendants. Shared nodes are returned once. Each call costs O(V + E)
/// in the newly visited graph; callers can extend a cache between calls without
/// walking previously summarized subgraphs. The caller must supply an acyclic
/// graph. Uses an explicit stack, so long dependency chains cannot overflow it.
pub fn dag_postorder<N, I, P, F>(roots: I, mut complete: P, mut children: F) -> Vec<N>
where
    N: Copy + Eq + Hash,
    I: IntoIterator<Item = N>,
    P: FnMut(N) -> bool,
    F: FnMut(N, &mut Vec<N>),
{
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    let mut pending = Vec::new();
    let mut next = Vec::new();
    for root in roots {
        pending.push((root, false));
        while let Some((node, exiting)) = pending.pop() {
            if exiting {
                result.push(node);
            } else if !complete(node) && seen.insert(node) {
                pending.push((node, true));
                next.clear();
                children(node, &mut next);
                pending.extend(next.iter().rev().map(|&child| (child, false)));
            }
        }
    }
    result
}

/// Half-open preorder range occupied by a subtree. Compare intervals only
/// within the same forest indexing. A node's own position is `start`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DfsInterval {
    pub start: usize,
    pub end: usize,
}

impl DfsInterval {
    /// No restriction on the preorder position.
    pub const ANY: Self = Self {
        start: 0,
        end: usize::MAX,
    };

    pub fn contains(self, point: usize) -> bool {
        self.start <= point && point < self.end
    }

    pub fn intersect(self, other: Self) -> Option<Self> {
        let result = Self {
            start: self.start.max(other.start),
            end: self.end.min(other.end),
        };
        (result.start < result.end).then_some(result)
    }
}

/// Index a rooted forest in O(V + E) time and O(V) space. Roots and children
/// are visited in the supplied order. Every node must have at most one parent;
/// this indexes a tree supplied by the caller, rather than computing dominators.
pub fn forest_intervals<N, I, F>(roots: I, mut children: F) -> HashMap<N, DfsInterval>
where
    N: Copy + Eq + Hash,
    I: IntoIterator<Item = N>,
    F: FnMut(N, &mut Vec<N>),
{
    let mut intervals = HashMap::<N, DfsInterval>::new();
    let mut position = 0;
    let mut pending = Vec::new();
    let mut next = Vec::new();
    for root in roots {
        pending.push((root, false));
        while let Some((node, exiting)) = pending.pop() {
            if exiting {
                if let Some(interval) = intervals.get_mut(&node) {
                    interval.end = position;
                }
            } else if let std::collections::hash_map::Entry::Vacant(entry) = intervals.entry(node) {
                entry.insert(DfsInterval {
                    start: position,
                    end: position,
                });
                position += 1;
                pending.push((node, true));
                next.clear();
                children(node, &mut next);
                pending.extend(next.iter().rev().map(|&child| (child, false)));
            }
        }
    }
    intervals
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
