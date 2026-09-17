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
#[path = "traversal_tests.rs"]
mod tests;
