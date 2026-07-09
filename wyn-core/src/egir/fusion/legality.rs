//! Read-only query layer over the semantic dependency DAG, used as the fusion
//! *legality oracle*. It indexes the edges that `collect_graph_dependencies`
//! already produces (`inner.semantic_dependencies`) — it computes no new edges
//! and owns no policy beyond "never fuse/reorder across a conflict".

use std::collections::{HashMap, HashSet};

use crate::egir::program::{SemanticDependency, SemanticDependencyKind, SemanticOpId};

/// An index over `&[SemanticDependency]` answering the questions fusion asks:
/// do two ops conflict, who consumes a producer's value, and is one op
/// transitively downstream of another.
pub struct SemanticGraph {
    /// Dense interning of every op that appears in any edge.
    index: HashMap<SemanticOpId, usize>,
    /// Value successors (consumers that read the producer's result).
    value_succ: Vec<Vec<usize>>,
    /// Unordered resource/effect reordering conflicts, stored both ways.
    conflict: HashSet<(usize, usize)>,
}

impl SemanticGraph {
    pub fn new(deps: &[SemanticDependency]) -> Self {
        let mut index: HashMap<SemanticOpId, usize> = HashMap::new();
        let mut intern = |op: &SemanticOpId| -> usize {
            if let Some(&i) = index.get(op) {
                return i;
            }
            let i = index.len();
            index.insert(op.clone(), i);
            i
        };

        let mut value_pairs = Vec::new();
        let mut conflict = HashSet::new();
        for dep in deps {
            let p = intern(&dep.producer);
            let c = intern(&dep.consumer);
            match dep.kind {
                SemanticDependencyKind::Value => value_pairs.push((p, c)),
                // Both explicit effect ordering and resource aliasing prohibit
                // moving another operation across this edge.  A directly
                // adjacent pair may still be fused in source order; callers use
                // this relation for the operations *between* that pair.
                SemanticDependencyKind::Effect | SemanticDependencyKind::Resource => {
                    conflict.insert((p, c));
                    conflict.insert((c, p));
                }
            }
        }

        let n = index.len();
        let mut value_succ = vec![Vec::new(); n];
        for (p, c) in value_pairs {
            value_succ[p].push(c);
        }

        Self {
            index,
            value_succ,
            conflict,
        }
    }

    /// Resource and Effect edges are both reordering conflicts. A caller may
    /// still combine a directly adjacent pair while preserving source order,
    /// but cannot move either operation across such an edge.
    pub fn conflicts(&self, a: &SemanticOpId, b: &SemanticOpId) -> bool {
        match (self.index.get(a), self.index.get(b)) {
            (Some(&i), Some(&j)) => self.conflict.contains(&(i, j)),
            _ => false,
        }
    }

    /// True iff `b` is transitively reachable from `a` along *value* edges —
    /// i.e. `a`'s result flows (directly or indirectly) into `b`, making them a
    /// producer/consumer chain rather than fusable siblings.
    pub fn reachable_between(&self, a: &SemanticOpId, b: &SemanticOpId) -> bool {
        let (Some(&start), Some(&target)) = (self.index.get(a), self.index.get(b)) else {
            return false;
        };
        wyn_graph::reaches_ordered(start, target, wyn_graph::WalkOrder::DepthFirst, |node, out| {
            out.extend(self.value_succ[node].iter().copied());
        })
    }

    /// Number of semantic operations that directly consume `producer`'s
    /// result.  Multiple uses inside one consumer count once because the DAG is
    /// operation-granular.
    pub fn value_consumer_count(&self, producer: &SemanticOpId) -> usize {
        self.index.get(producer).map(|&index| self.value_succ[index].len()).unwrap_or(0)
    }
}

#[cfg(test)]
#[path = "legality_tests.rs"]
mod legality_tests;
