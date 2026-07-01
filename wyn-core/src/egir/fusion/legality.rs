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
    /// Unordered resource-conflict pairs, stored both ways.
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
                // Effect edges are program *ordering*, not aliasing and not a
                // value chain. Two effect-adjacent ops that touch no common
                // binding may be fused (the fused op performs both effects in
                // order), so effects gate neither conflict nor reachability.
                SemanticDependencyKind::Effect => {}
                SemanticDependencyKind::Resource => {
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

    /// Two ops conflict if they share a binding with a non-Read access (a
    /// `Resource` edge) — i.e. they alias. Fusing across a resource conflict is
    /// never legal. Effect *ordering* alone is not a conflict. An op with no
    /// edges conflicts with nothing.
    pub fn conflicts(&self, a: &SemanticOpId, b: &SemanticOpId) -> bool {
        match (self.index.get(a), self.index.get(b)) {
            (Some(&i), Some(&j)) => self.conflict.contains(&(i, j)),
            _ => false,
        }
    }

    /// Number of value consumers of `producer` (cheaper than `value_consumers`).
    pub fn value_consumer_count(&self, producer: &SemanticOpId) -> usize {
        self.index.get(producer).map_or(0, |&i| self.value_succ[i].len())
    }

    /// True iff `b` is transitively reachable from `a` along *value* edges —
    /// i.e. `a`'s result flows (directly or indirectly) into `b`, making them a
    /// producer/consumer chain rather than fusable siblings.
    pub fn reachable_between(&self, a: &SemanticOpId, b: &SemanticOpId) -> bool {
        let (Some(&start), Some(&target)) = (self.index.get(a), self.index.get(b)) else {
            return false;
        };
        let mut seen = HashSet::new();
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if node == target {
                return true;
            }
            if seen.insert(node) {
                stack.extend(self.value_succ[node].iter().copied());
            }
        }
        false
    }
}

#[cfg(test)]
#[path = "legality_tests.rs"]
mod legality_tests;
