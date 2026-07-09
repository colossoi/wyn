//! Small graph algorithms for compiler data structures.
//!
//! This crate intentionally knows nothing about Wyn IR. Callers provide a node
//! universe and tiny successor/dependency callbacks; the crate supplies the
//! bookkeeping: reachability, topological ordering, and dominator trees.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use thiserror::Error;

/// Which frontier discipline a reachable-node walk should use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkOrder {
    /// First-in, first-out traversal.
    BreadthFirst,
    /// Last-in, first-out traversal, matching the common `Vec` stack pattern.
    DepthFirst,
}

impl WalkOrder {
    fn pop<N>(self, pending: &mut VecDeque<N>) -> Option<N> {
        match self {
            WalkOrder::BreadthFirst => pending.pop_front(),
            WalkOrder::DepthFirst => pending.pop_back(),
        }
    }

    fn push_all<N: Copy>(self, pending: &mut VecDeque<N>, nodes: &[N]) {
        pending.extend(nodes.iter().copied());
    }
}

/// Control flow returned by a reachable-node visitor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkDecision<T = ()> {
    /// Visit this node's successors.
    Continue,
    /// Keep this node in the visited set but do not visit its successors.
    Prune,
    /// Stop the traversal and return a value.
    Break(T),
}

/// Error returned when a topological traversal finds a cycle.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TopoError<N> {
    /// The nodes still carrying unsatisfied dependencies after Kahn's
    /// algorithm stops. The order follows the caller-provided node order.
    #[error("graph contains a cycle")]
    Cycle {
        remaining: Vec<N>,
    },
}

impl<N> TopoError<N> {
    /// Nodes that could not be scheduled because they are part of, or blocked
    /// by, a cycle.
    pub fn remaining(&self) -> &[N] {
        match self {
            TopoError::Cycle { remaining } => remaining,
        }
    }
}

/// Walk nodes reachable from `roots`.
///
/// The `successors` callback appends outgoing neighbors for a node into the
/// supplied buffer. Duplicate roots, duplicate edges, and cycles are harmless.
/// `visit` runs once per discovered node; returning [`WalkDecision::Prune`]
/// records the node but skips its outgoing edges, and
/// [`WalkDecision::Break`] returns immediately.
pub fn walk_reachable<N, I, S, V, T>(
    roots: I,
    order: WalkOrder,
    mut successors: S,
    mut visit: V,
) -> Option<T>
where
    N: Copy + Eq + Hash,
    I: IntoIterator<Item = N>,
    S: FnMut(N, &mut Vec<N>),
    V: FnMut(N) -> WalkDecision<T>,
{
    let mut seen = HashSet::new();
    let mut pending: VecDeque<N> = roots.into_iter().collect();
    let mut next = Vec::new();

    while let Some(node) = order.pop(&mut pending) {
        if !seen.insert(node) {
            continue;
        }

        match visit(node) {
            WalkDecision::Continue => {
                next.clear();
                successors(node, &mut next);
                order.push_all(&mut pending, &next);
            }
            WalkDecision::Prune => {}
            WalkDecision::Break(value) => return Some(value),
        }
    }

    None
}

/// Run a visitor over every reachable node.
pub fn for_each_reachable<N, I, S, V>(roots: I, order: WalkOrder, successors: S, mut visit: V)
where
    N: Copy + Eq + Hash,
    I: IntoIterator<Item = N>,
    S: FnMut(N, &mut Vec<N>),
    V: FnMut(N),
{
    let _: Option<()> = walk_reachable(roots, order, successors, |node| {
        visit(node);
        WalkDecision::Continue
    });
}

/// Return every node reachable from `roots`, in breadth-first discovery order.
///
/// The `successors` callback appends outgoing neighbors for a node into the
/// supplied buffer. Duplicate roots, duplicate edges, and cycles are harmless.
pub fn reachable_from<N, I, F>(roots: I, mut successors: F) -> Vec<N>
where
    N: Copy + Eq + Hash,
    I: IntoIterator<Item = N>,
    F: FnMut(N, &mut Vec<N>),
{
    reachable_from_ordered(roots, WalkOrder::BreadthFirst, |node, out| successors(node, out))
}

/// Return every node reachable from `roots` in the requested traversal order.
pub fn reachable_from_ordered<N, I, F>(roots: I, order: WalkOrder, successors: F) -> Vec<N>
where
    N: Copy + Eq + Hash,
    I: IntoIterator<Item = N>,
    F: FnMut(N, &mut Vec<N>),
{
    let mut reached = Vec::new();
    for_each_reachable(roots, order, successors, |node| reached.push(node));
    reached
}

/// Return every node reachable from `roots` as a set.
pub fn reachable_set<N, I, F>(roots: I, order: WalkOrder, successors: F) -> HashSet<N>
where
    N: Copy + Eq + Hash,
    I: IntoIterator<Item = N>,
    F: FnMut(N, &mut Vec<N>),
{
    reachable_from_ordered(roots, order, successors).into_iter().collect()
}

/// Return the first mapped value produced while walking reachable nodes.
pub fn find_map_reachable<N, I, F, V, T>(
    roots: I,
    order: WalkOrder,
    successors: F,
    mut visit: V,
) -> Option<T>
where
    N: Copy + Eq + Hash,
    I: IntoIterator<Item = N>,
    F: FnMut(N, &mut Vec<N>),
    V: FnMut(N) -> Option<T>,
{
    walk_reachable(roots, order, successors, |node| match visit(node) {
        Some(value) => WalkDecision::Break(value),
        None => WalkDecision::Continue,
    })
}

/// True when `target` is reachable from `start`.
pub fn reaches<N, F>(start: N, target: N, mut successors: F) -> bool
where
    N: Copy + Eq + Hash,
    F: FnMut(N, &mut Vec<N>),
{
    reaches_ordered(start, target, WalkOrder::BreadthFirst, |node, out| {
        successors(node, out);
    })
}

/// True when `target` is reachable from `start` using the requested traversal
/// order.
pub fn reaches_ordered<N, F>(start: N, target: N, order: WalkOrder, successors: F) -> bool
where
    N: Copy + Eq + Hash,
    F: FnMut(N, &mut Vec<N>),
{
    walk_reachable([start], order, successors, |node| {
        if node == target {
            WalkDecision::Break(())
        } else {
            WalkDecision::Continue
        }
    })
    .is_some()
}

/// Topologically sort a graph where the callback lists outgoing edges.
///
/// If `successors(a)` includes `b`, then `a` appears before `b` in the
/// returned order. Edges to nodes outside the supplied `nodes` universe are
/// ignored, which makes it convenient to sort an induced subgraph.
pub fn topo_sort<N, I, F>(nodes: I, mut successors: F) -> Result<Vec<N>, TopoError<N>>
where
    N: Copy + Eq + Hash,
    I: IntoIterator<Item = N>,
    F: FnMut(N, &mut Vec<N>),
{
    let nodes = unique_nodes(nodes);
    let universe: HashSet<N> = nodes.iter().copied().collect();
    let mut remaining: HashMap<N, usize> = nodes.iter().map(|&node| (node, 0)).collect();
    let mut dependents: HashMap<N, Vec<N>> = nodes.iter().map(|&node| (node, Vec::new())).collect();
    let mut next = Vec::new();

    for &node in &nodes {
        next.clear();
        successors(node, &mut next);

        let mut unique_edges = HashSet::new();
        for successor in next.iter().copied() {
            if !universe.contains(&successor) || !unique_edges.insert(successor) {
                continue;
            }
            if let Some(count) = remaining.get_mut(&successor) {
                *count += 1;
            } else {
                debug_assert!(
                    false,
                    "successor passed universe filter but has no remaining count"
                );
            }
            dependents.entry(node).or_default().push(successor);
        }
    }

    kahn(nodes, remaining, dependents)
}

/// Topologically sort a graph where the callback lists each node's
/// dependencies.
///
/// If `dependencies(a)` includes `b`, then `b` appears before `a` in the
/// returned order. Dependencies outside the supplied `nodes` universe are
/// ignored, which makes it convenient to sort an induced subgraph.
pub fn topo_sort_by_dependencies<N, I, F>(nodes: I, mut dependencies: F) -> Result<Vec<N>, TopoError<N>>
where
    N: Copy + Eq + Hash,
    I: IntoIterator<Item = N>,
    F: FnMut(N, &mut Vec<N>),
{
    let nodes = unique_nodes(nodes);
    let universe: HashSet<N> = nodes.iter().copied().collect();
    let mut remaining: HashMap<N, usize> = nodes.iter().map(|&node| (node, 0)).collect();
    let mut dependents: HashMap<N, Vec<N>> = nodes.iter().map(|&node| (node, Vec::new())).collect();
    let mut deps = Vec::new();

    for &node in &nodes {
        deps.clear();
        dependencies(node, &mut deps);

        let mut unique_deps = HashSet::new();
        for dependency in deps.iter().copied() {
            if !universe.contains(&dependency) || !unique_deps.insert(dependency) {
                continue;
            }
            if let Some(count) = remaining.get_mut(&node) {
                *count += 1;
            } else {
                debug_assert!(false, "node from node list has no remaining count");
            }
            dependents.entry(dependency).or_default().push(node);
        }
    }

    kahn(nodes, remaining, dependents)
}

/// Dominator tree for a flow graph rooted at one entry node.
#[derive(Clone, Debug)]
pub struct DominatorTree<N> {
    idom: HashMap<N, N>,
    children: HashMap<N, Vec<N>>,
    preorder: Vec<N>,
    reachable: HashSet<N>,
}

impl<N> DominatorTree<N>
where
    N: Copy + Eq + Hash,
{
    /// Build a dominator tree from a root and successor callback, discovering
    /// nodes depth-first.
    ///
    /// Unreachable nodes are excluded entirely: they have no immediate
    /// dominator and never appear in preorder traversal.
    pub fn build<F>(entry: N, successors: F) -> Self
    where
        F: FnMut(N, &mut Vec<N>),
    {
        Self::build_ordered(entry, WalkOrder::DepthFirst, successors)
    }

    /// Build a dominator tree, choosing how nodes are discovered.
    ///
    /// `order` fixes the order of each node's [`children`](Self::children) and
    /// hence of [`preorder`](Self::preorder). The tree itself — which node
    /// dominates which — does not depend on it. Callers that emit code by
    /// walking the tree see `order` in their output, so it is part of their
    /// contract rather than an implementation detail.
    pub fn build_ordered<F>(entry: N, order: WalkOrder, mut successors: F) -> Self
    where
        F: FnMut(N, &mut Vec<N>),
    {
        let reachable = reachable_from_ordered([entry], order, |node, out| successors(node, out));
        let reachable_set: HashSet<N> = reachable.iter().copied().collect();

        let mut predecessors: HashMap<N, Vec<N>> =
            reachable.iter().map(|&node| (node, Vec::new())).collect();
        let mut next = Vec::new();
        for &node in &reachable {
            next.clear();
            successors(node, &mut next);
            for successor in next.iter().copied() {
                if reachable_set.contains(&successor) {
                    predecessors.entry(successor).or_default().push(node);
                }
            }
        }

        let mut doms: HashMap<N, HashSet<N>> = HashMap::new();
        for &node in &reachable {
            if node == entry {
                doms.insert(node, [entry].into_iter().collect());
            } else {
                doms.insert(node, reachable_set.clone());
            }
        }

        loop {
            let mut changed = false;
            for &node in &reachable {
                if node == entry {
                    continue;
                }

                let Some(preds) = predecessors.get(&node) else {
                    debug_assert!(false, "reachable node has no predecessor entry");
                    continue;
                };
                let Some((first, rest)) = preds.split_first() else {
                    debug_assert!(false, "reachable non-entry node has no reachable predecessor");
                    continue;
                };
                let Some(first_doms) = doms.get(first) else {
                    debug_assert!(false, "predecessor has no dominator set");
                    continue;
                };
                let mut new_set = first_doms.clone();
                for pred in rest {
                    let Some(pred_doms) = doms.get(pred) else {
                        debug_assert!(false, "predecessor has no dominator set");
                        continue;
                    };
                    new_set = new_set.intersection(pred_doms).copied().collect();
                }
                new_set.insert(node);

                if doms.get(&node) != Some(&new_set) {
                    doms.insert(node, new_set);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        let mut idom = HashMap::new();
        for &node in &reachable {
            if node == entry {
                continue;
            }

            let Some(dom_set) = doms.get(&node) else {
                debug_assert!(false, "reachable node has no dominator set");
                continue;
            };
            let mut best = None;
            let mut best_depth = 0;
            for &dominator in dom_set {
                if dominator == node {
                    continue;
                }
                let Some(dominator_doms) = doms.get(&dominator) else {
                    debug_assert!(false, "dominator is not in the dominator map");
                    continue;
                };
                let depth = dominator_doms.len();
                if depth > best_depth {
                    best = Some(dominator);
                    best_depth = depth;
                }
            }

            if let Some(parent) = best {
                idom.insert(node, parent);
            }
        }

        let mut children: HashMap<N, Vec<N>> = reachable.iter().map(|&node| (node, Vec::new())).collect();
        for &node in &reachable {
            if let Some(&parent) = idom.get(&node) {
                children.entry(parent).or_default().push(node);
            }
        }

        let mut preorder = Vec::new();
        let mut stack = vec![entry];
        while let Some(node) = stack.pop() {
            preorder.push(node);
            if let Some(children) = children.get(&node) {
                for &child in children.iter().rev() {
                    stack.push(child);
                }
            }
        }

        Self {
            idom,
            children,
            preorder,
            reachable: reachable_set,
        }
    }

    /// Preorder traversal of the dominator tree.
    pub fn preorder(&self) -> &[N] {
        &self.preorder
    }

    /// Immediate dominator of `node`; the entry and unreachable nodes return
    /// `None`.
    pub fn idom(&self, node: N) -> Option<N> {
        self.idom.get(&node).copied()
    }

    /// Children of `node` in the dominator tree.
    pub fn children(&self, node: N) -> &[N] {
        self.children.get(&node).map_or(&[], Vec::as_slice)
    }

    /// True when `node` is reachable from the tree entry.
    pub fn is_reachable(&self, node: N) -> bool {
        self.reachable.contains(&node)
    }

    /// True when `dominator` dominates `node`.
    pub fn dominates(&self, dominator: N, node: N) -> bool {
        if !self.is_reachable(dominator) || !self.is_reachable(node) {
            return false;
        }
        if dominator == node {
            return true;
        }

        let mut current = node;
        while let Some(parent) = self.idom(current) {
            if parent == dominator {
                return true;
            }
            current = parent;
        }

        false
    }
}

fn unique_nodes<N, I>(nodes: I) -> Vec<N>
where
    N: Copy + Eq + Hash,
    I: IntoIterator<Item = N>,
{
    let mut seen = HashSet::new();
    nodes.into_iter().filter(|node| seen.insert(*node)).collect()
}

fn kahn<N>(
    nodes: Vec<N>,
    mut remaining: HashMap<N, usize>,
    dependents: HashMap<N, Vec<N>>,
) -> Result<Vec<N>, TopoError<N>>
where
    N: Copy + Eq + Hash,
{
    let mut ready: VecDeque<N> =
        nodes.iter().copied().filter(|node| remaining.get(node).copied().unwrap_or(0) == 0).collect();
    let mut result = Vec::with_capacity(nodes.len());

    while let Some(node) = ready.pop_front() {
        result.push(node);
        if let Some(next_nodes) = dependents.get(&node) {
            for &next in next_nodes {
                if let Some(count) = remaining.get_mut(&next) {
                    *count -= 1;
                    if *count == 0 {
                        ready.push_back(next);
                    }
                } else {
                    debug_assert!(false, "dependent has no remaining count");
                }
            }
        }
    }

    if result.len() == nodes.len() {
        Ok(result)
    } else {
        let remaining =
            nodes.into_iter().filter(|node| remaining.get(node).copied().unwrap_or(0) > 0).collect();
        Err(TopoError::Cycle { remaining })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diamond(node: u8, out: &mut Vec<u8>) {
        match node {
            0 => out.extend([1, 2]),
            1 | 2 => out.push(3),
            _ => {}
        }
    }

    #[test]
    fn reachability_handles_duplicates_and_cycles() {
        fn cyclic(node: u8, out: &mut Vec<u8>) {
            match node {
                0 => out.extend([1, 1, 2]),
                1 => out.extend([0, 3]),
                2 => out.push(3),
                _ => {}
            }
        }

        assert_eq!(reachable_from([0], cyclic), vec![0, 1, 2, 3]);
        assert!(reaches(0, 3, cyclic));
        assert!(!reaches(3, 0, cyclic));
    }

    #[test]
    fn ordered_reachability_supports_stack_order() {
        fn tree(node: u8, out: &mut Vec<u8>) {
            match node {
                0 => out.extend([1, 2]),
                1 => out.push(3),
                2 => out.push(4),
                _ => {}
            }
        }

        assert_eq!(
            reachable_from_ordered([0], WalkOrder::BreadthFirst, tree),
            vec![0, 1, 2, 3, 4]
        );
        assert_eq!(
            reachable_from_ordered([0], WalkOrder::DepthFirst, tree),
            vec![0, 2, 4, 1, 3]
        );
    }

    #[test]
    fn traversal_can_prune_or_break() {
        fn tree(node: u8, out: &mut Vec<u8>) {
            match node {
                0 => out.extend([1, 2]),
                1 => out.push(3),
                2 => out.push(4),
                _ => {}
            }
        }

        let mut visited = Vec::new();
        let _: Option<()> = walk_reachable([0], WalkOrder::BreadthFirst, tree, |node| {
            visited.push(node);
            if node == 1 {
                WalkDecision::Prune
            } else {
                WalkDecision::Continue
            }
        });
        assert_eq!(visited, vec![0, 1, 2, 4]);

        let found = find_map_reachable([0], WalkOrder::DepthFirst, tree, |node| {
            (node == 4).then_some(node * 10)
        });
        assert_eq!(found, Some(40));
        assert!(reachable_set([0], WalkOrder::DepthFirst, tree).contains(&3));
        assert!(reaches_ordered(0, 4, WalkOrder::DepthFirst, tree));
    }

    #[test]
    fn topological_sort_orders_successors_after_producers() {
        let order = match topo_sort([0, 1, 2, 3], diamond) {
            Ok(order) => order,
            Err(err) => panic!("diamond should be acyclic: {err}"),
        };
        let pos = |node| match order.iter().position(|&candidate| candidate == node) {
            Some(index) => index,
            None => panic!("topological order is missing node {node}"),
        };

        assert!(pos(0) < pos(1));
        assert!(pos(0) < pos(2));
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(3));
    }

    #[test]
    fn dependency_topological_sort_orders_dependencies_first() {
        fn deps(node: u8, out: &mut Vec<u8>) {
            match node {
                2 => out.extend([0, 1]),
                3 => out.push(2),
                _ => {}
            }
        }

        let order = match topo_sort_by_dependencies([0, 1, 2, 3], deps) {
            Ok(order) => order,
            Err(err) => panic!("dependency graph should be acyclic: {err}"),
        };
        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn topological_sort_reports_cycle_members() {
        fn deps(node: u8, out: &mut Vec<u8>) {
            match node {
                0 => out.push(1),
                1 => out.push(0),
                _ => {}
            }
        }

        let err = match topo_sort_by_dependencies([0, 1, 2], deps) {
            Ok(order) => panic!("cycle should be detected, got order {order:?}"),
            Err(err) => err,
        };
        assert_eq!(err.remaining(), &[0, 1]);
    }

    #[test]
    fn dominator_tree_excludes_unreachable_predecessors() {
        fn cfg(node: u8, out: &mut Vec<u8>) {
            match node {
                0 => out.push(1),
                1 => out.extend([2, 3]),
                2 => out.push(1),
                4 => out.push(3),
                _ => {}
            }
        }

        let tree = DominatorTree::build(0, cfg);
        assert_eq!(tree.idom(0), None);
        assert_eq!(tree.idom(1), Some(0));
        assert_eq!(tree.idom(2), Some(1));
        assert_eq!(tree.idom(3), Some(1));
        assert_eq!(tree.idom(4), None);
        assert_eq!(tree.preorder(), &[0, 1, 3, 2]);

        assert!(tree.dominates(1, 3));
        assert!(!tree.dominates(2, 3));
        assert!(!tree.is_reachable(4));
    }

    #[test]
    fn dominator_walk_order_selects_child_and_preorder_order() {
        // `build` discovers depth-first, so `0`'s successors land in the
        // reverse of the order the callback lists them.
        let dfs = DominatorTree::build(0, diamond);
        assert_eq!(dfs.children(0), &[2, 3, 1]);
        assert_eq!(dfs.preorder(), &[0, 2, 3, 1]);

        let bfs = DominatorTree::build_ordered(0, WalkOrder::BreadthFirst, diamond);
        assert_eq!(bfs.children(0), &[1, 2, 3]);
        assert_eq!(bfs.preorder(), &[0, 1, 2, 3]);

        // Only the ordering moves; domination itself is order-independent.
        for tree in [&dfs, &bfs] {
            assert_eq!(tree.idom(1), Some(0));
            assert_eq!(tree.idom(2), Some(0));
            assert_eq!(tree.idom(3), Some(0));
            assert!(tree.dominates(0, 3));
            assert!(!tree.dominates(1, 3));
        }
    }
}
