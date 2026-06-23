//! Dominator tree for the egraph pipeline.
//!
//! Can be built from either a skeleton CFG or an original SSA FuncBody.
//! Used by canonicalization (preorder traversal) and elaboration (scoping).

use crate::ssa::framework::BlockId;
use crate::{LookupMap, LookupSet};

#[cfg(test)]
#[path = "domtree_tests.rs"]
mod domtree_tests;

/// Dominator tree.
pub struct DomTree {
    /// Immediate dominator of each block (entry has no idom).
    idom: LookupMap<BlockId, BlockId>,
    /// Children in the dominator tree.
    children: LookupMap<BlockId, Vec<BlockId>>,
    /// Preorder traversal of the domtree.
    preorder_list: Vec<BlockId>,
}

/// Generic CFG trait so we can compute domtrees from either SSA or skeleton.
pub trait CfgView {
    fn entry(&self) -> BlockId;
    fn successors(&self, block: BlockId) -> Vec<BlockId>;
}

impl DomTree {
    /// Build a dominator tree from any CFG-like structure.
    ///
    /// Dominator analysis is defined on the flowgraph rooted at the
    /// entry — "X dominates Y" means "every path from *entry* to Y
    /// passes through X". Blocks not reachable from entry have no
    /// well-defined dominators, and including them in the iterative
    /// fixpoint poisons reachable blocks they appear as predecessors of
    /// (intersection with their garbage `doms` set collapses to ∅). So
    /// the analysis runs over the entry-reachable subgraph only;
    /// unreachable blocks left behind by upstream passes (e.g.
    /// `fold_constant_branches` dropping a dead arm) appear in the
    /// resulting tree only as "no idom, not in preorder".
    pub fn build(cfg: &dyn CfgView) -> Self {
        let entry = cfg.entry();

        // Forward BFS from entry to collect the reachable subgraph.
        // Everything else is left out of the analysis entirely.
        let reachable: LookupSet<BlockId> = {
            let mut visited: LookupSet<BlockId> = LookupSet::new();
            let mut stack = vec![entry];
            while let Some(b) = stack.pop() {
                if !visited.insert(b) {
                    continue;
                }
                for s in cfg.successors(b) {
                    if !visited.contains(&s) {
                        stack.push(s);
                    }
                }
            }
            visited
        };
        let reachable_blocks: Vec<BlockId> = reachable.iter().copied().collect();

        // Build predecessor map, restricted to edges between reachable
        // blocks. An unreachable predecessor would still poison the
        // intersection even if the destination is reachable.
        let mut preds: LookupMap<BlockId, Vec<BlockId>> = LookupMap::new();
        for &b in &reachable_blocks {
            preds.entry(b).or_default();
        }
        for &b in &reachable_blocks {
            for succ in cfg.successors(b) {
                if reachable.contains(&succ) {
                    preds.entry(succ).or_default().push(b);
                }
            }
        }

        // Iterative fixpoint dominator computation.
        let mut doms: LookupMap<BlockId, LookupSet<BlockId>> = LookupMap::new();
        for &b in &reachable_blocks {
            if b == entry {
                doms.insert(b, LookupSet::from([entry]));
            } else {
                doms.insert(b, reachable.clone());
            }
        }

        loop {
            let mut changed = false;
            for &b in &reachable_blocks {
                if b == entry {
                    continue;
                }
                let pred_list = &preds[&b];
                // Every non-entry reachable block has at least one
                // reachable predecessor (otherwise it wouldn't be
                // reachable), so the intersection is well-defined.
                let mut iter = pred_list.iter();
                let first = doms[iter.next().expect("reachable non-entry block has a predecessor")].clone();
                let mut new_set: LookupSet<BlockId> =
                    iter.fold(first, |acc, p| acc.intersection(&doms[p]).copied().collect());
                new_set.insert(b);
                if new_set != doms[&b] {
                    doms.insert(b, new_set);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // Rebind so the idom extraction below iterates the same set.
        let all_blocks = reachable_blocks;

        // Extract idom.
        let mut idom = LookupMap::new();
        for &b in &all_blocks {
            if b == entry {
                continue;
            }
            let dom_set = &doms[&b];
            let mut best: Option<BlockId> = None;
            let mut best_size = 0;
            for &d in dom_set {
                if d == b {
                    continue;
                }
                let sz = doms[&d].len();
                if sz > best_size {
                    best_size = sz;
                    best = Some(d);
                }
            }
            if let Some(parent) = best {
                idom.insert(b, parent);
            }
        }

        // Build children map.
        let mut children: LookupMap<BlockId, Vec<BlockId>> = LookupMap::new();
        for &b in &all_blocks {
            children.entry(b).or_default();
        }
        for (&child, &parent) in &idom {
            children.entry(parent).or_default().push(child);
        }

        // Preorder traversal.
        let mut preorder_list = Vec::new();
        let mut stack = vec![entry];
        while let Some(b) = stack.pop() {
            preorder_list.push(b);
            if let Some(ch) = children.get(&b) {
                for &c in ch.iter().rev() {
                    stack.push(c);
                }
            }
        }

        DomTree {
            idom,
            children,
            preorder_list,
        }
    }

    /// Preorder traversal of the domtree.
    pub fn preorder(&self) -> &[BlockId] {
        &self.preorder_list
    }

    /// Immediate dominator of a block. Entry block returns None.
    pub fn idom(&self, block: BlockId) -> Option<BlockId> {
        self.idom.get(&block).copied()
    }

    /// Children of a block in the dominator tree.
    pub fn dom_children(&self, block: BlockId) -> &[BlockId] {
        self.children.get(&block).map_or(&[], |v| v.as_slice())
    }
}

// ---------------------------------------------------------------------------
// CfgView implementations
// ---------------------------------------------------------------------------

/// CfgView adapter for a Skeleton CFG.
pub struct SkeletonCfgView<'a> {
    pub skeleton: &'a super::types::Skeleton,
}

impl CfgView for SkeletonCfgView<'_> {
    fn entry(&self) -> BlockId {
        self.skeleton.entry
    }

    fn successors(&self, block: BlockId) -> Vec<BlockId> {
        use super::types::SkeletonTerminator;
        match &self.skeleton.blocks[block].term {
            SkeletonTerminator::Return(_) | SkeletonTerminator::Unreachable => vec![],
            SkeletonTerminator::Branch { target, .. } => vec![*target],
            SkeletonTerminator::CondBranch {
                then_target,
                else_target,
                ..
            } => vec![*then_target, *else_target],
        }
    }
}
