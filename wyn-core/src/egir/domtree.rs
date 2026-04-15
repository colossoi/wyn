//! Dominator tree for the egraph pipeline.
//!
//! Can be built from either a skeleton CFG or an original SSA FuncBody.
//! Used by canonicalization (preorder traversal) and elaboration (scoping).

use crate::ssa::framework::BlockId;
use std::collections::{HashMap, HashSet};

/// Dominator tree.
pub struct DomTree {
    /// Immediate dominator of each block (entry has no idom).
    idom: HashMap<BlockId, BlockId>,
    /// Children in the dominator tree.
    children: HashMap<BlockId, Vec<BlockId>>,
    /// Preorder traversal of the domtree.
    preorder_list: Vec<BlockId>,
    /// Depth of each block in the domtree (entry = 0).
    depth: HashMap<BlockId, usize>,
    /// Entry block.
    entry: BlockId,
}

/// Generic CFG trait so we can compute domtrees from either SSA or skeleton.
pub trait CfgView {
    fn entry(&self) -> BlockId;
    fn all_blocks(&self) -> Vec<BlockId>;
    fn successors(&self, block: BlockId) -> Vec<BlockId>;
}

impl DomTree {
    /// Build a dominator tree from any CFG-like structure.
    pub fn build(cfg: &dyn CfgView) -> Self {
        let entry = cfg.entry();
        let all_blocks = cfg.all_blocks();

        // Build predecessor map.
        let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        for &b in &all_blocks {
            preds.entry(b).or_default();
        }
        for &b in &all_blocks {
            for succ in cfg.successors(b) {
                preds.entry(succ).or_default().push(b);
            }
        }

        // Iterative fixpoint dominator computation.
        let mut doms: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
        let all_set: HashSet<BlockId> = all_blocks.iter().copied().collect();
        for &b in &all_blocks {
            if b == entry {
                doms.insert(b, HashSet::from([entry]));
            } else {
                doms.insert(b, all_set.clone());
            }
        }

        loop {
            let mut changed = false;
            for &b in &all_blocks {
                if b == entry {
                    continue;
                }
                let pred_list = &preds[&b];
                let mut new_set = if pred_list.is_empty() {
                    HashSet::new()
                } else {
                    let mut iter = pred_list.iter();
                    let first = doms[iter.next().unwrap()].clone();
                    iter.fold(first, |acc, p| acc.intersection(&doms[p]).copied().collect())
                };
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

        // Extract idom.
        let mut idom = HashMap::new();
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
        let mut children: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        for &b in &all_blocks {
            children.entry(b).or_default();
        }
        for (&child, &parent) in &idom {
            children.entry(parent).or_default().push(child);
        }

        // Preorder traversal.
        let mut preorder_list = Vec::new();
        let mut depth = HashMap::new();
        let mut stack = vec![(entry, 0usize)];
        while let Some((b, d)) = stack.pop() {
            preorder_list.push(b);
            depth.insert(b, d);
            if let Some(ch) = children.get(&b) {
                for &c in ch.iter().rev() {
                    stack.push((c, d + 1));
                }
            }
        }

        DomTree {
            idom,
            children,
            preorder_list,
            depth,
            entry,
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

    /// Depth of a block in the domtree (entry = 0).
    pub fn block_depth(&self, block: BlockId) -> usize {
        self.depth.get(&block).copied().unwrap_or(0)
    }

    /// Does `a` dominate `b`?
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if a == b {
            return true;
        }
        let mut cur = b;
        while let Some(parent) = self.idom(cur) {
            if parent == a {
                return true;
            }
            cur = parent;
        }
        false
    }

    /// Entry block.
    pub fn entry(&self) -> BlockId {
        self.entry
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

    fn all_blocks(&self) -> Vec<BlockId> {
        self.skeleton.blocks.keys().collect()
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
