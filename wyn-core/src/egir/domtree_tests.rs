//! Unit tests for the dominator-tree builder.

use super::*;
use crate::ssa::framework::BlockId;
use slotmap::SlotMap;

/// Minimal `CfgView` over an in-memory adjacency map. Each test below
/// builds one of these directly so the CFG shape is staring you in the
/// face right next to the assertions.
struct MapCfg {
    entry: BlockId,
    succ: std::collections::HashMap<BlockId, Vec<BlockId>>,
}

impl CfgView for MapCfg {
    fn entry(&self) -> BlockId {
        self.entry
    }
    fn successors(&self, block: BlockId) -> Vec<BlockId> {
        self.succ.get(&block).cloned().unwrap_or_default()
    }
}

fn fresh_blocks(n: usize) -> Vec<BlockId> {
    let mut sm: SlotMap<BlockId, ()> = SlotMap::with_key();
    (0..n).map(|_| sm.insert(())).collect()
}

// --- baseline ---------------------------------------------------------------

#[test]
fn linear_chain_picks_predecessor_as_idom() {
    let b = fresh_blocks(3);
    let cfg = MapCfg {
        entry: b[0],
        succ: [(b[0], vec![b[1]]), (b[1], vec![b[2]]), (b[2], vec![])].into_iter().collect(),
    };
    let dt = DomTree::build(&cfg);
    assert_eq!(dt.idom(b[0]), None);
    assert_eq!(dt.idom(b[1]), Some(b[0]));
    assert_eq!(dt.idom(b[2]), Some(b[1]));
    assert_eq!(dt.preorder(), [b[0], b[1], b[2]].as_slice());
}

#[test]
fn diamond_picks_entry_as_merge_idom() {
    //         entry
    //         /    \
    //      then     else
    //         \    /
    //         merge
    let b = fresh_blocks(4);
    let cfg = MapCfg {
        entry: b[0],
        succ: [
            (b[0], vec![b[1], b[2]]),
            (b[1], vec![b[3]]),
            (b[2], vec![b[3]]),
            (b[3], vec![]),
        ]
        .into_iter()
        .collect(),
    };
    let dt = DomTree::build(&cfg);
    assert_eq!(dt.idom(b[0]), None);
    assert_eq!(dt.idom(b[1]), Some(b[0]));
    assert_eq!(dt.idom(b[2]), Some(b[0]));
    assert_eq!(dt.idom(b[3]), Some(b[0]));
}

// --- regression for the post-fold unreachable-arm bug -----------------------

#[test]
fn unreachable_predecessor_does_not_corrupt_reachable_merge_idom() {
    // This is the precise shape skel_opt::fold_constant_branches leaves
    // behind after rewriting `if literal_bool then ... else ...`:
    //
    //   entry --br--> else_arm --br(...)--> merge
    //       (then_arm is still in `skeleton.blocks`, but it has no
    //        predecessor and no successor that reaches merge from
    //        entry. It IS still listed by `all_blocks()`.)
    //
    // The merge block has preds [then_arm, else_arm] in `cfg.successors`
    // analysis. If the domtree dataflow treats `then_arm`'s empty-preds
    // case as `doms = ∅`, the intersection corrupts merge's dom set and
    // it ends up with no idom — which causes elaborate to skip it and
    // SPIR-V codegen to panic on the dangling branch.
    //
    // The fix is to restrict the analysis to entry-reachable blocks.
    let b = fresh_blocks(4);
    let entry = b[0];
    let then_arm = b[1]; // unreachable from entry after fold
    let else_arm = b[2];
    let merge = b[3];
    let cfg = MapCfg {
        entry,
        succ: [
            (entry, vec![else_arm]), // fold rewrote CondBranch -> Branch(else)
            (then_arm, vec![merge]), // dead arm's old branch survives
            (else_arm, vec![merge]),
            (merge, vec![]),
        ]
        .into_iter()
        .collect(),
    };
    let dt = DomTree::build(&cfg);

    // Reachable blocks must get correct idoms regardless of dead-block junk.
    assert_eq!(dt.idom(entry), None);
    assert_eq!(dt.idom(else_arm), Some(entry));
    assert_eq!(
        dt.idom(merge),
        Some(else_arm),
        "merge's idom must be its only reachable predecessor; an \
         unreachable sibling predecessor must not pollute the analysis"
    );

    // The merge block must appear in the preorder traversal — elaborate
    // walks this list and skipping the merge block is what produces the
    // dangling-branch SPIR-V panic.
    assert!(
        dt.preorder().contains(&merge),
        "preorder must visit reachable merge: got {:?}",
        dt.preorder()
    );
}

#[test]
fn unreachable_blocks_are_excluded_from_preorder() {
    // Same shape — the dead `then_arm` is unreachable from entry and
    // must not appear in the preorder. (Visiting it would let elaborate
    // emit code into a block no predecessor branches to.)
    let b = fresh_blocks(4);
    let cfg = MapCfg {
        entry: b[0],
        succ: [
            (b[0], vec![b[2]]),
            (b[1], vec![b[3]]),
            (b[2], vec![b[3]]),
            (b[3], vec![]),
        ]
        .into_iter()
        .collect(),
    };
    let dt = DomTree::build(&cfg);
    assert!(
        !dt.preorder().contains(&b[1]),
        "unreachable block must not appear in preorder: got {:?}",
        dt.preorder()
    );
    assert_eq!(dt.idom(b[1]), None);
}
