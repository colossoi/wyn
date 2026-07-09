//! Loop analysis over the skeleton CFG.
//!
//! A block is a loop header iff `control_headers` maps it to
//! `ControlHeader::Loop`. The body of that loop is the set of blocks
//! reachable from the header without crossing its `merge` block, which
//! sits at the loop's exit. Headers themselves are inside their own loop.
//!
//! For LICM, we only need:
//! - `is_header(b)` — do we need to push a loop frame when we enter `b`?
//! - `is_in_loop(b, header)` — is block `b` inside the loop rooted at `header`?

use crate::ssa::framework::BlockId;
use crate::{LookupMap, LookupSet};

use crate::ssa::types::ControlHeader;

use super::types::{Skeleton, SkeletonTerminator};

pub struct LoopAnalysis {
    /// All blocks inside each loop (key = loop header).
    bodies: LookupMap<BlockId, LookupSet<BlockId>>,
}

impl LoopAnalysis {
    pub fn build(skeleton: &Skeleton, control_headers: &LookupMap<BlockId, ControlHeader>) -> Self {
        let mut bodies: LookupMap<BlockId, LookupSet<BlockId>> = LookupMap::new();

        // Collect every header and DFS its body, stopping at `merge`.
        for (&header, ch) in control_headers {
            let merge = match ch {
                ControlHeader::Loop { merge, .. } => *merge,
                ControlHeader::Selection { .. } => continue,
            };
            let body = collect_loop_body(skeleton, header, merge);
            bodies.insert(header, body);
        }

        LoopAnalysis { bodies }
    }

    pub fn is_header(&self, b: BlockId) -> bool {
        self.bodies.contains_key(&b)
    }

    pub fn is_in_loop(&self, b: BlockId, header: BlockId) -> bool {
        self.bodies.get(&header).is_some_and(|s| s.contains(&b))
    }
}

/// DFS the skeleton from `header`, stopping at `merge`. The header itself
/// is included; `merge` is not.
fn collect_loop_body(skeleton: &Skeleton, header: BlockId, merge: BlockId) -> LookupSet<BlockId> {
    let mut body = LookupSet::new();
    let _: Option<()> = wyn_graph::walk_reachable(
        [header],
        wyn_graph::WalkOrder::DepthFirst,
        |b, out| match &skeleton.blocks[b].term {
            SkeletonTerminator::Return(_) | SkeletonTerminator::Unreachable => {}
            SkeletonTerminator::Branch { target, .. } => out.push(*target),
            SkeletonTerminator::CondBranch {
                then_target,
                else_target,
                ..
            } => {
                out.push(*then_target);
                out.push(*else_target);
            }
        },
        |b| {
            if b == merge {
                wyn_graph::WalkDecision::Prune
            } else {
                body.insert(b);
                wyn_graph::WalkDecision::Continue
            }
        },
    );
    body
}
