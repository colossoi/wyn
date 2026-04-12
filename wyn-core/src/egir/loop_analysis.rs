//! Loop analysis over the skeleton CFG.
//!
//! A block is a loop header iff `control_headers` maps it to
//! `ControlHeader::Loop`. The body of that loop is the set of blocks
//! reachable from the header without crossing its `merge` block, which
//! sits at the loop's exit. Headers themselves are inside their own loop.
//!
//! For LICM, we only need:
//! - `is_header(b)` — do we need to push a loop frame when we enter `b`?
//! - `innermost_loop_of(b)` — which loop header (if any) currently encloses `b`?
//! - `is_in_loop(b, header)` — is block `b` inside the loop rooted at `header`?
//!
//! "Innermost" is disambiguated by domtree depth: the header with the
//! greatest depth wins, because a nested-inner header is always strictly
//! deeper than its enclosing outer header.

use std::collections::{HashMap, HashSet};
use wyn_ssa::BlockId;

use crate::ssa::types::ControlHeader;

use super::domtree::DomTree;
use super::types::{Skeleton, SkeletonTerminator};

pub struct LoopAnalysis {
    /// Block → innermost loop header containing it (if any).
    innermost: HashMap<BlockId, BlockId>,
    /// All blocks inside each loop (key = loop header).
    bodies: HashMap<BlockId, HashSet<BlockId>>,
}

impl LoopAnalysis {
    pub fn build(
        skeleton: &Skeleton,
        control_headers: &HashMap<BlockId, ControlHeader>,
        domtree: &DomTree,
    ) -> Self {
        let mut bodies: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();

        // Collect every header and DFS its body, stopping at `merge`.
        for (&header, ch) in control_headers {
            let merge = match ch {
                ControlHeader::Loop { merge, .. } => *merge,
                ControlHeader::Selection { .. } => continue,
            };
            let body = collect_loop_body(skeleton, header, merge);
            bodies.insert(header, body);
        }

        // For each block, pick the innermost (deepest-header) loop.
        let mut innermost: HashMap<BlockId, BlockId> = HashMap::new();
        for (&header, body) in &bodies {
            let header_depth = domtree.block_depth(header);
            for &b in body {
                match innermost.get(&b) {
                    None => {
                        innermost.insert(b, header);
                    }
                    Some(&existing) => {
                        if domtree.block_depth(existing) < header_depth {
                            innermost.insert(b, header);
                        }
                    }
                }
            }
        }

        LoopAnalysis { innermost, bodies }
    }

    pub fn is_header(&self, b: BlockId) -> bool {
        self.bodies.contains_key(&b)
    }

    pub fn innermost_loop_of(&self, b: BlockId) -> Option<BlockId> {
        self.innermost.get(&b).copied()
    }

    pub fn is_in_loop(&self, b: BlockId, header: BlockId) -> bool {
        self.bodies.get(&header).is_some_and(|s| s.contains(&b))
    }
}

/// DFS the skeleton from `header`, stopping at `merge`. The header itself
/// is included; `merge` is not.
fn collect_loop_body(skeleton: &Skeleton, header: BlockId, merge: BlockId) -> HashSet<BlockId> {
    let mut body = HashSet::new();
    let mut stack = vec![header];
    while let Some(b) = stack.pop() {
        if b == merge || !body.insert(b) {
            continue;
        }
        match &skeleton.blocks[b].term {
            SkeletonTerminator::Return(_) | SkeletonTerminator::Unreachable => {}
            SkeletonTerminator::Branch { target, .. } => stack.push(*target),
            SkeletonTerminator::CondBranch {
                then_target,
                else_target,
                ..
            } => {
                stack.push(*then_target);
                stack.push(*else_target);
            }
        }
    }
    body
}
