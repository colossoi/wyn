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

use crate::flow::{BlockId, ControlHeader};
use crate::{LookupMap, LookupSet};

use super::types::{EGraph, ENode, EgirPhase, NodeId, Skeleton, SkeletonTerminator};

pub struct LoopAnalysis {
    /// All blocks inside each loop (key = loop header).
    bodies: LookupMap<BlockId, LookupSet<BlockId>>,
}

impl LoopAnalysis {
    pub fn build<P: EgirPhase>(
        skeleton: &Skeleton<P>,
        control_headers: &LookupMap<BlockId, ControlHeader>,
    ) -> Self {
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

    /// A value defined in `block` is invariant with respect to `header` when
    /// its definition lies outside that loop.
    pub fn block_is_invariant(&self, block: BlockId, header: BlockId) -> bool {
        !self.is_in_loop(block, header)
    }

    /// All operand definitions must be outside the loop before a pure
    /// operation can be placed outside it.
    pub fn operands_are_invariant(
        &self,
        header: BlockId,
        operand_blocks: impl IntoIterator<Item = BlockId>,
    ) -> bool {
        operand_blocks.into_iter().all(|block| self.block_is_invariant(block, header))
    }
}

/// Pre-elaboration view of the same loop-invariance rule used by placement.
/// Constants and function parameters have no CFG-local definition; block and
/// effect values use their defining block; pure values recursively require
/// every operand to be invariant.
pub struct LoopInvariance<'a, P: EgirPhase> {
    graph: &'a EGraph<P>,
    loops: &'a LoopAnalysis,
    header: BlockId,
    effect_blocks: LookupMap<NodeId, BlockId>,
    memo: LookupMap<NodeId, bool>,
}

impl<'a, P: EgirPhase> LoopInvariance<'a, P> {
    pub fn new(graph: &'a EGraph<P>, loops: &'a LoopAnalysis, header: BlockId) -> Self {
        let mut effect_blocks = LookupMap::new();
        for (block, body) in &graph.skeleton.blocks {
            for effect in &body.side_effects {
                if let Some(result) = effect.result {
                    effect_blocks.insert(result, block);
                }
            }
        }
        Self {
            graph,
            loops,
            header,
            effect_blocks,
            memo: LookupMap::new(),
        }
    }

    pub fn is_invariant(&mut self, node: NodeId) -> bool {
        if let Some(value) = self.memo.get(&node) {
            return *value;
        }
        let invariant = match self.graph.nodes[node].clone() {
            ENode::Constant(_) | ENode::FuncParam { .. } => true,
            ENode::BlockParam { block, .. } => self.loops.block_is_invariant(block, self.header),
            ENode::SideEffectResult => self
                .effect_blocks
                .get(&node)
                .is_some_and(|block| self.loops.block_is_invariant(*block, self.header)),
            ENode::Pure { operands, .. } => operands.iter().all(|operand| self.is_invariant(*operand)),
            // Extraction may select either branch, so both alternatives must
            // be invariant before the union is safe to classify as invariant.
            ENode::Union { left, right } => self.is_invariant(left) && self.is_invariant(right),
        };
        self.memo.insert(node, invariant);
        invariant
    }
}

/// DFS the skeleton from `header`, stopping at `merge`. The header itself
/// is included; `merge` is not.
fn collect_loop_body<P: EgirPhase>(
    skeleton: &Skeleton<P>,
    header: BlockId,
    merge: BlockId,
) -> LookupSet<BlockId> {
    let mut body = LookupSet::new();
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
