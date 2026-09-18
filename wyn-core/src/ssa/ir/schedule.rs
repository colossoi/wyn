//! Final placement of reachable floating SSA instructions.

use super::{BlockId, Function, InstId, InstPlacement, ValueDef, ValueId, ValueRef, VisitValues};
use crate::flow::ControlHeader;
use crate::{LookupMap, LookupSet};
use thiserror::Error;
use wyn_graph::DominatorTree;

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod tests;

/// Lexical loop containing each reachable control-flow block.
pub(crate) struct LoopScopes {
    scopes: LookupMap<BlockId, Option<BlockId>>,
    parents: LookupMap<BlockId, Option<BlockId>>,
}

impl LoopScopes {
    pub fn analyze<I, T>(function: &Function<I, T>) -> Self {
        let dominators = DominatorTree::build(function.entry, |block, successors| {
            successors.extend(function.blocks[block].term.successors())
        });
        let mut scopes = LookupMap::new();
        let mut parents = LookupMap::new();
        for &block in dominators.preorder() {
            let mut scope = dominators.idom(block).and_then(|parent| scopes[&parent]);
            while let Some(header) = scope {
                let Some(ControlHeader::Loop { merge, .. }) = function.blocks[header].control_header else {
                    break;
                };
                if merge != block {
                    break;
                }
                scope = parents[&header];
            }
            if matches!(
                function.blocks[block].control_header,
                Some(ControlHeader::Loop { .. })
            ) {
                parents.insert(block, scope);
                scope = Some(block);
            }
            scopes.insert(block, scope);
        }
        Self { scopes, parents }
    }

    /// Innermost lexical loop containing a reachable block.
    pub fn scope(&self, block: BlockId) -> Option<BlockId> {
        self.scopes.get(&block).copied().flatten()
    }

    /// Current and enclosing loop scopes, ending with function scope (`None`).
    pub fn enclosing_scopes(&self, block: BlockId) -> impl Iterator<Item = Option<BlockId>> + '_ {
        std::iter::successors(Some(self.scope(block)), |scope| {
            scope.map(|header| self.parents[&header])
        })
    }

    /// Whether a value is defined within a lexical loop scope.
    pub fn value_varies<I, T>(&self, function: &Function<I, T>, value: ValueRef) -> bool {
        let ValueRef::Ssa(value) = value else {
            return false;
        };
        function.block_of_value(value).and_then(|block| self.scope(block)).is_some()
    }
}

/// A malformed floating dependency graph that cannot be assigned to blocks.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub(crate) enum ScheduleError {
    #[error("cycle through floating SSA value {0:?}")]
    Cycle(ValueId),
    #[error("floating SSA operand {0:?} has no legal placement")]
    UnplacedOperand(ValueId),
    #[error("reachable SSA block {0:?} is missing scheduling state")]
    MissingBlock(BlockId),
}

/// Assign reachable floating instructions to the deepest operand block.
///
/// Pinned instructions and terminators are the roots. Floating dependencies
/// are emitted once, in dependency order, after every pinned operand and
/// before the first pinned use that requires them.
pub(crate) fn schedule_floating<I: VisitValues, T>(
    function: &mut Function<I, T>,
) -> Result<(), ScheduleError> {
    let dominators = DominatorTree::build(function.entry, |block, successors| {
        successors.extend(function.blocks[block].term.successors())
    });
    let mut depths = LookupMap::new();
    for &block in dominators.preorder() {
        let depth = if let Some(parent) = dominators.idom(block) {
            let Some(depth) = depths.get(&parent) else {
                return Err(ScheduleError::MissingBlock(parent));
            };
            depth + 1
        } else {
            0
        };
        depths.insert(block, depth);
    }

    let mut visiting = LookupSet::new();

    for &block in dominators.preorder() {
        let instructions = std::mem::take(&mut function.blocks[block].insts);
        for instruction in instructions {
            let operands = function.insts[instruction].data.values();
            for operand in operands {
                schedule_value(function, operand, &depths, &mut visiting)?;
            }
            function.blocks[block].insts.push(instruction);
        }
        let roots = function.blocks[block].term.referenced_nodes();
        for root in roots {
            schedule_value(function, root, &depths, &mut visiting)?;
        }
    }

    let dead = function
        .insts
        .iter()
        .filter_map(|(instruction, node)| {
            matches!(node.placement, InstPlacement::Floating).then_some(instruction)
        })
        .collect::<Vec<_>>();
    for instruction in dead {
        if let Some(node) = function.insts.remove(instruction) {
            if let Some(value) = node.result {
                function.values.remove(value);
            }
        }
    }
    Ok(())
}

fn schedule_value<I: VisitValues, T>(
    function: &mut Function<I, T>,
    value: ValueRef,
    depths: &LookupMap<BlockId, usize>,
    visiting: &mut LookupSet<InstId>,
) -> Result<(), ScheduleError> {
    let ValueRef::Ssa(value) = value else {
        return Ok(());
    };
    let ValueDef::Inst { inst } = function.values[value].def else {
        return Ok(());
    };
    if !matches!(function.insts[inst].placement, InstPlacement::Floating) {
        return Ok(());
    }
    if !visiting.insert(inst) {
        return Err(ScheduleError::Cycle(value));
    }

    let operands = function.insts[inst].data.values();
    for operand in &operands {
        schedule_value(function, *operand, depths, visiting)?;
    }
    let mut destination = function.entry;
    for operand in operands.into_iter().filter_map(ValueRef::as_ssa) {
        let Some(block) = function.block_of_value(operand) else {
            return Err(ScheduleError::UnplacedOperand(operand));
        };
        let (Some(block_depth), Some(destination_depth)) = (depths.get(&block), depths.get(&destination))
        else {
            return Err(ScheduleError::MissingBlock(block));
        };
        if block_depth > destination_depth {
            destination = block;
        }
    }
    function.insts[inst].placement = InstPlacement::Block(destination);
    function.blocks[destination].insts.push(inst);
    visiting.remove(&inst);
    Ok(())
}
