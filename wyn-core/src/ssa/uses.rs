//! Derived SSA value-use information.
//!
//! This is intentionally an analysis rather than mutable state on `FuncBody`:
//! rewrites cannot leave cached use counts stale, and consumers pay only one
//! linear walk when they need the information.

use crate::op::OpTag;
use crate::ssa::framework::InstId;
use crate::ssa::types::{FuncBody, InstKind, Terminator, ValueId, ValueRef};
use crate::LookupMap;

/// The location of one SSA value operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UseSite {
    Instruction {
        instruction: InstId,
        operand: usize,
    },
    Terminator,
}

/// Immutable use counts and use sites for the values in one SSA body.
#[derive(Clone, Debug, Default)]
pub struct ValueUses {
    users: LookupMap<ValueId, Vec<UseSite>>,
}

impl ValueUses {
    /// Analyze every instruction and CFG terminator operand in `body`.
    pub fn analyze(body: &FuncBody) -> Self {
        let mut result = Self::default();
        for (instruction, node) in &body.inner.insts {
            for (operand, value) in node.data.value_uses().into_iter().enumerate() {
                result.record(value, UseSite::Instruction { instruction, operand });
            }
        }
        for block in body.inner.blocks.values() {
            match &block.term {
                Terminator::Branch { args, .. } => {
                    for value in args {
                        result.record(*value, UseSite::Terminator);
                    }
                }
                Terminator::CondBranch {
                    cond,
                    then_args,
                    else_args,
                    ..
                } => {
                    result.record(*cond, UseSite::Terminator);
                    for value in then_args.iter().chain(else_args) {
                        result.record(*value, UseSite::Terminator);
                    }
                }
                Terminator::Return(Some(value)) => result.record(*value, UseSite::Terminator),
                Terminator::Return(None) | Terminator::Unreachable => {}
            }
        }
        result
    }

    fn record(&mut self, value: ValueRef, site: UseSite) {
        if let ValueRef::Ssa(value) = value {
            self.users.entry(value).or_default().push(site);
        }
    }

    pub fn count(&self, value: ValueId) -> usize {
        self.users.get(&value).map_or(0, Vec::len)
    }

    pub fn users(&self, value: ValueId) -> &[UseSite] {
        self.users.get(&value).map_or(&[], Vec::as_slice)
    }

    pub fn is_used_once(&self, value: ValueId) -> bool {
        self.count(value) == 1
    }
}

/// Remove recursively dead, side-effect-free SSA instructions.
///
/// This deliberately recognizes only operations whose purity is structural.
/// Calls, intrinsics, storage operations, and place operations are retained;
/// broadening that set belongs with an explicit effect classification.
pub fn eliminate_dead_pure_instructions(body: &mut FuncBody) {
    loop {
        let uses = ValueUses::analyze(body);
        let dead = body
            .inner
            .insts
            .iter()
            .filter_map(|(instruction, node)| {
                let result = node.result?;
                (uses.count(result) == 0 && is_structurally_pure(&node.data))
                    .then_some((instruction, result))
            })
            .collect::<Vec<_>>();
        if dead.is_empty() {
            return;
        }
        for block in body.inner.blocks.values_mut() {
            block.insts.retain(|instruction| !dead.iter().any(|(dead, _)| dead == instruction));
        }
        for (instruction, result) in dead {
            body.inner.insts.remove(instruction);
            body.inner.values.remove(result);
        }
    }
}

fn is_structurally_pure(instruction: &InstKind) -> bool {
    matches!(
        instruction,
        InstKind::Op {
            tag: OpTag::Int(_)
                | OpTag::Uint(_)
                | OpTag::Float(_)
                | OpTag::Bool(_)
                | OpTag::Unit
                | OpTag::Global(_)
                | OpTag::BinOp(_)
                | OpTag::UnaryOp(_)
                | OpTag::Tuple(_)
                | OpTag::Vector(_)
                | OpTag::Matrix { .. }
                | OpTag::ArrayLit(_)
                | OpTag::ArrayRange { .. }
                | OpTag::Project { .. }
                | OpTag::Index
                | OpTag::Materialize
                | OpTag::AddressableConstant(_)
                | OpTag::DynamicExtract,
            ..
        }
    )
}
