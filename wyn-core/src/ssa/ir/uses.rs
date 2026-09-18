//! Derived SSA use information that cannot become stale after rewrites.

use super::{Function, InstId, ValueId, ValueRef, VisitValues};
use crate::flow::Terminator;
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
    pub fn analyze<I: VisitValues, T>(function: &Function<I, T>) -> Self {
        let mut result = Self::default();
        for (instruction, node) in &function.insts {
            for (operand, value) in node.data.values().into_iter().enumerate() {
                result.record(value, UseSite::Instruction { instruction, operand });
            }
        }
        for block in function.blocks.values() {
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

    pub fn counts(&self) -> impl Iterator<Item = (ValueId, usize)> + '_ {
        self.users.iter().map(|(&value, users)| (value, users.len()))
    }

    pub fn is_used_once(&self, value: ValueId) -> bool {
        self.count(value) == 1
    }
}
