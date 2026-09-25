//! Cleanup of selections whose conditions became constant during SSA folding.

use super::{ConstantValue, Function, InstPlacement, Terminator, ValueDef, ValueRef};
use crate::flow::ControlHeader;
use crate::LookupSet;

#[cfg(test)]
#[path = "control_flow_tests.rs"]
mod tests;

pub(crate) fn fold_constant_selections<I, T>(function: &mut Function<I, T>) {
    let mut changed = false;
    for block in function.blocks.values_mut() {
        // Loop tests can live in separate, unannotated blocks. Simplifying
        // those also requires rebuilding the loop's merge/continue structure.
        let Some(ControlHeader::Selection { .. }) = block.control_header else {
            continue;
        };
        let Terminator::CondBranch {
            cond: ValueRef::Const(ConstantValue::Bool(condition)),
            then_target,
            then_args,
            else_target,
            else_args,
        } = &mut block.term
        else {
            continue;
        };
        let (target, args) = if *condition {
            (*then_target, std::mem::take(then_args))
        } else {
            (*else_target, std::mem::take(else_args))
        };
        block.term = Terminator::Branch { target, args };
        block.control_header = None;
        changed = true;
    }
    if !changed {
        return;
    }

    let mut live = LookupSet::new();
    let mut pending = vec![function.entry];
    while let Some(id) = pending.pop() {
        if !live.insert(id) {
            continue;
        }
        let block = &function.blocks[id];
        pending.extend(block.term.successors());
        // Backends also consume structured targets, even when no ordinary
        // edge reaches them. Only retained headers keep these blocks alive;
        // a loop inside a discarded arm must disappear in its entirety.
        match block.control_header {
            Some(ControlHeader::Selection { merge }) => pending.push(merge),
            Some(ControlHeader::Loop {
                merge,
                continue_block,
            }) => {
                pending.extend([merge, continue_block]);
            }
            None => {}
        }
    }
    function.blocks.retain(|id, _| live.contains(&id));
    function.insts.retain(|_, node| match node.placement {
        InstPlacement::Block(block) => live.contains(&block),
        InstPlacement::Floating => true,
    });
    function.values.retain(|_, value| match value.def {
        ValueDef::Param { block, .. } => live.contains(&block),
        ValueDef::Inst { inst } => function.insts.contains_key(inst),
        ValueDef::FunctionParam { .. } => true,
    });
}
