//! Value liveness across instructions, block parameters, and incoming edges.
//! The caller supplies discardability; control flow is preserved except for
//! empty, terminating selections with no live merge arguments.

use super::{BlockId, Function, Terminator, ValueDef, ValueId, ValueRef, VisitValues};
use crate::flow::ControlHeader;
use crate::{LookupMap, LookupSet};

#[cfg(test)]
#[path = "dce_tests.rs"]
mod tests;

pub(crate) fn eliminate_dead_values<I: VisitValues, T>(
    function: &mut Function<I, T>,
    is_discardable: impl Fn(&I) -> bool,
) {
    loop {
        sweep_values(function, &is_discardable);
        if !collapse_empty_selections(function) {
            break;
        }
    }
}

fn sweep_values<I: VisitValues, T>(function: &mut Function<I, T>, is_discardable: &impl Fn(&I) -> bool) {
    let mut incoming: LookupMap<ValueId, Vec<ValueRef>> = LookupMap::new();
    let mut pending = Vec::new();
    for node in function.insts.values() {
        if node.result.is_none() || !is_discardable(&node.data) {
            pending.extend(node.result.map(ValueRef::Ssa));
            pending.extend(node.data.values());
        }
    }
    let mut edge = |target: BlockId, args: &[ValueRef]| {
        for (&param, &arg) in function.blocks[target].params.iter().zip(args) {
            incoming.entry(param).or_default().push(arg);
        }
    };
    for block in function.blocks.values() {
        match &block.term {
            Terminator::Branch { target, args } => edge(*target, args),
            Terminator::CondBranch {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => {
                // Conditions remain live even for loops without used results:
                // removing them could change whether the program terminates.
                pending.push(*cond);
                edge(*then_target, then_args);
                edge(*else_target, else_args);
            }
            Terminator::Return(value) => pending.extend(value),
            Terminator::Unreachable => {}
        }
    }

    let mut live = LookupSet::new();
    while let Some(value) = pending.pop() {
        let ValueRef::Ssa(value) = value else { continue };
        if !live.insert(value) {
            continue;
        }
        match function.values[value].def {
            ValueDef::Inst { inst } => pending.extend(function.insts[inst].data.values()),
            ValueDef::Param { .. } => {
                if let Some(args) = incoming.get(&value) {
                    pending.extend(args);
                }
            }
            ValueDef::FunctionParam { .. } => {}
        }
    }

    function.insts.retain(|_, node| {
        if let Some(value) = node.result {
            if !live.contains(&value) {
                function.values.remove(value);
                return false;
            }
        }
        true
    });
    // Keep signatures intact; only internal block parameters are pruned.
    let retained: LookupMap<_, Vec<_>> = function
        .blocks
        .iter()
        .map(|(id, block)| (id, block.params.iter().map(|v| live.contains(v)).collect()))
        .collect();
    let retain_args = |target: BlockId, args: &mut Vec<ValueRef>| {
        let mask = &retained[&target];
        let mut index = 0;
        args.retain(|_| {
            let keep = mask[index];
            index += 1;
            keep
        });
    };
    for (id, block) in &mut function.blocks {
        block.insts.retain(|id| function.insts.contains_key(*id));
        block.params.retain(|value| {
            if live.contains(value) {
                true
            } else {
                function.values.remove(*value);
                false
            }
        });
        for (index, &value) in block.params.iter().enumerate() {
            function.values[value].def = ValueDef::Param { block: id, index };
        }
        match &mut block.term {
            Terminator::Branch { target, args } => retain_args(*target, args),
            Terminator::CondBranch {
                then_target,
                then_args,
                else_target,
                else_args,
                ..
            } => {
                retain_args(*then_target, then_args);
                retain_args(*else_target, else_args);
            }
            Terminator::Return(_) | Terminator::Unreachable => {}
        }
    }
}

fn collapse_empty_selections<I, T>(function: &mut Function<I, T>) -> bool {
    let headers = function.blocks.keys().collect::<Vec<_>>();
    let mut changed = false;
    for header in headers {
        let Some(block) = function.blocks.get(header) else {
            continue;
        };
        let Some(ControlHeader::Selection { merge }) = block.control_header else {
            continue;
        };
        if !function.blocks[merge].params.is_empty() {
            continue;
        }
        let Terminator::CondBranch {
            then_target,
            else_target,
            ..
        } = block.term
        else {
            continue;
        };
        let Some(mut removed) = empty_path(function, then_target, merge, header) else {
            continue;
        };
        let Some(other) = empty_path(function, else_target, merge, header) else {
            continue;
        };
        removed.extend(other);
        // Do not remove shared blocks or another construct's structural targets.
        let shared = function.blocks.iter().any(|(id, block)| {
            if id == header || removed.contains(&id) {
                return false;
            }
            block.term.successors().iter().any(|id| removed.contains(id))
                || match &block.control_header {
                    Some(ControlHeader::Selection { merge }) => removed.contains(merge),
                    Some(ControlHeader::Loop {
                        merge,
                        continue_block,
                    }) => removed.contains(merge) || removed.contains(continue_block),
                    None => false,
                }
        });
        if shared {
            continue;
        }
        function.blocks[header].term = Terminator::Branch {
            target: merge,
            args: vec![],
        };
        function.blocks[header].control_header = None;
        function.blocks.retain(|id, _| !removed.contains(&id));
        changed = true;
    }
    changed
}

/// Only empty linear paths qualify. Loops, returns, effects, and nested
/// control flow remain unless another iteration has safely removed them.
fn empty_path<I, T>(
    function: &Function<I, T>,
    mut block: BlockId,
    merge: BlockId,
    header: BlockId,
) -> Option<LookupSet<BlockId>> {
    let mut path = LookupSet::new();
    while block != merge {
        if block == header || block == function.entry || !path.insert(block) {
            return None;
        }
        let node = &function.blocks[block];
        if !node.insts.is_empty() || !node.params.is_empty() || node.control_header.is_some() {
            return None;
        }
        let Terminator::Branch { target, .. } = node.term else {
            return None;
        };
        block = target;
    }
    Some(path)
}
