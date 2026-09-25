//! Substitute forwarding block parameters without changing control flow.

use super::{Function, Substitutions, Terminator, ValueDef, ValueRef, VisitValues};
use crate::LookupMap;
use wyn_graph::DominatorTree;

#[cfg(test)]
#[path = "trivial_params_tests.rs"]
mod tests;

/// Remove parameters with exactly one incoming edge and a dominating value.
/// Run after instruction placement. This preserves SSA dominance without
/// changing structured control flow. Textual backends plan lexical declarations
/// separately, so loop exits need no artificial forwarding parameters.
pub(crate) fn eliminate_single_input_params<I: VisitValues, T>(function: &mut Function<I, T>) {
    let mut incoming = LookupMap::new();
    let mut edge = |target, args: &[ValueRef]| {
        incoming.entry(target).and_modify(|args| *args = None).or_insert_with(|| Some(args.to_vec()));
    };
    for block in function.blocks.values() {
        match &block.term {
            Terminator::Branch { target, args } => edge(*target, args),
            Terminator::CondBranch {
                then_target,
                then_args,
                else_target,
                else_args,
                ..
            } => {
                // Two edges from the same predecessor can pass different values.
                edge(*then_target, then_args);
                edge(*else_target, else_args);
            }
            Terminator::Return(_) | Terminator::Unreachable => {}
        }
    }
    let dominators = DominatorTree::build(function.entry, |block, successors| {
        successors.extend(function.blocks[block].term.successors());
    });
    let mut replacements = Substitutions::default();
    let mut retained = LookupMap::new();
    // Dominator order resolves chains even when storage order is different.
    // Unreachable cycles and entry parameters are deliberately left intact.
    for &block in dominators.preorder() {
        if block == function.entry || function.blocks[block].params.is_empty() {
            continue;
        }
        let Some(Some(args)) = incoming.get(&block) else {
            continue;
        };
        let mut mask = Vec::new();
        for (&param, &arg) in function.blocks[block].params.iter().zip(args) {
            let mut arg = arg;
            replacements.resolve(&mut arg);
            let available = match arg {
                ValueRef::Const(_) => true,
                ValueRef::Ssa(value) => function
                    .block_of_value(value)
                    .is_some_and(|producer| producer != block && dominators.dominates(producer, block)),
            };
            if available {
                replacements.insert(param, arg);
            }
            mask.push(!available);
        }
        retained.insert(block, mask);
    }
    let retain_args = |target, args: &mut Vec<ValueRef>| {
        if let Some(mask) = retained.get(&target) {
            let mut keep = mask.iter();
            args.retain(|_| *keep.next().expect("block argument matches parameter"));
        }
    };
    for (block, data) in &mut function.blocks {
        if let Some(mask) = retained.get(&block) {
            let mut keep = mask.iter();
            data.params.retain(|_| *keep.next().expect("block parameter has retention mask"));
            for (index, &param) in data.params.iter().enumerate() {
                function.values[param].def = ValueDef::Param { block, index };
            }
        }
        match &mut data.term {
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
    replacements.finish(function);
}
