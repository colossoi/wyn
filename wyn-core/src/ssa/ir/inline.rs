//! Representation-level cloning used by SSA inlining policies.

use super::{BlockId, Function, ValueRef, VisitValues};
use std::collections::HashMap;

/// Clone a single-block function into `caller` and return its mapped result.
/// Policy decisions such as size, purity, and linkage remain with the caller.
pub(crate) fn inline_single_block<I, T>(
    caller: &mut Function<I, T>,
    block: BlockId,
    callee: &Function<I, T>,
    arguments: &[ValueRef],
) -> Option<ValueRef>
where
    I: Clone + VisitValues,
    T: Clone,
{
    if callee.blocks.len() != 1 || callee.params.len() != arguments.len() {
        return None;
    }
    let super::Terminator::Return(Some(returned)) = callee.blocks[callee.entry].term else {
        return None;
    };
    let mut values: HashMap<_, _> = callee.params.iter().copied().zip(arguments.iter().copied()).collect();
    for &instruction in &callee.blocks[callee.entry].insts {
        let node = &callee.insts[instruction];
        let Some(result) = node.result else {
            return None;
        };
        let mut data = node.data.clone();
        data.visit_values_mut(&mut |value| {
            if let ValueRef::Ssa(id) = value {
                let Some(&mapped) = values.get(id) else {
                    unreachable!("single-block SSA operand must precede its use")
                };
                *value = mapped;
            }
        });
        let mapped = caller.append_inst_with_span(block, data, callee.values[result].ty.clone(), node.span);
        values.insert(result, mapped.into());
    }
    match returned {
        ValueRef::Ssa(value) => values.get(&value).copied(),
        constant => Some(constant),
    }
}
