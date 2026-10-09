//! Prove local value-update reuse after shared storage placement.

use super::ir::{LoopScopes, UseSite, ValueDef, ValueUses};
use super::storage::{contains_array, is_local_value, ResultStorage, StorageUse};
use super::types::{BlockId, FuncBody, InstId, InstKind};
use crate::builtins::catalog;
use crate::op::OpTag;
use crate::types::TypeExt;
use crate::{LookupMap, LookupSet};
use wyn_graph::DominatorTree;

pub(super) fn apply(program: &mut super::stage::Reachable) {
    for body in program
        .functions
        .iter_mut()
        .map(|f| &mut f.body)
        .chain(program.entry_points.iter_mut().map(|e| &mut e.body))
        .chain(program.constants.iter_mut().map(|c| &mut c.body))
    {
        promote_updates(body);
    }
}

fn promote_updates(body: &mut FuncBody) {
    let known = catalog().known();
    if !body.inner.insts.values().any(|node| {
        matches!(node.data, InstKind::Op { tag: OpTag::Intrinsic { id, .. }, .. }
            if id == known.array_with)
    }) {
        return;
    }
    let uses = ValueUses::analyze(&body.inner);
    let dominators = DominatorTree::build(body.inner.entry, |block, successors| {
        successors.extend(body.inner.blocks[block].term.successors());
    });
    let loops = LoopScopes::with_dominators(&body.inner, &dominators);
    let positions: LookupMap<InstId, (BlockId, usize)> = dominators
        .preorder()
        .iter()
        .flat_map(|&block| {
            body.inner.blocks[block].insts.iter().enumerate().map(move |(i, &id)| (id, (block, i + 1)))
        })
        .collect();
    let before = |(a, i), (b, j)| if a == b { i < j } else { dominators.dominates(a, b) };
    let instructions: Vec<_> = dominators
        .preorder()
        .iter()
        .flat_map(|&block| body.inner.blocks[block].insts.iter().copied())
        .collect();
    'updates: for instruction in instructions {
        let node = &body.inner.insts[instruction];
        let InstKind::Op {
            tag: OpTag::Intrinsic { id, .. },
            operands,
        } = &node.data
        else {
            continue;
        };
        if *id != known.array_with || operands.len() != 3 {
            continue;
        }
        let (Some(source), Some(result)) = (operands[0].as_ssa(), node.result) else {
            continue;
        };
        let ty = body.get_value_type(source);
        if !ty.is_array() || !is_local_value(ty) || ty != body.get_value_type(result) {
            continue;
        }
        let writer = positions[&instruction];
        let mut pending = vec![source];
        let mut visited = LookupSet::new();
        let mut has_origin = false;
        while let Some(value) = pending.pop() {
            if !visited.insert(value) {
                continue;
            }
            if !is_local_value(body.get_value_type(value)) {
                continue 'updates;
            }
            let origin = match body.inner.values[value].def {
                // Parameters are allocated once per call, even if entry is a loop header.
                ValueDef::FunctionParam { .. } => Some(((body.inner.entry, 0), None)),
                ValueDef::Param { block, .. } => Some(((block, 0), loops.scope(block))),
                ValueDef::Inst { inst } => {
                    let Some(&definition) = positions.get(&inst) else {
                        continue 'updates;
                    };
                    let data = &body.inner.insts[inst].data;
                    match data.result_storage() {
                        ResultStorage::Fresh => Some((definition, loops.scope(definition.0))),
                        ResultStorage::Unknown => continue 'updates,
                        storage => {
                            let mut found = false;
                            for (index, operand) in data.value_uses().into_iter().enumerate() {
                                if storage.aliases_operand(index) {
                                    if let Some(alias) =
                                        operand.as_ssa().filter(|&v| contains_array(body.get_value_type(v)))
                                    {
                                        pending.push(alias);
                                        found = true;
                                    }
                                }
                            }
                            if !found {
                                continue 'updates;
                            }
                            None
                        }
                    }
                }
            };
            if let Some((definition, scope)) = origin {
                if scope != loops.scope(writer.0) || !before(definition, writer) {
                    continue 'updates;
                }
                has_origin = true;
            }
            for usage in uses.users(value) {
                let UseSite::Instruction {
                    instruction: user,
                    operand,
                } = *usage
                else {
                    continue 'updates;
                };
                let data = &body.inner.insts[user].data;
                match data.storage_use(operand) {
                    StorageUse::Escape => continue 'updates,
                    StorageUse::Read
                        if user != instruction
                            && !positions.get(&user).is_some_and(|&at| before(at, writer)) =>
                    {
                        continue 'updates
                    }
                    _ => {}
                }
                if data.result_storage().aliases_operand(operand) {
                    if let Some(alias) =
                        body.inner.insts[user].result.filter(|&v| contains_array(body.get_value_type(v)))
                    {
                        pending.push(alias);
                    }
                }
            }
        }
        if has_origin {
            let InstKind::Op {
                tag: OpTag::Intrinsic { id, .. },
                ..
            } = &mut body.inner.insts[instruction].data
            else {
                unreachable!()
            };
            // The next traversal sees this result as an alias, not a fresh allocation.
            *id = known.array_with_in_place;
        }
    }
}

#[cfg(test)]
#[path = "ownership_tests.rs"]
mod tests;
