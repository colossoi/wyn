//! Wyn-specific cleanup driven by generic SSA use information.

use crate::builtins::{by_id, catalog, BuiltinLowering, Purity};
use crate::op::OpTag;
pub use crate::ssa::ir::{UseSite, ValueUses};
use crate::ssa::types::{FuncBody, InstKind};
use crate::LookupMap;

/// Remove recursively dead, side-effect-free SSA instructions.
///
/// Recognizes structural operations and explicitly discardable intrinsics.
/// Calls, opaque intrinsics, storage operations, and place operations remain.
/// Discardability does not grant permission to move or speculate an operation.
pub fn eliminate_dead_pure_instructions(body: &mut FuncBody) {
    let uses = ValueUses::analyze(&body.inner);
    let mut counts: LookupMap<_, _> = uses.counts().collect();
    let mut pending: Vec<_> = body
        .inner
        .insts
        .iter()
        .filter_map(|(id, node)| {
            (counts.get(&node.result?).copied().unwrap_or(0) == 0 && is_discardable(&node.data))
                .then_some(id)
        })
        .collect();
    while let Some(id) = pending.pop() {
        let Some(node) = body.inner.insts.remove(id) else {
            continue;
        };
        for operand in node.data.ssa_uses() {
            let count = counts.entry(operand).or_default();
            *count -= 1;
            if *count == 0 {
                if let Some(id) = body.inner.inst_of_value(operand) {
                    if is_discardable(&body.inner.insts[id].data) {
                        pending.push(id);
                    }
                }
            }
        }
        if let Some(result) = node.result {
            body.inner.values.remove(result);
        }
    }
    for block in body.inner.blocks.values_mut() {
        block.insts.retain(|id| body.inner.insts.contains_key(*id));
    }
}

fn is_discardable(instruction: &InstKind) -> bool {
    if let InstKind::Op {
        tag: OpTag::Intrinsic { id, overload_idx },
        ..
    } = instruction
    {
        let builtin = by_id(*id);
        if builtin.raw.purity != Purity::Pure {
            return false;
        }
        let Some(overload) = builtin.overloads().get(*overload_idx) else {
            return false;
        };
        return match overload.lowering {
            BuiltinLowering::PrimOp(_) | BuiltinLowering::ExtInstSplat { .. } => true,
            // Catalog purity alone is insufficient: array_with, for example,
            // can write through a storage view. Only allow known read-only
            // texture operations from the opaque backend dispatch family.
            BuiltinLowering::ByBuiltinId => {
                let known = catalog().known();
                *id == known.texture_load || *id == known.texture_sample
            }
            BuiltinLowering::LinkedSpirv(_) | BuiltinLowering::NotLowered => false,
        };
    }
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
