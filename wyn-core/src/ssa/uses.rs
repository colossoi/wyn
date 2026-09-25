//! Wyn-specific cleanup driven by generic SSA use information.

use crate::builtins::{by_id, catalog, BuiltinLowering, Purity};
use crate::op::OpTag;
pub use crate::ssa::ir::{UseSite, ValueUses};
use crate::ssa::types::{FuncBody, InstKind};

/// Remove dead instructions, block parameters, and empty selections.
///
/// Recognizes structural operations and explicitly discardable intrinsics.
/// Calls, opaque intrinsics, memory accesses, and place operations remain.
/// Discardability does not grant permission to move or speculate an operation.
pub fn eliminate_dead_pure_instructions(body: &mut FuncBody) {
    super::ir::eliminate_dead_values(&mut body.inner, is_discardable);
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
            // texture operations and length metadata from the opaque backend
            // dispatch family.
            BuiltinLowering::ByBuiltinId => {
                let known = catalog().known();
                *id == known.texture_load
                    || *id == known.texture_sample
                    || *id == known.storage_len
                    || *id == known.length
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
                | OpTag::DynamicExtract
                | OpTag::StorageView(_)
                | OpTag::StorageViewLen,
            ..
        }
    )
}
