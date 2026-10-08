//! Concrete SSA storage semantics for representation lowering and local reuse.

use super::types::InstKind;
use crate::builtins::catalog;
use crate::op::OpTag;
use crate::types::{
    is_array_variant_bounded, is_array_variant_composite, is_array_variant_view, Type, TypeExt, TypeName,
};

/// How an instruction obtains storage for an array-containing result.
pub(super) enum ResultStorage {
    /// A value copy with independent local contents (descriptors are excluded).
    Fresh,
    /// The backend may retain the selected operand's storage.
    Alias(usize),
    /// Aggregate construction may retain storage from any operand.
    AliasAll,
    /// No local allocation/alias contract is available.
    Unknown,
}

pub(super) enum StorageUse {
    /// Observes contents at the instruction, without retaining them.
    Read,
    /// Observes only shape, which an element update preserves.
    Metadata,
    /// Forwards storage into the result; its consumers determine liveness.
    Forward,
    /// May retain storage through a call, place, or an unmodeled operation.
    Escape,
}

impl ResultStorage {
    pub(super) fn aliases_operand(&self, index: usize) -> bool {
        matches!(self, Self::AliasAll) || matches!(self, Self::Alias(i) if *i == index)
    }
}

impl InstKind {
    pub(super) fn result_storage(&self) -> ResultStorage {
        match self {
            Self::Load { .. } => ResultStorage::Fresh,
            Self::Op { tag, .. } => match tag {
                OpTag::ArrayLit(_) => ResultStorage::Fresh,
                OpTag::Tuple(_) => ResultStorage::AliasAll,
                OpTag::Materialize | OpTag::Project { .. } | OpTag::Index | OpTag::DynamicExtract => {
                    ResultStorage::Alias(0)
                }
                OpTag::Intrinsic { id, .. } if *id == catalog().known().array_with => ResultStorage::Fresh,
                OpTag::Intrinsic { id, .. } if *id == catalog().known().array_with_in_place => {
                    ResultStorage::Alias(0)
                }
                _ => ResultStorage::Unknown,
            },
            _ => ResultStorage::Unknown,
        }
    }

    /// Classify each operand independently of the result's allocation behavior.
    pub(super) fn storage_use(&self, operand: usize) -> StorageUse {
        match self {
            Self::Op { tag, .. } => match tag {
                OpTag::Materialize | OpTag::Tuple(_) => StorageUse::Forward,
                OpTag::StorageViewLen => StorageUse::Metadata,
                OpTag::Index | OpTag::DynamicExtract | OpTag::Project { .. } | OpTag::ArrayLit(_) => {
                    StorageUse::Read
                }
                OpTag::Intrinsic { id, .. }
                    if operand == 0
                        && (*id == catalog().known().length || *id == catalog().known().storage_len) =>
                {
                    StorageUse::Metadata
                }
                OpTag::Intrinsic { id, .. }
                    if *id == catalog().known().array_with
                        || *id == catalog().known().array_with_in_place =>
                {
                    StorageUse::Read
                }
                _ => StorageUse::Escape,
            },
            _ => StorageUse::Escape,
        }
    }
}

/// View updates already mean writes. Make that explicit when lowering concrete
/// representations, before any optional local-storage reuse is considered.
pub(super) fn lower_array_update<R, C>(tag: &mut OpTag<R, C>, source: &Type) {
    if let OpTag::Intrinsic { id, .. } = tag {
        if *id == catalog().known().array_with && source.array_variant().is_some_and(is_array_variant_view)
        {
            *id = catalog().known().array_with_in_place;
        }
    }
}

pub(super) fn contains_array(ty: &Type) -> bool {
    ty.is_array()
        || matches!(ty,
            Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields)
                if fields.iter().any(contains_array))
}

/// Local value storage can be copied. Storage descriptors and unresolved array
/// representations cannot acquire mutation permission from this analysis.
pub(super) fn is_local_value(ty: &Type) -> bool {
    if let Some(variant) = ty.array_variant() {
        return (is_array_variant_composite(variant) || is_array_variant_bounded(variant))
            && ty.elem_type().is_some_and(is_local_value);
    }
    match ty {
        Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) => {
            fields.iter().all(is_local_value)
        }
        _ => true,
    }
}
