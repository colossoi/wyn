//! Unit tests for the pure operand/parameter-filtering logic. The
//! end-to-end behaviour (a real program lowering to SPIR-V/WGSL with no
//! runtime storage-image handle, validated by naga) is covered by the
//! integration tests; these pin the positional-ABI rewrite in isolation.

use super::{filter_smallvec, filter_vec, is_storage_image};
use crate::ast::TypeName;
use crate::egir::from_tlc::ConvertError;
use crate::egir::types::NodeId;
use crate::ssa::types::{ValueId, ValueRef};
use polytype::Type;
use slotmap::SlotMap;
use smallvec::{smallvec, SmallVec};

fn fresh_nodes(n: usize) -> Vec<NodeId> {
    let mut sm: SlotMap<NodeId, ()> = SlotMap::with_key();
    (0..n).map(|_| sm.insert(())).collect()
}

#[test]
fn is_storage_image_sees_through_unique() {
    let img = Type::Constructed(TypeName::StorageTexture, vec![]);
    assert!(is_storage_image(&img));
    // Written resources arrive `*storage_image`; erasure must see through it.
    let unique = Type::Constructed(TypeName::Unique, vec![img]);
    assert!(is_storage_image(&unique));
    // A view-array parameter is a real runtime value, not erasable.
    assert!(!is_storage_image(&Type::Constructed(TypeName::Array, vec![])));
}

#[test]
fn filter_smallvec_drops_masked_operands_preserving_order() {
    let n = fresh_nodes(3);
    let mut ops: SmallVec<[NodeId; 4]> = smallvec![n[0], n[1], n[2]];
    // Erase the middle operand (the storage-image argument).
    filter_smallvec(&mut ops, &[false, true, false], "helper").unwrap();
    assert_eq!(ops.as_slice(), &[n[0], n[2]]);
}

#[test]
fn filter_smallvec_rejects_arity_mismatch() {
    let n = fresh_nodes(2);
    let mut ops: SmallVec<[NodeId; 4]> = smallvec![n[0], n[1]];
    let err = filter_smallvec(&mut ops, &[false], "helper").unwrap_err();
    assert!(matches!(err, ConvertError::Internal(_)));
}

#[test]
fn filter_vec_drops_masked_value_refs() {
    let mut ops = vec![
        ValueRef::Ssa(ValueId::default()),
        ValueRef::Ssa(ValueId::default()),
    ];
    filter_vec(&mut ops, &[true, false], "helper").unwrap();
    assert_eq!(ops.len(), 1);
}
