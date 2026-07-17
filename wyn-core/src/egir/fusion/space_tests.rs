use super::*;
use crate::egir::program::SemanticResourceRef;
use crate::egir::types::{NodeId, SegExtent, SegSpace};
use crate::ResourceId;
use slotmap::SlotMap;

fn two_nodes() -> (NodeId, NodeId) {
    let mut sm: SlotMap<NodeId, ()> = SlotMap::with_key();
    (sm.insert(()), sm.insert(()))
}

#[test]
fn fixed_extents_compare_by_value() {
    assert!(seg_extent_fusable(&SegExtent::Fixed(64), &SegExtent::Fixed(64)));
    assert!(!seg_extent_fusable(&SegExtent::Fixed(64), &SegExtent::Fixed(65)));
}

#[test]
fn push_constant_compares_by_offset_ignoring_node() {
    let (a, b) = two_nodes();
    assert!(seg_extent_fusable(
        &SegExtent::PushConstant { node: a, offset: 4 },
        &SegExtent::PushConstant { node: b, offset: 4 },
    ));
    assert!(!seg_extent_fusable(
        &SegExtent::PushConstant { node: a, offset: 4 },
        &SegExtent::PushConstant { node: b, offset: 8 },
    ));
}

#[test]
fn resource_length_compares_by_resource_and_stride_ignoring_node() {
    let (a, b) = two_nodes();
    let mk = |node: NodeId, resource: u32, elem_bytes: u32| SegExtent::ResourceLength {
        node,
        resource: SemanticResourceRef(ResourceId(resource)),
        elem_bytes,
    };
    assert!(seg_extent_fusable(&mk(a, 1, 4), &mk(b, 1, 4)));
    assert!(
        !seg_extent_fusable(&mk(a, 1, 4), &mk(b, 2, 4)),
        "different resource"
    );
    assert!(
        !seg_extent_fusable(&mk(a, 1, 4), &mk(b, 1, 8)),
        "different stride"
    );
}

#[test]
fn value_extent_requires_same_node() {
    let (a, b) = two_nodes();
    assert!(seg_extent_fusable(&SegExtent::Value(a), &SegExtent::Value(a)));
    assert!(!seg_extent_fusable(&SegExtent::Value(a), &SegExtent::Value(b)));
}

#[test]
fn mismatched_variants_never_fuse() {
    let (a, _) = two_nodes();
    assert!(!seg_extent_fusable(&SegExtent::Fixed(4), &SegExtent::Value(a)));
}

#[test]
fn spaces_fuse_dimensionwise() {
    let space = |dims| SegSpace { dims };
    assert!(seg_space_fusable(
        &space(vec![SegExtent::Fixed(8)]),
        &space(vec![SegExtent::Fixed(8)]),
    ));
    assert!(!seg_space_fusable(
        &space(vec![SegExtent::Fixed(8)]),
        &space(vec![SegExtent::Fixed(9)]),
    ));
    assert!(
        !seg_space_fusable(
            &space(vec![SegExtent::Fixed(8)]),
            &space(vec![SegExtent::Fixed(8), SegExtent::Fixed(8)]),
        ),
        "different rank never fuses"
    );
}
