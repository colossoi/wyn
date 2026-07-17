//! Tests for `ssa::layout` — currently the `vertex_format` mapping
//! that drives `#[vertex_slot(n)]` vertex-input attribute formats.

use super::vertex_format;
use crate::ast::{Type, TypeName};
use wyn_pipeline_descriptor::VertexFormat;

fn scalar(tn: TypeName) -> Type {
    Type::Constructed(tn, vec![])
}

fn vec(elem: TypeName, n: usize) -> Type {
    Type::Constructed(
        TypeName::Vec,
        vec![scalar(elem), Type::Constructed(TypeName::Size(n), vec![])],
    )
}

#[test]
fn vertex_format_32bit_scalars() {
    assert_eq!(
        vertex_format(&scalar(TypeName::Float(32))),
        Some(VertexFormat::Float32)
    );
    assert_eq!(
        vertex_format(&scalar(TypeName::Int(32))),
        Some(VertexFormat::Sint32)
    );
    assert_eq!(
        vertex_format(&scalar(TypeName::UInt(32))),
        Some(VertexFormat::Uint32)
    );
}

#[test]
fn vertex_format_f32_vectors() {
    assert_eq!(
        vertex_format(&vec(TypeName::Float(32), 2)),
        Some(VertexFormat::Float32x2)
    );
    assert_eq!(
        vertex_format(&vec(TypeName::Float(32), 3)),
        Some(VertexFormat::Float32x3)
    );
    assert_eq!(
        vertex_format(&vec(TypeName::Float(32), 4)),
        Some(VertexFormat::Float32x4)
    );
}

#[test]
fn vertex_format_int_vectors() {
    assert_eq!(
        vertex_format(&vec(TypeName::Int(32), 3)),
        Some(VertexFormat::Sint32x3)
    );
    assert_eq!(
        vertex_format(&vec(TypeName::UInt(32), 2)),
        Some(VertexFormat::Uint32x2)
    );
}

#[test]
fn vertex_format_rejects_non_32bit_scalars() {
    assert_eq!(vertex_format(&scalar(TypeName::Float(64))), None);
    assert_eq!(vertex_format(&scalar(TypeName::Float(16))), None);
    assert_eq!(vertex_format(&scalar(TypeName::Int(64))), None);
    assert_eq!(vertex_format(&scalar(TypeName::UInt(8))), None);
}

#[test]
fn vertex_format_rejects_bad_vector_widths() {
    // vec1 and vec>4 are not vertex formats.
    assert_eq!(vertex_format(&vec(TypeName::Float(32), 1)), None);
    assert_eq!(vertex_format(&vec(TypeName::Float(32), 5)), None);
    // A vector of a non-32-bit element is also rejected.
    assert_eq!(vertex_format(&vec(TypeName::Float(64), 3)), None);
}

#[test]
fn vertex_format_rejects_aggregates() {
    // [4]f32 array.
    let arr = Type::Constructed(
        TypeName::Array,
        vec![
            scalar(TypeName::Float(32)),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Constructed(TypeName::Size(4), vec![]),
            crate::types::no_buffer(),
        ],
    );
    assert_eq!(vertex_format(&arr), None);
    // (f32, f32) tuple.
    let tup = Type::Constructed(
        TypeName::Tuple(2),
        vec![scalar(TypeName::Float(32)), scalar(TypeName::Float(32))],
    );
    assert_eq!(vertex_format(&tup), None);
    // Unit.
    assert_eq!(vertex_format(&scalar(TypeName::Unit)), None);
}

// ---- block_layout ---------------------------------------------------------

use super::{block_layout, std430_alignment, BlockLayout};
use crate::interface::StorageLayout;
use crate::types::RecordFields;

fn record(fields: &[(&str, Type)]) -> Type {
    Type::Constructed(
        TypeName::Record(RecordFields::new(
            fields.iter().map(|(n, _)| n.to_string()).collect(),
        )),
        fields.iter().map(|(_, t)| t.clone()).collect(),
    )
}

fn f32t() -> Type {
    scalar(TypeName::Float(32))
}

fn vecn(n: usize) -> Type {
    vec(TypeName::Float(32), n)
}

#[test]
fn block_layout_scalar_then_vec2_pads_to_vec2_alignment() {
    let ty = record(&[("a", f32t()), ("b", vecn(2))]);
    let l = block_layout(&ty, StorageLayout::Std430).expect("supported");
    assert_eq!(l.member_offsets, vec![0, 8]);
    assert_eq!(l.size, 16);
    assert_eq!(l.align, 8);
}

#[test]
fn block_layout_vec3_then_scalar_packs_into_padding() {
    let ty = record(&[("a", vecn(3)), ("b", f32t())]);
    let l = block_layout(&ty, StorageLayout::Std430).expect("supported");
    assert_eq!(l.member_offsets, vec![0, 12]);
    assert_eq!(l.size, 16);
    assert_eq!(l.align, 16);
}

#[test]
fn block_layout_std140_rounds_size_to_16() {
    let ty = record(&[("a", f32t()), ("b", f32t()), ("c", f32t()), ("d", vecn(2))]);
    let l430 = block_layout(&ty, StorageLayout::Std430).expect("supported");
    assert_eq!(l430.member_offsets, vec![0, 4, 8, 16]);
    assert_eq!(l430.size, 24);
    let l140 = block_layout(&ty, StorageLayout::Std140).expect("supported");
    assert_eq!(l140.member_offsets, l430.member_offsets);
    assert_eq!(l140.size, 32);
}

#[test]
fn block_layout_bare_scalar_and_vector() {
    assert_eq!(
        block_layout(&f32t(), StorageLayout::Std140),
        Some(BlockLayout {
            size: 16,
            align: 4,
            member_offsets: vec![0],
        })
    );
    assert_eq!(
        block_layout(&vecn(4), StorageLayout::Std430),
        Some(BlockLayout {
            size: 16,
            align: 16,
            member_offsets: vec![0],
        })
    );
}

#[test]
fn block_layout_offsets_agree_across_rule_sets() {
    // For the supported member set the rule sets differ only in size
    // rounding — pin the offset agreement the WGSL backend relies on.
    let cases = [
        record(&[("a", f32t()), ("b", vecn(2)), ("c", vecn(3)), ("d", f32t())]),
        record(&[("a", vecn(4)), ("b", scalar(TypeName::UInt(32)))]),
        record(&[("a", scalar(TypeName::Int(32))), ("b", vecn(2))]),
    ];
    for ty in cases {
        let a = block_layout(&ty, StorageLayout::Std140).expect("supported");
        let b = block_layout(&ty, StorageLayout::Std430).expect("supported");
        assert_eq!(a.member_offsets, b.member_offsets, "offsets differ for {ty:?}");
    }
}

#[test]
fn block_layout_rejects_unsupported_members() {
    let matrix = Type::Constructed(
        TypeName::Mat,
        vec![
            f32t(),
            Type::Constructed(TypeName::Size(3), vec![]),
            Type::Constructed(TypeName::Size(3), vec![]),
        ],
    );
    let array = Type::Constructed(
        TypeName::Array,
        vec![
            f32t(),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Constructed(TypeName::Size(4), vec![]),
            Type::Constructed(TypeName::NoBuffer, vec![]),
        ],
    );
    let nested = record(&[("inner", record(&[("x", f32t())]))]);
    for ty in [
        scalar(TypeName::Bool),
        scalar(TypeName::Float(16)),
        matrix.clone(),
        array.clone(),
        nested,
        record(&[("a", f32t()), ("m", matrix)]),
        record(&[("a", f32t()), ("arr", array.clone())]),
    ] {
        assert_eq!(
            block_layout(&ty, StorageLayout::Std140),
            None,
            "must reject {ty:?}"
        );
    }
    // Std430 additionally supports fixed arrays of supported members
    // (SOAC tuple elements like (u32, [4]u32)); std140 does not.
    let with_array = record(&[("n", scalar(TypeName::UInt(32))), ("taps", array)]);
    assert_eq!(block_layout(&with_array, StorageLayout::Std140), None);
    let l = block_layout(&with_array, StorageLayout::Std430).expect("std430 supports fixed arrays");
    assert_eq!(l.member_offsets, vec![0, 4]);
    assert_eq!(l.size, 20);
}

#[test]
fn std430_alignment_of_struct_is_max_member_alignment() {
    assert_eq!(
        std430_alignment(&record(&[("a", f32t()), ("b", vecn(2))])),
        Some(8)
    );
    assert_eq!(
        std430_alignment(&record(&[("a", f32t()), ("b", vecn(3))])),
        Some(16)
    );
    assert_eq!(std430_alignment(&record(&[("a", f32t())])), Some(4));
}
