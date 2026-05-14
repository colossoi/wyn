//! Tests for `ssa::layout` — currently the `vertex_format` mapping
//! that drives `#[location(n)]` vertex-input attribute formats.

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
            Type::Constructed(TypeName::Size(4), vec![]),
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
