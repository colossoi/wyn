use super::*;
use crate::diags;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn i32_ty() -> Type {
    Type::Constructed(TypeName::Int(32), vec![])
}
fn f32_ty() -> Type {
    Type::Constructed(TypeName::Float(32), vec![])
}
fn arrow(a: Type, b: Type) -> Type {
    Type::Constructed(TypeName::Arrow, vec![a, b])
}

#[test]
fn raster_formats_as_a_spellable_source_type() {
    let raster = Type::Constructed(TypeName::Raster, vec![f32_ty()]);
    assert_eq!(format_type(&raster), "raster<f32>");
    assert_eq!(diags::format_type(&raster), "raster<f32>");
}

#[test]
fn render_target_view_exposes_identity_but_formatting_hides_it() {
    let resource = Type::Variable(7);
    let target = Type::Constructed(TypeName::RenderTarget, vec![f32_ty(), resource.clone()]);

    let view = target.as_render_target().expect("well-formed render target");
    assert_eq!(view.color, &f32_ty());
    assert_eq!(view.resource, &resource);
    assert_eq!(format_type(&target), "render_target<f32>");
    assert_eq!(diags::format_type(&target), "render_target<f32>");
}

fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn record_fields_hash_matches_order_independent_equality() {
    let left = RecordFields::new(vec!["x".to_string(), "y".to_string()]);
    let right = RecordFields::new(vec!["y".to_string(), "x".to_string()]);

    assert_eq!(left, right);
    assert_eq!(hash_value(&left), hash_value(&right));

    let duplicate = RecordFields::new(vec!["x".to_string(), "x".to_string()]);
    let distinct = RecordFields::new(vec!["x".to_string(), "y".to_string()]);

    assert_ne!(duplicate, distinct);
    assert_ne!(distinct, duplicate);
}

#[test]
fn extract_function_signature_chains_arrows_in_order() {
    // `i32 -> f32 -> i32` -> ([i32, f32], i32)
    let ty = arrow(i32_ty(), arrow(f32_ty(), i32_ty()));
    let (params, ret) = extract_function_signature(&ty);
    assert_eq!(params, vec![i32_ty(), f32_ty()]);
    assert_eq!(ret, i32_ty());
}

#[test]
fn extract_function_signature_on_non_arrow_returns_empty_params() {
    let (params, ret) = extract_function_signature(&i32_ty());
    assert!(params.is_empty());
    assert_eq!(ret, i32_ty());
}

fn rank1_arr(size: usize) -> Type {
    Type::Constructed(
        TypeName::Array,
        vec![
            f32_ty(),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Constructed(TypeName::Size(size), vec![]),
            no_buffer(),
        ],
    )
}

fn size(n: usize) -> Type {
    Type::Constructed(TypeName::Size(n), vec![])
}

#[test]
fn tensor_view_separates_shape_from_array_storage() {
    let array = Type::Constructed(
        TypeName::Array,
        vec![
            f32_ty(),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            size(3),
            size(4),
            no_buffer(),
        ],
    );

    let tensor = array.as_tensor().unwrap();
    assert_eq!(tensor.elem, &f32_ty());
    assert_eq!(tensor.rank(), 2);
    assert_eq!(tensor.concrete_dim(0), Some(3));
    assert_eq!(tensor.concrete_dim(1), Some(4));

    let storage = array.array_storage().unwrap();
    assert!(matches!(
        storage.variant,
        Type::Constructed(TypeName::ArrayVariantComposite, _)
    ));
    assert_eq!(storage.region, &no_buffer());
}

#[test]
fn tensor_view_rejects_malformed_shapes() {
    let malformed_vector = Type::Constructed(TypeName::Vec, vec![f32_ty()]);
    let malformed_array = Type::Constructed(
        TypeName::Array,
        vec![
            f32_ty(),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            no_buffer(),
        ],
    );

    assert!(malformed_vector.as_tensor().is_none());
    assert!(malformed_array.as_tensor().is_none());
    assert!(malformed_array.array_storage().is_none());
}

#[test]
fn array_rank_reports_dim_count() {
    assert_eq!(rank1_arr(8).array_rank(), Some(1));
    assert_eq!(i32_ty().array_rank(), None);
}

#[test]
fn array_dims_returns_dim_slice() {
    let arr = rank1_arr(8);
    let dims = arr.array_dims().unwrap();
    assert_eq!(dims.len(), 1);
    assert!(matches!(&dims[0], Type::Constructed(TypeName::Size(8), _)));
}

#[test]
fn array_dim_indexes_dim_slice() {
    let arr = rank1_arr(8);
    assert!(matches!(
        arr.array_dim(0),
        Some(Type::Constructed(TypeName::Size(8), _))
    ));
    assert!(arr.array_dim(1).is_none());
}

#[test]
fn array_dim_zero_matches_array_size() {
    let arr = rank1_arr(8);
    assert_eq!(arr.array_dim(0), arr.array_size());
}

#[test]
fn diet_consuming_stops_at_function_arrows() {
    // A `*` nested in a tuple makes the whole aggregate consuming...
    let tuple_with_unique = Diet::Aggregate {
        unique: false,
        components: vec![Diet::Leaf(true), Diet::Leaf(false)],
    };
    assert!(tuple_with_unique.is_consuming());

    // ...but a `*` behind an arrow does not make the callback value
    // consuming (it is the callback's own parameter contract).
    let callback = Diet::Arrow(Box::new(Diet::Leaf(true)), Box::new(Diet::Leaf(false)));
    assert!(!callback.is_consuming());
    assert!(callback.mentions_consuming_function());
}

#[test]
fn diet_observing_mentions_no_consuming_function() {
    let observing = Diet::Aggregate {
        unique: false,
        components: vec![Diet::Leaf(false), Diet::Leaf(false)],
    };
    assert!(!observing.is_consuming());
    assert!(!observing.mentions_consuming_function());
}
