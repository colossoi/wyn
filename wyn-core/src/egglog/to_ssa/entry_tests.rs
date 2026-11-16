use super::{remap_saved, saved_slots, scalar_source, ArrayLocation, Extent, ExtentValue, Location};
use crate::egglog::to_ssa::Body;
use crate::host::{Expr, ScalarExpr, ScalarType};
use crate::types::{self, Type, TypeName};
use crate::{BindingRef, LookupMap};

fn array(binding: BindingRef, offset: ExtentValue, length: ExtentValue) -> Location {
    Location::Array(ArrayLocation {
        binding: Some(binding),
        element: types::i32(),
        offset: Extent {
            value: offset,
            host: None,
        },
        dimensions: vec![(
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            Extent {
                value: length,
                host: None,
            },
        )],
    })
}

#[test]
fn dispatch_transport_requires_materialized_extents() {
    let location = array(
        BindingRef::new(0, 0),
        ExtentValue::Constant(0),
        ExtentValue::Local(Body::number(4)),
    );
    assert!(!location.persistent());
    assert!(saved_slots(&location).is_err());
    let location = array(
        BindingRef::new(0, 0),
        ExtentValue::Constant(0),
        ExtentValue::Constant(4),
    );
    assert!(location.persistent());
    assert_eq!(saved_slots(&location).unwrap().len(), 1);
}

#[test]
fn descriptor_remapping_keeps_shape_and_storage_together() {
    let payload = BindingRef::new(0, 0);
    let descriptor = BindingRef::new(0, 1);
    let next_payload = BindingRef::new(0, 2);
    let next_descriptor = BindingRef::new(0, 3);
    let mut location = array(
        payload,
        ExtentValue::Stored(descriptor, 0),
        ExtentValue::Stored(descriptor, 1),
    );
    let Location::Array(array) = &mut location else {
        panic!("array location")
    };
    array.dimensions[0].1.host = Some(Expr::Integer(7));
    let bindings = LookupMap::from_iter([(payload, next_payload), (descriptor, next_descriptor)]);
    let mapped = remap_saved(&location, &bindings);
    assert!(mapped.persistent());
    assert_eq!(
        saved_slots(&mapped).unwrap(),
        vec![
            (next_payload, types::i32()),
            (next_descriptor, Body::number(0).ty)
        ]
    );
    let Location::Array(array) = mapped else {
        panic!("array location")
    };
    assert_eq!(
        array.dimensions[0].1.host(),
        Some(Expr::Scalar(ScalarExpr::Read {
            source: scalar_source(next_descriptor),
            offset: 4,
            ty: ScalarType::I32,
        }))
    );
}
