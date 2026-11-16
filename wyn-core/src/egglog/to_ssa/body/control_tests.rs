use super::joined_type;
use crate::types::{self, Type, TypeName};
use crate::BindingRef;

#[test]
fn local_array_joins_widen_capacity_inside_products() {
    let four = types::sized_array(4, types::i32());
    let five = types::sized_array(5, types::i32());
    let expected = types::make_array1(
        types::i32(),
        types::array_variant_bounded(),
        Type::Constructed(TypeName::Size(5), vec![]),
        types::no_buffer(),
    );
    assert_eq!(
        joined_type(&types::tuple(vec![four]), &types::tuple(vec![five])).unwrap(),
        types::tuple(vec![expected])
    );
}

#[test]
fn joins_do_not_erase_distinct_buffer_identities() {
    let view = |binding| {
        types::view_array_with_size(
            &types::i32(),
            Type::Constructed(TypeName::Size(4), vec![]),
            types::buffer_tag(BindingRef::new(0, binding)),
        )
    };
    assert!(joined_type(&view(0), &view(1)).is_err());
}
