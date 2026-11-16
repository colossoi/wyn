use super::output_type;
use crate::types::{self, Type, TypeName};

#[test]
fn output_nesting_preserves_existential_row_extents() {
    let size = Type::Constructed(TypeName::SizeVar("k".into()), vec![]);
    let row = types::make_array1(
        types::i32(),
        types::array_variant_abstract(),
        size.clone(),
        types::no_buffer(),
    );
    let lowered = types::sized_array(
        4,
        Type::Constructed(TypeName::Existential(vec!["k".into()]), vec![row]),
    );
    let expected = types::sized_array(
        4,
        types::make_array1(
            types::i32(),
            types::array_variant_composite(),
            size,
            types::no_buffer(),
        ),
    );
    assert_eq!(output_type(&expected, &lowered), Some(expected));
}
