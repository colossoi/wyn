use super::is_storage_image;
use crate::ast::TypeName;
use polytype::Type;

#[test]
fn is_storage_image_sees_through_unique() {
    let img = Type::Constructed(TypeName::StorageTexture, vec![]);
    assert!(is_storage_image(&img));
    // A view-array parameter is a real runtime value, not erasable.
    assert!(!is_storage_image(&Type::Constructed(TypeName::Array, vec![])));
}
