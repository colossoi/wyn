use super::*;
use crate::types;

#[test]
fn test_resolve_size_placeholder() {
    let mut resolver = PlaceholderResolver::new();

    let ty = types::unsized_array(types::f32(), types::storage_addrspace());
    let resolved = resolver.resolve_type(&ty);

    // Should have replaced SizePlaceholder with a variable
    match &resolved {
        Type::Constructed(TypeName::Array, args) => {
            assert!(matches!(&args[2], Type::Variable(_)));
        }
        _ => panic!("Expected array type"),
    }
}

#[test]
fn test_resolve_address_placeholder() {
    let mut resolver = PlaceholderResolver::new();

    let ty = Type::Constructed(
        TypeName::Array,
        vec![
            types::f32(),
            Type::Constructed(TypeName::AddressPlaceholder, vec![]),
            Type::Constructed(TypeName::Size(10), vec![]),
        ],
    );
    let resolved = resolver.resolve_type(&ty);

    // Should have replaced AddressPlaceholder with a variable
    match &resolved {
        Type::Constructed(TypeName::Array, args) => {
            assert!(matches!(&args[1], Type::Variable(_)));
        }
        _ => panic!("Expected array type"),
    }
}

#[test]
fn test_context_preserved() {
    let mut resolver = PlaceholderResolver::new();

    // Create two placeholders - should get different variables
    let ty1 = Type::Constructed(TypeName::SizePlaceholder, vec![]);
    let ty2 = Type::Constructed(TypeName::SizePlaceholder, vec![]);

    let resolved1 = resolver.resolve_type(&ty1);
    let resolved2 = resolver.resolve_type(&ty2);

    // Should be different variables
    match (&resolved1, &resolved2) {
        (Type::Variable(v1), Type::Variable(v2)) => {
            assert_ne!(v1, v2);
        }
        _ => panic!("Expected variables"),
    }

    // Context should have the variables
    let mut context = resolver.into_context();
    // The context should have allocated at least 2 variables
    let next_var = context.new_variable();
    match next_var {
        Type::Variable(v) => assert!(v >= 2),
        _ => panic!("Expected variable"),
    }
}
