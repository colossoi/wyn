use super::ModuleManager;
use crate::ast::{NodeCounter, TypeName};
use crate::types::checker::TypeChecker;
use polytype::Type;

use polytype::TypeScheme;

/// Check if a type is f32 -> f32
fn is_f32_to_f32(ty: &Type<TypeName>) -> bool {
    matches!(
        ty,
        Type::Constructed(TypeName::Arrow, args)
            if args.len() == 2
                && matches!(&args[0], Type::Constructed(TypeName::Float(32), _))
                && matches!(&args[1], Type::Constructed(TypeName::Float(32), _))
    )
}

/// Check if a type is an arrow (function) type
fn is_arrow_type(ty: &Type<TypeName>) -> bool {
    matches!(ty, Type::Constructed(TypeName::Arrow, _))
}

/// Extract the inner monotype from a TypeScheme (unwrapping any polytypes)
fn get_monotype(scheme: &TypeScheme<TypeName>) -> &Type<TypeName> {
    match scheme {
        TypeScheme::Monotype(ty) => ty,
        TypeScheme::Polytype { body, .. } => get_monotype(body),
    }
}

#[test]
fn test_query_f32_sin_from_math_prelude() {
    let mut node_counter = NodeCounter::new();
    let manager = ModuleManager::new(&mut node_counter);

    // Prelude files are automatically loaded on creation
    println!(
        "Loaded modules: {:?}",
        manager.elaborated_modules.keys().collect::<Vec<_>>()
    );

    // Use TypeChecker to get the function type schemes
    let mut checker = TypeChecker::new(&manager);
    checker.load_builtins().expect("Failed to load builtins");

    // Query for the f32 module's sin function type
    let sin_type = checker
        .get_module_function_type_scheme("f32", "sin")
        .expect("Failed to find f32.sin");

    // Should be f32 -> f32
    println!("Found f32.sin with type: {:?}", sin_type);
    let sin_mono = get_monotype(&sin_type);
    assert!(is_f32_to_f32(sin_mono), "f32.sin should be f32 -> f32");

    // Also test that f32.sum is found (from module body)
    let sum_type = checker
        .get_module_function_type_scheme("f32", "sum")
        .expect("Failed to find f32.sum");
    println!("Found f32.sum with type: {:?}", sum_type);

    // f32.sum takes an array of f32 and returns f32, so we just check it's a function type
    let sum_mono = get_monotype(&sum_type);
    assert!(is_arrow_type(sum_mono), "f32.sum should be a function type");
}
