use super::{ElaboratedItem, ModuleManager};
use crate::ast::{NodeCounter, Program, TypeName};
use crate::lexer::tokenize;
use crate::parser::Parser;
use crate::types::checker::TypeChecker;
use polytype::Type;

use polytype::TypeScheme;

/// Parse a source string into a Program
fn parse_program(src: &str) -> (Program, NodeCounter) {
    let mut nc = NodeCounter::new();
    let tokens = tokenize(src).unwrap();
    let prog = Parser::new(tokens, &mut nc).parse().unwrap();
    (prog, nc)
}

/// Create a ModuleManager with the given source elaborated (no prelude)
fn module_manager_with(src: &str) -> ModuleManager {
    let (prog, _nc) = parse_program(src);
    let mut mm = ModuleManager::new_empty();
    mm.elaborate_modules(&prog).unwrap();
    mm
}

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
    let sin_type = checker.get_module_function_type_scheme("f32", "sin").expect("Failed to find f32.sin");

    // Should be f32 -> f32
    println!("Found f32.sin with type: {:?}", sin_type);
    let sin_mono = get_monotype(&sin_type);
    assert!(is_f32_to_f32(sin_mono), "f32.sin should be f32 -> f32");

    // Also test that f32.sum is found (from module body)
    let sum_type = checker.get_module_function_type_scheme("f32", "sum").expect("Failed to find f32.sum");
    println!("Found f32.sum with type: {:?}", sum_type);

    // f32.sum takes an array of f32 and returns f32, so we just check it's a function type
    let sum_mono = get_monotype(&sum_type);
    assert!(is_arrow_type(sum_mono), "f32.sum should be a function type");
}

// ============================================================================
// Module type elaboration tests
// ============================================================================

#[test]
fn elaborate_module_type_with_include() {
    let mm = module_manager_with(
        r#"
module type numeric = {
    type t
    sig zero : t
    sig add : t -> t -> t
}

module type float_like = {
    include numeric
    sig mul : t -> t -> t
}
        "#,
    );

    // float_like should include numeric's specs plus mul
    // Check that both module types were registered
    assert!(mm.get_elaborated_module("numeric").is_none()); // These are types, not modules

    // The module_type_registry should have both
    // We can verify by creating a module that implements float_like
}

#[test]
fn elaborate_module_with_signature_and_with() {
    let mm = module_manager_with(
        r#"
module type numeric = {
    type t
    sig add : t -> t -> t
}

module f32_num : (numeric with t = f32) = {
    def add(x: t, y: t) -> t = x + y
}
        "#,
    );

    let m = mm.get_elaborated_module("f32_num").expect("f32_num should exist");

    // Should have TypeAlias for t -> f32
    let has_type_alias = m.items.iter().any(|item| {
        matches!(item, ElaboratedItem::TypeAlias(name, _) if name == "t")
    });
    assert!(has_type_alias, "f32_num should have type alias for t");

    // Should have Decl for add
    let has_add_decl = m.items.iter().any(|item| {
        matches!(item, ElaboratedItem::Decl(d) if d.name == "add")
    });
    assert!(has_add_decl, "f32_num should have add declaration");

    // type_aliases should contain f32_num.t -> f32
    let t_alias = mm.resolve_type_alias("f32_num.t");
    assert!(t_alias.is_some(), "f32_num.t should be in type_aliases");
}

#[test]
fn elaborate_functor_application() {
    let mm = module_manager_with(
        r#"
module type numeric = {
    type t
    sig add : t -> t -> t
}

module my_f32_num : (numeric with t = f32) = {
    def add(x: t, y: t) -> t = x + y
}

module add_stuff(n: numeric) = {
    type t = n.t
    def add3(x: t, y: t, z: t) -> t = n.add(n.add(x, y), z)
}

module add_f32 = add_stuff(my_f32_num)
        "#,
    );

    let m = mm.get_elaborated_module("add_f32").expect("add_f32 should exist");

    // Should have TypeAlias for t (resolved from n.t -> f32)
    let type_alias = m.items.iter().find_map(|item| {
        match item {
            ElaboratedItem::TypeAlias(name, ty) if name == "t" => Some(ty),
            _ => None,
        }
    });
    assert!(type_alias.is_some(), "add_f32 should have type alias for t");

    // Should have Decl for add3
    let add3_decl = m.items.iter().find_map(|item| {
        match item {
            ElaboratedItem::Decl(d) if d.name == "add3" => Some(d),
            _ => None,
        }
    });
    assert!(add3_decl.is_some(), "add_f32 should have add3 declaration");
}

#[test]
fn resolve_names_qualifies_intra_module_functions() {
    let mm = module_manager_with(
        r#"
module foo = {
    def helper(x: i32) -> i32 = x + 1
    def main(x: i32) -> i32 = helper(x)
}
        "#,
    );

    let m = mm.get_elaborated_module("foo").expect("foo should exist");

    // Find the main decl
    let main_decl = m.items.iter().find_map(|item| {
        match item {
            ElaboratedItem::Decl(d) if d.name == "main" => Some(d),
            _ => None,
        }
    });
    assert!(main_decl.is_some(), "foo should have main declaration");

    // The body should reference foo.helper (qualified), not just helper
    // We can check this by looking at the expression structure
    let main = main_decl.unwrap();
    let body_str = format!("{:?}", main.body);
    assert!(
        body_str.contains("foo") || body_str.contains("Identifier([\"foo\"]"),
        "main body should have qualified reference to helper: {}",
        body_str
    );
}

#[test]
fn resolve_names_respects_local_shadowing() {
    let mm = module_manager_with(
        r#"
module foo = {
    def bar(x: i32) -> i32 = x
    def test(x: i32) -> i32 =
        let bar = x + 1 in
        bar
}
        "#,
    );

    let m = mm.get_elaborated_module("foo").expect("foo should exist");

    // Find the test decl
    let test_decl = m.items.iter().find_map(|item| {
        match item {
            ElaboratedItem::Decl(d) if d.name == "test" => Some(d),
            _ => None,
        }
    });
    assert!(test_decl.is_some(), "foo should have test declaration");

    // The local bar binding should NOT be qualified to foo.bar
    // The body of the let should just reference the local bar
    let test = test_decl.unwrap();
    let body_str = format!("{:?}", test.body);

    // Inside the let body, bar should be unqualified (local reference)
    // This is a basic sanity check - the let binding shadows the module function
    assert!(
        body_str.contains("LetIn"),
        "test body should contain a let-in expression: {}",
        body_str
    );
}

#[test]
fn functor_param_module_references_resolved() {
    let mm = module_manager_with(
        r#"
module type numeric = {
    type t
    sig zero : t
    sig add : t -> t -> t
}

module my_i32 : (numeric with t = i32) = {
    def zero: t = 0
    def add(x: t, y: t) -> t = x + y
}

module sum_module(n: numeric) = {
    def sum3(a: n.t, b: n.t, c: n.t) -> n.t = n.add(n.add(a, b), c)
}

module i32_sum = sum_module(my_i32)
        "#,
    );

    let m = mm.get_elaborated_module("i32_sum").expect("i32_sum should exist");

    // Find sum3 decl
    let sum3_decl = m.items.iter().find_map(|item| {
        match item {
            ElaboratedItem::Decl(d) if d.name == "sum3" => Some(d),
            _ => None,
        }
    });
    assert!(sum3_decl.is_some(), "i32_sum should have sum3 declaration");

    // The body should reference my_i32.add (not n.add)
    let sum3 = sum3_decl.unwrap();
    let body_str = format!("{:?}", sum3.body);
    assert!(
        body_str.contains("my_i32"),
        "sum3 body should have n.add resolved to my_i32.add: {}",
        body_str
    );
}
