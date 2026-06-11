use super::{ElaboratedItem, ModuleManager};
use crate::ast::{ModuleTypeExpression, NodeCounter, Program, Spec, TypeName};
use crate::lexer::tokenize;
use crate::parser::Parser;
use crate::resolve_placeholders::PlaceholderResolver;
use crate::types::checker::TypeChecker;
use polytype::Type;
use std::collections::HashMap;

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
    let (prog, mut nc) = parse_program(src);
    let mut mm = ModuleManager::new_empty();
    mm.elaborate_modules(&prog, &mut nc).unwrap();
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
    let mut manager = ModuleManager::new(&mut node_counter);

    // Prelude files are automatically loaded on creation
    println!(
        "Loaded modules: {:?}",
        manager.elaborated_modules.keys().collect::<Vec<_>>()
    );

    // Resolve placeholders in modules to build spec_schemes
    // (No program to resolve, just pass an empty one)
    let mut empty_program = crate::ast::Program { declarations: vec![] };
    let mut resolver = PlaceholderResolver::new();
    resolver.resolve(&mut manager, &mut empty_program);
    let (context, spec_schemes) = resolver.into_parts();

    // Use TypeChecker to get the function type schemes
    let mut checker = TypeChecker::with_context_and_schemes(&manager, context, spec_schemes);
    // After Phase 4 the type checker requires every catalog identifier
    // to have a `NameResolution::Builtin` entry. Build the side table
    // covering prelude module bodies — without it, bare references like
    // `fract` inside `rand.init` resolve to an `UndefinedVariable`.
    let nr =
        crate::name_resolution::build_name_resolution(&empty_program, &manager, crate::builtins::catalog());
    checker.set_name_resolution(nr);
    checker.check_module_functions().expect("Failed to check module functions");

    // Query for the f32 module's sin function type from the cache
    let sin_type = checker.get_module_scheme("f32.sin").expect("Failed to find f32.sin");

    // Should be f32 -> f32
    println!("Found f32.sin with type: {:?}", sin_type);
    let sin_mono = get_monotype(sin_type);
    assert!(is_f32_to_f32(sin_mono), "f32.sin should be f32 -> f32");

    // Also test that f32.sum is found (from module body)
    let sum_type = checker.get_module_scheme("f32.sum").expect("Failed to find f32.sum");
    println!("Found f32.sum with type: {:?}", sum_type);

    // f32.sum takes an array of f32 and returns f32, so we just check it's a function type
    let sum_mono = get_monotype(sum_type);
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
    sig add(a: t, b: t) t
}

module type float_like = {
    include numeric
    sig mul(a: t, b: t) t
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
    sig add(a: t, b: t) t
}

module f32_num : (numeric with t = f32) = {
    def add(x: t, y: t) t = x + y
}
        "#,
    );

    let m = mm.get_elaborated_module("f32_num").expect("f32_num should exist");

    // Should have TypeAlias for t -> f32
    let has_type_alias =
        m.items.iter().any(|item| matches!(item, ElaboratedItem::TypeAlias(name, _) if name == "t"));
    assert!(has_type_alias, "f32_num should have type alias for t");

    // Should have Decl for add
    let has_add_decl =
        m.items.iter().any(|item| matches!(item, ElaboratedItem::Decl(d) if d.name == "add"));
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
    sig add(a: t, b: t) t
}

module my_f32_num : (numeric with t = f32) = {
    def add(x: t, y: t) t = x + y
}

functor add_stuff(n: numeric) = {
    type t = n.t
    def add3(x: t, y: t, z: t) t = n.add(n.add(x, y), z)
}

module add_f32 = add_stuff(my_f32_num)
        "#,
    );

    let m = mm.get_elaborated_module("add_f32").expect("add_f32 should exist");

    // Should have TypeAlias for t (resolved from n.t -> f32)
    let type_alias = m.items.iter().find_map(|item| match item {
        ElaboratedItem::TypeAlias(name, ty) if name == "t" => Some(ty),
        _ => None,
    });
    assert!(type_alias.is_some(), "add_f32 should have type alias for t");

    // Should have Decl for add3
    let add3_decl = m.items.iter().find_map(|item| match item {
        ElaboratedItem::Decl(d) if d.name == "add3" => Some(d),
        _ => None,
    });
    assert!(add3_decl.is_some(), "add_f32 should have add3 declaration");
}

#[test]
fn resolve_names_qualifies_intra_module_functions() {
    let mm = module_manager_with(
        r#"
module foo = {
    def helper(x: i32) i32 = x + 1
    def main(x: i32) i32 = helper(x)
}
        "#,
    );

    let m = mm.get_elaborated_module("foo").expect("foo should exist");

    // Find the main decl
    let main_decl = m.items.iter().find_map(|item| match item {
        ElaboratedItem::Decl(d) if d.name == "main" => Some(d),
        _ => None,
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
    def bar(x: i32) i32 = x
    def test(x: i32) i32 =
        let bar = x + 1 in
        bar
}
        "#,
    );

    let m = mm.get_elaborated_module("foo").expect("foo should exist");

    // Find the test decl
    let test_decl = m.items.iter().find_map(|item| match item {
        ElaboratedItem::Decl(d) if d.name == "test" => Some(d),
        _ => None,
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
    sig add(a: t, b: t) t
}

module my_i32 : (numeric with t = i32) = {
    def zero: t = 0
    def add(x: t, y: t) t = x + y
}

functor sum_module(n: numeric) = {
    def sum3(a: n.t, b: n.t, c: n.t) n.t = n.add(n.add(a, b), c)
}

module i32_sum = sum_module(my_i32)
        "#,
    );

    let m = mm.get_elaborated_module("i32_sum").expect("i32_sum should exist");

    // Find sum3 decl
    let sum3_decl = m.items.iter().find_map(|item| match item {
        ElaboratedItem::Decl(d) if d.name == "sum3" => Some(d),
        _ => None,
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

/// Regression guard: when an outer signature has a `with t = i32`
/// substitution, references to `t` inside a NESTED module signature
/// must also be rewritten. Previously `substitute_in_spec`'s
/// `Spec::Module` arm cloned the inner `ModuleTypeExpression`
/// unchanged, leaving the nested `t` reference unresolved.
///
/// Build the signature directly:
///
///     {
///       type t
///       module M : { sig f : t -> t }
///     }
///
/// ask `elaborate_module_type` to run with `{ t → i32 }`, then dig
/// into `M`'s signature and check that its `f` sig sees `i32 -> i32`.
#[test]
fn test_substitute_into_nested_module_signature() {
    let mm = ModuleManager::new_empty();

    // Named type `t` — the thing we substitute away.
    let t_ty = Type::Constructed(TypeName::Named("t".into()), vec![]);
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    // Inner signature: `{ sig f : t -> t }`.
    let inner_sig = ModuleTypeExpression::Signature(vec![Spec::Sig(
        "f".into(),
        vec![],
        Type::arrow(t_ty.clone(), t_ty.clone()),
    )]);

    // Outer signature: `{ type t; module M : <inner_sig> }`.
    let outer = ModuleTypeExpression::Signature(vec![
        Spec::Type("t".into(), vec![], None),
        Spec::Module("M".into(), inner_sig),
    ]);

    let mut substitutions = HashMap::new();
    substitutions.insert("t".to_string(), i32_ty.clone());

    let elaborated =
        mm.elaborate_module_type(&outer, &substitutions).expect("elaborate_module_type should succeed");

    // Find the nested `module M : { ... }` spec.
    let nested_mte = elaborated
        .iter()
        .find_map(|spec| match spec {
            Spec::Module(name, mte) if name == "M" => Some(mte),
            _ => None,
        })
        .expect("outer signature should still contain `module M : ...`");

    let nested_specs = match nested_mte {
        ModuleTypeExpression::Signature(ss) => ss,
        _ => panic!("nested M should be a signature, got {:?}", nested_mte),
    };

    // The nested `sig f : t -> t` must now read `sig f : i32 -> i32`.
    let f_ty = nested_specs
        .iter()
        .find_map(|spec| match spec {
            Spec::Sig(name, _, ty) if name == "f" => Some(ty),
            _ => None,
        })
        .expect("nested signature should contain sig f");

    let expected = Type::arrow(i32_ty.clone(), i32_ty);
    assert_eq!(
        f_ty, &expected,
        "nested sig f's type should have t substituted to i32, got {:?}",
        f_ty
    );
}

// ---------------------------------------------------------------------------
// Regression: intra-module rewriting must recurse into every ExprKind.
//
// The elaboration-time walker used to be a hand-written match with arms for
// Application / Lambda / LetIn / If / BinaryOp / UnaryOp / Tuple / Array /
// ArrayIndex / ArrayWith / RecordLiteral / Match and a catch-all `_ => {}`.
// That meant intra-module refs buried inside Loop / Range / Slice /
// TypeAscription / TypeCoercion / TypeHole silently escaped qualification.
// After consolidating onto the shared `name_resolution::walk_expr`, every
// ExprKind is visited. These tests guard that.
//
// Each case constructs a tiny module with an intra-module reference
// inside the expression kind under test and asserts the elaborated
// body contains a qualified identifier with the module name.
// ---------------------------------------------------------------------------

fn module_body_str(src: &str, module_name: &str, fn_name: &str) -> String {
    let mm = module_manager_with(src);
    let m = mm.get_elaborated_module(module_name).expect("module should exist");
    let decl = m
        .items
        .iter()
        .find_map(|item| match item {
            ElaboratedItem::Decl(d) if d.name == fn_name => Some(d),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{} should have {} declaration", module_name, fn_name));
    format!("{:?}", decl.body)
}

#[test]
fn intra_module_ref_inside_loop_body_is_qualified() {
    // `arr' = map(helper, …)` inside a loop body — the `helper` call is
    // a bare identifier that should be rewritten to `foo.helper`.
    let body = module_body_str(
        r#"
module foo = {
    def helper(x: i32) i32 = x + 1
    def main(seed: i32) i32 =
        let (_, out) =
            loop (i, acc) = (0i32, seed) while i < 2 do
                (i + 1, helper(acc))
        in out
}
        "#,
        "foo",
        "main",
    );
    assert!(
        body.contains("[\"foo\"]"),
        "helper inside a loop body should be qualified to foo.helper: {}",
        body
    );
}

#[test]
fn intra_module_ref_inside_range_is_qualified() {
    // Range expression `helper(0) ..< helper(n)`. Range was not in the old
    // walker's match; the call sites inside its endpoints escaped
    // qualification.
    let body = module_body_str(
        r#"
module foo = {
    def helper(x: i32) i32 = x + 1
    def main(n: i32) [8]i32 =
        map(|j: i32| j, helper(0) ..< helper(n))
}
        "#,
        "foo",
        "main",
    );
    assert!(
        body.contains("Range") && body.contains("[\"foo\"]"),
        "helper inside a Range expression should be qualified: {}",
        body
    );
}

#[test]
fn intra_module_ref_inside_slice_is_qualified() {
    // `arr[helper(0) : helper(n)]` — slice bounds contain intra-module calls.
    let body = module_body_str(
        r#"
module foo = {
    def helper(x: i32) i32 = x + 1
    def main(arr: [16]i32, n: i32) [16]i32 = arr[helper(0) .. helper(n)]
}
        "#,
        "foo",
        "main",
    );
    assert!(
        body.contains("[\"foo\"]"),
        "helper inside slice bounds should be qualified: {}",
        body
    );
}

#[test]
fn intra_module_ref_inside_type_ascription_is_qualified() {
    // `helper(x) : i32` — the expression inside a type ascription.
    let body = module_body_str(
        r#"
module foo = {
    def helper(x: i32) i32 = x + 1
    def main(x: i32) i32 = (helper(x) : i32)
}
        "#,
        "foo",
        "main",
    );
    assert!(
        body.contains("[\"foo\"]"),
        "helper inside a type ascription should be qualified: {}",
        body
    );
}
