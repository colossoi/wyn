//! Tests for `bind_pattern`, `bind_irrefutable_pattern`, and
//! `fresh_type_for_pattern`. We drive them through small wyn source
//! programs and assert that type-check accepts / rejects as expected.

use crate::Compiler;

fn type_check_source(source: &str) -> std::result::Result<(), String> {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = Compiler::parse(source, &mut node_counter).map_err(|e| format!("parse: {}", e))?;
    parsed
        .resolve(&mut module_manager)
        .map_err(|e| format!("resolve: {}", e))?
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .map(|_| ())
        .map_err(|e| format!("type_check: {}", e))
}

// ----- bind_pattern accepts every supported pattern variant -----

#[test]
fn name_pattern_in_let_binds() {
    let result = type_check_source("def f(n: i32) i32 = let x = n in x + 1");
    assert!(result.is_ok(), "got {:?}", result);
}

#[test]
fn wildcard_pattern_in_let_binds() {
    let result = type_check_source("def f(n: i32) i32 = let _ = n in 0");
    assert!(result.is_ok(), "got {:?}", result);
}

#[test]
fn tuple_pattern_in_let_destructures() {
    let result = type_check_source("def f(p: (i32, i32)) i32 = let (a, b) = p in a + b");
    assert!(result.is_ok(), "got {:?}", result);
}

#[test]
fn typed_name_pattern_in_let_binds() {
    let result = type_check_source("def f(n: i32) i32 = let x: i32 = n in x + 1");
    assert!(result.is_ok(), "got {:?}", result);
}

#[test]
fn lambda_param_with_tuple_pattern() {
    // Lambda parameters share bind_irrefutable_pattern. Drive the
    // lambda through map to get the parameter type inferred.
    let result = type_check_source(
        r#"
def f(arr: [3](i32, i32)) [3]i32 = map((|(a, b)| a + b), arr)
"#,
    );
    assert!(result.is_ok(), "got {:?}", result);
}

// ----- bind_irrefutable_pattern rejects refutable patterns -----

#[test]
fn literal_pattern_in_let_is_rejected() {
    let result = type_check_source(
        r#"
def f(n: i32) i32 = let 0 = n in 1
"#,
    );
    assert!(result.is_err(), "literal pattern in let should be refutable");
    let msg = result.unwrap_err();
    assert!(
        msg.to_lowercase().contains("refutable") || msg.to_lowercase().contains("literal"),
        "error should mention refutability or literal: {}",
        msg
    );
}

#[test]
fn multi_variant_constructor_in_let_is_rejected() {
    let result = type_check_source(
        r#"
def f(t: #a | #b) i32 = let #a = t in 1
"#,
    );
    assert!(result.is_err(), "multi-variant ctor in let should be refutable");
    let msg = result.unwrap_err();
    assert!(
        msg.to_lowercase().contains("refutable") || msg.contains("#b"),
        "error should mention refutability or #b: {}",
        msg
    );
}

#[test]
fn single_variant_constructor_in_let_is_accepted() {
    let result = type_check_source(
        r#"
def f(b: #boxed(i32)) i32 = let #boxed(x) = b in x + 1
"#,
    );
    assert!(
        result.is_ok(),
        "single-variant ctor in let is irrefutable: {:?}",
        result
    );
}

#[test]
fn tuple_pattern_with_refutable_sub_in_let_is_rejected() {
    let result = type_check_source(
        r#"
def f(p: (i32, i32)) i32 = let (a, 0) = p in a
"#,
    );
    assert!(
        result.is_err(),
        "literal in tuple element should make let refutable"
    );
}

// ----- match accepts refutable patterns (irrefutability NOT enforced) -----

#[test]
fn match_arm_with_literal_is_accepted() {
    let result = type_check_source(
        r#"
def f(n: i32) i32 =
  match n
  case 0 -> 1
  case _ -> 0
"#,
    );
    assert!(result.is_ok(), "literal match arm should be fine: {:?}", result);
}

#[test]
fn match_arm_with_single_variant_constructor_is_accepted() {
    let result = type_check_source(
        r#"
def f(b: #boxed(i32)) i32 =
  match b
  case #boxed(x) -> x
"#,
    );
    assert!(result.is_ok(), "got {:?}", result);
}

#[test]
fn match_arm_with_wildcard_is_accepted() {
    let result = type_check_source(
        r#"
def f(t: #a | #b | #c) i32 =
  match t
  case #a -> 1
  case _  -> 0
"#,
    );
    assert!(result.is_ok(), "got {:?}", result);
}

// ----- non-exhaustive match is rejected -----

#[test]
fn non_exhaustive_match_is_rejected() {
    let result = type_check_source(
        r#"
def f(t: #a | #b) i32 =
  match t
  case #a -> 1
"#,
    );
    assert!(result.is_err(), "missing #b should be flagged");
    let msg = result.unwrap_err();
    assert!(msg.contains("non-exhaustive") || msg.contains("missing"));
}

#[test]
fn match_with_redundant_arm_is_rejected() {
    let result = type_check_source(
        r#"
def f(t: #a | #b) i32 =
  match t
  case #a -> 1
  case #a -> 2
  case #b -> 0
"#,
    );
    assert!(result.is_err(), "duplicate #a should be flagged");
    let msg = result.unwrap_err();
    assert!(msg.contains("unreachable") || msg.contains("redundant"));
}

// ----- vec destructuring (positional inverse of the `@[…]` ctor) -----

#[test]
fn vec_pattern_in_let_destructures_vec2() {
    let result = type_check_source("def f(v: vec2f32) f32 = let @[a, b] = v in a + b");
    assert!(result.is_ok(), "got {:?}", result);
}

#[test]
fn vec_pattern_in_let_destructures_vec4() {
    let result = type_check_source("def f(v: vec4f32) f32 = let @[a, b, c, d] = v in a + b + c + d");
    assert!(result.is_ok(), "got {:?}", result);
}

#[test]
fn vec_pattern_supports_wildcards() {
    let result = type_check_source("def f(v: vec3f32) f32 = let @[a, _, c] = v in a + c");
    assert!(result.is_ok(), "got {:?}", result);
}

#[test]
fn vec_pattern_arity_mismatch_is_rejected() {
    // vec2 scrutinee, 3-element pattern.
    let result = type_check_source("def f(v: vec2f32) f32 = let @[a, b, c] = v in a + b + c");
    assert!(
        result.is_err(),
        "3-pattern destructure of vec2 should be flagged, got {:?}",
        result
    );
}

#[test]
fn vec_pattern_on_non_vec_is_rejected() {
    // Scrutinee is a scalar; vec destructure should not match.
    let result = type_check_source("def f(n: f32) f32 = let @[a, b] = n in a + b");
    assert!(
        result.is_err(),
        "vec destructure of scalar should be flagged, got {:?}",
        result
    );
}
