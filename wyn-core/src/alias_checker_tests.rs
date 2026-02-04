use crate::Compiler;
use crate::alias_checker::{AliasCheckResult, AliasChecker};

fn check_alias(source: &str) -> AliasCheckResult {
    let mut frontend = crate::cached_frontend();
    let parsed = Compiler::parse(source, &mut frontend.node_counter).expect("parse failed");
    let desugared = parsed.desugar(&mut frontend.node_counter).expect("desugar failed");
    let resolved = desugared.resolve(&mut frontend.module_manager).expect("resolve failed");
    let folded = resolved.fold_ast_constants();
    let type_checked =
        folded.type_check(&mut frontend.module_manager, &mut frontend.schemes).expect("type_check failed");

    let checker = AliasChecker::new(&type_checked.type_table, &type_checked.span_table);
    checker.check_program(&type_checked.ast).expect("alias check failed")
}

/// Helper to run through the pipeline up to alias checking (returns the AliasChecked stage)
fn alias_check_pipeline(source: &str) -> crate::AliasChecked {
    let mut frontend = crate::cached_frontend();
    let parsed = Compiler::parse(source, &mut frontend.node_counter).expect("parse failed");
    parsed
        .desugar(&mut frontend.node_counter)
        .expect("desugar failed")
        .resolve(&mut frontend.module_manager)
        .expect("resolve failed")
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .expect("type_check failed")
        .alias_check()
        .expect("alias_check failed")
}

#[test]
fn test_no_error_simple() {
    let result = check_alias(r#"def main(x: i32) i32 = x + 1"#);
    assert!(!result.has_errors());
}

#[test]
fn test_use_after_move() {
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let _ = consume(arr) in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(result.has_errors(), "Expected use-after-move error");
}

#[test]
fn test_alias_through_let() {
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let alias = arr in
    let _ = consume(arr) in
    alias[0]
"#;
    let result = check_alias(source);
    assert!(result.has_errors(), "Expected use-after-move error for alias");
}

#[test]
fn test_copy_type_no_tracking() {
    // i32 is copy, should work fine
    let source = r#"
def main(x: i32) i32 =
    let y = x in
    let z = x in
    y + z
"#;
    let result = check_alias(source);
    assert!(!result.has_errors());
}

// === Tricky edge cases ===

#[test]
fn test_transitive_aliasing() {
    // a -> b -> c, consume c, use a should error
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let a = arr in
    let b = a in
    let c = b in
    let _ = consume(c) in
    a[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "Expected error: a transitively aliases c which was consumed"
    );
}

#[test]
fn test_consume_alias_use_original() {
    // Consume the alias, then try to use the original - should error
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let alias = arr in
    let _ = consume(alias) in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "Expected error: arr's backing store was consumed via alias"
    );
}

#[test]
fn test_shadowing_does_not_affect_outer() {
    // Inner variable shadows outer with same name, consume inner, use outer
    // This should be OK because they're different backing stores
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(arr: [4]i32) i32 =
    let x = arr in
    let result =
        let x = [1, 2, 3, 4] in
        consume(x)
    in
    x[0]
"#;
    let result = check_alias(source);
    assert!(
        !result.has_errors(),
        "Should be OK: inner x is different from outer x"
    );
}

#[test]
fn test_if_branches_independent() {
    // Consume in one branch shouldn't affect use in other branch
    // (they're mutually exclusive execution paths)
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(t: (bool, [4]i32)) i32 =
    let (cond, arr) = t in
    if cond then consume(arr) else arr[0]
"#;
    let result = check_alias(source);
    // This SHOULD be OK since the branches are mutually exclusive
    // But a conservative implementation might reject it
    assert!(
        !result.has_errors(),
        "Should be OK: branches are mutually exclusive"
    );
}

#[test]
fn test_use_after_if_that_consumes() {
    // Use after an if expression where one branch consumed - should error
    let source = r#"
def consume(arr: *[4]i32) i32 = arr[0]

def main(t: (bool, [4]i32)) i32 =
    let (cond, arr) = t in
    let _ = if cond then consume(arr) else 0 in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(
        result.has_errors(),
        "Expected error: arr might have been consumed in if branch"
    );
}

#[test]
fn test_return_aliases_non_consumed_params() {
    // id3 takes: a (consumed via *), b (not consumed), c (not consumed)
    // Returns non-unique [4]i32, which should alias b and c
    // After the call: x is consumed, y and z are aliased by result
    // Use separate array literals so each has its own backing store

    let base = r#"
def id3(a: *[4]i32, b: [4]i32, c: [4]i32) [4]i32 = c
def consume(arr: *[4]i32) i32 = arr[0]
def main: i32 =
    let x = [1, 2, 3, 4] in
    let y = [5, 6, 7, 8] in
    let z = [9, 10, 11, 12] in
    let result = id3(x, y, z) in
"#;

    // Try consuming x - should error (x was already consumed by id3)
    let source_x = format!("{}    consume(x)", base);
    let result_x = check_alias(&source_x);
    assert!(result_x.has_errors(), "Expected error: x was consumed by id3");

    // Consume result, then try to use y - should error (result aliases y)
    let source_result_then_y = format!("{}    let _ = consume(result) in y[0]", base);
    let result_y = check_alias(&source_result_then_y);
    assert!(
        result_y.has_errors(),
        "Expected error: result aliases y, consuming result invalidates y"
    );

    // Consume result, then try to use z - should error (result aliases z)
    let source_result_then_z = format!("{}    let _ = consume(result) in z[0]", base);
    let result_z = check_alias(&source_result_then_z);
    assert!(
        result_z.has_errors(),
        "Expected error: result aliases z, consuming result invalidates z"
    );
}

#[test]
fn test_return_aliases_shared_backing_store() {
    // Using tuple destructuring - all three variables share a backing store
    // from the tuple. x, y, z all reference the same underlying memory.
    // Error messages now include alias information to help users understand
    // why seemingly unrelated variables are affected.

    let base = r#"
def id3(a: *[4]i32, b: [4]i32, c: [4]i32) [4]i32 = c
def consume(arr: *[4]i32) i32 = arr[0]
def main(t: ([4]i32, [4]i32, [4]i32)) i32 =
    let (x, y, z) = t in
    let result = id3(x, y, z) in
"#;

    // All three should error because they share a backing store with x,
    // which was consumed by id3
    let source_x = format!("{}    consume(x)", base);
    let result_x = check_alias(&source_x);
    assert!(result_x.has_errors(), "Expected error: x was consumed by id3");

    let source_y = format!("{}    consume(y)", base);
    let result_y = check_alias(&source_y);
    assert!(
        result_y.has_errors(),
        "Expected error: y shares backing store with consumed x"
    );

    let source_z = format!("{}    consume(z)", base);
    let result_z = check_alias(&source_z);
    assert!(
        result_z.has_errors(),
        "Expected error: z shares backing store with consumed x"
    );
}
