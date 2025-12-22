use crate::Compiler;
use crate::alias_checker::{AliasCheckResult, AliasChecker, InPlaceInfo, analyze_inplace};

fn check_alias(source: &str) -> AliasCheckResult {
    let mut frontend = crate::cached_frontend();
    let parsed = Compiler::parse(source, &mut frontend.node_counter).expect("parse failed");
    let desugared = parsed.desugar(&mut frontend.node_counter).expect("desugar failed");
    let resolved = desugared.resolve(&frontend.module_manager).expect("resolve failed");
    let folded = resolved.fold_ast_constants();
    let type_checked =
        folded.type_check(&frontend.module_manager, &mut frontend.schemes).expect("type_check failed");

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
        .resolve(&frontend.module_manager)
        .expect("resolve failed")
        .fold_ast_constants()
        .type_check(&frontend.module_manager, &mut frontend.schemes)
        .expect("type_check failed")
        .alias_check()
        .expect("alias_check failed")
}

#[test]
fn test_no_error_simple() {
    let result = check_alias(r#"def main(x: i32) -> i32 = x + 1"#);
    assert!(!result.has_errors());
}

#[test]
fn test_use_after_move() {
    let source = r#"
def consume(arr: *[4]i32) -> i32 = arr[0]

def main(arr: [4]i32) -> i32 =
    let _ = consume(arr) in
    arr[0]
"#;
    let result = check_alias(source);
    assert!(result.has_errors(), "Expected use-after-move error");
}

#[test]
fn test_alias_through_let() {
    let source = r#"
def consume(arr: *[4]i32) -> i32 = arr[0]

def main(arr: [4]i32) -> i32 =
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
def main(x: i32) -> i32 =
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
def consume(arr: *[4]i32) -> i32 = arr[0]

def main(arr: [4]i32) -> i32 =
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
def consume(arr: *[4]i32) -> i32 = arr[0]

def main(arr: [4]i32) -> i32 =
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
def consume(arr: *[4]i32) -> i32 = arr[0]

def main(arr: [4]i32) -> i32 =
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
#[ignore = "Conservative: doesn't track branch-specific state yet"]
fn test_if_branches_independent() {
    // Consume in one branch shouldn't affect use in other branch
    // (they're mutually exclusive execution paths)
    let source = r#"
def consume(arr: *[4]i32) -> i32 = arr[0]

def main(t: (bool, [4]i32)) -> i32 =
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
def consume(arr: *[4]i32) -> i32 = arr[0]

def main(t: (bool, [4]i32)) -> i32 =
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
def id3(a: *[4]i32, b: [4]i32, c: [4]i32) -> [4]i32 = c
def consume(arr: *[4]i32) -> i32 = arr[0]
def main() -> i32 =
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
def id3(a: *[4]i32, b: [4]i32, c: [4]i32) -> [4]i32 = c
def consume(arr: *[4]i32) -> i32 = arr[0]
def main(t: ([4]i32, [4]i32, [4]i32)) -> i32 =
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

// =============================================================================
// MIR In-Place Optimization Tests
// =============================================================================

/// Helper to analyze in-place opportunities in a source file
/// Runs through the full pipeline up to lifting (matching the real compilation flow)
fn analyze_inplace_ops(source: &str) -> InPlaceInfo {
    let mut frontend = crate::cached_frontend();
    let parsed = Compiler::parse(source, &mut frontend.node_counter).expect("parse failed");
    let desugared = parsed.desugar(&mut frontend.node_counter).expect("desugar failed");
    let resolved = desugared.resolve(&frontend.module_manager).expect("resolve failed");
    let folded = resolved.fold_ast_constants();
    let type_checked =
        folded.type_check(&frontend.module_manager, &mut frontend.schemes).expect("type_check failed");
    let alias_checked = type_checked.alias_check().expect("alias check failed");
    let (flattened, _backend) =
        alias_checked.flatten(&frontend.module_manager, &frontend.schemes).expect("flatten failed");

    // For tests without entry points, analyze after flattening (before monomorphization)
    // since filter_reachable would remove all defs without an entry point
    analyze_inplace(&flattened.mir)
}

#[test]
fn test_inplace_map_simple_dead_after() {
    // arr is not used after the map call - should be eligible for in-place
    let source = r#"
def double(x: i32) -> i32 = x * 2

def main(arr: [4]i32) -> [4]i32 =
    map(double, arr)
"#;
    let info = analyze_inplace_ops(source);
    assert!(
        !info.can_reuse_input.is_empty(),
        "Expected map call to be eligible for in-place optimization"
    );
}

#[test]
fn test_inplace_map_used_after() {
    // arr is used after the map call - should NOT be eligible for in-place
    let source = r#"
def double(x: i32) -> i32 = x * 2

def main(arr: [4]i32) -> (i32, [4]i32) =
    let result = map(double, arr) in
    (arr[0], result)
"#;
    let info = analyze_inplace_ops(source);
    assert!(
        info.can_reuse_input.is_empty(),
        "Expected no in-place optimization: arr is used after map"
    );
}

#[test]
fn test_inplace_map_alias_used_after() {
    // arr2 aliases arr, and arr2 is used after the map call
    // Should NOT be eligible for in-place
    let source = r#"
def double(x: i32) -> i32 = x * 2

def main(arr: [4]i32) -> (i32, [4]i32) =
    let arr2 = arr in
    let result = map(double, arr) in
    (arr2[0], result)
"#;
    let info = analyze_inplace_ops(source);
    assert!(
        info.can_reuse_input.is_empty(),
        "Expected no in-place optimization: alias arr2 is used after map"
    );
}

#[test]
fn test_inplace_map_nested() {
    // Nested maps - inner map result is used by outer map
    // The inner map on arr is dead after (only used by outer map)
    let source = r#"
def double(x: i32) -> i32 = x * 2

def main(arr: [4]i32) -> [4]i32 =
    map(double, map(double, arr))
"#;
    let info = analyze_inplace_ops(source);
    // At least one of the maps should be eligible
    // (Outer map can be in-place since inner result is dead after)
    assert!(
        !info.can_reuse_input.is_empty(),
        "Expected at least one map to be eligible for in-place"
    );
}

#[test]
fn test_inplace_map_in_let() {
    // Map result bound to name, original array dead
    let source = r#"
def double(x: i32) -> i32 = x * 2

def main(arr: [4]i32) -> [4]i32 =
    let result = map(double, arr) in
    result
"#;
    let info = analyze_inplace_ops(source);
    assert!(
        !info.can_reuse_input.is_empty(),
        "Expected map to be eligible for in-place when arr is dead after"
    );
}

// =============================================================================
// MIR In-Place ArrayWith Tests
// =============================================================================

#[test]
fn test_inplace_with_simple_dead_after() {
    // arr is not used after the with - should be eligible for in-place
    let source = r#"
def main(arr: [4]i32) -> [4]i32 =
    arr with [0] = 42
"#;
    let info = analyze_inplace_ops(source);
    assert!(
        !info.can_reuse_input.is_empty(),
        "Expected array_with to be eligible for in-place optimization"
    );
}

#[test]
fn test_inplace_with_used_after() {
    // arr is used after the with - should NOT be eligible for in-place
    let source = r#"
def main(arr: [4]i32) -> (i32, [4]i32) =
    let result = arr with [0] = 42 in
    (arr[1], result)
"#;
    let info = analyze_inplace_ops(source);
    assert!(
        info.can_reuse_input.is_empty(),
        "Expected no in-place optimization: arr is used after with"
    );
}

#[test]
fn test_inplace_with_discarded() {
    // Result is discarded with let _ = ... - should be eligible
    let source = r#"
def main(arr: [4]i32) -> () =
    let _ = arr with [0] = 42 in
    ()
"#;
    let info = analyze_inplace_ops(source);
    assert!(
        !info.can_reuse_input.is_empty(),
        "Expected array_with to be eligible when result is discarded"
    );
}

#[test]
fn test_inplace_with_chained() {
    // Chained with operations - each should be analyzed independently
    let source = r#"
def main(arr: [4]i32) -> [4]i32 =
    let arr2 = arr with [0] = 1 in
    arr2 with [1] = 2
"#;
    let info = analyze_inplace_ops(source);
    // Both operations should be eligible since arr is dead after first,
    // and arr2 is dead after second
    assert!(
        info.can_reuse_input.len() >= 1,
        "Expected at least one array_with to be eligible for in-place"
    );
}

// =============================================================================
// Pipeline Integration Tests
// =============================================================================

/// Test that has_alias_errors() returns true when alias errors exist
#[test]
fn test_pipeline_has_alias_errors_true() {
    let alias_checked = alias_check_pipeline(
        r#"
def consume(arr: *[4]i32) -> i32 = arr[0]

def main(arr: [4]i32) -> i32 =
    let _ = consume(arr) in
    arr[0]
"#,
    );
    assert!(
        alias_checked.has_alias_errors(),
        "Expected has_alias_errors() to return true for use-after-move"
    );
}

/// Test that has_alias_errors() returns false when no alias errors exist
#[test]
fn test_pipeline_has_alias_errors_false() {
    let alias_checked = alias_check_pipeline(r#"def main(x: i32) -> i32 = x + 1"#);
    assert!(
        !alias_checked.has_alias_errors(),
        "Expected has_alias_errors() to return false for valid code"
    );
}

// =============================================================================
// Liveness Analysis Tests
// =============================================================================

#[test]
fn test_liveness_simple_last_use() {
    // Array passed to function, not used after - should be alias_free=true, released=true
    let source = r#"
def f(arr: [4]i32) -> i32 = arr[0]

def main() -> i32 =
    let x = [1, 2, 3, 4] in
    f(x)
"#;
    let result = check_alias(source);
    assert!(!result.has_errors());

    // Should have liveness info for the array argument
    assert!(
        !result.liveness.is_empty(),
        "Expected liveness info for array argument"
    );

    // The single entry should be alias_free and released
    for (_, info) in &result.liveness {
        assert!(info.alias_free, "Expected alias_free=true for unaliased array");
        assert!(info.released, "Expected released=true for last use");
    }
}

#[test]
fn test_liveness_aliased_array() {
    // Array aliased by another variable - should be alias_free=false
    let source = r#"
def f(arr: [4]i32) -> i32 = arr[0]

def main() -> i32 =
    let x = [1, 2, 3, 4] in
    let y = x in
    f(x)
"#;
    let result = check_alias(source);
    assert!(!result.has_errors());

    // Should have liveness info
    assert!(
        !result.liveness.is_empty(),
        "Expected liveness info for array argument"
    );

    // The entry should NOT be alias_free (y aliases x)
    for (_, info) in &result.liveness {
        assert!(
            !info.alias_free,
            "Expected alias_free=false when array is aliased"
        );
    }
}

#[test]
fn test_liveness_multiple_uses() {
    // Array used twice - first use should have released=false
    let source = r#"
def f(arr: [4]i32) -> i32 = arr[0]

def main() -> i32 =
    let x = [1, 2, 3, 4] in
    f(x) + f(x)
"#;
    let result = check_alias(source);
    assert!(!result.has_errors());

    // Should have 2 liveness entries (one per call)
    assert_eq!(
        result.liveness.len(),
        2,
        "Expected 2 liveness entries for 2 array arguments"
    );

    // At least one should have released=false (first use)
    // At least one should have released=true (last use)
    let released_count = result.liveness.values().filter(|info| info.released).count();
    let not_released_count = result.liveness.values().filter(|info| !info.released).count();

    assert_eq!(
        released_count, 1,
        "Expected exactly one use to be released (last use)"
    );
    assert_eq!(
        not_released_count, 1,
        "Expected exactly one use to not be released (first use)"
    );
}

#[test]
fn test_liveness_fresh_array_literal() {
    // Array literal passed directly - should be alias_free=true, released=true
    let source = r#"
def f(arr: [4]i32) -> i32 = arr[0]

def main() -> i32 =
    f([1, 2, 3, 4])
"#;
    let result = check_alias(source);
    assert!(!result.has_errors());

    // Should have liveness info for the array literal
    assert!(
        !result.liveness.is_empty(),
        "Expected liveness info for array literal argument"
    );

    // Fresh array literal should be alias_free and released
    for (_, info) in &result.liveness {
        assert!(
            info.alias_free,
            "Expected alias_free=true for fresh array literal"
        );
        assert!(
            info.released,
            "Expected released=true for array literal (no other uses)"
        );
    }
}

#[test]
fn test_liveness_no_info_for_non_array() {
    // Non-array arguments should not have liveness info
    let source = r#"
def f(x: i32) -> i32 = x + 1

def main() -> i32 =
    f(42)
"#;
    let result = check_alias(source);
    assert!(!result.has_errors());

    // Should have NO liveness info (no array arguments)
    assert!(
        result.liveness.is_empty(),
        "Expected no liveness info for non-array arguments"
    );
}

// =============================================================================
// Slice and Range Alias Tests
// =============================================================================
// Note: Borrowed slices (arr[i:j]) alias their source array.
// This is the new semantics - slices are zero-copy views.

#[test]
fn test_borrowed_slice_consuming_invalidates_source() {
    // Borrowed slices alias the source array.
    // Consuming the slice invalidates the original array.
    let source = r#"
def consume(arr: *[3]i32) -> i32 = arr[0]

def main(arr: [10]i32) -> i32 =
    let sliced = arr[0:3] in
    let _ = consume(sliced) in
    arr[0]
"#;
    let result = check_alias(source);
    // Should ERROR: sliced aliases arr, consuming sliced invalidates arr
    assert!(
        result.has_errors(),
        "Borrowed slice aliases source; consuming slice should invalidate original"
    );
}

#[test]
fn test_multiple_borrowed_slices_alias_same_source() {
    // Multiple slices of the same array all alias the original.
    // Consuming one invalidates the others (they share the same backing store).
    let source = r#"
def consume(arr: *[3]i32) -> i32 = arr[0]

def main(arr: [10]i32) -> i32 =
    let s1 = arr[0:3] in
    let s2 = arr[3:6] in
    let _ = consume(s1) in
    consume(s2)
"#;
    let result = check_alias(source);
    // Should ERROR: s1 and s2 both alias arr, consuming s1 invalidates s2
    assert!(
        result.has_errors(),
        "Multiple borrowed slices alias same source; consuming one invalidates others"
    );
}

#[test]
fn test_consuming_source_invalidates_borrowed_slice() {
    // If we consume the original array, borrowed slices are invalidated.
    let source = r#"
def consume(arr: *[10]i32) -> i32 = arr[0]

def main(arr: [10]i32) -> i32 =
    let sliced = arr[0:3] in
    let _ = consume(arr) in
    sliced[0]
"#;
    let result = check_alias(source);
    // Should ERROR: sliced borrows from arr, consuming arr invalidates sliced
    assert!(
        result.has_errors(),
        "Borrowed slice is invalidated when source is consumed"
    );
}

#[test]
fn test_borrowed_slice_alias_chain() {
    // Slice bound to variable, then aliased - consuming alias invalidates original
    let source = r#"
def consume(arr: *[3]i32) -> i32 = arr[0]

def main(arr: [10]i32) -> i32 =
    let sliced = arr[0:3] in
    let alias = sliced in
    let _ = consume(alias) in
    arr[0]
"#;
    let result = check_alias(source);
    // Should ERROR: alias -> sliced -> arr, consuming alias invalidates arr
    assert!(
        result.has_errors(),
        "Borrowed slice alias chain: consuming alias invalidates original"
    );
}

#[test]
fn test_range_creates_fresh_backing_store() {
    // Range expression creates a fresh backing store
    let source = r#"
def consume(arr: *[5]i32) -> i32 = arr[0]

def main() -> i32 =
    let range = 0..<5 in
    consume(range)
"#;
    let result = check_alias(source);
    // Should be OK: range creates a fresh array
    assert!(!result.has_errors(), "Range should create fresh backing store");
}

#[test]
fn test_range_used_multiple_times() {
    // Range bound to variable can be used multiple times (before consuming)
    let source = r#"
def sum(arr: [5]i32) -> i32 = arr[0] + arr[1]

def main() -> i32 =
    let range = 0..<5 in
    sum(range) + sum(range)
"#;
    let result = check_alias(source);
    // Should be OK: sum doesn't consume, so range can be used multiple times
    assert!(
        !result.has_errors(),
        "Range can be used multiple times before being consumed"
    );
}

#[test]
fn test_borrowed_slice_aliases_base_array() {
    // Borrowed slices alias their base array
    // Consuming the slice should prevent using the original
    let source = r#"
def consume(arr: *[3]i32) -> i32 = arr[0]

def main(arr: [9]i32) -> i32 =
    let sliced = arr[1:4] in
    let _ = consume(sliced) in
    arr[0]
"#;
    let result = check_alias(source);
    // Should ERROR: sliced borrows from arr, consuming sliced invalidates arr
    assert!(
        result.has_errors(),
        "Borrowed slice should alias base array - using original after consuming slice is invalid"
    );
}

#[test]
fn test_borrowed_slice_non_consuming_ok() {
    // Using a borrowed slice without consuming allows original to still be used
    let source = r#"
def borrow(arr: [3]i32) -> i32 = arr[0]

def main(arr: [9]i32) -> i32 =
    let sliced = arr[1:4] in
    let _ = borrow(sliced) in
    arr[0]
"#;
    let result = check_alias(source);
    // Should be OK: borrow doesn't consume, so arr is still usable
    assert!(
        !result.has_errors(),
        "Non-consuming use of borrowed slice should allow original array to be used"
    );
}

#[test]
fn test_unzip_aliasing() {
    // unzip needs to use the input array twice (once per output array)
    // This tests whether we can map over the same array twice
    let source = r#"
def unzip(xys: [4](i32, i32)) -> ([4]i32, [4]i32) =
    (map(|xy| let (a, _) = xy in a, xys), map(|xy| let (_, b) = xy in b, xys))
"#;
    let result = check_alias(source);
    // map borrows (doesn't consume) its input, so xys can be used twice
    assert!(
        !result.has_errors(),
        "unzip should be able to map over input twice (non-consuming)"
    );
}
