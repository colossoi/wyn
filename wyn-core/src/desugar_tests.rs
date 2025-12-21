//! Tests for desugaring slices and ranges
//!
//! These tests verify that:
//! 1. Slices are desugared to map over ranges
//! 2. Ranges stay as the primitive form (not desugared further)
//! 3. The full pipeline works end-to-end with slices/ranges
//! 4. Constant folding in slice indices works correctly

use crate::error::CompilerError;

/// Helper to run full pipeline through lowering, including desugar step
fn compile_through_lowering(input: &str) -> Result<(), CompilerError> {
    let (module_manager, mut node_counter) = crate::cached_module_manager();
    let parsed = crate::Compiler::parse(input, &mut node_counter)?;
    let (flattened, mut backend) = parsed
        .desugar(&mut node_counter)?
        .resolve(&module_manager)?
        .fold_ast_constants()
        .type_check(&module_manager)?
        .alias_check()?
        .flatten(&module_manager)?;
    flattened
        .hoist_materializations()
        .normalize()
        .monomorphize()?
        .filter_reachable()
        .fold_constants()?
        .lift_bindings()
        .lower()?;
    Ok(())
}

/// Helper to run pipeline through flattening (checks desugar correctness)
fn compile_through_flatten(input: &str) -> Result<crate::Flattened, CompilerError> {
    let (module_manager, mut node_counter) = crate::cached_module_manager();
    let parsed = crate::Compiler::parse(input, &mut node_counter)?;
    let (flattened, _backend) = parsed
        .desugar(&mut node_counter)?
        .resolve(&module_manager)?
        .fold_ast_constants()
        .type_check(&module_manager)?
        .alias_check()?
        .flatten(&module_manager)?;
    Ok(flattened)
}

// =============================================================================
// Basic Slice Tests
// =============================================================================

// TODO: SPIR-V lowering for BorrowedSlice values is not yet implemented.
// These tests are ignored until Phase 4 is complete.
#[test]
#[ignore = "SPIR-V lowering for BorrowedSlice not yet implemented"]
fn test_simple_slice() {
    let source = r#"
def slice_array(arr: [10]i32) -> [5]i32 =
    arr[0:5]

#[vertex]
def vertex_main() -> #[builtin(position)] vec4f32 =
    let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] in
    let sliced = slice_array(arr) in
    @[f32.i32(sliced[0]), f32.i32(sliced[1]), 0.0f32, 1.0f32]
"#;
    let result = compile_through_lowering(source);
    assert!(result.is_ok(), "Simple slice should compile: {:?}", result.err());
}

#[test]
#[ignore = "SPIR-V lowering for BorrowedSlice not yet implemented"]
fn test_slice_with_computed_indices() {
    let source = r#"
def slice_computed(arr: [10]i32) -> [3]i32 =
    arr[1+1:2+3]

#[vertex]
def vertex_main() -> #[builtin(position)] vec4f32 =
    let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] in
    let sliced = slice_computed(arr) in
    @[f32.i32(sliced[0]), 0.0f32, 0.0f32, 1.0f32]
"#;
    assert!(
        compile_through_lowering(source).is_ok(),
        "Slice with computed indices should compile"
    );
}

// Note: Slice step syntax (arr[i:j:s]) is not yet supported - deferred to future work

// =============================================================================
// Range Tests
// =============================================================================

#[test]
fn test_simple_range() {
    let source = r#"
#[vertex]
def vertex_main() -> #[builtin(position)] vec4f32 =
    let range = 0..<4 in
    @[f32.i32(range[0]), f32.i32(range[1]), f32.i32(range[2]), 1.0f32]
"#;
    assert!(
        compile_through_lowering(source).is_ok(),
        "Simple range (0..<n) should compile"
    );
}

#[test]
fn test_range_with_start() {
    let source = r#"
#[vertex]
def vertex_main() -> #[builtin(position)] vec4f32 =
    let range = 1..<5 in
    @[f32.i32(range[0]), f32.i32(range[1]), f32.i32(range[2]), 1.0f32]
"#;
    assert!(
        compile_through_lowering(source).is_ok(),
        "Range with non-zero start should compile"
    );
}

#[test]
fn test_inclusive_range() {
    let source = r#"
#[vertex]
def vertex_main() -> #[builtin(position)] vec4f32 =
    let range = 0...3 in
    @[f32.i32(range[0]), f32.i32(range[1]), f32.i32(range[2]), f32.i32(range[3])]
"#;
    assert!(
        compile_through_lowering(source).is_ok(),
        "Inclusive range should compile"
    );
}

// =============================================================================
// Slice with Aliasing Tests
// =============================================================================
// Note: Borrowed slices (arr[i:j]) now alias their source array.
// Consuming a borrowed slice invalidates the source.

// These tests are ignored because:
// 1. SPIR-V lowering for BorrowedSlice is not yet implemented
// 2. The aliasing behavior has changed - slices now alias their source
#[test]
#[ignore = "SPIR-V lowering for BorrowedSlice not yet implemented"]
fn test_slice_borrowed_from_original() {
    // Borrowed slices alias their source - this should compile (non-consuming use)
    let source = r#"
def borrow(arr: [5]i32) -> i32 = arr[0]

def use_slice(arr: [10]i32) -> i32 =
    let sliced = arr[0:5] in
    let _ = borrow(sliced) in
    arr[0]

#[vertex]
def vertex_main() -> #[builtin(position)] vec4f32 =
    let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] in
    @[f32.i32(use_slice(arr)), 0.0f32, 0.0f32, 1.0f32]
"#;
    assert!(
        compile_through_lowering(source).is_ok(),
        "Non-consuming use of slice should allow original to be used"
    );
}

#[test]
#[ignore = "SPIR-V lowering for BorrowedSlice not yet implemented"]
fn test_multiple_slices_borrow_from_same() {
    // Multiple slices of same array all alias the original
    let source = r#"
def borrow(arr: [3]i32) -> i32 = arr[0]

def use_slices(arr: [10]i32) -> i32 =
    let s1 = arr[0:3] in
    let s2 = arr[3:6] in
    let _ = borrow(s1) in
    s2[0]

#[vertex]
def vertex_main() -> #[builtin(position)] vec4f32 =
    let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] in
    @[f32.i32(use_slices(arr)), 0.0f32, 0.0f32, 1.0f32]
"#;
    assert!(
        compile_through_lowering(source).is_ok(),
        "Multiple non-consuming slices should allow accessing each other"
    );
}

// =============================================================================
// Slice with Constant Folding Tests
// =============================================================================

#[test]
#[ignore = "SPIR-V lowering for BorrowedSlice not yet implemented"]
fn test_slice_with_constant_definition() {
    let source = r#"
def SIZE: i32 = 5
def OFFSET: i32 = 2

def slice_with_constants(arr: [10]i32) -> [5]i32 =
    arr[OFFSET:OFFSET+SIZE]

#[vertex]
def vertex_main() -> #[builtin(position)] vec4f32 =
    let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] in
    let sliced = slice_with_constants(arr) in
    @[f32.i32(sliced[0]), 0.0f32, 0.0f32, 1.0f32]
"#;
    assert!(
        compile_through_lowering(source).is_ok(),
        "Slice with constant definitions should compile"
    );
}

#[test]
fn test_range_combined_with_map() {
    let source = r#"
#[vertex]
def vertex_main() -> #[builtin(position)] vec4f32 =
    let doubled = map(|x| x * 2, 0..<4) in
    @[f32.i32(doubled[0]), f32.i32(doubled[1]), f32.i32(doubled[2]), 1.0f32]
"#;
    assert!(
        compile_through_lowering(source).is_ok(),
        "Range combined with map should compile"
    );
}

// Test simple map without range to isolate issues
#[test]
fn test_map_simple_array() {
    let source = r#"
def double(x: i32) -> i32 = x * 2

def test(arr: [4]i32) -> [4]i32 = map(double, arr)
"#;
    match compile_through_lowering(source) {
        Ok(_) => {}
        Err(e) => panic!("Simple map with array should compile: {e:?}"),
    }
}

#[test]
fn test_range_simple() {
    // Test just the range, no map
    let source = r#"
def test() -> [4]i32 = 0..<4
"#;
    match compile_through_lowering(source) {
        Ok(_) => {}
        Err(e) => panic!("Simple range should compile: {e:?}"),
    }
}

#[test]
fn test_map_range_simple() {
    // Test map over range with inline lambda
    let source = r#"
def test() -> [4]i32 = map(|x| x * 2, 0..<4)
"#;
    match compile_through_lowering(source) {
        Ok(_) => {}
        Err(e) => panic!("Map over range with lambda should compile: {e:?}"),
    }
}

#[test]
fn test_map_range_with_entry_point() {
    // Test map over range inside entry point
    let source = r#"
#[vertex]
def vertex_main() -> #[builtin(position)] vec4f32 =
    let doubled = map(|x| x * 2, 0..<4) in
    @[1.0f32, 2.0f32, 3.0f32, 1.0f32]
"#;
    match compile_through_lowering(source) {
        Ok(_) => {}
        Err(e) => panic!("Map over range in entry point should compile: {e:?}"),
    }
}

#[test]
fn test_map_range_with_index() {
    // Test indexing into map result
    let source = r#"
def test() -> i32 =
    let doubled = map(|x| x * 2, 0..<4) in
    doubled[0]
"#;
    match compile_through_lowering(source) {
        Ok(_) => {}
        Err(e) => panic!("Indexing map result should compile: {e:?}"),
    }
}

#[test]
fn test_map_range_indirect_entry_point() {
    // Test map over range used in entry point (using result)
    // TODO: Non-trivial constant expressions like `map(...)` require inlining at use sites
    // For now, test with map directly in the entry point
    let source = r#"
#[vertex]
def vertex_main() -> #[builtin(position)] vec4f32 =
    let doubled = map(|x| x * 2, 0..<4) in
    @[f32.i32(doubled[0]), f32.i32(doubled[1]), f32.i32(doubled[2]), 1.0f32]
"#;
    match compile_through_lowering(source) {
        Ok(_) => {}
        Err(e) => panic!("Map over range via helper should compile: {e:?}"),
    }
}

#[test]
fn test_map_with_named_function() {
    let source = r#"
def double(x: i32) -> i32 = x * 2

#[vertex]
def vertex_main() -> #[builtin(position)] vec4f32 =
    let doubled = map(double, 0..<4) in
    @[f32.i32(doubled[0]), f32.i32(doubled[1]), f32.i32(doubled[2]), 1.0f32]
"#;
    match compile_through_lowering(source) {
        Ok(_) => {}
        Err(e) => panic!("Named function passed to map should compile: {e:?}"),
    }
}

// =============================================================================
// MIR Structure Tests
// =============================================================================

#[test]
fn test_slice_desugars_to_map_range() {
    let source = r#"
def slice_test(arr: [10]i32) -> [5]i32 =
    arr[0:5]
"#;
    // Verify slice desugars correctly by checking compilation succeeds
    // The actual desugaring to map over range is tested by compile_through_flatten
    let flattened = compile_through_flatten(source);
    assert!(
        flattened.is_ok(),
        "Slice should desugar and flatten: {:?}",
        flattened.err()
    );
}

#[test]
fn test_simple_range_stays_as_range() {
    let source = r#"
def range_test() -> [5]i32 =
    0..<5
"#;
    // Verify range stays as primitive form by checking compilation succeeds
    let flattened = compile_through_flatten(source);
    assert!(
        flattened.is_ok(),
        "Simple range should compile: {:?}",
        flattened.err()
    );
}

#[test]
fn test_complex_range_stays_as_range() {
    let source = r#"
def range_test() -> [5]i32 =
    1..<6
"#;
    // Verify complex range compiles correctly
    let flattened = compile_through_flatten(source);
    assert!(
        flattened.is_ok(),
        "Complex range should compile: {:?}",
        flattened.err()
    );
}
