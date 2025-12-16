#![cfg(test)]

use crate::lexer::tokenize;
use crate::mir;
use crate::parser::Parser;
use crate::type_checker::TypeChecker;

fn flatten_program(input: &str) -> mir::Program {
    // Use the typestate API to ensure proper compilation pipeline
    let (module_manager, mut node_counter) = crate::cached_module_manager();
    let parsed = crate::Compiler::parse(input, &mut node_counter).expect("Parsing failed");
    let (flattened, _backend) = parsed
        .desugar(&mut node_counter)
        .expect("Desugaring failed")
        .resolve(&module_manager)
        .expect("Name resolution failed")
        .fold_ast_constants()
        .type_check(&module_manager)
        .expect("Type checking failed")
        .alias_check()
        .expect("Borrow checking failed")
        .flatten(&module_manager)
        .expect("Flattening failed");
    flattened.mir
}

fn flatten_to_string(input: &str) -> String {
    format!("{}", flatten_program(input))
}

/// Helper to check that code fails type checking (for testing error cases)
fn should_fail_type_check(input: &str) -> bool {
    let (module_manager, mut node_counter) = crate::cached_module_manager();
    let tokens = tokenize(input).expect("Tokenization failed");
    let mut parser = Parser::new(tokens, &mut node_counter);
    let ast = parser.parse().expect("Parsing failed");

    let mut type_checker = TypeChecker::new(&module_manager);
    type_checker.load_builtins().expect("Failed to load builtins");
    type_checker.check_program(&ast).is_err()
}

#[test]
fn test_simple_constant() {
    let mir = flatten_to_string("def x() = 42");
    assert!(mir.contains("def x:"));
    assert!(mir.contains("42"));
}

#[test]
fn test_simple_function() {
    let mir = flatten_to_string("def add(x, y) = x + y");
    // MIR format includes types: def add (x: type) (y: type): return_type =
    assert!(mir.contains("def add"));
    assert!(mir.contains("(x + y)"));
}

#[test]
fn test_let_binding() {
    let mir = flatten_to_string("def f() = let x = 1 in x + 2");
    // Format now includes binding ID: let x{N} = 1 in
    assert!(mir.contains("let x{"));
    assert!(mir.contains("} = 1 in"));
}

#[test]
fn test_tuple_pattern() {
    let mir = flatten_to_string("def f() = let (a, b) = (1, 2) in a + b");
    // Should generate tuple extraction
    assert!(mir.contains("tuple_access"));
}

#[test]
fn test_lambda_defunctionalization() {
    let mir = flatten_program("def f() = |x| x + 1");
    // Should generate a lambda function
    assert!(mir.defs.len() >= 2); // Original + lambda

    // Check that closure is created
    let mir_str = format!("{}", mir);
    assert!(mir_str.contains("_w_lam_f_"));
    // Closure is now an explicit @closure expression
    assert!(mir_str.contains("@closure(_w_lam_f_"));
}

#[test]
fn test_lambda_with_capture() {
    let mir = flatten_program("def f(y) = let g = |x| x + y in g(1)");
    let mir_str = format!("{}", mir);

    // Lambda should capture y
    assert!(mir_str.contains("_w_closure"));
    // Should reference y from closure
    assert!(mir_str.contains("record_access") || mir_str.contains("_w_closure"));
}

#[test]
fn test_nested_let() {
    let mir = flatten_to_string("def f() = let x = 1 in let y = 2 in x + y");
    // Format now includes binding ID: let x{N} = 1 in
    assert!(mir.contains("let x{"));
    assert!(mir.contains("let y{"));
}

#[test]
fn test_if_expression() {
    let mir = flatten_to_string("def f(x) = if x then 1 else 0");
    assert!(mir.contains("if x then 1 else 0"));
}

#[test]
fn test_function_call() {
    let mir = flatten_to_string("def g(y: i32) -> i32 = y + 1\ndef f(x) = g(x)");
    assert!(mir.contains("def g"));
    assert!(mir.contains("def f"));
}

#[test]
fn test_array_literal() {
    let mir = flatten_to_string("def arr() = [1, 2, 3]");
    assert!(mir.contains("[1, 2, 3]"));
}

#[test]
fn test_record_literal() {
    // Records are now represented as tuples in MIR, with fields in source order
    let mir = flatten_to_string("def r() = {x: 1, y: 2}");
    // Expect tuple representation (1, 2) instead of {x=1, y=2}
    assert!(mir.contains("(1, 2)"));
}

#[test]
fn test_while_loop() {
    let mir = flatten_to_string("def f() = loop x = 0 while x < 10 do x + 1");
    assert!(mir.contains("loop"));
    assert!(mir.contains("while"));
}

#[test]
fn test_for_range_loop() {
    let mir = flatten_to_string("def f() = loop acc = 0 for i < 10 do acc + i");
    assert!(mir.contains("loop"));
    assert!(mir.contains("for i <"));
}

#[test]
fn test_binary_ops() {
    let mir = flatten_to_string("def f(x, y) = x * y + x / y");
    assert!(mir.contains("*"));
    assert!(mir.contains("+"));
    assert!(mir.contains("/"));
}

#[test]
fn test_unary_op() {
    let mir = flatten_to_string("def f(x) = -x");
    assert!(mir.contains("(-x)"));
}

#[test]
fn test_array_index() {
    let mir = flatten_to_string("def f(arr, i) = arr[i]");
    assert!(mir.contains("index"));
}

#[test]
fn test_multiple_lambdas() {
    let mir = flatten_program(
        r#"
def f() =
    let a = |x| x + 1 in
    let b = |y| y * 2 in
    (a, b)
"#,
    );
    // Should have original + 2 lambdas
    assert!(mir.defs.len() >= 3);
}

#[test]
fn test_map_with_lambda() {
    // Test map with inline lambda
    let source = r#"
def test() -> [4]i32 =
    map((|x: i32| x + 1), [0, 1, 2, 3])
"#;

    // Parse
    let (module_manager, mut node_counter) = crate::cached_module_manager();
    let parsed = crate::Compiler::parse(source, &mut node_counter).expect("Parsing failed");
    let desugared = parsed.desugar(&mut node_counter).expect("Desugaring failed");

    // Resolve
    let resolved = desugared.resolve(&module_manager).expect("Name resolution failed");

    // Fold AST constants (before type checking for better size inference)
    let folded = resolved.fold_ast_constants();

    // Print AST to see what NodeId(6) is
    println!("AST:");
    println!("{:#?}", folded.ast);

    // Type check
    let typed = folded.type_check(&module_manager).expect("Type checking failed");
    println!("\nType table has {} entries", typed.type_table.len());
    for (id, scheme) in &typed.type_table {
        println!("  NodeId({:?}): {:?}", id, scheme);
    }

    println!("\nNodeId(6) is missing from type table!");

    // Alias check
    let alias_checked = typed.alias_check().expect("Alias checking failed");

    // Flatten
    let (flattened, _backend) = alias_checked.flatten(&module_manager).expect("Flattening failed");
    let mir_str = format!("{}", flattened.mir);
    println!("MIR: {}", mir_str);
    assert!(mir_str.contains("def test"));
}

#[test]
fn test_lambda_captures_typed_variable() {
    // This test reproduces an issue where a lambda captures a typed variable (like an array),
    // and the free variable rewriting creates _w_closure.mat, which then fails when trying
    // to resolve 'mat' as a field access on the closure.
    let mir = flatten_program(
        r#"
def test_capture(arr: [4]i32) -> i32 =
    let result = map((|i: i32| arr[i]), [0, 1, 2, 3]) in
    result[0]
"#,
    );
    let mir_str = format!("{}", mir);
    // Lambda should capture arr and access it via closure
    assert!(mir_str.contains("_w_closure") || mir_str.contains("record_access"));
}

#[test]
fn test_qualified_name_f32_sqrt() {
    // This test reproduces an issue where f32.sqrt is treated as field access
    // instead of a qualified builtin name. The identifier 'f32' has no type in
    // the type_table because it's a type name, not a variable.
    let mir = flatten_program(
        r#"
def length2(v: vec2f32) -> f32 =
    f32.sqrt((v.x * v.x + v.y * v.y))
"#,
    );
    let mir_str = format!("{}", mir);
    // Should contain f32.sqrt as a qualified name/call, not a field access error
    assert!(mir_str.contains("f32.sqrt"));
}

#[test]
fn test_map_with_closure_application() {
    // This test checks that map with a lambda generates a closure record
    // and registers the lambda in the registry for dispatch.
    let mir = flatten_program(
        r#"
def test_map(arr: [4]i32) -> [4]i32 =
    map((|x: i32| x + 1), arr)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    println!("Lambda registry: {:?}", mir.lambda_registry);

    // Should have generated lambda function
    assert!(mir_str.contains("_w_lam_test_map_"));
    // Closure is now an explicit @closure expression
    assert!(mir_str.contains("@closure(_w_lam_test_map_"));
    // Lambda registry should have the lambda function
    assert_eq!(mir.lambda_registry.len(), 1);
    let (_, info) = mir.lambda_registry.iter().next().unwrap();
    assert_eq!(info.name, "_w_lam_test_map_0");
    assert_eq!(info.arity, 1);
}

#[test]
fn test_direct_closure_call() {
    // This test checks that directly calling a closure generates a direct lambda call
    let mir = flatten_program(
        r#"
def test_apply(x: i32) -> i32 =
    let f = |y: i32| y + x in
    f(10)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    println!("Lambda registry: {:?}", mir.lambda_registry);

    // Should generate a direct call to _w_lam_test_apply_0 with the closure as first argument
    assert!(mir_str.contains("_w_lam_test_apply_0 f 10"));
    // Should NOT generate apply1 intrinsic
    assert!(!mir_str.contains("apply1"));
}

// Tests for function value restrictions (Futhark-style defunctionalization constraints)

#[test]
fn test_error_array_of_functions() {
    // Arrays of functions are not permitted
    assert!(
        should_fail_type_check(
            r#"
def test() -> [2](i32 -> i32) =
    [|x: i32| x + 1, |x: i32| x * 2]
"#
        ),
        "Should reject arrays of functions"
    );
}

#[test]
fn test_error_function_from_if() {
    // A function cannot be returned from an if expression
    assert!(
        should_fail_type_check(
            r#"
def choose(b: bool) -> (i32 -> i32) =
    if b then |x: i32| x + 1 else |x: i32| x * 2
"#
        ),
        "Should reject function returned from if expression"
    );
}

#[test]
fn test_error_loop_parameter_function() {
    // A loop parameter cannot be a function
    assert!(
        should_fail_type_check(
            r#"
def test() -> (i32 -> i32) =
    loop f = |x: i32| x while false do f
"#
        ),
        "Should reject function as loop parameter"
    );
}

#[test]
fn test_lambda_with_vector_literal() {
    // Test that vector literals work inside lambdas
    let mir = flatten_program(
        r#"
def test(v: vec3f32) -> vec4f32 =
    let f = |x: vec3f32| @[x.x, x.y, x.z, 1.0f32] in
    f(v)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);

    // Should contain vector literal
    assert!(mir_str.contains("@["));
}

#[test]
fn test_f32_sum_inline_definition() {
    // Test sum with inline definition (simpler than using prelude)
    let mir = flatten_program(
        r#"
def mysum<[n]>(arr: [n]f32) -> f32 =
  let (result, _) = loop (acc, i) = (0.0f32, 0) while i < length(arr) do
    (acc + arr[i], i + 1)
  in result

def test() -> f32 =
  let arr = [1.0f32, 2.0f32, 3.0f32] in
  mysum(arr)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("mysum"));
}

#[test]
fn test_f32_sum_simple() {
    // Test that f32.sum from prelude works through full compilation
    let source = r#"
def test() -> f32 =
  let arr = [1.0f32, 2.0f32, 3.0f32] in
  f32.sum(arr)
"#;

    // This should compile successfully
    let (mm, mut nc) = crate::cached_module_manager();
    let result = crate::Compiler::parse(source, &mut nc)
        .and_then(|p| p.desugar(&mut nc))
        .and_then(|d| d.resolve(&mm))
        .map(|r| r.fold_ast_constants())
        .and_then(|f| f.type_check(&mm))
        .and_then(|t| t.alias_check())
        .and_then(|a| a.flatten(&mm))
        .map(|(f, mut backend)| {
            let h = f.hoist_materializations();
            let n = h.normalize(&mut backend.node_counter);
            (n, backend)
        })
        .and_then(|(n, _backend): (crate::Normalized, _)| n.monomorphize())
        .map(|m| m.filter_reachable())
        .and_then(|r| r.fold_constants())
        .and_then(|f| f.lift_bindings())
        .and_then(|l| l.lower());
    assert!(result.is_ok(), "Compilation failed: {:?}", result.err());
}

#[test]
fn test_f32_conversions() {
    // Test f32 type conversion via module functions
    let mir = flatten_program(
        r#"
def test_conversions(x: i32) -> f32 =
  let f1 = f32.i32(x) in
  let i1 = f32.to_i64(f1) in
  let f2 = f32.i64(i1) in
  f2
"#,
    );
    let mir_str = format!("{}", mir);
    // Should contain the conversion calls
    assert!(mir_str.contains("f32.i32") || mir_str.contains("_w_builtin_f32_from"));
}

#[test]
fn test_f32_math_operations() {
    // Test f32 math operations including GLSL extended ops
    let mir = flatten_program(
        r#"
def test_math(x: f32) -> f32 =
  let a = f32.sin(x) in
  let b = f32.cos(x) in
  let c = f32.sqrt(a) in
  let d = f32.exp(b) in
  let e = f32.log(c) in
  let f = d ** 2.0f32 in
  let g = f32.sinh(x) in
  let h = f32.asinh(g) in
  let i = f32.atan2(x, a) in
  f32.fma(f, e, i)
"#,
    );
    let mir_str = format!("{}", mir);
    // Should contain f32 math operations
    assert!(mir_str.contains("f32.sin"));
    assert!(mir_str.contains("f32.cos"));
    assert!(mir_str.contains("f32.sqrt"));
    assert!(mir_str.contains("f32.sinh"));
    assert!(mir_str.contains("f32.asinh"));
    assert!(mir_str.contains("f32.atan2"));
    assert!(mir_str.contains("f32.fma"));
}

#[test]
fn test_operator_section_direct_application() {
    // Test operator section applied directly to arguments
    let mir = flatten_program(
        r#"
def test_add(x: i32, y: i32) -> i32 = (+)(x, y)
"#,
    );
    let mir_str = format!("{}", mir);
    // Operator section (+) applied to arguments should flatten correctly
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_add"));
}

#[test]
fn test_operator_section_with_map() {
    // Test operator section passed to map (special higher-order function)
    let mir = flatten_program(
        r#"
def test_map(arr: [3]i32) -> [3]i32 = map((|x: i32| (+)(x, 1)), arr)
"#,
    );
    let mir_str = format!("{}", mir);
    // Should successfully flatten with lambda wrapping operator section
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_map"));
}

#[test]
fn test_mul_overloads() {
    // Test all three overloaded versions of mul:
    // - mul_mat_mat: mat * mat -> mat
    // - mul_mat_vec: mat * vec -> vec
    // - mul_vec_mat: vec * mat -> vec
    // Note: Square matrices use mat4f32 syntax (not mat4x4f32)
    let mir = flatten_program(
        r#"
def test_mul_overloads(m1: mat4f32, m2: mat4f32, v: vec4f32) -> vec4f32 =
    let mat_result = mul(m1, m2) in          -- mul_mat_mat
    let vec_result1 = mul(mat_result, v) in  -- mul_mat_vec
    let vec_result2 = mul(v, m1) in          -- mul_vec_mat
    vec_result1
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);

    // All three should be desugared to their specific variants
    assert!(mir_str.contains("mul_mat_mat"), "Expected mul_mat_mat in MIR");
    assert!(mir_str.contains("mul_mat_vec"), "Expected mul_mat_vec in MIR");
    assert!(mir_str.contains("mul_vec_mat"), "Expected mul_vec_mat in MIR");
}

#[test]
fn test_no_redundant_materializations_simple_var() {
    // Test that accessing the same array with dynamic index in multiple branches
    // does not create redundant materializations for simple variables.
    // The backing store should be created once at the top level and reused.
    let mir = flatten_program(
        r#"
def test(arr: [3]i32, i: i32) -> i32 =
  let x = arr[i] in
  if true then arr[i] else x
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);

    // Count materializations - should be exactly 1 (for the backing store)
    let materialize_count = mir_str.matches("@materialize").count();
    assert!(
        materialize_count <= 1,
        "Expected at most 1 materialize, found {}. MIR:\n{}",
        materialize_count,
        mir_str
    );
}

#[test]
fn test_no_redundant_materializations_complex_expr() {
    // Test that indexing a complex expression (not a simple variable) in both
    // branches of an if does not create TWO separate materializations.
    // The materialize_hoisting pass should hoist the common materialization.
    let source = r#"
def identity(arr: [3]i32) -> [3]i32 = arr
def test(arr: [3]i32, i: i32) -> i32 =
  if true then (identity(arr))[i] else (identity(arr))[i]
"#;

    // Run through normalize + hoist_materializations
    let (mm, mut nc) = crate::cached_module_manager();
    let parsed = crate::Compiler::parse(source, &mut nc).expect("parse failed");
    let (flattened, _backend) = parsed
        .desugar(&mut nc)
        .expect("desugar failed")
        .resolve(&mm)
        .expect("resolve failed")
        .fold_ast_constants()
        .type_check(&mm)
        .expect("type_check failed")
        .alias_check()
        .expect("alias_check failed")
        .flatten(&mm)
        .expect("flatten failed");
    let hoisted = flattened.hoist_materializations();

    let mir_str = format!("{}", hoisted.mir);
    println!("MIR output after hoisting:\n{}", mir_str);

    // After hoisting, there should be at most 1 materialize
    let materialize_count = mir_str.matches("@materialize").count();
    assert!(
        materialize_count <= 1,
        "Expected at most 1 materialize, found {}. MIR:\n{}",
        materialize_count,
        mir_str
    );
}

#[test]
fn test_no_materialize_for_loop_tuple_destructuring() {
    // Loop tuple destructuring should use tuple_access directly, not materialize
    let mir = flatten_program(
        r#"
def test(arr: [10]i32) -> i32 =
  let (sum, _) = loop (acc, i) = (0, 0) while i < 10 do
    (acc + arr[i], i + 1)
  in sum
"#,
    );

    let mir_str = format!("{}", mir);
    println!("MIR:\n{}", mir_str);

    // The loop tuple pattern (acc, i) should NOT require materialize
    // Only the array index arr[i] might need it
    let materialize_count = mir_str.matches("@materialize").count();
    assert!(
        materialize_count <= 1,
        "Loop tuple destructuring should not use materialize. Found {} materializations. MIR:\n{}",
        materialize_count,
        mir_str
    );
}

#[test]
fn test_no_materialize_for_let_tuple_destructuring() {
    // Let tuple destructuring should use tuple_access directly, not materialize
    let mir = flatten_program(
        r#"
def test(x: i32) -> i32 =
  let pair = (x, x + 1) in
  let (a, b) = pair in
  a + b
"#,
    );

    let mir_str = format!("{}", mir);
    println!("MIR:\n{}", mir_str);

    // Tuple destructuring should NOT require materialize at all
    let materialize_count = mir_str.matches("@materialize").count();
    assert_eq!(
        materialize_count, 0,
        "Let tuple destructuring should not use materialize. Found {} materializations. MIR:\n{}",
        materialize_count, mir_str
    );
}

// =============================================================================
// Currying and partial application tests
// =============================================================================

#[test]
fn test_curried_function_definition() {
    // Curried function with parenthesized typed parameters
    let mir = flatten_program(
        r#"
def add(x: f32, y: f32) -> f32 = x + y
def test() -> f32 = add(1.0f32, 2.0f32)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR:\n{}", mir_str);
    assert!(mir_str.contains("def add"));
    assert!(mir_str.contains("def test"));
}

#[test]
fn test_curried_function_application() {
    // Curried function application with multiple arguments
    let mir = flatten_program(
        r#"
def add(x: f32, y: f32) -> f32 = x + y
def test() -> f32 =
    let a = 1.0f32 in
    let b = 2.0f32 in
    add(a, b)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR:\n{}", mir_str);
    assert!(mir_str.contains("add"));
}

#[test]
fn test_higher_order_function_parameter() {
    // Function parameter with arrow type
    let mir = flatten_program(
        r#"
def apply(f: f32 -> f32, x: f32) -> f32 = f(x)
def double(x: f32) -> f32 = x + x
def test() -> f32 = apply(double, 3.0f32)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR:\n{}", mir_str);
    assert!(mir_str.contains("def apply"));
    assert!(mir_str.contains("def double"));
}

#[test]
fn test_binary_function_parameter() {
    // Binary function parameter (f32 -> f32 -> f32)
    let mir = flatten_program(
        r#"
def fold2(op: f32 -> f32 -> f32, x: f32, y: f32) -> f32 = op(x, y)
def add(a: f32, b: f32) -> f32 = a + b
def test() -> f32 = fold2(add, 1.0f32, 2.0f32)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR:\n{}", mir_str);
    assert!(mir_str.contains("def fold2"));
}

#[test]
fn test_reduce_pattern() {
    // The reduce pattern with curried binary operator
    let mir = flatten_program(
        r#"
def reduce_f32(op: f32 -> f32 -> f32, init: f32, arr: [4]f32) -> f32 =
    let x0 = op(init, arr[0]) in
    let x1 = op(x0, arr[1]) in
    let x2 = op(x1, arr[2]) in
    op(x2, arr[3])

def test() -> f32 =
    let arr = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
    reduce_f32((|x, y| x + y), 0.0f32, arr)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR:\n{}", mir_str);
    assert!(mir_str.contains("def reduce_f32"));
}

#[test]
fn test_map_lambda_without_extra_parens() {
    // Reproduce issue: map with lambda without extra parentheses around the lambda
    // This matches the pattern: map(|v:vec3f32| ..., arr)
    // vs the working pattern: map((|v:vec3f32| ...), arr)
    let mir = flatten_program(
        r#"
def test_map_no_parens(arr: [4]i32) -> [4]i32 =
    map(|x: i32| x + 1, arr)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_map_no_parens"));
}

#[test]
fn test_map_lambda_with_capture() {
    // Reproduce issue: map with lambda that captures variable (like da_rasterizer)
    // Pattern: map(|v:type| func(v, captured_var), arr)
    let mir = flatten_program(
        r#"
def test_map_capture(arr: [4]i32, offset: i32) -> [4]i32 =
    map(|x: i32| x + offset, arr)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_map_capture"));
}

#[test]
fn test_generic_function_dot2() {
    // Reproduce issue from primitives.wyn: generic function dot2<E, T>
    let mir = flatten_program(
        r#"
def dot2<E, T>(v: T) -> E = dot(v, v)

def test_dot2(p: vec3f32) -> f32 =
    dot2(p)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_dot2"));
}

#[test]
fn test_map_with_mul_capture() {
    // Reproduce da_rasterizer issue: map with lambda capturing a mat4f32 and calling mul
    let mir = flatten_program(
        r#"
def cube_corners: [8]vec3f32 =
    [ @[-1.0, -1.0, 1.0],
      @[-1.0, 1.0, 1.0],
      @[1.0, 1.0, 1.0],
      @[1.0, -1.0, 1.0],
      @[-1.0, -1.0, -1.0],
      @[-1.0, 1.0, -1.0],
      @[1.0, 1.0, -1.0],
      @[1.0, -1.0, -1.0] ]

def test_map_mul(mat: mat4f32) -> [8]vec4f32 =
    map(|v:vec3f32| mul(@[v.x, v.y, v.z, 1.0], mat), cube_corners)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_map_mul"));
}

#[test]
fn test_multiline_map_lambda() {
    // Reproduce da_rasterizer pattern: multiline lambda with let-in inside map
    let mir = flatten_program(
        r#"
def arr: [4]f32 = [1.0, 2.0, 3.0, 4.0]

def test_multiline() -> [4]f32 =
    map(|q:f32|
              let zinv = 1.0 / q in
              zinv,
           arr)
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_multiline"));
}

#[test]
fn test_loop_with_map() {
    // Reproduce da_rasterizer pattern: map inside a loop
    let mir = flatten_program(
        r#"
def cube: [4]f32 = [1.0, 2.0, 3.0, 4.0]

def test_loop_map() -> f32 =
    let (_, acc) = loop (idx, acc) = (0i32, 0.0f32) while idx < 4 do
      let mat = 2.0f32 in
      let v4s = map(|v:f32| v * mat, cube) in
      (idx + 1, acc + v4s[0])
    in acc
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("test_loop_map"));
}

#[test]
fn test_da_rasterizer_minimal() {
    // Minimal reproduction of da_rasterizer.wyn structure
    let mir = flatten_program(
        r#"
#[uniform(set=1, binding=0)] def iResolution: vec2f32
#[uniform(set=1, binding=1)] def iTime: f32

def verts: [3]vec4f32 =
  [@[-1.0, -1.0, 0.0, 1.0],
   @[3.0, -1.0, 0.0, 1.0],
   @[-1.0, 3.0, 0.0, 1.0]]

#[vertex]
def vertex_main(#[builtin(vertex_index)] vertex_id:i32) -> #[builtin(position)] vec4f32 = verts[vertex_id]

def translation(p: vec3f32) -> mat4f32 =
  @[
    [1.0f32, 0.0f32, 0.0f32, p.x],
    [0.0f32, 1.0f32, 0.0f32, p.y],
    [0.0f32, 0.0f32, 1.0f32, p.z],
    [0.0f32, 0.0f32, 0.0f32, 1.0f32]
  ]

def rotation_euler(a: vec3f32) -> mat4f32 =
  let s = [f32.sin(a.x), f32.sin(a.y), f32.sin(a.z)] in
  let c = [f32.cos(a.x), f32.cos(a.y), f32.cos(a.z)] in
  @[
    [c[1]*c[2], c[1]*s[2], 0.0f32 - s[1], 0.0f32],
    [s[0]*s[1]*c[2] - c[0]*s[2], s[0]*s[1]*s[2] + c[0]*c[2], s[0]*c[1], 0.0f32],
    [c[0]*s[1]*c[2] + s[0]*s[2], c[0]*s[1]*s[2] - s[0]*c[2], c[0]*c[1], 0.0f32],
    [0.0f32, 0.0f32, 0.0f32, 1.0f32]
  ]

def cube_corners: [8]vec3f32 =
    [ @[-1.0, -1.0, 1.0],
      @[-1.0, 1.0, 1.0],
      @[1.0, 1.0, 1.0],
      @[1.0, -1.0, 1.0],
      @[-1.0, -1.0, -1.0],
      @[-1.0, 1.0, -1.0],
      @[1.0, 1.0, -1.0],
      @[1.0, -1.0, -1.0] ]

def main_image(iResolution:vec2f32, iTime:f32, fragCoord:vec2f32) -> vec4f32 =
  let cam = translation(@[0.0, 0.0, 10.0]) in
  let rot = rotation_euler(@[iTime, iTime*0.86, iTime*0.473]) in
  let rot_cam : mat4f32 = mul(rot, cam) in
  let mat  : mat4f32 = mul(translation(@[4.0, 4.0, -4.0]), rot_cam) in
  let v4s : [8]vec4f32 = map(|v:vec3f32| mul(@[v.x, v.y, v.z, 1.0], mat), cube_corners) in
  v4s[0]

#[fragment]
def fragment_main(#[builtin(frag_coord)] pos:vec4f32) -> #[location(0)] vec4f32 =
  main_image(@[iResolution.x, iResolution.y], iTime, @[pos.x, pos.y])
"#,
    );
    let mir_str = format!("{}", mir);
    println!("MIR output:\n{}", mir_str);
    assert!(mir_str.contains("fragment_main"));
}
