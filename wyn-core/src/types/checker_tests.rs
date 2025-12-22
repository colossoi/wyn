use super::checker::{TypeChecker, TypeWarning};
use super::{Type, TypeName};
use crate::error::CompilerError;

/// Helper to parse and type check source code, expecting success
fn typecheck_program(input: &str) {
    let result = try_typecheck_program(input);
    if let Err(e) = &result {
        eprintln!("\n=== TYPE CHECK ERROR ===");
        eprintln!("{:?}", e);
    }
    result.expect("Type checking should succeed");
}

/// Helper to parse and type check source code, returning Result
fn try_typecheck_program(input: &str) -> Result<(), CompilerError> {
    // Use the typestate API to ensure proper pipeline setup
    let mut frontend = crate::cached_frontend();
    let parsed = crate::Compiler::parse(input, &mut frontend.node_counter)?;
    let _type_checked = parsed
        .desugar(&mut frontend.node_counter)?
        .resolve(&frontend.module_manager)?
        .fold_ast_constants()
        .type_check(&frontend.module_manager, &mut frontend.schemes)?;
    Ok(())
}

#[test]
fn test_type_check_let() {
    typecheck_program("let x: i32 = 42");
}

#[test]
fn test_type_mismatch() {
    let result = try_typecheck_program("let x: i32 = 3.14f32");
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

#[test]
fn test_array_type_check() {
    typecheck_program("let arr: [2]f32 = [1.0f32, 2.0f32]");
}

#[test]
fn test_undefined_variable() {
    let result = try_typecheck_program("let x: i32 = undefined");
    match result {
        Err(CompilerError::UndefinedVariable(_, _)) => {}
        other => panic!("Expected UndefinedVariable error, got {:?}", other.err()),
    }
}

#[test]
fn test_simple_def() {
    typecheck_program("def identity(x) = x");
}

#[test]
fn test_two_length_and_replicate_calls() {
    // Simplified test: two calls to length/replicate with different array element types
    // This tests that type variables don't bleed between the two calls
    typecheck_program(
        r#"
def test: f32 =
    let v4s : [2]vec4f32 = [@[1.0f32, 2.0f32, 3.0f32, 4.0f32], @[5.0f32, 6.0f32, 7.0f32, 8.0f32]] in
    let len1 = length(v4s) in
    let out1 = replicate(len1, _w_uninit()) in

    let indices : [2]i32 = [0, 1] in
    let len2 = length(indices) in
    let out2 = replicate(len2, _w_uninit()) in

    42.0f32
        "#,
    );
}

#[test]
fn test_zip_arrays() {
    typecheck_program("def zip_arrays(xs, ys) = zip(xs, ys)");
}

#[test]
fn test_mul_concrete_matrix_types() {
    // This test verifies that polymorphic mul works with concrete matrix types
    typecheck_program(
        r#"
def test_mul(mat1: mat4f32, mat2: mat4f32) -> mat4f32 =
    mul(mat1, mat2)
        "#,
    );
}

/// Helper function to check a program with a type hole and return the inferred type
fn check_type_hole(source: &str) -> Type {
    use crate::lexer;
    use crate::parser::Parser;

    // Parse
    let (module_manager, mut node_counter) = crate::cached_module_manager();
    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens, &mut node_counter);
    let program = parser.parse().unwrap();

    // Type check
    let mut checker = TypeChecker::new(&module_manager);
    checker.load_builtins().unwrap();
    let _type_table = checker.check_program(&program).unwrap();

    // Check warnings
    let warnings = checker.warnings();
    assert_eq!(warnings.len(), 1, "Expected exactly one type hole warning");

    match &warnings[0] {
        TypeWarning::TypeHoleFilled { inferred_type, .. } => {
            // Apply the context to normalize type variables
            inferred_type.apply(checker.context())
        }
    }
}

#[test]
fn test_type_hole_in_array() {
    let inferred = check_type_hole("def arr = [1i32, ???, 3i32]");

    // ??? should be inferred as i32 (to match array elements)
    let expected = Type::Constructed(TypeName::Int(32), vec![]);
    assert_eq!(inferred, expected);
}

#[test]
fn test_type_hole_in_binop() {
    let inferred = check_type_hole("def result = 5i32 + ???");

    // ??? should be inferred as i32 (to match addition operand)
    let expected = Type::Constructed(TypeName::Int(32), vec![]);
    assert_eq!(inferred, expected);
}

#[test]
fn test_type_hole_function_arg() {
    let inferred = check_type_hole("def apply = (|x: i32| x + 1i32)(???)");

    // ??? should be inferred as i32 (the function argument type)
    let expected = Type::Constructed(TypeName::Int(32), vec![]);
    assert_eq!(inferred, expected);
}

#[test]
fn test_lambda_param_with_annotation() {
    // Test that lambda parameter works with type annotation (Futhark-style)
    // Field projection requires the parameter type to be known
    typecheck_program(
        "def test: [2]f32 = let arr : [2]vec3f32 = [@[1.0f32, 2.0f32, 3.0f32], @[4.0f32, 5.0f32, 6.0f32]] in map((|v: vec3f32| v.x), arr)",
    );
}

#[test]
fn test_bidirectional_with_concrete_type() {
    // Test bidirectional checking with a CONCRETE expected type
    // This demonstrates where bidirectional checking actually helps
    typecheck_program(
        r#"
            def apply_to_vec(f: vec3f32 -> f32) -> f32 =
              f(@[1.0f32, 2.0f32, 3.0f32])

            def test: f32 = apply_to_vec((|v| v.x))
        "#,
    );
}

#[test]
fn test_bidirectional_explicit_annotation_mismatch() {
    // Minimal test demonstrating bidirectional checking bug with explicit parameter annotations.
    // Two chained maps: vec3f32->vec4f32, then vec4f32->vec3f32
    // The second lambda's parameter annotation (q:vec4f32) is correct (v4s is [1]vec4f32),
    // but bidirectional checking incorrectly rejects it.
    typecheck_program(
        r#"
            def test =
              let arr : [1]vec3f32 = [@[1.0f32, 2.0f32, 3.0f32]] in
              let v4s : [1]vec4f32 = map((|v: vec3f32| @[v.x, v.y, v.z, 1.0f32]), arr) in
              map((|q: vec4f32| @[q.x, q.y, q.z]), v4s)
        "#,
    );
}

#[test]
fn test_map_with_unannotated_lambda_and_array_index() {
    // Test that bidirectional checking infers lambda parameter type from array type
    typecheck_program(
        r#"
            def test: [12]i32 =
              let edges : [12][2]i32 = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]] in
              map((|e| e[0]), edges)
        "#,
    );
}

#[test]
fn test_lambda_with_tuple_pattern() {
    // Test that lambdas with tuple patterns work
    typecheck_program(
        r#"
            def test: ((i32, i32) -> i32) =
              |(x, y)| x + y
        "#,
    );
}

#[test]
fn test_lambda_with_wildcard_in_tuple() {
    // Test that lambdas with wildcard in tuple patterns work
    typecheck_program(
        r#"
            def test: ((i32, i32) -> i32) =
              |(_, acc)| acc
        "#,
    );
}

// Tests for loop type checking will be added once Loop support is implemented
// in the type checker (currently todo!())

#[test]
fn test_map_with_array_size_inference() {
    typecheck_program(
        r#"
def test: [8]i32 =
  let arr = [1, 2, 3, 4, 5, 6, 7, 8] in
  map((|x| x + 1), arr)
"#,
    );
}

#[test]
fn test_let_polymorphism() {
    // Test that let-bound values are properly generalized
    // Without generalization, this would fail because id would be monomorphic
    typecheck_program(
        r#"
            def test: f32 =
                let id = |x| x in
                let test1 : i32 = id(42) in
                let test2 : f32 = id(3.14f32) in
                test2
        "#,
    );
}

#[test]
fn test_top_level_polymorphism() {
    // Test that top-level let/def declarations are generalized
    typecheck_program(
        r#"
            def id = |x| x
            def test1: i32 = id(42)
            def test2: f32 = id(3.14f32)
        "#,
    );
}

#[test]
fn test_polymorphic_id_tuple() {
    // Classic HM polymorphism test: let id = |x| x in (id 5, id true)
    typecheck_program(
        r#"
            def test =
                let id = |x| x in
                (id(5), id(true))
        "#,
    );
}

#[test]
fn test_qualified_name_sqrt() {
    // Test that qualified names like f32.sqrt type check correctly
    typecheck_program(
        r#"
            def test: f32 = f32.sqrt(4.0f32)
        "#,
    );
}

#[test]
fn test_nested_array_indexing() {
    // Test that nested array indexing type inference works
    // Reproduces the de_rasterizer.wyn issue: e[0] where e : [2]i32
    typecheck_program(
        r#"
            def test =
                let edges : [3][2]i32 = [[0,1], [1,2], [2,0]] in
                let verts : [4]f32 = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
                let e : [2]i32 = edges[0] in
                let idx : i32 = e[0] in
                verts[idx]
        "#,
    );
}

#[test]
fn test_nested_array_indexing_in_lambda() {
    // Test that nested array indexing works inside a lambda in map
    // This reproduces the actual de_rasterizer.wyn pattern:
    // map (|e| verts[e[0]]) edges
    typecheck_program(
        r#"
            def test =
                let edges : [3][2]i32 = [[0,1], [1,2], [2,0]] in
                let verts : [4]f32 = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
                map((|e| verts[e[0]]), edges)
        "#,
    );
}

#[test]
fn test_nested_array_indexing_with_literal() {
    // Test with array literal directly in map call, without type annotation
    // This is closer to the de_rasterizer pattern
    typecheck_program(
        r#"
            def test =
                let verts : [4]f32 = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
                map((|e| verts[e[0]]), [[0,1], [1,2], [2,0]])
        "#,
    );
}

#[test]
fn test_size_parameter_binding() {
    // Test that size parameters are properly bound and substituted
    typecheck_program(
        r#"
def identity<[n]>(xs: [n]i32) -> [n]i32 = xs

def test: [5]i32 =
  let arr = [1, 2, 3, 4, 5] in
  identity(arr)
"#,
    );
}

#[test]
fn test_f32_sum_simple() {
    // Test simple f32.sum call with array literal
    typecheck_program(
        r#"
def test: f32 = f32.sum([1.0f32, 2.0f32, 3.0f32])
    "#,
    );
}

#[test]
fn test_f32_sum_with_map_over_nested_array() {
    // Test f32.sum with map over nested arrays
    typecheck_program(
        r#"
def test: f32 =
    let edges : [3][2]i32 = [[0, 1], [1, 2], [2, 0]] in
    f32.sum(map((|e| 1.0f32), edges))
    "#,
    );
}

#[test]
fn test_map_lambda_param_type_unification() {
    // Test that lambda parameter type variable gets unified with input array element type
    typecheck_program(
        r#"
def test: [3]f32 =
    let edges : [3][2]i32 = [[0, 1], [1, 2], [2, 0]] in
    map((|e| let a = e[0] in 1.0f32), edges)
    "#,
    );
}

#[test]
fn test_map_with_capturing_closure() {
    // Test with dot product and f32.min
    typecheck_program(
        r#"
def line: f32 =
  let denom = dot(@[0.0f32, 0.0f32], @[9.0f32, 9.0f32]) in
  f32.min(denom, 1.0f32)
"#,
    );
}

#[test]
fn test_f32_sum_with_map_indexing_nested_array() {
    // Test f32.sum with map accessing elements of nested array
    typecheck_program(
        r#"
def test: f32 =
    let edges : [3][2]i32 = [[0, 1], [1, 2], [2, 0]] in
    f32.sum(map((|e| let a = e[0] in let b = e[1] in 1.0f32), edges))
    "#,
    );
}

// ===== Lambda/Closure Tests =====

#[test]
fn test_lambda_identity() {
    // Basic identity lambda
    typecheck_program("def id = |x| x");
}

#[test]
fn test_lambda_with_annotated_param() {
    // Lambda with type-annotated parameter
    typecheck_program("def inc = |x: i32| x + 1");
}

#[test]
fn test_lambda_with_annotated_return() {
    // Lambda with annotated return type via def
    typecheck_program("def inc: (i32 -> i32) = |x| x + 1");
}

#[test]
fn test_lambda_multi_param_curried() {
    // Multi-parameter lambda (curried form)
    typecheck_program("def add = |x| |y| x + y");
}

#[test]
fn test_lambda_application() {
    // Lambda applied to argument
    typecheck_program("def result: i32 = (|x| x + 1)(5)");
}

#[test]
fn test_lambda_curried_application() {
    // Curried lambda partially applied
    typecheck_program(
        r#"
def add = |x| |y| x + y
def add5 = add(5)
def result: i32 = add5(10)
"#,
    );
}

#[test]
fn test_lambda_capturing_variable() {
    // Lambda capturing outer variable (closure)
    typecheck_program(
        r#"
def test: i32 =
    let x = 10 in
    let f = |y| x + y in
    f(5)
"#,
    );
}

#[test]
fn test_lambda_capturing_multiple_variables() {
    // Lambda capturing multiple outer variables
    typecheck_program(
        r#"
def test: i32 =
    let a = 1 in
    let b = 2 in
    let c = 3 in
    let f = |x| a + b + c + x in
    f(4)
"#,
    );
}

#[test]
fn test_lambda_nested() {
    // Nested lambdas
    typecheck_program(
        r#"
def test: i32 =
    let outer = |x|
        let inner = |y| x + y in
        inner(10)
    in
    outer(5)
"#,
    );
}

#[test]
fn test_lambda_nested_capture() {
    // Nested lambda capturing from multiple scopes
    typecheck_program(
        r#"
def test: i32 =
    let a = 1 in
    let f = |x|
        let g = |y| a + x + y in
        g(3)
    in
    f(2)
"#,
    );
}

#[test]
fn test_lambda_as_argument() {
    // Lambda passed as argument to higher-order function
    typecheck_program(
        r#"
def apply(f: i32 -> i32, x: i32) -> i32 = f(x)
def result: i32 = apply((|x| x * 2), 5)
"#,
    );
}

#[test]
fn test_lambda_returned_from_function() {
    // Function returning a lambda
    typecheck_program(
        r#"
def make_adder(n: i32) -> (i32 -> i32) = |x| x + n
def add10 = make_adder(10)
def result: i32 = add10(5)
"#,
    );
}

#[test]
fn test_lambda_in_let_binding() {
    // Lambda bound in let expression
    typecheck_program(
        r#"
def test: i32 =
    let double = |x| x * 2 in
    double(21)
"#,
    );
}

#[test]
fn test_lambda_with_unit_param() {
    // Lambda with unit parameter
    typecheck_program("def get_five: (() -> i32) = |()| 5");
}

#[test]
fn test_lambda_returning_tuple() {
    // Lambda returning a tuple
    typecheck_program("def pair: (i32 -> (i32, i32)) = |x| (x, x + 1)");
}

#[test]
fn test_lambda_with_tuple_destructuring() {
    // Lambda with tuple pattern destructuring
    typecheck_program("def sum_pair: ((i32, i32) -> i32) = |(a, b)| a + b");
}

#[test]
fn test_lambda_chained_application() {
    // Chained lambda applications
    typecheck_program(
        r#"
def test: i32 =
    let f = |x| |y| |z| x + y + z in
    f(1)(2)(3)
"#,
    );
}

#[test]
fn test_lambda_polymorphic_usage() {
    // Lambda used polymorphically (via let generalization)
    typecheck_program(
        r#"
def test =
    let id = |x| x in
    let a : i32 = id(5) in
    let b : f32 = id(3.14f32) in
    (a, b)
"#,
    );
}

#[test]
fn test_lambda_with_array_operations() {
    // Lambda used with array operations
    typecheck_program(
        r#"
def test: [3]i32 =
    let arr = [1, 2, 3] in
    map((|x| x * 2), arr)
"#,
    );
}

#[test]
fn test_lambda_inferred_from_map_context() {
    // Lambda parameter type inferred from map's array element type
    typecheck_program(
        r#"
def test: [2]f32 =
    let vs : [2]vec3f32 = [@[1.0f32, 2.0f32, 3.0f32], @[4.0f32, 5.0f32, 6.0f32]] in
    map((|v| v.x), vs)
"#,
    );
}

#[test]
fn test_lambda_with_binary_ops() {
    // Lambda with various binary operations
    typecheck_program(
        r#"
def test: i32 =
    let f = |x| x * 2 + 3 - 1 in
    f(10)
"#,
    );
}

#[test]
fn test_lambda_with_comparison() {
    // Lambda with comparison operation
    typecheck_program(
        r#"
def gt10 = |x| x > 10
def test: bool = gt10(15)
"#,
    );
}

#[test]
fn test_lambda_compose() {
    // Function composition with lambdas
    typecheck_program(
        r#"
def compose(f: i32 -> i32, g: i32 -> i32) -> (i32 -> i32) =
    |x| f(g(x))
def double = |x| x * 2
def inc = |x| x + 1
def double_then_inc = compose(inc, double)
def result: i32 = double_then_inc(5)
"#,
    );
}

#[test]
fn test_lambda_type_error_param_mismatch() {
    // Type error: lambda parameter type mismatch
    assert!(
        try_typecheck_program(
            r#"
def test: i32 = (|x: f32| x + 1) 5
"#
        )
        .is_err()
    );
}

#[test]
fn test_lambda_type_error_return_mismatch() {
    // Type error: lambda return type mismatch
    assert!(
        try_typecheck_program(
            r#"
def test: f32 = (|x| x + 1) 5
"#
        )
        .is_err()
    );
}

// ===== Loop Tests =====

#[test]
fn test_loop_while_simple() {
    // Simple while loop that counts
    typecheck_program(
        r#"
def test: i32 =
    loop acc = 0 while acc < 10 do
        acc + 1
"#,
    );
}

#[test]
fn test_loop_while_tuple() {
    // While loop with tuple pattern
    typecheck_program(
        r#"
def test: (i32, i32) =
    loop (idx, acc) = (0, 0) while idx < 10 do
        (idx + 1, acc + idx)
"#,
    );
}

#[test]
fn test_loop_for_simple() {
    // Simple for loop
    typecheck_program(
        r#"
def test: i32 =
    loop acc = 0 for i < 10 do
        acc + i
"#,
    );
}

#[test]
fn test_loop_for_with_array() {
    // For loop building result
    typecheck_program(
        r#"
def test: i32 =
    loop sum = 0 for i < 5 do
        sum + i * 2
"#,
    );
}

#[test]
fn test_loop_forin_simple() {
    // For-in loop over array
    typecheck_program(
        r#"
def test: i32 =
    let arr = [1, 2, 3, 4, 5] in
    loop sum = 0 for x in arr do
        sum + x
"#,
    );
}

#[test]
fn test_loop_forin_nested_array() {
    // For-in loop over nested array
    typecheck_program(
        r#"
def test: i32 =
    let edges : [3][2]i32 = [[0, 1], [1, 2], [2, 0]] in
    loop sum = 0 for e in edges do
        sum + e[0] + e[1]
"#,
    );
}

#[test]
fn test_loop_return_type_inference() {
    // Loop return type is inferred from init
    typecheck_program(
        r#"
def test: f32 =
    loop acc = 0.0f32 while acc < 10.0f32 do
        acc + 1.0f32
"#,
    );
}

#[test]
fn test_loop_type_error_body_mismatch() {
    // Type error: body type doesn't match init type
    assert!(
        try_typecheck_program(
            r#"
def test: i32 =
    loop acc = 0 while acc < 10 do
        3.14f32
"#
        )
        .is_err()
    );
}

#[test]
fn test_loop_type_error_condition_not_bool() {
    // Type error: while condition must be bool
    assert!(
        try_typecheck_program(
            r#"
def test: i32 =
    loop acc = 0 while 42 do
        acc + 1
"#
        )
        .is_err()
    );
}

#[test]
fn test_loop_nested() {
    // Nested loops
    typecheck_program(
        r#"
def test: i32 =
    loop outer = 0 for i < 3 do
        loop inner = outer for j < 4 do
            inner + i + j
"#,
    );
}

#[test]
fn test_qualified_builtin_f32_sqrt() {
    // Test that qualified builtins like f32.sqrt work correctly
    typecheck_program(
        r#"
def length2(v: vec2f32) -> f32 =
    f32.sqrt(v.x * v.x + v.y * v.y)
"#,
    );
}

#[test]
fn test_mul_mat_vec_application() {
    // Test that mul with mat and vec arguments type checks correctly
    // mul : mat<n,m,a> -> vec<m,a> -> vec<n,a>
    typecheck_program(
        r#"
def test(mat: mat4f32) -> vec4f32 =
    mul(mat, @[1.0f32, 2.0f32, 3.0f32, 4.0f32])
"#,
    );
}

#[test]
fn test_mul_mat_vec_in_lambda() {
    // Test mul mat vec inside a lambda with captured matrix
    typecheck_program(
        r#"
def test(mat: mat4f32, verts: [3]vec3f32) -> [3]vec4f32 =
    map((|v| mul(mat, @[v.x, v.y, v.z, 1.0f32])), verts)
"#,
    );
}

#[test]
fn test_higher_order_reduce() {
    // Test reduce function with higher-order operations using explicit parentheses
    typecheck_program(
        r#"
def reduce_f32(op: f32 -> f32 -> f32, init: f32, arr: [4]f32) -> f32 =
  let x0 = op(init)(arr[0]) in
  let x1 = op(x0)(arr[1]) in
  let x2 = op(x1)(arr[2]) in
  op(x2)(arr[3])

def test: f32 = reduce_f32((|x| |y| x + y), 0.0f32, [1.0f32, 2.0f32, 3.0f32, 4.0f32])
"#,
    );
}

#[test]
fn test_higher_order_reduce_no_parens() {
    // Test reduce without parentheses - application is left-associative
    typecheck_program(
        r#"
def reduce_f32(op: f32 -> f32 -> f32, init: f32, arr: [4]f32) -> f32 =
  let x0 = op(init)(arr[0]) in
  let x1 = op(x0)(arr[1]) in
  let x2 = op(x1)(arr[2]) in
  op(x2)(arr[3])

def test: f32 = reduce_f32((|x| |y| x + y), 0.0f32, [1.0f32, 2.0f32, 3.0f32, 4.0f32])
"#,
    );
}

#[test]
fn test_nested_map_with_reduce() {
    // Test the nested map pattern used in matrix multiplication
    typecheck_program(
        r#"
def reduce_f32(op: f32 -> f32 -> f32, init: f32, arr: [4]f32) -> f32 =
  let x0 = op(init)(arr[0]) in
  let x1 = op(x0)(arr[1]) in
  let x2 = op(x1)(arr[2]) in
  op(x2)(arr[3])

def map2_f32(f: f32 -> f32 -> f32, xs: [4]f32, ys: [4]f32) -> [4]f32 =
  [f(xs[0])(ys[0]), f(xs[1])(ys[1]), f(xs[2])(ys[2]), f(xs[3])(ys[3])]

def test: f32 =
  let row = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
  let col = [5.0f32, 6.0f32, 7.0f32, 8.0f32] in
  reduce_f32((|x| |y| x + y), 0.0f32, map2_f32((|x| |y| x * y), row, col))
"#,
    );
}

/// Test that size-polymorphic functions work correctly when called with concrete arrays.
/// This tests that calling `sum<[n]>(arr:[n]f32)` with a concrete `[3]f32` properly
/// instantiates the size parameter to 3.
#[test]
fn test_size_param_instantiation() {
    // Simple test: size-polymorphic function called with concrete array
    typecheck_program(
        r#"
def sum<[n]>(arr:[n]f32) -> f32 =
  let (result, _) = loop (acc, i) = (0.0f32, 0) while i < length(arr) do
    (acc + arr[i], i + 1)
  in result

def test: f32 = sum([1.0f32, 2.0f32, 3.0f32])
"#,
    );
}

/// Test that size parameters work through multiple levels of function calls.
#[test]
fn test_size_param_through_calls() {
    typecheck_program(
        r#"
def sum<[n]>(arr:[n]f32) -> f32 =
  let (result, _) = loop (acc, i) = (0.0f32, 0) while i < length(arr) do
    (acc + arr[i], i + 1)
  in result

def double_sum<[m]>(arr:[m]f32) -> f32 = sum(arr) + sum(arr)

def test: f32 = double_sum([1.0f32, 2.0f32])
"#,
    );
}

#[test]
fn test_unique_return_from_non_unique_param() {
    // Returning *[4]f32 from a function that takes [4]f32 should be an error
    // because we're claiming to return a unique/owned value but we only borrowed it
    let result = try_typecheck_program("def id(a: [4]f32) -> *[4]f32 = a");
    assert!(
        result.is_err(),
        "Should error: cannot return unique type from non-unique parameter"
    );
}

// =============================================================================
// Vector/Matrix Literal Tests (@[...] syntax)
// =============================================================================

#[test]
fn test_vector_literal_vec2() {
    typecheck_program("def test: vec2f32 = @[1.0f32, 2.0f32]");
}

#[test]
fn test_vector_literal_vec3() {
    typecheck_program("def test: vec3f32 = @[1.0f32, 2.0f32, 3.0f32]");
}

#[test]
fn test_vector_literal_vec4() {
    typecheck_program("def test: vec4f32 = @[1.0f32, 2.0f32, 3.0f32, 4.0f32]");
}

#[test]
fn test_vector_literal_vec3i32() {
    typecheck_program("def test: vec3i32 = @[1, 2, 3]");
}

#[test]
fn test_vector_literal_type_mismatch() {
    // vec3f32 requires floats, but we're providing integers
    let result = try_typecheck_program("def test: vec3f32 = @[1, 2, 3]");
    assert!(
        result.is_err(),
        "Should error: type mismatch between i32 and f32 elements"
    );
}

#[test]
fn test_vector_literal_wrong_size() {
    // vec3f32 requires 3 elements, but we're providing 2
    let result = try_typecheck_program("def test: vec3f32 = @[1.0f32, 2.0f32]");
    assert!(result.is_err(), "Should error: wrong number of elements");
}

#[test]
fn test_vector_literal_mixed_element_types() {
    // All elements must have the same type
    let result = try_typecheck_program("def test: vec3f32 = @[1.0f32, 2, 3.0f32]");
    assert!(result.is_err(), "Should error: mixed element types");
}

#[test]
fn test_vector_literal_in_expression() {
    typecheck_program(
        r#"
        def add_vectors(a: vec3f32, b: vec3f32) -> vec3f32 =
            @[(a.x + b.x), (a.y + b.y), (a.z + b.z)]

        def test: vec3f32 = add_vectors(@[1.0f32, 0.0f32, 0.0f32], @[0.0f32, 1.0f32, 0.0f32])
        "#,
    );
}

// Square matrices support both matNf32 and matNxNf32 syntax
#[test]
fn test_matrix_literal_mat2x2() {
    typecheck_program("def test: mat2x2f32 = @[[1.0f32, 2.0f32], [3.0f32, 4.0f32]]");
}

#[test]
fn test_matrix_literal_mat2x3() {
    typecheck_program("def test: mat2x3f32 = @[[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]");
}

#[test]
fn test_matrix_literal_mat3x3() {
    typecheck_program(
        r#"
        def identity: mat3x3f32 = @[
            [1.0f32, 0.0f32, 0.0f32],
            [0.0f32, 1.0f32, 0.0f32],
            [0.0f32, 0.0f32, 1.0f32]
        ]
        "#,
    );
}

#[test]
fn test_matrix_literal_inconsistent_row_lengths() {
    // All rows must have the same length
    let result =
        try_typecheck_program("def test: mat2x3f32 = @[[1.0f32, 2.0f32], [3.0f32, 4.0f32, 5.0f32]]");
    assert!(result.is_err(), "Should error: inconsistent row lengths");
}

#[test]
fn test_matrix_literal_wrong_element_type() {
    let result = try_typecheck_program("def test: mat2f32 = @[[1, 2], [3, 4]]");
    assert!(result.is_err(), "Should error: i32 elements for f32 matrix");
}

#[test]
fn test_matrix_literal_with_array_indexing() {
    // Matrix literal with array indexing inside
    typecheck_program(
        r#"
        def test: mat2f32 =
            let a = [1.0f32, 2.0f32] in
            @[[a[0], a[1]], [a[1], a[0]]]
    "#,
    );
}

#[test]
fn test_matrix_literal_with_expressions() {
    // Matrix literal with arithmetic expressions
    typecheck_program(
        r#"
        def test: mat2f32 =
            let x = 1.0f32 in
            @[[x, x+1.0f32], [x+2.0f32, x+3.0f32]]
    "#,
    );
}

#[test]
fn test_matrix_literal_with_type_annotation() {
    // Matrix literal with explicit type annotation
    typecheck_program(
        r#"
        def test: mat2f32 =
            let a = [1.0f32, 2.0f32] in
            (@[[a[0], a[1]], [a[1], a[0]]] : mat2f32)
    "#,
    );
}

#[test]
fn test_matrix_literal_direct_return() {
    // Matrix literal as direct return value without let binding
    typecheck_program(
        r#"
        def test: mat2f32 =
            let a = [1.0f32, 2.0f32] in
            @[[a[0], a[1]], [a[1], a[0]]]
    "#,
    );
}

#[test]
fn test_matrix_literal_direct_return_with_type_annotation() {
    // Matrix literal with type annotation as direct return
    typecheck_program(
        r#"
        def test: mat2f32 =
            let a = [1.0f32, 2.0f32] in
            (@[[a[0], a[1]], [a[1], a[0]]] : mat2f32)
    "#,
    );
}

#[test]
fn test_vector_literal_invalid_size() {
    // Vector size must be 2, 3, or 4
    let result = try_typecheck_program("def test: f32 = @[1.0f32]");
    assert!(result.is_err(), "Should error: vector size 1 is invalid");
}

#[test]
fn test_vector_literal_too_large() {
    // Vector size must be 2, 3, or 4
    let result = try_typecheck_program("def test: f32 = @[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32]");
    assert!(result.is_err(), "Should error: vector size 5 is invalid");
}

#[test]
fn test_f32_min() {
    typecheck_program("def test: f32 = f32.min(1.0f32, 2.0f32)");
}

#[test]
fn test_f32_min_partial() {
    // Partial application is no longer supported - this test verifies it fails
    let result = try_typecheck_program("def test: (f32 -> f32) = f32.min(1.0f32)");
    assert!(result.is_err(), "Partial application should fail");
}

#[test]
fn test_f32_min_in_expression() {
    // Reproduce the da_rasterizer case: f32.min (s*s) 1.0
    typecheck_program(
        r#"
def test: f32 =
  let s = 2.0f32 in
  f32.min(s*s, 1.0f32)
"#,
    );
}

// Tests for polymorphic builtins with different vector sizes
// Bug: polymorphic builtins get monomorphized on first use
// See BUGS.txt for details

#[test]
fn test_polymorphic_builtin_magnitude_different_sizes() {
    // This test verifies that polymorphic builtins like magnitude can be
    // used with different vector sizes in the same file.
    // Bug: Type gets fixed on first use due to Monotype instead of Polytype
    typecheck_program(
        r#"
def test1(v:vec3f32) -> f32 = magnitude(v)
def test2(v:vec2f32) -> f32 = magnitude(v)
def test3(v:vec4f32) -> f32 = magnitude(v)
        "#,
    );
}

#[test]
fn test_polymorphic_builtin_normalize_different_sizes() {
    typecheck_program(
        r#"
def test1(v:vec3f32) -> vec3f32 = normalize(v)
def test2(v:vec2f32) -> vec2f32 = normalize(v)
        "#,
    );
}

#[test]
fn test_polymorphic_builtin_dot_different_sizes() {
    typecheck_program(
        r#"
def test1(a:vec3f32, b:vec3f32) -> f32 = dot(a, b)
def test2(a:vec2f32, b:vec2f32) -> f32 = dot(a, b)
        "#,
    );
}

// Tests for new GLSL builtins (abs, sign, floor, ceil, fract, min, max, clamp, mix, smoothstep)

#[test]
fn test_builtin_abs_scalar_and_vector() {
    typecheck_program(
        r#"
def test1(x:f32) -> f32 = abs(x)
def test2(v:vec3f32) -> vec3f32 = abs(v)
def test3(v:vec2f32) -> vec2f32 = abs(v)
def test4(x:i32) -> i32 = abs(x)
def test5(v:vec3i32) -> vec3i32 = abs(v)
        "#,
    );
}

#[test]
fn test_builtin_sign_scalar_and_vector() {
    typecheck_program(
        r#"
def test1(x:f32) -> f32 = sign(x)
def test2(v:vec3f32) -> vec3f32 = sign(v)
def test3(x:i32) -> i32 = sign(x)
        "#,
    );
}

#[test]
fn test_builtin_floor_ceil_fract() {
    typecheck_program(
        r#"
def test1(x:f32) -> f32 = floor(x)
def test2(x:f32) -> f32 = ceil(x)
def test3(x:f32) -> f32 = fract(x)
def test4(v:vec3f32) -> vec3f32 = floor(v)
def test5(v:vec3f32) -> vec3f32 = ceil(v)
def test6(v:vec3f32) -> vec3f32 = fract(v)
        "#,
    );
}

#[test]
fn test_builtin_min_max_overloaded() {
    typecheck_program(
        r#"
def test1(a:f32, b:f32) -> f32 = min(a, b)
def test2(a:vec3f32, b:vec3f32) -> vec3f32 = min(a, b)
def test3(a:f32, b:f32) -> f32 = max(a, b)
def test4(a:i32, b:i32) -> i32 = min(a, b)
def test5(a:u32, b:u32) -> u32 = max(a, b)
def test6(a:vec2i32, b:vec2i32) -> vec2i32 = min(a, b)
        "#,
    );
}

#[test]
fn test_builtin_clamp_curried() {
    typecheck_program(
        r#"
def test1(x:f32) -> f32 = clamp(0.0, 1.0, x)
def test2(v:vec3f32) -> vec3f32 = clamp(0.0, 1.0, v)
def test3(x:i32) -> i32 = clamp(0i32, 100i32, x)
def test4(lo:u32, hi:u32, x:u32) -> u32 = clamp(lo, hi, x)
        "#,
    );
}

#[test]
fn test_builtin_mix_smoothstep() {
    // Note: vector mix with scalar interpolant requires a helper function
    // because GLSL FMix requires all operands to be the same type
    typecheck_program(
        r#"
def test1(a:f32, b:f32, t:f32) -> f32 = mix(a, b, t)
def test3(x:f32) -> f32 = smoothstep(0.0, 1.0, x)
def test4(v:vec3f32) -> vec3f32 = smoothstep(0.0, 1.0, v)
        "#,
    );
}

#[test]
fn test_u32_literal() {
    // Test that u32 suffix on literals produces u32 type
    typecheck_program("let x: u32 = 42u32");
}

// ============================================================
// ArrayWith (a with [i] = v) tests
// ============================================================

#[test]
fn test_array_with_basic() {
    // Basic array with syntax: update element at index
    typecheck_program(
        r#"
def test: [3]i32 =
    let arr = [1, 2, 3] in
    arr with [1] = 42
        "#,
    );
}

#[test]
fn test_array_with_variable_index() {
    // Array with using a variable as index
    typecheck_program(
        r#"
def test(i: i32) -> [4]f32 =
    let arr = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
    arr with [i] = 0.0f32
        "#,
    );
}

#[test]
fn test_array_with_preserves_size() {
    // Result type preserves array size
    typecheck_program(
        r#"
def test: [5]i32 =
    let arr : [5]i32 = [1, 2, 3, 4, 5] in
    arr with [0] = 100
        "#,
    );
}

#[test]
fn test_array_with_chained() {
    // Chained array with operations
    typecheck_program(
        r#"
def test: [3]i32 =
    let arr = [1, 2, 3] in
    arr with [0] = 10 with [1] = 20 with [2] = 30
        "#,
    );
}

#[test]
fn test_array_with_in_let() {
    // Array with in let binding
    typecheck_program(
        r#"
def test: i32 =
    let arr = [1, 2, 3] in
    let arr2 = arr with [1] = 42 in
    arr2[1]
        "#,
    );
}

#[test]
fn test_array_with_nested_array() {
    // Array with on nested arrays
    typecheck_program(
        r#"
def test: [2][3]i32 =
    let arr : [2][3]i32 = [[1, 2, 3], [4, 5, 6]] in
    arr with [0] = [10, 20, 30]
        "#,
    );
}

#[test]
fn test_array_with_wrong_value_type() {
    // Error: value type doesn't match element type
    let result = try_typecheck_program(
        r#"
def test: [3]i32 =
    let arr = [1, 2, 3] in
    arr with [1] = 3.14f32
        "#,
    );
    assert!(result.is_err(), "Should error: f32 value for i32 array");
}

#[test]
fn test_array_with_wrong_index_type() {
    // Error: index must be integer type
    let result = try_typecheck_program(
        r#"
def test: [3]i32 =
    let arr = [1, 2, 3] in
    arr with [1.5f32] = 42
        "#,
    );
    assert!(result.is_err(), "Should error: f32 index");
}

#[test]
fn test_array_with_non_array() {
    // Error: cannot use with on non-array type
    let result = try_typecheck_program(
        r#"
def test: i32 =
    let x = 42 in
    x with [0] = 10
        "#,
    );
    assert!(result.is_err(), "Should error: with on non-array");
}

#[test]
fn test_array_with_in_function() {
    // Array with as function parameter and return
    typecheck_program(
        r#"
def update_at(arr: [4]i32, i: i32, v: i32) -> [4]i32 =
    arr with [i] = v

def test: [4]i32 =
    update_at([1, 2, 3, 4], 2, 99)
        "#,
    );
}

#[test]
fn test_array_with_in_loop() {
    // Array with inside a loop
    typecheck_program(
        r#"
def test: [5]i32 =
    let init = [0, 0, 0, 0, 0] in
    loop arr = init for i < 5 do
        arr with [i] = i
        "#,
    );
}
