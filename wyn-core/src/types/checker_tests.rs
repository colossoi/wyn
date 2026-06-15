use super::{TypeChecker, TypeWarning};
use crate::error::CompilerError;
use crate::types::{Type, TypeExt, TypeName, TypeScheme};

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
    crate::compile_thru_frontend(input).map(|_| ())
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
    let out1 = replicate(len1, _w_intrinsic_uninit()) in

    let indices : [2]i32 = [0, 1] in
    let len2 = length(indices) in
    let out2 = replicate(len2, _w_intrinsic_uninit()) in

    42.0f32
        "#,
    );
}

#[test]
fn test_zip_arrays() {
    typecheck_program("def zip_arrays(xs, ys) = zip(xs, ys)");
}

#[test]
fn weakening_unique_return_into_non_unique_declared() {
    // step2 returns `*[4]i32`; main's declared return is `[4]i32`.
    // Uniqueness can be discarded at the return boundary, so this
    // should typecheck via one-directional `*T → T` weakening.
    typecheck_program(
        r#"
def step1(x: *[4]i32) *[4]i32 = x with [0] = 1
def step2(x: *[4]i32) *[4]i32 = x with [1] = 2

def main(arr: *[4]i32) [4]i32 = step2(step1(arr))
"#,
    );
}

#[test]
fn weakening_unique_into_non_unique_tuple_element() {
    // (arr, x) has type (*[4]i32, i32) but main's declared return is
    // ([4]i32, i32). The recursive strip in `unify_or_err_weakening`
    // pushes the * removal down into the tuple, so this typechecks.
    typecheck_program(
        r#"
def main(arr: *[4]i32) ([4]i32, i32) =
    let x = arr[0] in
    (arr, x)
"#,
    );
}

#[test]
fn test_mul_concrete_matrix_types() {
    // This test verifies that polymorphic mul works with concrete matrix types
    typecheck_program(
        r#"
def test_mul(mat1: mat4f32, mat2: mat4f32) mat4f32 =
    mul(mat1, mat2)
        "#,
    );
}

/// Helper function to check a program with a type hole and return the inferred type
fn check_type_hole(source: &str) -> Type {
    use crate::lexer;
    use crate::parser::Parser;
    use crate::resolve_placeholders::PlaceholderResolver;

    // Parse
    let (mut module_manager, mut node_counter) = crate::cached_module_manager();
    let tokens = lexer::tokenize(source).unwrap();
    let mut parser = Parser::new(tokens, &mut node_counter);
    let mut program = parser.parse().unwrap();

    // Resolve placeholders (required before type checking)
    let mut resolver = PlaceholderResolver::new();
    resolver.resolve(&mut module_manager, &mut program);
    let (context, spec_schemes) = resolver.into_parts();

    // Type check with the context from resolve_placeholders
    let mut checker = TypeChecker::with_context_and_schemes(&module_manager, context, spec_schemes);
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
            def apply_to_vec(f: vec3f32 -> f32) f32 =
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
def identity<[n]>(xs: [n]i32) [n]i32 = xs

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
fn test_lambda_curried_application_rejected() {
    // Curried lambda partially applied - should be rejected
    assert!(
        try_typecheck_program(
            r#"
def add = |x| |y| x + y
def add5 = add(5)
def result: i32 = add5(10)
"#,
        )
        .is_err()
    );
}

#[test]
fn test_lambda_multi_param() {
    // Multi-parameter lambda (non-curried)
    typecheck_program(
        r#"
def add = |x, y| x + y
def result: i32 = add(5, 10)
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
def apply(f: i32 -> i32, x: i32) i32 = f(x)
def result: i32 = apply((|x| x * 2), 5)
"#,
    );
}

#[test]
fn test_lambda_returned_from_function_rejected() {
    // Function returning a lambda - rejected because it creates curried pattern
    // Type i32 -> (i32 -> i32) requires 2 args under no-curry policy
    assert!(
        try_typecheck_program(
            r#"
def make_adder(n: i32) (i32 -> i32) = |x| x + n
def add10 = make_adder(10)
def result: i32 = add10(5)
"#,
        )
        .is_err()
    );
}

#[test]
fn test_higher_order_apply() {
    // Higher-order function that takes and applies a function (not returning one)
    typecheck_program(
        r#"
def apply_twice(f: i32 -> i32, x: i32) i32 = f(f(x))
def result: i32 = apply_twice((|x| x * 2), 5)
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
fn test_lambda_chained_application_rejected() {
    // Chained lambda applications - rejected under no-curry policy
    assert!(
        try_typecheck_program(
            r#"
def test: i32 =
    let f = |x| |y| |z| x + y + z in
    f(1)(2)(3)
"#,
        )
        .is_err()
    );
}

#[test]
fn test_lambda_multi_param_application() {
    // Multi-parameter lambda called with all args at once
    typecheck_program(
        r#"
def test: i32 =
    let f = |x, y, z| x + y + z in
    f(1, 2, 3)
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
fn test_lambda_compose_rejected() {
    // Function composition returning a lambda - rejected under no-curry policy
    // Type (i32 -> i32) -> (i32 -> i32) -> (i32 -> i32) has 3 arrows, requires 3 args
    assert!(
        try_typecheck_program(
            r#"
def compose(f: i32 -> i32, g: i32 -> i32) -> (i32 -> i32) =
    |x| f(g(x))
def double = |x| x * 2
def inc = |x| x + 1
def double_then_inc = compose(inc, double)
def result: i32 = double_then_inc(5)
"#,
        )
        .is_err()
    );
}

#[test]
fn test_lambda_compose_inline() {
    // Function composition applied inline (no function-returning-function)
    typecheck_program(
        r#"
def compose_apply(f: i32 -> i32, g: i32 -> i32, x: i32) i32 =
    f(g(x))
def double = |x| x * 2
def inc = |x| x + 1
def result: i32 = compose_apply(inc, double, 5)
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
def length2(v: vec2f32) f32 =
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
def test(mat: mat4f32) vec4f32 =
    mat * @[1.0f32, 2.0f32, 3.0f32, 4.0f32]
"#,
    );
}

#[test]
fn test_mul_mat_vec_in_lambda() {
    // Test mul mat vec inside a lambda with captured matrix
    typecheck_program(
        r#"
def test(mat: mat4f32, verts: [3]vec3f32) [3]vec4f32 =
    map((|v| mat * @[v.x, v.y, v.z, 1.0f32]), verts)
"#,
    );
}

#[test]
fn test_higher_order_reduce() {
    // Test reduce function with 2-arg binary operator
    // Type f32 -> f32 -> f32 has 2 arrows = requires 2 args
    // Calling op(init, arr[0]) provides 2 args, so arity check passes
    typecheck_program(
        r#"
def reduce_f32(op: f32 -> f32 -> f32, init: f32, arr: [4]f32) f32 =
  let x0 = op(init, arr[0]) in
  let x1 = op(x0, arr[1]) in
  let x2 = op(x1, arr[2]) in
  op(x2, arr[3])

def test: f32 = reduce_f32((|x, y| x + y), 0.0f32, [1.0f32, 2.0f32, 3.0f32, 4.0f32])
"#,
    );
}

#[test]
fn test_higher_order_reduce_curried_rejected() {
    // Curried reduce pattern should be rejected
    assert!(
        try_typecheck_program(
            r#"
def reduce_f32(op: f32 -> f32 -> f32, init: f32, arr: [4]f32) f32 =
  let x0 = op(init)(arr[0]) in
  let x1 = op(x0)(arr[1]) in
  let x2 = op(x1)(arr[2]) in
  op(x2)(arr[3])

def test: f32 = reduce_f32((|x| |y| x + y), 0.0f32, [1.0f32, 2.0f32, 3.0f32, 4.0f32])
"#,
        )
        .is_err()
    );
}

#[test]
fn test_nested_map_with_reduce() {
    // Test the nested map pattern used in matrix multiplication
    // Using curried types but calling with all args at once
    typecheck_program(
        r#"
def reduce_f32(op: f32 -> f32 -> f32, init: f32, arr: [4]f32) f32 =
  let x0 = op(init, arr[0]) in
  let x1 = op(x0, arr[1]) in
  let x2 = op(x1, arr[2]) in
  op(x2, arr[3])

def map2_f32(f: f32 -> f32 -> f32, xs: [4]f32, ys: [4]f32) [4]f32 =
  [f(xs[0], ys[0]), f(xs[1], ys[1]), f(xs[2], ys[2]), f(xs[3], ys[3])]

def test: f32 =
  let row = [1.0f32, 2.0f32, 3.0f32, 4.0f32] in
  let col = [5.0f32, 6.0f32, 7.0f32, 8.0f32] in
  reduce_f32((|x, y| x + y), 0.0f32, map2_f32((|x, y| x * y), row, col))
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
def sum<[n]>(arr:[n]f32) f32 =
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
def sum<[n]>(arr:[n]f32) f32 =
  let (result, _) = loop (acc, i) = (0.0f32, 0) while i < length(arr) do
    (acc + arr[i], i + 1)
  in result

def double_sum<[m]>(arr:[m]f32) f32 = sum(arr) + sum(arr)

def test: f32 = double_sum([1.0f32, 2.0f32])
"#,
    );
}

#[test]
fn test_unique_return_from_non_unique_param() {
    // Returning *[4]f32 from a function that takes [4]f32 should be an error
    // because we're claiming to return a unique/owned value but we only borrowed it
    let result = try_typecheck_program("def id(a: [4]f32) *[4]f32 = a");
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
        def add_vectors(a: vec3f32, b: vec3f32) vec3f32 =
            a + b

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
    // Wyn does not support partial application — this should type-error.
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
def test1(v:vec3f32) f32 = magnitude(v)
def test2(v:vec2f32) f32 = magnitude(v)
def test3(v:vec4f32) f32 = magnitude(v)
        "#,
    );
}

#[test]
fn test_polymorphic_builtin_normalize_different_sizes() {
    typecheck_program(
        r#"
def test1(v:vec3f32) vec3f32 = normalize(v)
def test2(v:vec2f32) vec2f32 = normalize(v)
        "#,
    );
}

#[test]
fn test_polymorphic_builtin_dot_different_sizes() {
    typecheck_program(
        r#"
def test1(a:vec3f32, b:vec3f32) f32 = dot(a, b)
def test2(a:vec2f32, b:vec2f32) f32 = dot(a, b)
        "#,
    );
}

#[test]
fn test_builtin_abs_scalar_and_vector() {
    typecheck_program(
        r#"
def test1(x:f32) f32 = abs(x)
def test2(v:vec3f32) vec3f32 = abs(v)
def test3(v:vec2f32) vec2f32 = abs(v)
def test4(x:i32) i32 = abs(x)
def test5(v:vec3i32) vec3i32 = abs(v)
        "#,
    );
}

#[test]
fn test_builtin_sign_scalar_and_vector() {
    typecheck_program(
        r#"
def test1(x:f32) f32 = sign(x)
def test2(v:vec3f32) vec3f32 = sign(v)
def test3(x:i32) i32 = sign(x)
        "#,
    );
}

#[test]
fn test_builtin_floor_ceil_fract() {
    typecheck_program(
        r#"
def test1(x:f32) f32 = floor(x)
def test2(x:f32) f32 = ceil(x)
def test3(x:f32) f32 = fract(x)
def test4(v:vec3f32) vec3f32 = floor(v)
def test5(v:vec3f32) vec3f32 = ceil(v)
def test6(v:vec3f32) vec3f32 = fract(v)
        "#,
    );
}

#[test]
fn test_builtin_min_max_overloaded() {
    typecheck_program(
        r#"
def test1(a:f32, b:f32) f32 = min(a, b)
def test2(a:vec3f32, b:vec3f32) vec3f32 = min(a, b)
def test3(a:f32, b:f32) f32 = max(a, b)
def test4(a:i32, b:i32) i32 = min(a, b)
def test5(a:u32, b:u32) u32 = max(a, b)
def test6(a:vec2i32, b:vec2i32) vec2i32 = min(a, b)
        "#,
    );
}

#[test]
fn test_builtin_clamp_curried() {
    typecheck_program(
        r#"
def test1(x:f32) f32 = clamp(0.0, 1.0, x)
def test2(v:vec3f32) vec3f32 = clamp(0.0, 1.0, v)
def test3(x:i32) i32 = clamp(0i32, 100i32, x)
def test4(lo:u32, hi:u32, x:u32) u32 = clamp(lo, hi, x)
        "#,
    );
}

#[test]
fn test_builtin_mix_smoothstep() {
    // Note: vector mix with scalar interpolant requires a helper function
    // because GLSL FMix requires all operands to be the same type
    typecheck_program(
        r#"
def test1(a:f32, b:f32, t:f32) f32 = mix(a, b, t)
def test3(x:f32) f32 = smoothstep(0.0, 1.0, x)
def test4(v:vec3f32) vec3f32 = smoothstep(0.0, 1.0, v)
        "#,
    );
}

#[test]
fn test_u32_literal() {
    // Test that u32 suffix on literals produces u32 type
    typecheck_program("let x: u32 = 42u32");
}

#[test]
fn test_reduce_with_u32() {
    // Test that reduce with u32 initial value and u32 array works correctly
    typecheck_program(
        r#"
def sum_u32(arr: [4]u32) u32 =
    reduce(|a, b| a + b, 0u32, arr)
        "#,
    );
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
def test(i: i32) [4]f32 =
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
def update_at(arr: [4]i32, i: i32, v: i32) [4]i32 =
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

// =========================================================================
// Type alias resolution tests
// =========================================================================

#[test]
fn test_type_alias_cycle_is_fatal() {
    // Two modules whose type aliases refer to each other → cycle.
    // Previously the checker silently logged this and returned the
    // unresolved type; now it must surface as a fatal type error.
    let result = try_typecheck_program(
        r#"
module a = {
  type t = b.t
}
module b = {
  type t = a.t
}

def use_a(x: a.t) i32 = 0
        "#,
    );
    let err = result.expect_err("type alias cycle should be a fatal error");
    let rendered = format!("{:?}", err);
    assert!(
        rendered.contains("type alias cycle detected"),
        "expected cycle error, got: {}",
        rendered
    );
}

// =========================================================================
// Module scoping
// =========================================================================

#[test]
fn open_reexports_members() {
    // `open base` splices `base`'s members into `derived`, so `derived.foo`
    // resolves from outside — re-export, the linchpin for a functor-produced
    // numeric hierarchy (the result re-exports the param module's members).
    typecheck_program(
        r#"
module base = {
  def foo(x: i32) i32 = x + 1
}
module derived = {
  open base
}
def use_it: i32 = derived.foo(10)
        "#,
    );
}

#[test]
fn qualified_operator_member_resolves() {
    // `m.(+)` references a module's operator member by qualified name, the
    // operator counterpart of `m.foo`. Parses (the `(op)` after `.` reuses the
    // operator-section parser) and resolves to the `def (+)` member.
    typecheck_program(
        r#"
module m = {
  def (+)(a: i32, b: i32) i32 = a
}
def use_it: i32 = m.(+)(1, 2)
        "#,
    );
}

// =========================================================================
// Module-system gaps (aspirational, #[ignore]d)
//
// Each test below asserts the *desired* behavior of a module-system feature
// that Wyn currently lacks; all are `#[ignore]`d so the suite stays green.
// Surfaced while building the lib/ statistics generators (rng/dist/disttest)
// and spiking a Futhark-style functor-produced numeric hierarchy. Drop the
// `#[ignore]` when the corresponding gap is closed.
// =========================================================================

/// Regression: a nested `module` body closes over the enclosing
/// file scope (ML/Futhark-style). Used to fail with
/// `UndefinedVariable("K")` because `check_module_functions` runs
/// before `check_program`'s user-decl loop, so file-scope `def K`
/// hadn't been inserted into the scope when module `m`'s `f` body
/// was checked. Now fixed by a forward-declaration pass at the top
/// of `check_program` that pre-binds file-scope `def`s with full
/// ascription (return type + all param types) — without disturbing
/// SOAC builtin resolution, since names that already collide with
/// something in scope are deliberately skipped (else a user
/// `def map(x) = …` would shadow the SOAC during prelude function
/// checking and break `unzip`'s `map(|...|, xys)`).
#[test]
fn nested_module_can_reference_outer_top_level_def() {
    typecheck_program(
        r#"
def K: u32 = 5u32
module m = {
  def f(x: u32) u32 = x + K
}
def use_k: u32 = m.f(1u32)
        "#,
    );
}

/// Regression: `open base` inside `module derived = { … }` brings
/// `base`'s members into local scope, so `derived`'s body can call
/// `foo` unqualified. Used to fail with `UndefinedVariable("foo")`
/// because the open-resolver's index was built from `spec_schemes` +
/// catalog only, and user-defined module `Decl` members never reach
/// `spec_schemes` — so the resolver didn't know `base` had any
/// members and the bare `foo` reference fell through. Module
/// elaboration already splices `open base`'s items into `derived`'s
/// items; the index now sees them too.
#[test]
fn open_brings_members_into_local_scope() {
    typecheck_program(
        r#"
module base = {
  def foo(x: i32) i32 = x + 1
}
module derived = {
  open base
  def bar(x: i32) i32 = foo(x) + 1
}
def use_it: i32 = derived.bar(10)
        "#,
    );
}

/// Regression: `f32.(+)` references f32's reified `+` function value
/// (operators can be forwarded through a functor the way named builtins
/// like `f32.max` already are). Used to fail with
/// `UndefinedVariable("f32.(+)")` because the catalog mis-stored
/// operator members under the binop spelling `f32.+`. `+` is the
/// binop; `(+)` is the function value, and member names are always
/// functions — so the catalog now stores `f32.(+)` and Spec::SigOp's
/// qualified key uses the same spelling.
#[test]
fn qualified_operator_member_resolves_builtin() {
    typecheck_program(
        r#"
def use_it: f32 = f32.(+)(1.0f32, 2.0f32)
        "#,
    );
}

/// A self-contained stand-in for a module with a concrete type alias
/// (`state = f32`) plus functions that use it internally. The qualified-type-
/// alias tests below prepend this so they exercise alias resolution without
/// depending on any prelude module.
const ALIAS_FIXTURE: &str = r#"
module rand = {
  type state = f32
  def init(seed: f32) state = seed
  def next(s: state) (state, f32) = (s, s)
}
"#;

/// Type check `body` with `ALIAS_FIXTURE` prepended, expecting success.
fn typecheck_with_alias_fixture(body: &str) {
    typecheck_program(&format!("{ALIAS_FIXTURE}{body}"));
}

#[test]
fn test_qualified_type_alias_resolves() {
    // rand.state is a type alias for f32 - qualified names should resolve
    typecheck_with_alias_fixture(
        r#"
def test: rand.state = 0.5f32
        "#,
    );
}

#[test]
fn test_qualified_type_alias_in_function_param() {
    // rand.state in parameter position should resolve to f32
    typecheck_with_alias_fixture(
        r#"
def use_state(s: rand.state) f32 = s + 1.0f32
def test: f32 = use_state(0.5f32)
        "#,
    );
}

#[test]
fn test_module_function_uses_internal_alias() {
    // Module functions like rand.init use 'state' internally - this should type check
    // because the module context provides alias resolution
    typecheck_with_alias_fixture(
        r#"
def test: rand.state = rand.init(0.123f32)
        "#,
    );
}

#[test]
fn test_module_function_returns_resolved_alias() {
    // rand.next returns (state, f32) which should resolve to (f32, f32)
    typecheck_with_alias_fixture(
        r#"
def test: (f32, f32) =
    let s = rand.init(0.5f32) in
    rand.next(s)
        "#,
    );
}

#[test]
fn test_lambda_param_with_qualified_alias() {
    // Lambda parameter annotations should resolve qualified type aliases
    // This tests that |x: rand.state| resolves rand.state -> f32
    typecheck_with_alias_fixture(
        r#"
def test: f32 =
    let f = |x: rand.state| x + 1.0f32 in
    f(0.5f32)
        "#,
    );
}

#[test]
fn test_entry_output_with_qualified_alias() {
    // Entry output types should resolve qualified type aliases
    // rand.state -> f32, so returning 1.0f32 should be valid
    typecheck_with_alias_fixture(
        r#"
#[fragment]
entry test(x: f32) rand.state =
    1.0f32
        "#,
    );
}

#[test]
fn test_type_ascription_with_qualified_alias() {
    // Type ascription should resolve qualified type aliases
    // rand.state -> f32
    typecheck_with_alias_fixture(
        r#"
def test(x: f32) f32 =
    let y = (1.0f32 : rand.state) in
    y + x
        "#,
    );
}

#[test]
fn test_let_annotation_with_qualified_alias() {
    // Let binding type annotations should resolve qualified type aliases
    // rand.state -> f32
    typecheck_with_alias_fixture(
        r#"
def test(x: f32) f32 =
    let y: rand.state = 1.0f32 in
    y + x
        "#,
    );
}

#[test]
fn test_vector_arithmetic() {
    // Vector addition/subtraction should work element-wise
    typecheck_program(
        r#"
def vec_add(a: vec3f32, b: vec3f32) vec3f32 = a + b
def vec_sub(a: vec3f32, b: vec3f32) vec3f32 = a - b
def vec_scale(a: vec3f32, s: f32) vec3f32 = a * s
        "#,
    );
}

#[test]
fn test_parameterized_module() {
    // Functors should allow generic module implementations
    typecheck_program(
        r#"
module type my_numeric = {
  type t
  sig add(a: t, b: t) t
}

module my_f32_num : (my_numeric with t = f32) = {
  def add(x: t, y: t) t = x + y
}

module my_f64_num : (my_numeric with t = f64) = {
  def add(x: t, y: t) t = x + y
}

functor add_stuff(n: my_numeric) = {
  type t = n.t

  def add3(x: t, y: t, z: t) t =
    n.add(n.add(x, y), z)
}

module add_f32 = add_stuff(my_f32_num)
module add_f64 = add_stuff(my_f64_num)

def add3_f32(a: f32, b: f32, c: f32) f32 = add_f32.add3(a, b, c)
def add3_f64(a: f64, b: f64, c: f64) f64 = add_f64.add3(a, b, c)
        "#,
    );
}

#[test]
fn test_stats32_erf() {
    // Test using the stats32 module from the fstats functor
    typecheck_program(
        r#"
def test_erf(x: f32) f32 = stats32.erf(x)
def test_erfc(x: f32) f32 = stats32.erfc(x)
def test_gamma(x: f32) f32 = stats32.gamma(x)
def test_lgamma(x: f32) f32 = stats32.lgamma(x)
        "#,
    );
}

#[test]
fn test_trig32() {
    // Test using the trig32 module from the ftrig functor
    typecheck_program(
        r#"
def test_sinpi(x: f32) f32 = trig32.sinpi(x)
def test_cospi(x: f32) f32 = trig32.cospi(x)
def test_tanpi(x: f32) f32 = trig32.tanpi(x)
def test_asinpi(x: f32) f32 = trig32.asinpi(x)
def test_acospi(x: f32) f32 = trig32.acospi(x)
def test_atanpi(x: f32) f32 = trig32.atanpi(x)
def test_atan2pi(x: f32, y: f32) f32 = trig32.atan2pi(x, y)
def test_hypot(x: f32, y: f32) f32 = trig32.hypot(x, y)
        "#,
    );
}

#[test]
fn test_builtin_soac_can_be_shadowed() {
    // Test that local definitions can shadow builtin SOACs
    typecheck_program(
        r#"
def test: i32 =
    let map = 42 in
    map
"#,
    );
}

#[test]
fn test_builtin_soac_works_when_not_shadowed() {
    typecheck_program(
        r#"
def test: [2]i32 =
    map(|x| x + 1, [1, 2])
"#,
    );
}

// --- Tuple positional access (.0, .1) ---

#[test]
fn test_tuple_field_access_first() {
    typecheck_program("def x(t: (i32, f32)) i32 = t.0");
}

#[test]
fn test_tuple_field_access_second() {
    typecheck_program("def x(t: (i32, f32)) f32 = t.1");
}

#[test]
fn test_tuple_field_access_out_of_bounds() {
    let result = try_typecheck_program("def x(t: (i32, f32)) i32 = t.2");
    assert!(
        result.is_err(),
        "Expected type error for out-of-bounds tuple index"
    );
}

#[test]
fn test_tuple_field_access_on_non_tuple() {
    let result = try_typecheck_program("def x(n: i32) i32 = n.0");
    assert!(result.is_err(), "Expected type error for .0 on non-tuple type");
}

// ---- Spec examples: structural sum-type and record-projection ambiguity --
//
// SPECIFICATION.md "Sum Types" and "Record Types" sections cite these as
// the working forms — a bare `#left(3)` or `r.x` is ambiguous in
// isolation, but with the type pinned by an annotation each type-checks.

#[test]
fn test_record_projection_with_annotation() {
    typecheck_program(
        r#"
def get_x: f32 =
    let r: { x: f32, y: f32 } = {x: 1.0f32, y: 2.0f32} in
    r.x
        "#,
    );
}

#[test]
fn test_sum_constructor_with_annotation() {
    typecheck_program("let x: #left(i32) | #right(f32) = #left(3)");
}

#[test]
fn test_sum_constructor_nullary() {
    typecheck_program("let x: #some(i32) | #none = #none");
}

#[test]
fn test_sum_constructor_multi_arg() {
    typecheck_program("let p: #point(i32, i32) | #origin = #point(3, 4)");
}

#[test]
fn test_sum_constructor_ambiguous_without_annotation() {
    let result = try_typecheck_program("let x = #left(3)");
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected type error for ambiguous constructor, got {:?}",
        result
    );
}

#[test]
fn test_sum_constructor_arity_mismatch() {
    let result = try_typecheck_program("let x: #left(i32) | #right(f32) = #left(3, 4)");
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected arity-mismatch type error, got {:?}",
        result
    );
}

#[test]
fn test_sum_constructor_unknown_variant() {
    let result = try_typecheck_program("let x: #left(i32) | #right(f32) = #middle(3)");
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected unknown-variant type error, got {:?}",
        result
    );
}

#[test]
fn test_sum_constructor_payload_type_mismatch() {
    let result = try_typecheck_program("let x: #left(i32) | #right(f32) = #left(1.5f32)");
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected payload-type-mismatch error, got {:?}",
        result
    );
}

// Match expressions can't run end-to-end yet (Phase C lowering hasn't
// landed), but the type checker should accept a well-typed match and
// reject obvious shape violations. The tests below stop after
// type-checking, before the AST→TLC step that would panic.

#[test]
fn test_match_well_typed() {
    typecheck_program(
        r#"
def pick(v: #left(i32) | #right(i32)) i32 =
    match v
    case #left(x) -> x
    case #right(y) -> y
        "#,
    );
}

#[test]
fn test_match_non_exhaustive() {
    let result = try_typecheck_program(
        r#"
def pick(v: #left(i32) | #right(i32)) i32 =
    match v
    case #left(x) -> x
        "#,
    );
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected non-exhaustive-match error, got {:?}",
        result
    );
}

#[test]
fn test_match_arm_type_mismatch() {
    let result = try_typecheck_program(
        r#"
def pick(v: #left(i32) | #right(f32)) i32 =
    match v
    case #left(x) -> x
    case #right(y) -> y
        "#,
    );
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected arm-result-mismatch error, got {:?}",
        result
    );
}

#[test]
fn test_match_scrutinee_not_a_sum() {
    let result = try_typecheck_program(
        r#"
def pick(v: i32) i32 =
    match v
    case #zero -> 0
        "#,
    );
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected scrutinee-not-a-sum error, got {:?}",
        result
    );
}

// Unit tests targeting bind_pattern's PatternKind::Constructor arm.
// Distinct from the constructor-expression tests above: these exercise
// the pattern side, where args are sub-patterns to be bound, not
// expressions to be checked.

#[test]
fn test_match_pattern_arity_mismatch() {
    // Pattern #left(x, y) against scrutinee variant #left(i32) — arity
    // mismatch must be caught by bind_pattern's args.len() check.
    let result = try_typecheck_program(
        r#"
def pick(v: #left(i32) | #right(i32)) i32 =
    match v
    case #left(x, y) -> x
    case #right(z) -> z
        "#,
    );
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected pattern arity-mismatch error, got {:?}",
        result
    );
}

#[test]
fn test_match_pattern_unknown_variant() {
    // Pattern #middle against a sum that has no #middle variant — must
    // be caught by bind_pattern's variants.iter().find() lookup.
    let result = try_typecheck_program(
        r#"
def pick(v: #left(i32) | #right(i32)) i32 =
    match v
    case #left(x) -> x
    case #middle(y) -> y
        "#,
    );
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected pattern unknown-variant error, got {:?}",
        result
    );
}

// =============================================================================
// `with` expression tests: swizzle, array, record
// =============================================================================
//
// Three families track the three LHS forms `with` accepts:
//
//   - swizzle:  `v with .yz = e`
//   - array:    `a with [i] = e`
//   - record:   `r with field = e` (single-level and nested `r with a.x = e`)

// --- Swizzle-with ---

#[test]
fn test_vec_with_swizzle_yz() {
    typecheck_program(
        r#"
def update(v: vec3f32, e: vec2f32) vec3f32 =
    v with .yz = e
        "#,
    );
}

#[test]
fn test_vec_with_swizzle_xz() {
    // Non-contiguous swizzle (.x and .z, skipping .y) must work.
    typecheck_program(
        r#"
def update(v: vec3f32, e: vec2f32) vec3f32 =
    v with .xz = e
        "#,
    );
}

#[test]
fn test_vec_with_swizzle_single_component() {
    // One-component LHS: RHS is scalar (elem type), not vec1.
    typecheck_program(
        r#"
def update(v: vec3f32, e: f32) vec3f32 =
    v with .x = e
        "#,
    );
}

#[test]
fn test_vec_with_swizzle_compound_mul() {
    // Compound `*=` desugars to `v with .yz = v.yz * m` and must
    // match against vec2 * mat2 → vec2.
    typecheck_program(
        r#"
def update(v: vec3f32, m: mat2f32) vec3f32 =
    v with .yz *= m
        "#,
    );
}

#[test]
fn test_vec_with_swizzle_arity_mismatch() {
    // `.yz` swizzle requires a vec2 RHS; passing vec3 is wrong.
    let result = try_typecheck_program(
        r#"
def update(v: vec3f32, e: vec3f32) vec3f32 =
    v with .yz = e
        "#,
    );
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected arity-mismatch error, got {:?}",
        result
    );
}

#[test]
fn test_vec_with_swizzle_duplicate_components() {
    // `.xx` writes the same slot twice — caller intent is ambiguous;
    // parser rejects at distinctness check.
    let result = try_typecheck_program(
        r#"
def update(v: vec3f32, e: vec2f32) vec3f32 =
    v with .xx = e
        "#,
    );
    assert!(
        result.is_err(),
        "expected error for duplicate components, got {:?}",
        result
    );
}

#[test]
fn test_vec_with_swizzle_out_of_range() {
    // `.w` is slot 3, but a vec3 only has slots 0-2.
    let result = try_typecheck_program(
        r#"
def update(v: vec3f32, e: f32) vec3f32 =
    v with .w = e
        "#,
    );
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected out-of-range error, got {:?}",
        result
    );
}

#[test]
fn test_vec_with_swizzle_non_vec_target() {
    // Target must be a vec; passing an i32 should error.
    let result = try_typecheck_program(
        r#"
def update(v: i32, e: f32) i32 =
    v with .x = e
        "#,
    );
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected non-vec target error, got {:?}",
        result
    );
}

#[test]
fn test_vec_with_swizzle_chained_rotations() {
    typecheck_program(
        r#"
def rot(a: f32) mat2f32 =
    let c = f32.cos(a) in
    let s = f32.sin(a) in
    @[[c, s], [0.0f32 - s, c]]

def transform(dir0: vec3f32, mx: f32, my: f32, rotview1: mat2f32, rotview2: mat2f32) vec3f32 =
    let dir1 = dir0 with .yz *= rot(my) in
    let dir2 = dir1 with .xz *= rot(mx) in
    let dir3 = dir2 with .yz *= rotview2 in
    dir3 with .xz *= rotview1
        "#,
    );
}

// (Array-with already has 11 tests starting at `test_array_with_basic`,
//  covering basic / variable-index / preserves-size / chained / in-let /
//  nested / wrong-value / wrong-index / non-array / in-function / in-loop.
//  No new array-with tests added here.)

// --- Record-with (spec'd in SPECIFICATION.md but never built) ---

#[test]
fn test_record_with_field() {
    typecheck_program(
        r#"
def update(r: { x: i32, y: i32 }) { x: i32, y: i32 } =
    r with x = 99
        "#,
    );
}

#[test]
fn test_record_with_nested_field() {
    typecheck_program(
        r#"
def update(r: { a: { x: i32 } }) { a: { x: i32 } } =
    r with a.x = 99
        "#,
    );
}

#[test]
fn test_record_with_unknown_field() {
    let result = try_typecheck_program(
        r#"
def update(r: { x: i32 }) { x: i32 } =
    r with z = 99
        "#,
    );
    assert!(result.is_err(), "expected unknown-field error, got {:?}", result);
}

#[test]
fn test_record_with_value_type_mismatch() {
    let result = try_typecheck_program(
        r#"
def update(r: { x: i32 }) { x: i32 } =
    r with x = 1.5f32
        "#,
    );
    assert!(result.is_err(), "expected value-type mismatch, got {:?}", result);
}

#[test]
fn test_match_pattern_binds_payload_type() {
    // The variable bound by a constructor pattern must take the
    // payload type, so an arm body that uses it as f32 should
    // type-check when the payload is f32, and fail when the payload
    // is i32. This exercises bind_pattern's recursive call into the
    // sub-pattern with payload_ty.
    typecheck_program(
        r#"
def pick(v: #left(f32) | #right(f32)) f32 =
    match v
    case #left(x) -> x + 1.0f32
    case #right(y) -> y
        "#,
    );
    let result = try_typecheck_program(
        r#"
def pick(v: #left(i32) | #right(i32)) f32 =
    match v
    case #left(x) -> x + 1.0f32
    case #right(y) -> 0.0f32
        "#,
    );
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected type-mismatch on bound payload, got {:?}",
        result
    );
}

#[test]
fn builtin_exp2() {
    let scalar = try_typecheck_program("open f32\ndef f(x: f32) f32 = exp2(x)");
    assert!(
        scalar.is_ok(),
        "exp2 should be a scalar f32 builtin: {:?}",
        scalar
    );
    let vec3 = try_typecheck_program("def f(v: vec3f32) vec3f32 = vec.exp2(v)");
    assert!(vec3.is_ok(), "vec.exp2 should accept vec3f32: {:?}", vec3);
}

#[test]
fn builtin_step() {
    let scalar = try_typecheck_program("def f(edge: f32, x: f32) f32 = step(edge, x)");
    assert!(
        scalar.is_ok(),
        "step should be a scalar f32 builtin: {:?}",
        scalar
    );
    let vec3 = try_typecheck_program("def f(edge: f32, v: vec3f32) vec3f32 = step(edge, v)");
    assert!(
        vec3.is_ok(),
        "step should accept (f32 edge, vec3f32 x): {:?}",
        vec3
    );
}

// --- Vertex-shader #[location(n)] input attributes ---------------------

#[test]
fn vertex_location_inputs_typecheck() {
    typecheck_program(
        "#[vertex]\n\
         entry vs(#[location(0)] position: vec3f32, #[location(1)] color: vec3f32)\n\
           (#[builtin(position)] vec4f32, #[location(0)] vec3f32) =\n\
           (@[position.x, position.y, position.z, 1.0], color)",
    );
}

#[test]
fn vertex_location_duplicate_rejected() {
    let result = try_typecheck_program(
        "#[vertex]\n\
         entry vs(#[location(0)] a: vec3f32, #[location(0)] b: vec3f32)\n\
           #[builtin(position)] vec4f32 = @[a.x, a.y, a.z, 1.0]",
    );
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

#[test]
fn vertex_location_non_format_type_rejected() {
    // [4]f32 is not a valid vertex-buffer attribute format.
    let result = try_typecheck_program(
        "#[vertex]\n\
         entry vs(#[location(0)] a: [4]f32)\n\
           #[builtin(position)] vec4f32 = @[a[0], a[1], a[2], 1.0]",
    );
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

#[test]
fn vertex_bare_param_rejected() {
    // A vertex param with neither #[location] nor #[builtin] is an error.
    let result = try_typecheck_program(
        "#[vertex]\n\
         entry vs(a: vec3f32) #[builtin(position)] vec4f32 = @[a.x, a.y, a.z, 1.0]",
    );
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

#[test]
fn fragment_location_varying_still_typechecks() {
    // Regression guard: #[location(n)] on a *fragment* param is a varying
    // input, not a vertex attribute — the vertex-format restriction must
    // not touch it.
    typecheck_program(
        "#[fragment]\n\
         entry fs(#[location(0)] color: vec3f32) #[location(0)] vec4f32 =\n\
           @[color.x, color.y, color.z, 1.0]",
    );
}

// =============================================================================
// Lifted-type restrictions (spec lines 458-465)
// =============================================================================
//
// Per the spec, certain shapes that contain functions or existential sizes
// must be rejected at type-check time:
//
//  * Lifted types (size-lifted or fully-lifted) cannot be put in arrays.
//  * Fully-lifted types additionally cannot be returned from `if`/`match`
//    branches or `loop` parameters.
//
// Each restriction is tested in two flavors:
//  - The "direct" form, where a function or existential type literally
//    appears in array position — caught by inspecting the type shape
//    directly.
//  - The "via an abbreviation" form, where the bad shape hides behind a
//    `type~`/`type^` declared abbreviation. This case unambiguously
//    requires the lifted-marker plumbing.

// ----- Lifted (existential / function) types in arrays -----

/// Direct: `[]( ?n. [n]i32 )` — array of existentially-sized arrays.
/// Different array instances would carry different existential sizes,
/// which can't share an outer array's uniform element shape.
#[test]
fn aspiration_existential_size_array_in_array_rejected_direct() {
    let result = try_typecheck_program("def junk: [](?n. [n]i32) = [[1, 2, 3], [4, 5]]");
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

/// Via `type~`: same shape but hidden behind a size-lifted abbreviation.
/// Even with the RHS hidden, the array-of-bags use should be rejected
/// because the abbreviation's lifting marker says so.
#[test]
fn aspiration_size_lifted_abbreviation_in_array_rejected() {
    let result = try_typecheck_program("type~ bag = ?n. [n]i32\ndef bags: []bag = []");
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

/// Direct: `[](i32 -> i32)` — array of functions.
#[test]
fn aspiration_function_in_array_rejected_direct() {
    let result = try_typecheck_program("def fns: [](i32 -> i32) = []");
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

/// Via `type^`: array of fully-lifted abbreviation.
#[test]
fn aspiration_fully_lifted_abbreviation_in_array_rejected() {
    let result = try_typecheck_program("type^ cmp = i32 -> i32 -> i32\ndef cmps: []cmp = []");
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

// ----- Fully-lifted types out of branches / loops (the `type^`-only rule) -----

/// A direct function returned from an `if`/`else` branch.
#[test]
fn aspiration_function_returned_from_if_rejected_direct() {
    let result = try_typecheck_program(
        "def choose(p: bool) (i32 -> i32) = if p then (|x: i32| x + 1) else (|x: i32| x - 1)",
    );
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

/// Same shape behind a `type^` abbreviation.
#[test]
fn aspiration_fully_lifted_abbreviation_returned_from_if_rejected() {
    let result = try_typecheck_program(
        "type^ cmp = i32 -> i32 -> i32\n\
         def asc: cmp = |x: i32, y: i32| x - y\n\
         def dsc: cmp = |x: i32, y: i32| y - x\n\
         def pick(p: bool) cmp = if p then asc else dsc",
    );
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

/// Fully-lifted as a `loop` parameter. The accumulator type flows through
/// every iteration; a function-typed accumulator violates the rule.
#[test]
fn aspiration_fully_lifted_loop_parameter_rejected() {
    let result = try_typecheck_program(
        "type^ cmp = i32 -> i32 -> i32\n\
         def chain(k: i32) cmp = loop f: cmp = (|x: i32, y: i32| x - y) for i < k do f",
    );
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

// ----- Plain `type` with a shape that requires a lifted marker -----
//
// Spec line 460: an unmarked `type` declaration whose RHS contains an
// existential size or a function must be rejected — the user is required
// to mark it `type~` or `type^` respectively. Each test below uses the
// abbreviation in a downstream `def` to ensure some part of the pipeline
// surfaces the mismatch.

#[test]
fn plain_type_with_existential_rhs_is_rejected() {
    let result = try_typecheck_program("type bag = ?n. [n]i32\ndef x: bag = [1, 2, 3]");
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

#[test]
fn plain_type_with_function_rhs_is_rejected() {
    let result =
        try_typecheck_program("type cmp = i32 -> i32 -> i32\ndef ascending: cmp = |x: i32, y: i32| x - y");
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

// =============================================================================
// Causality restriction (spec lines 1076-1098)
// =============================================================================
//
// A size parameter must be "used as the size of some parameter" — i.e. it
// must appear as a concrete array size on at least one of the function's
// value parameters. The spec gives the minimal counterexample
// `def f [n] (x: i32) = n` and says it should be rejected.

#[test]
#[ignore = "spec section \"Size Types > Causality Restriction\": a size parameter never used as a concrete array size should be a type error; today it is silently accepted"]
fn aspiration_causality_unused_size_param_rejected() {
    let result = try_typecheck_program("def f [n] (x: i32) i32 = n");
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

// =============================================================================
// Multi-dimensional chained slicing (spec line 720-722)
// =============================================================================
//
// Per the spec, multi-dim slicing is expressed by chaining single-dim
// slices: `a[i:j][k:l]` slices the outer dim then the inner. Single-dim
// `a[i:j]` works on rank-1 arrays today; applying a second slice to the
// sub-array returned by the outer slice is not yet supported.

#[test]
#[ignore = "spec section \"Expressions > Semantics of Simple Expressions > a[i:j:s]\": chained multi-dim slicing a[i:j][k:l] should type-check; today applying a slice to the sub-array returned by an outer slice fails"]
fn aspiration_chained_multidim_slice() {
    let source = "def f(a: [3][4]i32) [2][3]i32 = a[0:2][0:3]";
    let result = try_typecheck_program(source);
    assert!(result.is_ok());
}

#[test]
fn storage_slice_with_literal_bounds_stays_view() {
    let checked = crate::compile_thru_frontend(
        r#"
#[compute]
entry e(data: []i32) i32 = length(data[0..4096])
"#,
    )
    .expect("storage slice should typecheck");

    let entry = checked
        .ast
        .declarations
        .iter()
        .find_map(|decl| match decl {
            crate::ast::Declaration::Entry(entry) if entry.name == "e" => Some(entry),
            _ => None,
        })
        .expect("entry e should exist");

    let slice_expr = match &entry.body.kind {
        crate::ast::ExprKind::Application(_, args) => args
            .iter()
            .find(|arg| matches!(arg.kind, crate::ast::ExprKind::Slice(_)))
            .expect("length argument should be the slice expression"),
        other => panic!("expected length application, got {other:?}"),
    };

    let ty = match checked
        .type_table
        .get(&slice_expr.h.id)
        .expect("slice expression should have an inferred type")
    {
        TypeScheme::Monotype(ty) => ty,
        TypeScheme::Polytype { .. } => panic!("slice expression should be monomorphic"),
    };

    assert!(
        crate::types::is_array_variant_view(ty.array_variant().expect("slice type has variant")),
        "literal-bounded storage slice should stay View, got {ty:?}"
    );
    assert!(
        matches!(
            ty.array_size().expect("slice type has size"),
            Type::Constructed(TypeName::Size(4096), _)
        ),
        "literal bounds should still record the static slice length, got {ty:?}"
    );
}

// =============================================================================
// Empty array literals (spec lines 1100-1106)
// =============================================================================
//
// Constructing an empty array via `[] : t` requires the element type's
// shape to be statically known at the point of construction. The spec
// example pairs an empty `[][]i32` with a later `[filter (>0) xs]`,
// whose inner array size is unknown at the point `a` is constructed —
// that should be a type error.

#[test]
#[ignore = "spec section \"Size Types > Empty Array Literals\": empty array whose element shape isn't known at the construction point should be a type error; today this program is silently accepted"]
fn aspiration_empty_array_with_unknown_inner_size_rejected() {
    let source = "\
def main (b: bool) (xs: []i32) bool =\n\
  let a = [] : [][]i32 in\n\
  let bs = [filter (>0) xs] in\n\
  a[0] == bs[0]\n";
    let result = try_typecheck_program(source);
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

// =============================================================================
// Size coercion `:>` (spec lines 1062-1069)
// =============================================================================
//
// `:>` is documented as a runtime-checked coercion. Today the checker
// accepts the expression and the backend emits no length assertion, so
// a statically-impossible coercion compiles silently with no diagnostic
// or runtime trap.

#[test]
#[ignore = "spec section \"Size Types > Size Coercion\": a :> coercion with statically-known mismatched sizes (here [3]i32 -> [5]i32) should be rejected at compile time or emit a runtime check; today the program compiles with no diagnostic and no runtime assertion"]
fn aspiration_size_coercion_statically_mismatched_rejected() {
    let result = try_typecheck_program("def f() [5]i32 = [1, 2, 3] :> [5]i32");
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

// =============================================================================
// Size variable sharing (spec section "Size Types", lines 956-1156)
// =============================================================================
//
// A function `def f [n] (a: [n]i32) (b: [n]i32) = ...` declares that the
// two arguments share size `n`. The checker should reject a call site
// passing arrays of statically-known different sizes. Today size
// variables are mostly treated as generic type variables, so the
// constraint isn't enforced — calls with statically different lengths
// compile.

#[test]
#[ignore = "spec section \"Size Types\": a function with a shared size parameter `[n]` should reject call sites with statically different array sizes; today the size variable is treated generically and the call compiles"]
fn aspiration_size_param_shared_across_args_constraint_enforced() {
    let source = "\
def eq [n] (a: [n]i32) (b: [n]i32) i32 = a[0] + b[0]\n\
def main () i32 = eq [1, 2, 3] [4, 5]\n";
    let result = try_typecheck_program(source);
    assert!(matches!(result, Err(CompilerError::TypeError(_, _))));
}

/// Spec §x binop y: `**` is the one binary op that admits a
/// heterogeneous shape — a float base may take an integer exponent
/// (signed or unsigned, any width); every other op still requires same
/// types, and the exception is one-directional (integer base + float
/// exponent stays a same-typed error).
#[test]
fn pow_operator_heterogeneous_cases() {
    // 1. Permitted: `f32 ** i32` (default int literal is i32).
    typecheck_program("def pow5(x: f32) f32 = x ** 5");

    // 2. Permitted: `f32 ** u32` (explicit unsigned exponent).
    typecheck_program("def pow5u(x: f32) f32 = x ** 5u32");

    // 3. Regression guard: existing homogeneous shape keeps working.
    typecheck_program("def pow5f(x: f32) f32 = x ** 5.0f32");

    // 4. Rejected: integer base, float exponent — the exception only
    //    runs in the float-base direction.
    let int_base_float_exp = try_typecheck_program("def bad(x: i32) i32 = x ** 5.0f32");
    assert!(matches!(int_base_float_exp, Err(CompilerError::TypeError(_, _))));

    // 5. Rejected: other binary operators stay strictly same-typed. The
    //    exception is `**`-only.
    let plus_het = try_typecheck_program("def bad(x: f32) f32 = x + 5");
    assert!(matches!(plus_het, Err(CompilerError::TypeError(_, _))));
}

// ---- Constructor-style type conversion `T(value)` ----

/// `i32(x)` where `x: f32` dispatches via the existing
/// `i32.f32` catalog entry — no new intrinsic needed.
#[test]
fn ctor_scalar_f32_to_i32_typechecks() {
    typecheck_program("def to_i(x: f32) i32 = i32(x)");
}

#[test]
fn ctor_scalar_u32_to_i32_typechecks() {
    typecheck_program("def to_i(x: u32) i32 = i32(x)");
}

#[test]
fn ctor_scalar_i32_to_f32_typechecks() {
    typecheck_program("def to_f(x: i32) f32 = f32(x)");
}

#[test]
fn ctor_scalar_f32_to_u32_typechecks() {
    typecheck_program("def to_u(x: f32) u32 = u32(x)");
}

/// Unknown name falls through to the standard undefined-variable
/// error — the constructor hook must not swallow non-type names.
#[test]
fn ctor_unknown_name_returns_undefined() {
    let result = try_typecheck_program("def bad(x: f32) f32 = xyzzy(x)");
    assert!(
        matches!(result, Err(CompilerError::UndefinedVariable(_, _))),
        "expected UndefinedVariable, got {result:?}"
    );
}

/// Vec constructors desugar to N componentwise scalar conversions.
/// vec2 / vec3 / vec4 over the realistic source/target pairs.
#[test]
fn ctor_vec2_f32_to_i32_typechecks() {
    typecheck_program("def to_v2i(v: vec2f32) vec2i32 = vec2i32(v)");
}

#[test]
fn ctor_vec3_i32_to_f32_typechecks() {
    typecheck_program("def to_v3f(v: vec3i32) vec3f32 = vec3f32(v)");
}

#[test]
fn ctor_vec4_u32_to_f32_typechecks() {
    typecheck_program("def to_v4f(v: vec4u32) vec4f32 = vec4f32(v)");
}

/// Vec constructor over a scalar arg is a type error — the
/// hook synthesises an arrow `vec<a,N> -> vec<T,N>` and unification
/// against an `i32` arg must fail.
#[test]
fn ctor_vec_rejects_scalar_arg() {
    let result = try_typecheck_program("def bad(x: i32) vec2i32 = vec2i32(x)");
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected TypeError, got {result:?}"
    );
}

/// Vec3 constructor over a vec2 arg fails on arity.
#[test]
fn ctor_vec_rejects_arity_mismatch() {
    let result = try_typecheck_program("def bad(v: vec2f32) vec3i32 = vec3i32(v)");
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected TypeError, got {result:?}"
    );
}

/// The legacy dot-form `i32.f32(x)` must keep working — the
/// constructor hook is additive, the existing `lookup_module_scheme`
/// path is untouched.
#[test]
fn ctor_legacy_dot_syntax_still_works() {
    typecheck_program("def to_i(x: f32) i32 = i32.f32(x)");
}

// ---- Bare `[]T` size sharing in `def` scope ----
//
// Bare `[]T` inside a single `def` resolves to one shared size
// variable, so `def f(a: []i32, b: []i32)` is sugar for
// `def f<[n]>(a: [n]i32, b: [n]i32)`.

#[test]
fn def_bare_array_sizes_share_within_decl() {
    // Both args resolve to the same size variable; matching literal
    // sizes typecheck cleanly.
    typecheck_program("def add_first(a: []i32, b: []i32) i32 = a[0] + b[0]");
}

#[test]
fn def_bare_array_size_mismatch_rejected() {
    // Same `def` as above, but the call site passes literal arrays of
    // different sizes. Sharing the size variable across `a` and `b`
    // means the two sizes must unify; they don't, so it's a TypeError.
    let result = try_typecheck_program(
        r#"
def add_first(a: []i32, b: []i32) i32 = a[0] + b[0]
def test: i32 = add_first([1, 2, 3], [4, 5])
"#,
    );
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected TypeError for size-mismatched call to def with shared bare-[]: {result:?}"
    );
}

// ---- Representation-polymorphic existential array (filter output) ----
//
// `filter`'s return scheme is `?k. Array[a, Abstract, k, no_region]`.
// After `open_existential` (let-binding), the bound value has type
// `Array[a, Abstract, Skolem(k), no_region]`. Operations on the
// Abstract-variant array must typecheck cleanly (length, index, slice,
// passing to a size-poly helper). The previous Composite pinning froze
// the consumer signature against Composite before the producer's
// representation existed — see issues/slice-view-provenance.md and
// `egir::verify_no_abstract`.

#[test]
fn filter_result_typechecks_in_let_binding() {
    // Smallest possible case: bind a filter result.
    typecheck_program(
        r#"
def is_even(x: i32) bool = x % 2 == 0
def evens(arr: [8]i32) ?k. [k]i32 = filter(is_even, arr)
"#,
    );
}

#[test]
fn filter_result_length_and_index_typecheck() {
    // Length and index on the opened (Abstract-variant) array. The
    // type rule must not force a concrete variant here.
    typecheck_program(
        r#"
def is_even(x: i32) bool = x % 2 == 0
#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
  let e = filter(is_even, [1, 2, 3, 4, 5, 6, 7, 8]) in
  @[f32.i32(length(e)), f32.i32(e[0]), 0.0, 1.0]
"#,
    );
}

#[test]
fn filter_into_reduce_typechecks() {
    // Canonical "filter then size-polymorphic consumer" pattern. With
    // filter's old `Composite` pinning the consumer (reduce) was
    // specialized against Composite — but at EGIR time the producer
    // chose Bounded/View, mismatching. With Abstract, reduce's
    // signature stays variant-polymorphic and the producer-side
    // resolution is what fixes the runtime representation.
    typecheck_program(
        r#"
#[compute]
entry tick(#[storage(set=2, binding=0, access=read)] xs: []f32) f32 =
  let ys = filter(|x: f32| x > 0.0, xs) in
  reduce(|a: f32, b: f32| a + b, 0.0, ys)
"#,
    );
}

// ---- Infix bitwise / shift operators ----
//
// `^ & | << >>` typecheck over integer operands as `t -> t -> t`, and are
// rejected on non-integer operands. SPIR-V emission is covered separately.

#[test]
fn bitwise_ops_typecheck_on_integers() {
    typecheck_program(
        r#"
def mix(a: u32, b: u32) u32 = (a ^ b) & (a | b)
def shifts(x: i32, n: i32) i32 = (x << n) >> n
"#,
    );
}

#[test]
fn bitwise_op_rejects_float_operands() {
    let result = try_typecheck_program(
        r#"
def bad(x: f32, y: f32) f32 = x ^ y
"#,
    );
    assert!(
        matches!(result, Err(CompilerError::TypeError(_, _))),
        "expected TypeError for bitwise '^' on f32 operands: {result:?}"
    );
}

// ---- Top-level integer constants keep their annotated/suffixed type ----
//
// Regression: a top-level `def C: u32 = <lit>` (or any non-i32 integer
// constant) dropped its type at use sites and defaulted to i32, so using it
// where a u32 was required failed with "requires same-typed operands, got
// u32 and i32". Float constants were unaffected.

#[test]
fn top_level_u32_constant_keeps_its_type() {
    typecheck_program(
        r#"
def SEED: u32 = 7u32
def use_it(x: u32) u32 = x + SEED
"#,
    );
}

#[test]
fn top_level_i64_constant_keeps_its_type() {
    typecheck_program(
        r#"
def BIG: i64 = 7i64
def use_it(x: i64) i64 = x + BIG
"#,
    );
}

// ---- Functor member-function abstract-type substitution ----
//
// Regression: a functor over a module-type with more than one member, whose
// body calls a member that *consumes* the abstract type (`at(k: key, ...)`),
// fails to substitute the abstract `key` for that member at instantiation —
// the argument resolves to the concrete type while the parameter stays
// abstract ("Function argument type mismatch: expected key, got u32"). A
// single-member signature, or a functor that never calls the key-consuming
// member, both type-check fine — so the trigger is the *other* member
// shifting which signature the abstract type is substituted into.
#[test]
fn functor_substitutes_abstract_type_in_member_call() {
    typecheck_program(
        r#"
module type IFC = {
  type key
  sig other(s: u32) u32
  sig at(k: key, p: u32) u32
}
module a : IFC = {
  type key = u32
  def other(s: u32) u32 = s
  def at(k: key, p: u32) u32 = k + p
}
functor F(G: IFC) = {
  def call(k: G.key, p: u32) u32 = G.at(k, p)
}
module fa = F(a)
def use_it(x: u32) u32 = fa.call(x, 1u32)
"#,
    );
}
