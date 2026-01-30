#![cfg(test)]
//! Integration tests for the full compilation pipeline.
//!
//! These tests verify that source code compiles correctly through all stages:
//! parse → desugar → resolve → type_check → alias_check → TLC → monomorphize → MIR
//!
//! All tests include entry points to ensure monomorphization can find reachable code.

use crate::mir;

/// Run source through the pipeline up to MIR (after hoisting).
fn compile_to_mir(input: &str) -> mir::Program {
    let mut frontend = crate::cached_frontend();
    let parsed = crate::Compiler::parse(input, &mut frontend.node_counter).expect("Parsing failed");
    let alias_checked = parsed
        .desugar(&mut frontend.node_counter)
        .expect("Desugaring failed")
        .resolve(&mut frontend.module_manager)
        .expect("Name resolution failed")
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .expect("Type checking failed")
        .alias_check()
        .expect("Borrow checking failed");

    let known_defs = crate::build_known_defs(&alias_checked.ast, &mut frontend.module_manager);
    alias_checked
        .to_tlc(known_defs, &frontend.schemes, &mut frontend.module_manager)
        .skip_partial_eval()
        .defunctionalize()
        .monomorphize()
        .to_mir()
        .hoist_materializations()
        .mir
}

/// Compile to GLSL (Shadertoy target) through the full pipeline.
fn compile_to_glsl(input: &str) -> String {
    let mut frontend = crate::cached_frontend();
    let parsed = crate::Compiler::parse(input, &mut frontend.node_counter).expect("Parsing failed");
    let alias_checked = parsed
        .desugar(&mut frontend.node_counter)
        .expect("Desugaring failed")
        .resolve(&mut frontend.module_manager)
        .expect("Name resolution failed")
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .expect("Type checking failed")
        .alias_check()
        .expect("Alias checking failed");
    let builtins = crate::build_known_defs(&alias_checked.ast, &mut frontend.module_manager);
    alias_checked
        .to_tlc(builtins, &frontend.schemes, &mut frontend.module_manager)
        .partial_eval()
        .defunctionalize()
        .monomorphize()
        .to_mir()
        .hoist_materializations()
        .normalize()
        .default_address_spaces()
        .parallelize_soacs()
        .filter_reachable()
        .lift_bindings()
        .lower_shadertoy()
        .expect("GLSL lowering failed")
}

/// Helper to check that code fails type checking (for testing error cases).
fn should_fail_type_check(input: &str) -> bool {
    let mut frontend = crate::cached_frontend();
    let result = crate::Compiler::parse(input, &mut frontend.node_counter)
        .and_then(|parsed| parsed.desugar(&mut frontend.node_counter))
        .and_then(|desugared| desugared.resolve(&mut frontend.module_manager))
        .map(|resolved| resolved.fold_ast_constants())
        .and_then(|folded| folded.type_check(&mut frontend.module_manager, &mut frontend.schemes));
    result.is_err()
}

/// Find a definition by name.
fn find_def<'a>(mir: &'a mir::Program, name: &str) -> &'a mir::Def {
    mir.defs
        .iter()
        .find(|d| match d {
            mir::Def::Function { name: n, .. } => n == name,
            mir::Def::Constant { name: n, .. } => n == name,
            _ => false,
        })
        .unwrap_or_else(|| panic!("Definition '{}' not found", name))
}

/// Get the body from a definition.
fn get_body<'a>(def: &'a mir::Def) -> &'a mir::Body {
    match def {
        mir::Def::Function { body, .. } => body,
        mir::Def::Constant { body, .. } => body,
        other => panic!("Expected Function or Constant, got {:?}", other),
    }
}

/// Extract just one function's MIR from the full program output.
fn extract_function_mir(mir_str: &str, fn_name: &str) -> String {
    let prefix = format!("def {} ", fn_name);
    let mut result = String::new();
    let mut capturing = false;

    for line in mir_str.lines() {
        if line.starts_with(&prefix) {
            capturing = true;
        } else if capturing && line.starts_with("def ") {
            break;
        }
        if capturing {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

// =============================================================================
// Basic Expressions
// =============================================================================

#[test]
fn test_basic_expressions() {
    // Tests: functions, let bindings, if expressions, binary/unary ops
    // Note: simple constants like `def x = 42` get inlined, so we test functions instead
    let mir = compile_to_mir(
        r#"
def add(x: i32, y: i32) i32 = x + y

def with_let(a: i32, b: i32) i32 =
    let x = a in
    let y = b in
    x + y

def with_if(x: bool) i32 = if x then 1 else 0

def with_ops(x: i32, y: i32) i32 = x * y + x / y - (-x)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let b = add(1, 2) in
    let c = with_let(3, 4) in
    let d = with_if(true) in
    let e = with_ops(5, 6) in
    @[f32.i32(b + c + d + e), 0.0, 0.0, 1.0]
"#,
    );

    // Verify key definitions exist
    assert!(mir.defs.iter().any(|d| matches!(d, mir::Def::Function { name, .. } if name == "add")));
    assert!(mir.defs.iter().any(|d| matches!(d, mir::Def::Function { name, .. } if name == "with_let")));
    assert!(mir.defs.iter().any(|d| matches!(d, mir::Def::Function { name, .. } if name == "with_if")));
    assert!(mir.defs.iter().any(|d| matches!(d, mir::Def::Function { name, .. } if name == "with_ops")));
}

// =============================================================================
// Data Structures
// =============================================================================

#[test]
fn test_data_structures() {
    // Tests: arrays, tuples, records, tuple patterns
    let mir = compile_to_mir(
        r#"
def arr = [1, 2, 3]

def record = {x: 1, y: 2}

def tuple_destruct: i32 =
    let (a, b) = (1, 2) in a + b

def nested_tuple: i32 =
    let ((a, b), c) = ((1, 2), 3) in a + b + c

def array_index(arr: [4]i32, i: i32) i32 = arr[i]

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = arr[0] in
    let b = record.x in
    let c = tuple_destruct in
    let d = nested_tuple in
    let e = array_index([1, 2, 3, 4], 0) in
    @[f32.i32(a + b + c + d + e), 0.0, 0.0, 1.0]
"#,
    );

    // Verify array literal exists
    let arr_def = find_def(&mir, "arr");
    let body = get_body(arr_def);
    assert!(
        matches!(body.get_expr(body.root), mir::Expr::Array { .. }),
        "Expected Array expression for arr"
    );
}

// =============================================================================
// Loops
// =============================================================================

#[test]
fn test_loops() {
    // Tests: while loops, for-range loops, for-in loops
    let mir = compile_to_mir(
        r#"
def while_loop: i32 =
    loop x = 0 while x < 10 do x + 1

def for_range_loop: i32 =
    loop acc = 0 for i < 10 do acc + i

def for_in_loop(arr: [5]i32) i32 =
    loop acc = 0 for x in arr do acc + x

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = while_loop in
    let b = for_range_loop in
    let c = for_in_loop([1, 2, 3, 4, 5]) in
    @[f32.i32(a + b + c), 0.0, 0.0, 1.0]
"#,
    );

    // Verify loop expressions exist
    let while_def = find_def(&mir, "while_loop");
    let body = get_body(while_def);
    assert!(
        matches!(
            body.get_expr(body.root),
            mir::Expr::Loop {
                kind: mir::LoopKind::While { .. },
                ..
            }
        ),
        "Expected While loop"
    );

    let for_def = find_def(&mir, "for_range_loop");
    let body = get_body(for_def);
    assert!(
        matches!(
            body.get_expr(body.root),
            mir::Expr::Loop {
                kind: mir::LoopKind::ForRange { .. },
                ..
            }
        ),
        "Expected ForRange loop"
    );
}

// =============================================================================
// Lambdas and Closures
// =============================================================================

#[test]
fn test_lambdas_and_closures() {
    // Tests: lambdas with captures, nested lambdas, direct calls, tuple params
    let mir = compile_to_mir(
        r#"
def with_capture(y: i32) i32 =
    let f = |x: i32| x + y in
    f(10)

def nested_lambda(x: i32) i32 =
    let outer = |a: i32|
        let inner = |b: i32| a + b + x in
        inner(a)
    in
    outer(5)

def tuple_param_lambda: i32 =
    let add = |(x, y): (i32, i32)| x + y in
    add((1, 2))

def direct_call: i32 =
    let inc = |x: i32| x + 1 in
    inc(5)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let b = with_capture(10) in
    let c = nested_lambda(100) in
    let d = tuple_param_lambda in
    let e = direct_call in
    @[f32.i32(b + c + d + e), 0.0, 0.0, 1.0]
"#,
    );

    // Should have lifted lambda functions
    let lambda_count = mir
        .defs
        .iter()
        .filter(|d| matches!(d, mir::Def::Function { name, .. } if name.contains("_w_lambda_")))
        .count();
    assert!(lambda_count >= 1, "Expected at least one lifted lambda");
}

// =============================================================================
// Higher-Order Functions (map, reduce, filter)
// =============================================================================

#[test]
fn test_higher_order_functions() {
    // Tests: map, reduce, filter with lambdas and named functions
    let mir = compile_to_mir(
        r#"
def double(x: i32) i32 = x * 2

def map_named(arr: [4]i32) [4]i32 = map(double, arr)

def map_lambda(arr: [4]i32) [4]i32 = map(|x: i32| x + 1, arr)

def map_with_capture(arr: [4]i32, offset: i32) [4]i32 =
    map(|x: i32| x + offset, arr)

def reduce_sum(arr: [4]f32) f32 =
    reduce(|acc: f32, x: f32| acc + x, 0.0, arr)

def reduce_tuple(hits: [4](f32, i32)) (f32, i32) =
    reduce(|(t1, m1): (f32, i32), (t2, m2): (f32, i32)|
             if t1 < t2 then (t1, m1) else (t2, m2),
           (1000.0, 0),
           hits)

def is_positive(x: i32) bool = x > 0

def filter_positive(arr: [5]i32) ?k. [k]i32 =
    filter(is_positive, arr)

def filter_lambda(arr: [4]i32) ?k. [k]i32 =
    filter(|x: i32| x % 2 == 0, arr)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = map_named([1, 2, 3, 4]) in
    let b = map_lambda([1, 2, 3, 4]) in
    let c = map_with_capture([1, 2, 3, 4], 10) in
    let d = reduce_sum([1.0, 2.0, 3.0, 4.0]) in
    let (t, _) = reduce_tuple([(1.0, 0), (2.0, 1), (0.5, 2), (3.0, 3)]) in
    let e = filter_positive([1, -2, 3, -4, 5]) in
    let f = filter_lambda([1, 2, 3, 4]) in
    @[d + t, f32.i32(a[0] + b[0] + c[0] + length(e) + length(f)), 0.0, 1.0]
"#,
    );

    // Should have map/reduce intrinsics
    let has_map = mir.defs.iter().any(|d| {
        if let mir::Def::Function { body, .. } = d {
            body.exprs
                .iter()
                .any(|e| matches!(e, mir::Expr::Intrinsic { name, .. } if name.contains("map")))
        } else {
            false
        }
    });
    assert!(has_map, "Expected map intrinsic");

    let has_reduce = mir.defs.iter().any(|d| {
        if let mir::Def::Function { body, .. } = d {
            body.exprs
                .iter()
                .any(|e| matches!(e, mir::Expr::Intrinsic { name, .. } if name.contains("reduce")))
        } else {
            false
        }
    });
    assert!(has_reduce, "Expected reduce intrinsic");
}

// =============================================================================
// Defunctionalization Scenarios
// =============================================================================

#[test]
fn test_defunctionalization() {
    // Tests various defunctionalization scenarios:
    // - Multiple HOF calls with different captures
    // - Nested captures (grandparent scope)
    // - Same lambda reused
    // - Chain of HOF calls
    let mir = compile_to_mir(
        r#"
def different_captures(x: i32, y: i32, arr: [4]i32) ([4]i32, [4]i32) =
    let result1 = map(|e: i32| e + x, arr) in
    let result2 = map(|e: i32| e * y, arr) in
    (result1, result2)

def nested_capture(x: i32, arr: [4]i32) [4]i32 =
    let outer = |y: i32|
        let inner = |z: i32| x + y + z in
        inner(y)
    in
    map(outer, arr)

def reused_lambda(x: i32, arr1: [4]i32, arr2: [4]i32) ([4]i32, [4]i32) =
    let adder = |e: i32| e + x in
    let result1 = map(adder, arr1) in
    let result2 = map(adder, arr2) in
    (result1, result2)

def hof_chain(scale: i32, offset: i32, arr: [4]i32) i32 =
    let scaled = map(|x: i32| x * scale, arr) in
    let shifted = map(|x: i32| x + offset, scaled) in
    reduce(|a: i32, b: i32| a + b, 0, shifted)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let (a, b) = different_captures(1, 2, [1, 2, 3, 4]) in
    let c = nested_capture(10, [1, 2, 3, 4]) in
    let (d, e) = reused_lambda(5, [1, 2, 3, 4], [5, 6, 7, 8]) in
    let f = hof_chain(2, 10, [1, 2, 3, 4]) in
    @[f32.i32(a[0] + b[0] + c[0] + d[0] + e[0] + f), 0.0, 0.0, 1.0]
"#,
    );

    // Should have multiple lifted lambdas
    let lambda_count = mir
        .defs
        .iter()
        .filter(|d| matches!(d, mir::Def::Function { name, .. } if name.contains("_w_lambda_")))
        .count();
    assert!(
        lambda_count >= 4,
        "Expected at least 4 lifted lambdas, found {}",
        lambda_count
    );
}

// =============================================================================
// Type Checking Errors
// =============================================================================

#[test]
fn test_type_errors() {
    // These fail during type checking, before monomorphization, so no entry points needed

    // Arrays of functions are not permitted
    assert!(
        should_fail_type_check(
            r#"
def test: [2](i32 -> i32) =
    [|x: i32| x + 1, |x: i32| x * 2]
"#
        ),
        "Should reject arrays of functions"
    );

    // Function from if expression
    assert!(
        should_fail_type_check(
            r#"
def choose(b: bool) (i32 -> i32) =
    if b then |x: i32| x + 1 else |x: i32| x * 2
"#
        ),
        "Should reject function returned from if expression"
    );

    // Loop parameter cannot be a function
    assert!(
        should_fail_type_check(
            r#"
def test: (i32 -> i32) =
    loop f = |x: i32| x while false do f
"#
        ),
        "Should reject function as loop parameter"
    );
}

// =============================================================================
// Materialization Optimization
// =============================================================================

#[test]
fn test_materialization_optimization() {
    // Tests that materialization hoisting works correctly
    let mir = compile_to_mir(
        r#"
def identity(arr: [3]i32) [3]i32 = arr

def no_redundant_complex(arr: [3]i32, i: i32) i32 =
    if true then (identity(arr))[i] else (identity(arr))[i]

def no_materialize_tuple(x: i32) i32 =
    let pair = (x, x + 1) in
    let (a, b) = pair in
    a + b

def no_materialize_loop_tuple(arr: [10]i32) i32 =
    let (sum, _) = loop (acc, i) = (0, 0) while i < 10 do
        (acc + arr[i], i + 1)
    in sum

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = no_redundant_complex([1, 2, 3], 0) in
    let b = no_materialize_tuple(5) in
    let c = no_materialize_loop_tuple([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) in
    @[f32.i32(a + b + c), 0.0, 0.0, 1.0]
"#,
    );

    let mir_str = format!("{}", mir);

    // Check no_redundant_complex has at most 1 materialize after hoisting
    let complex_fn = extract_function_mir(&mir_str, "no_redundant_complex");
    let complex_count = complex_fn.matches("@materialize").count();
    assert!(
        complex_count <= 1,
        "Expected at most 1 materialize in no_redundant_complex, found {}",
        complex_count
    );

    // Check tuple destructuring doesn't use materialize
    let tuple_fn = extract_function_mir(&mir_str, "no_materialize_tuple");
    let tuple_count = tuple_fn.matches("@materialize").count();
    assert_eq!(
        tuple_count, 0,
        "Tuple destructuring should not use materialize, found {}",
        tuple_count
    );
}

// =============================================================================
// Math Functions and Conversions
// =============================================================================

#[test]
fn test_math_and_conversions() {
    // Tests: f32 conversions, math operations, qualified names
    let mir = compile_to_mir(
        r#"
def conversions(x: i32, y: i64) f32 =
    let f1 = f32.i32(x) in
    let f2 = f32.i64(y) in
    f1 + f2

def math_ops(x: f32) f32 =
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

def vector_length(v: vec2f32) f32 =
    f32.sqrt(v.x * v.x + v.y * v.y)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let a = conversions(1, 2i64) in
    let b = math_ops(1.0) in
    let c = vector_length(@[3.0, 4.0]) in
    @[a + b + c, 0.0, 0.0, 1.0]
"#,
    );

    // Check for math function calls
    let math_def = find_def(&mir, "math_ops");
    let body = get_body(math_def);
    let call_names: Vec<_> = body
        .exprs
        .iter()
        .filter_map(|e| if let mir::Expr::Call { func, .. } = e { Some(func.as_str()) } else { None })
        .collect();

    assert!(call_names.iter().any(|n| n.contains("sin")), "Expected sin call");
    assert!(call_names.iter().any(|n| n.contains("cos")), "Expected cos call");
    assert!(
        call_names.iter().any(|n| n.contains("sqrt")),
        "Expected sqrt call"
    );
}

// =============================================================================
// Matrix Operations
// =============================================================================

#[test]
fn test_matrix_operations() {
    // Tests: mul overloads (mat*mat, mat*vec, vec*mat)
    let mir = compile_to_mir(
        r#"
def test_mul(m1: mat4f32, m2: mat4f32, v: vec4f32) vec4f32 =
    let mat_result = mul(m1, m2) in
    let vec_result1 = mul(mat_result, v) in
    let vec_result2 = mul(v, m1) in
    vec_result1

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let m = @[[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]] in
    let v = @[1.0, 2.0, 3.0, 1.0] in
    test_mul(m, m, v)
"#,
    );

    // Check for mul variant calls
    let test_def = find_def(&mir, "test_mul");
    let body = get_body(test_def);
    let call_names: Vec<_> = body
        .exprs
        .iter()
        .filter_map(|e| if let mir::Expr::Call { func, .. } = e { Some(func.as_str()) } else { None })
        .collect();

    assert!(
        call_names.iter().any(|n| n.contains("mul_mat_mat")),
        "Expected mul_mat_mat"
    );
    assert!(
        call_names.iter().any(|n| n.contains("mul_mat_vec")),
        "Expected mul_mat_vec"
    );
    assert!(
        call_names.iter().any(|n| n.contains("mul_vec_mat")),
        "Expected mul_vec_mat"
    );
}

// =============================================================================
// GLSL Output
// =============================================================================

#[test]
fn test_glsl_output() {
    // Tests GLSL-specific output: vector constructors, let bindings, normalization
    let glsl = compile_to_glsl(
        r#"
#[uniform(set=0, binding=0)] def iResolution: vec2f32
#[uniform(set=0, binding=1)] def iTime: f32

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let coord = @[pos.x, pos.y] in
    let uv = @[coord.x / iResolution.x, coord.y / iResolution.y] in
    let x = pos.x + pos.y * iTime in
    @[uv.x, uv.y, x, 1.0]
"#,
    );

    // Should use vec2() not float() for vectors
    assert!(glsl.contains("vec2("), "Expected vec2() constructor");
    assert!(!glsl.contains("float("), "Should not have float() for vectors");

    // Should have local variables
    assert!(glsl.contains("vec2 coord"), "Expected coord local variable");
    assert!(glsl.contains("vec2 uv"), "Expected uv local variable");

    // Should have normalization variables
    assert!(glsl.contains("_w_norm_"), "Expected normalization variables");
}

// =============================================================================
// Complex Shader Integration
// =============================================================================

#[test]
fn test_complex_shader() {
    // Full shader with uniforms, matrices, map, multiple functions
    let mir = compile_to_mir(
        r#"
#[uniform(set=1, binding=0)] def iResolution: vec2f32
#[uniform(set=1, binding=1)] def iTime: f32

def verts: [3]vec4f32 =
    [@[-1.0, -1.0, 0.0, 1.0],
     @[3.0, -1.0, 0.0, 1.0],
     @[-1.0, 3.0, 0.0, 1.0]]

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vertex_id: i32) #[builtin(position)] vec4f32 =
    verts[vertex_id]

def translation(p: vec3f32) mat4f32 =
    @[[1.0f32, 0.0f32, 0.0f32, p.x],
      [0.0f32, 1.0f32, 0.0f32, p.y],
      [0.0f32, 0.0f32, 1.0f32, p.z],
      [0.0f32, 0.0f32, 0.0f32, 1.0f32]]

def rotation_y(angle: f32) mat4f32 =
    let s = f32.sin(angle) in
    let c = f32.cos(angle) in
    @[[c, 0.0f32, s, 0.0f32],
      [0.0f32, 1.0f32, 0.0f32, 0.0f32],
      [0.0 - s, 0.0f32, c, 0.0f32],
      [0.0f32, 0.0f32, 0.0f32, 1.0f32]]

def cube_corners: [8]vec3f32 =
    [@[-1.0, -1.0, 1.0], @[-1.0, 1.0, 1.0],
     @[1.0, 1.0, 1.0], @[1.0, -1.0, 1.0],
     @[-1.0, -1.0, -1.0], @[-1.0, 1.0, -1.0],
     @[1.0, 1.0, -1.0], @[1.0, -1.0, -1.0]]

def main_image(res: vec2f32, time: f32, fragCoord: vec2f32) vec4f32 =
    let cam = translation(@[0.0, 0.0, 10.0]) in
    let rot = rotation_y(time) in
    let mat = mul(rot, cam) in
    let v4s = map(|v: vec3f32| mul(@[v.x, v.y, v.z, 1.0], mat), cube_corners) in
    v4s[0]

#[fragment]
entry fragment_main(#[builtin(frag_coord)] pos: vec4f32) #[location(0)] vec4f32 =
    main_image(@[iResolution.x, iResolution.y], iTime, @[pos.x, pos.y])
"#,
    );

    let mir_str = format!("{}", mir);

    // Should have all the key functions
    assert!(mir_str.contains("vertex_main"), "Expected vertex_main");
    assert!(mir_str.contains("fragment_main"), "Expected fragment_main");
    assert!(mir_str.contains("translation"), "Expected translation");
    assert!(mir_str.contains("rotation_y"), "Expected rotation_y");
    assert!(mir_str.contains("main_image"), "Expected main_image");

    // Should have lambda for map
    let has_lambda = mir
        .defs
        .iter()
        .any(|d| matches!(d, mir::Def::Function { name, .. } if name.contains("_w_lambda_")));
    assert!(has_lambda, "Expected lifted lambda for map");
}

// =============================================================================
// Full Pipeline to SPIR-V
// =============================================================================

#[test]
fn test_full_pipeline_to_spirv() {
    // Verify the full pipeline compiles successfully to SPIR-V
    let source = r#"
#[uniform(set=0, binding=0)] def iTime: f32

def compute(x: f32, y: f32) f32 =
    let a = f32.sin(x) in
    let b = f32.cos(y) in
    a + b

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let s = compute(pos.x, pos.y) in
    @[s + iTime, 0.0, 0.0, 1.0]
"#;

    let mut frontend = crate::cached_frontend();
    let alias_checked = crate::Compiler::parse(source, &mut frontend.node_counter)
        .and_then(|p| p.desugar(&mut frontend.node_counter))
        .and_then(|d| d.resolve(&mut frontend.module_manager))
        .map(|r| r.fold_ast_constants())
        .and_then(|f| f.type_check(&mut frontend.module_manager, &mut frontend.schemes))
        .and_then(|t| t.alias_check())
        .expect("Failed before TLC transform");

    let builtins = crate::build_known_defs(&alias_checked.ast, &mut frontend.module_manager);
    let result = alias_checked
        .to_tlc(builtins, &frontend.schemes, &mut frontend.module_manager)
        .skip_partial_eval()
        .defunctionalize()
        .monomorphize()
        .to_mir()
        .hoist_materializations()
        .normalize()
        .default_address_spaces()
        .parallelize_soacs()
        .filter_reachable()
        .lift_bindings()
        .lower();

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}
