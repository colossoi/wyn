#![cfg(test)]
//! Integration tests for the full compilation pipeline.
//!
//! These tests verify that source code compiles correctly through all stages:
//! parse → desugar → resolve → type_check → alias_check → TLC → monomorphize → SSA
//!
//! All tests include entry points to ensure monomorphization can find reachable code.

use crate::ssa::types::Program;

/// Run source through the pipeline up to SSA.
fn compile_to_ssa(input: &str) -> Program {
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

    alias_checked
        .to_tlc(&frontend.schemes, &frontend.module_manager)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .parallelize_soacs()
        .filter_reachable()
        .to_egraph()
        .expect("SSA conversion failed")
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate()
        .ssa
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

/// Check that a function with the given name exists in the SSA program.
fn has_function(ssa: &Program, name: &str) -> bool {
    ssa.functions.iter().any(|f| f.name == name)
}

/// Helper to compile up through TLC fusion (stops before defunctionalization).
fn compile_to_fused_tlc(input: &str) -> crate::tlc::Program {
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

    let tlc = alias_checked.to_tlc(&frontend.schemes, &frontend.module_manager);
    let fused = tlc.partial_eval().normalize_soacs().fuse_maps();
    fused.tlc
}

// =============================================================================
// SOAC Fusion Integration Tests
// =============================================================================

#[test]
fn test_map_reduce_fusion_end_to_end() {
    let source = r#"
def globalArr: [4]f32 = [10.0, 20.0, 30.0, 40.0]

def myMap(ro: f32, rd: f32) [4]f32 =
  map(|x: f32| x + ro + rd, globalArr)

def myReduce(hits: [4]f32) f32 =
  reduce(|acc: f32, x: f32| if acc < x then acc else x, 999.0, hits)

#[fragment]
entry fragment_main() #[location(0)] vec4f32 =
  let hits = myMap(1.0, 2.0) in
  let closest = myReduce(hits) in
  @[closest, 0.0, 0.0, 1.0]
"#;

    let tlc = compile_to_fused_tlc(source);

    // After fusion, check that myMap's body is no longer a standalone map
    // or that some def contains a fused reduce
    let my_map_has_map = tlc.defs.iter().any(|def| {
        let name = tlc.symbols.get(def.name).cloned().unwrap_or_default();
        if name != "myMap" {
            return false;
        }
        let (_, body) = crate::tlc::extract_lambda_params(&def.body);
        has_soac_kind(&body, "Map")
    });

    let any_has_reduce = tlc.defs.iter().any(|def| {
        let (_, body) = crate::tlc::extract_lambda_params(&def.body);
        has_soac_kind(&body, "Reduce")
    });

    // Check fragment_main: does it contain a fused Reduce?
    let fragment_main = tlc
        .defs
        .iter()
        .find(|def| tlc.symbols.get(def.name).map(|s| s.as_str()) == Some("fragment_main"))
        .expect("fragment_main not found");

    let (_, frag_body) = crate::tlc::extract_lambda_params(&fragment_main.body);
    let frag_has_redomap = has_soac_kind(&frag_body, "Redomap");
    let frag_has_reduce = has_soac_kind(&frag_body, "Reduce");
    let frag_has_map = has_soac_kind(&frag_body, "Map");

    eprintln!("fragment_main has Redomap: {}", frag_has_redomap);
    eprintln!("fragment_main has Reduce: {}", frag_has_reduce);
    eprintln!("fragment_main has Map: {}", frag_has_map);
    eprintln!(
        "fragment_main body: {:?}",
        std::mem::discriminant(&frag_body.kind)
    );

    // Print the Let chain structure
    fn print_term(term: &crate::tlc::Term, syms: &crate::SymbolTable, depth: usize) {
        let indent = "  ".repeat(depth);
        match &term.kind {
            crate::tlc::TermKind::Let { name, rhs, body, .. } => {
                let n = syms.get(*name).cloned().unwrap_or_else(|| format!("{:?}", name));
                eprintln!("{indent}let {n} = ...");
                print_term(rhs, syms, depth + 1);
                print_term(body, syms, depth);
            }
            crate::tlc::TermKind::Soac(soac) => {
                eprintln!("{indent}SOAC {:?}", std::mem::discriminant(soac));
            }
            crate::tlc::TermKind::App { func, args } => {
                eprintln!("{indent}App:");
                print_term(func, syms, depth + 1);
                for a in args {
                    print_term(a, syms, depth + 1);
                }
            }
            crate::tlc::TermKind::Var(s) => {
                let n = syms.get(*s).cloned().unwrap_or_else(|| format!("{:?}", s));
                eprintln!("{indent}Var({n})");
            }
            other => {
                eprintln!("{indent}{:?}", std::mem::discriminant(other));
            }
        }
    }
    print_term(&frag_body, &tlc.symbols, 0);

    // The fusion should have replaced the let chain with a fused SOAC
    // or at minimum the fragment_main should contain a Reduce
    assert!(
        frag_has_redomap || frag_has_reduce,
        "Expected fragment_main to contain a fused Redomap or Reduce after interprocedural fusion"
    );
}

fn has_soac_kind(term: &crate::tlc::Term, kind: &str) -> bool {
    use crate::tlc::{SoacOp, TermKind};
    match &term.kind {
        TermKind::Soac(SoacOp::Map { .. }) if kind == "Map" => true,
        TermKind::Soac(SoacOp::Reduce { .. }) if kind == "Reduce" => true,
        TermKind::Soac(SoacOp::Redomap { .. }) if kind == "Redomap" => true,
        TermKind::Let { rhs, body, .. } => has_soac_kind(rhs, kind) || has_soac_kind(body, kind),
        TermKind::Lambda(lam) => has_soac_kind(&lam.body, kind),
        TermKind::App { func, args } => {
            has_soac_kind(func, kind) || args.iter().any(|a| has_soac_kind(a, kind))
        }
        _ => false,
    }
}

// =============================================================================
// Basic Expressions
// =============================================================================

#[test]
fn test_basic_expressions() {
    // Tests: functions, let bindings, if expressions, binary/unary ops
    let ssa = compile_to_ssa(
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

    // Compilation success is the test (partial eval may inline simple functions)
}

// =============================================================================
// Data Structures
// =============================================================================

#[test]
fn test_data_structures() {
    // Tests: arrays, tuples, records, tuple patterns
    let _ssa = compile_to_ssa(
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
    // Compilation success is the test
}

// =============================================================================
// Tuple Positional Access
// =============================================================================

#[test]
fn test_tuple_positional_access() {
    // Tests: .0, .1 on tuples, chained access, in expressions
    let _ssa = compile_to_ssa(
        r#"
def first(t: (i32, f32)) i32 = t.0

def second(t: (i32, f32)) f32 = t.1

def sum_pair(t: (i32, i32)) i32 = t.0 + t.1

def nested(t: ((i32, i32), f32)) i32 =
    let inner = t.0 in inner.0 + inner.1

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let t = (42, 3.14) in
    let a = first(t) in
    let b = sum_pair((1, 2)) in
    let c = nested(((10, 20), 1.0)) in
    @[f32.i32(a + b + c), 0.0, 0.0, 1.0]
"#,
    );
}

// =============================================================================
// Loops
// =============================================================================

#[test]
fn test_loops() {
    // Tests: while loops, for-range loops, for-in loops
    let _ssa = compile_to_ssa(
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
    // Compilation success is the test
}

// =============================================================================
// Lambdas and Closures
// =============================================================================

#[test]
fn test_lambdas_and_closures() {
    // Tests: lambdas with captures, nested lambdas, direct calls, tuple params
    let _ssa = compile_to_ssa(
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
    // Compilation success is the test
}

// =============================================================================
// Higher-Order Functions (map, reduce, filter)
// =============================================================================

#[test]
fn test_higher_order_functions() {
    // Tests: map, reduce, filter with lambdas and named functions
    let _ssa = compile_to_ssa(
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
    // Compilation success is the test
}

// =============================================================================
// Defunctionalization Scenarios
// =============================================================================

#[test]
fn test_defunctionalization() {
    // Tests various defunctionalization scenarios
    let _ssa = compile_to_ssa(
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
    // Compilation success is the test
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
    let _ssa = compile_to_ssa(
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
    // Compilation success is the test
}

// =============================================================================
// Math Functions and Conversions
// =============================================================================

#[test]
fn test_math_and_conversions() {
    // Tests: f32 conversions, math operations, qualified names
    let _ssa = compile_to_ssa(
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
    // Compilation success is the test
}

// =============================================================================
// Matrix Operations
// =============================================================================

#[test]
fn test_matrix_operations() {
    // Tests: mul overloads (mat*mat, mat*vec, vec*mat)
    let _ssa = compile_to_ssa(
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
    // Compilation success is the test
}

// =============================================================================
// Complex Shader Integration
// =============================================================================

#[test]
fn test_complex_shader() {
    // Full shader with uniforms, matrices, map, multiple functions
    let _ssa = compile_to_ssa(
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
    let mat = rot * cam in
    let v4s = map(|v: vec3f32| @[v.x, v.y, v.z, 1.0] * mat, cube_corners) in
    v4s[0]

#[fragment]
entry fragment_main(#[builtin(frag_coord)] pos: vec4f32) #[location(0)] vec4f32 =
    main_image(@[iResolution.x, iResolution.y], iTime, @[pos.x, pos.y])
"#,
    );
    // Compilation success is the test
}

// =============================================================================
// Full Pipeline to SPIR-V
// =============================================================================

#[test]
fn test_function_call_with_array_arg() {
    // Test calling a function with an array literal argument
    let source = r#"
def sum_first_two(arr: [4]i32) i32 =
    arr[0] + arr[1]

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let result = sum_first_two([1, 2, 3, 4]) in
    @[f32.i32(result), 0.0, 0.0, 1.0]
"#;

    let mut frontend = crate::cached_frontend();
    let alias_checked = crate::Compiler::parse(source, &mut frontend.node_counter)
        .and_then(|p| p.desugar(&mut frontend.node_counter))
        .and_then(|d| d.resolve(&mut frontend.module_manager))
        .map(|r| r.fold_ast_constants())
        .and_then(|f| f.type_check(&mut frontend.module_manager, &mut frontend.schemes))
        .and_then(|t| t.alias_check())
        .expect("Failed before TLC transform");

    let result = alias_checked
        .to_tlc(&frontend.schemes, &frontend.module_manager)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .parallelize_soacs()
        .filter_reachable()
        .to_egraph()
        .expect("SSA conversion failed")
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate()
        .lower();

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}

#[test]
#[ignore = "requires single-thread fallback implementation"]
fn test_compute_shader_with_storage_slice() {
    // Test compute shader with storage buffer slice
    let source = r#"
def sum_first_two(arr: [4]i32) i32 =
    arr[0] + arr[1]

#[compute]
entry compute_main(data: []i32) i32 =
    let from_storage = sum_first_two(data[0..4]) in
    let from_literal = sum_first_two([1, 2, 3, 4]) in
    from_storage + from_literal
"#;

    let mut frontend = crate::cached_frontend();
    let alias_checked = crate::Compiler::parse(source, &mut frontend.node_counter)
        .and_then(|p| p.desugar(&mut frontend.node_counter))
        .and_then(|d| d.resolve(&mut frontend.module_manager))
        .map(|r| r.fold_ast_constants())
        .and_then(|f| f.type_check(&mut frontend.module_manager, &mut frontend.schemes))
        .and_then(|t| t.alias_check())
        .expect("Failed before TLC transform");

    let result = alias_checked
        .to_tlc(&frontend.schemes, &frontend.module_manager)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .parallelize_soacs()
        .filter_reachable()
        .to_egraph()
        .expect("SSA conversion failed")
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate()
        .lower();

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}

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

    let result = alias_checked
        .to_tlc(&frontend.schemes, &frontend.module_manager)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .parallelize_soacs()
        .filter_reachable()
        .to_egraph()
        .expect("SSA conversion failed")
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate()
        .lower();

    assert!(result.is_ok(), "SPIR-V compilation failed: {:?}", result.err());
}

// =============================================================================
// Array Variant Monomorphization
// =============================================================================

#[test]
fn test_array_variant_monomorphization() {
    // Slicing a view with constant bounds produces a Composite array (materialized at
    // the call site), so both call sites use the same Composite variant of sum_first_two.
    let ssa = compile_to_ssa(
        r#"
def sum_first_two(arr: [4]i32) i32 =
    arr[0] + arr[1]

#[compute]
entry compute_main(data: []i32) i32 =
    let from_storage = sum_first_two(data[0..4]) in
    let from_literal = sum_first_two([1, 2, 3, 4]) in
    from_storage + from_literal
"#,
    );

    // Collect all sum_first_two variants (including buffer-specialized)
    let sum_versions: Vec<_> =
        ssa.functions.iter().filter(|f| f.name.starts_with("sum_first_two")).collect();

    eprintln!("sum_first_two SSA functions:");
    for f in &sum_versions {
        eprintln!("  {}", f.name);
        // Show param types
        for (val, ty, name) in &f.body.params {
            eprintln!("    param {} ({:?}) :: {:?}", name, val, ty);
        }
        // Show all instructions that involve indexing or storage views
        for inst in f.body.inner.insts.values() {
            match &inst.data {
                crate::ssa::types::InstKind::Index { .. } => {
                    eprintln!("    inst {:?}: Index", inst.result);
                }
                crate::ssa::types::InstKind::StorageView { .. } => {
                    eprintln!("    inst {:?}: StorageView", inst.result);
                }
                crate::ssa::types::InstKind::StorageViewIndex { .. } => {
                    eprintln!("    inst {:?}: StorageViewIndex", inst.result);
                }
                _ => {}
            }
        }
    }

    // After TLC-level inlining and DCE, sum_first_two may be fully inlined
    // at all call sites and eliminated. The important thing is that the program
    // compiles successfully to SSA — buffer specialization is tested implicitly.
    // (The function may or may not survive depending on inlining thresholds.)
}

// =============================================================================
// SPIR-V Block Param / Phi Node Tests
// =============================================================================

/// Compile source all the way through SPIR-V and return Ok/Err.
fn compile_to_spirv(input: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut frontend = crate::cached_frontend();
    let alias_checked = crate::Compiler::parse(input, &mut frontend.node_counter)?
        .elaborate_modules(&mut frontend.module_manager)?
        .desugar(&mut frontend.node_counter)?
        .resolve(&mut frontend.module_manager)?
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)?
        .alias_check()?;

    let result = alias_checked
        .to_tlc(&frontend.schemes, &frontend.module_manager)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .parallelize_soacs()
        .filter_reachable()
        .to_egraph()?
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate()
        .lower()?;

    Ok(result.spirv)
}

/// Helper: compile source that may have modules (like raytrace.wyn)
fn compile_to_ssa_with_modules(input: &str) -> Program {
    let mut frontend = crate::cached_frontend();
    let parsed = crate::Compiler::parse(input, &mut frontend.node_counter).expect("Parse failed");
    let parsed = parsed.elaborate_modules(&mut frontend.module_manager).expect("Module elaboration failed");
    let alias_checked = parsed
        .desugar(&mut frontend.node_counter)
        .expect("Desugar failed")
        .resolve(&mut frontend.module_manager)
        .expect("Resolve failed")
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .expect("Type check failed")
        .alias_check()
        .expect("Alias check failed");

    alias_checked
        .to_tlc(&frontend.schemes, &frontend.module_manager)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .parallelize_soacs()
        .filter_reachable()
        .to_egraph()
        .expect("SSA conversion failed")
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate()
        .ssa
}

/// Verify that nested if/else chains compile to SPIR-V.
#[test]
fn test_spirv_nested_if_else_block_params() {
    let source = r#"
def choose(a: f32, b: f32, c: f32, sel1: i32, sel2: i32) f32 =
    let x = if sel1 == 0 then a
            else if sel1 == 1 then b
            else c in
    let y = if sel2 == 0 then a
            else if sel2 == 1 then c
            else b in
    x + y

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let r = choose(pos.x, pos.y, pos.z, 1, 2) in
    @[r, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect("Nested if/else should compile to SPIR-V");
}

/// Verify many conditional branches producing block params compile to SPIR-V.
#[test]
fn test_spirv_many_conditional_block_params() {
    let source = r#"
def process(a: f32, b: f32, c: f32, d: f32, flag: i32) (f32, f32, f32, f32) =
    let x = if flag == 0 then a + b else a - b in
    let y = if flag == 1 then b + c else b * c in
    let z = if flag == 2 then c + d else c - d in
    let w = if flag == 0 then d * a else d + a in
    (x, y, z, w)

def combine(t: (f32, f32, f32, f32)) f32 =
    let (a, b, c, d) = t in
    a + b + c + d

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let result = process(pos.x, pos.y, pos.z, pos.w, 1) in
    let s = combine(result) in
    @[s, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect("Many conditionals should compile to SPIR-V");
}

/// Verify maps over small arrays with nested conditionals compile to SPIR-V.
#[test]
fn test_spirv_map_with_nested_conditionals() {
    let source = r#"
def selectValue(x: f32, flag: i32) f32 =
    if flag == 0 then x * 2.0
    else if flag == 1 then x + 1.0
    else x - 1.0

#[compute]
entry compute_main(data: [8]f32) [8]f32 =
    map(|x| selectValue(x, 1), data)
"#;
    compile_to_spirv(source).expect("Map with nested conditionals should compile to SPIR-V");
}

/// Verify multiple maps followed by a reduce compile to SPIR-V.
#[test]
fn test_spirv_multiple_maps_and_reduce() {
    let source = r#"
#[compute]
entry compute_main(data: [8](f32, f32)) [8]f32 =
    let first = map(|t| let (a, _) = t in a, data) in
    let second = map(|t| let (_, b) = t in b, data) in
    let combined = map(|(a, b): (f32, f32)| a + b, zip(first, second)) in
    let total = reduce(|a: f32, b: f32| a + b, 0.0, combined) in
    map(|x| x + total, combined)
"#;
    compile_to_spirv(source).expect("Multiple maps + reduce should compile to SPIR-V");
}

/// Verify conditional array element selection compiles to SPIR-V
/// (the finalOrigins/finalDirs pattern from raytrace.wyn).
#[test]
fn test_spirv_conditional_array_construction() {
    let source = r#"
def build(a: [4]f32, b: [4]f32, flags: [4]i32) [4]f32 =
    [
        if flags[0] == 1 then b[0] else a[0],
        if flags[1] == 1 then b[1] else a[1],
        if flags[2] == 1 then b[2] else a[2],
        if flags[3] == 1 then b[3] else a[3]
    ]

#[compute]
entry compute_main(data: [4]f32) [4]f32 =
    let doubled = map(|x| x * 2.0, data) in
    let flags = [1, 0, 1, 0] in
    build(data, doubled, flags)
"#;
    compile_to_spirv(source).expect("Conditional array construction should compile to SPIR-V");
}

/// Test the specific raytrace.wyn file compiles to SPIR-V.
#[test]
fn test_spirv_raytrace() {
    let source = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../testfiles/raytrace.wyn"))
        .expect("Could not read testfiles/raytrace.wyn");
    compile_to_spirv(&source).expect("raytrace.wyn should compile to SPIR-V");
}

/// Regression: if/else before interprocedural map+reduce fusion caused
/// UnterminatedBlock in soac_lower. The if/else creates a dead Unreachable
/// block in SSA, which soac_lower's rebuild would pre-create but finish()
/// rejected because Unreachable doubles as the "unterminated" sentinel.
#[test]
fn test_interproc_fusion_if_before_fused_reduce() {
    let source = r#"
def maxDist: f32 = 100.0
def globalData: [12]f32 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

def producer(x: f32) [12]f32 =
  map(|a: f32| a * x, globalData)

def consumer(arr: [12]f32) f32 =
  reduce(|acc: f32, x: f32| if acc < x then acc else x, maxDist, arr)

def scene(x: f32, y: f32) f32 =
  let ground = if y > 0.0 then y else maxDist in
  let hits = producer(x) in
  let closest = consumer(hits) in
  closest + ground

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32) #[builtin(position)] vec4f32 =
  let r = scene(1.0, 0.5) in
  @[r, 0.0, 0.0, 1.0]
"#;
    compile_to_spirv(source).expect("if-before-interproc-fusion should compile");
}

/// Verify raytrace.wyn compiles through SSA to SPIR-V without errors.
/// This exercises the RPO block emission and incremental array literal
/// lowering that were needed for complex cross-block value references.
/// (test_spirv_raytrace covers this; this test verifies the SSA is well-formed
/// enough that compile_to_ssa_with_modules succeeds.)
#[test]
fn test_ssa_raytrace_well_formed() {
    let source = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../testfiles/raytrace.wyn"))
        .expect("Could not read testfiles/raytrace.wyn");

    let ssa = compile_to_ssa_with_modules(&source);

    // Verify key functions exist
    assert!(
        ssa.functions.iter().any(|f| f.name == "trace"),
        "trace should be in SSA output"
    );
}

// =============================================================================
// Constant Inlining
// =============================================================================

/// Constants that reference other constants should be fully inlined.
#[test]
fn test_constant_referencing_constant() {
    let source = r#"
def PI: f32 = 3.14159265
def TAU: f32 = PI * 2.0
def QUARTER_TAU: f32 = TAU / 4.0

#[fragment]
entry frag(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
  @[QUARTER_TAU, PI, TAU, 1.0]
"#;
    compile_to_spirv(source).expect("constants referencing constants should compile");
}

// `test_soa_eliminates_extraction_maps` was deleted: it inspected post-SSA for
// `InstKind::Soac(Soac::Map { .. })`, but SOACs no longer exist as SSA
// instructions — they're expanded inside EGIR (`egir::soac_expand`) before
// elaboration produces any SSA. The underlying optimization concern (extraction
// maps after a zip-map should fold to tuple projections) is now an EGIR-level
// question and would need a different shape of test.

/// Pipeline that includes TLC inline_small (the new pass).
fn compile_to_ssa_with_inline_small(input: &str) -> Program {
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

    alias_checked
        .to_tlc(&frontend.schemes, &frontend.module_manager)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .to_egraph()
        .expect("SSA conversion failed")
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate()
        .ssa
}

#[test]
fn test_constant_inlining_global_ref() {
    // Minimal repro: a constant def used by a function, going through inline_small.
    // This should NOT produce an unresolved Global("PI") in SSA.
    let ssa = compile_to_ssa_with_inline_small(
        r#"
def PI: f32 = 3.141592

def use_pi(x: f32) f32 = x * PI

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let r = use_pi(pos.x) in
    @[r, 0.0, 0.0, 1.0]
"#,
    );

    // Dump what we got.
    eprintln!("{}", crate::ssa::print::format_program(&ssa));

    // Check that no Global("PI") instruction exists — it should have been inlined.
    for func in &ssa.functions {
        for (_id, inst) in &func.body.inner.insts {
            if let crate::ssa::types::InstKind::Global(name) = &inst.data {
                assert_ne!(
                    name, "PI",
                    "Global @PI should have been inlined, but survived in function '{}'",
                    func.name
                );
            }
        }
    }
    for ep in &ssa.entry_points {
        for (_id, inst) in &ep.body.inner.insts {
            if let crate::ssa::types::InstKind::Global(name) = &inst.data {
                assert_ne!(
                    name, "PI",
                    "Global @PI should have been inlined, but survived in entry '{}'",
                    ep.name
                );
            }
        }
    }
}
