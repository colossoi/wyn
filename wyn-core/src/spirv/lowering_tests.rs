use super::lower;
use crate::error::Result;

fn compile_to_spirv(source: &str) -> Result<Vec<u32>> {
    // Use the typestate API to ensure proper compilation pipeline
    let mut frontend = crate::cached_frontend();
    let parsed = crate::Compiler::parse(source, &mut frontend.node_counter).expect("Parsing failed");
    let (flattened, _backend) = parsed
        .desugar(&mut frontend.node_counter)
        .expect("Desugaring failed")
        .resolve(&frontend.module_manager)
        .expect("Name resolution failed")
        .fold_ast_constants()
        .type_check(&frontend.module_manager, &mut frontend.schemes)
        .expect("Type checking failed")
        .alias_check()
        .expect("Alias checking failed")
        .lower_to_sir()
        .expect("SIR lowering failed")
        .transform()
        .flatten()
        .expect("Flattening failed");

    let inplace_info = crate::alias_checker::analyze_inplace(&flattened.mir);
    lower(&flattened.mir, &inplace_info)
}

#[test]
fn test_simple_constant() {
    let spirv = compile_to_spirv("def x = 42").unwrap();
    assert!(!spirv.is_empty());
    // SPIR-V magic number
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_simple_function() {
    let spirv = compile_to_spirv("def add(x, y) = x + y").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_let_binding() {
    let spirv = compile_to_spirv("def f = let x = 1 in x + 2").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_arithmetic() {
    let spirv = compile_to_spirv("def f(x, y) = x * y + x / y - 1").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_nested_let() {
    let spirv = compile_to_spirv("def f = let a = 1 in let b = 2 in a + b").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_if_expression() {
    let spirv = compile_to_spirv("def f(x) = if x == 0 then 1 else 2").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_comparisons() {
    let spirv = compile_to_spirv("def f(x, y) = if x < y then 1 else if x > y then 2 else 0").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_tuple_literal() {
    let spirv = compile_to_spirv("def f = (1, 2, 3)").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_array_literal() {
    let spirv = compile_to_spirv("def f = [1, 2, 3]").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_unary_negation() {
    let spirv = compile_to_spirv("def f(x) = -x").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_record_field_access() {
    let spirv = compile_to_spirv(
        r#"
def get_x(r:{x:i32, y:i32}) i32 = r.x
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_closure_capture_access() {
    // This test uses tuple_access intrinsic for closure field access
    let spirv = compile_to_spirv(
        r#"
def test(x:i32) i32 =
    let f = |y:i32| x + y in
    f(10)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_polymorphic_dot2() {
    // Test polymorphic function with type parameters that need proper instantiation
    // This reproduces the primitives.wyn issue where Vec type has unresolved size variable
    let spirv = compile_to_spirv(
        r#"
def dot2<E, T>(v: T) E = dot(v, v)

def test_dot2_vec3(v: vec3f32) f32 = dot2(v)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_polymorphic_dot2_in_expression() {
    // Test dot2 used in a more complex expression like in primitives.wyn
    // sdCappedTorus: f32.sqrt(dot2(p) + ra*ra - 2.0*ra*k) - rb
    let spirv = compile_to_spirv(
        r#"
def dot2<E, T>(v: T) E = dot(v, v)

def sdCappedTorus(p: vec3f32, ra: f32, rb: f32, k: f32) f32 =
  f32.sqrt(dot2(p) + ra*ra - 2.0*ra*k) - rb
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_polymorphic_dot2_vec2_and_vec3() {
    // Test dot2 with both vec2 and vec3 in same program (like primitives.wyn)
    let spirv = compile_to_spirv(
        r#"
def dot2<E, T>(v: T) E = dot(v, v)

def test_vec3(v: vec3f32) f32 = dot2(v)
def test_vec2(v: vec2f32) f32 = dot2(v)
def test_both(v3: vec3f32, v2: vec2f32) f32 = dot2(v3) + dot2(v2)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_scan_inclusive() {
    // Inclusive scan (prefix sum): scan (+) 0 [1,2,3] = [1, 3, 6]
    let spirv = compile_to_spirv(
        r#"
def sum_scan(arr: [4]i32) [4]i32 = scan((|a, b| a + b), 0, arr)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_map_variants() {
    // Test all map variants: map (desugars to map1), map2, map3, map4, map5
    // map2-map5 take functions with tuple arguments: (A, B) -> C, etc.
    let spirv = compile_to_spirv(
        r#"
def double(x: i32) i32 = x * 2

def test_map(arr: [3]i32) [3]i32 = map(double, arr)
def test_map2(xs: [3]i32, ys: [3]i32) [3]i32 = map2(|(x, y)| x + y, xs, ys)
def test_map3(xs: [3]i32, ys: [3]i32, zs: [3]i32) [3]i32 = map3(|(x, y, z)| x + y + z, xs, ys, zs)
def test_map4(a: [3]i32, b: [3]i32, c: [3]i32, d: [3]i32) [3]i32 = map4(|(a, b, c, d)| a + b + c + d, a, b, c, d)
def test_map5(a: [3]i32, b: [3]i32, c: [3]i32, d: [3]i32, e: [3]i32) [3]i32 = map5(|(a, b, c, d, e)| a + b + c + d + e, a, b, c, d, e)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_scatter_update() {
    // Scatter: write values to array at given indices
    let spirv = compile_to_spirv(
        r#"
def scatter_test(dest: [5]i32, indices: [2]i32, values: [2]i32) [5]i32 =
    scatter(dest, indices, values)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_reduce_by_index() {
    // Histogram / reduce_by_index: accumulate values at indices using operator
    let spirv = compile_to_spirv(
        r#"
def hist_test(dest: [3]i32, indices: [4]i32, values: [4]i32) [3]i32 =
    reduce_by_index(dest, |a, b| a + b, 0, indices, values)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_hist() {
    // hist is an alias for reduce_by_index
    let spirv = compile_to_spirv(
        r#"
def hist_alias_test(dest: [3]i32, indices: [4]i32, values: [4]i32) [3]i32 =
    hist(dest, |a, b| a + b, 0, indices, values)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_reduce_u32() {
    // Test reduce with u32 types - the initial value 0u32 must generate
    // an unsigned constant, not a signed one
    let spirv = compile_to_spirv(
        r#"
def sum_u32(arr: [4]u32) u32 =
    reduce(|a, b| a + b, 0u32, arr)
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_algebraic_simplifications() {
    // Test all algebraic identity simplifications compile correctly
    let spirv = compile_to_spirv(
        r#"
def test(x: f32) f32 =
    (0.0 - x) +     -- 0 - x → -x
    (0.0 + x) +     -- 0 + x → x
    (x + 0.0) +     -- x + 0 → x
    (x - 0.0) +     -- x - 0 → x
    (0.0 * x) +     -- 0 * x → 0
    (x * 0.0) +     -- x * 0 → 0
    (1.0 * x) +     -- 1 * x → x
    (x * 1.0) +     -- x * 1 → x
    (x / 1.0) +     -- x / 1 → x
    (-1.0 * x) +    -- -1 * x → -x
    (x * -1.0)      -- x * -1 → -x
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

/// Compile source to SPIR-V through the full pipeline including partial_eval
fn compile_to_spirv_with_partial_eval(source: &str) -> Result<Vec<u32>> {
    let mut frontend = crate::cached_frontend();
    let parsed = crate::Compiler::parse(source, &mut frontend.node_counter).expect("Parsing failed");
    let flattened = parsed
        .desugar(&mut frontend.node_counter)
        .expect("Desugaring failed")
        .resolve(&frontend.module_manager)
        .expect("Name resolution failed")
        .fold_ast_constants()
        .type_check(&frontend.module_manager, &mut frontend.schemes)
        .expect("Type checking failed")
        .alias_check()
        .expect("Alias checking failed")
        .lower_to_sir()
        .expect("SIR lowering failed")
        .transform()
        .flatten()
        .expect("Flattening failed")
        .0
        .hoist_materializations()
        .normalize()
        .monomorphize()
        .expect("Monomorphization failed")
        .partial_eval()
        .expect("Partial eval failed")
        .filter_reachable()
        .lift_bindings();

    let inplace_info = crate::alias_checker::analyze_inplace(&flattened.mir);
    lower(&flattened.mir, &inplace_info)
}

#[test]
fn test_partial_eval_inlined_function_local_id_collision() {
    // This test reproduces a bug where partial_eval inlining causes LocalId collision.
    //
    // The bug: When partial_eval inlines a function with all known args, it evaluates
    // the inlined function's body. If that body has a let binding with an Unknown RHS
    // (e.g., uses a uniform), it calls map_local() to allocate a LocalId. But local_map
    // still contains mappings from the OUTER function, causing LocalId collisions.
    //
    // Setup:
    // - helper() has a let binding that uses a uniform (Unknown)
    // - fragment_main has multiple locals before calling helper() with known args
    // - The helper's local collides with fragment_main's locals in local_map
    let spirv = compile_to_spirv_with_partial_eval(
        r#"
#[uniform(set=0, binding=0)] def iTime: f32

-- Helper function that will be inlined when called with known args.
-- The let binding 'weight' uses iTime which is unknown, so it gets residualized.
def helper(x: f32) f32 =
    let weight = x * iTime in
    weight

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    -- Create multiple locals to ensure LocalId collision
    let a = pos.x in
    let b = pos.y in
    let c = pos.z in
    -- Call helper with known arg - this gets inlined.
    -- helper's 'weight' local may collide with our locals.
    let d = helper(3.0) in
    @[d, a, b, c]
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_partial_eval_intrinsic_arg_types() {
    // This test reproduces a bug where partial_eval reifies intrinsic arguments
    // using the result type instead of the argument types.
    //
    // The bug: When residualizing an intrinsic like dot(vec3, vec3) -> f32,
    // the code was reifying the vector arguments with type f32 (the result type)
    // instead of vec3f32 (the argument type). This causes OpCompositeConstruct
    // to use a scalar type with multiple values.
    //
    // Setup:
    // - dot() takes two vec3 arguments and returns f32
    // - When the vec3 arguments are known vectors, they get reified with wrong type
    let spirv = compile_to_spirv_with_partial_eval(
        r#"
def verts: [3]vec4f32 =
  [@[-1.0, -1.0, 0.0, 1.0],
   @[3.0, -1.0, 0.0, 1.0],
   @[-1.0, 3.0, 0.0, 1.0]]

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vertex_id: i32) #[builtin(position)] vec4f32 = verts[vertex_id]

#[fragment]
entry fragment_main() #[location(0)] vec4f32 =
  let g = @[1.0, 2.0, 3.0] in
  let h = dot(g, @[127.1, 311.7, 74.7]) in
  @[h, h, h, 1.0]
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}
