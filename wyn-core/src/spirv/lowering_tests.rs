use crate::error::Result;

fn compile_to_glsl(source: &str) -> Result<crate::glsl::GlslOutput> {
    // Mirror compile_to_spirv through the `.lower_glsl()` terminal.
    let mut frontend = crate::cached_frontend();
    let parsed = crate::Compiler::parse(source, &mut frontend.node_counter).expect("Parsing failed");
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
        .expand_soacs(false)
        .optimize_skeleton()
        .elaborate()
        .lower_glsl()
}

fn compile_to_spirv(source: &str) -> Result<Vec<u32>> {
    // Use the typestate API to ensure proper compilation pipeline
    let mut frontend = crate::cached_frontend();
    let parsed = crate::Compiler::parse(source, &mut frontend.node_counter).expect("Parsing failed");
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

    let ssa = alias_checked
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
        .elaborate();

    ssa.lower().map(|l| l.spirv)
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
    // Test map with zip variants: map over zipped inputs
    let spirv = compile_to_spirv(
        r#"
def double(x: i32) i32 = x * 2

def test_map(arr: [3]i32) [3]i32 = map(double, arr)
def test_map2(xs: [3]i32, ys: [3]i32) [3]i32 = map(|(x, y)| x + y, zip(xs, ys))
def test_map3(xs: [3]i32, ys: [3]i32, zs: [3]i32) [3]i32 = map(|(x, y, z)| x + y + z, zip3(xs, ys, zs))
def test_map4(a: [3]i32, b: [3]i32, c: [3]i32, d: [3]i32) [3]i32 = map(|(a, b, c, d)| a + b + c + d, zip4(a, b, c, d))
def test_map5(a: [3]i32, b: [3]i32, c: [3]i32, d: [3]i32, e: [3]i32) [3]i32 = map(|(a, b, c, d, e)| a + b + c + d + e, zip5(a, b, c, d, e))
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
fn test_scatter_reachable_entry() {
    // Entry-point test: scatter is actually reached by the compiler pipeline.
    let spirv = compile_to_spirv(
        r#"
def scatter_demo(dest: [5]f32, indices: [2]i32, values: [2]f32) [5]f32 =
    scatter(dest, indices, values)

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let dest: [5]f32 = [0.0, 0.0, 0.0, 0.0, 0.0] in
    let indices: [2]i32 = [1, 3] in
    let values: [2]f32 = [7.0, 9.0] in
    let r = scatter_demo(dest, indices, values) in
    @[r[0], r[1], r[2], r[3]]
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
fn test_filter_over_tuple_array() {
    // Regression 1: filter over an array whose elements are tuples. The
    // tlc::soa pass distributes `Array[Tuple, _, _]` into
    // `Tuple(Array[Ti, _, _])`, so build_filter_loop sees an SoA tuple as
    // input. Previously this panicked with "Filter input must be an array
    // type, got Tuple(...)". After the SoA fix, filter allocates one
    // composite buffer per SoA component and emits one AWI per component
    // inside the body.
    //
    // Regression 2: the hash-cons of multiple `_w_intrinsic_uninit()`
    // calls with different return types now correctly produces distinct
    // nodes (previously collided into one because NodeKey didn't include
    // the attached type).
    let spirv = compile_to_spirv(
        r#"
def pairs: [4](i32, i32) = [(1, 2), (3, 4), (2, 5), (4, 8)]

def is_big(p: (i32, i32)) bool = p.0 > 2

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let big = filter(is_big, pairs) in
    @[f32.i32(length(big)), f32.i32(big[0].0), f32.i32(big[0].1), 1.0]
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_map_producing_tuple_elements_soa_output() {
    // Regression: Map whose lambda returns a tuple produces an Array of
    // tuples, which tlc::soa distributes into a tuple of arrays. The Map
    // lowering in egir::soac_expand previously allocated a single
    // loop-carried buffer with tuple-of-arrays type and emitted one AWI
    // per iteration on that tuple — SPIR-V's AWI backend can't handle
    // that because the tuple is not a composite array. This pattern was
    // previously hidden because raytrace.wyn's equivalent
    // `intersectAllSpheres` fuses with the downstream `findClosestHit`
    // reduce via fuse_maps, which bypasses the map's own output buffer.
    //
    // Here the map's output is consumed by indexing + length, so no
    // fusion happens and the bug surfaces.
    let spirv = compile_to_spirv(
        r#"
def pair_up(x: i32) (i32, i32) = (x, x * 2)

def xs: [4]i32 = [1, 2, 3, 4]

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let pairs = map(pair_up, xs) in
    let n = length(pairs) in
    let (a, b) = pairs[0] in
    @[f32.i32(n), f32.i32(a), f32.i32(b), 1.0]
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_uninit_different_types_hash_cons_distinct() {
    // Regression for NodeKey hash-consing: prior to including `ty` in the
    // key, two `_w_intrinsic_uninit()` calls with different return types
    // collapsed to one node (the first one's type stuck). The fallout
    // showed up as filter producing a single "tuple" buffer instead of N
    // per-component composite buffers, but the root cause is type-agnostic
    // hash-consing of nullary pure ops.
    //
    // This test constructs a scenario where two filter results have
    // different buffer types: one over [4]i32 (Composite<i32, 4>) and one
    // over [6]f32 (Composite<f32, 6>). Both trigger uninit allocations;
    // with the fix they're distinct nodes.
    let spirv = compile_to_spirv(
        r#"
def xs: [4]i32 = [1, -2, 3, -4]
def ys: [6]f32 = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]

def pos_i(x: i32) bool = x > 0
def pos_f(y: f32) bool = y > 0.0

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let a = filter(pos_i, xs) in
    let b = filter(pos_f, ys) in
    @[f32.i32(length(a)), f32.i32(length(b)), 0.0, 1.0]
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_filter_reachable_entry() {
    // Entry-point test: filter lowers to an OwnedView struct; length() and
    // indexing on the result work via variant dispatch.
    let spirv = compile_to_spirv(
        r#"
def is_positive(x: i32) bool = x > 0

def filter_demo(arr: [5]i32) = filter(is_positive, arr)

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let r = filter_demo([1, -2, 3, -4, 5]) in
    @[f32.i32(length(r)), f32.i32(r[0]), 0.0, 1.0]
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_filter_reachable_entry_glsl() {
    // Same program as test_filter_reachable_entry, compiled to GLSL. Exercises
    // the OwnedView struct emission + .valid_len / .buffer[i] dispatch.
    let out = compile_to_glsl(
        r#"
def is_positive(x: i32) bool = x > 0

def filter_demo(arr: [5]i32) = filter(is_positive, arr)

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let r = filter_demo([1, -2, 3, -4, 5]) in
    @[f32.i32(length(r)), f32.i32(r[0]), 0.0, 1.0]
"#,
    )
    .unwrap();
    let frag = out.fragment.expect("fragment shader");
    assert!(frag.contains("struct OwnedView"), "expected OwnedView struct: {}", frag);
    assert!(frag.contains(".valid_len"), "expected .valid_len access: {}", frag);
    assert!(frag.contains(".buffer["), "expected .buffer[] access: {}", frag);
}

#[test]
fn test_reduce_by_index_reachable_entry() {
    // Entry-point test: reduce_by_index is actually reached by the compiler pipeline.
    let spirv = compile_to_spirv(
        r#"
def hist_demo(dest: [3]i32, indices: [4]i32, values: [4]i32) [3]i32 =
    reduce_by_index(dest, |a: i32, b: i32| a + b, 0, indices, values)

#[fragment]
entry fragment_main(#[builtin(position)] pos: vec4f32) #[location(0)] vec4f32 =
    let dest: [3]i32 = [0, 0, 0] in
    let indices: [4]i32 = [0, 1, 0, 2] in
    let values: [4]i32 = [10, 20, 30, 40] in
    let r = hist_demo(dest, indices, values) in
    @[f32.i32(r[0]), f32.i32(r[1]), f32.i32(r[2]), 1.0]
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
fn test_reduce_with_tuple_destructuring() {
    // Test reduce with tuple pattern destructuring in the combiner.
    // This pattern from raytrace.wyn's findClosestHit was causing:
    // "Undefined global: _w_lambda_N" error because HOF specialization
    // was not eliminating function parameters correctly.
    let result = compile_to_spirv(
        r#"
def minPair(hits: [4](f32, i32)) (f32, i32) =
  reduce(|(t1, m1): (f32, i32), (t2, m2): (f32, i32)|
           if t1 < t2 then (t1, m1) else (t2, m2),
         (1000.0, 0),
         hits)

def testHits: [4](f32, i32) = [(1.0, 1), (2.0, 2), (0.5, 3), (3.0, 4)]

#[fragment]
entry fragment_main(#[builtin(frag_coord)] pos: vec4f32) #[location(0)] vec4f32 =
  let (t, m) = minPair(testHits) in
  @[t, t, 0.0, 1.0]
"#,
    );
    // Should compile without "Undefined global" error
    assert!(
        result.is_ok(),
        "Expected successful compilation, got: {:?}",
        result.err()
    );
    let spirv = result.unwrap();
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

/// Compile source to SPIR-V through the full pipeline including TLC partial_eval
fn compile_to_spirv_with_partial_eval(source: &str) -> Result<Vec<u32>> {
    let mut frontend = crate::cached_frontend();
    let parsed = crate::Compiler::parse(source, &mut frontend.node_counter).expect("Parsing failed");
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
    let ssa = alias_checked
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
        .elaborate();

    ssa.lower().map(|l| l.spirv)
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

#[test]
fn test_nested_if_else_in_entry_point() {
    // Regression test: nested if-else directly in entry point body
    // was causing NestedBlock error in SPIR-V builder.
    //
    // The issue: when an entry point has nested conditionals, the SSA
    // lowering's special handling for entry points (re-selecting the
    // "last block" for emitting OpReturn) was interfering with the
    // multi-block structure created by nested if-else.
    let result = compile_to_spirv(
        r#"
#[fragment]
entry fragment_main(#[builtin(position)] fragCoord: vec4f32) #[location(0)] vec4f32 =
  let x = fragCoord.x in
  if x < 0.5 then @[1.0, 0.0, 0.0, 1.0]
  else if x < 1.5 then @[0.0, 1.0, 0.0, 1.0]
  else @[0.0, 0.0, 1.0, 1.0]
"#,
    );
    assert!(
        result.is_ok(),
        "Nested if-else in entry point should compile. Got: {:?}",
        result.err()
    );
    let spirv = result.unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}
