use crate::error::Result;

/// Compile a source string to SPIR-V words. Thin wrapper around the
/// canonical `compile_thru_spirv` so test failures lift through `.unwrap()`.
fn compile_to_spirv(source: &str) -> Result<Vec<u32>> {
    crate::compile_thru_spirv(source).map(|l| l.spirv).map_err(|e| crate::err_spirv!("{}", e))
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
fn test_if_literal_true_lowers() {
    // `fold_constant_branches` leaves the dead arm in `skeleton.blocks`
    // without a predecessor. Dominator analysis must ignore that arm when
    // deciding whether to emit the live path's merge block.
    let spirv = compile_to_spirv("def f(x) = if true then x + 1 else x + 2").unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_if_literal_false_lowers() {
    let spirv = compile_to_spirv("def f(x) = if false then x + 1 else x + 2").unwrap();
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
fn test_reduce_with_tuple_destructuring() {
    // HOF specialization must eliminate the tuple-destructuring combiner's
    // function parameters before SPIR-V lowering.
    let result = compile_to_spirv(
        r#"
def minPair(hits: [4](f32, i32)) (f32, i32) =
  reduce(|(t1, m1): (f32, i32), (t2, m2): (f32, i32)|
           if t1 < t2 then (t1, m1) else (t2, m2),
         (1000.0, 0),
         hits)

def testHits: [4](f32, i32) = [(1.0, 1), (2.0, 2), (0.5, 3), (3.0, 4)]

#[fragment]
entry fragment_main(#[builtin(frag_coord)] pos: vec4f32) #[target(screen)] vec4f32 =
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

#[test]
fn test_partial_eval_inlined_function_local_id_collision() {
    // Partial evaluation of an inlined helper with an unknown let RHS must
    // allocate locals independently of the caller's local map.
    let spirv = compile_to_spirv(
        r#"
-- Helper function that will be inlined when called with known args.
-- The let binding 'weight' uses iTime which is unknown, so it gets residualized.
def helper(x: f32, iTime: f32) f32 =
    let weight = x * iTime in
    weight

#[fragment]
entry fragment_main(#[uniform(set=1, binding=0)] iTime: f32, #[builtin(position)] pos: vec4f32) #[target(screen)] vec4f32 =
    -- Create multiple locals to ensure LocalId collision
    let a = pos.x in
    let b = pos.y in
    let c = pos.z in
    -- Call helper with known arg - this gets inlined.
    -- helper's 'weight' local may collide with our locals.
    let d = helper(3.0, iTime) in
    @[d, a, b, c]
"#,
    )
    .unwrap();
    assert!(!spirv.is_empty());
    assert_eq!(spirv[0], 0x07230203);
}

#[test]
fn test_partial_eval_intrinsic_arg_types() {
    // Residualizing dot(vec3, vec3) -> f32 must reify each known argument with
    // its vec3 type rather than the scalar result type.
    let spirv = compile_to_spirv(
        r#"
def verts: [3]vec4f32 =
  [@[-1.0, -1.0, 0.0, 1.0],
   @[3.0, -1.0, 0.0, 1.0],
   @[-1.0, 3.0, 0.0, 1.0]]

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vertex_id: i32) #[builtin(position)] vec4f32 = verts[vertex_id]

#[fragment]
entry fragment_main() #[target(screen)] vec4f32 =
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
    // Entry-point return emission must preserve the multi-block structure of
    // nested conditionals.
    let result = compile_to_spirv(
        r#"
#[fragment]
entry fragment_main(#[builtin(position)] fragCoord: vec4f32) #[target(screen)] vec4f32 =
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

#[test]
fn test_two_compute_entries_share_one_global_invocation_id() {
    // VUID-StandaloneSpirv-OpEntryPoint-09658 permits exactly one module-level
    // GlobalInvocationId input variable. Count its BuiltIn decoration to
    // ensure multiple compute entries share that variable.
    let src = "\
def verts: [3]vec4f32 =
  [@[0.0 - 1.0, 0.0 - 1.0, 0.0, 1.0],
   @[3.0, 0.0 - 1.0, 0.0, 1.0],
   @[0.0 - 1.0, 3.0, 0.0, 1.0]]

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32)
  #[builtin(position)] vec4f32 = verts[vid]

#[compute]
entry a(#[builtin(global_invocation_id)] gid: vec3u32) () = ()

#[compute]
entry b(#[builtin(global_invocation_id)] gid: vec3u32) () = ()

#[fragment]
entry fragment_main(#[builtin(position)] _p: vec4f32)
  #[target(screen)] vec4f32 = @[0.0, 0.0, 0.0, 1.0]
";
    let spirv = compile_to_spirv(src).expect("two-compute-entry shader should compile");
    // BuiltIn enum value for GlobalInvocationId is 28
    // (spirv::BuiltIn::GlobalInvocationId as u32).
    let global_invocation_id_builtin = spirv::BuiltIn::GlobalInvocationId as u32;
    // OpDecorate is opcode 71 with word_count 4 for BuiltIn:
    // [opcode|word_count<<16, target_id, Decoration::BuiltIn(11),
    //  BuiltIn_enum_value]. Scan for that pattern.
    const OP_DECORATE: u32 = 71;
    const DECORATION_BUILTIN: u32 = 11;
    let mut count = 0;
    let mut i = 5; // skip 5-word header
    while i + 3 < spirv.len() {
        let word = spirv[i];
        let opcode = word & 0xFFFF;
        let word_count = (word >> 16) as usize;
        if opcode == OP_DECORATE
            && word_count == 4
            && spirv[i + 2] == DECORATION_BUILTIN
            && spirv[i + 3] == global_invocation_id_builtin
        {
            count += 1;
        }
        if word_count == 0 {
            break;
        }
        i += word_count;
    }
    assert_eq!(
        count, 1,
        "expected exactly one `OpDecorate ... BuiltIn GlobalInvocationId`, got {count}"
    );
}

/// `scatter` into a `#[storage]` framebuffer lowers end-to-end: the full
/// `SoacKind::Scatter` → `SoacOp::Scatter` → `EgirSoac::Hist` →
/// `build_scatter_loop` path emits indexed `OpStore`s into the destination
/// view (one per scattered element; N=5 here, unrolled).
#[test]
fn scatter_into_storage_buffer_lowers() {
    let spirv = compile_to_spirv(
        r#"
def N:i32 = 5
#[compute]
entry rasterize(#[storage(set=2, binding=0, access=read)] positions: []vec4f32,
                #[storage(set=2, binding=1, access=write)] fb: []vec4f32) () =
  let pts  = positions[0..N] in
  let idxs = map(|p:vec4f32| i32.f32(p.y) * 512 + i32.f32(p.x), pts) in
  let vals = map(|p:vec4f32| @[1.0, 1.0, 1.0, 1.0], pts) in
  let _ = scatter(fb, idxs, vals) in ()
"#,
    )
    .expect("scatter rasterizer must lower to SPIR-V");
    assert_eq!(spirv[0], 0x07230203, "SPIR-V magic number");
    const OP_STORE: u32 = 62;
    let stores = spirv.iter().skip(5).filter(|w| (*w & 0xFFFF) == OP_STORE).count();
    assert!(
        stores >= 5,
        "expected >= 5 OpStore (one per scattered particle), got {stores}"
    );
}

/// Disassemble SPIR-V words to text for instruction-level assertions.
fn disasm(words: &[u32]) -> String {
    use wspirv::binary::Disassemble;
    wspirv::dr::load_words(words).expect("backend emitted valid SPIR-V").disassemble()
}

/// Lines per `(set, binding)` storage class help locate hoisted globals: a
/// compile-time-constant array indexed by a runtime value is materialized once
/// into a module-scope `Private` global (initializer = the constant), not
/// `OpStore`d wholesale into a per-occurrence `Function` array variable.
#[test]
fn dynamic_const_array_index_hoists_to_private_global() {
    let spirv = compile_to_spirv(
        "def t: [4]i32 = [10, 20, 30, 40]\n\
         #[compute]\n\
         entry pick() []i32 = map(|i| t[i % 4], iota(100))",
    )
    .unwrap();
    let text = disasm(&spirv);
    let private_vars: Vec<&str> =
        text.lines().filter(|l| l.contains("OpVariable") && l.contains(" Private ")).collect();
    assert_eq!(
        private_vars.len(),
        1,
        "expected one hoisted Private global:\n{text}"
    );
    // The Private global carries a constant initializer (trailing operand).
    let last = private_vars[0].trim_end().rsplit(' ').next().unwrap();
    assert!(
        last.starts_with('%'),
        "Private global must have an initializer: {}",
        private_vars[0]
    );
    // The constant is not also materialized into a Function-storage array.
    assert!(
        !text.contains("_ptr_Function__arr"),
        "constant array must not be stored into a Function variable:\n{text}"
    );
}

/// The hoist is deduped by constant value: the same constant array indexed at
/// two sites collapses to a single `Private` global.
#[test]
fn const_array_hoist_is_deduped() {
    let spirv = compile_to_spirv(
        "def t: [4]i32 = [10, 20, 30, 40]\n\
         #[compute]\n\
         entry pick() []i32 = map(|i| t[i % 4] + t[(i + 1) % 4], iota(100))",
    )
    .unwrap();
    let text = disasm(&spirv);
    let n = text.lines().filter(|l| l.contains("OpVariable") && l.contains(" Private ")).count();
    assert_eq!(
        n, 1,
        "two indexings of one constant must share one Private global:\n{text}"
    );
}
