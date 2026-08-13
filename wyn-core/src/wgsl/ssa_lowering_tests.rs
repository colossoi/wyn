//! WGSL backend unit tests.

use super::{validate_wgsl_identifier, wgsl_mangle, TypeEmitter};
use crate::ast::TypeName;
use polytype::Type as PolyType;

// ---------- wgsl_mangle: per-char cases ----------

#[test]
fn mangle_plain_passthrough() {
    assert_eq!(wgsl_mangle("foo"), "w_foo");
    assert_eq!(wgsl_mangle("a1b2c3"), "w_a1b2c3");
}

#[test]
fn mangle_dot() {
    assert_eq!(wgsl_mangle("a.b"), "w_a_Db");
    assert_eq!(
        wgsl_mangle("materials.pbrDistributionGGX"),
        "w_materials_DpbrDistributionGGX"
    );
}

#[test]
fn mangle_dollar() {
    assert_eq!(wgsl_mangle("foo$0"), "w_foo_S0");
}

#[test]
fn mangle_underscore_inside() {
    // `_` inside a name → `_U`.
    assert_eq!(wgsl_mangle("a_b"), "w_a_Ub");
}

#[test]
fn mangle_leading_underscore_contracts_prefix() {
    // Leading non-alnum char contracts the prefix from `w_` to `w` so we
    // never produce `__` at the prefix/body boundary.
    assert_eq!(wgsl_mangle("_foo"), "w_Ufoo");
    assert_eq!(wgsl_mangle("_w_intrinsic_foo"), "w_Uw_Uintrinsic_Ufoo");
}

#[test]
fn mangle_empty() {
    // Empty input: no leading char, so no prefix underscore. Prefix alone.
    assert_eq!(wgsl_mangle(""), "w");
}

#[test]
fn mangle_non_ascii_fallback() {
    // Non-ASCII char exercises the `_X<hex>_` fallback.
    assert_eq!(wgsl_mangle("a-b"), "w_a_X2d_b");
    assert_eq!(wgsl_mangle("-ab"), "w_X2d_ab");
}

#[test]
fn mangle_no_double_underscore() {
    // Invariant: no mangled output contains `__`. Spot-check a few edge
    // cases; the mangler's design prevents it structurally.
    for input in ["_w_lambda_13", "a__b", "foo_bar_baz", "_", "__", "a._b", "a.$b"] {
        let out = wgsl_mangle(input);
        assert!(
            !out.contains("__"),
            "mangle({:?}) produced {:?} which contains `__`",
            input,
            out
        );
    }
}

// ---------- validate_wgsl_identifier ----------

#[test]
fn validate_accepts_plain_ident() {
    assert!(validate_wgsl_identifier("iResolution").is_ok());
    assert!(validate_wgsl_identifier("my_buffer_0").is_ok());
    assert!(validate_wgsl_identifier("_leading_underscore_ok_for_host").is_ok());
}

#[test]
fn validate_rejects_empty() {
    assert!(validate_wgsl_identifier("").is_err());
}

#[test]
fn validate_rejects_digit_leading() {
    assert!(validate_wgsl_identifier("0invalid").is_err());
}

#[test]
fn validate_rejects_illegal_chars() {
    assert!(validate_wgsl_identifier("foo-bar").is_err());
    assert!(validate_wgsl_identifier("foo.bar").is_err());
}

#[test]
fn validate_rejects_reserved_keyword() {
    for kw in &["fn", "let", "var", "struct", "return", "loop"] {
        assert!(
            validate_wgsl_identifier(kw).is_err(),
            "keyword {} must be rejected",
            kw
        );
    }
}

#[test]
fn validate_rejects_reserved_type_name() {
    for ty in &["f32", "i32", "u32", "bool", "vec3", "mat4x4", "array"] {
        assert!(
            validate_wgsl_identifier(ty).is_err(),
            "type name {} must be rejected",
            ty
        );
    }
}

#[test]
fn validate_rejects_double_underscore_prefix() {
    // WGSL reserves `__...` for the implementation.
    assert!(validate_wgsl_identifier("__foo").is_err());
    assert!(validate_wgsl_identifier("_").is_err());
}

// ---------- type_to_wgsl ----------

fn scalar_ty(name: TypeName) -> PolyType<TypeName> {
    PolyType::Constructed(name, vec![])
}

#[test]
fn type_f32() {
    let mut e = TypeEmitter::new();
    assert_eq!(e.type_to_wgsl(&scalar_ty(TypeName::Float(32))).unwrap(), "f32");
}

#[test]
fn type_i32() {
    let mut e = TypeEmitter::new();
    assert_eq!(e.type_to_wgsl(&scalar_ty(TypeName::Int(32))).unwrap(), "i32");
}

#[test]
fn type_u32() {
    let mut e = TypeEmitter::new();
    assert_eq!(e.type_to_wgsl(&scalar_ty(TypeName::UInt(32))).unwrap(), "u32");
}

#[test]
fn type_bool() {
    let mut e = TypeEmitter::new();
    assert_eq!(e.type_to_wgsl(&scalar_ty(TypeName::Bool)).unwrap(), "bool");
}

#[test]
fn type_f64_rejected() {
    let mut e = TypeEmitter::new();
    let result = e.type_to_wgsl(&scalar_ty(TypeName::Float(64)));
    assert!(result.is_err(), "f64 must be rejected by WGSL type lowering");
}

#[test]
fn type_u64_requires_explicit_emulation_policy() {
    let ty = scalar_ty(TypeName::UInt(64));
    let mut default_emitter = TypeEmitter::new();
    assert!(default_emitter.type_to_wgsl(&ty).is_err());

    let mut emulating = TypeEmitter::with_options(crate::wgsl::WgslOptions::U64_EMULATION);
    assert_eq!(emulating.type_to_wgsl(&ty).unwrap(), "vec2<u32>");
}

#[test]
fn type_vec3f32() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Vec,
        vec![
            scalar_ty(TypeName::Float(32)),
            PolyType::Constructed(TypeName::Size(3), vec![]),
        ],
    );
    assert_eq!(e.type_to_wgsl(&ty).unwrap(), "vec3<f32>");
}

#[test]
fn type_mat4x4f32() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Mat,
        vec![
            scalar_ty(TypeName::Float(32)),
            PolyType::Constructed(TypeName::Size(4), vec![]),
            PolyType::Constructed(TypeName::Size(4), vec![]),
        ],
    );
    assert_eq!(e.type_to_wgsl(&ty).unwrap(), "mat4x4<f32>");
}

#[test]
fn type_array_sized() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Array,
        vec![
            scalar_ty(TypeName::Float(32)),
            PolyType::Constructed(TypeName::ArrayVariantComposite, vec![]),
            PolyType::Constructed(TypeName::Size(8), vec![]),
            crate::types::no_buffer(),
        ],
    );
    assert_eq!(e.type_to_wgsl(&ty).unwrap(), "array<f32, 8>");
}

#[test]
fn type_tuple_creates_struct() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Tuple(2),
        vec![scalar_ty(TypeName::Float(32)), scalar_ty(TypeName::Int(32))],
    );
    let name = e.type_to_wgsl(&ty).unwrap();
    assert_eq!(name, "T0");
    assert_eq!(
        e.tuple_structs.get("T0").unwrap(),
        &vec!["f32".to_string(), "i32".to_string()]
    );
}

#[test]
fn type_tuple_caches_by_signature() {
    let mut e = TypeEmitter::new();
    let ty = PolyType::Constructed(
        TypeName::Tuple(2),
        vec![scalar_ty(TypeName::Float(32)), scalar_ty(TypeName::Int(32))],
    );
    let n1 = e.type_to_wgsl(&ty).unwrap();
    let n2 = e.type_to_wgsl(&ty).unwrap();
    assert_eq!(n1, n2);
    assert_eq!(e.tuple_structs.len(), 1);
}

#[test]
fn type_tuple_distinct_shapes_distinct_structs() {
    let mut e = TypeEmitter::new();
    let ty_a = PolyType::Constructed(
        TypeName::Tuple(2),
        vec![scalar_ty(TypeName::Float(32)), scalar_ty(TypeName::Int(32))],
    );
    let ty_b = PolyType::Constructed(
        TypeName::Tuple(2),
        vec![scalar_ty(TypeName::Int(32)), scalar_ty(TypeName::Float(32))],
    );
    let na = e.type_to_wgsl(&ty_a).unwrap();
    let nb = e.type_to_wgsl(&ty_b).unwrap();
    assert_ne!(na, nb);
    assert_eq!(e.tuple_structs.len(), 2);
}

// ---------- end-to-end lowering ----------

#[test]
fn lower_empty_program_succeeds() {
    // Empty program: no functions, no entries — lower emits just the
    // header comment and returns a string.
    let program = crate::ssa::types::Program::<crate::ssa::stage::WgslReadyTag, _>::from_parts(
        Vec::new(),
        Vec::new(),
        Vec::new(),
        crate::ssa::context::BackendGlobal {
            pipeline: Default::default(),
            profile: crate::LoweringProfile::new(
                crate::CodegenTarget::Wgsl,
                crate::SchedulePolicy::Parallel,
            ),
            kernel_plan: Default::default(),
        },
    );
    let out = super::lower(&program).expect("empty program should lower");
    assert!(out.contains("WGSL backend"));
}

// ---------- naga-validated end-to-end ----------

/// Parse + validate WGSL text through naga. Panics with naga's diagnostic
/// message on failure so test output points directly at the offending
/// line. Used by end-to-end tests that compile a `.wyn` source through
/// the full pipeline to WGSL.
fn validate_wgsl(source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("naga parse failed:\n{}\n\n--- source ---\n{}", e, source));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("naga validation failed:\n{:?}\n\n--- source ---\n{}", e, source));
}

/// Compile a Wyn source through the full pipeline to WGSL text.
fn compile_to_wgsl(source: &str) -> crate::error::Result<String> {
    let program = crate::compile_thru_ssa(source).map_err(|e| crate::err_spirv!("{}", e))?;
    crate::lower_ssa_to_wgsl(program)
}

fn compile_to_wgsl_with_u64_emulation(source: &str) -> crate::error::Result<String> {
    let program = crate::compile_thru_ssa(source).map_err(|e| crate::err_spirv!("{}", e))?;
    crate::lower_ssa_to_wgsl_with_options(program, crate::wgsl::WgslOptions::U64_EMULATION)
}

#[test]
fn wgsl_u64_emulation_is_opt_in_and_naga_valid() {
    let source = r#"
def rotr16(x: u64) u64 =
  (x >> 16u64) | (x << 48u64)

entry rotate_and_add(xs: []u32) []u32 =
  map(|x: u32|
    let wide = u64.u32(x) + 81985529216486895u64
    let rotated = rotr16(wide) ^ 18446744073709551615u64
    let shifted = rotated >> u64.u32(x & 63u32) in
    if shifted != 0u64 then u32.u64(shifted) else 0u32,
    xs)
"#;

    let default_error = compile_to_wgsl(source).expect_err("default WGSL policy must reject u64");
    assert!(default_error.to_string().contains("64-bit scalars"));

    let wgsl = compile_to_wgsl_with_u64_emulation(source).expect("emulated u64 compile");
    validate_wgsl(&wgsl);
    assert!(wgsl.contains("fn _wyn_u64_add"));
    assert!(!wgsl.contains("fn _wyn_u64_shl"));
    assert!(wgsl.contains("fn _wyn_u64_shr"));
    assert!(wgsl.contains("vec2<u32>(2309737967u, 19088743u)"));
    assert!(
        !wgsl.lines().any(|line| {
            line.trim_start().starts_with("var v") && line.contains("vec2<u32>(2309737967u, 19088743u)")
        }),
        "residual typed literal should be substituted directly at its uses:\n{wgsl}"
    );
    assert!(wgsl.contains("any("));
}

#[test]
fn wgsl_fragment_trivial() {
    let wgsl = compile_to_wgsl(
        r#"
entry frame(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      if vertex.vertex_index == 0u32 then @[-0.5, -0.5, 0.0, 1.0]
      else if vertex.vertex_index == 1u32 then @[0.5, -0.5, 0.0, 1.0]
      else @[0.0, 0.5, 0.0, 1.0],
      @[0.0, 0.0, 0.0])) in
  shade(target, covered, |fragment| @[1.0, 0.5, 0.0, 1.0])
"#,
    )
    .expect("compile");
    validate_wgsl(&wgsl);
    assert!(wgsl.contains("@fragment"));
    assert!(wgsl.contains("@builtin(position)"));
    assert!(wgsl.contains("@location(0)"));
}
#[test]
fn wgsl_vertex_full_screen_triangle() {
    let wgsl = compile_to_wgsl(
        r#"
def verts: [3]vec4f32 =
  [@[-1.0, -1.0, 0.0, 1.0],
   @[ 3.0, -1.0, 0.0, 1.0],
   @[-1.0,  3.0, 0.0, 1.0]]

entry frame(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      verts[i32(vertex.vertex_index)],
      @[0.0, 0.0, 0.0])) in
  shade(target, covered, |fragment| @[1.0, 1.0, 1.0, 1.0])
"#,
    )
    .expect("compile");
    validate_wgsl(&wgsl);
    assert!(wgsl.contains("@vertex"));
    assert!(wgsl.contains("@builtin(vertex_index)"));
}
#[test]
fn wgsl_vertex_multi_output_struct() {
    let wgsl = compile_to_wgsl(
        r#"
entry frame(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      @[0.0, 0.0, 0.0, 1.0],
      @[1.0, 0.0, 0.0])) in
  shade(target, covered,
    |fragment| @[fragment.value.x, fragment.value.y, fragment.value.z, 1.0])
"#,
    )
    .expect("compile");
    validate_wgsl(&wgsl);
    assert!(wgsl.contains("struct VsOut0"));
    assert!(wgsl.contains("@builtin(position) f0: vec4<f32>,"));
    assert!(wgsl.contains("@location(0) f1: vec3<f32>,"));
    assert!(wgsl.contains("-> VsOut0"));
    assert!(wgsl.contains("var _out_struct: VsOut0;"));
    assert!(wgsl.contains("_out_struct.f0 ="));
    assert!(wgsl.contains("_out_struct.f1 ="));
    assert!(wgsl.contains("return _out_struct;"));
}
#[test]
fn wgsl_testfile_red_triangle() {
    validate_testfile_wgsl("testfiles/red_triangle.wyn");
}

#[test]
fn wgsl_testfile_red_triangle_curried() {
    validate_testfile_wgsl("testfiles/red_triangle_curried.wyn");
}

#[test]
fn wgsl_testfile_map_iota() {
    // Exercises ArrayRange lowering: `iota(10)` produces a virtual
    // array that's consumed by `map`. Virtual arrays lower to a
    // generated `VirtRange{N}` struct, indexed as start + i*step.
    validate_testfile_wgsl("testfiles/map_iota.wyn");
}

#[test]
fn wgsl_testfile_array_call_demo() {
    // Exercises `_w_intrinsic_slice` view-to-view lowering. `data[0..4]`
    // with `data: []f32` remains a storage-backed view and is passed to
    // a view-specialized user function.
    validate_testfile_wgsl("testfiles/array_call_demo.wyn");
}

#[test]
fn wgsl_testfile_pc_echo_test() {
    // Exercises push-constant-backed compute inputs — broadcast scalars
    // and small arrays — routed through a synthesized storage-read
    // block (WGSL uniform alignment would reject the array stride).
    validate_testfile_wgsl("testfiles/pc_echo_test.wyn");
}

#[test]
fn wgsl_testfile_reduce_compute() {
    // Exercises the function-scope hoist of SSA inst-result `var`
    // declarations. Without hoisting, a storage-view offset declared
    // inside the reduction loop would be out-of-scope at the post-loop
    // write site, which WGSL's textual block scoping rejects.
    validate_testfile_wgsl("testfiles/reduce_compute.wyn");
}

/// Compile a source file from disk through the full pipeline to WGSL
/// and naga-validate the result. Used for testfile sweeps. Resolves
/// paths relative to the workspace root so tests work regardless of
/// the crate under `-p`.
fn validate_testfile_wgsl(rel_path: &str) {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = format!("{}/../{}", manifest, rel_path);
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {}: {}", path, e));
    let wgsl =
        compile_to_wgsl(&src).unwrap_or_else(|e| panic!("wgsl compile failed for {}:\n{}", rel_path, e));
    validate_wgsl(&wgsl);
}

#[test]
fn wgsl_testfile_creation() {
    validate_testfile_wgsl("testfiles/playground/creation.wyn");
}

#[test]
fn wgsl_testfile_lava() {
    validate_testfile_wgsl("testfiles/playground/lava.wyn");
}

#[test]
fn wgsl_testfile_seascape() {
    validate_testfile_wgsl("testfiles/playground/seascape.wyn");
}

#[test]
fn wgsl_testfile_raytrace() {
    validate_testfile_wgsl("testfiles/playground/raytrace.wyn");
}

#[test]
fn wgsl_testfile_mandelbulb() {
    validate_testfile_wgsl("testfiles/playground/mandelbulb.wyn");
}

#[test]
fn wgsl_testfile_da_rasterizer() {
    validate_testfile_wgsl("testfiles/playground/da_rasterizer.wyn");
}

#[test]
fn wgsl_testfile_sum_demo() {
    // Structural sum types lowered into flattened tuples at the
    // AST→TLC boundary. Mixed-arity variants, including a nullary
    // case, exercise the dead-slot zero-fill path and the
    // tag-checked match dispatch.
    validate_testfile_wgsl("testfiles/sum_demo.wyn");
}

#[test]
fn wgsl_testfile_swizzle_with_demo() {
    // GLSL-style chained `dir.yz *= mat2` rotations expressed via
    // `with .swizzle *= m`. Lowers to let-bound vec rebuilds at
    // AST→TLC; SSA / WGSL never see VecWith.
    validate_testfile_wgsl("testfiles/swizzle_with_demo.wyn");
}

#[test]
fn wgsl_testfile_loopingspline() {
    validate_testfile_wgsl("testfiles/playground/loopingspline.wyn");
}

#[test]
fn wgsl_compiler_assigned_scalar_capture_emits_uniform_binding() {
    let wgsl = compile_to_wgsl(
        r#"
entry frame(i_time: f32,
            target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      @[0.0, 0.0, 0.0, 1.0],
      @[0.0, 0.0, 0.0])) in
  shade(target, covered, |fragment| @[i_time, 0.0, 0.0, 1.0])
"#,
    )
    .expect("compile");
    validate_wgsl(&wgsl);
    assert!(
        wgsl.contains("var<uniform>"),
        "expected a compiler-assigned uniform capture, got:\n{wgsl}"
    );
}
#[test]
fn wgsl_compute_reduce_writes_to_storage_buffer() {
    // A parallelized `reduce` compute shader's terminal write must hit
    // the storage buffer directly:
    //
    //     _buf_0_1[(i32(off) + i32(tid))] = v_accum;
    //
    // Writing only a local `var` that mirrors the buffer slot is a no-op at
    // runtime even though naga accepts it:
    //
    //     var v27_1: f32 = _buf_0_1[(i32(off) + i32(tid))];
    //     v27_1 = v_accum;
    let wgsl = compile_to_wgsl(
        r#"
entry sum_array(#[size_hint(1024)] data: []f32) f32 =
    reduce(|a: f32, b: f32| a + b, 0.0, data)
"#,
    )
    .expect("compile");
    validate_wgsl(&wgsl);

    // The bug pattern: `var vNN: f32 = _buf_...[...];` immediately
    // followed by `vNN = ...;` on the next non-blank line.
    let lines: Vec<&str> = wgsl.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("var ") {
            if let Some(eq_pos) = rest.find(" = ") {
                let name = &rest[..eq_pos].split(':').next().unwrap_or("").trim();
                let init = rest[eq_pos + 3..].trim_end_matches(';').trim();
                if init.starts_with("_buf_") && init.contains('[') {
                    // Look at the next non-blank line.
                    if let Some(next) = lines[i + 1..].iter().find(|l| !l.trim().is_empty()) {
                        let nt = next.trim();
                        let expected_bug = format!("{} = ", name);
                        assert!(
                            !nt.starts_with(&expected_bug),
                            "WGSL storage-write bug: `var {0}: ... = {1};` is followed by \
                             `{0} = ...;` — the write targets a dead local instead of \
                             the storage buffer.\n\n--- offending pair ---\n{2}\n{3}\n\n\
                             --- full WGSL ---\n{4}",
                            name,
                            init,
                            line,
                            next,
                            wgsl
                        );
                    }
                }
            }
        }
    }

    // Positive assertion: at least one direct storage-buffer write must
    // appear. The phase-1 terminal store writes the partial to
    // `_buf_0_1[offset + tid]`.
    assert!(
        wgsl.lines().any(|l| {
            let t = l.trim();
            t.starts_with("_buf_") && t.contains("] = ") && !t.contains(" = _buf_")
        }),
        "expected at least one direct `_buf_N_M[idx] = val;` write in emitted WGSL:\n{}",
        wgsl
    );
}

/// An `i32`-range reduce lowers to valid WGSL and provides the signed
/// counterpart to the `u32` case below.
#[test]
fn wgsl_i32_range_reduce_validates() {
    let wgsl = compile_to_wgsl(
        r#"
entry mn(n: i32) i32 =
  reduce(|a: i32, b: i32| if a < b then a else b, 2147483647, 0..<n)
"#,
    )
    .expect("compile");
    validate_wgsl(&wgsl);
}

/// A reduce over a u32 virtual range still uses an i32 loop counter.
/// `_w_intrinsic_length` must honor `wants_i32` by casting the range's
/// element-typed `.f2` field at the comparison site; each element is converted
/// back with `u32(idx)`.
#[test]
fn wgsl_u32_range_reduce_validates() {
    let wgsl = compile_to_wgsl(
        r#"
entry mn(n: u32) u32 =
  reduce(|a: u32, b: u32| if a < b then a else b, 4294967295u32, 0u32..<n)
"#,
    )
    .expect("compile");
    validate_wgsl(&wgsl);
}

/// A parallel consumer requests its range extent as `u32`, even when the
/// source range is element-typed as `i32`. The virtual-array length lowering
/// must bridge that signedness boundary explicitly for WGSL.
#[test]
fn wgsl_i32_range_filter_then_map_validates() {
    let wgsl = compile_to_wgsl(
        r#"
entry filtered() []i32 =
  let kept = filter(|i| i % 2 == 0, iota(64)) in
  map(|i| i + 1, kept)
"#,
    )
    .expect("compile");
    validate_wgsl(&wgsl);
}

#[test]
fn wgsl_compute_multi_output_runtime_sized_arrays() {
    // A compute entry returning a tuple of >1 runtime-sized array: each
    // field's producing `map` must stream into its own bound output storage
    // view.
    let wgsl = compile_to_wgsl(
        r#"
entry gen(src: []f32) ([]f32, []f32) =
    (map(|x: f32| x * 2.0, src), map(|x: f32| x * 3.0, src))
"#,
    )
    .expect("compile");
    validate_wgsl(&wgsl);

    // Two distinct output buffers, each written directly by its own map.
    assert!(
        wgsl.contains("_buf_0_1[") && wgsl.contains("] = "),
        "expected a direct write to output buffer 1:\n{wgsl}"
    );
    assert!(
        wgsl.contains("_buf_0_2[") && wgsl.contains("] = "),
        "expected a direct write to output buffer 2:\n{wgsl}"
    );
}

#[test]
fn wgsl_fragment_with_helper_function() {
    let wgsl = compile_to_wgsl(
        r#"
def brighten(c: vec4f32, amount: f32) vec4f32 =
  @[c.x + amount, c.y + amount, c.z + amount, c.w]

entry frame(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      @[0.0, 0.0, 0.0, 1.0],
      @[0.0, 0.0, 0.0])) in
  shade(target, covered,
    |fragment| brighten(@[0.1, 0.2, 0.3, 1.0], 0.5))
"#,
    )
    .expect("compile");
    validate_wgsl(&wgsl);
    assert!(wgsl.contains("@fragment"));
    assert!(wgsl.contains("@location(0)"));
    assert!(
        wgsl.lines().any(|line| line.trim_start().starts_with("let v")),
        "ordinary immutable SSA results should use let bindings:\n{wgsl}"
    );
}
#[test]
fn size_hint_large_bumps_workgroup_to_256() {
    // size_hint > 64K should pick a workgroup of 256 (per
    // `pick_workgroup_size`); the choice has to land on the shader's
    // `@workgroup_size` directive, not just the host descriptor.
    let wgsl = compile_to_wgsl(
        r#"
entry sum_array(#[size_hint(100000)] data: []f32) f32 =
    reduce(|a: f32, b: f32| a + b, 0.0, data)
"#,
    )
    .expect("compile");
    assert!(
        wgsl.contains("@workgroup_size(256, 1, 1)"),
        "size_hint(100000) should select workgroup_size=256 in the emitted WGSL, \
         got:\n{}",
        wgsl
    );
}

#[test]
fn size_hint_default_stays_workgroup_64() {
    // No hint → workgroup remains the default 64 (current behaviour).
    let wgsl = compile_to_wgsl(
        r#"
entry sum_array(data: []f32) f32 =
    reduce(|a: f32, b: f32| a + b, 0.0, data)
"#,
    )
    .expect("compile");
    assert!(
        wgsl.contains("@workgroup_size(64, 1, 1)"),
        "no size_hint should keep workgroup_size=64, got:\n{}",
        wgsl
    );
}

#[test]
fn wgsl_gather_computed_array() {
    // A randomly-indexed computed array is materialized into its own
    // storage buffer; the consumer reads it by index. The WGSL backend must
    // emit the gather buffer as a module-scope `var<storage>` and validate
    // (naga) end-to-end. Three buffers: input `bh` (0), consumer output (1),
    // gather intermediate (2).
    let wgsl = compile_to_wgsl(
        "\
entry gen(bh: []vec4f32) []i32 =
  let counts = map(|h:vec4f32| 4 + 5*(if h.x>4.0 then 3 else 1), bh) in
  map(|i:i32| counts[i % 256], iota(6144))
",
    )
    .expect("gather program must lower to WGSL");

    validate_wgsl(&wgsl);

    assert!(
        wgsl.contains("@group(0) @binding(2)") && wgsl.contains("var<storage"),
        "the gather buffer must be declared as a storage binding:\n{wgsl}"
    );
    // The producer and consumer share one physical pipeline, so the consumer
    // indexes the same read_write global selected by that pipeline layout.
    assert!(
        wgsl.contains("@group(0) @binding(2) var<storage, read_write> _buf_0_2:")
            && wgsl.contains("_buf_0_2["),
        "consumer must read the gather buffer by index:\n{wgsl}"
    );
}

/// `scatter` into a `#[storage(access=write)]` framebuffer: the destination
/// binding must be emitted `read_write` (WGSL requires it for the Store, and
/// the declared `access=write` must propagate from the param), and the scatter
/// emits indexed stores into it.
#[test]
fn wgsl_scatter_into_storage_buffer() {
    let source = r#"
def N:i32 = 5
entry rasterize(positions: []vec4f32,
                fb: *[]vec4f32) *[]vec4f32 =
  let pts  = positions[0..N] in
  let idxs = map(|p:vec4f32| i32.f32(p.y) * 512 + i32.f32(p.x), pts) in
  let vals = map(|p:vec4f32| @[1.0, 1.0, 1.0, 1.0], pts) in
  let updated = scatter(fb, idxs, vals) in
  map(|x: vec4f32| x, updated)
"#;
    let wgsl = compile_to_wgsl(source).expect("compile to WGSL");
    assert!(
        wgsl.contains("@group(0) @binding(1) var<storage, read_write>"),
        "scatter destination must be read_write:\n{wgsl}"
    );
    assert!(
        wgsl.contains("_buf_0_1["),
        "scatter must emit indexed stores into the framebuffer:\n{wgsl}"
    );
}

#[test]
fn wgsl_map_over_unique_storage_view_updates_backing_buffer() {
    let wgsl = compile_to_wgsl(
        r#"
entry draw(fb: *[]vec4f32) *[]vec4f32 =
  let cleared = map(|_p: vec4f32| @[0.0, 0.0, 0.0, 1.0], fb) in
  let idxs = map(|i: i32| i, 0i32 ..< 4i32) in
  let vals = map(|_i: i32| @[1.0, 1.0, 1.0, 1.0], 0i32 ..< 4i32) in
  scatter(cleared, idxs, vals)
"#,
    )
    .expect("map/scatter over a unique storage view must lower");

    validate_wgsl(&wgsl);
    assert!(
        wgsl.contains("_buf_0_1[") && wgsl.contains("] ="),
        "in-place view updates must target the backing storage buffer:\n{wgsl}"
    );
}

#[test]
fn wgsl_const_array_dynamic_index_hoists_to_private_global() {
    // A compile-time-constant array indexed by a runtime value is hoisted
    // once to a module-scope `var<private>` (the WGSL analog of the SPIR-V
    // Private-global hoist) and indexed there, instead of being copied into
    // a per-call `var<function>` materialization.
    let wgsl = compile_to_wgsl(
        "def t: [4]i32 = [10, 20, 30, 40]\n\
         \n\
         entry pick() []i32 = map(|i| t[i % 4], iota(100))",
    )
    .expect("compile");
    validate_wgsl(&wgsl);
    assert!(
        wgsl.contains("var<private> _const_global_0:"),
        "expected a hoisted private global:\n{wgsl}"
    );
    assert!(
        wgsl.contains("_const_global_0["),
        "the runtime index must address the global:\n{wgsl}"
    );
}

#[test]
fn wgsl_dynamic_index_reuses_addressable_fixed_array_values() {
    // Both the captured function parameter and the row extracted from it are
    // already backed by WGSL references. Materialize must alias those values
    // instead of producing full-array copies before each dynamic index.
    let wgsl = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            compile_to_wgsl(
                r#"
def expand(root: [2]u32, table: [1024][2]u32) [2]u32 =
  map(|i: i32|
    let reference = root[i]
    in if i == 0i32
       then table[i32.u32(reference)][0]
       else table[i32.u32(reference)][1],
    iota(2))

entry pick(roots: [1]([2]u32), table: [1024][2]u32) [1]([2]u32) =
  map(|i: i32| expand(roots[i], table), iota(1))
"#,
            )
        })
        .expect("spawn fixed-array regression compiler thread")
        .join()
        .expect("fixed-array regression compiler thread panicked")
        .expect("compile dynamic indexing of captured fixed arrays");

    validate_wgsl(&wgsl);
    assert!(
        wgsl.contains("w_roots[w_i]"),
        "the dynamic index should address the captured parameter directly:\n{wgsl}"
    );
    assert!(
        !wgsl.lines().any(|line| {
            line.trim_start().starts_with("var ") && line.contains("array<") && line.contains(" = w_roots;")
        }),
        "Materialize must not copy the captured fixed array parameter:\n{wgsl}"
    );
}

#[test]
fn wgsl_distinctness_indexes_512_element_parameter_without_copying_it() {
    let wgsl = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            compile_to_wgsl(
                r#"
def indices_distinct(indices: [512]u32) bool =
  loop unique = true for i < 512 do
    let duplicate = loop found = false for j < i do
      found || indices[i] == indices[j]
    in unique && !duplicate

entry check(inputs: [1]([512]u32)) [1]u32 =
  map(|i: i32|
    if indices_distinct(inputs[i]) then 1u32 else 0u32,
    iota(1))
"#,
            )
        })
        .expect("spawn distinctness regression compiler thread")
        .join()
        .expect("distinctness regression compiler thread panicked")
        .expect("compile 512-element distinctness check");

    validate_wgsl(&wgsl);
    let start = wgsl
        .find("fn w_indices_Udistinct")
        .unwrap_or_else(|| panic!("expected a distinctness helper:\n{wgsl}"));
    let helper_tail = &wgsl[start..];
    let end = helper_tail
        .find("\n}\n")
        .unwrap_or_else(|| panic!("expected the end of the distinctness helper:\n{helper_tail}"));
    let helper = &helper_tail[..end];
    assert!(
        helper.contains("w_indices["),
        "distinctness should index its parameter directly:\n{helper}"
    );
    assert!(
        !helper.lines().any(|line| line.contains("array<u32, 512> = w_indices;")),
        "distinctness must not copy its 512-element parameter:\n{helper}"
    );
}

#[test]
fn wgsl_const_array_hoist_is_deduped() {
    // The same constant array indexed at two sites shares one global.
    let wgsl = compile_to_wgsl(
        "def t: [4]i32 = [10, 20, 30, 40]\n\
         \n\
         entry pick() []i32 = map(|i| t[i % 4] + t[(i + 1) % 4], iota(100))",
    )
    .expect("compile");
    validate_wgsl(&wgsl);
    let n = wgsl.matches("var<private> _const_global").count();
    assert_eq!(
        n, 1,
        "two indexings of one constant must share one global:\n{wgsl}"
    );
}

#[test]
fn wgsl_composite_constant_inlines_transitive_constant_dependencies() {
    let wgsl = compile_to_wgsl(
        r#"
type word64 = (u32, u32)

def WORDS: [2]word64 = [
  (0x11223344u32, 0x55667788u32),
  (0x99aabbccu32, 0xddeeff00u32)
]

def COPIED_WORDS: [2]word64 = [WORDS[0], WORDS[1]]

def select_word(words: [2]word64, index: i32) word64 = words[index]

entry composite_constant(index: u32) word64 =
  select_word(COPIED_WORDS, i32.u32(index & 1u32))
"#,
    )
    .expect("compile a composite constant with a constant dependency");

    validate_wgsl(&wgsl);
    assert!(
        !wgsl.contains("w_WORDS"),
        "transitive constants must be expanded before WGSL lowering:\n{wgsl}"
    );
}

#[test]
fn wgsl_structural_capture_emits_uniform_struct_and_validates() {
    let wgsl = compile_to_wgsl(
        r#"
type block = { radius: f32, tint: vec2f32, center: vec2f32 }

entry frame(c: block,
            target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      @[0.0, 0.0, 0.0, 1.0],
      @[0.0, 0.0, 0.0])) in
  shade(target, covered,
    |fragment| @[c.tint.x, c.tint.y, c.radius + c.center.x, 1.0])
"#,
    )
    .expect("compile");
    validate_wgsl(&wgsl);
    assert!(wgsl.contains("struct T0"));
    assert!(
        wgsl.contains("var<uniform>"),
        "expected a compiler-assigned structural uniform capture, got:\n{wgsl}"
    );
}
#[test]
fn wgsl_duplicate_source_parameter_names_are_uniquified() {
    let wgsl = compile_to_wgsl(
        r#"
def advance(p: i32, i: i32, t: f32) i32 = p + i + i32.f32(t)

entry tick(prev: []i32, t: f32) []i32 =
  if t < 0.1 then
    map(|i: i32| i, 0i32 ..< 16i32)
  else
    let ps = prev[0i32..16i32] in
    map(|i: i32| advance(ps[i], i, t), 0i32 ..< 16i32)
"#,
    )
    .expect("conditional map envelope must lower");

    validate_wgsl(&wgsl);
    assert!(
        wgsl.contains("w_i__p1: i32"),
        "duplicate source parameter names must be uniquified:\n{wgsl}"
    );
}
