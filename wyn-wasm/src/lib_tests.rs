//! Host-side unit tests for the playground interface builder.

use super::*;

/// Compile a Wyn source through the same pipeline used by
/// `compile_to_wgsl_impl` and return the SSA program so tests can inspect
/// the interface shape without going through JSON serialization.
fn compile_to_ssa(source: &str) -> wyn_core::ssa::stage::Elaborated {
    let (node_counter, module_manager) = wyn_core::init_compiler_with_options(
        wyn_core::CompilerOptions { graphics: true },
    )
    .expect("compiler initialization failed");
    let program = wyn_core::parser::parse(source, node_counter, module_manager).expect("parse failed");
    let program = wyn_core::resolve_imports::resolve_imports(program, std::path::Path::new("."))
        .expect("resolve_imports failed");
    let program =
        wyn_core::elaborate_modules::elaborate_modules(program).expect("elaborate_modules failed");
    let program = wyn_core::name_resolution::resolve_names(program);
    let program =
        wyn_core::resolve_resources::resolve_resources(program).expect("resolve_resources failed");
    let program = wyn_core::ast_const_fold::fold_constants(program);
    let program = wyn_core::resolve_placeholders::resolve_type_placeholders(program);
    let program = wyn_core::resolve_opens::resolve_opens(program).expect("resolve_opens failed");
    let program = wyn_core::types::run::type_check(program).expect("type_check failed");
    let program = wyn_core::ast_type_holes::reject_type_holes(program).expect("type holes");
    let program = wyn_core::tlc::lower_from_ast(program).expect("lower_from_ast");
    let program = wyn_core::tlc::pin_entry_buffers(program).expect("pin_entry_buffers");
    let program = wyn_core::tlc::validate_ownership(program).expect("validate_ownership");
    let program = wyn_core::tlc::partial_eval(program);
    let program = wyn_core::tlc::normalize_soacs(program);
    let program = wyn_core::tlc::monomorphize(program).expect("monomorphize");
    let program = wyn_core::tlc::rep_specialize(program);
    let program = wyn_core::tlc::inline_small(program);
    let program = wyn_core::tlc::force_inline_soac_helpers(program);
    let program = wyn_core::tlc::renormalize_inlined_soa(program);
    let program = wyn_core::tlc::canonicalize_conditional_producers(program);
    let program = wyn_core::tlc::normalize_soacs_to_anf(program);
    let program = wyn_core::tlc::float_runtime_index_nested_producers(program);
    let program = wyn_core::tlc::defunctionalize(program);
    let program = wyn_core::tlc::fold_generated_lambdas(program);
    let program = wyn_core::tlc::apply_ownership(program);
    let program = wyn_core::tlc::filter_reachable(program);
    let program = wyn_core::tlc::infer_input_slice_bounds(program);
    let program = wyn_core::to_egraph(program).expect("to_egraph failed");
    let program = wyn_core::egir::reify_soacs(program);
    let program = wyn_core::egir::optimize_semantic_operations(program)
        .expect("semantic EGIR optimization failed");
    let program = wyn_core::egir::lift_stage_uniform_values(program);
    let program = wyn_core::egir::plan_logical_resources(program).expect("semantic EGIR allocation failed");
    let program = wyn_core::egir::plan(
        program,
        wyn_core::LoweringProfile::new(wyn_core::CodegenTarget::Wgsl, wyn_core::SchedulePolicy::Parallel),
    )
    .expect("semantic EGIR planning failed");
    wyn_core::lower_egir_to_ssa(program).expect("planned EGIR lowering failed")
}

/// A fragment shader whose body contains a fragment-invariant reduce
/// gets an EGIR scalar materialization scheduled as a compute pre-pass. EGIR
/// lowers that into two compute entries (`phase1_chunks` +
/// `phase2_combine`), wires two compiler-allocated storage buffers
/// (partials + result), and rewrites the fragment body to load the
/// result scalar. The playground driver (`webgpu.ts`) needs the
/// `ProgramInterface` to surface all of this so it can: allocate the
/// storage buffers, build compute pipelines for the pre-passes, and
/// dispatch them before each render pass.
///
/// Every materialization storage binding appears in `interface.storage`, and
/// each compute entry exposes the `(set, binding)` coordinates it reads and
/// writes.
#[test]
fn interface_surfaces_materialization_storage_bindings() {
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(interface_surfaces_materialization_storage_bindings_impl)
        .expect("spawn test thread")
        .join()
        .expect("test thread panicked");
}

fn interface_surfaces_materialization_storage_bindings_impl() {
    let src = r#"
def vertex_main(vertex: vertex_invocation) vertex<vec2f32> =
  vertex_output(
    if vertex.vertex_index == 0u32 then @[-1.0, -1.0, 0.0, 1.0]
    else if vertex.vertex_index == 1u32 then @[3.0, -1.0, 0.0, 1.0]
    else @[-1.0, 3.0, 0.0, 1.0],
    @[0.0, 0.0])

def fragment_main(iTime: f32,
                  fragment: fragment_invocation<vec2f32>) vec4f32 =
  let samples = map(|i: i32| f32.cos(iTime + f32.i32(i)), 0..<64) in
  let breath = reduce(|a: f32, b: f32| a + b, 0.0, samples) in
  @[breath, 0.0, 0.0, 1.0]

entry image(iTime: f32,
            screen: render_target<vec4f32>) render_target<vec4f32> =
  let raster = rasterize_triangles(direct_draw(3u32, 1u32), vertex_main) in
  shade(screen, raster, |fragment| fragment_main(iTime, fragment))
"#;
    let program = compile_to_ssa(src);
    let iface = program_interface(&program);

    // The root scalar `iTime` (used inside the lifted reduce) must surface in
    // `interface.uniforms` for the driver to bind the uniform buffer — on
    // whichever entry ends up carrying it.
    assert!(
        iface.uniforms.iter().any(|u| u.name == "iTime"),
        "uniform iTime missing from interface.uniforms; got {:?}",
        iface.uniforms.iter().map(|u| (u.set, u.binding, u.name.clone())).collect::<Vec<_>>()
    );

    let kinds: Vec<&str> = iface.entries.iter().map(|e| e.kind.as_str()).collect();
    let compute_count = kinds.iter().filter(|k| **k == "compute").count();
    assert!(
        compute_count >= 2,
        "expected at least 2 compute pre-pass entries (phase1+phase2), got entries={:?}",
        iface.entries.iter().map(|e| (e.name.clone(), e.kind.clone())).collect::<Vec<_>>()
    );
    assert!(
        iface.entries.iter().any(|e| e.kind == "fragment"),
        "fragment entry missing"
    );

    // The lifted partials + result buffers must be visible at top level so
    // the driver can allocate them. There should be at least two
    // compiler-introduced storage bindings beyond any user-declared ones.
    assert!(
        iface.storage.len() >= 2,
        "expected ≥2 storage bindings (partials + result); got {:?}",
        iface.storage.iter().map(|s| (s.set, s.binding, s.name.clone())).collect::<Vec<_>>()
    );

    // Every compute pre-pass entry must expose the storage bindings it
    // touches via its inputs (the driver uses this to build bind groups).
    for entry in iface.entries.iter().filter(|e| e.kind == "compute") {
        let storage_inputs: Vec<&EntryBinding> =
            entry.inputs.iter().filter(|b| b.decoration.starts_with("storage(")).collect();
        assert!(
            !storage_inputs.is_empty(),
            "compute entry '{}' has no storage-binding inputs — driver has nothing \
             to bind. inputs={:?}",
            entry.name,
            entry.inputs.iter().map(|b| (b.name.clone(), b.decoration.clone())).collect::<Vec<_>>()
        );
    }

    // The fragment entry must expose the result-buffer binding it reads
    // (the lift rewrote the `breath` let-RHS into a storage load, so
    // without this binding the WGSL reference to `_buf_0_N` is dangling
    // from the driver's perspective).
    let fragment = iface.entries.iter().find(|e| e.kind == "fragment").unwrap();
    let fragment_storage: Vec<&EntryBinding> =
        fragment.inputs.iter().filter(|b| b.decoration.starts_with("storage(")).collect();
    assert!(
        !fragment_storage.is_empty(),
        "fragment entry '{}' loads the lifted pre-pass result but exposes no \
         storage binding in its inputs. inputs={:?}",
        fragment.name,
        fragment.inputs.iter().map(|b| (b.name.clone(), b.decoration.clone())).collect::<Vec<_>>()
    );
}
