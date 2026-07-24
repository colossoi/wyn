//! Host-side unit tests for the playground interface builder.

use super::*;

/// Compile a Wyn source through the same pipeline used by
/// `compile_to_wgsl_impl` and return the SSA program so tests can inspect
/// the interface shape without going through JSON serialization.
fn compile_to_ssa(source: &str) -> wyn_core::ssa::types::Program {
    let (mut node_counter, mut module_manager) =
        wyn_core::init_compiler().expect("compiler initialization failed");
    let parsed = wyn_core::Compiler::parse(source, &mut node_counter).expect("parse failed");
    let parsed =
        parsed.elaborate_modules(&mut module_manager, &mut node_counter).expect("elaborate_modules failed");
    let type_checked = parsed
        .resolve(&module_manager)
        .expect("resolve failed")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check failed");
    let program = type_checked.to_tlc(&module_manager, false);
    let program = wyn_core::tlc::pin_entry_buffers(program).expect("pin_entry_buffers");
    let program = wyn_core::tlc::validate_ownership(program).expect("validate_ownership");
    let program = wyn_core::tlc::partial_eval(program);
    let program = wyn_core::tlc::normalize_soacs(program);
    let program = wyn_core::tlc::monomorphize(program);
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
    let ssa = wyn_core::to_egraph(program)
        .expect("to_egraph failed")
        .realize_outputs()
        .expect("realize_outputs failed")
        .segment()
        .optimize()
        .allocate()
        .expect("semantic EGIR allocation failed")
        .plan(wyn_core::LoweringProfile::new(
            wyn_core::CodegenTarget::Wgsl,
            wyn_core::SchedulePolicy::Parallel,
        ))
        .expect("semantic EGIR planning failed")
        .lower_to_ssa()
        .expect("planned EGIR lowering failed");
    ssa.ssa
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
/// Regression guard: every materialization storage binding must appear in
/// `interface.storage`, and each compute entry's inputs must expose
/// the `(set, binding)` coordinates it reads/writes.
#[test]
fn interface_surfaces_materialization_storage_bindings() {
    let src = r#"
#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32)
  #[builtin(position)] vec4f32 =
  @[-1.0, -1.0, 0.0, 1.0]

#[fragment]
entry fragment_main(
  #[uniform(set=1, binding=0)] iTime: f32,
  #[builtin(position)] fragCoord: vec4f32
) #[target(screen)] vec4f32 =
  let samples = map(|i: i32| f32.cos(iTime + f32.i32(i)), 0..<64) in
  let breath = reduce(|a: f32, b: f32| a + b, 0.0, samples) in
  @[breath, 0.0, 0.0, 1.0]
"#;
    let program = compile_to_ssa(src);
    let iface = program_interface(&program);

    // The `#[uniform(set=1, binding=0)]` fragment param `iTime` (used inside
    // the lifted reduce) must surface in `interface.uniforms` for the driver
    // to bind the uniform buffer — on whichever entry ends up carrying it.
    assert!(
        iface.uniforms.iter().any(|u| u.set == 1 && u.binding == 0),
        "uniform iTime (set=1, binding=0) missing from interface.uniforms; got {:?}",
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
