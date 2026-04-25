//! Host-side unit tests for the playground interface builder.

use super::*;

/// Compile a Wyn source through the same pipeline used by
/// `compile_to_wgsl_impl` and return the SSA program so tests can inspect
/// the interface shape without going through JSON serialization.
fn compile_to_ssa(source: &str) -> wyn_core::ssa::types::Program {
    let mut frontend = wyn_core::FrontEnd::new();
    let parsed = wyn_core::Compiler::parse(source, &mut frontend.node_counter).expect("parse failed");
    let parsed = parsed.elaborate_modules(&mut frontend.module_manager).expect("elaborate_modules failed");
    let alias_checked = parsed
        .desugar(&mut frontend.node_counter)
        .expect("desugar failed")
        .resolve(&mut frontend.module_manager)
        .expect("resolve failed")
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .expect("type_check failed")
        .alias_check()
        .expect("alias_check failed");
    let ssa = alias_checked
        .to_tlc(&frontend.schemes, &frontend.module_manager, false)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .parallelize_soacs(false)
        .filter_reachable()
        .to_egraph()
        .expect("to_egraph failed")
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate();
    ssa.ssa
}

/// A fragment shader whose body contains a fragment-invariant reduce
/// gets its reduce lifted into a compute pre-pass. The lift pass
/// multi-stages that into two compute entries (`phase1_chunks` +
/// `phase2_combine`), wires two compiler-allocated storage buffers
/// (partials + result), and rewrites the fragment body to load the
/// result scalar. The playground driver (`webgpu.ts`) needs the
/// `ProgramInterface` to surface all of this so it can: allocate the
/// storage buffers, build compute pipelines for the pre-passes, and
/// dispatch them before each render pass.
///
/// Regression guard: every lifted storage binding must appear in
/// `interface.storage`, and each compute entry's inputs must expose
/// the `(set, binding)` coordinates it reads/writes.
#[test]
fn interface_surfaces_lifted_prepass_storage_bindings() {
    let src = r#"
#[uniform(set=0, binding=0)] def iTime: f32

#[vertex]
entry vertex_main(#[builtin(vertex_index)] vid: i32)
  #[builtin(position)] vec4f32 =
  @[-1.0, -1.0, 0.0, 1.0]

#[fragment]
entry fragment_main(#[builtin(position)] fragCoord: vec4f32)
  #[location(0)] vec4f32 =
  let samples = map(|i: i32| f32.cos(iTime + f32.i32(i)), 0..<64) in
  let breath = reduce(|a: f32, b: f32| a + b, 0.0, samples) in
  @[breath, 0.0, 0.0, 1.0]
"#;
    let program = compile_to_ssa(src);
    let iface = program_interface(&program);

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
