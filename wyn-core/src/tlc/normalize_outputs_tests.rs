use crate::Compiler;

/// Compile a Wyn source string through the full pipeline (parse →
/// type-check → TLC → EGIR → SSA → SPIR-V) and return the SPIR-V words.
/// Panics on any pipeline error — intended for tests that expect the
/// source to compile cleanly.
fn compile_to_spirv(source: &str) -> Vec<u32> {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = Compiler::parse(source, &mut node_counter).expect("parse");
    let type_checked = parsed
        .resolve(&mut module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");

    let ssa = type_checked
        .to_tlc(&module_manager, false)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .normalize_outputs()
        .expect("normalize_outputs")
        .lift_gathers()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .parallelize_soacs(false)
        .expect("parallelize_soacs")
        .filter_reachable()
        .to_egraph()
        .expect("egraph")
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate();

    ssa.lower().expect("lower").spirv
}

/// True iff the SPIR-V module contains at least one instruction with the
/// given opcode. SPIR-V instruction encoding: high 16 bits = word count,
/// low 16 bits = opcode. The first 5 words are the module header.
fn spirv_contains_opcode(spirv: &[u32], opcode: u32) -> bool {
    spirv.iter().skip(5).any(|w| (w & 0xFFFF) == opcode)
}

/// Compute entries that declare unit return (`()`) with a side-effectful
/// tail (e.g. `image_store(img, xy, color)`) must preserve the tail
/// through normalize_outputs. The pass has no slot writes to emit for a
/// zero-output entry, but discarding the tail silently drops the
/// imperative builtin's effect.
///
/// Regression: prior to the fix, `emit_slot_writes`'s `n_outputs == 0`
/// branch returned a fresh `UnitLit`, dropping the `image_store` App
/// entirely; the compiled SPIR-V contained no `OpImageWrite` instruction.
#[test]
fn unit_return_compute_entry_preserves_image_store() {
    let source = r#"
#[compute]
entry paint(#[storage_image(set=0, binding=0, format=rgba8unorm, access=write_only)] img: storage_image,
            #[builtin(global_invocation_id)] gid: vec3u32) () =
  let xy = @[i32.u32(gid.x), i32.u32(gid.y)] in
  image_store(img, xy, @[1.0, 0.0, 0.0, 1.0])
"#;

    let spirv = compile_to_spirv(source);

    const OP_IMAGE_WRITE: u32 = 99;
    assert!(
        spirv_contains_opcode(&spirv, OP_IMAGE_WRITE),
        "expected OpImageWrite in compiled SPIR-V (regression: normalize_outputs dropping unit-return tail)",
    );
}
