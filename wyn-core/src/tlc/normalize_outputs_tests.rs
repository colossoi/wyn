use crate::Compiler;
use crate::ast::TypeName;
use crate::tlc::{Program, Term, TermKind};
use polytype::Type;

/// Compile a Wyn source string up to (and including) `normalize_outputs`,
/// returning the resulting TLC `Program`. Used by tests that need to
/// inspect post-normalize_outputs term structure directly.
fn compile_to_normalized_tlc(source: &str) -> Program {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = Compiler::parse(source, &mut node_counter).expect("parse");
    let type_checked = parsed
        .resolve(&mut module_manager)
        .expect("resolve")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("type_check");
    let normalized = type_checked
        .to_tlc(&module_manager, false)
        .pin_entry_regions()
        .expect("pin_entry_regions")
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .normalize_outputs()
        .expect("normalize_outputs");
    normalized.0.tlc
}

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
        .pin_entry_regions()
        .expect("pin_entry_regions")
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
        .fold_generated_lambdas()
        .inline_small()
        .rep_specialize()
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

/// Walk a Term tree and assert the TLC invariant `let.ty == let.body.ty`
/// and `lambda.ret_ty == lambda.body.ty` at every layer. Panics on
/// mismatch, reporting the offending node's kind.
fn assert_let_lambda_type_invariants(term: &Term) {
    match &term.kind {
        TermKind::Lambda(lam) => {
            assert_eq!(lam.ret_ty, lam.body.ty, "Lambda.ret_ty must match Lambda.body.ty",);
            assert_let_lambda_type_invariants(&lam.body);
        }
        TermKind::Let { body, .. } => {
            assert_eq!(term.ty, body.ty, "Let.ty must match Let.body.ty");
            assert_let_lambda_type_invariants(body);
        }
        _ => {}
    }
}

/// Multi-output entry whose tail is NOT a `Tuple` literal — e.g. a
/// `reduce` returning a tuple. `emit_slot_writes` falls through and
/// returns the tail unchanged with its tuple type. The enclosing
/// `rewrite_body` Lambda/Let wrappers must reflect that type, NOT
/// force `Unit` unconditionally.
///
/// Regression: prior to the fix, both arms set `ret_ty: unit_ty` /
/// `ty: Unit` regardless of the rewritten body's actual type, so a
/// `Lambda` whose body is tuple-typed claimed `ret_ty: Unit`.
#[test]
fn multi_output_non_tuple_tail_preserves_lambda_let_types() {
    let source = r#"
def pair_min(a: (i32, i32), b: (i32, i32)) (i32, i32) =
  if a.0 < b.0 then a else b

#[compute]
entry foo(xs: []i32) (i32, i32) =
  reduce(pair_min, (0, 0), map(|x: i32| (x, x), xs))
"#;
    let program = compile_to_normalized_tlc(source);
    let entry = program
        .defs
        .iter()
        .find(|d| program.symbols.get(d.name).map(|n| n == "foo").unwrap_or(false))
        .expect("foo entry");
    let unit_ty = Type::Constructed(TypeName::Unit, vec![]);
    assert_ne!(
        entry.body.ty, unit_ty,
        "precondition: this test only exercises the multi-output non-Tuple \
         fallthrough; if the outer Lambda's ty is Unit, the fallthrough \
         didn't fire and the test is moot"
    );
    assert_let_lambda_type_invariants(&entry.body);
}

/// After `normalize_outputs`, a compute entry whose body decomposes into
/// `OutputSlotStore` writes has no runtime return value — slot writes
/// are the work. The body's tail must therefore type as `SideEffect`
/// (not `Unit`, which is a value), and `def.ty == def.body.ty` so both
/// views of the def's signature agree.
#[test]
fn normalize_outputs_yields_side_effect_typed_body() {
    let source = r#"
#[compute]
entry foo(xs: []i32) (i32, i32) =
    (xs[0], xs[1])
"#;
    let program = compile_to_normalized_tlc(source);
    let entry = program
        .defs
        .iter()
        .find(|d| program.symbols.get(d.name).map(|n| n == "foo").unwrap_or(false))
        .expect("foo entry");
    assert_eq!(
        entry.ty, entry.body.ty,
        "def.ty must equal def.body.ty after normalize_outputs"
    );
    let (_params, ret) = crate::types::extract_function_signature(&entry.ty);
    assert_eq!(
        ret,
        Type::Constructed(TypeName::SideEffect, vec![]),
        "normalized entry's return position must be SideEffect, not Unit"
    );
}
