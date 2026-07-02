// ============================================================================
// Tests
// ============================================================================

use super::{run, Converter};
use crate::ast::TypeName;
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::ssa::types::{FuncBody, InstKind, Program};
use crate::tlc::VarRef;
use crate::tlc::{Term, TermKind};
use crate::SymbolId;
use crate::SymbolTable;
use polytype::Type;
use std::collections::{HashMap, HashSet};

/// Compile a source string through the full TLC pipeline, then convert
/// through the full EGIR chain (`from_tlc → expand_soacs → optimize_skeleton
/// → elaborate`) to a `Program`. No `materialize` — tests don't exercise
/// SPIR-V-specific dynamic-index rewrites.
fn compile_via_egir(src: &str) -> Program {
    let tlc = crate::test_pipeline::compile_to_reachable(src, false);

    let bounds = crate::tlc::input_slice_bounds::compute_for_program(&tlc.tlc);
    let mut binding_ids = crate::IdSource::<u32>::new();
    crate::EgirRaw(
        run(&tlc.tlc, PipelineDescriptor::default(), &bounds, &mut binding_ids)
            .expect("egir::from_tlc conversion failed"),
    )
    .realize_outputs()
    .expect("egir::realize_outputs failed")
    .segment()
    .optimize()
    .allocate(&binding_ids)
    .lower_to_ssa(crate::LoweringProfile::PORTABLE)
    .expect("semantic EGIR lowering failed")
    .ssa
}

use crate::ast::Span;
use crate::tlc::TermIdSource;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn mk_term(ty: Type<TypeName>, kind: TermKind) -> Term {
    Term {
        id: TermIdSource::new().next_id(),
        ty,
        span: Span::dummy(),
        kind,
    }
}

/// Build a minimal TLC def and convert it via EGraph.
fn convert_simple_def(body: Term, params: Vec<(SymbolId, Type<TypeName>)>) -> FuncBody {
    let symbols = SymbolTable::new();
    let top_level = HashMap::new();
    let constants_by_name = HashMap::new();
    let pure_constants = HashSet::new();

    let ret_ty = body.ty.clone();
    let param_info: Vec<(Type<TypeName>, String)> =
        params.iter().enumerate().map(|(i, (_, ty))| (ty.clone(), format!("p{}", i))).collect();

    let mut binding_ids = crate::IdSource::<u32>::new();
    let region_interner = std::cell::RefCell::new(crate::egir::program::RegionInterner::default());
    let mut converter = Converter::new(
        &top_level,
        &constants_by_name,
        &symbols,
        pure_constants,
        &mut binding_ids,
        &region_interner,
    );
    for (i, (sym, ty)) in params.iter().enumerate() {
        let nid = converter.graph.add_func_param(i, ty.clone());
        converter.locals.insert(*sym, nid);
    }
    let result = converter.convert_term(&body).expect("conversion failed");
    converter.set_return(Some(result));
    converter.elaborate_to_funcbody(&param_info, ret_ty).expect("elaboration failed")
}

#[test]
fn test_int_literal_roundtrip() {
    let body = mk_term(i32_ty(), TermKind::IntLit("42".into()));
    let func = convert_simple_def(body, vec![]);
    let entry = func.get_block(func.entry_block());
    // Should have one Int instruction.
    assert!(entry.insts.iter().any(|&iid| {
        matches!(&func.get_inst(iid).data, InstKind::Op { tag: crate::op::OpTag::Int(s), .. } if s == "42")
    }));
}

#[test]
fn test_add_roundtrip() {
    let mut symbols = SymbolTable::new();
    let a_sym = symbols.alloc("a".into());
    let b_sym = symbols.alloc("b".into());

    // Build: a + b
    let a_var = mk_term(i32_ty(), TermKind::Var(VarRef::Symbol(a_sym)));
    let b_var = mk_term(i32_ty(), TermKind::Var(VarRef::Symbol(b_sym)));
    let add_op = mk_term(
        i32_ty(), // simplified — real type would be arrow
        TermKind::BinOp(crate::ast::BinaryOp { op: "+".into() }),
    );
    let app = mk_term(
        i32_ty(),
        TermKind::App {
            func: Box::new(add_op),
            args: vec![a_var, b_var],
        },
    );

    let top_level = HashMap::new();
    let constants_by_name = HashMap::new();
    let pure_constants = HashSet::new();

    let mut binding_ids = crate::IdSource::<u32>::new();
    let region_interner = std::cell::RefCell::new(crate::egir::program::RegionInterner::default());
    let mut converter = Converter::new(
        &top_level,
        &constants_by_name,
        &symbols,
        pure_constants,
        &mut binding_ids,
        &region_interner,
    );
    let a_nid = converter.graph.add_func_param(0, i32_ty());
    converter.locals.insert(a_sym, a_nid);
    let b_nid = converter.graph.add_func_param(1, i32_ty());
    converter.locals.insert(b_sym, b_nid);

    let result = converter.convert_term(&app).expect("conversion failed");
    converter.set_return(Some(result));

    let params = vec![(i32_ty(), "a".into()), (i32_ty(), "b".into())];
    let func = converter.elaborate_to_funcbody(&params, i32_ty()).expect("elaboration failed");

    let entry = func.get_block(func.entry_block());
    // Should have a BinOp(+) instruction.
    assert!(
        entry
            .insts
            .iter()
            .any(|&iid| { matches!(&func.get_inst(iid).data, InstKind::Op { tag: crate::op::OpTag::BinOp(op), .. } if op == "+") })
    );
}

#[test]
fn test_gvn_via_let() {
    // let x = 42 in let y = 42 in (x, y)
    // GVN should deduplicate the two 42 constants into a single node.
    // (A `+` would be constant-folded to `84`, erasing the evidence.)
    use polytype::Type;
    let pair_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), i32_ty()]);

    let lit42 = mk_term(i32_ty(), TermKind::IntLit("42".into()));
    let lit42b = mk_term(i32_ty(), TermKind::IntLit("42".into()));

    let mut symbols = SymbolTable::new();
    let x_sym = symbols.alloc("x".into());
    let y_sym = symbols.alloc("y".into());

    let x_ref = mk_term(i32_ty(), TermKind::Var(VarRef::Symbol(x_sym)));
    let y_ref = mk_term(i32_ty(), TermKind::Var(VarRef::Symbol(y_sym)));
    let pair_app = mk_term(pair_ty.clone(), TermKind::Tuple(vec![x_ref, y_ref]));

    let inner_let = mk_term(
        pair_ty.clone(),
        TermKind::Let {
            name: y_sym,
            name_ty: i32_ty(),
            rhs: Box::new(lit42b),
            body: Box::new(pair_app),
        },
    );
    let outer_let = mk_term(
        pair_ty.clone(),
        TermKind::Let {
            name: x_sym,
            name_ty: i32_ty(),
            rhs: Box::new(lit42),
            body: Box::new(inner_let),
        },
    );

    let top_level = HashMap::new();
    let constants_by_name = HashMap::new();
    let pure_constants = HashSet::new();

    let mut binding_ids = crate::IdSource::<u32>::new();
    let region_interner = std::cell::RefCell::new(crate::egir::program::RegionInterner::default());
    let mut converter = Converter::new(
        &top_level,
        &constants_by_name,
        &symbols,
        pure_constants,
        &mut binding_ids,
        &region_interner,
    );
    let result = converter.convert_term(&outer_let).expect("conversion failed");
    converter.set_return(Some(result));

    let func = converter.elaborate_to_funcbody(&[], pair_ty).expect("elaboration failed");

    let entry = func.get_block(func.entry_block());
    // GVN: should have only ONE Int("42") instruction, not two.
    let const_count = entry
        .insts
        .iter()
        .filter(|&&iid| matches!(&func.get_inst(iid).data, InstKind::Op { tag: crate::op::OpTag::Int(s), .. } if s == "42"))
        .count();
    assert_eq!(
        const_count, 1,
        "GVN should deduplicate: found {} copies of 42",
        const_count
    );
}

#[test]
fn test_hash_cons_distinguishes_by_result_type() {
    // Interning the same intrinsic with the same operands but different
    // result types must produce distinct NodeIds — otherwise the
    // first-inserted type silently wins at the merged node. Extends the
    // 3b8cb24 Int/Uint-literal split to cover every pure op. Regression
    // for the conway.wyn `_w_intrinsic_storage_len` i32/u32 collision.
    use crate::egir::types::{EGraph, PureOp};
    use smallvec::smallvec;

    let mut g = EGraph::new();
    let i32_ty = i32_ty();
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);

    let zero_u32 = g.intern_pure(
        PureOp::Uint("0".into()),
        smallvec::SmallVec::new(),
        u32_ty.clone(),
    );

    let storage_len_id = crate::builtins::catalog().known().storage_len;
    let a = g.intern_pure(
        PureOp::Intrinsic {
            id: storage_len_id,
            overload_idx: 0,
        },
        smallvec![zero_u32, zero_u32],
        i32_ty,
    );
    let b = g.intern_pure(
        PureOp::Intrinsic {
            id: storage_len_id,
            overload_idx: 0,
        },
        smallvec![zero_u32, zero_u32],
        u32_ty,
    );
    assert_ne!(
        a, b,
        "different result types must not hash-cons to the same NodeId"
    );
}

#[test]
fn test_if_else_roundtrip() {
    // if cond then 1 else 0
    let mut symbols = SymbolTable::new();
    let c_sym = symbols.alloc("c".into());
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    let cond = mk_term(bool_ty.clone(), TermKind::Var(VarRef::Symbol(c_sym)));
    let then_br = mk_term(i32_ty(), TermKind::IntLit("1".into()));
    let else_br = mk_term(i32_ty(), TermKind::IntLit("0".into()));
    let if_term = mk_term(
        i32_ty(),
        TermKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_br),
            else_branch: Box::new(else_br),
        },
    );

    let top_level = HashMap::new();
    let constants_by_name = HashMap::new();
    let pure_constants = HashSet::new();

    let mut binding_ids = crate::IdSource::<u32>::new();
    let region_interner = std::cell::RefCell::new(crate::egir::program::RegionInterner::default());
    let mut converter = Converter::new(
        &top_level,
        &constants_by_name,
        &symbols,
        pure_constants,
        &mut binding_ids,
        &region_interner,
    );
    let c_nid = converter.graph.add_func_param(0, bool_ty);
    converter.locals.insert(c_sym, c_nid);

    let result = converter.convert_term(&if_term).expect("conversion failed");
    converter.set_return(Some(result));

    let params = vec![(Type::Constructed(TypeName::Bool, vec![]), "c".into())];
    let func = converter.elaborate_to_funcbody(&params, i32_ty()).expect("elaboration failed");

    // Should have 4 blocks: entry, then, else, merge
    assert_eq!(func.inner.blocks.len(), 4, "if/else should produce 4 blocks");

    // Entry should end with CondBranch
    let entry = func.get_block(func.entry_block());
    assert!(
        matches!(&entry.term, crate::ssa::framework::Terminator::CondBranch { .. }),
        "Entry should end with CondBranch, got {:?}",
        entry.term
    );
}

// ====================================================================
// Full pipeline integration tests
// ====================================================================

#[test]
fn test_full_pipeline_simple() {
    let program = compile_via_egir(
        r#"
def add(a: i32, b: i32) i32 = a + b

#[vertex]
entry main() #[builtin(position)] vec4f32 =
    let x = add(1, 2) in
    @[f32.i32(x), 0.0, 0.0, 1.0]
"#,
    );
    // 'add' may be inlined by TLC passes — just verify the program is valid
    assert!(!program.entry_points.is_empty(), "Should have entry points");
}

#[test]
fn test_full_pipeline_if_else() {
    let program = compile_via_egir(
        r#"
def pick(c: bool, a: i32, b: i32) i32 = if c then a else b

#[vertex]
entry main() #[builtin(position)] vec4f32 =
    let x = pick(true, 1, 2) in
    @[f32.i32(x), 0.0, 0.0, 1.0]
"#,
    );
    // 'pick' may be inlined — just verify compilation succeeds
    assert!(!program.entry_points.is_empty(), "Should have entry points");
}

#[test]
fn test_full_pipeline_loop() {
    let program = compile_via_egir(
        r#"
def sum_to(n: i32) i32 =
    loop acc = 0 for i < n do acc + i

#[vertex]
entry main() #[builtin(position)] vec4f32 =
    let x = sum_to(10) in
    @[f32.i32(x), 0.0, 0.0, 1.0]
"#,
    );
    // 'sum_to' may be inlined — just verify compilation succeeds
    assert!(!program.entry_points.is_empty(), "Should have entry points");
}

#[test]
fn test_filter_compiles_end_to_end() {
    // Exercises the EGIR Filter path:
    //   * surface `filter(...)` is reachable (not eliminated by
    //     partial_eval / inline_small for this shape),
    //   * `convert_soac_filter` rewrites the existential `?k. [k]T`
    //     result to `Array[T, Size(N), Bounded]`,
    //   * `expand_one`'s Filter arm builds the loop with a Selection
    //     inside the loop body,
    //   * `length()` projects the runtime count from member 1 of the
    //     resulting Bounded struct.
    let program = compile_via_egir(
        r#"
def is_even(x: i32) bool = x % 2 == 0

def evens(arr: [4]i32) ?k. [k]i32 =
    filter(is_even, arr)

#[vertex]
entry vertex_main() #[builtin(position)] vec4f32 =
    let e = evens([1, 2, 3, 4]) in
    @[f32.i32(length(e)), 0.0, 0.0, 1.0]
"#,
    );
    assert!(!program.entry_points.is_empty(), "Should have entry points");
}

// --- vertex_inputs population from #[location(n)] params ---------------

#[test]
fn vertex_inputs_populated_from_location_params() {
    use crate::pipeline_descriptor::{Pipeline, VertexFormat};

    let src = "#[vertex]\n\
               entry vs(#[location(0)] position: vec3f32, #[location(1)] color: vec3f32)\n\
                 (#[builtin(position)] vec4f32, #[location(0)] vec3f32) =\n\
                 (@[position.x, position.y, position.z, 1.0], color)\n\
               #[fragment]\n\
               entry fs(#[location(0)] color: vec3f32) #[location(0)] vec4f32 =\n\
                 @[color.x, color.y, color.z, 1.0]";
    let converted = crate::compile_thru_ssa(src).expect("compile thru ssa");

    let find = |name: &str| {
        converted.pipeline.pipelines.iter().find_map(|p| match p {
            Pipeline::Graphics(gp) if gp.stages.iter().any(|s| s.entry_point == name) => Some(gp),
            _ => None,
        })
    };

    // Vertex stage: both #[location] params surface as vertex_inputs.
    let vs = find("vs").expect("vertex graphics pipeline");
    assert_eq!(vs.vertex_inputs.len(), 2);
    assert_eq!(vs.vertex_inputs[0].location, 0);
    assert_eq!(vs.vertex_inputs[0].name, "position");
    assert_eq!(vs.vertex_inputs[0].format, VertexFormat::Float32x3);
    assert_eq!(vs.vertex_inputs[1].location, 1);
    assert_eq!(vs.vertex_inputs[1].name, "color");
    assert_eq!(vs.vertex_inputs[1].format, VertexFormat::Float32x3);

    // Fragment stage: #[location] params there are varyings, not vertex
    // buffers — vertex_inputs stays empty.
    let fs = find("fs").expect("fragment graphics pipeline");
    assert!(fs.vertex_inputs.is_empty());
}

// --- fragment_outputs population from #[location(n)] returns ------------

#[test]
fn fragment_outputs_populated_from_location_returns() {
    use crate::pipeline_descriptor::Pipeline;

    let src = "#[vertex]\n\
               entry vs(#[location(0)] position: vec3f32)\n\
                 #[builtin(position)] vec4f32 =\n\
                 @[position.x, position.y, position.z, 1.0]\n\
               #[fragment]\n\
               entry fs() #[location(0)] vec4f32 =\n\
                 @[1.0, 0.0, 0.0, 1.0]";
    let converted = crate::compile_thru_ssa(src).expect("compile thru ssa");

    let fs = converted
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Graphics(gp) if gp.stages.iter().any(|s| s.entry_point == "fs") => Some(gp),
            _ => None,
        })
        .expect("fragment graphics pipeline");

    // The single #[location(0)] return surfaces as one fragment output.
    assert_eq!(fs.fragment_outputs.len(), 1);
    assert_eq!(fs.fragment_outputs[0].location, 0);
    assert_eq!(fs.fragment_outputs[0].name, "fs_output");
}

// --- vertex shaders may read #[uniform] params -------------------------

#[test]
fn vertex_uniform_param_compiles_and_surfaces_binding() {
    use crate::pipeline_descriptor::{Binding, Pipeline};

    // A vertex shader reading a uniform is standard; the type checker
    // must not reject the `#[uniform]` param as a missing vertex
    // attribute, and the binding must surface in the descriptor.
    let src = "#[vertex]\n\
               entry vs(#[location(0)] position: vec3f32,\n\
                        #[uniform(set=1, binding=0)] scale: f32)\n\
                 #[builtin(position)] vec4f32 =\n\
                 @[position.x * scale, position.y * scale, position.z * scale, 1.0]\n\
               #[fragment]\n\
               entry fs() #[location(0)] vec4f32 = @[1.0, 0.0, 0.0, 1.0]";
    let converted = crate::compile_thru_ssa(src).expect("compile thru ssa");

    let vs = converted
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Graphics(gp) if gp.stages.iter().any(|s| s.entry_point == "vs") => Some(gp),
            _ => None,
        })
        .expect("vertex graphics pipeline");

    // `position` surfaces as a vertex input; `scale` as a uniform binding.
    assert_eq!(vs.vertex_inputs.len(), 1);
    assert_eq!(vs.vertex_inputs[0].name, "position");
    assert!(
        vs.bindings.iter().any(|b| matches!(
            b,
            Binding::Uniform { set: 1, binding: 0, name, .. } if name == "scale"
        )),
        "vertex uniform `scale` should surface as a binding, got {:?}",
        vs.bindings
    );
}

// --- texture + sampler bindings surface in the descriptor --------------

#[test]
fn texture_and_sampler_params_surface_bindings() {
    use crate::pipeline_descriptor::{Binding, Pipeline};

    let src = "#[vertex]\n\
               entry vs(#[location(0)] pos: vec2f32)\n\
                 (#[builtin(position)] vec4f32, #[location(0)] vec2f32) =\n\
                 (@[pos.x, pos.y, 0.0, 1.0], pos)\n\
               #[fragment]\n\
               entry fs( #[location(0)] uv: vec2f32\n\
                       , #[texture(set=0, binding=0)] tex: texture2d\n\
                       , #[sampler(set=0, binding=1)] samp: sampler\n\
                       ) #[location(0)] vec4f32 =\n\
                 texture_sample(tex, samp, uv, 0.0) + texture_load(tex, @[0, 0], 0)";
    let converted = crate::compile_thru_ssa(src).expect("compile thru ssa");

    let fs = converted
        .pipeline
        .pipelines
        .iter()
        .find_map(|p| match p {
            Pipeline::Graphics(gp) if gp.stages.iter().any(|s| s.entry_point == "fs") => Some(gp),
            _ => None,
        })
        .expect("fragment graphics pipeline");

    assert!(
        fs.bindings.iter().any(|b| matches!(
            b,
            Binding::Texture { set: 0, binding: 0, name, .. } if name == "tex"
        )),
        "texture `tex` should surface as a Texture binding, got {:?}",
        fs.bindings
    );
    assert!(
        fs.bindings.iter().any(|b| matches!(
            b,
            Binding::Sampler { set: 0, binding: 1, name, .. } if name == "samp"
        )),
        "sampler `samp` should surface as a Sampler binding, got {:?}",
        fs.bindings
    );
}

/// Graphics entries must NOT derive `EgirEntry.return_ty` from
/// `def.ty`'s arrow-return position — that's the post-`normalize_outputs`
/// shortcut for compute entries, and graphics entries don't go through
/// `normalize_outputs`. The prior contract used `inner_body.ty`, which
/// matches the body's actual produced shape after ownership /
/// monomorphize / etc. Forcing the def.ty-derived form here makes any
/// future divergence (e.g. a uniqueness wrapper that ownership doesn't
/// strip from `def.ty` for non-compute entries) silently change how
/// `build_entry_outputs` classifies the return — a tuple wrapped in
/// `Unique` no longer matches the `Tuple` arm, and an N-output entry
/// gets collapsed to a single output.
///
/// This test constructs the divergence directly by mutating the graphics
/// entry's `def.ty` after type-checking. With the guard (`if is_compute
/// { sig_ret } else { inner_body.ty.clone() }`) the conversion still
/// reads the unmodified `inner_body.ty` and produces two outputs; without
/// the guard the wrapped `def.ty` propagates and produces one.
#[test]
fn graphics_entry_ret_type_comes_from_inner_body_not_def_ty() {
    use crate::tlc::DefMeta;

    let src = r#"
#[vertex]
entry vertex_main(#[location(0)] position: vec3f32, #[location(1)] color: vec3f32)
  (#[builtin(position)] vec4f32, #[location(0)] vec3f32) =
  (@[position.x, position.y, position.z, 1.0], color)
"#;
    let tlc = crate::test_pipeline::compile_to_reachable(src, false);

    let mut tlc_program = tlc.tlc.clone();

    // Mutate the vertex entry's `def.ty` arrow-return position to wrap
    // the result tuple in `TypeName::Unique`. `inner_body.ty` stays
    // unchanged. This synthesises the divergence that a future ownership
    // / lowering change could produce naturally.
    let def = tlc_program
        .defs
        .iter_mut()
        .find(|d| tlc_program.symbols.get(d.name).map(|n| n == "vertex_main").unwrap_or(false))
        .expect("vertex_main def");
    assert!(
        matches!(&def.meta, DefMeta::EntryPoint(e) if !e.entry_type.is_compute()),
        "precondition: vertex_main is a graphics entry"
    );
    wrap_arrow_return_in_unique(&mut def.ty);

    let bounds = crate::tlc::input_slice_bounds::compute_for_program(&tlc_program);
    let mut binding_ids = crate::IdSource::<u32>::new();
    let egir = super::run(
        &tlc_program,
        PipelineDescriptor::default(),
        &bounds,
        &mut binding_ids,
    )
    .expect("from_tlc::run on graphics entry must succeed");
    let entry = egir.entry_points.iter().find(|e| e.name == "vertex_main").expect("vertex_main EgirEntry");

    assert_eq!(
        entry.outputs.len(),
        2,
        "graphics entry's outputs must be derived from inner_body.ty (preserved as a tuple) \
         — not from def.ty (mutated to Unique-wrap the tuple). got {:?}",
        entry.outputs.iter().map(|o| &o.ty).collect::<Vec<_>>()
    );
}

/// Walk an arrow chain `P1 -> P2 -> ... -> Pn -> R` and replace `R` with
/// `Unique(R)`. Used to synthesise a divergence between `def.ty`'s
/// arrow-return position and `inner_body.ty`.
fn wrap_arrow_return_in_unique(mut ty: &mut Type<TypeName>) {
    loop {
        let inner = match ty {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => &mut args[1],
            _ => break,
        };
        if !matches!(inner, Type::Constructed(TypeName::Arrow, _)) {
            let old = std::mem::replace(inner, Type::Constructed(TypeName::Unit, vec![]));
            *inner = Type::Constructed(TypeName::Unique, vec![old]);
            return;
        }
        ty = inner;
    }
}

/// Correctness risk #2 — terminal lowering of a parallel scan synthesizes a
/// swap-wrapper region (`{entry}_scan_op_swap`). That region is interned during
/// `lower`, not present in the pre-lowering arena, and `soac_expand` recovers
/// its SSA `Call` name through the interner. If the synthesized region were not
/// interned, name recovery would panic. Compiling a parallel scan to SSA drives
/// that path end to end.
#[test]
fn parallel_scan_synthesized_swap_region_is_name_recoverable() {
    let src = "#[compute] entry prefix(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)";
    crate::compile_thru_ssa(src)
        .expect("parallel scan lowers, recovering its synthesized swap-wrapper region");
}
