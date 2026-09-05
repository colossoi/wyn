// ============================================================================
// Tests
// ============================================================================

use super::{convert_program, ConversionPlan, Converter};
use crate::ast;
use crate::ast::TypeName;
use crate::builtins;
use crate::compile_thru_ssa;
use crate::egir;
use crate::egir::types::{callable_parameter, Parameters, WynLanguage};
use crate::interface;
use crate::lower_egir_to_ssa;
use crate::op;
use crate::ssa;
use crate::ssa::types::{ConstantValue, FuncBody, InstKind, ValueRef};
use crate::test_pipeline;
use crate::tlc;
use crate::tlc::data::{ExplicitCapturesPayload, ExplicitClosurePayload};
use crate::tlc::VarRef;
use crate::tlc::{Term, TermKind};
use crate::types;
use crate::LoweringProfile;
use crate::SymbolTable;
use crate::{BindingRef, SymbolId};
use polytype::Type;
use wyn_base::IdSource;

/// Compile a source string through the full TLC pipeline, then convert
/// through the full EGIR chain (`from_tlc → expand_soacs → optimize_skeleton
/// → elaborate`) to a `Program`. No `materialize` — tests don't exercise
/// SPIR-V-specific dynamic-index rewrites.
fn compile_via_egir(src: &str) -> ssa::stage::Elaborated {
    let tlc = tlc::infer_input_slice_bounds(test_pipeline::compile_to_reachable(src));
    let program = convert_program(&tlc, IdSource::<u32>::new(), IdSource::new())
        .expect("egir::from_tlc conversion failed");
    let program = egir::reify_soacs(program);
    let program = egir::optimize_semantic_operations(program).expect("semantic EGIR optimization failed");
    let program = egir::lift_stage_uniform_values(program);
    let program = egir::plan_logical_resources(program).expect("semantic EGIR allocation failed");
    let program = egir::plan(program, LoweringProfile::PORTABLE).expect("semantic EGIR planning failed");
    lower_egir_to_ssa(program).expect("semantic EGIR lowering failed")
}

use crate::ast::Span;
use crate::tlc::TermIdSource;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn mk_term(
    ids: &mut TermIdSource,
    ty: Type<TypeName>,
    kind: TermKind<ExplicitClosurePayload, ExplicitCapturesPayload>,
) -> Term<ExplicitClosurePayload, ExplicitCapturesPayload> {
    Term {
        id: ids.next_id(),
        ty,
        span: Span::generated(),
        kind,
    }
}

fn elaborate_converter(
    converter: Converter<'_, '_>,
    params: &Parameters<BindingRef, Type<TypeName>>,
    return_ty: Type<TypeName>,
) -> FuncBody {
    let graph = converter.into_graph();
    let (graph, _, _) = graph
        .try_map_resources_and_phase::<egir::types::Physical, String>(
            |resource| {
                Err(format!(
                    "unit-test graph unexpectedly references resource {resource:?}"
                ))
            },
            |_, _, _, _| Err("unit-test graph unexpectedly contains an unexpanded SOAC".into()),
        )
        .expect("unit-test graph should be directly physicalizable");
    egir::elaborate::elaborate_one_body(graph, params, return_ty)
}

/// Build a minimal TLC def and convert it via EGraph.
fn convert_simple_def(
    body: Term<ExplicitClosurePayload, ExplicitCapturesPayload>,
    params: Vec<(SymbolId, Type<TypeName>)>,
) -> FuncBody {
    let symbols = SymbolTable::new();

    let ret_ty = body.ty.clone();
    let param_info = params
        .iter()
        .enumerate()
        .map(|(i, (_, ty))| callable_parameter::<BindingRef, WynLanguage>(format!("p{i}"), ty.clone()))
        .collect::<Parameters<_, _>>();

    let mut binding_ids = IdSource::<u32>::new();
    let mut effect_ids = IdSource::new();
    let plan = ConversionPlan::empty();
    let mut converter = Converter::new(&symbols, &mut binding_ids, &mut effect_ids, &plan);
    for ((sym, ty), parameter) in params.iter().zip(param_info.ids()) {
        let nid = converter.graph.add_test_value_parameter(parameter, ty.clone());
        converter.locals.insert(*sym, nid);
    }
    let result = converter.convert_term(&body).expect("conversion failed");
    converter.set_return(Some(converter.graph.value_result(result)));
    elaborate_converter(converter, &param_info, ret_ty)
}

#[test]
fn test_int_literal_roundtrip() {
    let mut term_ids = TermIdSource::new();
    let body = mk_term(&mut term_ids, i32_ty(), TermKind::IntLit("42".into()));
    let func = convert_simple_def(body, vec![]);
    let entry = func.get_block(func.entry_block());
    assert!(
        entry.insts.is_empty(),
        "a representable literal needs no SSA instruction"
    );
    assert!(matches!(
        entry.term,
        ssa::framework::Terminator::Return(Some(ValueRef::Const(ConstantValue::I32(42))))
    ));
}

#[test]
fn test_add_roundtrip() {
    let mut symbols = SymbolTable::new();
    let mut term_ids = TermIdSource::new();
    let a_sym = symbols.alloc("a".into());
    let b_sym = symbols.alloc("b".into());

    // Build: a + b
    let a_var = mk_term(&mut term_ids, i32_ty(), TermKind::Var(VarRef::Symbol(a_sym)));
    let b_var = mk_term(&mut term_ids, i32_ty(), TermKind::Var(VarRef::Symbol(b_sym)));
    let add_op = mk_term(
        &mut term_ids,
        i32_ty(), // simplified — real type would be arrow
        TermKind::BinOp(ast::BinaryOp {
            op: op::BinaryOperator::Add,
        }),
    );
    let app = mk_term(
        &mut term_ids,
        i32_ty(),
        TermKind::App {
            func: Box::new(add_op),
            args: vec![a_var, b_var],
        },
    );

    let mut binding_ids = IdSource::<u32>::new();
    let mut effect_ids = IdSource::new();
    let plan = ConversionPlan::empty();
    let mut converter = Converter::new(&symbols, &mut binding_ids, &mut effect_ids, &plan);
    let params = [
        callable_parameter::<BindingRef, WynLanguage>("a".into(), i32_ty()),
        callable_parameter::<BindingRef, WynLanguage>("b".into(), i32_ty()),
    ]
    .into_iter()
    .collect::<Parameters<_, _>>();
    let parameter_ids = params.ids().collect::<Vec<_>>();
    let a_nid = converter.graph.add_test_value_parameter(parameter_ids[0], i32_ty());
    converter.locals.insert(a_sym, a_nid);
    let b_nid = converter.graph.add_test_value_parameter(parameter_ids[1], i32_ty());
    converter.locals.insert(b_sym, b_nid);

    let result = converter.convert_term(&app).expect("conversion failed");
    converter.set_return(Some(converter.graph.value_result(result)));

    let func = elaborate_converter(converter, &params, i32_ty());

    let entry = func.get_block(func.entry_block());
    // Should have a BinOp(+) instruction.
    assert!(entry.insts.iter().any(|&iid| {
        matches!(
            &func.get_inst(iid).data,
            InstKind::Op {
                tag: op::OpTag::BinOp(op::BinaryOperator::Add),
                ..
            }
        )
    }));
}

#[test]
fn test_gvn_via_let() {
    // let x = 42 in let y = 42 in (x, y)
    // GVN should deduplicate the two 42 constants into a single node.
    // (A `+` would be constant-folded to `84`, erasing the evidence.)
    use polytype::Type;
    let mut term_ids = TermIdSource::new();
    let pair_ty = Type::Constructed(TypeName::Tuple(2), vec![i32_ty(), i32_ty()]);

    let lit42 = mk_term(&mut term_ids, i32_ty(), TermKind::IntLit("42".into()));
    let lit42b = mk_term(&mut term_ids, i32_ty(), TermKind::IntLit("42".into()));

    let mut symbols = SymbolTable::new();
    let x_sym = symbols.alloc("x".into());
    let y_sym = symbols.alloc("y".into());

    let x_ref = mk_term(&mut term_ids, i32_ty(), TermKind::Var(VarRef::Symbol(x_sym)));
    let y_ref = mk_term(&mut term_ids, i32_ty(), TermKind::Var(VarRef::Symbol(y_sym)));
    let pair_app = mk_term(
        &mut term_ids,
        pair_ty.clone(),
        TermKind::Tuple(vec![x_ref, y_ref]),
    );

    let inner_let = mk_term(
        &mut term_ids,
        pair_ty.clone(),
        TermKind::Let {
            name: y_sym,
            name_ty: i32_ty(),
            rhs: Box::new(lit42b),
            body: Box::new(pair_app),
        },
    );
    let outer_let = mk_term(
        &mut term_ids,
        pair_ty.clone(),
        TermKind::Let {
            name: x_sym,
            name_ty: i32_ty(),
            rhs: Box::new(lit42),
            body: Box::new(inner_let),
        },
    );

    let mut binding_ids = IdSource::<u32>::new();
    let mut effect_ids = IdSource::new();
    let plan = ConversionPlan::empty();
    let mut converter = Converter::new(&symbols, &mut binding_ids, &mut effect_ids, &plan);
    let result = converter.convert_term(&outer_let).expect("conversion failed");
    converter.set_return(Some(converter.graph.value_result(result)));

    let func = elaborate_converter(converter, &Parameters::new(), pair_ty);

    let entry = func.get_block(func.entry_block());
    // The hash-consed constant reaches both tuple fields directly; EGIR
    // elaboration must not expand either occurrence back into an instruction.
    let tuple_operands = entry
        .insts
        .iter()
        .find_map(|&iid| match &func.get_inst(iid).data {
            InstKind::Op {
                tag: op::OpTag::Tuple(2),
                operands,
            } => Some(operands),
            _ => None,
        })
        .expect("tuple construction should remain in SSA");
    assert_eq!(
        tuple_operands,
        &[
            ValueRef::Const(ConstantValue::I32(42)),
            ValueRef::Const(ConstantValue::I32(42)),
        ]
    );
    assert!(!entry.insts.iter().any(|&iid| {
        matches!(
            &func.get_inst(iid).data,
            InstKind::Op {
                tag: op::OpTag::Int(_),
                ..
            }
        )
    }));
}

#[test]
fn test_hash_cons_distinguishes_by_result_type() {
    // Interning the same intrinsic with the same operands but different
    // result types must produce distinct ValueNodeIds; otherwise the first-inserted
    // type silently wins at the merged node. This applies to every pure op,
    // including `_w_intrinsic_storage_len` instantiated as i32 and u32.
    use crate::egir::types::{EGraph, PureOp};
    use smallvec::smallvec;

    let mut g = EGraph::<egir::types::Semantic>::new();
    let i32_ty = i32_ty();
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);

    let zero_u32 = g.intern_pure(
        PureOp::Uint("0".into()),
        smallvec::SmallVec::new(),
        u32_ty.clone(),
        None,
    );

    let storage_len_id = builtins::catalog().known().storage_len;
    let a = g.intern_pure(
        PureOp::Intrinsic {
            id: storage_len_id,
            overload_idx: 0,
        },
        smallvec![zero_u32, zero_u32],
        i32_ty,
        None,
    );
    let b = g.intern_pure(
        PureOp::Intrinsic {
            id: storage_len_id,
            overload_idx: 0,
        },
        smallvec![zero_u32, zero_u32],
        u32_ty,
        None,
    );
    assert_ne!(
        a, b,
        "different result types must not hash-cons to the same ValueId"
    );
}

#[test]
fn test_if_else_roundtrip() {
    // if cond then 1 else 0
    let mut symbols = SymbolTable::new();
    let mut term_ids = TermIdSource::new();
    let c_sym = symbols.alloc("c".into());
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    let cond = mk_term(
        &mut term_ids,
        bool_ty.clone(),
        TermKind::Var(VarRef::Symbol(c_sym)),
    );
    let then_br = mk_term(&mut term_ids, i32_ty(), TermKind::IntLit("1".into()));
    let else_br = mk_term(&mut term_ids, i32_ty(), TermKind::IntLit("0".into()));
    let if_term = mk_term(
        &mut term_ids,
        i32_ty(),
        TermKind::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_br),
            else_branch: Box::new(else_br),
        },
    );

    let mut binding_ids = IdSource::<u32>::new();
    let mut effect_ids = IdSource::new();
    let plan = ConversionPlan::empty();
    let mut converter = Converter::new(&symbols, &mut binding_ids, &mut effect_ids, &plan);
    let params = Parameters::from_ordered([callable_parameter::<BindingRef, WynLanguage>(
        "c".into(),
        Type::Constructed(TypeName::Bool, vec![]),
    )]);
    let c_nid = converter.graph.add_test_value_parameter(params.ids().next().unwrap(), bool_ty);
    converter.locals.insert(c_sym, c_nid);

    let result = converter.convert_term(&if_term).expect("conversion failed");
    converter.set_return(Some(converter.graph.value_result(result)));

    let func = elaborate_converter(converter, &params, i32_ty());

    // Should have 4 blocks: entry, then, else, merge
    assert_eq!(func.inner.blocks.len(), 4, "if/else should produce 4 blocks");

    // Entry should end with CondBranch
    let entry = func.get_block(func.entry_block());
    assert!(
        matches!(&entry.term, ssa::framework::Terminator::CondBranch { .. }),
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


entry main() vec4f32 =
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


entry main() vec4f32 =
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


entry main() vec4f32 =
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


entry vertex_main() vec4f32 =
    let e = evens([1, 2, 3, 4]) in
    @[f32.i32(length(e)), 0.0, 0.0, 1.0]
"#,
    );
    assert!(!program.entry_points.is_empty(), "Should have entry points");
}

#[test]
fn conversion_completes_output_routes_and_filter_abi_policy() {
    use crate::pipeline_descriptor::BufferLen;

    let source = r#"
entry evens(xs: []i32) []i32 =
  filter(|x: i32| x % 2 == 0, xs)
"#;
    let tlc = tlc::infer_input_slice_bounds(test_pipeline::compile_to_reachable(source));
    let raw = convert_program(&tlc, IdSource::<u32>::new(), IdSource::new())
        .expect("TLC-to-EGIR construction succeeds");
    let entry = raw.entry_points.iter().find(|entry| entry.name == "evens").unwrap();
    let output = &entry.outputs[0];

    assert!(
        !output.routes.is_empty(),
        "Converted entries own complete output routes"
    );
    assert!(output.routes.iter().all(|route| route.writers.is_empty()));
    assert!(matches!(
        output.storage_length(),
        Some(BufferLen::LikeInput {
            elem_bytes: 4,
            src_elem_bytes: 4,
            ..
        })
    ));
    assert!(
        entry.graph.skeleton.blocks.values().flat_map(|block| &block.side_effects).any(|effect| matches!(
            effect.kind(),
            egir::types::SideEffectKind::Soac(egir::types::SoacEffect(
                (),
                egir::types::Soac::Filter(filter)
            )) if matches!(filter.state.output, egir::soac::filter::RawOutput::Runtime { .. })
        ))
    );
}

#[test]
fn reification_links_filter_publication_uniformly() {
    use crate::ResourceAccess;

    let source = r#"
entry evens(xs: []i32) []i32 =
  filter(|x: i32| x % 2 == 0, xs)
"#;
    let tlc = tlc::infer_input_slice_bounds(test_pipeline::compile_to_reachable(source));
    let raw = convert_program(&tlc, IdSource::<u32>::new(), IdSource::new())
        .expect("TLC-to-EGIR construction succeeds");
    let semantic = egir::reify_soacs(raw);
    let entry = semantic.entry_points.iter().find(|entry| entry.name == "evens").unwrap();
    let output_resource = entry.outputs[0].resource.expect("compute output is storage-backed");

    assert!(entry.outputs[0].routes.iter().all(|route| !route.writers.is_empty()));
    let filter = entry
        .graph
        .skeleton
        .blocks
        .values()
        .flat_map(|block| &block.side_effects)
        .find_map(|effect| match effect.kind() {
            egir::types::SideEffectKind::Soac(egir::types::SoacEffect(
                _,
                egir::types::Soac::Filter(filter),
            )) => Some(filter),
            _ => None,
        })
        .expect("entry retains a semantic Filter");
    assert_eq!(filter.state.output_slots, [egir::program::OutputSlotId(0)]);
    assert!(filter
        .state
        .resources
        .iter()
        .any(|access| { access.resource == output_resource && access.access != ResourceAccess::Read }));
    assert!(matches!(
        filter.state.output,
        egir::soac::filter::Output::Runtime(egir::soac::filter::RuntimeOutput {
            backing: egir::soac::filter::RuntimeBacking::Deferred,
            length: egir::soac::filter::RuntimeLength::Implicit,
            ..
        })
    ));
}

// --- vertex_inputs population from params ------------

#[test]
fn pure_user_calls_enter_egir_as_canonical_call_sites_during_construction() {
    use crate::egir::types::ValueKind;

    let source = r#"
def choose(x: i32) i32 =
  if x > 0 then x + 1 else x - 1

def wrapper(x: i32) i32 =
  if x > 0 then choose(x) else choose(0 - x)

entry e(xs: []i32) []i32 = map(wrapper, xs)
"#;
    let tlc = tlc::infer_input_slice_bounds(test_pipeline::compile_to_reachable(source));
    let binding_ids = IdSource::<u32>::new();
    let effect_ids = IdSource::new();
    let raw = convert_program(&tlc, binding_ids, effect_ids).expect("TLC-to-EGIR construction succeeds");
    let choose = raw.functions.iter().find(|function| function.name == "choose").unwrap().region;
    let wrapper = raw.functions.iter().find(|function| function.name == "wrapper").unwrap();

    assert!(wrapper.graph.nodes.values().any(|node| {
        let ValueKind::CallResult { call, .. } = &node.kind else {
            return false;
        };
        wrapper.graph.call(*call).callee() == choose
    }));
    let calls = wrapper.graph.calls().keys().collect::<Vec<_>>();
    assert_eq!(calls.len(), 2);
    let effects = wrapper.graph.side_effect_index();
    assert!(calls.into_iter().all(|call| effects.call_site(call).is_some()));
    assert!(!wrapper.graph.has_ordered_effects());
}

#[test]
fn construction_purity_propagates_effectful_builtin_calls() {
    use crate::tlc::{Def, DefMeta};

    let mut symbols = SymbolTable::new();
    let mut term_ids = TermIdSource::new();
    let pure_leaf = symbols.alloc("pure_leaf".into());
    let reads_storage = symbols.alloc("reads_storage".into());
    let calls_reader = symbols.alloc("calls_reader".into());
    let array_consumer = symbols.alloc("array_consumer".into());
    let ty = i32_ty();
    let definition = |name, body| Def {
        data: (),
        name,
        package: None,
        ty: ty.clone(),
        body,
        meta: DefMeta::Function,
        arity: 1,
        param_diets: vec![types::Diet::observing()],
        return_diet: types::Diet::observing(),
    };
    let reader_ref = mk_term(
        &mut term_ids,
        ty.clone(),
        TermKind::Var(VarRef::Symbol(reads_storage)),
    );
    let direct_call = mk_term(
        &mut term_ids,
        ty.clone(),
        TermKind::App {
            func: Box::new(reader_ref),
            args: vec![],
        },
    );
    let storage_index = mk_term(
        &mut term_ids,
        ty.clone(),
        TermKind::Var(VarRef::Builtin {
            id: builtins::catalog().known().storage_index,
            overload_idx: 0,
        }),
    );
    let storage_read = mk_term(
        &mut term_ids,
        ty.clone(),
        TermKind::App {
            func: Box::new(storage_index),
            args: vec![],
        },
    );
    let pure_body = mk_term(&mut term_ids, ty.clone(), TermKind::IntLit("1".into()));
    let array_consumer_body = mk_term(&mut term_ids, ty.clone(), TermKind::IntLit("1".into()));
    let program = tlc::Program::from_parts(
        vec![
            definition(pure_leaf, pure_body),
            definition(reads_storage, storage_read),
            definition(calls_reader, direct_call),
            Def {
                data: (),
                name: array_consumer,
                package: None,
                ty: Type::Constructed(
                    TypeName::Arrow,
                    vec![Type::Constructed(TypeName::Array, vec![]), ty.clone()],
                ),
                body: array_consumer_body,
                meta: DefMeta::Function,
                arity: 1,
                param_diets: vec![types::Diet::observing()],
                return_diet: types::Diet::observing(),
            },
        ],
        symbols,
        term_ids,
        tlc::context::BackendGlobal {
            auto_storage_binding_ids: IdSource::new(),
        },
    );

    let pure = super::infer_pure_definitions(&program);
    assert!(pure.contains(&pure_leaf));
    assert!(!pure.contains(&reads_storage));
    assert!(!pure.contains(&calls_reader));
    assert!(
        !pure.contains(&array_consumer),
        "non-copy ABIs remain anchored until semantic array dependencies no longer rely on effects"
    );
}

/// Graphics entries must NOT derive `Entry<Semantic>.return_ty` from
/// `def.ty`'s arrow-return position. Compute output routes use that signature,
/// while graphics entries use `inner_body.ty`, which
/// matches the body's actual produced shape after ownership and
/// monomorphization. A divergence between the signature and body must not
/// change how `build_entry_outputs` classifies a graphics return; wrapping
/// the signature's tuple would otherwise collapse N outputs to one.
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
entry frame(target: render_target<vec4f32>) render_target<vec4f32> =
  let covered = rasterize_triangles(
    direct_draw(3u32, 1u32),
    |vertex| vertex_output(
      @[f32(vertex.vertex_index), 0.0, 0.0, 1.0],
      @[1.0, 0.0, 0.0])) in
  shade(target, covered,
    |fragment| @[fragment.value.x, fragment.value.y, fragment.value.z, 1.0])
"#;
    let mut tlc_program = tlc::infer_input_slice_bounds(test_pipeline::compile_to_reachable(src));

    // Wrap only the vertex entry's signature return. `inner_body.ty` stays
    // unchanged, giving the test two deliberately divergent type sources.
    let vertex_symbol = tlc_program
        .defs
        .iter()
        .find(|definition| {
            matches!(
                &definition.meta,
                DefMeta::EntryPoint(entry)
                    if entry.declaration.entry_kind == interface::EntryKind::Vertex
            )
        })
        .map(|definition| definition.name)
        .expect("extracted vertex definition");
    let vertex_name = tlc_program.symbols.get(vertex_symbol).expect("vertex name").to_string();
    let def = tlc_program
        .defs
        .iter_mut()
        .find(|definition| definition.name == vertex_symbol)
        .expect("extracted vertex definition");
    assert!(
        matches!(
            &def.meta,
            DefMeta::EntryPoint(e)
                if e.declaration.entry_kind != interface::EntryKind::Compute
        ),
        "precondition: vertex_main is a graphics entry"
    );
    wrap_arrow_return_in_marker(&mut def.ty);

    let binding_ids = IdSource::<u32>::new();
    let effect_ids = IdSource::new();
    let egir = super::convert_program(&tlc_program, binding_ids, effect_ids)
        .expect("from_tlc::convert_program on graphics entry must succeed");
    let entry =
        egir.entry_points.iter().find(|entry| entry.name == vertex_name).expect("vertex Entry<Semantic>");

    assert_eq!(
        entry.outputs.len(),
        2,
        "graphics entry's outputs must be derived from inner_body.ty (preserved as a tuple) \
         — not from def.ty (mutated to Unique-wrap the tuple). got {:?}",
        entry.outputs.iter().map(|o| &o.ty).collect::<Vec<_>>()
    );
}

/// Walk an arrow chain `P1 -> P2 -> ... -> Pn -> R` and wrap `R` in a
/// spurious single-element `Tuple(1)`. This creates a divergence between
/// `def.ty`'s arrow-return position and the real two-element
/// `inner_body.ty`.
fn wrap_arrow_return_in_marker(mut ty: &mut Type<TypeName>) {
    loop {
        let inner = match ty {
            Type::Constructed(TypeName::Arrow, args) if args.len() == 2 => &mut args[1],
            _ => break,
        };
        if !matches!(inner, Type::Constructed(TypeName::Arrow, _)) {
            let old = std::mem::replace(inner, Type::Constructed(TypeName::Unit, vec![]));
            *inner = Type::Constructed(TypeName::Tuple(1), vec![old]);
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
    let src = " entry prefix(xs: []i32) []i32 = scan(|a: i32, b: i32| a + b, 0, xs)";
    compile_thru_ssa(src).expect("parallel scan lowers, recovering its synthesized swap-wrapper region");
}
