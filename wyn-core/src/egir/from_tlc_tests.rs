// ============================================================================
// Tests
// ============================================================================

use super::{Converter, run};
use crate::SymbolId;
use crate::SymbolTable;
use crate::ast::TypeName;
use crate::pipeline_descriptor::PipelineDescriptor;
use crate::ssa::types::{FuncBody, InstKind, Program};
use crate::tlc::VarRef;
use crate::tlc::{Term, TermKind};
use polytype::Type;
use std::collections::{HashMap, HashSet};

/// Compile a source string through the full TLC pipeline, then convert
/// through the full EGIR chain (`from_tlc → expand_soacs → optimize_skeleton
/// → elaborate`) to a `Program`. No `materialize` — tests don't exercise
/// SPIR-V-specific dynamic-index rewrites.
fn compile_via_egir(src: &str) -> Program {
    let (mut node_counter, mut module_manager) = crate::cached_compiler_init();
    let parsed = crate::Compiler::parse(src, &mut node_counter).expect("Parsing failed");
    let type_checked = parsed
        .desugar(&mut node_counter)
        .expect("Desugaring failed")
        .resolve(&mut module_manager)
        .expect("Name resolution failed")
        .fold_ast_constants()
        .type_check(&mut module_manager)
        .expect("Type checking failed");

    let tlc = type_checked
        .to_tlc(&module_manager, false)
        .partial_eval()
        .normalize_soacs()
        .fuse_maps()
        .apply_ownership()
        .expect("apply_ownership")
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .parallelize_soacs(false)
        .filter_reachable();

    crate::EgirRaw(run(&tlc.tlc, PipelineDescriptor::default()).expect("egir::from_tlc conversion failed"))
        .expand_soacs(true)
        .optimize_skeleton()
        .elaborate()
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

    let mut converter = Converter::new(&top_level, &constants_by_name, &symbols, pure_constants);
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
    assert!(
        entry
            .insts
            .iter()
            .any(|&iid| { matches!(&func.get_inst(iid).data, InstKind::Int(s) if s == "42") })
    );
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

    let mut converter = Converter::new(&top_level, &constants_by_name, &symbols, pure_constants);
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
            .any(|&iid| { matches!(&func.get_inst(iid).data, InstKind::BinOp { op, .. } if op == "+") })
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

    let mut converter = Converter::new(&top_level, &constants_by_name, &symbols, pure_constants);
    let result = converter.convert_term(&outer_let).expect("conversion failed");
    converter.set_return(Some(result));

    let func = converter.elaborate_to_funcbody(&[], pair_ty).expect("elaboration failed");

    let entry = func.get_block(func.entry_block());
    // GVN: should have only ONE Int("42") instruction, not two.
    let const_count = entry
        .insts
        .iter()
        .filter(|&&iid| matches!(&func.get_inst(iid).data, InstKind::Int(s) if s == "42"))
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

    let mut converter = Converter::new(&top_level, &constants_by_name, &symbols, pure_constants);
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
#[should_panic(expected = "is not Map/Scan")]
fn rewrite_map_scan_to_into_panics_on_reduce() {
    use super::rewrite_map_scan_to_into;
    use crate::egir::types::{EGraph, PendingSoac, SideEffect, SideEffectKind};
    use smallvec::SmallVec;

    let mut graph = EGraph::new();
    let target = graph.alloc_side_effect_result(i32_ty());
    let output_view = graph.alloc_side_effect_result(i32_ty());

    let entry = graph.skeleton.entry;
    graph.skeleton.blocks[entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Pending(PendingSoac::Reduce {
            func: "f".to_string(),
            input_array_type: i32_ty(),
            input_elem_type: i32_ty(),
        }),
        operand_nodes: SmallVec::new(),
        result: Some(target),
        effects: None,
        span: None,
    });

    rewrite_map_scan_to_into(&mut graph, target, output_view);
}

#[test]
#[should_panic(expected = "runtime-sized array")]
fn emit_compute_output_stores_panics_on_unsized_array() {
    use super::emit_compute_output_stores;
    use crate::ssa::types::EntryOutput;

    // Build a runtime-sized array type: [n]f32 where n is a free variable.
    let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);
    let unsized_arr_ty = Type::Constructed(TypeName::Array, vec![f32_ty.clone(), Type::Variable(99)]);

    let symbols = SymbolTable::new();
    let top_level = HashMap::new();
    let constants_by_name = HashMap::new();
    let mut converter = Converter::new(&top_level, &constants_by_name, &symbols, HashSet::new());

    let result_nid = converter.graph.alloc_side_effect_result(unsized_arr_ty.clone());

    let outputs = vec![EntryOutput {
        ty: unsized_arr_ty,
        decoration: None,
        storage_binding: Some((0, 1)),
    }];

    emit_compute_output_stores(&mut converter, result_nid, &outputs);
}

#[test]
#[should_panic(expected = "no side effect produced")]
fn rewrite_map_scan_to_into_panics_when_target_missing() {
    use super::rewrite_map_scan_to_into;
    use crate::egir::types::EGraph;

    let mut graph = EGraph::new();
    let target = graph.alloc_side_effect_result(i32_ty());
    let output_view = graph.alloc_side_effect_result(i32_ty());

    rewrite_map_scan_to_into(&mut graph, target, output_view);
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
