fn empty_compute_entry(index: usize) -> crate::ssa::types::EntryPoint {
    use crate::ssa::{
        builder::FuncBuilder,
        types::{EntryPoint, Terminator},
    };
    let mut builder = FuncBuilder::new(vec![], crate::types::unit());
    builder.terminate(Terminator::Return(None)).unwrap();
    EntryPoint {
        id: crate::EntryId::from_index(index),
        name: format!("entry_{index}"),
        body: builder.finish().unwrap(),
        execution_model: crate::flow::ExecutionModel::Compute {
            local_size: (1, 1, 1),
        },
        inputs: vec![],
        parameter_inputs: vec![],
        outputs: vec![],
        storage_bindings: vec![],
        stage_descriptor_storage_accesses: Default::default(),
        pipeline_storage_accesses: Default::default(),
        span: crate::ast::Span::generated(),
    }
}

fn emit_entries(entries: Vec<crate::ssa::types::EntryPoint>) -> Vec<u32> {
    let program = crate::ssa::stage::SpirvReady::from_parts(
        vec![],
        entries,
        crate::ssa::context::BackendGlobal {
            pipeline: Default::default(),
            target: crate::CodegenTarget::Spirv,
        },
    );
    super::lower_ssa_program_impl(&program).unwrap()
}

#[test]
fn explicit_storage_view_length_needs_no_array_length() {
    let words = crate::ssa::test_support::spirv(vec![crate::ssa::test_support::storage_write()]);
    let module = wspirv::dr::load_words(&words).unwrap();
    let lengths: Vec<_> =
        module.all_inst_iter().filter(|i| i.class.opcode == spirv::Op::ArrayLength).collect();
    assert_eq!(
        lengths.len(),
        0,
        "the logical input length is supplied explicitly"
    );
    assert!(
        module.all_inst_iter().any(|i| i.class.opcode == spirv::Op::Store),
        "map writes its output"
    );
}

#[test]
fn literal_composite_indices_do_not_emit_unused_constant_ids() {
    use crate::op::OpTag;
    use crate::ssa::types::InstKind;
    let entry =
        crate::ssa::test_support::scalar_entry(vec![crate::types::f32()], crate::types::f32(), |builder| {
            let array = builder
                .push_inst(
                    InstKind::Op {
                        tag: OpTag::ArrayLit,
                        operands: vec![builder.get_param(0).into(); 48],
                    },
                    crate::types::sized_array(48, crate::types::f32()),
                )
                .unwrap();
            builder
                .push_inst(
                    InstKind::Op {
                        tag: OpTag::Project { index: 38 },
                        operands: vec![array.into()],
                    },
                    crate::types::f32(),
                )
                .unwrap()
                .into()
        });
    let module = wspirv::dr::load_words(crate::ssa::test_support::spirv(vec![entry])).unwrap();
    // Field forwarding can remove the extraction; its literal index must not
    // leave an unused scalar constant in either representation.
    assert!(!module
        .types_global_values
        .iter()
        .any(|i| i.class.opcode == spirv::Op::Constant
            && i.operands == [wspirv::dr::Operand::LiteralBit32(38)]));
}

#[test]
fn storage_length_descriptors_do_not_emit_unused_constant_ids() {
    let entry = crate::ssa::test_support::storage_write_at(crate::BindingRef::new(0, 38));
    let module = wspirv::dr::load_words(crate::ssa::test_support::spirv(vec![entry])).unwrap();
    let buffers: Vec<_> = module
        .annotations
        .iter()
        .filter_map(|inst| match inst.operands.as_slice() {
            [wspirv::dr::Operand::IdRef(id), wspirv::dr::Operand::Decoration(spirv::Decoration::Binding), wspirv::dr::Operand::LiteralBit32(38)] => Some(*id),
            _ => None,
        })
        .collect();
    assert!(!buffers.is_empty(), "input keeps its declared binding");
    assert!(
        module.functions.iter().flat_map(|f| &f.blocks).flat_map(|b| &b.instructions).any(|i| i
            .class
            .opcode
            == spirv::Op::AccessChain
            && matches!(i.operands.first(), Some(wspirv::dr::Operand::IdRef(id)) if buffers.contains(id)))
    );
    for constant in module.types_global_values.iter().filter(|i| {
        i.class.opcode == spirv::Op::Constant && i.operands == [wspirv::dr::Operand::LiteralBit32(38)]
    }) {
        // The packed length table can use index 38; it must not emit an unused ID.
        let id = constant.result_id.unwrap();
        assert!(module.all_inst_iter().any(|inst| inst.operands.contains(&wspirv::dr::Operand::IdRef(id))));
    }
}

#[test]
fn unrelated_entries_preserve_storage_buffer_element_types() {
    use crate::BindingRef;
    use crate::{types, EntryId, LookupMap, ResourceAccess};
    let binding = BindingRef::new(0, 0);
    for reordered in [false, true] {
        let mut constructor = super::Constructor::new();
        let mut entries = vec![
            (EntryId::from_index(0), types::f32()),
            (EntryId::from_index(1), types::vec(4, types::f32())),
            (EntryId::from_index(2), types::f32()),
        ];
        if reordered {
            entries.swap(0, 1);
        }
        let accesses = LookupMap::from_iter([(binding, ResourceAccess::Read)]);
        let mut created = Vec::new();
        for (entry, ty) in &entries {
            constructor.select_storage_accesses(&accesses, Some(*entry));
            let variable = constructor.create_storage_buffer_for_element(ty, 0, 0, false).unwrap();
            let element = constructor.storage_polytype_to_spirv(ty).unwrap();
            created.push((*entry, variable, element));
        }
        created.sort_by_key(|(entry, _, _)| *entry);
        assert_eq!(created[0].1, created[2].1, "identical types share a global");
        assert_ne!(
            created[0].1, created[1].1,
            "different types need distinct globals"
        );
        for (entry, variable, element) in created {
            constructor.select_storage_accesses(&accesses, Some(entry));
            let buffer = constructor.storage_buffer(binding).unwrap();
            assert_eq!(buffer.variable, variable);
            assert_eq!(buffer.element_type, element);
        }
    }
}

#[test]
fn test_linked_extern_call_uses_structural_function_identity() {
    use crate::ssa::{
        builder::FuncBuilder,
        types::{Function, InstKind},
    };
    let function = Function {
        id: crate::FunctionId::from_index(0),
        name: "display_name".into(),
        linkage_name: Some("external_increment".into()),
        span: crate::ast::Span::generated(),
        body: FuncBuilder::new(vec![crate::types::i32()], crate::types::i32()).finish_unchecked(),
    };
    let entry =
        crate::ssa::test_support::scalar_entry(vec![crate::types::i32()], crate::types::i32(), |builder| {
            builder
                .push_inst(
                    InstKind::Op {
                        tag: crate::op::OpTag::Call(function.id),
                        operands: vec![builder.get_param(0).into()],
                    },
                    crate::types::i32(),
                )
                .unwrap()
                .into()
        });
    let program = crate::ssa::stage::SpirvReady::from_parts(
        vec![function],
        vec![entry],
        crate::ssa::context::BackendGlobal {
            pipeline: Default::default(),
            target: crate::CodegenTarget::Spirv,
        },
    );
    let words = super::lower_ssa_program_impl(&program).unwrap();
    let module = wspirv::dr::load_words(words).unwrap();
    assert!(module.annotations.iter().any(|instruction| instruction
        .operands
        .contains(&wspirv::dr::Operand::LiteralString("external_increment".into()))
        && instruction.operands.contains(&wspirv::dr::Operand::LinkageType(spirv::LinkageType::Import))));
    assert!(module.all_inst_iter().any(|instruction| instruction.class.opcode == spirv::Op::FunctionCall));
}

#[test]
fn distinct_entry_push_constant_layouts_have_distinct_variables() {
    use std::collections::HashSet;
    use wspirv::dr::Operand;
    use wspirv::spirv::{Op, StorageClass};
    let entries = [
        crate::types::i32(),
        crate::types::sized_array(2, crate::types::i32()),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, ty)| {
        let mut entry = empty_compute_entry(index);
        entry.inputs.push(crate::interface::EntryInput {
            name: "input".into(),
            ty,
            host: (),
            kind: crate::interface::EntryInputKind::PushConstant {
                slot: crate::interface::PushConstantSlot {
                    offset: 0,
                    size: if index == 0 { 4 } else { 8 },
                },
            },
        });
        entry
    })
    .collect();
    let module = wspirv::dr::load_words(emit_entries(entries)).unwrap();
    let variables: HashSet<_> = module
        .types_global_values
        .iter()
        .filter_map(|i| {
            (i.class.opcode == Op::Variable
                && i.operands.first() == Some(&Operand::StorageClass(StorageClass::PushConstant)))
            .then_some(i.result_id)
            .flatten()
        })
        .collect();
    assert!(variables.len() >= 2);
    for entry in &module.entry_points {
        assert_eq!(
            entry
                .operands
                .iter()
                .filter(|operand| matches!(operand, Operand::IdRef(id) if variables.contains(id)))
                .count(),
            1
        );
    }
}

#[test]
fn test_two_compute_entries_share_one_global_invocation_id() {
    // VUID-StandaloneSpirv-OpEntryPoint-09658 permits exactly one module-level
    // GlobalInvocationId input variable. Count its BuiltIn decoration to
    // ensure multiple compute entries share that variable.
    let spirv = emit_entries(vec![empty_compute_entry(0), empty_compute_entry(1)]);
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
/// `SoacKind::Scatter` → `SoacOp::Scatter` → `egglog::OperationKind::Scatter` →
/// `build_scatter_loop` path emits indexed `OpStore`s into the destination
/// view (one indexed store executed by the scatter loop).
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
    let spirv = crate::lower_ssa_to_spirv(crate::ssa::test_support::indexed_constant(false)).unwrap().spirv;
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
    let spirv = crate::lower_ssa_to_spirv(crate::ssa::test_support::indexed_constant(true)).unwrap().spirv;
    let text = disasm(&spirv);
    let n = text.lines().filter(|l| l.contains("OpVariable") && l.contains(" Private ")).count();
    assert_eq!(
        n, 1,
        "two indexings of one constant must share one Private global:\n{text}"
    );
}

#[test]
fn small_source_programs_typecheck() {
    for (name, source) in [
        ("test_simple_constant", "def x = 42"),
        ("test_simple_function", "def add(x, y) = x + y"),
        ("test_let_binding", "def f = let x = 1 in x + 2"),
        ("test_arithmetic", "def f(x, y) = x * y + x / y - 1"),
        ("test_nested_let", "def f = let a = 1 in let b = 2 in a + b"),
        ("test_if_expression", "def f(x) = if x == 0 then 1 else 2"),
        (
            "test_comparisons",
            "def f(x, y) = if x < y then 1 else if x > y then 2 else 0",
        ),
        ("test_tuple_literal", "def f = (1, 2, 3)"),
        ("test_array_literal", "def f = [1, 2, 3]"),
        ("test_unary_negation", "def f(x) = -x"),
        (
            "test_record_field_access",
            r#"
def get_x(r:{x:i32, y:i32}) i32 = r.x
"#,
        ),
        (
            "test_closure_capture_access",
            r#"
def test(x:i32) i32 =
    let f = |y:i32| x + y in
    f(10)
"#,
        ),
        (
            "test_polymorphic_dot2",
            r#"
def dot2<E, T>(v: T) E = dot(v, v)

def test_dot2_vec3(v: vec3f32) f32 = dot2(v)
"#,
        ),
        (
            "test_polymorphic_dot2_in_expression",
            r#"
def dot2<E, T>(v: T) E = dot(v, v)

def sdCappedTorus(p: vec3f32, ra: f32, rb: f32, k: f32) f32 =
  f32.sqrt(dot2(p) + ra*ra - 2.0*ra*k) - rb
"#,
        ),
        (
            "test_polymorphic_dot2_vec2_and_vec3",
            r#"
def dot2<E, T>(v: T) E = dot(v, v)

def test_vec3(v: vec3f32) f32 = dot2(v)
def test_vec2(v: vec2f32) f32 = dot2(v)
def test_both(v3: vec3f32, v2: vec2f32) f32 = dot2(v3) + dot2(v2)
"#,
        ),
        (
            "test_scan_inclusive",
            r#"
def sum_scan(arr: [4]i32) [4]i32 = scan((|a, b| a + b), 0, arr)
"#,
        ),
        (
            "test_scatter_update",
            r#"
def scatter_test(dest: [5]i32, indices: [2]i32, values: [2]i32) [5]i32 =
    scatter(dest, indices, values)
"#,
        ),
        (
            "test_reduce_by_index",
            r#"
def hist_test(dest: [3]i32, indices: [4]i32, values: [4]i32) [3]i32 =
    reduce_by_index(dest, |a, b| a + b, 0, indices, values)
"#,
        ),
        (
            "test_hist",
            r#"
def hist_alias_test(dest: [3]i32, indices: [4]i32, values: [4]i32) [3]i32 =
    hist(dest, |a, b| a + b, 0, indices, values)
"#,
        ),
        (
            "test_reduce_u32",
            r#"
def sum_u32(arr: [4]u32) u32 =
    reduce(|a, b| a + b, 0u32, arr)
"#,
        ),
    ] {
        crate::compile_thru_frontend(source).unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}
