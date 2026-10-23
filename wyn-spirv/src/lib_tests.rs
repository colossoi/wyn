// Test assertions intentionally panic on unexpected errors; production code
// retains the parent module's unwrap/expect bans.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;

#[test]
fn integer_constants_keep_their_width_and_required_capability() {
    for (width, capability) in [
        (8, spirv::Capability::Int8),
        (16, spirv::Capability::Int16),
        (64, spirv::Capability::Int64),
    ] {
        let mut builder = SpirvBuilder::new();
        let signed = builder.const_integer(width, true, u64::MAX);
        let unsigned = builder.const_integer(width, false, (1u64 << (width - 1)) - 1);
        let module = builder.into_module();
        assert_eq!(
            module
                .capabilities
                .iter()
                .filter(|instruction| {
                    instruction.operands == [rspirv::dr::Operand::Capability(capability)]
                })
                .count(),
            1
        );
        for (constant, signedness) in [(signed, 1), (unsigned, 0)] {
            let declaration = module
                .types_global_values
                .iter()
                .find(|instruction| instruction.result_id == Some(*constant))
                .unwrap();
            let ty = module
                .types_global_values
                .iter()
                .find(|instruction| instruction.result_id == declaration.result_type)
                .unwrap();
            assert_eq!(ty.class.opcode, spirv::Op::TypeInt);
            assert_eq!(
                ty.operands,
                [
                    rspirv::dr::Operand::LiteralBit32(width),
                    rspirv::dr::Operand::LiteralBit32(signedness),
                ]
            );
        }
    }
}

#[test]
fn buffer_array_layout_is_distinct_and_decorated_once() {
    let mut builder = SpirvBuilder::new();
    let element = builder.u32_type();
    let count = builder.const_u32(4);
    let logical = builder.type_array(element, *count);
    let storage = builder.type_buffer_array(element, *count, 4);
    assert_ne!(logical, storage);
    builder.decorate_array_stride_once(storage, 4);
    let module = builder.into_module();
    let strides: Vec<_> = module
        .annotations
        .iter()
        .filter(|i| {
            i.operands.get(1) == Some(&rspirv::dr::Operand::Decoration(spirv::Decoration::ArrayStride))
        })
        .collect();
    assert_eq!(strides.len(), 1);
    assert_eq!(strides[0].operands[0], rspirv::dr::Operand::IdRef(*storage));
}

/// `Id<K>` for distinct `K`s are distinct types — a function expecting
/// `TypeId` cannot be called with a `ValueId`. Compile-only check; if
/// this file builds, the safety invariant holds.
#[test]
fn id_kinds_are_distinct_types() {
    fn takes_type(_: TypeId) {}
    fn takes_value(_: ValueId) {}
    let t: TypeId = Id::new(7);
    let v: ValueId = Id::new(11);
    takes_type(t);
    takes_value(v);
    // `takes_type(v)` here would fail to compile — that's the point.
}

#[test]
fn deref_extracts_raw_word() {
    let t: TypeId = Id::new(42);
    assert_eq!(*t, 42);
    // Function signatures wanting `spirv::Word` still need explicit
    // `*id` — auto-coercion via Deref doesn't apply to function args.
    fn takes_word(_: spirv::Word) {}
    takes_word(*t);
}

#[test]
fn id_is_copy_eq_hash() {
    use std::collections::HashSet;
    let a: TypeId = Id::new(1);
    let b: TypeId = Id::new(1);
    let c: TypeId = Id::new(2);
    assert_eq!(a, b);
    assert_ne!(a, c);
    let mut set = HashSet::new();
    set.insert(a);
    set.insert(b);
    set.insert(c);
    assert_eq!(set.len(), 2);
    // `a` is still usable after insertion — proves `Copy`.
    let _ = a;
}

#[test]
fn builder_emits_minimal_valid_module() {
    let mut b = SpirvBuilder::new();
    // Add an OpTypeVoid + a no-op function so the assembled module
    // has *something* in it. Pure smoke test that the setup is right.
    let void = b.void_type();
    let (_fn_id, _params, _code_block) = b.begin_function(None, &[], void).expect("begin_function");
    b.ret().expect("ret");
    b.end_function().expect("end_function");
    let module = b.into_module();
    assert!(
        !module.functions.is_empty(),
        "module should contain at least one function"
    );
}

#[test]
fn aggregate_forwarding_preserves_types_and_nested_paths() {
    for case in 0..3 {
        let mut b = SpirvBuilder::new();
        let scalar = b.f32_type();
        let pair = b.type_struct(vec![scalar, scalar]);
        let wide = b.type_struct(vec![pair, scalar, scalar]);
        let narrow = b.type_struct(vec![pair, scalar]);
        let result_type = if case == 1 { scalar } else { wide };
        let (_, p, _) = b.begin_function(None, &[wide], result_type).unwrap();
        let fields = [
            b.composite_extract(*pair, None, p[0], [0]).unwrap(),
            b.composite_extract(*scalar, None, p[0], [1]).unwrap(),
            b.composite_extract(*scalar, None, p[0], [2]).unwrap(),
        ];
        let result = match case {
            0 => b.composite_construct(*wide, None, fields).unwrap(),
            1 => {
                let subset = b.composite_construct(*narrow, None, [fields[0], fields[1]]).unwrap();
                b.composite_extract(*scalar, None, subset, [0, 1]).unwrap()
            }
            _ => b.composite_construct(*wide, None, [fields[0], fields[2], fields[1]]).unwrap(),
        };
        b.ret_value(result).unwrap();
        b.end_function().unwrap();
        let module = b.into_module();
        let instructions: Vec<_> = module.all_inst_iter().collect();
        if case == 0 {
            assert_eq!(instructions.last().unwrap().class.opcode, spirv::Op::FunctionEnd);
            assert!(instructions
                .iter()
                .any(|i| i.class.opcode == spirv::Op::ReturnValue
                    && i.operands == [dr::Operand::IdRef(p[0])]));
        } else {
            let definition = instructions.iter().find(|i| i.result_id == Some(result)).unwrap();
            let expected = if case == 1 {
                vec![dr::Operand::IdRef(fields[0]), dr::Operand::LiteralBit32(1)]
            } else {
                [fields[0], fields[2], fields[1]].into_iter().map(dr::Operand::IdRef).collect()
            };
            assert_eq!(definition.operands, expected);
        }
    }
}

#[test]
fn aggregate_forwarding_keeps_storage_layout_conversions() {
    let mut b = SpirvBuilder::new();
    let scalar = b.u32_type();
    let logical = b.type_struct(vec![scalar, scalar]);
    let storage = b.type_buffer_struct(vec![scalar, scalar], &[0, 4]);
    assert_ne!(logical, storage);
    let (_, p, _) = b.begin_function(None, &[storage], logical).unwrap();
    let x = b.composite_extract(*scalar, None, p[0], [0]).unwrap();
    let y = b.composite_extract(*scalar, None, p[0], [1]).unwrap();
    let converted = b.composite_construct(*logical, None, [x, y]).unwrap();
    b.ret_value(converted).unwrap();
    b.end_function().unwrap();
    let module = b.into_module();
    let definition = module.all_inst_iter().find(|i| i.result_id == Some(converted)).unwrap();
    assert_eq!(definition.class.opcode, spirv::Op::CompositeConstruct);
    assert_eq!(definition.result_type, Some(*logical));
}

#[test]
fn aggregate_forwarding_handles_scalar_vectors_arrays_and_matrices() {
    let mut b = SpirvBuilder::new();
    let scalar = b.f32_type();
    let vector = b.type_vec(scalar, 2);
    let count = b.const_u32(2);
    let array = b.type_array(vector, *count);
    let matrix = b.type_matrix(vector, 2);
    let (_, p, _) = b.begin_function(None, &[scalar, scalar, vector], scalar).unwrap();
    let v = b.composite_construct(*vector, None, [p[0], p[1]]).unwrap();
    let mut sum = p[0];
    for ty in [*array, *matrix] {
        let aggregate = b.composite_construct(ty, None, [v, p[2]]).unwrap();
        let extracted = b.composite_extract(*scalar, None, aggregate, [0, 1]).unwrap();
        sum = b.f_add(*scalar, None, sum, extracted).unwrap();
    }
    let large = b.type_vec(scalar, 4);
    let packed = b.composite_construct(*large, None, [v, p[2]]).unwrap();
    let lane = b.composite_extract(*scalar, None, packed, [1]).unwrap();
    // A vector constituent is not a scalar lane; packed vectors fall back.
    let result = b.f_add(*scalar, None, sum, lane).unwrap();
    b.ret_value(result).unwrap();
    b.end_function().unwrap();
    let module = b.into_module();
    let additions: Vec<_> = module.all_inst_iter().filter(|i| i.class.opcode == spirv::Op::FAdd).collect();
    assert_eq!(additions.len(), 3);
    assert!(additions[..2].iter().all(|i| i.operands.last() == Some(&dr::Operand::IdRef(p[1]))));
    let projection = module.all_inst_iter().find(|i| i.result_id == Some(lane)).unwrap();
    assert_eq!(
        projection.operands,
        [dr::Operand::IdRef(packed), dr::Operand::LiteralBit32(1)]
    );
}

#[test]
fn aggregate_forwarding_does_not_reuse_sibling_branch_definitions() {
    let mut b = SpirvBuilder::new();
    let scalar = b.f32_type();
    let pair = b.type_struct(vec![scalar, scalar]);
    let boolean = b.bool_type();
    let (_, p, _) = b.begin_function(None, &[boolean, scalar, scalar], scalar).unwrap();
    let yes = b.id();
    let no = b.id();
    let merge = b.id();
    b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
    b.branch_conditional(p[0], yes, no, []).unwrap();
    b.begin_block(Some(yes)).unwrap();
    let a = b.composite_construct(*pair, None, [p[1], p[2]]).unwrap();
    b.branch(merge).unwrap();
    b.begin_block(Some(no)).unwrap();
    let c = b.composite_construct(*pair, None, [p[1], p[2]]).unwrap();
    b.branch(merge).unwrap();
    b.begin_block(Some(merge)).unwrap();
    let phi = b.phi(*pair, None, [(a, yes), (c, no)]).unwrap();
    let projected = b.composite_extract(*scalar, None, phi, [0]).unwrap();
    b.ret_value(projected).unwrap();
    b.end_function().unwrap();
    let module = b.into_module();
    assert_eq!(
        module.all_inst_iter().filter(|i| i.class.opcode == spirv::Op::CompositeConstruct).count(),
        2
    );
}

#[test]
fn aggregate_forwarding_reconstructs_a_deferred_loop_phi() {
    let mut b = SpirvBuilder::new();
    let scalar = b.f32_type();
    let vector = b.type_vec(scalar, 2);
    let pair = b.type_struct(vec![scalar, vector]);
    let two = b.const_u32(2);
    let array = b.type_array(pair, *two);
    let condition = b.bool_type();
    let (_, p, entry) = b.begin_function(None, &[array, condition], array).unwrap();
    let header = b.id();
    let merge = b.id();
    let continuing = b.id();
    let phi = b.id();
    b.branch(header).unwrap();
    b.begin_block(Some(header)).unwrap();
    let header_index = b.selected_block().unwrap();
    let x = b.composite_extract(*pair, None, phi, [0]).unwrap();
    let y = b.composite_extract(*pair, None, phi, [1]).unwrap();
    let reconstructed = b.composite_construct(*array, None, [x, y]).unwrap();
    b.loop_merge(merge, continuing, spirv::LoopControl::NONE, []).unwrap();
    b.branch_conditional(p[1], continuing, merge, []).unwrap();
    b.begin_block(Some(continuing)).unwrap();
    b.branch(header).unwrap();
    b.begin_block(Some(merge)).unwrap();
    b.ret_value(reconstructed).unwrap();
    b.select_block(Some(header_index)).unwrap();
    b.insert_value_phi(*array, phi, &[(p[0], *entry), (phi, continuing)]).unwrap();
    b.end_function().unwrap();
    let module = b.into_module();
    let blocks = &module.functions[0].blocks;
    let header = &blocks[header_index];
    let phis: Vec<_> =
        header.instructions.iter().take_while(|i| i.class.opcode == spirv::Op::Phi).collect();
    assert_eq!(phis.len(), 4);
    assert!(phis.iter().all(|i| i.result_type == Some(*scalar) || i.result_type == Some(*vector)));
    // Every use in the header follows its definition, including nested reconstruction.
    let mut available: std::collections::HashSet<_> = phis.iter().filter_map(|i| i.result_id).collect();
    for instruction in header.instructions.iter().skip(phis.len()) {
        if instruction.class.opcode == spirv::Op::CompositeConstruct {
            for operand in &instruction.operands {
                let dr::Operand::IdRef(id) = operand else {
                    panic!("constituent")
                };
                assert!(available.contains(id));
            }
        }
        available.extend(instruction.result_id);
    }
    assert!(available.contains(&phi));
}

#[test]
fn aggregate_phis_preserve_merge_adjacency_and_phi_order() {
    let mut b = SpirvBuilder::new();
    let scalar = b.f32_type();
    let pair = b.type_struct(vec![scalar, scalar]);
    let condition = b.bool_type();
    let (_, p, entry) = b.begin_function(None, &[condition, pair, pair, scalar], pair).unwrap();
    let other = b.id();
    let merge = b.id();
    b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
    b.branch_conditional(p[0], merge, other, []).unwrap();
    b.begin_block(Some(other)).unwrap();
    b.branch(merge).unwrap();
    b.begin_block(Some(merge)).unwrap();
    let merge_index = b.selected_block().unwrap();
    let phi = b.id();
    b.ret_value(phi).unwrap();
    b.select_block(Some(merge_index)).unwrap();
    b.insert_value_phi(*pair, phi, &[(p[1], *entry), (p[2], other)]).unwrap();
    let scalar_phi = b.id();
    b.insert_value_phi(*scalar, scalar_phi, &[(p[3], *entry), (p[3], other)]).unwrap();
    b.end_function().unwrap();
    let module = b.into_module();
    let blocks = &module.functions[0].blocks;
    let entry_instructions = &blocks.iter().find(|b| b.label_id() == Some(*entry)).unwrap().instructions;
    assert_eq!(
        entry_instructions[entry_instructions.len() - 2].class.opcode,
        spirv::Op::SelectionMerge
    );
    let merge = &blocks.iter().find(|b| b.label_id() == Some(merge)).unwrap().instructions;
    assert_eq!(
        merge.iter().take_while(|i| i.class.opcode == spirv::Op::Phi).count(),
        3
    );
    assert_eq!(merge[3].result_id, Some(phi));
    assert!(merge[..3].iter().all(|i| i.result_type == Some(*scalar)));
}

#[test]
fn aggregate_cleanup_reuses_same_block_values_and_removes_dead_chains() {
    let mut b = SpirvBuilder::new();
    let scalar = b.f32_type();
    let pair = b.type_struct(vec![scalar, scalar]);
    let nested = b.type_struct(vec![pair]);
    let outer = b.type_struct(vec![nested]);
    let (_, p, _) = b.begin_function(None, &[pair], pair).unwrap();
    let x = b.composite_extract(*scalar, None, p[0], [0]).unwrap();
    let duplicate = b.composite_extract(*scalar, None, p[0], [0]).unwrap();
    let a = b.composite_construct(*pair, None, [x, duplicate]).unwrap();
    let other = b.composite_construct(*pair, None, [duplicate, x]).unwrap();
    let dead = b.composite_construct(*nested, None, [a]).unwrap();
    let dead_again = b.composite_construct(*outer, None, [dead]).unwrap();
    // The dead chain is deliberately unused. Forwarding alone retains it.
    let _ = dead_again;
    b.ret_value(other).unwrap();
    b.end_function().unwrap();
    let module = b.into_module();
    let instructions: Vec<_> = module.all_inst_iter().collect();
    assert_eq!(
        instructions.iter().filter(|i| i.class.opcode == spirv::Op::CompositeExtract).count(),
        1
    );
    assert_eq!(
        instructions.iter().filter(|i| i.class.opcode == spirv::Op::CompositeConstruct).count(),
        1
    );
    assert!(instructions
        .iter()
        .any(|i| i.class.opcode == spirv::Op::ReturnValue && i.operands == [dr::Operand::IdRef(a)]));
}

#[test]
fn dead_aggregate_cleanup_preserves_operand_evaluation_and_decorations() {
    let mut b = SpirvBuilder::new();
    let scalar = b.i32_type();
    let pair = b.type_struct(vec![scalar, scalar]);
    let (_, p, _) = b.begin_function(None, &[pair, scalar], scalar).unwrap();
    let x = b.composite_extract(*scalar, None, p[0], [0]).unwrap();
    let decorated = b.composite_extract(*scalar, None, p[0], [0]).unwrap();
    b.decorate(decorated, spirv::Decoration::RelaxedPrecision, []);
    let division = b.s_div(*scalar, None, x, p[1]).unwrap();
    b.composite_construct(*pair, None, [division, division]).unwrap();
    b.ret_value(x).unwrap();
    b.end_function().unwrap();
    let module = b.into_module();
    assert!(module.all_inst_iter().any(|i| i.result_id == Some(division)));
    assert!(module.all_inst_iter().any(|i| i.result_id == Some(decorated)));
    assert!(!module.all_inst_iter().any(|i| i.class.opcode == spirv::Op::CompositeConstruct));
    assert_eq!(
        module.all_inst_iter().filter(|i| i.class.opcode == spirv::Op::CompositeExtract).count(),
        2
    );
}
