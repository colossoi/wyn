use super::*;
use crate::builtins::catalog;
use crate::op::BinaryOperator;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::ConstantValue;
use crate::types::{self, Type};

fn diamond(ty: Type) -> (FuncBody, BlockId, BlockId, BlockId) {
    let mut body = FuncBuilder::new(
        vec![
            (bool_type(), "c".into()),
            (ty.clone(), "x".into()),
            (ty.clone(), "y".into()),
        ],
        ty.clone(),
    )
    .finish_unchecked();
    let h = body.inner.entry;
    let yes = body.inner.create_block();
    let no = body.inner.create_block();
    let merge = body.inner.create_block();
    let result = body.inner.add_block_param(merge, ty);
    body.inner.blocks[h].control_header = Some(ControlHeader::Selection { merge });
    body.inner.blocks[h].term = Terminator::CondBranch {
        cond: body.inner.params[0].into(),
        then_target: yes,
        then_args: vec![],
        else_target: no,
        else_args: vec![],
    };
    for (block, param) in [(yes, 1), (no, 2)] {
        body.inner.blocks[block].term = Terminator::Branch {
            target: merge,
            args: vec![body.inner.params[param].into()],
        };
    }
    body.inner.blocks[merge].term = Terminator::Return(Some(result.into()));
    (body, yes, no, merge)
}

fn select_count(body: &FuncBody) -> usize {
    body.inner
        .insts
        .values()
        .filter(|node| {
            matches!(node.data,
        InstKind::Op { tag: OpTag::Intrinsic { id, .. }, .. } if id == catalog().known().select)
        })
        .count()
}

fn binary(body: &mut FuncBody, block: BlockId, op: BinaryOperator, operand: ValueRef) -> ValueRef {
    body.inner
        .append_inst(
            block,
            InstKind::Op {
                tag: OpTag::BinOp(op),
                operands: vec![operand, body.inner.params[1].into()],
            },
            types::i32(),
        )
        .into()
}

fn arm_result(body: &mut FuncBody, block: BlockId, value: ValueRef) {
    let Terminator::Branch { args, .. } = &mut body.inner.blocks[block].term else {
        panic!("arm")
    };
    args[0] = value;
}

#[test]
fn select_diamond_uses_existing_values_and_removes_merge_parameters() {
    for ty in [types::i32(), bool_type(), types::vec(4, types::f32())] {
        let (mut body, _, _, merge) = diamond(ty);
        let expected = vec![
            body.inner.params[2].into(),
            body.inner.params[1].into(),
            body.inner.params[0].into(),
        ];
        convert(&mut body, 0);
        assert_eq!(body.num_blocks(), 2);
        assert_eq!(select_count(&body), 1);
        assert!(body.inner.blocks[merge].params.is_empty());
        let InstKind::Op { operands, .. } = &body.inner.insts.values().next().unwrap().data else {
            panic!("select")
        };
        assert_eq!(operands, &expected);
    }
}

#[test]
fn select_conversion_limits_new_work_and_preserves_guarded_partial_operations() {
    for (op, count, expected) in [
        (BinaryOperator::Add, 4, 1),
        (BinaryOperator::Add, 5, 0),
        (BinaryOperator::Divide, 1, 0),
        (BinaryOperator::Remainder, 1, 0),
        (BinaryOperator::ShiftLeft, 1, 0),
    ] {
        let (mut body, yes, _, _) = diamond(types::i32());
        let mut value = body.inner.params[1].into();
        for _ in 0..count {
            value = binary(&mut body, yes, op, value);
        }
        arm_result(&mut body, yes, value);
        run(&mut body);
        assert_eq!(select_count(&body), expected, "{op:?}, {count}");
        if expected == 0 {
            assert!(body.inner.blocks.contains_key(yes));
        }
    }
}

#[test]
fn select_conversion_accepts_already_evaluated_division() {
    let (mut body, yes, _, _) = diamond(types::i32());
    let header = body.inner.entry;
    let x = body.inner.params[1].into();
    let divided = binary(&mut body, header, BinaryOperator::Divide, x);
    arm_result(&mut body, yes, divided);
    convert(&mut body, 0);
    assert_eq!(select_count(&body), 1);
    assert_eq!(body.inner.block_of_value(divided.as_ssa().unwrap()), Some(header));
}

#[test]
fn select_conversion_rejects_effects_shared_blocks_and_aggregate_results() {
    let (mut body, yes, _, _) = diamond(types::i32());
    body.inner.append_void_inst(yes, InstKind::ControlBarrier);
    run(&mut body);
    assert_eq!(select_count(&body), 0);
    let (mut body, yes, _, _) = diamond(types::i32());
    let shared = body.inner.create_block();
    body.inner.blocks[shared].term = Terminator::Branch {
        target: yes,
        args: vec![],
    };
    run(&mut body);
    assert_eq!(select_count(&body), 0);
    let (mut body, _, _, _) = diamond(types::sized_array(3, types::i32()));
    run(&mut body);
    assert_eq!(select_count(&body), 0);
}

#[test]
fn select_conversion_preserves_other_constructs_structural_targets() {
    let (mut body, yes, _, merge) = diamond(types::i32());
    let outer = body.inner.create_block();
    body.inner.blocks[outer].control_header = Some(ControlHeader::Loop {
        merge,
        continue_block: yes,
    });
    run(&mut body);
    assert_eq!(select_count(&body), 0);
}

#[test]
fn select_conversion_rejects_memory_partial_math_derivatives_and_expensive_math() {
    for name in ["f32.sqrt", "f32.d_fdx", "f32.sin"] {
        let (mut body, yes, _, _) = diamond(types::f32());
        let value = body.inner.append_inst(
            yes,
            InstKind::Op {
                tag: OpTag::Intrinsic {
                    id: catalog().lookup_by_any_name(name).unwrap().id,
                    overload_idx: 0,
                },
                operands: vec![body.inner.params[1].into()],
            },
            types::f32(),
        );
        arm_result(&mut body, yes, value.into());
        run(&mut body);
        assert_eq!(select_count(&body), 0, "{name}");
    }
    let (mut body, yes, _, _) = diamond(types::i32());
    let place = body.places.insert(crate::ssa::types::PlaceInfo {
        elem_ty: types::i32(),
        origin: crate::ssa::types::PlaceOrigin::Parameter { index: 1 },
    });
    let loaded = body.inner.append_inst(yes, InstKind::Load { place }, types::i32());
    arm_result(&mut body, yes, loaded.into());
    run(&mut body);
    assert_eq!(select_count(&body), 0);
}

#[test]
fn select_folding_keeps_eager_nondiscardable_producers() {
    let mut body = FuncBuilder::new(vec![(types::i32(), "x".into())], types::i32()).finish_unchecked();
    let h = body.inner.entry;
    let called = body.inner.append_inst(
        h,
        InstKind::Op {
            tag: OpTag::Intrinsic {
                id: catalog().known().uninit,
                overload_idx: 0,
            },
            operands: vec![],
        },
        types::i32(),
    );
    let value = body.inner.append_inst(
        h,
        InstKind::select(
            called.into(),
            body.inner.params[0].into(),
            ValueRef::Const(ConstantValue::Bool(true)),
        ),
        types::i32(),
    );
    body.inner.blocks[h].term = Terminator::Return(Some(value.into()));
    super::super::constant_folding::fold(&mut body);
    super::super::eliminate_dead_pure_instructions(&mut body);
    assert_eq!(select_count(&body), 0);
    assert!(body.inner.values.contains_key(called));
}

#[test]
fn select_conversion_handles_multiple_results_and_identical_alternatives() {
    let (mut body, yes, no, merge) = diamond(types::i32());
    let common = body.inner.params[1];
    let extra = body.inner.add_block_param(merge, types::i32());
    for block in [yes, no] {
        let Terminator::Branch { args, .. } = &mut body.inner.blocks[block].term else {
            panic!("arm")
        };
        args.push(common.into());
    }
    let first = body.inner.blocks[merge].params[0];
    let result = body.inner.append_inst(
        merge,
        InstKind::Op {
            tag: OpTag::BinOp(BinaryOperator::Add),
            operands: vec![first.into(), extra.into()],
        },
        types::i32(),
    );
    body.inner.blocks[merge].term = Terminator::Return(Some(result.into()));
    run(&mut body);
    assert_eq!(select_count(&body), 1);
    assert!(body.inner.blocks[merge].params.is_empty());
    assert!(!body.inner.values.contains_key(extra));
}

#[test]
fn select_conversion_stays_inside_possibly_empty_loop() {
    let (mut body, yes, _, merge) = diamond(types::i32());
    let selection = body.inner.entry;
    let preheader = body.inner.create_block();
    let loop_header = body.inner.create_block();
    let exit = body.inner.create_block();
    body.inner.entry = preheader;
    let state = body.inner.add_block_param(loop_header, types::i32());
    body.inner.blocks[preheader].term = Terminator::Branch {
        target: loop_header,
        args: vec![body.inner.params[1].into()],
    };
    body.inner.blocks[loop_header].control_header = Some(ControlHeader::Loop {
        merge: exit,
        continue_block: merge,
    });
    body.inner.blocks[loop_header].term = Terminator::CondBranch {
        cond: body.inner.params[0].into(),
        then_target: selection,
        then_args: vec![],
        else_target: exit,
        else_args: vec![],
    };
    let value = binary(&mut body, yes, BinaryOperator::Add, state.into());
    arm_result(&mut body, yes, value);
    body.inner.blocks[merge].term = Terminator::Branch {
        target: loop_header,
        args: vec![body.inner.blocks[merge].params[0].into()],
    };
    body.inner.blocks[exit].term = Terminator::Return(Some(state.into()));
    run(&mut body);
    assert_eq!(select_count(&body), 1);
    assert_eq!(
        body.inner.block_of_value(value.as_ssa().unwrap()),
        Some(selection)
    );
    assert!(body.inner.blocks[preheader].insts.is_empty());
    assert!(matches!(
        body.inner.blocks[loop_header].control_header,
        Some(ControlHeader::Loop { .. })
    ));
}

#[test]
fn select_fold_preserves_selected_float_bits() {
    for bits in [0x8000_0000, 0x7fc0_1234, 0x7f80_0000] {
        let mut body = FuncBuilder::new(vec![], types::f32()).finish_unchecked();
        let h = body.inner.entry;
        let chosen = ValueRef::Const(ConstantValue::F32(bits));
        let result = body.inner.append_inst(
            h,
            InstKind::select(
                ValueRef::Const(ConstantValue::F32(0)),
                chosen,
                ValueRef::Const(ConstantValue::Bool(true)),
            ),
            types::f32(),
        );
        body.inner.blocks[h].term = Terminator::Return(Some(result.into()));
        super::super::constant_folding::fold(&mut body);
        assert!(matches!(body.inner.blocks[h].term, Terminator::Return(Some(value)) if value == chosen));
    }
}

#[test]
fn select_codegen_and_guarded_fallback_validate_on_both_backends() {
    for (source, expected) in [
        ("entry choose(c:bool,x:i32,y:i32) i32 = if c then x else y", true),
        ("entry choose(c:bool) i32 = if c then 1 else 0", true),
        (
            "entry choose(c:bool,d:bool,x:i32,y:i32,z:i32) i32 = if c then (if d then x else y) else z",
            true,
        ),
        (
            "entry choose(c:bool,x:vec4f32,y:vec4f32) vec4f32 = if c then x else y",
            true,
        ),
        (
            "entry choose(c:bool,x:i32,y:i32) i32 = if c then x/y else 0",
            false,
        ),
    ] {
        let program = crate::compile_thru_ssa(source).unwrap();
        let wgsl = crate::lower_ssa_to_wgsl(program.clone()).unwrap();
        assert_eq!(wgsl.contains("select("), expected, "{wgsl}");
        assert_eq!(wgsl.contains("if "), !expected, "{wgsl}");
        let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let words = crate::lower_ssa_to_spirv(program).unwrap().spirv;
        let module = wspirv::dr::load_words(&words).unwrap();
        let instructions = module
            .functions
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.instructions)
            .collect::<Vec<_>>();
        assert_eq!(
            instructions.iter().any(|i| i.class.opcode == spirv::Op::Select),
            expected
        );
        assert_eq!(
            instructions.iter().any(|i| i.class.opcode == spirv::Op::SelectionMerge),
            !expected
        );
        let bytes = words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>();
        let module = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
