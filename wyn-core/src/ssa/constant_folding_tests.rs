use super::*;
use crate::builtins::catalog;
use crate::op::BinaryOperator;
use crate::ssa;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::Terminator;
use crate::types;
use wspirv::dr;

const FIXTURE: &str = include_str!("../../../testfiles/constant_shader_math.wyn");

#[test]
fn constant_shader_math_folds_after_inlining_and_preserves_dynamic_depth_order() {
    let program = crate::compile_thru_ssa(FIXTURE).unwrap();
    let placed = ssa::place_floating(ssa::optimize(program.clone())).unwrap();
    let entry = placed.entry_points.iter().find(|e| e.name == "constant_math_compute").unwrap();
    let function = &entry.body.inner;
    let depth = ValueRef::Const(ConstantValue::from_f32(1.0 / (0.1f32 - 1000.0)));
    let first = function
        .insts
        .values()
        .find_map(|node| match &node.data {
            InstKind::Op {
                tag: OpTag::BinOp(BinaryOperator::Multiply),
                operands,
            } if operands.get(1) == Some(&ValueRef::Const(ConstantValue::from_f32(1000.0))) => node.result,
            _ => None,
        })
        .expect("p.z * far must retain its own rounding step");
    assert!(function.insts.values().any(|node| matches!(&node.data,
        InstKind::Op { tag: OpTag::BinOp(BinaryOperator::Multiply), operands }
        if operands == &[first.into(), depth])));

    let lowered = crate::lower_ssa_to_spirv(program.clone()).unwrap();
    let module = dr::load_words(&lowered.spirv).unwrap();
    let compute = entry_function(&module, "constant_math_compute");
    let count = |op| instructions(compute).filter(|i| i.class.opcode == op).count();
    assert_eq!(
        count(spirv::Op::FDiv),
        2,
        "only dynamic aspect/projection divisions remain"
    );
    assert_eq!(count(spirv::Op::FSub), 0);
    assert!(!instructions(compute).any(|i| matches!(
        i.operands.get(1),
        Some(dr::Operand::LiteralExtInstInteger(11 | 15 | 46))
    )));

    let wgsl = crate::lower_ssa_to_wgsl(program).unwrap();
    assert!(!wgsl.contains("radians("));
    assert!(!wgsl.contains("tan("));
    assert!(!wgsl.contains("mix("));
    validate(naga::front::wgsl::parse_str(&wgsl).unwrap());
    let bytes = lowered.spirv.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>();
    validate(naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap());
}

#[test]
fn unused_shader_inputs_keep_their_interfaces_without_runtime_setup() {
    let program = crate::compile_thru_ssa(FIXTURE).unwrap();
    // Check before SSA cleanup: lowering should never manufacture dead views
    // merely to preserve an entry's resource declarations.
    for (name, expected_lengths) in [("unused_inputs_vertex", 1), ("unused_inputs_fragment", 0)] {
        let entry = program.entry_points.iter().find(|e| e.name == name).unwrap();
        assert_eq!(entry.inputs.len(), 4);
        assert_eq!(
            entry.inputs.iter().filter(|i| i.storage_binding().is_some()).count(),
            2
        );
        assert_eq!(
            entry.inputs.iter().filter(|i| i.uniform_binding().is_some()).count(),
            1
        );
        assert_eq!(
            entry
                .body
                .inner
                .insts
                .values()
                .filter(|node| matches!(&node.data,
            InstKind::Op { tag: OpTag::Intrinsic { id, .. }, .. } if *id == catalog().known().storage_len))
                .count(),
            expected_lengths
        );
    }
    let lowered = crate::lower_ssa_to_spirv(program).unwrap();
    let module = dr::load_words(&lowered.spirv).unwrap();
    for (name, lengths, loads) in [("unused_inputs_vertex", 1, 2), ("unused_inputs_fragment", 0, 1)] {
        let function = entry_function(&module, name);
        assert_eq!(
            instructions(function).filter(|i| i.class.opcode == spirv::Op::ArrayLength).count(),
            lengths
        );
        // Vertex index + point, or fragment color. Neither shader reads config.
        assert_eq!(
            instructions(function).filter(|i| i.class.opcode == spirv::Op::Load).count(),
            loads
        );
    }
}

fn entry_function<'a>(module: &'a dr::Module, name: &str) -> &'a dr::Function {
    let entry = module
        .entry_points
        .iter()
        .find(|e| matches!(e.operands.get(2), Some(dr::Operand::LiteralString(n)) if n == name))
        .unwrap();
    let dr::Operand::IdRef(id) = entry.operands[1] else {
        panic!("entry id")
    };
    module.functions.iter().find(|f| f.def.as_ref().and_then(|i| i.result_id) == Some(id)).unwrap()
}

fn instructions(function: &dr::Function) -> impl Iterator<Item = &dr::Instruction> {
    function.blocks.iter().flat_map(|b| &b.instructions)
}

fn validate(module: naga::Module) {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn late_folding_preserves_signed_zero_and_leaves_overflow_residual() {
    let ty = Type::Constructed(TypeName::Float(32), vec![]);
    for (x, folded_bits) in [(-0.0f32, Some((-0.0f32).to_bits())), (f32::MAX, None)] {
        let mut body = FuncBuilder::new(vec![], ty.clone()).finish_unchecked();
        let first = body.inner.append_inst(
            body.inner.entry,
            InstKind::Op {
                tag: OpTag::BinOp(BinaryOperator::Multiply),
                operands: vec![
                    ValueRef::Const(ConstantValue::from_f32(x)),
                    ValueRef::Const(ConstantValue::from_f32(2.0)),
                ],
            },
            ty.clone(),
        );
        let second = body.inner.append_inst(
            body.inner.entry,
            InstKind::Op {
                tag: OpTag::BinOp(BinaryOperator::Divide),
                operands: vec![first.into(), ValueRef::Const(ConstantValue::from_f32(2.0))],
            },
            ty.clone(),
        );
        body.inner.blocks[body.inner.entry].term = Terminator::Return(Some(second.into()));
        fold(&mut body);
        if let Some(bits) = folded_bits {
            assert_eq!(body.num_insts(), 0);
            assert!(
                matches!(body.inner.blocks[body.inner.entry].term, Terminator::Return(Some(ValueRef::Const(ConstantValue::F32(actual)))) if actual == bits)
            );
        } else {
            assert_eq!(body.num_insts(), 2, "overflow must not be hidden by cancellation");
        }
    }
}

#[test]
fn late_vector_conversions_preserve_unsigned_lane_values() {
    let signed = types::vec(2, types::i32());
    let unsigned = types::vec(2, Type::Constructed(TypeName::UInt(32), vec![]));
    let float = types::vec(2, types::f32());
    let mut body = FuncBuilder::new(vec![], float.clone()).finish_unchecked();
    let mut value = body.inner.append_inst(
        body.inner.entry,
        InstKind::Op {
            tag: OpTag::Vector(2),
            operands: vec![ValueRef::Const(ConstantValue::I32(-1)); 2],
        },
        signed,
    );
    for (source, target, ty) in [
        (TypeName::Int(32), TypeName::UInt(32), unsigned),
        (TypeName::UInt(32), TypeName::Float(32), float),
    ] {
        value = body.inner.append_inst(
            body.inner.entry,
            InstKind::Op {
                tag: OpTag::Intrinsic {
                    id: catalog().conversion(&target, &source).unwrap(),
                    overload_idx: 0,
                },
                operands: vec![value.into()],
            },
            ty,
        );
    }
    body.inner.blocks[body.inner.entry].term = Terminator::Return(Some(value.into()));
    fold(&mut body);
    let result = body.inner.inst_of_value(value).unwrap();
    assert!(matches!(&body.inner.insts[result].data,
        InstKind::Op { tag: OpTag::Vector(2), operands }
        if operands == &[ValueRef::Const(ConstantValue::from_f32(u32::MAX as f32)); 2]));
}

#[test]
fn dead_view_cleanup_removes_its_length_query_but_retains_live_views() {
    let binding = BindingRef::new(0, 0);
    let view_ty = types::view_array_of(&types::i32(), types::buffer_tag(binding));
    for live in [false, true] {
        let mut body = FuncBuilder::new(vec![], view_ty.clone()).finish_unchecked();
        let length = body.inner.append_inst(
            body.inner.entry,
            InstKind::Op {
                tag: OpTag::Intrinsic {
                    id: catalog().known().storage_len,
                    overload_idx: 0,
                },
                operands: vec![ValueRef::Const(ConstantValue::U32(0)); 2],
            },
            Type::Constructed(TypeName::UInt(32), vec![]),
        );
        let view = body.inner.append_inst(
            body.inner.entry,
            InstKind::Op {
                tag: OpTag::StorageView(crate::op::PureViewSource::Storage(binding)),
                operands: vec![ValueRef::Const(ConstantValue::U32(0)), length.into()],
            },
            view_ty.clone(),
        );
        body.inner.blocks[body.inner.entry].term = Terminator::Return(live.then_some(view.into()));
        ssa::eliminate_dead_pure_instructions(&mut body);
        assert_eq!(body.num_insts(), if live { 2 } else { 0 });
    }
}
