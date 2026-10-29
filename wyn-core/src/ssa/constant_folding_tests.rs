use super::*;
use crate::builtins::catalog;
use crate::op::BinaryOperator;
use crate::ssa;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::Terminator;
use crate::types;

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
            tag: OpTag::Vector,
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
        InstKind::Op { tag: OpTag::Vector, operands }
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
