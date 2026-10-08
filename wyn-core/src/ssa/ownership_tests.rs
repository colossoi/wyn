use super::*;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{ConstantValue, ControlHeader, Terminator, ValueId, ValueRef};
use crate::types::{i32, sized_array};

fn int(n: i32) -> ValueRef {
    ValueRef::Const(ConstantValue::I32(n))
}

fn update(builder: &mut FuncBuilder, array: ValueId) -> ValueId {
    builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Intrinsic {
                    id: catalog().known().array_with,
                    overload_idx: 0,
                },
                operands: vec![array.into(), int(0), int(7)],
            },
            sized_array(4, i32()),
        )
        .unwrap()
}

fn array(builder: &mut FuncBuilder) -> ValueId {
    builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::ArrayLit(4),
                operands: vec![int(1), int(2), int(3), int(4)],
            },
            sized_array(4, i32()),
        )
        .unwrap()
}

fn in_place(body: &FuncBody) -> usize {
    body.inner
        .insts
        .values()
        .filter(|node| {
            matches!(node.data,
        InstKind::Op { tag: OpTag::Intrinsic { id, .. }, .. }
            if id == catalog().known().array_with_in_place)
        })
        .count()
}

#[test]
fn chained_updates_reuse_a_dead_local_after_placement() {
    let mut builder = FuncBuilder::new(vec![], sized_array(4, i32()));
    let xs = array(&mut builder);
    let ys = update(&mut builder, xs);
    let zs = update(&mut builder, ys);
    builder.terminate(Terminator::Return(Some(zs.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    promote_updates(&mut body);
    assert_eq!(in_place(&body), 2);
}

#[test]
fn later_reads_of_the_old_value_prevent_mutation() {
    let mut builder = FuncBuilder::new(vec![], i32());
    let xs = array(&mut builder);
    update(&mut builder, xs);
    let old = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Index,
                operands: vec![xs.into(), int(0)],
            },
            i32(),
        )
        .unwrap();
    builder.terminate(Terminator::Return(Some(old.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    promote_updates(&mut body);
    assert_eq!(in_place(&body), 0);
}

#[test]
fn materialize_alias_keeps_the_original_live() {
    let mut builder = FuncBuilder::new(vec![], i32());
    let xs = array(&mut builder);
    let alias = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Materialize,
                operands: vec![xs.into()],
            },
            sized_array(4, i32()),
        )
        .unwrap();
    update(&mut builder, alias);
    let old = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Index,
                operands: vec![xs.into(), int(0)],
            },
            i32(),
        )
        .unwrap();
    builder.terminate(Terminator::Return(Some(old.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    promote_updates(&mut body);
    assert_eq!(in_place(&body), 0);
}

#[test]
fn earlier_scalar_reads_do_not_keep_a_local_live() {
    let mut builder = FuncBuilder::new(vec![], sized_array(4, i32()));
    let xs = array(&mut builder);
    builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Index,
                operands: vec![xs.into(), int(0)],
            },
            i32(),
        )
        .unwrap();
    let result = update(&mut builder, xs);
    builder.terminate(Terminator::Return(Some(result.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    promote_updates(&mut body);
    assert_eq!(in_place(&body), 1);
}

#[test]
fn dead_donors_from_a_dominating_block_can_be_overwritten() {
    let mut builder = FuncBuilder::new(vec![], sized_array(4, i32()));
    let xs = array(&mut builder);
    let inner = builder.create_block();
    builder
        .terminate(Terminator::Branch {
            target: inner,
            args: vec![],
        })
        .unwrap();
    builder.switch_to_block(inner).unwrap();
    let result = update(&mut builder, xs);
    builder.terminate(Terminator::Return(Some(result.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    promote_updates(&mut body);
    assert_eq!(in_place(&body), 1);
}

#[test]
fn an_escaping_local_is_not_overwritten() {
    let mut builder = FuncBuilder::new(vec![], sized_array(4, i32()));
    let xs = array(&mut builder);
    let place = builder.new_place(sized_array(4, i32()));
    builder
        .push_void_inst(InstKind::Alloca {
            elem_ty: sized_array(4, i32()),
            result: place,
        })
        .unwrap();
    builder
        .push_void_inst(InstKind::Store {
            place,
            value: xs.into(),
        })
        .unwrap();
    let result = update(&mut builder, xs);
    builder.terminate(Terminator::Return(Some(result.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    promote_updates(&mut body);
    assert_eq!(in_place(&body), 0);
}

fn read(builder: &mut FuncBuilder, array: ValueId) -> ValueId {
    builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Index,
                operands: vec![array.into(), int(0)],
            },
            i32(),
        )
        .unwrap()
}

#[test]
fn later_length_through_a_materialize_alias_does_not_keep_contents_live() {
    let mut builder = FuncBuilder::new(vec![], i32());
    let xs = array(&mut builder);
    update(&mut builder, xs);
    let alias = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Materialize,
                operands: vec![xs.into()],
            },
            sized_array(4, i32()),
        )
        .unwrap();
    let length = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Intrinsic {
                    id: catalog().known().length,
                    overload_idx: 0,
                },
                operands: vec![alias.into()],
            },
            i32(),
        )
        .unwrap();
    builder.terminate(Terminator::Return(Some(length.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    promote_updates(&mut body);
    assert_eq!(in_place(&body), 1);
}

#[test]
fn a_later_read_in_a_successor_keeps_contents_live() {
    let mut builder = FuncBuilder::new(vec![], i32());
    let xs = array(&mut builder);
    update(&mut builder, xs);
    let next = builder.create_block();
    builder
        .terminate(Terminator::Branch {
            target: next,
            args: vec![],
        })
        .unwrap();
    builder.switch_to_block(next).unwrap();
    let old = read(&mut builder, xs);
    builder.terminate(Terminator::Return(Some(old.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    promote_updates(&mut body);
    assert_eq!(in_place(&body), 0);
}

#[test]
fn an_earlier_read_in_a_dominating_block_does_not_keep_contents_live() {
    let mut builder = FuncBuilder::new(vec![], sized_array(4, i32()));
    let xs = array(&mut builder);
    read(&mut builder, xs);
    let next = builder.create_block();
    builder
        .terminate(Terminator::Branch {
            target: next,
            args: vec![],
        })
        .unwrap();
    builder.switch_to_block(next).unwrap();
    let result = update(&mut builder, xs);
    builder.terminate(Terminator::Return(Some(result.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    promote_updates(&mut body);
    assert_eq!(in_place(&body), 1);
}

#[test]
fn an_old_alias_observed_after_a_chain_prevents_reusing_that_alias() {
    let mut builder = FuncBuilder::new(vec![], i32());
    let xs = array(&mut builder);
    let ys = update(&mut builder, xs);
    let alias = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Materialize,
                operands: vec![ys.into()],
            },
            sized_array(4, i32()),
        )
        .unwrap();
    update(&mut builder, alias);
    let old = read(&mut builder, ys);
    builder.terminate(Terminator::Return(Some(old.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    promote_updates(&mut body);
    // xs can be consumed to produce ys; ys must survive the second update.
    assert_eq!(in_place(&body), 1);
}

#[test]
fn loop_reuse_requires_storage_recreated_in_the_same_iteration() {
    for loop_local in [false, true] {
        let mut builder = FuncBuilder::new(vec![], i32());
        let invariant = (!loop_local).then(|| array(&mut builder));
        let header = builder.create_block();
        let inner = builder.create_block();
        let exit = builder.create_block();
        builder
            .terminate(Terminator::Branch {
                target: header,
                args: vec![],
            })
            .unwrap();
        builder.switch_to_block(header).unwrap();
        builder.func_mut().blocks[header].control_header = Some(ControlHeader::Loop {
            merge: exit,
            continue_block: inner,
        });
        let xs = invariant.unwrap_or_else(|| array(&mut builder));
        read(&mut builder, xs);
        builder
            .terminate(Terminator::CondBranch {
                cond: ValueRef::Const(ConstantValue::Bool(true)),
                then_target: inner,
                then_args: vec![],
                else_target: exit,
                else_args: vec![],
            })
            .unwrap();
        builder.switch_to_block(inner).unwrap();
        update(&mut builder, xs);
        builder
            .terminate(Terminator::Branch {
                target: header,
                args: vec![],
            })
            .unwrap();
        builder.switch_to_block(exit).unwrap();
        builder.terminate(Terminator::Return(Some(int(0)))).unwrap();
        let mut body = builder.finish().unwrap();
        promote_updates(&mut body);
        assert_eq!(
            in_place(&body),
            usize::from(loop_local),
            "loop-local donor: {loop_local}"
        );
    }
}

#[test]
fn forwarded_arrays_keep_their_backing_storage_live() {
    let mut builder = FuncBuilder::new(vec![], i32());
    let xs = array(&mut builder);
    let tuple = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Tuple(1),
                operands: vec![xs.into()],
            },
            crate::types::tuple(vec![sized_array(4, i32())]),
        )
        .unwrap();
    update(&mut builder, xs);
    let projected = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Project { index: 0 },
                operands: vec![tuple.into()],
            },
            sized_array(4, i32()),
        )
        .unwrap();
    let old = read(&mut builder, projected);
    builder.terminate(Terminator::Return(Some(old.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    promote_updates(&mut body);
    assert_eq!(in_place(&body), 0);
}

#[test]
fn function_parameters_are_invariant_even_when_entry_is_a_loop_header() {
    let mut builder = FuncBuilder::new(vec![(sized_array(4, i32()), "xs".into())], i32());
    let xs = builder.get_param(0);
    let header = builder.entry();
    let inner = builder.create_block();
    let exit = builder.create_block();
    builder.func_mut().blocks[header].control_header = Some(ControlHeader::Loop {
        merge: exit,
        continue_block: inner,
    });
    read(&mut builder, xs);
    builder
        .terminate(Terminator::CondBranch {
            cond: ValueRef::Const(ConstantValue::Bool(true)),
            then_target: inner,
            then_args: vec![],
            else_target: exit,
            else_args: vec![],
        })
        .unwrap();
    builder.switch_to_block(inner).unwrap();
    update(&mut builder, xs);
    builder
        .terminate(Terminator::Branch {
            target: header,
            args: vec![],
        })
        .unwrap();
    builder.switch_to_block(exit).unwrap();
    builder.terminate(Terminator::Return(Some(int(0)))).unwrap();
    let mut body = builder.finish().unwrap();
    promote_updates(&mut body);
    assert_eq!(in_place(&body), 0);
}

#[test]
fn materializing_an_addressable_constant_does_not_grant_reuse() {
    let mut builder = FuncBuilder::new(vec![], sized_array(4, i32()));
    let constant = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::AddressableConstant(crate::op::AddressableConstantId(0)),
                operands: vec![],
            },
            sized_array(4, i32()),
        )
        .unwrap();
    let alias = builder
        .push_inst(
            InstKind::Op {
                tag: OpTag::Materialize,
                operands: vec![constant.into()],
            },
            sized_array(4, i32()),
        )
        .unwrap();
    let result = update(&mut builder, alias);
    builder.terminate(Terminator::Return(Some(result.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    promote_updates(&mut body);
    assert_eq!(in_place(&body), 0);
}
