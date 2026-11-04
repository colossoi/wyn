use super::*;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{ConstantValue, PlaceInfo, PlaceOrigin, Terminator, ValueRef};
use crate::types::{self, TypeName};

fn apply(body: &mut FuncBody) -> Result<()> {
    super::optimize_body(body, &LookupMap::default())
}

fn buffer(body: &mut FuncBody, block: BlockId, id: u32, count: u32) -> ValueRef {
    body.inner
        .append_inst(
            block,
            InstKind::Op {
                tag: OpTag::StorageView(PureViewSource::Workgroup { id, count }),
                operands: vec![
                    ValueRef::Const(ConstantValue::U32(0)),
                    ValueRef::Const(ConstantValue::U32(count)),
                ],
            },
            types::view_array_with_size(
                &types::i32(),
                Type::Constructed(TypeName::SizePlaceholder, vec![]),
                types::no_buffer(),
            ),
        )
        .into()
}

fn access(body: &mut FuncBody, block: BlockId, view: ValueRef, write: bool) {
    let place = body.places.insert(PlaceInfo {
        elem_ty: types::i32(),
        origin: PlaceOrigin::Instruction,
    });
    body.inner.append_void_inst(
        block,
        InstKind::ViewIndex {
            view,
            index: ValueRef::Const(ConstantValue::I32(0)),
            result: place,
        },
    );
    if write {
        body.inner.append_void_inst(
            block,
            InstKind::Store {
                place,
                value: ValueRef::Const(ConstantValue::I32(7)),
            },
        );
    } else {
        body.inner.append_inst(block, InstKind::Load { place }, types::i32());
    }
}

fn allocations(body: &FuncBody) -> LookupSet<u32> {
    body.inner
        .insts
        .values()
        .filter_map(|node| match node.data {
            InstKind::Op {
                tag: OpTag::StorageView(PureViewSource::Workgroup { id, .. }),
                ..
            } => Some(id),
            _ => None,
        })
        .collect()
}

fn barriers(body: &FuncBody) -> usize {
    body.inner.insts.values().filter(|node| matches!(node.data, InstKind::ControlBarrier)).count()
}

#[test]
fn reuse_requires_all_lanes_to_finish_and_preserves_that_barrier() {
    for synchronize in [false, true] {
        let mut body = FuncBuilder::new(vec![], types::unit()).finish_unchecked();
        let entry = body.inner.entry;
        let a = buffer(&mut body, entry, 0, 65);
        access(&mut body, entry, a, true);
        if synchronize {
            body.inner.append_void_inst(entry, InstKind::ControlBarrier);
        }
        let b = buffer(&mut body, entry, 1, 17);
        access(&mut body, entry, b, true);
        apply(&mut body).unwrap();
        assert_eq!(allocations(&body).len(), if synchronize { 1 } else { 2 });
        assert_eq!(barriers(&body), usize::from(synchronize));
    }
}

#[test]
fn old_contents_reached_through_a_descriptor_alias_prevent_reuse() {
    let mut body = FuncBuilder::new(vec![], types::unit()).finish_unchecked();
    let entry = body.inner.entry;
    let a = buffer(&mut body, entry, 0, 65);
    access(&mut body, entry, a, true);
    body.inner.append_void_inst(entry, InstKind::ControlBarrier);
    let b = buffer(&mut body, entry, 1, 65);
    access(&mut body, entry, b, true);
    body.inner.append_void_inst(entry, InstKind::ControlBarrier);
    let alias = body.inner.append_inst(
        entry,
        InstKind::Op {
            tag: OpTag::StorageView(PureViewSource::Inherited),
            operands: vec![
                ValueRef::Const(ConstantValue::U32(0)),
                ValueRef::Const(ConstantValue::U32(65)),
                a,
            ],
        },
        types::view_array_with_size(
            &types::i32(),
            Type::Constructed(TypeName::SizePlaceholder, vec![]),
            types::no_buffer(),
        ),
    );
    access(&mut body, entry, alias.into(), false);
    access(&mut body, entry, b, false);
    apply(&mut body).unwrap();
    assert_eq!(allocations(&body).len(), 2);
    // The final barrier publishes both independent buffers.
    assert_eq!(barriers(&body), 1);
}

#[test]
fn barrier_coalescing_across_blocks_keeps_the_memory_dependency() {
    let mut body = FuncBuilder::new(vec![], types::unit()).finish_unchecked();
    let entry = body.inner.entry;
    let middle = body.inner.create_block();
    let exit = body.inner.create_block();
    let a = buffer(&mut body, entry, 0, 65);
    access(&mut body, entry, a, true);
    body.inner.append_void_inst(entry, InstKind::ControlBarrier);
    body.inner.blocks[entry].term = Terminator::Branch {
        target: middle,
        args: vec![],
    };
    body.inner.append_void_inst(middle, InstKind::ControlBarrier);
    body.inner.blocks[middle].term = Terminator::Branch {
        target: exit,
        args: vec![],
    };
    access(&mut body, exit, a, false);
    body.inner.append_void_inst(exit, InstKind::ControlBarrier);
    apply(&mut body).unwrap();
    assert_eq!(barriers(&body), 1);
}

#[test]
fn next_iteration_reads_keep_loop_storage_live() {
    let mut body = FuncBuilder::new(vec![], types::unit()).finish_unchecked();
    let entry = body.inner.entry;
    let header = body.inner.create_block();
    let iteration = body.inner.create_block();
    let exit = body.inner.create_block();
    let a = buffer(&mut body, entry, 0, 65);
    access(&mut body, entry, a, true);
    body.inner.append_void_inst(entry, InstKind::ControlBarrier);
    body.inner.blocks[entry].term = Terminator::Branch {
        target: header,
        args: vec![],
    };
    access(&mut body, header, a, false);
    body.inner.blocks[header].term = Terminator::CondBranch {
        cond: ValueRef::Const(ConstantValue::Bool(true)),
        then_target: iteration,
        then_args: vec![],
        else_target: exit,
        else_args: vec![],
    };
    body.inner.append_void_inst(iteration, InstKind::ControlBarrier);
    let b = buffer(&mut body, iteration, 1, 65);
    access(&mut body, iteration, b, true);
    body.inner.append_void_inst(iteration, InstKind::ControlBarrier);
    body.inner.blocks[iteration].term = Terminator::Branch {
        target: header,
        args: vec![],
    };
    apply(&mut body).unwrap();
    assert_eq!(allocations(&body).len(), 2);
}

#[test]
fn transitive_call_effects_distinguish_scalar_helpers_from_shared_memory() {
    use crate::ast::Span;
    use crate::ssa::types::{Function, Program};
    let helper = |index, calls: &[FunctionId], shared| {
        let mut body = FuncBuilder::new(vec![], types::unit()).finish_unchecked();
        let entry = body.inner.entry;
        for &callee in calls {
            body.inner.append_inst(
                entry,
                InstKind::Op {
                    tag: OpTag::Call(callee),
                    operands: vec![],
                },
                types::unit(),
            );
        }
        if shared {
            let a = buffer(&mut body, entry, 0, 65);
            access(&mut body, entry, a, true);
        }
        Function {
            id: FunctionId::from_index(index),
            name: format!("helper_{index}"),
            body,
            span: Span::generated(),
            linkage_name: None,
        }
    };
    let scalar = FunctionId::from_index(0);
    let wrapper = FunctionId::from_index(1);
    let writer = FunctionId::from_index(2);
    let unknown = FunctionId::from_index(3);
    let program = Program::bare(
        vec![
            helper(0, &[], false),
            helper(1, &[scalar], false),
            helper(2, &[], true),
            helper(3, &[FunctionId::from_index(9)], false),
        ],
        vec![],
    );
    let effects = crate::ssa::storage::workgroup_call_effects(&program);
    assert_eq!(effects[&scalar], None);
    assert_eq!(effects[&wrapper], None);
    assert_eq!(effects[&writer], Some(ResourceAccess::Write));
    assert_eq!(effects[&unknown], Some(ResourceAccess::ReadWrite));
    for callee in [wrapper, unknown] {
        let mut body = FuncBuilder::new(vec![], types::unit()).finish_unchecked();
        let entry = body.inner.entry;
        let a = buffer(&mut body, entry, 0, 65);
        access(&mut body, entry, a, true);
        body.inner.append_void_inst(entry, InstKind::ControlBarrier);
        body.inner.append_inst(
            entry,
            InstKind::Op {
                tag: OpTag::Call(callee),
                operands: vec![],
            },
            types::unit(),
        );
        body.inner.append_void_inst(entry, InstKind::ControlBarrier);
        access(&mut body, entry, a, false);
        super::optimize_body(&mut body, &effects).unwrap();
        assert_eq!(barriers(&body), if callee == wrapper { 1 } else { 2 });
    }
}

#[test]
fn a_view_hidden_in_an_escaping_array_blocks_reuse() {
    let view_ty = types::view_array_with_size(
        &types::i32(),
        Type::Constructed(TypeName::SizePlaceholder, vec![]),
        types::no_buffer(),
    );
    let result_ty = types::sized_array(1, view_ty);
    let mut body = FuncBuilder::new(vec![], result_ty.clone()).finish_unchecked();
    let entry = body.inner.entry;
    let a = buffer(&mut body, entry, 0, 65);
    access(&mut body, entry, a, true);
    body.inner.append_void_inst(entry, InstKind::ControlBarrier);
    let b = buffer(&mut body, entry, 1, 65);
    access(&mut body, entry, b, true);
    let result = body.inner.append_inst(
        entry,
        InstKind::Op {
            tag: OpTag::ArrayLit,
            operands: vec![a],
        },
        result_ty,
    );
    body.inner.blocks[entry].term = Terminator::Return(Some(result.into()));
    apply(&mut body).unwrap();
    assert_eq!(allocations(&body).len(), 2);
    assert_eq!(
        barriers(&body),
        1,
        "the returned view's contents must be published"
    );
}
