//! Expand retained memory transfers after storage selection, for both backends.
use super::types::{
    BlockId, ConstantValue, ControlHeader, FuncBody, InstId, InstKind, PlaceId, PlaceInfo, PlaceOrigin,
    Terminator, ValueRef,
};
use crate::op::{BinaryOperator, OpTag};
use crate::types::{self, Type, TypeExt, TypeName};

/// A memory transfer preserves element representation; rank is unrestricted.
pub(crate) fn matching_view_element<'a>(source: &'a Type, destination: &Type) -> Option<&'a Type> {
    if source.array_variant().is_some_and(types::is_array_variant_view)
        && destination.array_variant().is_some_and(types::is_array_variant_view)
    {
        source.elem_type().filter(|element| Some(*element) == destination.elem_type())
    } else {
        None
    }
}

pub(super) fn lower(body: &mut FuncBody) {
    let copies: Vec<_> = body
        .inner
        .blocks
        .iter()
        .flat_map(|(block, data)| {
            data.insts
                .iter()
                .filter(|&&id| matches!(body.inner.insts[id].data, InstKind::Copy { .. }))
                .map(move |&id| (block, id))
        })
        .collect();
    // Earlier transfers can split the original block containing later ones.
    for (_, copy) in copies {
        let block = body.inner.insts[copy].placement.block().expect("placed copy");
        expand(body, block, copy);
    }
}

fn operation(
    body: &mut FuncBody,
    block: BlockId,
    tag: OpTag<crate::BindingRef, crate::FunctionId>,
    operands: Vec<ValueRef>,
    ty: Type,
) -> ValueRef {
    body.inner.append_inst(block, InstKind::Op { tag, operands }, ty).into()
}

fn expand(body: &mut FuncBody, block: BlockId, copy: InstId) {
    let InstKind::Copy {
        destination,
        source,
        length,
        start,
        step,
        ..
    } = body.inner.insts[copy].data
    else {
        unreachable!()
    };
    let source_ty = body.value_ref_type(source);
    let destination_ty = body.value_ref_type(destination);
    let element = matching_view_element(&source_ty, &destination_ty).expect("copy element");
    let end = body.inner.create_block();
    let at = body.inner.blocks[block].insts.iter().position(|&id| id == copy).expect("copy in block");
    let tail = body.inner.blocks[block].insts.split_off(at + 1);
    body.inner.blocks[block].insts.pop();
    body.inner.insts.remove(copy);
    for &id in &tail {
        body.inner.insts[id].placement = super::ir::InstPlacement::Block(end);
    }
    body.inner.blocks[end].insts = tail;
    body.inner.blocks[end].term =
        std::mem::replace(&mut body.inner.blocks[block].term, Terminator::Unreachable);
    // A loop header must remain the target of its backedges. A selection's
    // header follows its conditional terminator to the end of the copy.
    if matches!(
        body.inner.blocks[block].control_header,
        Some(ControlHeader::Selection { .. })
    ) {
        body.inner.blocks[end].control_header = body.inner.blocks[block].control_header.take();
    }
    counted(body, block, end, start, length, step, |body, block, index| {
        let source = view_place(body, block, source, index, element);
        let destination = view_place(body, block, destination, index, element);
        copy_place(body, block, source, destination, element)
    });
}

fn counted(
    body: &mut FuncBody,
    before: BlockId,
    end: BlockId,
    start: ValueRef,
    length: ValueRef,
    step: ValueRef,
    emit: impl FnOnce(&mut FuncBody, BlockId, ValueRef) -> BlockId,
) {
    let index_ty = body.value_ref_type(start);
    let header = body.inner.create_block();
    let iteration = body.inner.create_block();
    let continuing = body.inner.create_block();
    let index: ValueRef = body.inner.add_block_param(header, index_ty.clone()).into();
    body.inner.blocks[before].term = Terminator::Branch {
        target: header,
        args: vec![start],
    };
    body.inner.blocks[header].control_header = Some(ControlHeader::Loop {
        merge: end,
        continue_block: continuing,
    });
    let condition = operation(
        body,
        header,
        OpTag::BinOp(BinaryOperator::Less),
        vec![index, length],
        types::bool_type(),
    );
    body.inner.blocks[header].term = Terminator::CondBranch {
        cond: condition,
        then_target: iteration,
        then_args: vec![],
        else_target: end,
        else_args: vec![],
    };
    let last = emit(body, iteration, index);
    body.inner.blocks[last].term = Terminator::Branch {
        target: continuing,
        args: vec![],
    };
    let next = operation(
        body,
        continuing,
        OpTag::BinOp(BinaryOperator::Add),
        vec![index, step],
        index_ty,
    );
    body.inner.blocks[continuing].term = Terminator::Branch {
        target: header,
        args: vec![next],
    };
}

fn view_place(body: &mut FuncBody, block: BlockId, view: ValueRef, index: ValueRef, ty: &Type) -> PlaceId {
    let result = body.places.insert(PlaceInfo {
        elem_ty: ty.clone(),
        origin: PlaceOrigin::Instruction,
    });
    body.inner.append_void_inst(block, InstKind::ViewIndex { view, index, result });
    result
}

fn child(body: &mut FuncBody, block: BlockId, place: PlaceId, index: ValueRef, ty: &Type) -> PlaceId {
    let result = body.places.insert(PlaceInfo {
        elem_ty: ty.clone(),
        origin: PlaceOrigin::Instruction,
    });
    body.inner.append_void_inst(block, InstKind::PlaceIndex { place, index, result });
    result
}

fn copy_place(
    body: &mut FuncBody,
    mut block: BlockId,
    source: PlaceId,
    destination: PlaceId,
    ty: &Type,
) -> BlockId {
    if ty.array_variant().is_some_and(types::is_array_variant_composite) {
        let Some(Type::Constructed(TypeName::Size(count), _)) = ty.array_size() else {
            panic!("concrete array copy extent");
        };
        let element = ty.elem_type().expect("array element");
        let end = body.inner.create_block();
        counted(
            body,
            block,
            end,
            ValueRef::Const(ConstantValue::U32(0)),
            ValueRef::Const(ConstantValue::U32(*count as u32)),
            ValueRef::Const(ConstantValue::U32(1)),
            |body, block, index| {
                let source = child(body, block, source, index, element);
                let destination = child(body, block, destination, index, element);
                copy_place(body, block, source, destination, element)
            },
        );
        return end;
    }
    if let Type::Constructed(TypeName::Tuple(_) | TypeName::Record(_), fields) = ty {
        for (i, field) in fields.iter().enumerate() {
            let index = ValueRef::Const(ConstantValue::U32(i as u32));
            let source = child(body, block, source, index, field);
            let destination = child(body, block, destination, index, field);
            block = copy_place(body, block, source, destination, field);
        }
        return block;
    }
    let value = body.inner.append_inst(block, InstKind::Load { place: source }, ty.clone());
    body.inner.append_void_inst(
        block,
        InstKind::Store {
            place: destination,
            value: value.into(),
        },
    );
    block
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::PureViewSource;
    use crate::ssa::builder::FuncBuilder;

    #[test]
    fn nested_transfers_use_element_places_without_aggregate_snapshots() {
        for element in [
            types::sized_array(3, types::i32()),
            types::sized_array(3, types::sized_array(5, types::i32())),
            types::tuple(vec![types::sized_array(2, types::i32()), types::i32()]),
        ] {
            let view = types::view_array_with_size(
                &element,
                Type::Constructed(TypeName::SizePlaceholder, vec![]),
                types::no_buffer(),
            );
            assert_eq!(matching_view_element(&view, &view), Some(&element));
            let mut builder = FuncBuilder::new(vec![], types::unit());
            let mut views = Vec::new();
            for id in 0..2 {
                views.push(
                    builder
                        .push_inst(
                            InstKind::Op {
                                tag: OpTag::StorageView(PureViewSource::Workgroup { id, count: 2 }),
                                operands: vec![
                                    ValueRef::Const(ConstantValue::U32(0)),
                                    ValueRef::Const(ConstantValue::U32(2)),
                                ],
                            },
                            view.clone(),
                        )
                        .unwrap()
                        .into(),
                );
            }
            builder
                .push_void_inst(InstKind::Copy {
                    source: views[0],
                    destination: views[1],
                    length: ValueRef::Const(ConstantValue::U32(2)),
                    start: ValueRef::Const(ConstantValue::U32(0)),
                    step: ValueRef::Const(ConstantValue::U32(1)),
                    collective: false,
                })
                .unwrap();
            builder.terminate(Terminator::Return(None)).unwrap();
            let mut body = builder.finish().unwrap();
            lower(&mut body);
            assert!(!body.inner.insts.values().any(|n| matches!(n.data, InstKind::Copy { .. })));
            for node in body.inner.insts.values() {
                if let (InstKind::Load { .. }, Some(value)) = (&node.data, node.result) {
                    assert_eq!(body.get_value_type(value), &types::i32());
                }
            }
        }
    }
}
