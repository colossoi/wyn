#![cfg(test)]

use crate::ast::TypeName;
use crate::compile_thru_spirv;
use crate::op;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{InstKind, Terminator, ValueId, ValueRef};
use crate::ssa::{eliminate_dead_pure_instructions, UseSite, ValueUses};
use polytype::Type;

#[test]
fn test_func_body_params() {
    let i32_ty = || Type::Constructed(TypeName::Int(32), vec![]);
    let mut builder = FuncBuilder::new(
        vec![(i32_ty(), "x".to_string()), (i32_ty(), "y".to_string())],
        i32_ty(),
    );

    let x = builder.get_param(0);
    let y = builder.get_param(1);
    let sum = builder
        .push_inst(
            InstKind::Op {
                tag: op::OpTag::BinOp(op::BinaryOperator::Add),
                operands: vec![ValueRef::Ssa(x), ValueRef::Ssa(y)],
            },
            i32_ty(),
        )
        .unwrap();
    builder.terminate(Terminator::Return(Some(ValueRef::Ssa(sum)))).unwrap();

    let body = builder.finish().unwrap();

    assert_eq!(body.params().len(), 2);
    assert_ne!(body.param(0).unwrap().0, body.param(1).unwrap().0);
    assert_eq!(body.num_blocks(), 1);
    assert_eq!(body.num_insts(), 1);
}

#[test]
fn value_uses_counts_instruction_and_terminator_operands() {
    let i32_ty = || Type::Constructed(TypeName::Int(32), vec![]);
    let mut builder = FuncBuilder::new(vec![(i32_ty(), "x".to_string())], i32_ty());
    let x = builder.get_param(0);
    let sum = builder
        .push_inst(
            InstKind::Op {
                tag: op::OpTag::BinOp(op::BinaryOperator::Add),
                operands: vec![x.into(), x.into()],
            },
            i32_ty(),
        )
        .unwrap();
    builder.terminate(Terminator::Return(Some(sum.into()))).unwrap();
    let body = builder.finish().unwrap();

    let uses = ValueUses::analyze(&body.inner);
    assert_eq!(uses.count(x), 2);
    assert_eq!(uses.count(sum), 1);
    assert!(matches!(uses.users(sum), [UseSite::Terminator]));
}

#[test]
fn dead_pure_elimination_removes_an_entire_unused_expression_tree() {
    let i32_ty = || Type::Constructed(TypeName::Int(32), vec![]);
    let mut builder = FuncBuilder::new(vec![(i32_ty(), "x".to_string())], i32_ty());
    let x = builder.get_param(0);
    let dead_sum = builder
        .push_inst(
            InstKind::Op {
                tag: op::OpTag::BinOp(op::BinaryOperator::Add),
                operands: vec![
                    x.into(),
                    ValueRef::Const(crate::ssa::types::ConstantValue::I32(1)),
                ],
            },
            i32_ty(),
        )
        .unwrap();
    let _dead_product = builder
        .push_inst(
            InstKind::Op {
                tag: op::OpTag::BinOp(op::BinaryOperator::Multiply),
                operands: vec![
                    dead_sum.into(),
                    ValueRef::Const(crate::ssa::types::ConstantValue::I32(2)),
                ],
            },
            i32_ty(),
        )
        .unwrap();
    builder.terminate(Terminator::Return(Some(x.into()))).unwrap();
    let mut body = builder.finish().unwrap();

    eliminate_dead_pure_instructions(&mut body);
    assert_eq!(body.num_insts(), 0);
    assert_eq!(ValueUses::analyze(&body.inner).count(x), 1);
}

#[test]
fn dead_pure_elimination_removes_unused_intrinsic_chains() {
    use crate::types::{f32, i32};

    let mut builder = FuncBuilder::new(vec![(f32(), "x".into())], f32());
    let x = builder.get_param(0);
    let mut operand = x;
    // Neither sqrt nor float-to-int conversion is safe to speculate, but an
    // unused chain of these computations can be discarded.
    for (name, ty) in [("f32.sqrt", f32()), ("i32.f32", i32())] {
        operand = builder
            .push_inst(
                InstKind::Op {
                    tag: op::OpTag::Intrinsic {
                        id: crate::builtins::catalog().lookup_by_any_name(name).unwrap().id,
                        overload_idx: 0,
                    },
                    operands: vec![operand.into()],
                },
                ty,
            )
            .unwrap();
    }
    builder.terminate(Terminator::Return(Some(x.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    eliminate_dead_pure_instructions(&mut body);
    assert_eq!(body.num_insts(), 0);
}

#[test]
fn dead_pure_elimination_removes_unused_texture_samples() {
    use crate::ssa::types::ConstantValue;
    use crate::types::{f32, vec};

    let mut builder = FuncBuilder::new(
        vec![
            (Type::Constructed(TypeName::Texture2D, vec![]), "texture".into()),
            (Type::Constructed(TypeName::Sampler, vec![]), "sampler".into()),
            (vec(2, f32()), "uv".into()),
        ],
        f32(),
    );
    let sample = builder
        .push_inst(
            InstKind::Op {
                tag: op::OpTag::Intrinsic {
                    id: crate::builtins::catalog().known().texture_sample,
                    overload_idx: 0,
                },
                operands: vec![
                    builder.get_param(0).into(),
                    builder.get_param(1).into(),
                    builder.get_param(2).into(),
                    ValueRef::Const(ConstantValue::from_f32(0.0)),
                ],
            },
            vec(4, f32()),
        )
        .unwrap();
    builder
        .push_inst(
            InstKind::Op {
                tag: op::OpTag::Project { index: 0 },
                operands: vec![sample.into()],
            },
            f32(),
        )
        .unwrap();
    builder
        .terminate(Terminator::Return(Some(ValueRef::Const(
            ConstantValue::from_f32(1.0),
        ))))
        .unwrap();
    let mut body = builder.finish().unwrap();
    eliminate_dead_pure_instructions(&mut body);
    assert_eq!(body.num_insts(), 0);
}

#[test]
fn dead_pure_elimination_preserves_storage_updates_and_opaque_calls() {
    use crate::ssa::types::ConstantValue;
    use crate::types::{buffer_tag, i32, view_array_with_size};
    use crate::{BindingRef, FunctionId};

    let view_ty = view_array_with_size(
        &i32(),
        Type::Constructed(TypeName::Size(4), vec![]),
        buffer_tag(BindingRef::new(0, 0)),
    );
    let mut builder = FuncBuilder::new(vec![(view_ty.clone(), "dest".into()), (i32(), "x".into())], i32());
    let dest = builder.get_param(0);
    let x = builder.get_param(1);
    let value = builder
        .push_inst(
            InstKind::Op {
                tag: op::OpTag::BinOp(op::BinaryOperator::Add),
                operands: vec![x.into(), ValueRef::Const(ConstantValue::I32(1))],
            },
            i32(),
        )
        .unwrap();
    let known = crate::builtins::catalog().known();
    let mut results = Vec::new();
    // array_with is catalogued Pure, yet its view representation writes to
    // storage. Its result being unused must not discard the write or operands.
    for tag in [
        op::OpTag::Intrinsic {
            id: known.array_with,
            overload_idx: 0,
        },
        op::OpTag::Intrinsic {
            id: known.array_with_in_place,
            overload_idx: 0,
        },
        op::OpTag::Call(FunctionId::from(99)),
    ] {
        results.push(
            builder
                .push_inst(
                    InstKind::Op {
                        tag,
                        operands: vec![dest.into(), ValueRef::Const(ConstantValue::I32(0)), value.into()],
                    },
                    view_ty.clone(),
                )
                .unwrap(),
        );
    }
    builder.terminate(Terminator::Return(Some(x.into()))).unwrap();
    let mut body = builder.finish().unwrap();
    eliminate_dead_pure_instructions(&mut body);
    assert_eq!(body.num_insts(), 4);
    let uses = ValueUses::analyze(&body.inner);
    assert!(results
        .iter()
        .all(|&result| uses.count(result) == 0 && body.inner.inst_of_value(result).is_some()));
    assert_eq!(uses.count(value), 3);
}

/// `InstKind::remap` rewrites `ValueId` operands but must leave `PlaceId`s
/// (the identity of `ViewIndex` / `OutputSlot` / `Alloca` results and the
/// operand slot of `Load` / `Store`) untouched. Place identity lives directly
/// in the IR, so remapping the value axis cannot desynchronize place metadata.
#[test]
fn value_remap_preserves_place_identity() {
    use crate::ssa::types::PlaceId;
    use slotmap::KeyData;

    // Build three PlaceIds by hand — the numeric keys don't matter for the
    // remap's `result:` field, which is what we're testing.
    let place_a = PlaceId::from(KeyData::from_ffi(1));
    let place_b = PlaceId::from(KeyData::from_ffi(2));
    let value_a = ValueId::from(KeyData::from_ffi(10));
    let value_b = ValueId::from(KeyData::from_ffi(11));
    let renamed_a = ValueId::from(KeyData::from_ffi(100));
    let renamed_b = ValueId::from(KeyData::from_ffi(101));

    // A ValueId remap that renames every a→100, b→101, leaves others alone.
    let rv = |v: ValueId| {
        if v == value_a {
            renamed_a
        } else if v == value_b {
            renamed_b
        } else {
            v
        }
    };

    // ViewIndex { view, index, result }: remap renames view & index (values)
    // but preserves the place result.
    let view_index: InstKind = InstKind::ViewIndex {
        view: ValueRef::Ssa(value_a),
        index: ValueRef::Ssa(value_b),
        result: place_a,
    };
    let remapped = view_index.remap(&rv);
    match remapped {
        InstKind::ViewIndex { view, index, result } => {
            assert_eq!(view.as_ssa(), Some(renamed_a), "view operand should remap");
            assert_eq!(index.as_ssa(), Some(renamed_b), "index operand should remap");
            assert_eq!(
                result, place_a,
                "result PlaceId must not change under value remap"
            );
        }
        other => panic!("expected ViewIndex, got {:?}", other),
    }

    // Store { place, value }: value operand remaps, place stays.
    let store: InstKind = InstKind::Store {
        place: place_b,
        value: ValueRef::Ssa(value_a),
    };
    let remapped = store.remap(&rv);
    match remapped {
        InstKind::Store { place, value } => {
            assert_eq!(place, place_b, "Store place must not change under value remap");
            assert_eq!(value.as_ssa(), Some(renamed_a), "Store value should remap");
        }
        other => panic!("expected Store, got {:?}", other),
    }

    // Load { place }: place stays.
    let load: InstKind = InstKind::Load { place: place_b };
    let remapped = load.remap(&rv);
    assert!(matches!(remapped, InstKind::Load { place } if place == place_b));

    // OutputSlot { index, result }: no value operands; place stays.
    let slot: InstKind = InstKind::OutputSlot {
        index: 0,
        result: place_a,
    };
    let remapped = slot.remap(&rv);
    assert!(matches!(remapped, InstKind::OutputSlot { result, .. } if result == place_a));

    // Alloca { elem_ty, result }: place stays.
    let alloca: InstKind = InstKind::Alloca {
        elem_ty: Type::Constructed(TypeName::Int(32), vec![]),
        result: place_a,
    };
    let remapped = alloca.remap(&rv);
    assert!(matches!(remapped, InstKind::Alloca { result, .. } if result == place_a));
}

/// `value_uses()` only returns value-carrying operand slots. `Load`'s
/// place, `Store`'s place, and place-producing insts' `result:` fields
/// must not appear. Places get their own traversal via `place_uses()` /
/// `place_result()`.
#[test]
fn value_uses_does_not_traverse_places() {
    use crate::ssa::types::PlaceId;
    use slotmap::KeyData;

    let place = PlaceId::from(KeyData::from_ffi(1));
    let v = ValueId::from(KeyData::from_ffi(10));

    let load: InstKind = InstKind::Load { place };
    assert!(
        load.value_uses().is_empty(),
        "Load's place is a PlaceId operand, not a ValueRef"
    );
    assert_eq!(load.place_uses(), vec![place]);

    let store: InstKind = InstKind::Store {
        place,
        value: ValueRef::Ssa(v),
    };
    assert_eq!(
        store.value_uses(),
        vec![ValueRef::Ssa(v)],
        "Store reports only its value operand, not its place"
    );
    assert_eq!(store.place_uses(), vec![place]);

    let vi: InstKind = InstKind::ViewIndex {
        view: ValueRef::Ssa(v),
        index: ValueRef::Ssa(v),
        result: place,
    };
    // ViewIndex's value operands (view, index) are present; result is a place.
    assert_eq!(vi.value_uses().len(), 2);
    assert!(
        vi.place_uses().is_empty(),
        "ViewIndex only produces a place, doesn't consume one"
    );
    assert_eq!(vi.place_result(), Some(place));
}

/// Compile a compute shader whose MapInto lowers to a `ViewIndex` → `Store`
/// chain and confirm the place/value split emits valid SPIR-V without
/// ValueId-keyed place tracking.
#[test]
fn spirv_storage_write_chain_lowers_cleanly() {
    // Minimal compute shader: the map's `[]f32 → []f32` writeback forces
    // the MapInto path → ViewIndex (place) + Store.
    let source = "\nentry double(arr: []f32) []f32 = map(|x: f32| x * 2.0, arr)\n";

    let spirv = compile_thru_spirv(source)
        .expect("SPIR-V lowering of StorageView → ViewIndex → Store chain must succeed");

    assert_eq!(spirv.spirv[0], 0x07230203, "SPIR-V magic number");

    // Confirm the chain actually emitted: OpAccessChain (opcode 65) and
    // OpStore (opcode 62) must both appear.
    let mut has_access_chain = false;
    let mut has_store = false;
    let mut i = 5; // skip header
    while i < spirv.spirv.len() {
        let word = spirv.spirv[i];
        let op = (word & 0xFFFF) as u16;
        let len = ((word >> 16) & 0xFFFF) as usize;
        if op == 65 {
            has_access_chain = true;
        }
        if op == 62 {
            has_store = true;
        }
        if len == 0 {
            break;
        }
        i += len;
    }
    assert!(has_access_chain, "ViewIndex must lower to OpAccessChain");
    assert!(has_store, "Store(place) must lower to OpStore");
}
