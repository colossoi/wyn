#![cfg(test)]

use crate::ast::TypeName;
use crate::ssa::builder::FuncBuilder;
use crate::ssa::types::{InstKind, Terminator, ValueId, ValueRef};
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
            InstKind::BinOp {
                op: "+".to_string(),
                lhs: ValueRef::Ssa(x),
                rhs: ValueRef::Ssa(y),
            },
            i32_ty(),
        )
        .unwrap();
    builder.terminate(Terminator::Return(Some(sum))).unwrap();

    let body = builder.finish().unwrap();

    assert_eq!(body.params.len(), 2);
    assert_ne!(body.params[0].0, body.params[1].0);
    assert_eq!(body.num_blocks(), 1);
    assert_eq!(body.num_insts(), 1);
}

/// `InstKind::remap` rewrites `ValueId` operands but must leave `PlaceId`s
/// (the identity of `ViewIndex` / `OutputSlot` / `Alloca` results and the
/// operand slot of `Load` / `Store`) untouched. This is the property that
/// was silently violated in the pre-split world: a generic value-remap
/// would rename the `StorageViewIndex` result ValueId *and* the Load/Store
/// pointers referencing it, but the separate backend shadow maps
/// (`view_buffer_id`, `view_handles`) stayed keyed by the *old* ValueId,
/// desynchronising the IR from its own metadata. With a dedicated
/// `PlaceId` the identity is in the IR, so remaps of the value axis can't
/// corrupt the place axis.
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
    let view_index = InstKind::ViewIndex {
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
    let store = InstKind::Store {
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
    let load = InstKind::Load { place: place_b };
    let remapped = load.remap(&rv);
    assert!(matches!(remapped, InstKind::Load { place } if place == place_b));

    // OutputSlot { index, result }: no value operands; place stays.
    let slot = InstKind::OutputSlot {
        index: 0,
        result: place_a,
    };
    let remapped = slot.remap(&rv);
    assert!(matches!(remapped, InstKind::OutputSlot { result, .. } if result == place_a));

    // Alloca { elem_ty, result }: place stays.
    let alloca = InstKind::Alloca {
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

    let load = InstKind::Load { place };
    assert!(
        load.value_uses().is_empty(),
        "Load's place is a PlaceId operand, not a ValueRef"
    );
    assert_eq!(load.place_uses(), vec![place]);

    let store = InstKind::Store {
        place,
        value: ValueRef::Ssa(v),
    };
    assert_eq!(
        store.value_uses(),
        vec![ValueRef::Ssa(v)],
        "Store reports only its value operand, not its place"
    );
    assert_eq!(store.place_uses(), vec![place]);

    let vi = InstKind::ViewIndex {
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
/// chain and confirm we emit valid SPIR-V without the removed
/// `view_buffer_id` fallback path ever being exercised. This is the
/// integration-level regression test for the place/value split: if any
/// backend ever resurrects `ValueId`-keyed place tracking, the
/// `simple_compute` path is what'll bitrot first.
#[test]
fn spirv_storage_write_chain_lowers_cleanly() {
    // Minimal compute shader: the map's `[]f32 → []f32` writeback forces
    // the MapInto path → ViewIndex (place) + Store. If lowering were still
    // using `view_buffer_id` cross-lookups, a mismatched ValueId would
    // surface as `resolve_buffer_by_id: not a u32 constant`.
    let source = "#[compute]\nentry double(arr: []f32) []f32 = map(|x: f32| x * 2.0, arr)\n";

    let mut frontend = crate::cached_frontend();
    let parsed = crate::Compiler::parse(source, &mut frontend.node_counter).unwrap();
    let spirv = parsed
        .desugar(&mut frontend.node_counter)
        .unwrap()
        .resolve(&mut frontend.module_manager)
        .unwrap()
        .fold_ast_constants()
        .type_check(&mut frontend.module_manager, &mut frontend.schemes)
        .unwrap()
        .to_tlc(&frontend.schemes, &frontend.module_manager, false)
        .partial_eval()
        .normalize_soacs()
        .promote_inplace()
        .expect("promote_inplace")
        .fuse_maps()
        .defunctionalize()
        .monomorphize()
        .buffer_specialize()
        .fold_generated_lambdas()
        .inline_small()
        .parallelize_soacs(false)
        .filter_reachable()
        .to_egraph()
        .unwrap()
        .expand_soacs(true)
        .materialize()
        .optimize_skeleton()
        .elaborate()
        .lower()
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
