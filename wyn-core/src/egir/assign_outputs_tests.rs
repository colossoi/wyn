use super::*;
use crate::egir::types::{EGraph, PendingSoac, SideEffect, SideEffectKind};
use smallvec::SmallVec;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

#[test]
#[should_panic(expected = "is not Map/Scan")]
fn rewrite_map_scan_to_into_panics_on_reduce() {
    let mut graph = EGraph::new();
    let target = graph.alloc_side_effect_result(i32_ty());
    let output_view = graph.alloc_side_effect_result(i32_ty());

    let entry = graph.skeleton.entry;
    graph.skeleton.blocks[entry].side_effects.push(SideEffect {
        kind: SideEffectKind::Pending(PendingSoac::Reduce {
            func: "f".to_string(),
            input_array_type: i32_ty(),
            input_elem_type: i32_ty(),
        }),
        operand_nodes: SmallVec::new(),
        result: Some(target),
        effects: None,
        span: None,
    });

    rewrite_map_scan_to_into(&mut graph, target, output_view);
}

#[test]
#[should_panic(expected = "no side effect produced")]
fn rewrite_map_scan_to_into_panics_when_target_missing() {
    let mut graph = EGraph::new();
    let target = graph.alloc_side_effect_result(i32_ty());
    let output_view = graph.alloc_side_effect_result(i32_ty());

    rewrite_map_scan_to_into(&mut graph, target, output_view);
}

/// A runtime-sized compute output that no retargetable Map/Scan produced
/// must surface a clean `Unsupported` error (where the old code path
/// panicked at `emit_compute_output_stores`).
#[test]
fn lower_slot_rejects_unsized_array_without_soac() {
    let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);
    let unsized_arr_ty = Type::Constructed(TypeName::Array, vec![f32_ty.clone(), Type::Variable(99)]);

    let mut graph = EGraph::new();
    let source = graph.alloc_side_effect_result(unsized_arr_ty.clone());
    let block = graph.skeleton.entry;
    let mut next_effect = 1u32;

    let slot = Slot {
        index: 0,
        ty: unsized_arr_ty,
        source,
        dest: Dest::StorageView(crate::BindingRef::new(0, 1)),
    };

    let err = lower_slot(&mut graph, block, &mut next_effect, &slot)
        .expect_err("runtime-sized array without a producing SOAC must be rejected");
    match err {
        ConvertError::Unsupported(msg) => {
            assert!(msg.contains("runtime-sized array"), "unexpected message: {msg}")
        }
        other => panic!("expected ConvertError::Unsupported, got {other:?}"),
    }
}
