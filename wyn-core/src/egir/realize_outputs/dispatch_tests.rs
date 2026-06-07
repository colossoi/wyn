//! Unit tests for the dispatch helpers — translated from the old
//! `assign_outputs_tests.rs` against the renamed module.

use super::*;
use crate::egir::types::{EGraph, PendingSoac, SideEffect, SideEffectKind};
use smallvec::SmallVec;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

#[test]
#[should_panic(expected = "is not Map/Scan")]
fn retarget_map_scan_panics_on_reduce() {
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

    retarget_map_scan(&mut graph, target, output_view);
}

#[test]
#[should_panic(expected = "no side effect produced")]
fn retarget_map_scan_panics_when_target_missing() {
    let mut graph = EGraph::new();
    let target = graph.alloc_side_effect_result(i32_ty());
    let output_view = graph.alloc_side_effect_result(i32_ty());

    retarget_map_scan(&mut graph, target, output_view);
}

/// A runtime-sized compute output that no retargetable Map/Scan produced
/// must surface a clean `Unsupported` error.
#[test]
fn compute_slot_source_rejects_unsized_array_without_soac() {
    let f32_ty = Type::Constructed(TypeName::Float(32), vec![]);
    // Array args = [elem, variant, size]. Use a View variant + a free
    // type variable for the size to model "unsized runtime array."
    let unsized_arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            f32_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(99),
        ],
    );

    let mut graph = EGraph::new();
    let source = graph.alloc_side_effect_result(unsized_arr_ty.clone());
    let block = graph.skeleton.entry;
    let mut next_effect = 1u32;
    let mut aliases = std::collections::HashMap::new();

    let err = compute_slot_source(
        &mut graph,
        &mut aliases,
        &mut next_effect,
        block,
        source,
        0,
        &unsized_arr_ty,
        crate::BindingRef::new(0, 1),
        false,
    )
    .expect_err("runtime-sized array without a producing SOAC must be rejected");
    match err {
        ConvertError::Unsupported(msg) => {
            assert!(msg.contains("runtime-sized array"), "unexpected message: {msg}")
        }
        other => panic!("expected ConvertError::Unsupported, got {other:?}"),
    }
}
