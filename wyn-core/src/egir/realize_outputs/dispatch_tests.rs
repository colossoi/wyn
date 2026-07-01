//! Unit tests for the dispatch helpers.

use super::*;
use crate::egir::types::{EGraph, RegionId, ScremaAccumulator, ScremaOperator, SegBody, SideEffect};
use smallvec::smallvec;

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
            crate::types::no_region(),
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

// ---------------------------------------------------------------------
// rewrite_sibling_index_consumers — operand-region classifier contract.
//
// These tests construct a single downstream `SideEffect` whose
// `operand_nodes` contains the slot's `source` NodeId at a *non-
// input* position. The classifier must reject each position with a
// clear `Unsupported` naming the side-effect kind. Positions that
// fall in the input region get rewritten in place — covered end-to-
// end by the integration test
// `compute_entry_returns_screma_result_and_scatters_through_it`.
// ---------------------------------------------------------------------

fn f32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn vec4_ty() -> Type<TypeName> {
    Type::Constructed(
        TypeName::Vec,
        vec![f32_ty(), Type::Constructed(TypeName::Size(4), vec![])],
    )
}

fn view_arr_ty(elem: Type<TypeName>, binding: crate::BindingRef) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            elem,
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(0),
            Type::Constructed(TypeName::Region(binding), vec![]),
        ],
    )
}

fn composite_arr_ty(elem: Type<TypeName>, n: usize) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            elem,
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Constructed(TypeName::Size(n), vec![]),
            crate::types::no_region(),
        ],
    )
}

/// `source` at a `Screma` operand position past `input_array_types.len()`
/// (an init-accumulator / output-view slot) must be rejected as non-input —
/// only the leading input operands are per-element view reads.
#[test]
fn rewrite_sibling_index_consumers_rejects_screma_noninput_operand() {
    let mut graph = EGraph::new();
    let block = graph.skeleton.entry;
    let elem = vec4_ty();
    let arr_ty = composite_arr_ty(elem.clone(), 4);

    // The slot's source — a `SideEffectResult` typed as a (logical)
    // composite array. (The producer side-effect itself isn't needed
    // for this classifier-level test.)
    let source = graph.alloc_side_effect_result(arr_ty.clone());

    // The output view we'd retarget to.
    let view = graph_ops::intern_storage_view(&mut graph, crate::BindingRef::new(0, 1), elem.clone(), None);

    // A downstream Screma with one input. The input operand is a distinct
    // dummy; `source` is placed at operand index 1, past the single input
    // (an init-accumulator / output-view slot).
    let dummy_input = graph.alloc_side_effect_result(arr_ty.clone());
    let result_nid =
        graph.alloc_side_effect_result(Type::Constructed(TypeName::Tuple(1), vec![arr_ty.clone()]));
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(EgirSoac::Screma {
            map_bodies: vec![SegBody {
                region: RegionId::from_index(0),
                captures: vec![source],
            }],
            accumulators: vec![],
            input_array_types: vec![arr_ty.clone()],
            input_elem_types: vec![elem.clone()],
            map_output_elem_types: vec![elem.clone()],
            map_input_indices: vec![vec![0]],
            map_destinations: vec![SoacDestination::Fresh],
            acc_destinations: vec![],
        }),
        operand_nodes: smallvec![dummy_input, source],
        result: Some(result_nid),
        effects: None,
        span: None,
    });

    let mut aliases = std::collections::HashMap::new();
    let mut next_effect = 1u32;
    let err = rewrite_sibling_index_consumers(
        &mut graph,
        &mut aliases,
        block,
        &mut next_effect,
        source,
        view,
        elem,
        0,
    )
    .expect_err("Screma capture-position consumer of `source` must be rejected");
    match err {
        ConvertError::Unsupported(msg) => {
            assert!(
                msg.contains("Screma") && msg.contains("not an array input"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected ConvertError::Unsupported, got {other:?}"),
    }
}

/// `source` at the `Scatter` dest slot (operand 0) must be rejected —
/// the dest is a write-storage view, not an input read.
#[test]
fn rewrite_sibling_index_consumers_rejects_scatter_dest_position() {
    let mut graph = EGraph::new();
    let block = graph.skeleton.entry;
    let elem = vec4_ty();
    let arr_ty = composite_arr_ty(elem.clone(), 4);

    let source = graph.alloc_side_effect_result(arr_ty.clone());
    let view = graph_ops::intern_storage_view(&mut graph, crate::BindingRef::new(0, 1), elem.clone(), None);

    // Scatter with `source` placed at the dest_view operand (index 0)
    // instead of the legitimate input region (`1..1+inputs.len()`).
    let dummy_input = graph.alloc_side_effect_result(arr_ty.clone());
    let result_nid = graph.alloc_side_effect_result(Type::Constructed(TypeName::Bool, vec![]));
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(EgirSoac::Hist {
            body: crate::egir::types::SegBody {
                region: crate::egir::types::RegionId::from_index(0),
                captures: vec![],
            },
            input_array_types: vec![arr_ty.clone()],
            input_elem_types: vec![elem.clone()],
            index_type: Type::Constructed(TypeName::Int(32), vec![]),
            value_type: elem.clone(),
            dest_elem_type: elem.clone(),
            update_policy: crate::egir::types::HistUpdatePolicy::OrderedOverwrite,
            space: None,
        }),
        operand_nodes: smallvec![source, dummy_input],
        result: Some(result_nid),
        effects: None,
        span: None,
    });

    let mut aliases = std::collections::HashMap::new();
    let mut next_effect = 1u32;
    let err = rewrite_sibling_index_consumers(
        &mut graph,
        &mut aliases,
        block,
        &mut next_effect,
        source,
        view,
        elem,
        0,
    )
    .expect_err("Scatter dest-position consumer of `source` must be rejected");
    match err {
        ConvertError::Unsupported(msg) => {
            assert!(
                msg.contains("Scatter") && msg.contains("not an array input"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected ConvertError::Unsupported, got {other:?}"),
    }
}

/// `source` at a `Scatter` capture slot (past `1+inputs.len()`) must
/// be rejected — captures feed the envelope's free vars, not a
/// per-element view read.
#[test]
fn rewrite_sibling_index_consumers_rejects_scatter_capture_position() {
    let mut graph = EGraph::new();
    let block = graph.skeleton.entry;
    let elem = vec4_ty();
    let arr_ty = composite_arr_ty(elem.clone(), 4);

    let source = graph.alloc_side_effect_result(arr_ty.clone());
    let view = graph_ops::intern_storage_view(&mut graph, crate::BindingRef::new(0, 1), elem.clone(), None);

    // Scatter with one input + one capture. The capture slot (index 2,
    // past `1+inputs.len()=2`) is where `source` lives.
    let dummy_dest =
        graph.alloc_side_effect_result(view_arr_ty(elem.clone(), crate::BindingRef::new(0, 1)));
    let dummy_input = graph.alloc_side_effect_result(arr_ty.clone());
    let result_nid = graph.alloc_side_effect_result(Type::Constructed(TypeName::Bool, vec![]));
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(EgirSoac::Hist {
            body: crate::egir::types::SegBody {
                region: crate::egir::types::RegionId::from_index(0),
                captures: vec![source],
            },
            input_array_types: vec![arr_ty.clone()],
            input_elem_types: vec![elem.clone()],
            index_type: Type::Constructed(TypeName::Int(32), vec![]),
            value_type: elem.clone(),
            dest_elem_type: elem.clone(),
            update_policy: crate::egir::types::HistUpdatePolicy::OrderedOverwrite,
            space: None,
        }),
        operand_nodes: smallvec![dummy_dest, dummy_input, source],
        result: Some(result_nid),
        effects: None,
        span: None,
    });

    let mut aliases = std::collections::HashMap::new();
    let mut next_effect = 1u32;
    let err = rewrite_sibling_index_consumers(
        &mut graph,
        &mut aliases,
        block,
        &mut next_effect,
        source,
        view,
        elem,
        0,
    )
    .expect_err("Scatter capture-position consumer of `source` must be rejected");
    match err {
        ConvertError::Unsupported(msg) => {
            assert!(
                msg.contains("Scatter") && msg.contains("not an array input"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected ConvertError::Unsupported, got {other:?}"),
    }
}

/// `source` at a `Screma` accumulator-init slot (past `inputs.len()`)
/// must be rejected — init values are scalars/values, not per-element
/// view reads.
#[test]
fn rewrite_sibling_index_consumers_rejects_screma_init_acc_position() {
    let mut graph = EGraph::new();
    let block = graph.skeleton.entry;
    let elem = vec4_ty();
    let arr_ty = composite_arr_ty(elem.clone(), 4);

    let source = graph.alloc_side_effect_result(arr_ty.clone());
    let view = graph_ops::intern_storage_view(&mut graph, crate::BindingRef::new(0, 1), elem.clone(), None);

    // Screma with one input + one Reduce accumulator. `source` at the
    // init_acc slot (operand index 1, past the input region at index 0).
    let dummy_input = graph.alloc_side_effect_result(arr_ty.clone());
    let result_nid =
        graph.alloc_side_effect_result(Type::Constructed(TypeName::Tuple(1), vec![elem.clone()]));
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(EgirSoac::Screma {
            map_bodies: vec![],
            accumulators: vec![ScremaOperator {
                kind: ScremaAccumulator::Reduce,
                step: SegBody {
                    region: RegionId::from_index(0),
                    captures: vec![],
                },
                combine: SegBody {
                    region: RegionId::from_index(1),
                    captures: vec![],
                },
                input_indices: vec![0],
            }],
            input_array_types: vec![arr_ty.clone()],
            input_elem_types: vec![elem.clone()],
            map_output_elem_types: vec![],
            map_input_indices: vec![],
            map_destinations: vec![],
            acc_destinations: vec![SoacDestination::Fresh],
        }),
        operand_nodes: smallvec![dummy_input, source],
        result: Some(result_nid),
        effects: None,
        span: None,
    });

    let mut aliases = std::collections::HashMap::new();
    let mut next_effect = 1u32;
    let err = rewrite_sibling_index_consumers(
        &mut graph,
        &mut aliases,
        block,
        &mut next_effect,
        source,
        view,
        elem,
        0,
    )
    .expect_err("Screma init-acc consumer of `source` must be rejected");
    match err {
        ConvertError::Unsupported(msg) => {
            assert!(
                msg.contains("Screma") && msg.contains("not an array input"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected ConvertError::Unsupported, got {other:?}"),
    }
}
