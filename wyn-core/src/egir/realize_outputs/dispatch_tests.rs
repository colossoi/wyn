//! Unit tests for the dispatch helpers.

use super::*;
use crate::egir::soac::{hist, screma};
use crate::egir::types::{EGraph, Raw, RegionId, SegBody, SideEffect, Soac, SoacEffect, SoacInputType};
use smallvec::smallvec;

fn raw_map_soac(
    input: SoacInputType,
    map_body: SegBody,
    output_element_type: Type<TypeName>,
    _result_type: Type<TypeName>,
) -> Soac<Raw> {
    let input_element = input.element();
    Soac::Screma(screma::Op {
        inputs: vec![input],
        form: screma::ScremaForm {
            pre: screma::Lambda::region(map_body, vec![input_element], vec![output_element_type.clone()]),
            scans: vec![],
            reductions: vec![],
            post: screma::Lambda::identity(vec![output_element_type]),
        },
        result_state: vec![screma::ResultState {
            destination: SoacDestination::fresh().placed(SoacPlacement::OutputView),
        }],
        state: screma::RawState,
    })
}

fn raw_hist_soac(
    inputs: Vec<SoacInputType>,
    captures: Vec<NodeId>,
    index_type: Type<TypeName>,
    value_type: Type<TypeName>,
    destination: NodeId,
    shape: NodeId,
    race_factor: NodeId,
) -> Soac<Raw> {
    Soac::Hist(hist::Op {
        inputs: inputs.clone(),
        form: hist::HistForm {
            bucket: screma::Lambda::region(
                SegBody {
                    region: RegionId::from_index(0),
                    captures,
                },
                inputs.iter().map(SoacInputType::element).collect(),
                vec![index_type, value_type.clone()],
            ),
            operations: vec![hist::HistOp {
                emission: hist::Emission::Always,
                shape: vec![shape],
                race_factor,
                destinations: vec![destination],
                update: hist::Update::OrderedOverwrite {
                    value_types: vec![value_type],
                },
            }],
        },
        state: hist::RawState,
    })
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
            crate::types::no_buffer(),
        ],
    );

    let mut graph = EGraph::<Raw>::new();
    let source = graph.alloc_side_effect_result(unsized_arr_ty.clone());
    let block = graph.skeleton.entry;
    let mut next_effect = crate::IdSource::new();
    let effect_index = graph.side_effect_index();
    let err = compute_slot_source(
        &mut graph,
        &effect_index,
        &mut next_effect,
        block,
        source,
        0,
        &unsized_arr_ty,
        crate::ResourceId::for_test(0),
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

fn composite_arr_ty(elem: Type<TypeName>, n: usize) -> Type<TypeName> {
    Type::Constructed(
        TypeName::Array,
        vec![
            elem,
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            Type::Constructed(TypeName::Size(n), vec![]),
            crate::types::no_buffer(),
        ],
    )
}

/// `source` at a `Screma` operand position past `input_array_types.len()`
/// (an init-accumulator / output-view slot) must be rejected as non-input —
/// only the leading input operands are per-element view reads.
#[test]
fn rewrite_sibling_index_consumers_rejects_map_output_view_operand() {
    let mut graph = EGraph::<Raw>::new();
    let block = graph.skeleton.entry;
    let elem = vec4_ty();
    let arr_ty = composite_arr_ty(elem.clone(), 4);

    // The slot's source — a `SideEffectResult` typed as a (logical)
    // composite array. (The producer side-effect itself isn't needed
    // for this classifier-level test.)
    let source = graph.alloc_side_effect_result(arr_ty.clone());

    // The output view we'd retarget to.
    let view =
        graph_ops::intern_resource_view(&mut graph, crate::ResourceId::for_test(1), elem.clone(), None);

    // A downstream Screma with one input. The input operand is a distinct
    // dummy; `source` is placed at operand index 1, past the single input
    // (an init-accumulator / output-view slot).
    let dummy_input = graph.alloc_side_effect_result(arr_ty.clone());
    let result_nid =
        graph.alloc_side_effect_result(Type::Constructed(TypeName::Tuple(1), vec![arr_ty.clone()]));
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(
            (),
            raw_map_soac(
                SoacInputType::array(arr_ty.clone()),
                SegBody {
                    region: RegionId::from_index(0),
                    captures: vec![],
                },
                elem.clone(),
                arr_ty.clone(),
            ),
        )),
        operand_nodes: smallvec![dummy_input, source],
        result: Some(result_nid),
        effects: None,
        span: None,
    });

    let mut next_effect = crate::IdSource::new();
    let err = rewrite_sibling_index_consumers(&mut graph, block, &mut next_effect, source, view, elem, 0)
        .expect_err("Screma output-view consumer of `source` must be rejected");
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

/// Hist's compact operands contain only co-iterated inputs. Destination,
/// shape, race-factor, and capture references live in the canonical form and
/// cannot be mistaken for retargetable array inputs.
#[test]
fn rewrite_sibling_index_consumers_rewrites_hist_input_only() {
    let mut graph = EGraph::<Raw>::new();
    let block = graph.skeleton.entry;
    let elem = vec4_ty();
    let arr_ty = composite_arr_ty(elem.clone(), 4);
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    let source = graph.alloc_side_effect_result(arr_ty.clone());
    let view =
        graph_ops::intern_resource_view(&mut graph, crate::ResourceId::for_test(1), elem.clone(), None);
    let destination =
        graph_ops::intern_resource_view(&mut graph, crate::ResourceId::for_test(2), elem.clone(), None);
    let shape = graph.intern_pure(PureOp::Int("4".into()), smallvec![], i32_ty.clone(), None);
    let race_factor = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone(), None);
    let result_nid = graph.alloc_side_effect_result(Type::Constructed(TypeName::Bool, vec![]));
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(
            (),
            raw_hist_soac(
                vec![SoacInputType::array(arr_ty)],
                vec![],
                i32_ty,
                elem.clone(),
                destination,
                shape,
                race_factor,
            ),
        )),
        operand_nodes: smallvec![source],
        result: Some(result_nid),
        effects: None,
        span: None,
    });

    let mut next_effect = crate::IdSource::new();
    rewrite_sibling_index_consumers(&mut graph, block, &mut next_effect, source, view, elem, 0)
        .expect("Hist input should retarget to the output view");
    assert_eq!(
        graph.skeleton.blocks[block].side_effects[0].operand_nodes.as_slice(),
        &[view]
    );
}
/// `source` at a `Screma` accumulator-init slot (past `inputs.len()`)
/// must be rejected — init values are scalars/values, not per-element
/// view reads.
#[test]
fn rewrite_sibling_index_consumers_rejects_accumulator_output_view_operand() {
    let mut graph = EGraph::<Raw>::new();
    let block = graph.skeleton.entry;
    let elem = vec4_ty();
    let arr_ty = composite_arr_ty(elem.clone(), 4);

    let source = graph.alloc_side_effect_result(arr_ty.clone());
    let view =
        graph_ops::intern_resource_view(&mut graph, crate::ResourceId::for_test(1), elem.clone(), None);

    // Screma with one input + one Reduce accumulator. `source` at the
    // init_acc slot (operand index 1, past the input region at index 0).
    let dummy_input = graph.alloc_side_effect_result(arr_ty.clone());
    let result_nid =
        graph.alloc_side_effect_result(Type::Constructed(TypeName::Tuple(1), vec![elem.clone()]));
    graph.skeleton.blocks[block].side_effects.push(SideEffect {
        kind: SideEffectKind::Soac(SoacEffect(
            (),
            Soac::Screma(screma::Op {
                inputs: vec![SoacInputType::array(arr_ty.clone())],
                form: screma::ScremaForm {
                    pre: screma::Lambda::identity(vec![elem.clone()]),
                    scans: vec![],
                    reductions: vec![screma::Reduce {
                        operator: screma::Lambda::region(
                            SegBody {
                                region: RegionId::from_index(0),
                                captures: vec![],
                            },
                            vec![elem.clone(), elem.clone()],
                            vec![elem.clone()],
                        ),
                        neutral: vec![source],
                        commutative: false,
                    }],
                    post: screma::Lambda::identity(vec![]),
                },
                result_state: vec![screma::ResultState {
                    destination: SoacDestination::fresh().placed(SoacPlacement::OutputView),
                }],
                state: screma::RawState,
            }),
        )),
        operand_nodes: smallvec![dummy_input, source],
        result: Some(result_nid),
        effects: None,
        span: None,
    });

    let mut next_effect = crate::IdSource::new();
    let err = rewrite_sibling_index_consumers(&mut graph, block, &mut next_effect, source, view, elem, 0)
        .expect_err("Screma accumulator output-view consumer of `source` must be rejected");
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
