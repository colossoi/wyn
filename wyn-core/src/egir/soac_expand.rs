//! Expand `SideEffectKind::Pending(PendingSoac::...)` skeleton side-effects
//! into explicit loop subgraphs with pure ops in the sea and block params
//! carrying accumulators.
//!
//! Runs after `from_tlc` populates the EGraph and before `elaborate` produces
//! the final `FuncBody`. Every variant must be handled here — there is no
//! fallback. Any `Pending` left in the skeleton at elaboration time is a bug.

use crate::builtins::catalog;
use crate::LookupMap;

use crate::ssa::framework::BlockId;
use polytype::Type;
use smallvec::{smallvec, SmallVec};

use super::graph_ops::{
    alloc_effect, emit_alloca, emit_load, emit_place_index_store, emit_store, next_effect_token,
};
use super::program::EgirInner;
use crate::ast::TypeName;
use crate::ssa::types::{ControlHeader, InstKind, ValueRef};
use crate::types::TypeExt;
use crate::types::{is_array_variant_view, is_virtual_array};

use super::types::{
    EGraph, ENode, NodeId, PendingSoac, PureOp, SideEffect, SideEffectKind, SkeletonTerminator,
    SoacDestination,
};

/// Run `run_one_body` on every function and entry point in the program.
pub fn run(inner: &mut EgirInner) {
    for f in &mut inner.functions {
        run_one_body(&mut f.graph, &mut f.control_headers);
    }
    for e in &mut inner.entry_points {
        run_one_body(&mut e.graph, &mut e.control_headers);
    }
}

/// Expand every `SideEffectKind::Pending(PendingSoac::...)` in the skeleton.
pub fn run_one_body(graph: &mut EGraph, control_headers: &mut LookupMap<BlockId, ControlHeader>) {
    // Collect (block, index) of every handleable Soac in a stable order.
    // Process back-to-front within each block so earlier indices stay valid.
    let mut targets: Vec<(BlockId, usize)> = Vec::new();
    for (bid, block) in &graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            if is_handleable_soac(&se.kind) {
                targets.push((bid, i));
            }
        }
    }
    // Sort by (block, descending index) so removals within the same block
    // don't shift earlier target indices.
    targets.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));

    let mut next_effect = next_effect_token(graph);
    for (bid, idx) in targets {
        expand_one(graph, control_headers, bid, idx, &mut next_effect);
    }
    let _ = next_effect; // silence unused when no view-array reductions
}

// Effect-token helpers (`next_effect_token`, `alloc_effect`) live in
// `graph_ops` and are imported up top.

/// Does this SOAC kind have a TLC→EGIR expansion implemented here?
fn is_handleable_soac(kind: &SideEffectKind) -> bool {
    let SideEffectKind::Pending(soac) = kind else {
        return false;
    };
    match soac {
        PendingSoac::Screma {
            input_array_types, ..
        } => input_array_types.iter().all(is_plain_array_source),
        PendingSoac::Filter { input_array_type, .. } => is_plain_array_source(input_array_type),
        // Scatter reads all input arrays per element; loop length comes from
        // the first input, but every input must support the read path.
        PendingSoac::Scatter {
            input_array_types, ..
        } => !input_array_types.is_empty() && input_array_types.iter().all(is_plain_array_source),
        // A reified parallel map/reduce/scan; same source rules as its inputs.
        PendingSoac::Seg {
            input_array_types, ..
        } => input_array_types.iter().all(is_plain_array_source),
    }
}

/// Element type to read from an input array: the buffer's own element type
/// (uniqueness stripped). For a map-fused scan/reduce the raw input element
/// differs from the accumulator element carried by `input_elem_type` (e.g.
/// `scan(+, 0, map(|h:vec4f32| ..:i32, bh))` reads `vec4f32` but accumulates
/// `i32`), so the read must follow the array type, not the accumulator.
/// Falls back to `acc_elem` when the array type has no extractable element
/// (e.g. a SoA-tuple source, handled separately).

/// Input-array shape handled today: rank-1 composite/view/virtual
/// arrays, or SoA tuples `([n]A, [n]B, ...)` (produced by `tlc::soa`)
/// whose components are themselves handleable.
fn is_plain_array_source(arr_ty: &Type<TypeName>) -> bool {
    // Rank-1 invariant: [elem, variant, size, region] (4 args).
    if matches!(arr_ty, Type::Constructed(TypeName::Array, args) if args.len() == 4) {
        return true;
    }
    if let Some(components) = as_soa_tuple(arr_ty) {
        return components.iter().all(is_plain_array_source);
    }
    false
}

/// If `ty` is a SoA tuple (tuple where every component is an Array or itself
/// a SoA tuple), return the component types. Mirrors the helper in
/// `ssa::soa_helpers`.
pub(super) fn as_soa_tuple(ty: &Type<TypeName>) -> Option<&[Type<TypeName>]> {
    let Type::Constructed(TypeName::Tuple(_), components) = ty else {
        return None;
    };
    if components.is_empty() {
        return None;
    }
    // Rank-1 invariant on each component ([elem, variant, size, region]).
    let all_soa = components.iter().all(|ct| {
        matches!(ct, Type::Constructed(TypeName::Array, args) if args.len() == 4)
            || as_soa_tuple(ct).is_some()
    });
    if all_soa {
        Some(components)
    } else {
        None
    }
}

/// Element type of a SoA tuple: `([n]A, [n]B)` → `(A, B)`. Nested SoA tuples
/// recurse into their own element types.
pub(super) fn soa_element_type(soa_ty: &Type<TypeName>) -> Type<TypeName> {
    let Type::Constructed(TypeName::Tuple(n), components) = soa_ty else {
        panic!("soa_element_type: expected tuple, got {:?}", soa_ty)
    };
    let elem_tys: Vec<Type<TypeName>> = components
        .iter()
        .map(|ct| {
            if ct.is_array() {
                ct.elem_type().expect("Array has elem").clone()
            } else if as_soa_tuple(ct).is_some() {
                soa_element_type(ct)
            } else {
                ct.clone()
            }
        })
        .collect();
    Type::Constructed(TypeName::Tuple(*n), elem_tys)
}

fn is_view_source(arr_ty: &Type<TypeName>) -> bool {
    matches!(
        arr_ty,
        Type::Constructed(TypeName::Array, args)
            // args = [elem, variant, size, region]
            if args.len() == 4 && is_array_variant_view(&args[1])
    )
}

fn is_virtual_source(arr_ty: &Type<TypeName>) -> bool {
    is_virtual_array(arr_ty)
}

fn expand_one(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx: usize,
    next_effect: &mut u32,
) {
    let se = graph.skeleton.blocks[bid].side_effects.remove(idx);
    match &se.kind {
        SideEffectKind::Pending(PendingSoac::Screma {
            map_funcs,
            accumulators,
            input_array_types,
            input_elem_types,
            map_output_elem_types,
            map_input_indices,
            map_capture_counts,
            map_destinations,
            acc_destinations,
        }) => {
            let map_input_indices = map_input_indices.clone();
            let map_funcs = map_funcs.clone();
            let acc_specs = accumulators.clone();
            let arr_tys = input_array_types.clone();
            let elem_tys = input_elem_types.clone();
            let map_output_elem_types = map_output_elem_types.clone();
            let map_capture_counts = map_capture_counts.clone();
            let map_destinations = map_destinations.clone();
            let acc_destinations = acc_destinations.clone();
            let n_maps = map_funcs.len();
            let n_accs = acc_specs.len();
            let n_inputs = arr_tys.len();
            let input_nids: Vec<NodeId> = se.operand_nodes[..n_inputs].to_vec();
            let init_acc_nids: Vec<NodeId> = se.operand_nodes[n_inputs..n_inputs + n_accs].to_vec();
            let mut cursor = n_inputs + n_accs;
            let mut map_captures = Vec::with_capacity(n_maps);
            for count in &map_capture_counts {
                map_captures.push(se.operand_nodes[cursor..cursor + *count].to_vec());
                cursor += *count;
            }
            let mut acc_step_captures = Vec::with_capacity(n_accs);
            for acc in &acc_specs {
                acc_step_captures.push(se.operand_nodes[cursor..cursor + acc.step_capture_count].to_vec());
                cursor += acc.step_capture_count;
            }
            for acc in &acc_specs {
                cursor += acc.reduce_op_capture_count;
            }
            let result_nid = se.result.expect("Screma has a result");
            let result_ty = graph.types[&result_nid].clone();
            let Type::Constructed(TypeName::Tuple(_), result_fields) = &result_ty else {
                panic!("Screma result must be a tuple");
            };
            assert_eq!(
                result_fields.len(),
                n_maps + n_accs,
                "Screma result is (mapped..., accumulator...)"
            );

            let mut view_cursor = cursor;
            let mut map_output_views = Vec::with_capacity(n_maps);
            let mut map_input_buffer_inits = Vec::with_capacity(n_maps);
            for (map_idx, dest) in map_destinations.iter().enumerate() {
                match dest {
                    SoacDestination::Fresh => {
                        map_output_views.push(None);
                        map_input_buffer_inits.push(None);
                    }
                    SoacDestination::OutputView => {
                        let view = *se
                            .operand_nodes
                            .get(view_cursor)
                            .expect("Screma[OutputView] has mapped output_view operand");
                        view_cursor += 1;
                        map_output_views.push(Some(view));
                        map_input_buffer_inits.push(None);
                    }
                    SoacDestination::InputBuffer => {
                        // Consuming map: loop carries `inputs[map_idx]`
                        // (or `inputs[0]` for single-input Screma) as the
                        // initial output, so the result aliases the input
                        // buffer in place (same shape as
                        // `PendingSoac::Map[InputBuffer]`).
                        let carry_from = input_nids.get(map_idx).copied().unwrap_or(input_nids[0]);
                        map_output_views.push(None);
                        map_input_buffer_inits.push(Some(carry_from));
                    }
                }
            }
            let mut acc_output_views = Vec::with_capacity(n_accs);
            let mut acc_input_buffer_inits = Vec::with_capacity(n_accs);
            for dest in &acc_destinations {
                match dest {
                    SoacDestination::Fresh => {
                        acc_output_views.push(None);
                        acc_input_buffer_inits.push(None);
                    }
                    SoacDestination::OutputView => {
                        let view = *se
                            .operand_nodes
                            .get(view_cursor)
                            .expect("Screma[OutputView] has accumulator output_view operand");
                        view_cursor += 1;
                        acc_output_views.push(Some(view));
                        acc_input_buffer_inits.push(None);
                    }
                    SoacDestination::InputBuffer => {
                        // Consuming Scan accumulator: writes back to the
                        // input buffer in-place. Loop carries inputs[0]
                        // as the initial scan-output value.
                        acc_output_views.push(None);
                        acc_input_buffer_inits.push(Some(input_nids[0]));
                    }
                }
            }

            let map_result_tys: Vec<Type<TypeName>> = result_fields[..n_maps].to_vec();
            let acc_result_tys: Vec<Type<TypeName>> = result_fields[n_maps..].to_vec();
            let acc_elem_tys: Vec<Type<TypeName>> = acc_specs
                .iter()
                .zip(acc_result_tys.iter())
                .map(|(acc, result_ty)| match acc.kind {
                    crate::tlc::ScremaAccumulator::Reduce => result_ty.clone(),
                    crate::tlc::ScremaAccumulator::Scan => {
                        if result_ty.is_array() {
                            result_ty.elem_type().expect("Array has elem").clone()
                        } else if as_soa_tuple(result_ty).is_some() {
                            soa_element_type(result_ty)
                        } else {
                            panic!("Screma[Scan] accumulator result must be an array or SoA tuple")
                        }
                    }
                })
                .collect();

            let read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)> = input_nids
                .iter()
                .zip(arr_tys.iter().zip(elem_tys.iter()))
                .map(|(n, (a, e))| (*n, a.clone(), e.clone()))
                .collect();
            let len_input = (input_nids[0], arr_tys[0].clone());

            let uninit_id = catalog().known().uninit;
            let mut carried = Vec::new();
            let mut result_indices = Vec::with_capacity(n_maps + n_accs);
            let mut map_carried_indices = Vec::with_capacity(n_maps);
            let mut acc_scan_carried_indices = Vec::with_capacity(n_accs);
            let mut acc_current_carried_indices = Vec::with_capacity(n_accs);
            let mut result_field_tys = Vec::with_capacity(n_maps + n_accs);
            // Per-map type used as `arr_ty` for the in-loop
            // `emit_write_element` call. Differs from
            // `map_result_tys[map_idx]` when the destination is
            // `InputBuffer` or `OutputView`: the carried buffer is
            // the actual input (a `View`), not the Screma's pre-
            // decision Composite tuple-field type. Passing the
            // pre-decision type to `emit_write_element` produces
            // an `array_with_inplace` node whose declared result
            // type disagrees with the carried block param's type;
            // the SPIR-V backend then panics trying to lower the
            // bogus `Array[..., Composite, Variable, NoRegion]`.
            let mut map_carried_tys: Vec<Type<TypeName>> = Vec::with_capacity(n_maps);

            for map_idx in 0..n_maps {
                let init = if let Some(view_nid) = map_output_views[map_idx] {
                    view_nid
                } else if let Some(input_nid) = map_input_buffer_inits[map_idx] {
                    // InputBuffer: carry the input array as the initial
                    // output; the loop's `emit_write_element` will fold
                    // updates into it in place.
                    input_nid
                } else {
                    graph.intern_pure(
                        PureOp::Intrinsic {
                            id: uninit_id,
                            overload_idx: 0,
                        },
                        smallvec![],
                        map_result_tys[map_idx].clone(),
                    )
                };
                let carried_ty = map_output_views[map_idx]
                    .and_then(|view_nid| graph.types.get(&view_nid).cloned())
                    .or_else(|| {
                        map_input_buffer_inits[map_idx].and_then(|nid| graph.types.get(&nid).cloned())
                    })
                    .unwrap_or_else(|| map_result_tys[map_idx].clone());
                map_carried_indices.push(carried.len());
                result_indices.push(carried.len());
                result_field_tys.push(carried_ty.clone());
                map_carried_tys.push(carried_ty.clone());
                carried.push((carried_ty, init));
            }
            // Per-Scan-accumulator carried type, used for
            // `emit_write_element` exactly like `map_carried_tys` above.
            // `None` for Reduce accumulators (no buffer carried).
            let mut acc_scan_carried_tys: Vec<Option<Type<TypeName>>> = Vec::with_capacity(n_accs);
            for acc_idx in 0..n_accs {
                match acc_specs[acc_idx].kind {
                    crate::tlc::ScremaAccumulator::Reduce => {
                        acc_scan_carried_indices.push(None);
                        acc_current_carried_indices.push(carried.len());
                        result_indices.push(carried.len());
                        result_field_tys.push(acc_result_tys[acc_idx].clone());
                        acc_scan_carried_tys.push(None);
                        carried.push((acc_elem_tys[acc_idx].clone(), init_acc_nids[acc_idx]));
                    }
                    crate::tlc::ScremaAccumulator::Scan => {
                        let init_scan_out = if let Some(view_nid) = acc_output_views[acc_idx] {
                            view_nid
                        } else if let Some(input_nid) = acc_input_buffer_inits[acc_idx] {
                            // Consuming Scan: carry the input array as
                            // the initial scan output; folds updates in
                            // place via emit_write_element.
                            input_nid
                        } else {
                            graph.intern_pure(
                                PureOp::Intrinsic {
                                    id: uninit_id,
                                    overload_idx: 0,
                                },
                                smallvec![],
                                acc_result_tys[acc_idx].clone(),
                            )
                        };
                        let scan_ty = acc_output_views[acc_idx]
                            .and_then(|view_nid| graph.types.get(&view_nid).cloned())
                            .or_else(|| {
                                acc_input_buffer_inits[acc_idx]
                                    .and_then(|nid| graph.types.get(&nid).cloned())
                            })
                            .unwrap_or_else(|| acc_result_tys[acc_idx].clone());
                        acc_scan_carried_indices.push(Some(carried.len()));
                        result_indices.push(carried.len());
                        result_field_tys.push(scan_ty.clone());
                        acc_scan_carried_tys.push(Some(scan_ty.clone()));
                        carried.push((scan_ty, init_scan_out));
                        acc_current_carried_indices.push(carried.len());
                        carried.push((acc_elem_tys[acc_idx].clone(), init_acc_nids[acc_idx]));
                    }
                }
            }
            let result_tuple_ty = Type::Constructed(TypeName::Tuple(n_maps + n_accs), result_field_tys);
            graph.types.insert(result_nid, result_tuple_ty.clone());
            let result = ResultBinding::TupleFromCarried {
                result_node: result_nid,
                tuple_ty: result_tuple_ty,
                indices: result_indices,
            };

            expand_loop(
                graph,
                control_headers,
                bid,
                idx,
                &len_input,
                &carried,
                &result,
                next_effect,
                false,
                |graph, next_effect, body_bid, idx_nid, carried_nids| {
                    let mut elem_nids = Vec::with_capacity(read_inputs.len());
                    for (arr, arr_ty, elem_ty) in &read_inputs {
                        elem_nids.push(emit_read_element(
                            graph,
                            body_bid,
                            *arr,
                            idx_nid,
                            arr_ty,
                            elem_ty,
                            next_effect,
                        ));
                    }

                    let mut new_carried = Vec::with_capacity(carried_nids.len());
                    for map_idx in 0..n_maps {
                        let out_nid = carried_nids[map_carried_indices[map_idx]];
                        let mut map_operands: smallvec::SmallVec<[NodeId; 4]> =
                            map_input_indices[map_idx].iter().map(|&i| elem_nids[i]).collect();
                        map_operands.extend(map_captures[map_idx].iter().copied());
                        let mapped = graph.intern_pure(
                            PureOp::Call(map_funcs[map_idx].clone()),
                            map_operands,
                            map_output_elem_types[map_idx].clone(),
                        );
                        let new_out = if map_output_views[map_idx].is_some() {
                            let ptr_nid = graph.intern_pure(
                                PureOp::ViewIndex,
                                smallvec![out_nid, idx_nid],
                                map_output_elem_types[map_idx].clone(),
                            );
                            let eff_in = alloc_effect(next_effect);
                            let eff_out = alloc_effect(next_effect);
                            graph.skeleton.blocks[body_bid].side_effects.push(SideEffect {
                                kind: SideEffectKind::Inst(InstKind::Store {
                                    place: Default::default(),
                                    value: ValueRef::Ssa(Default::default()),
                                }),
                                operand_nodes: smallvec![ptr_nid, mapped],
                                result: None,
                                effects: Some((eff_in, eff_out)),
                                span: None,
                            });
                            out_nid
                        } else {
                            emit_write_element(
                                graph,
                                out_nid,
                                idx_nid,
                                mapped,
                                &map_carried_tys[map_idx],
                                &map_output_elem_types[map_idx],
                            )
                        };
                        new_carried.push(new_out);
                    }

                    for acc_idx in 0..n_accs {
                        let acc_nid = carried_nids[acc_current_carried_indices[acc_idx]];
                        let mut reduce_operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![acc_nid];
                        reduce_operands.extend(elem_nids.iter().copied());
                        reduce_operands.extend(acc_step_captures[acc_idx].iter().copied());
                        let new_acc = graph.intern_pure(
                            PureOp::Call(acc_specs[acc_idx].step_func.clone()),
                            reduce_operands,
                            acc_elem_tys[acc_idx].clone(),
                        );
                        if let Some(scan_idx) = acc_scan_carried_indices[acc_idx] {
                            let scan_out_nid = carried_nids[scan_idx];
                            let new_scan_out = if acc_output_views[acc_idx].is_some() {
                                let ptr_nid = graph.intern_pure(
                                    PureOp::ViewIndex,
                                    smallvec![scan_out_nid, idx_nid],
                                    acc_elem_tys[acc_idx].clone(),
                                );
                                let eff_in = alloc_effect(next_effect);
                                let eff_out = alloc_effect(next_effect);
                                graph.skeleton.blocks[body_bid].side_effects.push(SideEffect {
                                    kind: SideEffectKind::Inst(InstKind::Store {
                                        place: Default::default(),
                                        value: ValueRef::Ssa(Default::default()),
                                    }),
                                    operand_nodes: smallvec![ptr_nid, new_acc],
                                    result: None,
                                    effects: Some((eff_in, eff_out)),
                                    span: None,
                                });
                                scan_out_nid
                            } else {
                                emit_write_element(
                                    graph,
                                    scan_out_nid,
                                    idx_nid,
                                    new_acc,
                                    acc_scan_carried_tys[acc_idx]
                                        .as_ref()
                                        .expect("Scan accumulator must have a carried type"),
                                    &acc_elem_tys[acc_idx],
                                )
                            };
                            new_carried.push(new_scan_out);
                            new_carried.push(new_acc);
                        } else {
                            new_carried.push(new_acc);
                        }
                    }
                    new_carried
                },
            );
        }
        SideEffectKind::Pending(PendingSoac::Filter {
            map_func,
            map_capture_count,
            output_elem_type,
            pred_func,
            input_array_type,
            input_elem_type,
            output_capacity_size,
            destination,
            scratch_out,
            len_out,
        }) => {
            let map_func = map_func.clone();
            let map_capture_count = *map_capture_count;
            let output_elem_ty = output_elem_type.clone();
            let pred_func = pred_func.clone();
            let arr_ty = input_array_type.clone();
            let elem_ty = input_elem_type.clone();
            let capacity_size = output_capacity_size.clone();
            let destination = *destination;
            let scratch_out = *scratch_out;
            let len_out = *len_out;

            // Operand layout: [input, ...map_captures, ...pred_captures].
            let arr_nid = se.operand_nodes[0];
            let map_captures: Vec<NodeId> = se.operand_nodes[1..1 + map_capture_count].to_vec();
            let captures: Vec<NodeId> = se.operand_nodes[1 + map_capture_count..].to_vec();
            let result_nid = se.result.expect("Filter has a result");

            build_filter_loop(
                graph,
                control_headers,
                bid,
                idx,
                FilterLoop {
                    arr_nid,
                    arr_ty,
                    elem_ty,
                    output_elem_ty,
                    capacity_size,
                    map_func,
                    map_captures,
                    pred_func,
                    captures,
                    result_node: result_nid,
                    destination,
                    scratch_out,
                    len_out,
                },
                next_effect,
            );
        }
        SideEffectKind::Pending(PendingSoac::Scatter {
            func,
            input_array_types,
            input_elem_types,
            capture_count,
            index_type,
            value_type,
            dest_elem_type,
            space,
        }) => {
            // Operands: [dest_view, inputs.., captures..].
            let dest_view = se.operand_nodes[0];
            let n_inputs = input_array_types.len();
            let input_nids = &se.operand_nodes[1..1 + n_inputs];
            let captures: Vec<NodeId> =
                se.operand_nodes[1 + n_inputs..1 + n_inputs + capture_count].to_vec();
            let read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)> = input_nids
                .iter()
                .zip(input_array_types.iter())
                .zip(input_elem_types.iter())
                .map(|((nid, arr_ty), elem_ty)| (*nid, arr_ty.clone(), elem_ty.clone()))
                .collect();
            let len_input = (input_nids[0], input_array_types[0].clone());
            let result_nid = se.result.expect("Scatter has a result");

            let scatter = ScatterLoop {
                dest_view,
                dest_elem_ty: dest_elem_type.clone(),
                func: func.clone(),
                read_inputs,
                captures,
                index_type: index_type.clone(),
                value_type: value_type.clone(),
                len_input,
                result_node: result_nid,
            };
            if space.is_some() {
                build_parallel_scatter(graph, control_headers, bid, idx, scatter, next_effect);
            } else {
                build_scatter_loop(graph, control_headers, bid, idx, scatter, next_effect);
            }
        }
        SideEffectKind::Pending(PendingSoac::Seg {
            map_funcs,
            accumulators,
            input_array_types,
            input_elem_types,
            map_output_elem_types,
            map_input_indices,
            map_capture_counts,
            map_destinations,
            acc_destinations,
            ..
        }) => {
            // Only the pointwise (no-accumulator) map case expands here;
            // reduce/scan Segs are lowered into chunked phase entries by
            // `egir::parallelize` before expansion reaches them.
            assert!(
                accumulators.is_empty() && acc_destinations.is_empty(),
                "reduce/scan Seg should be lowered before soac_expand"
            );
            let n_inputs = input_array_types.len();
            let input_nids: Vec<NodeId> = se.operand_nodes[..n_inputs].to_vec();
            let mut cursor = n_inputs;
            let mut map_captures = Vec::with_capacity(map_capture_counts.len());
            for count in map_capture_counts {
                map_captures.push(se.operand_nodes[cursor..cursor + *count].to_vec());
                cursor += *count;
            }
            let output_views =
                if map_destinations.iter().all(|dest| *dest == SoacDestination::InputBuffer) {
                    vec![input_nids[0]; map_funcs.len()]
                } else {
                    se.operand_nodes[cursor..].to_vec()
                };
            assert_eq!(
                output_views.len(),
                map_funcs.len(),
                "Seg map has one output view per map"
            );
            let result_nid = se.result.expect("Seg has a result");
            let read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)> = input_nids
                .iter()
                .zip(input_array_types.iter().zip(input_elem_types.iter()))
                .map(|(n, (a, e))| (*n, a.clone(), e.clone()))
                .collect();
            let len_input = (input_nids[0], input_array_types[0].clone());
            build_parallel_maps(
                graph,
                control_headers,
                bid,
                idx,
                ScremaMapsIntoLoop {
                    len_input,
                    read_inputs,
                    func_input_indices: map_input_indices.clone(),
                    output_views,
                    output_elem_tys: map_output_elem_types.clone(),
                    result_node: result_nid,
                    funcs: map_funcs.clone(),
                    captures: map_captures,
                },
                next_effect,
            );
        }

        _ => unreachable!("is_handleable_soac filtered to supported variants"),
    }
}

/// Emit a real loop via `build_loop_skeleton`, invoking `emit_body` in the
/// body block to produce the new carried values, then wire the back-edge.
fn build_loop<F>(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    len_input: &(NodeId, Type<TypeName>),
    carried: &[(Type<TypeName>, NodeId)],
    result: &ResultBinding,
    next_effect: &mut u32,
    mut emit_body: F,
) where
    F: FnMut(&mut EGraph, &mut u32, BlockId, NodeId, &[NodeId]) -> Vec<NodeId>,
{
    let handles = build_loop_skeleton(
        graph,
        control_headers,
        bid,
        idx_in_block,
        LoopSkeletonSpec {
            carried: carried.to_vec(),
            result: result.clone(),
            len_input: len_input.clone(),
        },
    );
    let new_carried = emit_body(
        graph,
        next_effect,
        handles.body,
        handles.idx_nid,
        &handles.carried,
    );
    debug_assert_eq!(new_carried.len(), carried.len());
    let next_i_nid = increment(graph, handles.idx_nid);
    let mut args = new_carried;
    args.push(next_i_nid);
    graph.skeleton.blocks[handles.body].term = SkeletonTerminator::Branch {
        target: handles.header,
        args,
    };
}

/// Try to unroll a small loop; if the trip count isn't statically small (or
/// `allow_unroll` is false), fall back to a real loop. Both paths share the
/// same `emit_body` closure — write iteration logic once.
fn expand_loop<F>(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    len_input: &(NodeId, Type<TypeName>),
    carried: &[(Type<TypeName>, NodeId)],
    result: &ResultBinding,
    next_effect: &mut u32,
    allow_unroll: bool,
    mut emit_body: F,
) where
    F: FnMut(&mut EGraph, &mut u32, BlockId, NodeId, &[NodeId]) -> Vec<NodeId>,
{
    if allow_unroll
        && try_unroll(
            graph,
            bid,
            idx_in_block,
            len_input,
            carried,
            result,
            next_effect,
            &mut emit_body,
        )
    {
        return;
    }
    build_loop(
        graph,
        control_headers,
        bid,
        idx_in_block,
        len_input,
        carried,
        result,
        next_effect,
        emit_body,
    );
}

/// Generic small-loop unroller. Returns `true` if the loop was unrolled
/// straight-line into `bid`; `false` if the trip count isn't statically
/// known to be small, and the caller should fall back to emitting a real
/// loop via `build_loop_skeleton`.
///
/// `emit_body(graph, next_effect, bid, idx_const_nid, carried_in)` produces
/// the `carried_out` NodeIds for one iteration, inlined into `bid`.
fn try_unroll<F>(
    graph: &mut EGraph,
    bid: BlockId,
    idx_in_block: usize,
    len_input: &(NodeId, Type<TypeName>),
    carried: &[(Type<TypeName>, NodeId)],
    result: &ResultBinding,
    next_effect: &mut u32,
    mut emit_body: F,
) -> bool
where
    F: FnMut(&mut EGraph, &mut u32, BlockId, NodeId, &[NodeId]) -> Vec<NodeId>,
{
    const UNROLL_THRESHOLD: usize = 16;

    // SoA-tuple driving inputs don't have a direct `array_size`; skip.
    if as_soa_tuple(&len_input.1).is_some() {
        return false;
    }
    let Some(size_ty) = len_input.1.array_size() else {
        return false;
    };
    let n = match size_ty {
        Type::Constructed(TypeName::Size(n), _) => *n,
        _ => return false,
    };
    if n > UNROLL_THRESHOLD {
        return false;
    }

    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);

    // Stash side-effects that followed the (already-removed) Soac so any
    // effectful reads/writes emitted by `emit_body` land at the Soac's
    // original position.
    let suffix: Vec<SideEffect> = graph.skeleton.blocks[bid].side_effects.drain(idx_in_block..).collect();

    let mut carried_nids: Vec<NodeId> = carried.iter().map(|(_, init)| *init).collect();
    for i in 0..n {
        let idx_nid = graph.intern_pure(PureOp::Int(i.to_string()), smallvec![], i32_ty.clone());
        carried_nids = emit_body(graph, next_effect, bid, idx_nid, &carried_nids);
        debug_assert_eq!(carried_nids.len(), carried.len());
    }

    // Rebind the original SOAC result NodeId from the carried tuple.
    match result {
        ResultBinding::TupleFromCarried {
            result_node,
            tuple_ty,
            indices,
        } => {
            let tuple_parts: smallvec::SmallVec<[NodeId; 4]> =
                indices.iter().map(|idx| carried_nids[*idx]).collect();
            graph.nodes[*result_node] = ENode::Pure {
                op: PureOp::Tuple(tuple_parts.len()),
                operands: tuple_parts,
            };
            graph.types.insert(*result_node, tuple_ty.clone());
        }
        ResultBinding::DummyBool { result_node } => {
            graph.nodes[*result_node] = ENode::Constant(crate::ssa::types::ConstantValue::Bool(false));
        }
    }

    graph.skeleton.blocks[bid].side_effects.extend(suffix);
    true
}

/// `Scan[OutputView]`: `new_acc = func(acc, elem, ...caps); view[i] = new_acc`
/// per iteration. One loop-carried value (scalar accumulator). Writes are
/// effectful so the SOAC's `result_node` is bound to a dummy.

/// MapInto: `y = func(elem1, ..., ...caps); view[i] = y` per iteration. No
/// loop-carried state (writes are effectful); the SOAC "result" is a dummy.

/// Parallel pointwise Screma: one lane per input element, guarded by
/// `if tid < len then body else ()`. Reads the shared inputs once and
/// writes every mapped output field to its corresponding output view —
/// no loop, no phi.
struct ScremaMapsIntoLoop {
    len_input: (NodeId, Type<TypeName>),
    read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)>,
    /// Which `read_inputs` each lane consumes: `func_input_indices[k]` lists the
    /// input positions whose elements feed `funcs[k]`, in order, before its
    /// captures. One entry per func.
    func_input_indices: Vec<Vec<usize>>,
    output_views: Vec<NodeId>,
    output_elem_tys: Vec<Type<TypeName>>,
    result_node: NodeId,
    funcs: Vec<String>,
    captures: Vec<Vec<NodeId>>,
}

fn build_parallel_maps(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    spec: ScremaMapsIntoLoop,
    next_effect: &mut u32,
) {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    let after = graph.skeleton.create_block();
    let suffix: Vec<SideEffect> = graph.skeleton.blocks[bid].side_effects.drain(idx_in_block..).collect();
    let old_term = std::mem::replace(
        &mut graph.skeleton.blocks[bid].term,
        SkeletonTerminator::Unreachable,
    );
    graph.skeleton.blocks[after].side_effects = suffix;
    graph.skeleton.blocks[after].term = old_term;
    if let Some(header_meta) = control_headers.remove(&bid) {
        control_headers.insert(after, header_meta);
    }

    graph.nodes[spec.result_node] = ENode::Constant(crate::ssa::types::ConstantValue::Bool(false));

    let body = graph.skeleton.create_block();
    let known = catalog().known();
    let tid_nid = graph.intern_pure(
        PureOp::Intrinsic {
            id: known.thread_id,
            overload_idx: 0,
        },
        smallvec![],
        u32_ty,
    );
    let i32_from_u32 = catalog().lookup_by_any_name("i32.u32").expect("catalog has i32.u32 bitcast");
    let i_nid = graph.intern_pure(
        PureOp::Intrinsic {
            id: i32_from_u32.id,
            overload_idx: 0,
        },
        smallvec![tid_nid],
        i32_ty.clone(),
    );
    let len_nid = emit_length(graph, spec.len_input.0, &spec.len_input.1, &i32_ty);
    let cond_nid = graph.intern_pure(PureOp::BinOp("<".into()), smallvec![i_nid, len_nid], bool_ty);

    graph.skeleton.blocks[bid].term = SkeletonTerminator::CondBranch {
        cond: cond_nid,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args: vec![],
    };
    control_headers.insert(bid, ControlHeader::Selection { merge: after });

    let mut elems = Vec::with_capacity(spec.read_inputs.len());
    for (arr, arr_ty, elem_ty) in &spec.read_inputs {
        elems.push(emit_read_element(
            graph,
            body,
            *arr,
            i_nid,
            arr_ty,
            elem_ty,
            next_effect,
        ));
    }

    for map_idx in 0..spec.funcs.len() {
        let mut call_operands: smallvec::SmallVec<[NodeId; 4]> =
            spec.func_input_indices[map_idx].iter().map(|&i| elems[i]).collect();
        call_operands.extend(spec.captures[map_idx].iter().copied());
        let y_nid = graph.intern_pure(
            PureOp::Call(spec.funcs[map_idx].clone()),
            call_operands,
            spec.output_elem_tys[map_idx].clone(),
        );
        let ptr_nid = graph.intern_pure(
            PureOp::ViewIndex,
            smallvec![spec.output_views[map_idx], i_nid],
            spec.output_elem_tys[map_idx].clone(),
        );
        let eff_in = alloc_effect(next_effect);
        let eff_out = alloc_effect(next_effect);
        graph.skeleton.blocks[body].side_effects.push(SideEffect {
            kind: SideEffectKind::Inst(InstKind::Store {
                place: Default::default(),
                value: ValueRef::Ssa(Default::default()),
            }),
            operand_nodes: smallvec![ptr_nid, y_nid],
            result: None,
            effects: Some((eff_in, eff_out)),
            span: None,
        });
    }

    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: after,
        args: vec![],
    };
}

/// Scan: `new_acc = func(acc, elem, ...caps); out[i] = new_acc` per iteration.
/// Two loop-carried values: the output array (built via `_w_intrinsic_array_with`)
/// and the scalar accumulator.

/// Filter: per iteration `keep = pred(elem, ...caps); buf' = array_with(buf, count, elem);
/// count' = if keep then count+1 else count`. The buffer write is unconditional —
/// non-passing iterations overwrite the same slot on the next iteration that
/// advances `count`. Two loop-carried values: the buffer and the runtime count.
struct FilterLoop {
    /// The input array node, used both for the read path and for length.
    arr_nid: NodeId,
    arr_ty: Type<TypeName>,
    /// The input element type (what `emit_read_element` yields).
    elem_ty: Type<TypeName>,
    /// The output element type: `map_func`'s return type when a map is fused,
    /// else equal to `elem_ty`. The buffer/result hold this type.
    output_elem_ty: Type<TypeName>,
    /// `Size(N)` — the input's static capacity, reused as the output buffer's
    /// capacity (the upper bound on filtered count).
    capacity_size: Type<TypeName>,
    /// `Some(name)` folds a producer `map(f, …)` in: per element compute
    /// `v = f(elem, ...map_captures)` and keep/test `v` instead of `elem`.
    map_func: Option<String>,
    map_captures: Vec<NodeId>,
    pred_func: String,
    captures: Vec<NodeId>,
    /// The original SOAC result NodeId. After expansion this becomes a
    /// `Tuple(buffer, count)` whose type is `Array[T, Size(N), Bounded]`.
    result_node: NodeId,
    /// `Fresh` allocates a new capacity-N buffer via `uninit`. `InputBuffer`
    /// reuses the input as the output buffer (the result `View` aliases
    /// the input's backing slot). Ownership analysis decides which. Ignored
    /// when `scratch_out` is set (runtime lowering always writes the scratch
    /// buffer).
    destination: SoacDestination,
    /// `Some(br)` selects the runtime scratch-view lowering: compact kept
    /// elements into the storage buffer at `br` and rebind the result to a
    /// runtime-length view over it. `None` is the static Bounded lowering.
    scratch_out: Option<crate::BindingRef>,
    /// `Some(br)` (runtime lowering): also store the final surviving count into
    /// `br[0]`, a host-readable length cell paired with the output buffer (set
    /// when the filter is a compute-entry output). `None` otherwise.
    len_out: Option<crate::BindingRef>,
}

/// The value the filter keeps and tests for a read element: `f(elem, ..caps)`
/// when a producer map is fused, else the element itself.
fn filter_kept_value(graph: &mut EGraph, elem_nid: NodeId, spec: &FilterLoop) -> NodeId {
    match &spec.map_func {
        Some(name) => {
            let mut ops: SmallVec<[NodeId; 4]> = smallvec![elem_nid];
            ops.extend(spec.map_captures.iter().copied());
            graph.intern_pure(PureOp::Call(name.clone()), ops, spec.output_elem_ty.clone())
        }
        None => elem_nid,
    }
}

fn build_filter_loop(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    spec: FilterLoop,
    next_effect: &mut u32,
) {
    if let Some(scratch_out) = spec.scratch_out {
        build_runtime_filter_loop(
            graph,
            control_headers,
            bid,
            idx_in_block,
            spec,
            scratch_out,
            next_effect,
        );
        return;
    }
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    // Composite buffer type — the underlying storage of the Bounded result. It
    // holds the kept (output) elements, so it is typed in `output_elem_ty`.
    let buf_ty = Type::Constructed(
        TypeName::Array,
        vec![
            spec.output_elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            spec.capacity_size.clone(),
            crate::types::no_region(),
        ],
    );

    // Split `bid` into preheader (bid) + after. The suffix (side-effects
    // that followed the Filter) is held in `suffix` until the buffer `Load`
    // is emitted at the head of `after` — any suffix side-effect that
    // references the filter result resolves through `Tuple(buf_load, count)`
    // and must see `buf_load` already elaborated.
    let after = graph.skeleton.create_block();
    let suffix: Vec<SideEffect> = graph.skeleton.blocks[bid].side_effects.drain(idx_in_block..).collect();
    let old_term = std::mem::replace(
        &mut graph.skeleton.blocks[bid].term,
        SkeletonTerminator::Unreachable,
    );
    graph.skeleton.blocks[after].term = old_term;
    if let Some(header_meta) = control_headers.remove(&bid) {
        control_headers.insert(after, header_meta);
    }

    // `after` block param: the surviving count. The buffer place is
    // referenced directly through `buf_place_nid`. Count is `i32`
    // throughout — matches the index type taken by element-place stores,
    // the result type of `length()` at the backend boundary, and
    // `Bounded`'s on-disk `len` field.
    let after_count_nid = graph.alloc_side_effect_result(i32_ty.clone());
    graph.nodes[after_count_nid] = ENode::BlockParam {
        block: after,
        index: 0,
    };
    graph.skeleton.blocks[after].params.push(after_count_nid);

    // Build header, body, then, else_, sel_merge, continue blocks. The
    // SPIR-V structured-control-flow rules need the inner selection's
    // merge to be distinct from the loop's continue target, so this loop
    // uses a separate continue block that's just a forwarder back to the
    // header.
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let then_blk = graph.skeleton.create_block();
    let else_blk = graph.skeleton.create_block();
    let sel_merge = graph.skeleton.create_block();
    let continue_blk = graph.skeleton.create_block();

    // Header block params: count_in, i_in. The buffer place is referenced
    // through `buf_place_nid` directly.
    let count_in_nid = graph.add_block_param(header, 0, i32_ty.clone());
    graph.skeleton.blocks[header].params.push(count_in_nid);
    let i_in_nid = graph.add_block_param(header, 1, i32_ty.clone());
    graph.skeleton.blocks[header].params.push(i_in_nid);

    // Preheader: allocate the function-local buffer place; for an
    // `InputBuffer` destination, seed it with the input array so the result
    // observably aliases the input. `Fresh` skips the init store — every
    // surviving element is written through `PlaceIndex` before `count`
    // advances past it, so unread slots are never observed.
    let buf_place_nid = emit_alloca(graph, bid, buf_ty.clone(), next_effect, None);
    if matches!(spec.destination, SoacDestination::InputBuffer) {
        let _ = emit_store(graph, bid, buf_place_nid, spec.arr_nid, next_effect, None);
    } else if !matches!(spec.destination, SoacDestination::Fresh) {
        panic!("Filter[OutputView] not supported — see filter-consuming-input.md");
    }
    let zero_i32_nid = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone());
    graph.skeleton.blocks[bid].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![zero_i32_nid, zero_i32_nid],
    };

    // Header → cond_br(i<N, body, after(count)). The buffer place is
    // referenced through `buf_place_nid` directly, no block-param carry.
    let len_nid = emit_length(graph, spec.arr_nid, &spec.arr_ty, &i32_ty);
    let cond_nid = graph.intern_pure(
        PureOp::BinOp("<".into()),
        smallvec![i_in_nid, len_nid],
        bool_ty.clone(),
    );
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond: cond_nid,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args: vec![count_in_nid],
    };
    control_headers.insert(
        header,
        ControlHeader::Loop {
            merge: after,
            continue_block: continue_blk,
        },
    );

    // Body: elem = arr[i]; pred = pred_func(elem, captures).
    let elem_nid = emit_read_element(
        graph,
        body,
        spec.arr_nid,
        i_in_nid,
        &spec.arr_ty,
        &spec.elem_ty,
        next_effect,
    );
    // A fused producer map computes the kept value `v = f(elem)`; `pred` tests
    // `v` and `v` is what's written. A plain filter keeps the input element.
    let kept_nid = filter_kept_value(graph, elem_nid, &spec);
    let mut pred_operands: SmallVec<[NodeId; 4]> = smallvec![kept_nid];
    pred_operands.extend(spec.captures.iter().copied());
    let pred_nid = graph.intern_pure(PureOp::Call(spec.pred_func), pred_operands, bool_ty.clone());

    // Body → cond_br(pred, then, else_).
    graph.skeleton.blocks[body].term = SkeletonTerminator::CondBranch {
        cond: pred_nid,
        then_target: then_blk,
        then_args: vec![],
        else_target: else_blk,
        else_args: vec![],
    };
    control_headers.insert(body, ControlHeader::Selection { merge: sel_merge });

    // then: write the accepted value into `buf_place[count_in]`, bump
    // count; Branch(sel_merge, [count_bumped]).
    emit_place_index_store(
        graph,
        then_blk,
        buf_place_nid,
        count_in_nid,
        kept_nid,
        spec.output_elem_ty.clone(),
        next_effect,
        None,
    );
    let one_i32_nid = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone());
    let count_bumped_nid = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![count_in_nid, one_i32_nid],
        i32_ty.clone(),
    );
    graph.skeleton.blocks[then_blk].term = SkeletonTerminator::Branch {
        target: sel_merge,
        args: vec![count_bumped_nid],
    };

    // else_: Branch(sel_merge, [count_in]).
    graph.skeleton.blocks[else_blk].term = SkeletonTerminator::Branch {
        target: sel_merge,
        args: vec![count_in_nid],
    };

    // sel_merge: param count_next; Branch(continue, [count_next]).
    let count_next_nid = graph.add_block_param(sel_merge, 0, i32_ty.clone());
    graph.skeleton.blocks[sel_merge].params.push(count_next_nid);
    graph.skeleton.blocks[sel_merge].term = SkeletonTerminator::Branch {
        target: continue_blk,
        args: vec![count_next_nid],
    };

    // continue: param (count_for_continue);
    // i_next = i+1; Branch(header, [count_for_continue, i_next]).
    let cont_count_nid = graph.add_block_param(continue_blk, 0, i32_ty.clone());
    graph.skeleton.blocks[continue_blk].params.push(cont_count_nid);
    let next_i_nid = increment(graph, i_in_nid);
    graph.skeleton.blocks[continue_blk].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![cont_count_nid, next_i_nid],
    };

    // `after` opens with one whole-array `Load` of the buffer place,
    // then the suffix (held back since the split) — so any suffix
    // side-effect that demands the filter result finds `buf_loaded_nid`
    // already elaborated. Rebind the original result NodeId to a
    // `Tuple(buf, count)` matching the `Bounded` struct layout.
    let buf_loaded_nid = emit_load(graph, after, buf_place_nid, buf_ty.clone(), next_effect, None);
    graph.skeleton.blocks[after].side_effects.extend(suffix);
    graph.nodes[spec.result_node] = ENode::Pure {
        op: PureOp::Tuple(2),
        operands: smallvec![buf_loaded_nid, after_count_nid],
    };
}

/// Runtime-sized `filter` lowering: a single-thread serial scatter into the
/// reserved scratch storage buffer `scratch_out`. The loop carries only a
/// surviving `count` and the input index `i` (both `u32`); kept elements are
/// stored into `scratch_out[count]` and `count` is bumped. The original result
/// node is rebound to a runtime-length view `StorageView(scratch_out)[0, count]`
/// over the buffer — its type (set by `convert_soac_filter`) already carries
/// `Region(scratch_out)`, so the backend recovers the descriptor from the type.
/// All offsets/lengths are `u32` to match the view `{offset, len}` convention.
/// Inputs for a sequential `scatter` expansion. The per-element envelope
/// `func` maps the read input elements (plus captures) to an `(index, value)`
/// pair; the loop projects the pair and writes `dest[index] = value`.
struct ScatterLoop {
    dest_view: NodeId,
    dest_elem_ty: Type<TypeName>,
    func: String,
    /// `(array_nid, array_type, elem_type)` per input, read per iteration.
    read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)>,
    captures: Vec<NodeId>,
    index_type: Type<TypeName>,
    value_type: Type<TypeName>,
    /// Loop bound source — the first input `(nid, array_type)`.
    len_input: (NodeId, Type<TypeName>),
    result_node: NodeId,
}

/// Sequential `scatter`: for each `i`, `dest_view[indices[i]] = values[i]`.
/// The per-element writes are effectful, so the SOAC result binds to a dummy.
/// Out-of-bounds indices are not guarded here (Futhark ignores them; v1 trusts
/// the producer to emit in-bounds indices). This is the serial cut — a parallel
/// (one-thread-per-element) version is deferred; sequential semantics make it a
/// pure optimization.
fn build_scatter_loop(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    spec: ScatterLoop,
    next_effect: &mut u32,
) {
    use super::graph_ops::emit_storage_store;
    let ScatterLoop {
        dest_view,
        dest_elem_ty,
        func,
        read_inputs,
        captures,
        index_type,
        value_type,
        len_input,
        result_node,
    } = spec;

    let result = ResultBinding::DummyBool { result_node };

    expand_loop(
        graph,
        control_headers,
        bid,
        idx_in_block,
        &len_input,
        &[],
        &result,
        next_effect,
        true,
        move |graph, next_effect, blk, i_nid, _carried| {
            // (index, value) = func(inputs[i].., ..captures)
            let mut call_operands: smallvec::SmallVec<[NodeId; 4]> = SmallVec::new();
            for (arr, arr_ty, elem_ty) in &read_inputs {
                let elem = emit_read_element(graph, blk, *arr, i_nid, arr_ty, elem_ty, next_effect);
                call_operands.push(elem);
            }
            call_operands.extend(captures.iter().copied());
            let pair_ty =
                Type::Constructed(TypeName::Tuple(2), vec![index_type.clone(), value_type.clone()]);
            let pair_nid = graph.intern_pure(PureOp::Call(func.clone()), call_operands, pair_ty);
            let scatter_idx = graph.intern_pure(
                PureOp::Project { index: 0 },
                smallvec![pair_nid],
                index_type.clone(),
            );
            let val = graph.intern_pure(
                PureOp::Project { index: 1 },
                smallvec![pair_nid],
                value_type.clone(),
            );
            emit_storage_store(
                graph,
                blk,
                dest_view,
                scatter_idx,
                val,
                dest_elem_ty.clone(),
                next_effect,
                None,
            );
            vec![]
        },
    );
}

fn build_parallel_scatter(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    spec: ScatterLoop,
    next_effect: &mut u32,
) {
    use super::graph_ops::emit_storage_store;
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let ScatterLoop {
        dest_view,
        dest_elem_ty,
        func,
        read_inputs,
        captures,
        index_type,
        value_type,
        len_input,
        result_node,
    } = spec;

    let after = graph.skeleton.create_block();
    let suffix: Vec<SideEffect> = graph.skeleton.blocks[bid].side_effects.drain(idx_in_block..).collect();
    let old_term = std::mem::replace(
        &mut graph.skeleton.blocks[bid].term,
        SkeletonTerminator::Unreachable,
    );
    graph.skeleton.blocks[after].side_effects = suffix;
    graph.skeleton.blocks[after].term = old_term;
    if let Some(header_meta) = control_headers.remove(&bid) {
        control_headers.insert(after, header_meta);
    }

    graph.nodes[result_node] = ENode::Constant(crate::ssa::types::ConstantValue::Bool(false));

    let body = graph.skeleton.create_block();
    let known = catalog().known();
    let tid_nid = graph.intern_pure(
        PureOp::Intrinsic {
            id: known.thread_id,
            overload_idx: 0,
        },
        smallvec![],
        u32_ty,
    );
    let i32_from_u32 = catalog().lookup_by_any_name("i32.u32").expect("catalog has i32.u32 bitcast");
    let i_nid = graph.intern_pure(
        PureOp::Intrinsic {
            id: i32_from_u32.id,
            overload_idx: 0,
        },
        smallvec![tid_nid],
        i32_ty.clone(),
    );
    let len_nid = emit_length(graph, len_input.0, &len_input.1, &i32_ty);
    let cond_nid = graph.intern_pure(PureOp::BinOp("<".into()), smallvec![i_nid, len_nid], bool_ty);

    graph.skeleton.blocks[bid].term = SkeletonTerminator::CondBranch {
        cond: cond_nid,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args: vec![],
    };
    control_headers.insert(bid, ControlHeader::Selection { merge: after });

    let mut call_operands: smallvec::SmallVec<[NodeId; 4]> = SmallVec::new();
    for (arr, arr_ty, elem_ty) in &read_inputs {
        let elem = emit_read_element(graph, body, *arr, i_nid, arr_ty, elem_ty, next_effect);
        call_operands.push(elem);
    }
    call_operands.extend(captures.iter().copied());
    let pair_ty = Type::Constructed(TypeName::Tuple(2), vec![index_type.clone(), value_type.clone()]);
    let pair_nid = graph.intern_pure(PureOp::Call(func), call_operands, pair_ty);
    let scatter_idx = graph.intern_pure(PureOp::Project { index: 0 }, smallvec![pair_nid], index_type);
    let val = graph.intern_pure(PureOp::Project { index: 1 }, smallvec![pair_nid], value_type);
    emit_storage_store(
        graph,
        body,
        dest_view,
        scatter_idx,
        val,
        dest_elem_ty,
        next_effect,
        None,
    );
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: after,
        args: vec![],
    };
}

fn build_runtime_filter_loop(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    spec: FilterLoop,
    scratch_out: crate::BindingRef,
    next_effect: &mut u32,
) {
    use super::graph_ops::{emit_storage_store, intern_storage_view, intern_u32};
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    // A view over the whole scratch buffer, for the per-iteration store. The
    // scratch holds the kept (output) elements.
    let scratch_view = intern_storage_view(graph, scratch_out, spec.output_elem_ty.clone(), None);

    // Split `bid` into preheader (bid) + after, moving the suffix + terminator.
    let after = graph.skeleton.create_block();
    let suffix: Vec<SideEffect> = graph.skeleton.blocks[bid].side_effects.drain(idx_in_block..).collect();
    let old_term = std::mem::replace(
        &mut graph.skeleton.blocks[bid].term,
        SkeletonTerminator::Unreachable,
    );
    graph.skeleton.blocks[after].side_effects = suffix;
    graph.skeleton.blocks[after].term = old_term;
    if let Some(header_meta) = control_headers.remove(&bid) {
        control_headers.insert(after, header_meta);
    }

    // After-block param: the final surviving count.
    let after_count_nid = graph.add_block_param(after, 0, u32_ty.clone());
    graph.skeleton.blocks[after].params.push(after_count_nid);

    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let then_blk = graph.skeleton.create_block();
    let else_blk = graph.skeleton.create_block();
    let sel_merge = graph.skeleton.create_block();
    let continue_blk = graph.skeleton.create_block();

    // Header params: count_in, i_in.
    let count_in_nid = graph.add_block_param(header, 0, u32_ty.clone());
    graph.skeleton.blocks[header].params.push(count_in_nid);
    let i_in_nid = graph.add_block_param(header, 1, u32_ty.clone());
    graph.skeleton.blocks[header].params.push(i_in_nid);

    // Preheader → header(0, 0).
    let zero_count_nid = intern_u32(graph, 0, None);
    let zero_i_nid = intern_u32(graph, 0, None);
    graph.skeleton.blocks[bid].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![zero_count_nid, zero_i_nid],
    };

    // Header → cond_br(i < len, body, after(count)).
    let len_nid = graph.intern_pure(PureOp::StorageViewLen, smallvec![spec.arr_nid], u32_ty.clone());
    let cond_nid = graph.intern_pure(
        PureOp::BinOp("<".into()),
        smallvec![i_in_nid, len_nid],
        bool_ty.clone(),
    );
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond: cond_nid,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args: vec![count_in_nid],
    };
    control_headers.insert(
        header,
        ControlHeader::Loop {
            merge: after,
            continue_block: continue_blk,
        },
    );

    // Body: elem = arr[i]; pred = pred_func(elem, captures); cond_br(pred, then, else).
    let elem_nid = emit_read_element(
        graph,
        body,
        spec.arr_nid,
        i_in_nid,
        &spec.arr_ty,
        &spec.elem_ty,
        next_effect,
    );
    // A fused producer map computes the kept value `v = f(elem)`; `pred` tests
    // `v` and `v` is what's compacted into the scratch buffer.
    let kept_nid = filter_kept_value(graph, elem_nid, &spec);
    let mut pred_operands: SmallVec<[NodeId; 4]> = smallvec![kept_nid];
    pred_operands.extend(spec.captures.iter().copied());
    let pred_nid = graph.intern_pure(PureOp::Call(spec.pred_func), pred_operands, bool_ty.clone());
    graph.skeleton.blocks[body].term = SkeletonTerminator::CondBranch {
        cond: pred_nid,
        then_target: then_blk,
        then_args: vec![],
        else_target: else_blk,
        else_args: vec![],
    };
    control_headers.insert(body, ControlHeader::Selection { merge: sel_merge });

    // then: scratch_out[count] = v; count_bumped = count + 1; Branch(sel_merge, [count_bumped]).
    emit_storage_store(
        graph,
        then_blk,
        scratch_view,
        count_in_nid,
        kept_nid,
        spec.output_elem_ty.clone(),
        next_effect,
        None,
    );
    let one_u32_nid = intern_u32(graph, 1, None);
    let count_bumped_nid = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![count_in_nid, one_u32_nid],
        u32_ty.clone(),
    );
    graph.skeleton.blocks[then_blk].term = SkeletonTerminator::Branch {
        target: sel_merge,
        args: vec![count_bumped_nid],
    };

    // else: Branch(sel_merge, [count_in]).
    graph.skeleton.blocks[else_blk].term = SkeletonTerminator::Branch {
        target: sel_merge,
        args: vec![count_in_nid],
    };

    // sel_merge: param count_next; Branch(continue, [count_next]).
    let count_next_nid = graph.add_block_param(sel_merge, 0, u32_ty.clone());
    graph.skeleton.blocks[sel_merge].params.push(count_next_nid);
    graph.skeleton.blocks[sel_merge].term = SkeletonTerminator::Branch {
        target: continue_blk,
        args: vec![count_next_nid],
    };

    // continue: param cont_count; i_next = i + 1; Branch(header, [cont_count, i_next]).
    let cont_count_nid = graph.add_block_param(continue_blk, 0, u32_ty.clone());
    graph.skeleton.blocks[continue_blk].params.push(cont_count_nid);
    let one_i_nid = intern_u32(graph, 1, None);
    let next_i_nid = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![i_in_nid, one_i_nid],
        u32_ty.clone(),
    );
    graph.skeleton.blocks[continue_blk].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![cont_count_nid, next_i_nid],
    };

    // When the filter is a compute-entry output, store the final surviving
    // count into the paired length cell `len_out[0]` so the host can read how
    // many elements are valid in the (capacity-n) output buffer.
    if let Some(len_br) = spec.len_out {
        let len_view = intern_storage_view(graph, len_br, u32_ty.clone(), None);
        let zero_idx = intern_u32(graph, 0, None);
        emit_storage_store(
            graph,
            after,
            len_view,
            zero_idx,
            after_count_nid,
            u32_ty.clone(),
            next_effect,
            None,
        );
    }

    // Rebind the original result NodeId to the runtime-length view
    // `StorageView(scratch_out)[offset = 0, len = after_count]`. The node's
    // type (carrying `Region(scratch_out)`) is preserved from emit_soac.
    let zero_off_nid = intern_u32(graph, 0, None);
    graph.nodes[spec.result_node] = ENode::Pure {
        op: PureOp::StorageView(crate::op::PureViewSource::Storage(scratch_out)),
        operands: smallvec![zero_off_nid, after_count_nid],
    };
}

/// Description of an accumulator-only SOAC (Reduce, reducing Screma): loop over one or
/// more input arrays, thread a scalar accumulator through a per-iteration call,
/// and yield the final accumulator as the result. No output array.

/// Common skeleton shared by every SOAC expansion: split the enclosing block
/// at the SOAC's index, create header/body/after blocks, wire the preheader
/// branch, and install the condbr on the header.
struct LoopSkeletonSpec {
    /// Per loop-carried value: (type, initial value in preheader).
    /// These become `header`'s block params, in order, followed by the index.
    carried: Vec<(Type<TypeName>, NodeId)>,
    /// How the original SOAC result NodeId should be rebound after expansion.
    result: ResultBinding,
    /// Input array for length calculation: (arr_nid, arr_ty).
    len_input: (NodeId, Type<TypeName>),
}

#[derive(Clone)]
enum ResultBinding {
    /// Rebind `result_node` as a tuple of carried values. Used by
    /// Screma, which produces N maps + N accumulators into one tuple.
    TupleFromCarried {
        result_node: NodeId,
        tuple_ty: Type<TypeName>,
        indices: Vec<usize>,
    },
    /// Rebind `result_node` as a constant `Bool(false)` (dummy) — the SOAC
    /// produces no consumed value (the OutputView destination's writes
    /// are effectful and the "result" is discarded by the entry-point
    /// finalize step).
    DummyBool {
        result_node: NodeId,
    },
}

struct LoopHandles {
    header: BlockId,
    body: BlockId,
    /// One NodeId per loop-carried, matching the order in `spec.carried`.
    /// These are the header block-param NodeIds, available inside body and
    /// on the else branch into `after`.
    carried: Vec<NodeId>,
    /// The header's index block param.
    idx_nid: NodeId,
}

fn build_loop_skeleton(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    spec: LoopSkeletonSpec,
) -> LoopHandles {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    // Split `bid` into preheader (bid) + after (holding suffix side-effects + old term).
    let after = graph.skeleton.create_block();
    let suffix: Vec<SideEffect> = graph.skeleton.blocks[bid].side_effects.drain(idx_in_block..).collect();
    let old_term = std::mem::replace(
        &mut graph.skeleton.blocks[bid].term,
        SkeletonTerminator::Unreachable,
    );
    graph.skeleton.blocks[after].side_effects = suffix;
    graph.skeleton.blocks[after].term = old_term;

    // The split moves the branching terminator to `after`: if `bid`
    // carries structured-control-flow header metadata (e.g. a Selection
    // whose CondBranch is in `old_term`), that metadata follows to
    // `after`, since `bid`'s new terminator is an unconditional branch to
    // the loop header — `after` is the selection/loop header now.
    if let Some(header_meta) = control_headers.remove(&bid) {
        control_headers.insert(after, header_meta);
    }

    // Rebind the SOAC's original result NodeId:
    //   - Carried: becomes the `after` block's param, populated from
    //     `carried[idx]` via the header's else branch below.
    //   - DummyBool: becomes an inline `Bool(false)` constant node in place.
    //     Consumers (if any) see a scalar false, matching the SSA pass's
    //     dummy-result convention for effect-only variants.
    match &spec.result {
        ResultBinding::TupleFromCarried {
            result_node,
            tuple_ty,
            indices,
        } => {
            let mut operands = smallvec::SmallVec::new();
            for (param_idx, carried_idx) in indices.iter().enumerate() {
                let Some((part_ty, _)) = spec.carried.get(*carried_idx) else {
                    continue;
                };
                let part_nid = graph.add_block_param(after, param_idx, part_ty.clone());
                graph.skeleton.blocks[after].params.push(part_nid);
                operands.push(part_nid);
            }
            graph.nodes[*result_node] = ENode::Pure {
                op: PureOp::Tuple(operands.len()),
                operands,
            };
            graph.types.insert(*result_node, tuple_ty.clone());
        }
        ResultBinding::DummyBool { result_node } => {
            graph.nodes[*result_node] = ENode::Constant(crate::ssa::types::ConstantValue::Bool(false));
        }
    }

    // Build header with one block-param per carried plus the index.
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let mut carried_nids = Vec::with_capacity(spec.carried.len());
    for (i, (ty, _)) in spec.carried.iter().enumerate() {
        let nid = graph.add_block_param(header, i, ty.clone());
        graph.skeleton.blocks[header].params.push(nid);
        carried_nids.push(nid);
    }
    let idx_nid = graph.add_block_param(header, spec.carried.len(), i32_ty.clone());
    graph.skeleton.blocks[header].params.push(idx_nid);

    // Preheader terminator: br header(init_carried..., 0).
    let zero_nid = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone());
    let mut preheader_args: Vec<NodeId> = spec.carried.iter().map(|(_, init)| *init).collect();
    preheader_args.push(zero_nid);
    graph.skeleton.blocks[bid].term = SkeletonTerminator::Branch {
        target: header,
        args: preheader_args,
    };

    // Header terminator: condbr i<len -> body / after(result_carried).
    let len_nid = emit_length(graph, spec.len_input.0, &spec.len_input.1, &i32_ty);
    let cond_nid = graph.intern_pure(PureOp::BinOp("<".into()), smallvec![idx_nid, len_nid], bool_ty);
    let else_args: Vec<NodeId> = match &spec.result {
        ResultBinding::TupleFromCarried { indices, .. } => {
            indices.iter().map(|idx| carried_nids[*idx]).collect()
        }
        // No `after` block param in the dummy case — branch with empty args.
        ResultBinding::DummyBool { .. } => vec![],
    };
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond: cond_nid,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args,
    };

    control_headers.insert(
        header,
        ControlHeader::Loop {
            merge: after,
            continue_block: body,
        },
    );

    LoopHandles {
        header,
        body,
        carried: carried_nids,
        idx_nid,
    }
}

/// Emit `idx + 1` as a pure op.
fn increment(graph: &mut EGraph, idx_nid: NodeId) -> NodeId {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let one_nid = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone());
    graph.intern_pure(PureOp::BinOp("+".into()), smallvec![idx_nid, one_nid], i32_ty)
}

/// Emit the length of an input array as i32.
/// Composite, view, and virtual arrays share `_w_intrinsic_length`. For a SoA
/// tuple, the length is the length of component 0 (all components share it
/// post-`tlc::soa`).
fn emit_length(
    graph: &mut EGraph,
    arr_nid: NodeId,
    arr_ty: &Type<TypeName>,
    i32_ty: &Type<TypeName>,
) -> NodeId {
    let actual_arr_ty = graph.types.get(&arr_nid).filter(|ty| is_plain_array_source(ty)).cloned();
    let arr_ty = actual_arr_ty.as_ref().unwrap_or(arr_ty);
    if let Some(components) = as_soa_tuple(arr_ty) {
        let first_arr = graph.intern_pure(
            PureOp::Project { index: 0 },
            smallvec![arr_nid],
            components[0].clone(),
        );
        return emit_length(graph, first_arr, &components[0], i32_ty);
    }
    let length_id = catalog().known().length;
    graph.intern_pure(
        PureOp::Intrinsic {
            id: length_id,
            overload_idx: 0,
        },
        smallvec![arr_nid],
        i32_ty.clone(),
    )
}

/// Emit a per-iteration read of `arr[idx]` at the given body block.
/// Composite arrays use a pure `Index`; view arrays use `StorageViewIndex` +
/// effectful `Load`.
fn emit_read_element(
    graph: &mut EGraph,
    body: BlockId,
    arr_nid: NodeId,
    idx_nid: NodeId,
    arr_ty: &Type<TypeName>,
    elem_ty: &Type<TypeName>,
    next_effect: &mut u32,
) -> NodeId {
    let actual_arr_ty = graph.types.get(&arr_nid).filter(|ty| is_plain_array_source(ty)).cloned();
    let arr_ty = actual_arr_ty.as_ref().unwrap_or(arr_ty);
    // SoA tuple: project each component array, recursively read element i
    // from each, repack as the element tuple.
    if let Some(components) = as_soa_tuple(arr_ty) {
        let elem_components: Vec<Type<TypeName>> = components
            .iter()
            .map(|ct| {
                if ct.is_array() {
                    ct.elem_type().expect("Array has elem").clone()
                } else if as_soa_tuple(ct).is_some() {
                    soa_element_type(ct)
                } else {
                    ct.clone()
                }
            })
            .collect();
        let mut elem_nids: SmallVec<[NodeId; 4]> = SmallVec::with_capacity(components.len());
        for (i, (comp_ty, comp_elem_ty)) in components.iter().zip(elem_components.iter()).enumerate() {
            let comp_arr = graph.intern_pure(
                PureOp::Project { index: i as u32 },
                smallvec![arr_nid],
                comp_ty.clone(),
            );
            let e = emit_read_element(graph, body, comp_arr, idx_nid, comp_ty, comp_elem_ty, next_effect);
            elem_nids.push(e);
        }
        return graph.intern_pure(PureOp::Tuple(components.len()), elem_nids, elem_ty.clone());
    }
    if is_view_source(arr_ty) {
        // View array: ViewIndex (pure, PlaceId) + Load (effectful).
        let ptr_nid = graph.intern_pure(PureOp::ViewIndex, smallvec![arr_nid, idx_nid], elem_ty.clone());
        let load_result = graph.alloc_side_effect_result(elem_ty.clone());
        let eff_in = alloc_effect(next_effect);
        let eff_out = alloc_effect(next_effect);
        graph.skeleton.blocks[body].side_effects.push(SideEffect {
            kind: SideEffectKind::Inst(InstKind::Load {
                place: Default::default(),
            }),
            operand_nodes: smallvec![ptr_nid],
            result: Some(load_result),
            effects: Some((eff_in, eff_out)),
            span: None,
        });
        load_result
    } else if is_virtual_source(arr_ty) {
        // Virtual {start, step, len}: elem = start + i * step.
        let start_nid =
            graph.intern_pure(PureOp::Project { index: 0 }, smallvec![arr_nid], elem_ty.clone());
        let step_nid = graph.intern_pure(PureOp::Project { index: 1 }, smallvec![arr_nid], elem_ty.clone());
        let mul_nid = graph.intern_pure(
            PureOp::BinOp("*".into()),
            smallvec![idx_nid, step_nid],
            elem_ty.clone(),
        );
        graph.intern_pure(
            PureOp::BinOp("+".into()),
            smallvec![start_nid, mul_nid],
            elem_ty.clone(),
        )
    } else {
        graph.intern_pure(PureOp::Index, smallvec![arr_nid, idx_nid], elem_ty.clone())
    }
}

/// Emit a per-iteration write `arr[idx] = val`, producing the new array node.
///
/// `elem_ty` must be the logical element type of `arr_ty`:
/// - Plain composite array: `arr_ty.elem_type()`.
/// - SoA tuple: `soa_element_type(arr_ty)` (a tuple whose components line
///   up with `as_soa_tuple(arr_ty)`).
///
/// For a SoA tuple, this projects each component array out of `arr_nid`,
/// projects the matching component out of `val_nid`, recursively writes,
/// and repacks a `PureOp::Tuple`. For a plain composite array, this emits
/// `_w_intrinsic_array_with_inplace` directly. Any other `arr_ty` (view,
/// virtual, tuple whose elements aren't all arrays) is a bug in the caller
/// — soac_expand's output arrays are always freshly-built composites.
fn emit_write_element(
    graph: &mut EGraph,
    arr_nid: NodeId,
    idx_nid: NodeId,
    val_nid: NodeId,
    arr_ty: &Type<TypeName>,
    elem_ty: &Type<TypeName>,
) -> NodeId {
    // Invariant: the supplied elem_ty must match what arr_ty implies.
    // A mismatch means an upstream pass produced inconsistent types.
    // Hard panic — emitting silently-wrong IR in release is worse than
    // crashing loudly.
    let expected_elem_ty = derive_elem_ty(arr_ty);
    if elem_ty != &expected_elem_ty {
        panic!(
            "emit_write_element: elem_ty {:?} disagrees with arr_ty {:?} (expected elem {:?})",
            elem_ty, arr_ty, expected_elem_ty
        );
    }

    if let Some(components) = as_soa_tuple(arr_ty) {
        let Type::Constructed(TypeName::Tuple(_), elem_components) = elem_ty else {
            panic!(
                "emit_write_element: SoA-tuple arr_ty {:?} paired with non-tuple elem_ty {:?}",
                arr_ty, elem_ty
            );
        };
        if components.len() != elem_components.len() {
            panic!(
                "emit_write_element: SoA tuple arity mismatch — arr_ty has {} components, elem_ty has {}",
                components.len(),
                elem_components.len()
            );
        }
        let mut new_component_arrs: SmallVec<[NodeId; 4]> = SmallVec::with_capacity(components.len());
        for (i, (comp_arr_ty, comp_elem_ty)) in components.iter().zip(elem_components.iter()).enumerate() {
            let comp_arr = graph.intern_pure(
                PureOp::Project { index: i as u32 },
                smallvec![arr_nid],
                comp_arr_ty.clone(),
            );
            let comp_val = graph.intern_pure(
                PureOp::Project { index: i as u32 },
                smallvec![val_nid],
                comp_elem_ty.clone(),
            );
            let new_comp =
                emit_write_element(graph, comp_arr, idx_nid, comp_val, comp_arr_ty, comp_elem_ty);
            new_component_arrs.push(new_comp);
        }
        return graph.intern_pure(
            PureOp::Tuple(components.len()),
            new_component_arrs,
            arr_ty.clone(),
        );
    }

    let inplace_id = catalog().known().array_with_in_place;
    graph.intern_pure(
        PureOp::Intrinsic {
            id: inplace_id,
            overload_idx: 0,
        },
        smallvec![arr_nid, idx_nid, val_nid],
        arr_ty.clone(),
    )
}

/// The logical element type implied by `arr_ty`: `arr_ty.elem_type()` for
/// composite arrays, `soa_element_type(arr_ty)` for SoA tuples. Only used
/// by `emit_write_element`'s debug_assert.
fn derive_elem_ty(arr_ty: &Type<TypeName>) -> Type<TypeName> {
    if as_soa_tuple(arr_ty).is_some() {
        soa_element_type(arr_ty)
    } else {
        arr_ty.elem_type().expect("composite array has elem").clone()
    }
}

#[cfg(test)]
#[path = "soac_expand_tests.rs"]
mod soac_expand_tests;
