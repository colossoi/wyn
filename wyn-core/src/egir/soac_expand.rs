//! Expand `SideEffectKind::Pending(PendingSoac::...)` skeleton side-effects
//! into explicit loop subgraphs with pure ops in the sea and block params
//! carrying accumulators.
//!
//! Runs after `from_tlc` populates the EGraph and before `elaborate` produces
//! the final `FuncBody`. Every variant must be handled here — there is no
//! fallback. Any `Pending` left in the skeleton at elaboration time is a bug.

use std::collections::HashMap;

use crate::ssa::framework::BlockId;
use polytype::Type;
use smallvec::{SmallVec, smallvec};

use super::types::EffectToken;
use crate::ast::TypeName;
use crate::intrinsics::{INTRINSIC_ARRAY_WITH_INPLACE, INTRINSIC_LENGTH, INTRINSIC_UNINIT};
use crate::ssa::types::{ControlHeader, InstKind, ValueRef};
use crate::types::TypeExt;
use crate::types::{is_array_variant_composite, is_array_variant_view, is_virtual_array};

use super::types::{
    EGraph, ENode, NodeId, PendingSoac, PureOp, SideEffect, SideEffectKind, SkeletonTerminator,
    SoacDestination,
};

/// Expand every `SideEffectKind::Pending(PendingSoac::...)` in the skeleton.
///
/// `unroll_maps`: when true, Map over statically-sized arrays up to 16
/// elements is unrolled into straight-line code. GLSL targets pass `false`
/// (the GLSL structurizer prefers explicit loops; GLSL drivers unroll on
/// their own).
pub fn run(graph: &mut EGraph, control_headers: &mut HashMap<BlockId, ControlHeader>, unroll_maps: bool) {
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
        expand_one(graph, control_headers, bid, idx, &mut next_effect, unroll_maps);
    }
    let _ = next_effect; // silence unused when no view-array reductions
}

/// Find the next unused EffectToken by scanning all existing skeleton side-effects.
fn next_effect_token(graph: &EGraph) -> u32 {
    let mut max = 0u32;
    for (_, block) in &graph.skeleton.blocks {
        for se in &block.side_effects {
            if let Some((a, b)) = se.effects {
                max = max.max(a.0).max(b.0);
            }
        }
    }
    max + 1
}

fn alloc_effect(next: &mut u32) -> EffectToken {
    let t = EffectToken(*next);
    *next += 1;
    t
}

/// Does this SOAC kind have a TLC→EGIR expansion implemented here?
fn is_handleable_soac(kind: &SideEffectKind) -> bool {
    let SideEffectKind::Pending(soac) = kind else {
        return false;
    };
    match soac {
        PendingSoac::Reduce { input_array_type, .. } => is_plain_array_source(input_array_type),
        PendingSoac::Redomap {
            input_array_types, ..
        } => input_array_types.iter().all(is_plain_array_source),
        PendingSoac::Scan { input_array_type, .. } => is_plain_composite(input_array_type),
        PendingSoac::Map {
            input_array_types, ..
        } => input_array_types.iter().all(is_plain_array_source),
        PendingSoac::ScanInto { input_array_type, .. } => is_plain_array_source(input_array_type),
    }
}

fn is_plain_composite(arr_ty: &Type<TypeName>) -> bool {
    match arr_ty {
        Type::Constructed(TypeName::Array, args) if args.len() == 3 => {
            is_array_variant_composite(&args[2]) && !is_virtual_array(arr_ty)
        }
        _ => false,
    }
}

/// Input-array shape handled today: composite/view/virtual arrays, or SoA
/// tuples `([n]A, [n]B, ...)` (produced by `tlc::soa`) whose components are
/// themselves handleable.
fn is_plain_array_source(arr_ty: &Type<TypeName>) -> bool {
    if matches!(arr_ty, Type::Constructed(TypeName::Array, args) if args.len() == 3) {
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
    let all_soa = components.iter().all(|ct| {
        matches!(ct, Type::Constructed(TypeName::Array, args) if args.len() == 3)
            || as_soa_tuple(ct).is_some()
    });
    if all_soa { Some(components) } else { None }
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
            if args.len() == 3 && is_array_variant_view(&args[2])
    )
}

fn is_virtual_source(arr_ty: &Type<TypeName>) -> bool {
    is_virtual_array(arr_ty)
}

fn expand_one(
    graph: &mut EGraph,
    control_headers: &mut HashMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx: usize,
    next_effect: &mut u32,
    unroll_maps: bool,
) {
    let se = graph.skeleton.blocks[bid].side_effects.remove(idx);
    match &se.kind {
        SideEffectKind::Pending(PendingSoac::Reduce {
            func,
            input_array_type,
            input_elem_type,
            ..
        }) => {
            let func = func.clone();
            let arr_ty = input_array_type.clone();
            let elem_ty = input_elem_type.clone();

            // Decode operands: [arr, init, ...captures].
            let arr_nid = se.operand_nodes[0];
            let init_nid = se.operand_nodes[1];
            let captures: Vec<NodeId> = se.operand_nodes[2..].to_vec();
            let result_nid = se.result.expect("Reduce has a result");
            let acc_ty = graph.types[&result_nid].clone();

            build_accumulator_loop(
                graph,
                control_headers,
                bid,
                idx,
                AccumulatorLoop {
                    len_input: (arr_nid, arr_ty.clone()),
                    read_inputs: vec![(arr_nid, arr_ty, elem_ty)],
                    init_acc: init_nid,
                    acc_ty,
                    result_node: result_nid,
                    func,
                    captures,
                },
                next_effect,
            );
        }
        SideEffectKind::Pending(PendingSoac::Redomap {
            func,
            input_array_types,
            input_elem_types,
            ..
        }) => {
            let func = func.clone();
            let arr_tys = input_array_types.clone();
            let elem_tys = input_elem_types.clone();

            // Operand layout: [input_0, input_1, ..., init, ...captures, ...reduce_captures]
            let n_inputs = arr_tys.len();
            let input_nids: Vec<NodeId> = se.operand_nodes[..n_inputs].to_vec();
            let init_nid = se.operand_nodes[n_inputs];
            // The remaining operands (captures + reduce_captures) all get
            // passed to the per-iteration `func` — the lowering doesn't call
            // `reduce_func` (that's only used for parallel phase-2 reductions,
            // which the SSA lowering also skips when expanding a sequential loop).
            let captures: Vec<NodeId> = se.operand_nodes[n_inputs + 1..].to_vec();
            let result_nid = se.result.expect("Redomap has a result");
            let acc_ty = graph.types[&result_nid].clone();

            let read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)> = input_nids
                .iter()
                .zip(arr_tys.iter().zip(elem_tys.iter()))
                .map(|(n, (a, e))| (*n, a.clone(), e.clone()))
                .collect();
            let len_input = (input_nids[0], arr_tys[0].clone());

            build_accumulator_loop(
                graph,
                control_headers,
                bid,
                idx,
                AccumulatorLoop {
                    len_input,
                    read_inputs,
                    init_acc: init_nid,
                    acc_ty,
                    result_node: result_nid,
                    func,
                    captures,
                },
                next_effect,
            );
        }
        SideEffectKind::Pending(PendingSoac::Map {
            func,
            input_array_types,
            input_elem_types,
            output_elem_type,
            destination,
        }) => {
            let func = func.clone();
            let arr_tys = input_array_types.clone();
            let elem_tys = input_elem_types.clone();
            let out_elem_ty = output_elem_type.clone();
            let destination = *destination;

            let n_inputs = arr_tys.len();
            let input_nids: Vec<NodeId> = se.operand_nodes[..n_inputs].to_vec();
            let result_nid = se.result.expect("Map has a result");

            let read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)> = input_nids
                .iter()
                .zip(arr_tys.iter().zip(elem_tys.iter()))
                .map(|(n, (a, e))| (*n, a.clone(), e.clone()))
                .collect();
            let len_input = (input_nids[0], arr_tys[0].clone());

            match destination {
                SoacDestination::Fresh => {
                    // Operand layout: [input_0, ..., input_{n-1}, ...captures].
                    let captures: Vec<NodeId> = se.operand_nodes[n_inputs..].to_vec();
                    let out_arr_ty = graph.types[&result_nid].clone();

                    let init_out_nid = graph.intern_pure(
                        PureOp::Call(INTRINSIC_UNINIT.into()),
                        smallvec![],
                        out_arr_ty.clone(),
                    );
                    let carried = vec![(out_arr_ty.clone(), init_out_nid)];
                    let result = ResultBinding::Carried {
                        result_node: result_nid,
                        idx: 0,
                    };
                    // Don't allow unroll when output or any input is a SoA tuple:
                    // `_w_intrinsic_array_with` targets composite arrays only.
                    let allow_unroll = unroll_maps
                        && as_soa_tuple(&out_arr_ty).is_none()
                        && read_inputs.iter().all(|(_, a, _)| as_soa_tuple(a).is_none());
                    expand_loop(
                        graph,
                        control_headers,
                        bid,
                        idx,
                        &len_input,
                        &carried,
                        &result,
                        next_effect,
                        allow_unroll,
                        |graph, next_effect, body_bid, idx_nid, carried_nids| {
                            let out_nid = carried_nids[0];
                            let mut call_operands: SmallVec<[NodeId; 4]> = SmallVec::new();
                            for (arr, arr_ty, elem_ty) in &read_inputs {
                                let elem_nid = emit_read_element(
                                    graph,
                                    body_bid,
                                    *arr,
                                    idx_nid,
                                    arr_ty,
                                    elem_ty,
                                    next_effect,
                                );
                                call_operands.push(elem_nid);
                            }
                            call_operands.extend(captures.iter().copied());
                            let y_nid = graph.intern_pure(
                                PureOp::Call(func.clone()),
                                call_operands,
                                out_elem_ty.clone(),
                            );
                            // Loop-carried phi kills the previous iteration's
                            // value on the back-edge, so the in-place variant
                            // is always safe for SOAC-generated output arrays.
                            // For SoA-tuple outputs, `emit_write_element` splits
                            // the update into per-component ArrayWith calls
                            // plus a Tuple repack.
                            let new_out = emit_write_element(
                                graph,
                                out_nid,
                                idx_nid,
                                y_nid,
                                &out_arr_ty,
                                &out_elem_ty,
                            );
                            vec![new_out]
                        },
                    );
                }
                SoacDestination::OutputView => {
                    // Operand layout: [input_0, ..., input_{n-1}, ...captures, output_view].
                    let view_nid =
                        *se.operand_nodes.last().expect("Map[OutputView] has output_view operand");
                    let captures: Vec<NodeId> =
                        se.operand_nodes[n_inputs..se.operand_nodes.len() - 1].to_vec();

                    build_map_into_loop(
                        graph,
                        control_headers,
                        bid,
                        idx,
                        MapIntoLoop {
                            len_input,
                            read_inputs,
                            view_nid,
                            out_elem_ty,
                            result_node: result_nid,
                            func,
                            captures,
                        },
                        next_effect,
                    );
                }
                SoacDestination::InputBuffer => {
                    // Operand layout: [input_0, ..., input_{n-1}, ...captures] —
                    // identical to Fresh. The difference is the loop carries
                    // `inputs[0]` instead of a fresh uninit allocation, so
                    // the result aliases the input buffer.
                    let captures: Vec<NodeId> = se.operand_nodes[n_inputs..].to_vec();
                    let buf_nid = input_nids[0];
                    let buf_arr_ty = arr_tys[0].clone();

                    let carried = vec![(buf_arr_ty.clone(), buf_nid)];
                    let result = ResultBinding::Carried {
                        result_node: result_nid,
                        idx: 0,
                    };
                    let allow_unroll = unroll_maps
                        && as_soa_tuple(&buf_arr_ty).is_none()
                        && read_inputs.iter().all(|(_, a, _)| as_soa_tuple(a).is_none());
                    expand_loop(
                        graph,
                        control_headers,
                        bid,
                        idx,
                        &len_input,
                        &carried,
                        &result,
                        next_effect,
                        allow_unroll,
                        |graph, next_effect, body_bid, idx_nid, carried_nids| {
                            let cur_buf = carried_nids[0];
                            let mut call_operands: SmallVec<[NodeId; 4]> = SmallVec::new();
                            for (arr, arr_ty, elem_ty) in &read_inputs {
                                let elem_nid = emit_read_element(
                                    graph,
                                    body_bid,
                                    *arr,
                                    idx_nid,
                                    arr_ty,
                                    elem_ty,
                                    next_effect,
                                );
                                call_operands.push(elem_nid);
                            }
                            call_operands.extend(captures.iter().copied());
                            let y_nid = graph.intern_pure(
                                PureOp::Call(func.clone()),
                                call_operands,
                                out_elem_ty.clone(),
                            );
                            let new_buf = emit_write_element(
                                graph,
                                cur_buf,
                                idx_nid,
                                y_nid,
                                &buf_arr_ty,
                                &out_elem_ty,
                            );
                            vec![new_buf]
                        },
                    );
                }
            }
        }
        SideEffectKind::Pending(PendingSoac::ScanInto {
            func,
            input_array_type,
            input_elem_type,
            ..
        }) => {
            let func = func.clone();
            let arr_ty = input_array_type.clone();
            let elem_ty = input_elem_type.clone();

            // Operand layout: [input, init, ...captures, output_view].
            let arr_nid = se.operand_nodes[0];
            let init_nid = se.operand_nodes[1];
            let view_nid = *se.operand_nodes.last().expect("ScanInto has output_view");
            let captures: Vec<NodeId> = se.operand_nodes[2..se.operand_nodes.len() - 1].to_vec();
            let result_nid = se.result.expect("ScanInto has a (dummy) result");

            build_scan_into_loop(
                graph,
                control_headers,
                bid,
                idx,
                ScanIntoLoop {
                    len_input: (arr_nid, arr_ty.clone()),
                    input: (arr_nid, arr_ty, elem_ty.clone()),
                    init_acc: init_nid,
                    acc_ty: elem_ty,
                    view_nid,
                    result_node: result_nid,
                    func,
                    captures,
                },
                next_effect,
            );
        }
        SideEffectKind::Pending(PendingSoac::Scan {
            func,
            input_array_type,
            input_elem_type,
            ..
        }) => {
            let func = func.clone();
            let arr_ty = input_array_type.clone();
            let elem_ty = input_elem_type.clone();

            // Operand layout for Scan: [input, init, ...captures].
            let arr_nid = se.operand_nodes[0];
            let init_nid = se.operand_nodes[1];
            let captures: Vec<NodeId> = se.operand_nodes[2..].to_vec();
            let result_nid = se.result.expect("Scan has a result");
            // Scan's result is the output array; its type lives on result_nid.
            let out_arr_ty = graph.types[&result_nid].clone();

            build_scan_loop(
                graph,
                control_headers,
                bid,
                idx,
                ScanLoop {
                    len_input: (arr_nid, arr_ty.clone()),
                    input: (arr_nid, arr_ty, elem_ty.clone()),
                    init_acc: init_nid,
                    acc_ty: elem_ty,
                    out_arr_ty,
                    result_node: result_nid,
                    func,
                    captures,
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
    control_headers: &mut HashMap<BlockId, ControlHeader>,
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
    control_headers: &mut HashMap<BlockId, ControlHeader>,
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

    // Rebind the original SOAC result NodeId. For `Carried`, alias to the
    // final carried value by cloning its ENode into `result_node`'s slot.
    match result {
        ResultBinding::Carried { result_node, idx } => {
            let final_nid = carried_nids[*idx];
            if final_nid != *result_node {
                let final_enode = graph.nodes[final_nid].clone();
                graph.nodes[*result_node] = final_enode;
                graph.types.insert(*result_node, carried[*idx].0.clone());
            }
        }
        ResultBinding::DummyBool { result_node } => {
            graph.nodes[*result_node] = ENode::Constant(crate::ssa::types::ConstantValue::Bool(false));
        }
    }

    graph.skeleton.blocks[bid].side_effects.extend(suffix);
    true
}

/// ScanInto: `new_acc = func(acc, elem, ...caps); view[i] = new_acc` per iteration.
/// One loop-carried value (scalar accumulator). Writes are effectful.
struct ScanIntoLoop {
    len_input: (NodeId, Type<TypeName>),
    input: (NodeId, Type<TypeName>, Type<TypeName>),
    init_acc: NodeId,
    acc_ty: Type<TypeName>,
    view_nid: NodeId,
    result_node: NodeId,
    func: String,
    captures: Vec<NodeId>,
}

fn build_scan_into_loop(
    graph: &mut EGraph,
    control_headers: &mut HashMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    spec: ScanIntoLoop,
    next_effect: &mut u32,
) {
    let handles = build_loop_skeleton(
        graph,
        control_headers,
        bid,
        idx_in_block,
        LoopSkeletonSpec {
            carried: vec![(spec.acc_ty.clone(), spec.init_acc)],
            // Result is dummy — writes are effectful.
            result: ResultBinding::DummyBool {
                result_node: spec.result_node,
            },
            len_input: spec.len_input,
        },
    );

    // ScanInto's result is dummy but scalar acc is still threaded; override
    // the header's else-branch to carry acc (build_loop_skeleton defaults to
    // empty for DummyBool, which is correct when there are no carried values,
    // but here we DO have carried — DummyBool just says "don't pass to after").
    // The after block has no params in the dummy case; we drop the acc at exit.

    let acc_nid = handles.carried[0];
    let idx_nid = handles.idx_nid;

    let (arr, arr_ty, elem_ty) = spec.input;
    let elem_nid = emit_read_element(graph, handles.body, arr, idx_nid, &arr_ty, &elem_ty, next_effect);

    let mut call_operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![acc_nid, elem_nid];
    call_operands.extend(spec.captures.iter().copied());
    let new_acc_nid = graph.intern_pure(PureOp::Call(spec.func), call_operands, spec.acc_ty.clone());

    // view[i] = new_acc: ViewIndex (pure, produces a PlaceId) + Store (effectful).
    let ptr_nid = graph.intern_pure(PureOp::ViewIndex, smallvec![spec.view_nid, idx_nid], spec.acc_ty);
    let eff_in = alloc_effect(next_effect);
    let eff_out = alloc_effect(next_effect);
    graph.skeleton.blocks[handles.body].side_effects.push(SideEffect {
        kind: SideEffectKind::Inst(InstKind::Store {
            place: Default::default(),
            value: ValueRef::Ssa(Default::default()),
        }),
        operand_nodes: smallvec![ptr_nid, new_acc_nid],
        result: None,
        effects: Some((eff_in, eff_out)),
        span: None,
    });

    let next_i_nid = increment(graph, idx_nid);
    graph.skeleton.blocks[handles.body].term = SkeletonTerminator::Branch {
        target: handles.header,
        args: vec![new_acc_nid, next_i_nid],
    };
}

/// MapInto: `y = func(elem1, ..., ...caps); view[i] = y` per iteration. No
/// loop-carried state (writes are effectful); the SOAC "result" is a dummy.
struct MapIntoLoop {
    len_input: (NodeId, Type<TypeName>),
    read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)>,
    /// The storage view to write into.
    view_nid: NodeId,
    out_elem_ty: Type<TypeName>,
    result_node: NodeId,
    func: String,
    captures: Vec<NodeId>,
}

fn build_map_into_loop(
    graph: &mut EGraph,
    control_headers: &mut HashMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    spec: MapIntoLoop,
    next_effect: &mut u32,
) {
    let handles = build_loop_skeleton(
        graph,
        control_headers,
        bid,
        idx_in_block,
        LoopSkeletonSpec {
            carried: vec![],
            result: ResultBinding::DummyBool {
                result_node: spec.result_node,
            },
            len_input: spec.len_input,
        },
    );

    let idx_nid = handles.idx_nid;

    // y = func(elem1, ..., ...caps)
    let mut call_operands: smallvec::SmallVec<[NodeId; 4]> = SmallVec::new();
    for (arr, arr_ty, elem_ty) in &spec.read_inputs {
        let elem_nid = emit_read_element(graph, handles.body, *arr, idx_nid, arr_ty, elem_ty, next_effect);
        call_operands.push(elem_nid);
    }
    call_operands.extend(spec.captures.iter().copied());
    let y_nid = graph.intern_pure(PureOp::Call(spec.func), call_operands, spec.out_elem_ty.clone());

    // view[i] = y: ViewIndex (pure, produces a PlaceId) + Store (effectful).
    let ptr_nid = graph.intern_pure(
        PureOp::ViewIndex,
        smallvec![spec.view_nid, idx_nid],
        spec.out_elem_ty,
    );
    let eff_in = alloc_effect(next_effect);
    let eff_out = alloc_effect(next_effect);
    graph.skeleton.blocks[handles.body].side_effects.push(SideEffect {
        kind: SideEffectKind::Inst(InstKind::Store {
            place: Default::default(),
            value: ValueRef::Ssa(Default::default()),
        }),
        operand_nodes: smallvec![ptr_nid, y_nid],
        result: None,
        effects: Some((eff_in, eff_out)),
        span: None,
    });

    let next_i_nid = increment(graph, idx_nid);
    graph.skeleton.blocks[handles.body].term = SkeletonTerminator::Branch {
        target: handles.header,
        args: vec![next_i_nid],
    };
}

/// Scan: `new_acc = func(acc, elem, ...caps); out[i] = new_acc` per iteration.
/// Two loop-carried values: the output array (built via `_w_intrinsic_array_with`)
/// and the scalar accumulator.
struct ScanLoop {
    len_input: (NodeId, Type<TypeName>),
    input: (NodeId, Type<TypeName>, Type<TypeName>),
    init_acc: NodeId,
    acc_ty: Type<TypeName>,
    out_arr_ty: Type<TypeName>,
    result_node: NodeId,
    func: String,
    captures: Vec<NodeId>,
}

fn build_scan_loop(
    graph: &mut EGraph,
    control_headers: &mut HashMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    spec: ScanLoop,
    next_effect: &mut u32,
) {
    // Preheader initial for the output array: a pure `_w_intrinsic_uninit` call
    // returning an array of the result type. The per-iteration array_with chain
    // fills it. (The SPIR-V backend recognizes the pattern for in-place update.)
    let init_out_nid = graph.intern_pure(
        PureOp::Call(INTRINSIC_UNINIT.into()),
        smallvec![],
        spec.out_arr_ty.clone(),
    );

    let handles = build_loop_skeleton(
        graph,
        control_headers,
        bid,
        idx_in_block,
        LoopSkeletonSpec {
            carried: vec![
                (spec.out_arr_ty.clone(), init_out_nid),
                (spec.acc_ty.clone(), spec.init_acc),
            ],
            result: ResultBinding::Carried {
                result_node: spec.result_node,
                idx: 0,
            }, // the output array is the result

            len_input: spec.len_input,
        },
    );

    let out_nid = handles.carried[0];
    let acc_nid = handles.carried[1];
    let idx_nid = handles.idx_nid;

    // elem = read_element(input, i)
    let (arr, arr_ty, elem_ty) = spec.input;
    let elem_nid = emit_read_element(graph, handles.body, arr, idx_nid, &arr_ty, &elem_ty, next_effect);

    // new_acc = func(acc, elem, ...caps)
    let mut call_operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![acc_nid, elem_nid];
    call_operands.extend(spec.captures.iter().copied());
    let acc_ty = spec.acc_ty;
    let new_acc_nid = graph.intern_pure(PureOp::Call(spec.func), call_operands, acc_ty.clone());

    // out' = array_with_inplace(out, i, new_acc)
    // Loop-carried phi kills the previous iteration's value on the back-edge,
    // so in-place mutation is always safe here. For SoA-tuple outputs,
    // `emit_write_element` splits into per-component ArrayWith + Tuple repack.
    let new_out_nid = emit_write_element(graph, out_nid, idx_nid, new_acc_nid, &spec.out_arr_ty, &acc_ty);

    let next_i_nid = increment(graph, idx_nid);
    graph.skeleton.blocks[handles.body].term = SkeletonTerminator::Branch {
        target: handles.header,
        args: vec![new_out_nid, new_acc_nid, next_i_nid],
    };
}

/// Description of an accumulator-only SOAC (Reduce, Redomap): loop over one or
/// more input arrays, thread a scalar accumulator through a per-iteration call,
/// and yield the final accumulator as the result. No output array.
struct AccumulatorLoop {
    /// The input whose length drives the loop. Usually `read_inputs[0]`.
    len_input: (NodeId, Type<TypeName>),
    /// Input arrays to read per iteration: (arr_nid, arr_ty, elem_ty).
    read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)>,
    /// Initial accumulator value; its NodeId flows into the header's acc param.
    init_acc: NodeId,
    /// Accumulator type.
    acc_ty: Type<TypeName>,
    /// Existing NodeId that consumers of this SOAC's result reference. Rebound
    /// as the `after` block's single param.
    result_node: NodeId,
    /// Function called per iteration as `func(acc, elem1, [elem2, ...], ...captures)`.
    func: String,
    /// Captured values appended to the call's argument list.
    captures: Vec<NodeId>,
}

fn build_accumulator_loop(
    graph: &mut EGraph,
    control_headers: &mut HashMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    spec: AccumulatorLoop,
    next_effect: &mut u32,
) {
    // One loop-carried value: the scalar accumulator. Its exit value is the result.
    let handles = build_loop_skeleton(
        graph,
        control_headers,
        bid,
        idx_in_block,
        LoopSkeletonSpec {
            carried: vec![(spec.acc_ty.clone(), spec.init_acc)],
            result: ResultBinding::Carried {
                result_node: spec.result_node,
                idx: 0,
            },

            len_input: spec.len_input.clone(),
        },
    );

    // --- Body: read each input, call func(acc, elems, caps), br header(new_acc, i+1). ---
    let acc_nid = handles.carried[0];
    let idx_nid = handles.idx_nid;
    let mut call_operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![acc_nid];
    for (arr_nid, arr_ty, elem_ty) in &spec.read_inputs {
        let elem_nid = emit_read_element(
            graph,
            handles.body,
            *arr_nid,
            idx_nid,
            arr_ty,
            elem_ty,
            next_effect,
        );
        call_operands.push(elem_nid);
    }
    call_operands.extend(spec.captures.iter().copied());
    let new_acc_nid = graph.intern_pure(PureOp::Call(spec.func), call_operands, spec.acc_ty);

    let next_i_nid = increment(graph, idx_nid);
    graph.skeleton.blocks[handles.body].term = SkeletonTerminator::Branch {
        target: handles.header,
        args: vec![new_acc_nid, next_i_nid],
    };
}

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
    /// Rebind `result_node` as `after`'s block param populated by
    /// `carried[idx]` when the loop exits. Used for Reduce/Redomap/Scan/Map.
    Carried {
        result_node: NodeId,
        idx: usize,
    },
    /// Rebind `result_node` as a constant `Bool(false)` (dummy) — the SOAC
    /// produces no consumed value (MapInto/ScanInto's writes are effectful
    /// and the "result" is discarded by the entry-point finalize step).
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
    control_headers: &mut HashMap<BlockId, ControlHeader>,
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

    // If `bid` was previously a structured-control-flow header (e.g. a
    // Selection whose CondBranch is in `old_term`), that metadata now
    // describes `after` — bid's new terminator is an unconditional branch to
    // the loop header and is no longer a selection/loop header itself.
    if let Some(header_meta) = control_headers.remove(&bid) {
        control_headers.insert(after, header_meta);
    }

    // Rebind the SOAC's original result NodeId:
    //   - Carried: becomes the `after` block's param, populated from
    //     `carried[idx]` via the header's else branch below.
    //   - DummyBool: becomes an inline `Bool(false)` constant node in place.
    //     Consumers (if any) see a scalar false, matching the SSA pass's
    //     dummy-result convention for effect-only variants.
    match spec.result {
        ResultBinding::Carried { result_node, .. } => {
            graph.nodes[result_node] = ENode::BlockParam {
                block: after,
                index: 0,
            };
            graph.skeleton.blocks[after].params.push(result_node);
        }
        ResultBinding::DummyBool { result_node } => {
            graph.nodes[result_node] = ENode::Constant(crate::ssa::types::ConstantValue::Bool(false));
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
    let else_args: Vec<NodeId> = match spec.result {
        ResultBinding::Carried { idx, .. } => vec![carried_nids[idx]],
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
    if let Some(components) = as_soa_tuple(arr_ty) {
        let first_arr = graph.intern_pure(
            PureOp::Project { index: 0 },
            smallvec![arr_nid],
            components[0].clone(),
        );
        return emit_length(graph, first_arr, &components[0], i32_ty);
    }
    graph.intern_pure(
        PureOp::Intrinsic(INTRINSIC_LENGTH.into()),
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

    graph.intern_pure(
        PureOp::Call(INTRINSIC_ARRAY_WITH_INPLACE.into()),
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
