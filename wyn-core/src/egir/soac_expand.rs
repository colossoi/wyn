//! Expand physical `SideEffectKind::Soac(_, ...)` skeleton side-effects
//! into explicit loop subgraphs with pure ops in the sea and block params
//! carrying accumulators.
//!
//! Runs after `from_tlc` populates the EGraph and before `elaborate` produces
//! the final `FuncBody`. Every variant must be handled here — there is no
//! fallback. Any semantic SOAC left in the skeleton at elaboration time is a bug.

use crate::builtins::catalog;
use crate::flow::{BlockId, ControlHeader};
use crate::LookupMap;

use polytype::Type;
use smallvec::{smallvec, SmallVec};

use super::graph_ops::{alloc_effect, emit_alloca, emit_load, emit_place_index_store, emit_store};
use super::program::{
    PhysicalEGraph as EGraph, PhysicalFilterOutput, PhysicalFilterWorkBuffers as FilterWorkBuffers,
    PhysicalProgram, PhysicalSegSpace as SegSpace, PhysicalSideEffect as SideEffect,
    PhysicalSideEffectKind as SideEffectKind, PhysicalSoac as Soac, RegionInterner,
};
use super::soac::{filter, screma};
use crate::ast::TypeName;
use crate::types::{is_array_variant_view, is_virtual_array, TypeExt};

use super::types::{
    as_soa_tuple, soac_element_type, ENode, EffectOp, EffectToken, NodeId, PureOp, SkeletonTerminator,
    SoacOwnership, SoacPlacement,
};

/// Run `run_one_body` on every function and entry point in the program.
pub fn run(inner: &mut PhysicalProgram, effect_ids: &mut crate::IdSource<EffectToken>) {
    // Borrow the region interner disjointly from the bodies being expanded; it
    // is read-only here (recovering the SSA `Call` name for each region).
    let super::ir::Program {
        functions,
        entry_points,
        region_interner,
        ..
    } = &mut inner.ir;
    for f in functions.iter_mut() {
        run_one_body(&mut f.graph, &mut f.control_headers, region_interner, effect_ids);
    }
    for e in entry_points.iter_mut() {
        run_one_body(&mut e.graph, &mut e.control_headers, region_interner, effect_ids);
    }
}

/// Expand every physical SOAC in the skeleton.
pub fn run_one_body(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    regions: &RegionInterner,
    effect_ids: &mut crate::IdSource<EffectToken>,
) {
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

    for (bid, idx) in targets {
        expand_one(graph, control_headers, bid, idx, effect_ids, regions);
    }
}

/// Does this SOAC kind have a TLC→EGIR expansion implemented here?
fn is_handleable_soac(kind: &SideEffectKind) -> bool {
    let SideEffectKind::Soac(_, soac) = kind else {
        return false;
    };
    match soac {
        Soac::Screma(op) if op.is_serial() => {
            op.lanes().inputs.iter().all(|input| is_plain_array_source(&input.array))
        }
        Soac::Filter(op) => {
            let input = match &op.body.input {
                filter::Input::Plain(input) | filter::Input::Mapped { input, .. } => input,
            };
            is_plain_array_source(&input.array)
        }
        // Scatter reads all input arrays per element; loop length comes from
        // the first input, but every input must support the read path.
        Soac::Hist(op) => {
            !op.body.inputs.is_empty()
                && op.body.inputs.iter().all(|input| is_plain_array_source(&input.array))
        }
        // A reified parallel map/reduce/scan; same source rules as its inputs.
        Soac::Screma(screma::Op::Map {
            lanes,
            state: screma::ScheduledState::Segmented(_),
        }) => lanes.inputs.iter().all(|input| is_plain_array_source(&input.array)),
        Soac::Screma(_) => false,
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

/// Element type of a SoA tuple: `([n]A, [n]B)` → `(A, B)`. Nested SoA tuples
/// recurse into their own element types.
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
    next_effect: &mut crate::IdSource<EffectToken>,
    regions: &RegionInterner,
) {
    let se = graph.skeleton.blocks[bid].side_effects.remove(idx);
    match &se.kind {
        SideEffectKind::Soac(_, Soac::Screma(op)) if op.is_serial() => {
            let lanes = op.lanes();
            let map_input_indices =
                lanes.maps.iter().map(|map| map.input_indices.clone()).collect::<Vec<_>>();
            // Captures and the callee region are explicit on each `SegBody`;
            // the serial loop reads them directly rather than reslicing the
            // operand list by a separate capture-count layout.
            let map_funcs = regions.resolve_cloned(lanes.maps.iter().map(|map| map.body.region));
            let map_captures: Vec<Vec<NodeId>> =
                lanes.maps.iter().map(|map| map.body.captures.clone()).collect();
            let acc_specs = op.operators().into_iter().cloned().collect::<Vec<_>>();
            let acc_is_scan = (0..acc_specs.len()).map(|index| op.is_scan(index)).collect::<Vec<_>>();
            let acc_step_captures: Vec<Vec<NodeId>> =
                acc_specs.iter().map(|acc| acc.step.captures.clone()).collect();
            let arr_tys = lanes.inputs.iter().map(|input| input.array.clone()).collect::<Vec<_>>();
            let elem_tys = lanes.inputs.iter().map(|input| input.element()).collect::<Vec<_>>();
            let map_output_elem_types =
                lanes.maps.iter().map(|map| map.output_element_type.clone()).collect::<Vec<_>>();
            let map_destinations = lanes.maps.iter().map(|map| map.destination).collect::<Vec<_>>();
            let acc_destinations =
                acc_specs.iter().map(|operator| operator.destination).collect::<Vec<_>>();
            let n_maps = map_funcs.len();
            let n_accs = acc_specs.len();
            let n_inputs = arr_tys.len();
            let input_nids: Vec<NodeId> = se.operand_nodes[..n_inputs].to_vec();
            let init_acc_nids = acc_specs.iter().map(|operator| operator.neutral).collect::<Vec<_>>();
            // Operand layout is `[inputs.., output_views..]`; accumulator
            // neutrals and callable captures are explicit in the body.
            let cursor = n_inputs;
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
                match (dest.ownership, dest.placement) {
                    (SoacOwnership::UniqueInput, None) => {
                        panic!("unresolved UniqueInput destination reached physical expansion")
                    }
                    (SoacOwnership::Fresh, None) => {
                        map_output_views.push(None);
                        map_input_buffer_inits.push(None);
                    }
                    (_, Some(SoacPlacement::OutputView)) => {
                        let view = *se
                            .operand_nodes
                            .get(view_cursor)
                            .expect("Screma[OutputView] has mapped output_view operand");
                        view_cursor += 1;
                        map_output_views.push(Some(view));
                        map_input_buffer_inits.push(None);
                    }
                    (_, Some(SoacPlacement::InputBuffer)) => {
                        // Consuming map: loop carries `inputs[map_idx]`
                        // (or `inputs[0]` for single-input Screma) as the
                        // initial output, so the result aliases the input
                        // buffer in place (same shape as
                        // a map with an `InputBuffer` destination).
                        let carry_from = input_nids.get(map_idx).copied().unwrap_or(input_nids[0]);
                        map_output_views.push(None);
                        map_input_buffer_inits.push(Some(carry_from));
                    }
                }
            }
            let mut acc_output_views = Vec::with_capacity(n_accs);
            let mut acc_input_buffer_inits = Vec::with_capacity(n_accs);
            for dest in &acc_destinations {
                match (dest.ownership, dest.placement) {
                    (SoacOwnership::UniqueInput, None) => {
                        panic!("unresolved UniqueInput destination reached physical expansion")
                    }
                    (SoacOwnership::Fresh, None) => {
                        acc_output_views.push(None);
                        acc_input_buffer_inits.push(None);
                    }
                    (_, Some(SoacPlacement::OutputView)) => {
                        let view = *se
                            .operand_nodes
                            .get(view_cursor)
                            .expect("Screma[OutputView] has accumulator output_view operand");
                        view_cursor += 1;
                        acc_output_views.push(Some(view));
                        acc_input_buffer_inits.push(None);
                    }
                    (_, Some(SoacPlacement::InputBuffer)) => {
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
                .enumerate()
                .map(|(index, (_acc, result_ty))| {
                    if !acc_is_scan[index] {
                        result_ty.clone()
                    } else {
                        if result_ty.is_array() {
                            result_ty.elem_type().expect("Array has elem").clone()
                        } else if as_soa_tuple(result_ty).is_some() {
                            soac_element_type(result_ty)
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
            // bogus `Array[..., Composite, Variable, NoBuffer]`.
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
                        None,
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
                if !acc_is_scan[acc_idx] {
                    acc_scan_carried_indices.push(None);
                    acc_current_carried_indices.push(carried.len());
                    result_indices.push(carried.len());
                    result_field_tys.push(acc_result_tys[acc_idx].clone());
                    acc_scan_carried_tys.push(None);
                    carried.push((acc_elem_tys[acc_idx].clone(), init_acc_nids[acc_idx]));
                } else {
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
                            None,
                        )
                    };
                    let scan_ty = acc_output_views[acc_idx]
                        .and_then(|view_nid| graph.types.get(&view_nid).cloned())
                        .or_else(|| {
                            acc_input_buffer_inits[acc_idx].and_then(|nid| graph.types.get(&nid).cloned())
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
            let result_tuple_ty = Type::Constructed(TypeName::Tuple(n_maps + n_accs), result_field_tys);
            graph.retype_node(result_nid, result_tuple_ty.clone());
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
                        let mut map_operands: smallvec::SmallVec<[NodeId; 4]> = map_input_indices[map_idx]
                            .iter()
                            .map(|input| elem_nids[input.index()])
                            .collect();
                        map_operands.extend(map_captures[map_idx].iter().copied());
                        let mapped = graph.intern_pure(
                            PureOp::Call(map_funcs[map_idx].clone()),
                            map_operands,
                            map_output_elem_types[map_idx].clone(),
                            None,
                        );
                        let new_out = if map_output_views[map_idx].is_some() {
                            let ptr_nid = graph.intern_pure(
                                PureOp::ViewIndex,
                                smallvec![out_nid, idx_nid],
                                map_output_elem_types[map_idx].clone(),
                                None,
                            );
                            let eff_in = alloc_effect(next_effect);
                            let eff_out = alloc_effect(next_effect);
                            graph.skeleton.blocks[body_bid].side_effects.push(SideEffect {
                                kind: SideEffectKind::Effect(EffectOp::Store),
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
                        if acc_specs[acc_idx].input_indices.is_empty() {
                            reduce_operands.extend(elem_nids.iter().copied());
                        } else {
                            reduce_operands.extend(
                                acc_specs[acc_idx]
                                    .input_indices
                                    .iter()
                                    .map(|input| elem_nids[input.index()]),
                            );
                        }
                        reduce_operands.extend(acc_step_captures[acc_idx].iter().copied());
                        let new_acc = graph.intern_pure(
                            PureOp::Call(regions.resolve(acc_specs[acc_idx].step.region).to_string()),
                            reduce_operands,
                            acc_elem_tys[acc_idx].clone(),
                            None,
                        );
                        if let Some(scan_idx) = acc_scan_carried_indices[acc_idx] {
                            let scan_out_nid = carried_nids[scan_idx];
                            let new_scan_out = if acc_output_views[acc_idx].is_some() {
                                let ptr_nid = graph.intern_pure(
                                    PureOp::ViewIndex,
                                    smallvec![scan_out_nid, idx_nid],
                                    acc_elem_tys[acc_idx].clone(),
                                    None,
                                );
                                let eff_in = alloc_effect(next_effect);
                                let eff_out = alloc_effect(next_effect);
                                graph.skeleton.blocks[body_bid].side_effects.push(SideEffect {
                                    kind: SideEffectKind::Effect(EffectOp::Store),
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
        SideEffectKind::Soac(_, Soac::Filter(op)) => {
            let (input, map_body) = match &op.body.input {
                filter::Input::Plain(input) => (input, None),
                filter::Input::Mapped { input, body, .. } => (input, Some(body)),
            };
            let map_func = map_body.map(|body| regions.resolve(body.region).to_string());
            let output_elem_ty = op.body.output_element_type();
            let pred_func = regions.resolve(op.body.predicate.region).to_string();
            let arr_ty = input.array.clone();
            let elem_ty = input.element();
            let (output, plan) = match &op.state {
                filter::ScheduledState::Serial { storage, .. } => (storage.clone(), filter::Plan::Serial),
                filter::ScheduledState::Parallel { storage, plan, .. } => {
                    let output = filter::Output::Runtime {
                        scratch: storage.scratch,
                        length: storage.length,
                    };
                    let plan = match plan.stage {
                        filter::ParallelStage::Flags => filter::Plan::Flags(plan.buffers),
                        filter::ParallelStage::Scan => filter::Plan::Scan(plan.buffers),
                        filter::ParallelStage::Scatter => filter::Plan::Scatter(plan.buffers),
                    };
                    (output, plan)
                }
            };

            // Operand layout: [input, ...map_captures, ...pred_captures].
            let arr_nid = se.operand_nodes[0];
            let map_captures = map_body.map(|body| body.captures.clone()).unwrap_or_default();
            let captures = op.body.predicate.captures.clone();
            let result_nid = se.result.expect("Filter has a result");

            let spec = FilterLoop {
                arr_nid,
                arr_ty,
                elem_ty,
                output_elem_ty,
                output,
                map_func,
                map_captures,
                pred_func,
                captures,
                result_node: result_nid,
            };
            match plan {
                filter::Plan::Flags(work) => {
                    build_filter_flags(graph, control_headers, bid, idx, spec, work.flags, next_effect)
                }
                filter::Plan::Scan(work) => {
                    build_filter_scan(graph, control_headers, bid, idx, spec, work, next_effect)
                }
                filter::Plan::Scatter(work) => {
                    build_filter_scatter(graph, control_headers, bid, idx, spec, work, next_effect)
                }
                filter::Plan::Serial => {
                    build_filter_loop(graph, control_headers, bid, idx, spec, next_effect)
                }
            }
        }
        SideEffectKind::Soac(_, Soac::Hist(op)) => {
            // Operands: [dest_view, inputs.., captures..].
            let dest_view = se.operand_nodes[0];
            let n_inputs = op.body.inputs.len();
            let input_nids = &se.operand_nodes[1..1 + n_inputs];
            let captures = op.body.body.captures.clone();
            let read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)> = input_nids
                .iter()
                .zip(op.body.inputs.iter())
                .map(|(nid, input)| (*nid, input.array.clone(), input.element()))
                .collect();
            let len_input = (input_nids[0], op.body.inputs[0].array.clone());
            let result_nid = se.result.expect("Scatter has a result");

            let scatter = ScatterLoop {
                dest_view,
                dest_elem_ty: op.body.dest_elem_type.clone(),
                func: regions.resolve(op.body.body.region).to_string(),
                read_inputs,
                captures,
                index_type: op.body.index_type.clone(),
                value_type: op.body.value_type.clone(),
                len_input,
                result_node: result_nid,
            };
            // Ordered overwrite is non-commutative when indices conflict.
            // Preserve source order until a future update policy proves a
            // conflict-safe parallel implementation.
            build_scatter_loop(graph, control_headers, bid, idx, scatter, next_effect);
        }
        SideEffectKind::Soac(
            _,
            Soac::Screma(screma::Op::Map {
                lanes: screma::Lanes { inputs, maps },
                state: screma::ScheduledState::Segmented(screma::Segmented { space, .. }),
            }),
        ) => {
            // SegRed/SegScan are consumed by `egir::parallelize::lower`
            // before expansion. This arm is therefore semantically map-only.
            let n_inputs = inputs.len();
            let input_nids: Vec<NodeId> = se.operand_nodes[..n_inputs].to_vec();
            // `[inputs.., output_views..]`: views start right after the inputs.
            let cursor = n_inputs;
            let map_captures: Vec<Vec<NodeId>> = maps.iter().map(|map| map.body.captures.clone()).collect();
            let map_funcs = regions.resolve_cloned(maps.iter().map(|map| map.body.region));
            let output_views = if maps.iter().all(|map| map.destination.is_input_buffer()) {
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
                .zip(inputs.iter())
                .map(|(nid, input)| (*nid, input.array.clone(), input.element()))
                .collect();
            let len_input = (input_nids[0], inputs[0].array.clone());
            build_parallel_maps(
                graph,
                control_headers,
                bid,
                idx,
                ScremaMapsIntoLoop {
                    space: space.clone(),
                    len_input,
                    read_inputs,
                    func_input_indices: maps.iter().map(|map| map.input_indices.clone()).collect(),
                    output_views,
                    output_elem_tys: maps.iter().map(|map| map.output_element_type.clone()).collect(),
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
    next_effect: &mut crate::IdSource<EffectToken>,
    mut emit_body: F,
) where
    F: FnMut(&mut EGraph, &mut crate::IdSource<EffectToken>, BlockId, NodeId, &[NodeId]) -> Vec<NodeId>,
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
    next_effect: &mut crate::IdSource<EffectToken>,
    allow_unroll: bool,
    mut emit_body: F,
) where
    F: FnMut(&mut EGraph, &mut crate::IdSource<EffectToken>, BlockId, NodeId, &[NodeId]) -> Vec<NodeId>,
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
    next_effect: &mut crate::IdSource<EffectToken>,
    mut emit_body: F,
) -> bool
where
    F: FnMut(&mut EGraph, &mut crate::IdSource<EffectToken>, BlockId, NodeId, &[NodeId]) -> Vec<NodeId>,
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
        let idx_nid = graph.intern_pure(PureOp::Int(i.to_string()), smallvec![], i32_ty.clone(), None);
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
            graph.replace_pure_node(*result_node, PureOp::Tuple(tuple_parts.len()), tuple_parts);
            graph.retype_node(*result_node, tuple_ty.clone());
        }
        ResultBinding::DummyBool { result_node } => {
            graph.replace_node_preserving_type(
                *result_node,
                ENode::Constant(crate::ssa::types::ConstantValue::Bool(false)),
            );
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
    space: SegSpace,
    len_input: (NodeId, Type<TypeName>),
    read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)>,
    /// Which `read_inputs` each lane consumes: `func_input_indices[k]` lists the
    /// input positions whose elements feed `funcs[k]`, in order, before its
    /// captures. One entry per func.
    func_input_indices: Vec<Vec<screma::InputId>>,
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
    next_effect: &mut crate::IdSource<EffectToken>,
) {
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    let after = graph.skeleton.split_block_before_effect(bid, idx_in_block);
    if let Some(header_meta) = control_headers.remove(&bid) {
        control_headers.insert(after, header_meta);
    }

    graph.replace_node_preserving_type(
        spec.result_node,
        ENode::Constant(crate::ssa::types::ConstantValue::Bool(false)),
    );

    let body = graph.skeleton.create_block();
    let known = catalog().known();
    let tid_nid = graph.intern_pure(
        PureOp::Intrinsic {
            id: known.thread_id,
            overload_idx: 0,
        },
        smallvec![],
        u32_ty,
        None,
    );
    let i32_from_u32 = catalog().lookup_by_any_name("i32.u32").expect("catalog has i32.u32 bitcast");
    let i_nid = graph.intern_pure(
        PureOp::Intrinsic {
            id: i32_from_u32.id,
            overload_idx: 0,
        },
        smallvec![tid_nid],
        i32_ty.clone(),
        None,
    );
    let len_nid = emit_seg_space_len(graph, &spec.space, &spec.len_input, &i32_ty);
    let cond_nid = graph.intern_pure(
        PureOp::BinOp("<".into()),
        smallvec![i_nid, len_nid],
        bool_ty,
        None,
    );

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
            spec.func_input_indices[map_idx].iter().map(|input| elems[input.index()]).collect();
        call_operands.extend(spec.captures[map_idx].iter().copied());
        let y_nid = graph.intern_pure(
            PureOp::Call(spec.funcs[map_idx].clone()),
            call_operands,
            spec.output_elem_tys[map_idx].clone(),
            None,
        );
        let ptr_nid = graph.intern_pure(
            PureOp::ViewIndex,
            smallvec![spec.output_views[map_idx], i_nid],
            spec.output_elem_tys[map_idx].clone(),
            None,
        );
        let eff_in = alloc_effect(next_effect);
        let eff_out = alloc_effect(next_effect);
        graph.skeleton.blocks[body].side_effects.push(SideEffect {
            kind: SideEffectKind::Effect(EffectOp::Store),
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

fn emit_seg_space_len(
    graph: &mut EGraph,
    space: &SegSpace,
    fallback: &(NodeId, Type<TypeName>),
    i32_ty: &Type<TypeName>,
) -> NodeId {
    use crate::egir::types::SegExtent;

    let mut dimensions = Vec::with_capacity(space.dims.len());
    for extent in &space.dims {
        let dimension = match extent {
            SegExtent::Fixed(count) => {
                graph.intern_pure(PureOp::Int(count.to_string()), smallvec![], i32_ty.clone(), None)
            }
            SegExtent::PushConstant { node, .. } => *node,
            SegExtent::Value(node) => {
                let ty = graph.types[node].clone();
                if is_plain_array_source(&ty) {
                    emit_length(graph, *node, &ty, i32_ty)
                } else {
                    *node
                }
            }
            SegExtent::ResourceLength { node, .. } => {
                let ty = graph.types[node].clone();
                emit_length(graph, *node, &ty, i32_ty)
            }
        };
        dimensions.push(dimension);
    }
    let Some(first) = dimensions.first().copied() else {
        return emit_length(graph, fallback.0, &fallback.1, i32_ty);
    };
    dimensions.into_iter().skip(1).fold(first, |product, dimension| {
        graph.intern_pure(
            PureOp::BinOp("*".into()),
            smallvec![product, dimension],
            i32_ty.clone(),
            None,
        )
    })
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
    output: PhysicalFilterOutput,
    /// `Some(name)` folds a producer `map(f, …)` in: per element compute
    /// `v = f(elem, ...map_captures)` and keep/test `v` instead of `elem`.
    map_func: Option<String>,
    map_captures: Vec<NodeId>,
    pred_func: String,
    captures: Vec<NodeId>,
    /// The original SOAC result NodeId. After expansion this becomes a
    /// `Tuple(buffer, count)` whose type is `Array[T, Size(N), Bounded]`.
    result_node: NodeId,
}

/// The value the filter keeps and tests for a read element: `f(elem, ..caps)`
/// when a producer map is fused, else the element itself.
fn filter_kept_value(graph: &mut EGraph, elem_nid: NodeId, spec: &FilterLoop) -> NodeId {
    match &spec.map_func {
        Some(name) => {
            let mut ops: SmallVec<[NodeId; 4]> = smallvec![elem_nid];
            ops.extend(spec.map_captures.iter().copied());
            graph.intern_pure(PureOp::Call(name.clone()), ops, spec.output_elem_ty.clone(), None)
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
    next_effect: &mut crate::IdSource<EffectToken>,
) {
    let runtime_scratch = match &spec.output {
        filter::Output::Runtime { scratch, .. } => Some(*scratch),
        filter::Output::Local { .. } => None,
    };
    if let Some(scratch) = runtime_scratch {
        build_runtime_filter_loop(
            graph,
            control_headers,
            bid,
            idx_in_block,
            spec,
            scratch,
            next_effect,
        );
        return;
    }
    let filter::Output::Local {
        ref capacity,
        destination,
    } = &spec.output
    else {
        unreachable!()
    };
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    // Composite buffer type — the underlying storage of the Bounded result. It
    // holds the kept (output) elements, so it is typed in `output_elem_ty`.
    let buf_ty = Type::Constructed(
        TypeName::Array,
        vec![
            spec.output_elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            capacity.clone(),
            crate::types::no_buffer(),
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
    let after_count_nid = graph.add_block_param(after, i32_ty.clone());

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
    let count_in_nid = graph.add_block_param(header, i32_ty.clone());
    let i_in_nid = graph.add_block_param(header, i32_ty.clone());

    // Preheader: allocate the function-local buffer place; for an
    // `InputBuffer` destination, seed it with the input array so the result
    // observably aliases the input. `Fresh` skips the init store — every
    // surviving element is written through `PlaceIndex` before `count`
    // advances past it, so unread slots are never observed.
    let buf_place_nid = emit_alloca(graph, bid, buf_ty.clone(), next_effect, None);
    if destination.is_input_buffer() {
        let _ = emit_store(graph, bid, buf_place_nid, spec.arr_nid, next_effect, None);
    } else if !destination.is_unplaced_fresh() {
        panic!("Filter[OutputView] not supported — see filter-consuming-input.md");
    }
    let zero_i32_nid = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone(), None);
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
        None,
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
    let pred_nid = graph.intern_pure(PureOp::Call(spec.pred_func), pred_operands, bool_ty.clone(), None);

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
    let one_i32_nid = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone(), None);
    let count_bumped_nid = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![count_in_nid, one_i32_nid],
        i32_ty.clone(),
        None,
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
    let count_next_nid = graph.add_block_param(sel_merge, i32_ty.clone());
    graph.skeleton.blocks[sel_merge].term = SkeletonTerminator::Branch {
        target: continue_blk,
        args: vec![count_next_nid],
    };

    // continue: param (count_for_continue);
    // i_next = i+1; Branch(header, [count_for_continue, i_next]).
    let cont_count_nid = graph.add_block_param(continue_blk, i32_ty.clone());
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
    graph.replace_pure_node(
        spec.result_node,
        PureOp::Tuple(2),
        smallvec![buf_loaded_nid, after_count_nid],
    );
}

/// Runtime-sized `filter` lowering: a single-thread serial scatter into the
/// reserved scratch storage buffer `scratch_out`. The loop carries only a
/// surviving `count` and the input index `i` (both `u32`); kept elements are
/// stored into `scratch_out[count]` and `count` is bumped. The original result
/// node is rebound to a runtime-length view `StorageView(scratch_out)[0, count]`
/// over the buffer — its type (set by `convert_soac_filter`) already carries
/// `Buffer(scratch_out)`, so the backend recovers the descriptor from the type.
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
    next_effect: &mut crate::IdSource<EffectToken>,
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
            let pair_nid = graph.intern_pure(PureOp::Call(func.clone()), call_operands, pair_ty, None);
            let scatter_idx = graph.intern_pure(
                PureOp::Project { index: 0 },
                smallvec![pair_nid],
                index_type.clone(),
                None,
            );
            let val = graph.intern_pure(
                PureOp::Project { index: 1 },
                smallvec![pair_nid],
                value_type.clone(),
                None,
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

#[allow(dead_code)]
fn build_parallel_scatter(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    spec: ScatterLoop,
    next_effect: &mut crate::IdSource<EffectToken>,
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

    let after = graph.skeleton.split_block_before_effect(bid, idx_in_block);
    if let Some(header_meta) = control_headers.remove(&bid) {
        control_headers.insert(after, header_meta);
    }

    graph.replace_node_preserving_type(
        result_node,
        ENode::Constant(crate::ssa::types::ConstantValue::Bool(false)),
    );

    let body = graph.skeleton.create_block();
    let known = catalog().known();
    let tid_nid = graph.intern_pure(
        PureOp::Intrinsic {
            id: known.thread_id,
            overload_idx: 0,
        },
        smallvec![],
        u32_ty,
        None,
    );
    let i32_from_u32 = catalog().lookup_by_any_name("i32.u32").expect("catalog has i32.u32 bitcast");
    let i_nid = graph.intern_pure(
        PureOp::Intrinsic {
            id: i32_from_u32.id,
            overload_idx: 0,
        },
        smallvec![tid_nid],
        i32_ty.clone(),
        None,
    );
    let len_nid = emit_length(graph, len_input.0, &len_input.1, &i32_ty);
    let cond_nid = graph.intern_pure(
        PureOp::BinOp("<".into()),
        smallvec![i_nid, len_nid],
        bool_ty,
        None,
    );

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
    let pair_nid = graph.intern_pure(PureOp::Call(func), call_operands, pair_ty, None);
    let scatter_idx = graph.intern_pure(
        PureOp::Project { index: 0 },
        smallvec![pair_nid],
        index_type,
        None,
    );
    let val = graph.intern_pure(
        PureOp::Project { index: 1 },
        smallvec![pair_nid],
        value_type,
        None,
    );
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

fn filter_thread_index(graph: &mut EGraph) -> NodeId {
    graph.intern_pure(
        PureOp::Intrinsic {
            id: catalog().known().thread_id,
            overload_idx: 0,
        },
        smallvec![],
        Type::Constructed(TypeName::UInt(32), vec![]),
        None,
    )
}

fn build_filter_flags(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx: usize,
    spec: FilterLoop,
    flags: crate::BindingRef,
    next_effect: &mut crate::IdSource<EffectToken>,
) {
    use super::graph_ops::{emit_storage_store, intern_storage_view, intern_u32};
    graph.skeleton.blocks[bid].side_effects.drain(idx..);
    let after = graph.skeleton.create_block();
    let in_range = graph.skeleton.create_block();
    let keep = graph.skeleton.create_block();
    let drop = graph.skeleton.create_block();
    let pred_merge = graph.skeleton.create_block();
    graph.skeleton.blocks[after].term = SkeletonTerminator::Return(None);
    let gid = filter_thread_index(graph);
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let len = emit_length(graph, spec.arr_nid, &spec.arr_ty, &u32_ty);
    let bounded = graph.intern_pure(
        PureOp::BinOp("<".into()),
        smallvec![gid, len],
        Type::Constructed(TypeName::Bool, vec![]),
        None,
    );
    graph.skeleton.blocks[bid].term = SkeletonTerminator::CondBranch {
        cond: bounded,
        then_target: in_range,
        then_args: vec![],
        else_target: after,
        else_args: vec![],
    };
    control_headers.insert(bid, ControlHeader::Selection { merge: after });
    let elem = emit_read_element(
        graph,
        in_range,
        spec.arr_nid,
        gid,
        &spec.arr_ty,
        &spec.elem_ty,
        next_effect,
    );
    let kept = filter_kept_value(graph, elem, &spec);
    let mut operands: SmallVec<[NodeId; 4]> = smallvec![kept];
    operands.extend(spec.captures.iter().copied());
    let pred = graph.intern_pure(
        PureOp::Call(spec.pred_func.clone()),
        operands,
        Type::Constructed(TypeName::Bool, vec![]),
        None,
    );
    graph.skeleton.blocks[in_range].term = SkeletonTerminator::CondBranch {
        cond: pred,
        then_target: keep,
        then_args: vec![],
        else_target: drop,
        else_args: vec![],
    };
    control_headers.insert(in_range, ControlHeader::Selection { merge: pred_merge });
    let view = intern_storage_view(graph, flags, Type::Constructed(TypeName::UInt(32), vec![]), None);
    for (block, value) in [(keep, 1), (drop, 0)] {
        let flag = intern_u32(graph, value, None);
        emit_storage_store(
            graph,
            block,
            view,
            gid,
            flag,
            Type::Constructed(TypeName::UInt(32), vec![]),
            next_effect,
            None,
        );
        graph.skeleton.blocks[block].term = SkeletonTerminator::Branch {
            target: pred_merge,
            args: vec![],
        };
    }
    graph.skeleton.blocks[pred_merge].term = SkeletonTerminator::Branch {
        target: after,
        args: vec![],
    };
    graph.replace_node_preserving_type(
        spec.result_node,
        ENode::Constant(crate::ssa::types::ConstantValue::Bool(false)),
    );
}

fn build_filter_scan(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx: usize,
    spec: FilterLoop,
    work: FilterWorkBuffers,
    next_effect: &mut crate::IdSource<EffectToken>,
) {
    use super::graph_ops::{emit_load, emit_storage_store, intern_storage_view, intern_u32};
    graph.skeleton.blocks[bid].side_effects.drain(idx..);
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let after = graph.skeleton.create_block();
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let zero = intern_u32(graph, 0, None);
    let one = intern_u32(graph, 1, None);
    let gid = filter_thread_index(graph);
    let input_len = emit_length(graph, spec.arr_nid, &spec.arr_ty, &u32_ty);
    let nwg = graph.intern_pure(
        PureOp::Intrinsic {
            id: catalog().known().num_workgroups,
            overload_idx: 0,
        },
        smallvec![],
        u32_ty.clone(),
        None,
    );
    let wg_width = graph.intern_pure(
        PureOp::Uint(super::parallelize::REDUCE_PHASE1_WIDTH.to_string()),
        smallvec![],
        u32_ty.clone(),
        None,
    );
    let total_threads = graph.intern_pure(
        PureOp::BinOp("*".into()),
        smallvec![nwg, wg_width],
        u32_ty.clone(),
        None,
    );
    let total_minus_one = graph.intern_pure(
        PureOp::BinOp("-".into()),
        smallvec![total_threads, one],
        u32_ty.clone(),
        None,
    );
    let len_plus = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![input_len, total_minus_one],
        u32_ty.clone(),
        None,
    );
    let chunk_size = graph.intern_pure(
        PureOp::BinOp("/".into()),
        smallvec![len_plus, total_threads],
        u32_ty.clone(),
        None,
    );
    let raw_chunk_start = graph.intern_pure(
        PureOp::BinOp("*".into()),
        smallvec![gid, chunk_size],
        u32_ty.clone(),
        None,
    );
    let u32_min = catalog().lookup_by_any_name("u32.min").expect("catalog has u32.min");
    let chunk_start = graph.intern_pure(
        PureOp::Intrinsic {
            id: u32_min.id,
            overload_idx: 0,
        },
        smallvec![raw_chunk_start, input_len],
        u32_ty.clone(),
        None,
    );
    let remaining = graph.intern_pure(
        PureOp::BinOp("-".into()),
        smallvec![input_len, chunk_start],
        u32_ty.clone(),
        None,
    );
    let chunk_len = graph.intern_pure(
        PureOp::Intrinsic {
            id: u32_min.id,
            overload_idx: 0,
        },
        smallvec![chunk_size, remaining],
        u32_ty.clone(),
        None,
    );
    graph.skeleton.blocks[bid].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![zero, zero],
    };
    let i = graph.add_block_param(header, u32_ty.clone());
    let acc = graph.add_block_param(header, u32_ty.clone());
    let cond = graph.intern_pure(
        PureOp::BinOp("<".into()),
        smallvec![i, chunk_len],
        Type::Constructed(TypeName::Bool, vec![]),
        None,
    );
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args: vec![acc],
    };
    control_headers.insert(
        header,
        ControlHeader::Loop {
            merge: after,
            continue_block: body,
        },
    );
    let flags = intern_storage_view(graph, work.flags, u32_ty.clone(), None);
    let offsets = intern_storage_view(graph, work.offsets, u32_ty.clone(), None);
    let global_i = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![chunk_start, i],
        u32_ty.clone(),
        None,
    );
    let flag_place = graph.intern_pure(
        PureOp::ViewIndex,
        smallvec![flags, global_i],
        u32_ty.clone(),
        None,
    );
    let flag = emit_load(graph, body, flag_place, u32_ty.clone(), next_effect, None);
    let next = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![acc, flag],
        u32_ty.clone(),
        None,
    );
    emit_storage_store(
        graph,
        body,
        offsets,
        global_i,
        next,
        u32_ty.clone(),
        next_effect,
        None,
    );
    let next_i = graph.intern_pure(PureOp::BinOp("+".into()), smallvec![i, one], u32_ty.clone(), None);
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![next_i, next],
    };
    let final_count = graph.add_block_param(after, u32_ty.clone());
    let block_sums = intern_storage_view(graph, work.block_sums, u32_ty.clone(), None);
    emit_storage_store(
        graph,
        after,
        block_sums,
        gid,
        final_count,
        u32_ty.clone(),
        next_effect,
        None,
    );
    graph.skeleton.blocks[after].term = SkeletonTerminator::Return(None);
    graph.replace_node_preserving_type(
        spec.result_node,
        ENode::Constant(crate::ssa::types::ConstantValue::Bool(false)),
    );
}

fn build_filter_scatter(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx: usize,
    spec: FilterLoop,
    work: FilterWorkBuffers,
    next_effect: &mut crate::IdSource<EffectToken>,
) {
    use super::graph_ops::{emit_load, emit_storage_store, intern_storage_view, intern_u32};
    let after = graph.skeleton.split_block_before_effect(bid, idx);
    let in_range = graph.skeleton.create_block();
    let write = graph.skeleton.create_block();
    let skip = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let gid = filter_thread_index(graph);
    let len = emit_length(graph, spec.arr_nid, &spec.arr_ty, &u32_ty);
    let bounded = graph.intern_pure(
        PureOp::BinOp("<".into()),
        smallvec![gid, len],
        bool_ty.clone(),
        None,
    );
    graph.skeleton.blocks[bid].term = SkeletonTerminator::CondBranch {
        cond: bounded,
        then_target: in_range,
        then_args: vec![],
        else_target: after,
        else_args: vec![],
    };
    control_headers.insert(bid, ControlHeader::Selection { merge: after });
    let flags = intern_storage_view(graph, work.flags, u32_ty.clone(), None);
    let offsets = intern_storage_view(graph, work.offsets, u32_ty.clone(), None);
    let flag_place = graph.intern_pure(PureOp::ViewIndex, smallvec![flags, gid], u32_ty.clone(), None);
    let flag = emit_load(graph, in_range, flag_place, u32_ty.clone(), next_effect, None);
    let one = intern_u32(graph, 1, None);
    let keep = graph.intern_pure(PureOp::BinOp("==".into()), smallvec![flag, one], bool_ty, None);
    graph.skeleton.blocks[in_range].term = SkeletonTerminator::CondBranch {
        cond: keep,
        then_target: write,
        then_args: vec![],
        else_target: skip,
        else_args: vec![],
    };
    control_headers.insert(in_range, ControlHeader::Selection { merge });
    let offset_place = graph.intern_pure(PureOp::ViewIndex, smallvec![offsets, gid], u32_ty.clone(), None);
    let inclusive = emit_load(graph, write, offset_place, u32_ty.clone(), next_effect, None);
    let output_index = graph.intern_pure(
        PureOp::BinOp("-".into()),
        smallvec![inclusive, one],
        u32_ty.clone(),
        None,
    );
    let elem = emit_read_element(
        graph,
        write,
        spec.arr_nid,
        gid,
        &spec.arr_ty,
        &spec.elem_ty,
        next_effect,
    );
    let kept = filter_kept_value(graph, elem, &spec);
    let (out_binding, len_binding) = match &spec.output {
        filter::Output::Runtime {
            scratch,
            length: filter::RuntimeLength::Stored(length),
        } => (scratch, length),
        _ => panic!("parallel filter scatter requires runtime entry output"),
    };
    let out_binding = *out_binding;
    let len_binding = *len_binding;
    let output = intern_storage_view(graph, out_binding, spec.output_elem_ty.clone(), None);
    emit_storage_store(
        graph,
        write,
        output,
        output_index,
        kept,
        spec.output_elem_ty.clone(),
        next_effect,
        None,
    );
    graph.skeleton.blocks[write].term = SkeletonTerminator::Branch {
        target: merge,
        args: vec![],
    };
    graph.skeleton.blocks[skip].term = SkeletonTerminator::Branch {
        target: merge,
        args: vec![],
    };
    graph.skeleton.blocks[merge].term = SkeletonTerminator::Branch {
        target: after,
        args: vec![],
    };
    let len_view = intern_storage_view(graph, len_binding, u32_ty.clone(), None);
    let zero = intern_u32(graph, 0, None);
    let len_place = graph.intern_pure(PureOp::ViewIndex, smallvec![len_view, zero], u32_ty.clone(), None);
    let count = emit_load(graph, bid, len_place, u32_ty.clone(), next_effect, None);
    graph.replace_pure_node(
        spec.result_node,
        PureOp::StorageView(crate::op::PureViewSource::Storage(out_binding)),
        smallvec![zero, count],
    );
}

fn build_runtime_filter_loop(
    graph: &mut EGraph,
    control_headers: &mut LookupMap<BlockId, ControlHeader>,
    bid: BlockId,
    idx_in_block: usize,
    spec: FilterLoop,
    scratch_out: crate::BindingRef,
    next_effect: &mut crate::IdSource<EffectToken>,
) {
    use super::graph_ops::{emit_storage_store, intern_storage_view, intern_u32};
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);

    // A view over the whole scratch buffer, for the per-iteration store. The
    // scratch holds the kept (output) elements.
    let scratch_view = intern_storage_view(graph, scratch_out, spec.output_elem_ty.clone(), None);

    // Split `bid` into preheader (bid) + after, moving the suffix + terminator.
    let after = graph.skeleton.split_block_before_effect(bid, idx_in_block);
    if let Some(header_meta) = control_headers.remove(&bid) {
        control_headers.insert(after, header_meta);
    }

    // After-block param: the final surviving count.
    let after_count_nid = graph.add_block_param(after, u32_ty.clone());

    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let then_blk = graph.skeleton.create_block();
    let else_blk = graph.skeleton.create_block();
    let sel_merge = graph.skeleton.create_block();
    let continue_blk = graph.skeleton.create_block();

    // Header params: count_in, i_in.
    let count_in_nid = graph.add_block_param(header, u32_ty.clone());
    let i_in_nid = graph.add_block_param(header, u32_ty.clone());

    // Preheader → header(0, 0).
    let zero_count_nid = intern_u32(graph, 0, None);
    let zero_i_nid = intern_u32(graph, 0, None);
    graph.skeleton.blocks[bid].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![zero_count_nid, zero_i_nid],
    };

    // Header → cond_br(i < len, body, after(count)).
    let len_nid = emit_length(graph, spec.arr_nid, &spec.arr_ty, &u32_ty);
    let cond_nid = graph.intern_pure(
        PureOp::BinOp("<".into()),
        smallvec![i_in_nid, len_nid],
        bool_ty.clone(),
        None,
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
    let pred_nid = graph.intern_pure(PureOp::Call(spec.pred_func), pred_operands, bool_ty.clone(), None);
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
        None,
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
    let count_next_nid = graph.add_block_param(sel_merge, u32_ty.clone());
    graph.skeleton.blocks[sel_merge].term = SkeletonTerminator::Branch {
        target: continue_blk,
        args: vec![count_next_nid],
    };

    // continue: param cont_count; i_next = i + 1; Branch(header, [cont_count, i_next]).
    let cont_count_nid = graph.add_block_param(continue_blk, u32_ty.clone());
    let one_i_nid = intern_u32(graph, 1, None);
    let next_i_nid = graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![i_in_nid, one_i_nid],
        u32_ty.clone(),
        None,
    );
    graph.skeleton.blocks[continue_blk].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![cont_count_nid, next_i_nid],
    };

    // When the filter is a compute-entry output, store the final surviving
    // count into the paired length cell `len_out[0]` so the host can read how
    // many elements are valid in the (capacity-n) output buffer.
    if let filter::Output::Runtime {
        length: filter::RuntimeLength::Stored(len_br),
        ..
    } = &spec.output
    {
        let len_view = intern_storage_view(graph, *len_br, u32_ty.clone(), None);
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
    // type (carrying `Buffer(scratch_out)`) is preserved from emit_soac.
    let zero_off_nid = intern_u32(graph, 0, None);
    graph.replace_pure_node(
        spec.result_node,
        PureOp::StorageView(crate::op::PureViewSource::Storage(scratch_out)),
        smallvec![zero_off_nid, after_count_nid],
    );
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
    let after = graph.skeleton.split_block_before_effect(bid, idx_in_block);

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
            for carried_idx in indices {
                let Some((part_ty, _)) = spec.carried.get(*carried_idx) else {
                    continue;
                };
                let part_nid = graph.add_block_param(after, part_ty.clone());
                operands.push(part_nid);
            }
            graph.replace_pure_node(*result_node, PureOp::Tuple(operands.len()), operands);
            graph.retype_node(*result_node, tuple_ty.clone());
        }
        ResultBinding::DummyBool { result_node } => {
            graph.replace_node_preserving_type(
                *result_node,
                ENode::Constant(crate::ssa::types::ConstantValue::Bool(false)),
            );
        }
    }

    // Build header with one block-param per carried plus the index.
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let mut carried_nids = Vec::with_capacity(spec.carried.len());
    for (ty, _) in &spec.carried {
        let nid = graph.add_block_param(header, ty.clone());
        carried_nids.push(nid);
    }
    let idx_nid = graph.add_block_param(header, i32_ty.clone());

    // Preheader terminator: br header(init_carried..., 0).
    let zero_nid = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone(), None);
    let mut preheader_args: Vec<NodeId> = spec.carried.iter().map(|(_, init)| *init).collect();
    preheader_args.push(zero_nid);
    graph.skeleton.blocks[bid].term = SkeletonTerminator::Branch {
        target: header,
        args: preheader_args,
    };

    // Header terminator: condbr i<len -> body / after(result_carried).
    let len_nid = emit_length(graph, spec.len_input.0, &spec.len_input.1, &i32_ty);
    let cond_nid = graph.intern_pure(
        PureOp::BinOp("<".into()),
        smallvec![idx_nid, len_nid],
        bool_ty,
        None,
    );
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
    let one_nid = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone(), None);
    graph.intern_pure(
        PureOp::BinOp("+".into()),
        smallvec![idx_nid, one_nid],
        i32_ty,
        None,
    )
}

/// Emit the length of an input array in the requested integer type.
/// Composite, view, and virtual arrays share `_w_intrinsic_length`. For a SoA
/// tuple, the length is the length of component 0 (all components share it
/// post-`tlc::soa`).
fn emit_length(
    graph: &mut EGraph,
    arr_nid: NodeId,
    arr_ty: &Type<TypeName>,
    result_ty: &Type<TypeName>,
) -> NodeId {
    let actual_arr_ty = graph.types.get(&arr_nid).filter(|ty| is_plain_array_source(ty)).cloned();
    let arr_ty = actual_arr_ty.as_ref().unwrap_or(arr_ty);
    if let Some(components) = as_soa_tuple(arr_ty) {
        let first_arr = graph.intern_pure(
            PureOp::Project { index: 0 },
            smallvec![arr_nid],
            components[0].clone(),
            None,
        );
        return emit_length(graph, first_arr, &components[0], result_ty);
    }
    let length_id = catalog().known().length;
    graph.intern_pure(
        PureOp::Intrinsic {
            id: length_id,
            overload_idx: 0,
        },
        smallvec![arr_nid],
        result_ty.clone(),
        None,
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
    next_effect: &mut crate::IdSource<EffectToken>,
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
                    soac_element_type(ct)
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
                None,
            );
            let e = emit_read_element(graph, body, comp_arr, idx_nid, comp_ty, comp_elem_ty, next_effect);
            elem_nids.push(e);
        }
        return graph.intern_pure(PureOp::Tuple(components.len()), elem_nids, elem_ty.clone(), None);
    }
    if is_view_source(arr_ty) {
        // View array: ViewIndex (pure, PlaceId) + Load (effectful).
        let ptr_nid = graph.intern_pure(
            PureOp::ViewIndex,
            smallvec![arr_nid, idx_nid],
            elem_ty.clone(),
            None,
        );
        let load_result = graph.alloc_side_effect_result(elem_ty.clone());
        let eff_in = alloc_effect(next_effect);
        let eff_out = alloc_effect(next_effect);
        graph.skeleton.blocks[body].side_effects.push(SideEffect {
            kind: SideEffectKind::Effect(EffectOp::Load),
            operand_nodes: smallvec![ptr_nid],
            result: Some(load_result),
            effects: Some((eff_in, eff_out)),
            span: None,
        });
        load_result
    } else if is_virtual_source(arr_ty) {
        // Virtual {start, step, len}: elem = start + i * step.
        let start_nid = graph.intern_pure(
            PureOp::Project { index: 0 },
            smallvec![arr_nid],
            elem_ty.clone(),
            None,
        );
        let step_nid = graph.intern_pure(
            PureOp::Project { index: 1 },
            smallvec![arr_nid],
            elem_ty.clone(),
            None,
        );
        let mul_nid = graph.intern_pure(
            PureOp::BinOp("*".into()),
            smallvec![idx_nid, step_nid],
            elem_ty.clone(),
            None,
        );
        graph.intern_pure(
            PureOp::BinOp("+".into()),
            smallvec![start_nid, mul_nid],
            elem_ty.clone(),
            None,
        )
    } else {
        graph.intern_pure(PureOp::Index, smallvec![arr_nid, idx_nid], elem_ty.clone(), None)
    }
}

/// Emit a per-iteration write `arr[idx] = val`, producing the new array node.
///
/// `elem_ty` must be the logical element type of `arr_ty`:
/// - Plain composite array: `arr_ty.elem_type()`.
/// - SoA tuple: `soac_element_type(arr_ty)` (a tuple whose components line
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
                None,
            );
            let comp_val = graph.intern_pure(
                PureOp::Project { index: i as u32 },
                smallvec![val_nid],
                comp_elem_ty.clone(),
                None,
            );
            let new_comp =
                emit_write_element(graph, comp_arr, idx_nid, comp_val, comp_arr_ty, comp_elem_ty);
            new_component_arrs.push(new_comp);
        }
        return graph.intern_pure(
            PureOp::Tuple(components.len()),
            new_component_arrs,
            arr_ty.clone(),
            None,
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
        None,
    )
}

/// The logical element type implied by `arr_ty`: `arr_ty.elem_type()` for
/// composite arrays, `soac_element_type(arr_ty)` for SoA tuples. Only used
/// by `emit_write_element`'s debug_assert.
fn derive_elem_ty(arr_ty: &Type<TypeName>) -> Type<TypeName> {
    if as_soa_tuple(arr_ty).is_some() {
        soac_element_type(arr_ty)
    } else {
        arr_ty.elem_type().expect("composite array has elem").clone()
    }
}

#[cfg(test)]
#[path = "soac_expand_tests.rs"]
mod soac_expand_tests;
