//! Expand physical `SideEffectKind::Soac(SoacEffect(_, ...))` skeleton side-effects
//! into explicit loop subgraphs with pure ops in the sea and block params
//! carrying accumulators.
//!
//! Consumes target-planned physical EGIR before graph cleanup and SSA
//! elaboration. Every physical variant must be handled here; any SOAC left in
//! the skeleton after this stage is a bug.

/// Physical EGIR whose SOAC effects have been expanded into explicit CFGs.
#[derive(Debug, Clone, Copy)]
pub enum SoacsExpandedTag {}
pub type SoacsExpanded = super::program::Program<
    SoacsExpandedTag,
    super::ir::ProgramFamily<
        super::types::Physical,
        crate::interface::StorageBindingDecl,
        super::ir::RealizedOutputRoute,
        super::program::CoreProgramData,
    >,
    super::program::PlannedGlobal,
>;

use crate::builtins::catalog;
use crate::flow::{BlockId, ControlHeader};

use polytype::Type;
use smallvec::{smallvec, SmallVec};

use super::graph_ops::{
    alloc_effect, emit_alloca, emit_load, emit_place_index_store, emit_storage_store, emit_store,
};
use super::program::{
    PhysicalEGraph as EGraph, PhysicalFilterOutput, PhysicalFilterWorkBuffers as FilterWorkBuffers,
    PhysicalSegSpace as SegSpace, PhysicalSideEffect as SideEffect,
    PhysicalSideEffectKind as SideEffectKind, PhysicalSoac as Soac, ProgramIdentities,
};
use super::soac::{filter, hist, screma};
use crate::ast::TypeName;
use crate::types::{is_array_variant_view, is_virtual_array, TypeExt};

use super::types::{
    as_soa_tuple, soac_element_type, ENode, EffectOp, EffectToken, NodeId, PureOp, RegionId,
    SkeletonTerminator, SoacDestination, SoacEffect,
};

mod array_io;
mod filter_lowering;
mod hist_lowering;
mod loop_builder;
mod screma_lowering;

use array_io::{emit_read_element, emit_write_element};
use filter_lowering::{
    build_filter_flags, build_filter_loop, build_filter_scan, build_filter_scatter, FilterLoop,
};
use hist_lowering::{
    build_bucket_finish, build_bucket_init, build_bucket_insert, build_hist_atomic, build_hist_loop,
    HistLoop,
};
use loop_builder::{expand_loop, LoopBody, ResultBinding};
use screma_lowering::{build_parallel_screma_map, emit_screma_lambda};

/// Expand every graph-bearing body and rebuild the program at the
/// post-expansion checkpoint.
pub fn expand_soacs(program: super::parallelize::Planned) -> Result<SoacsExpanded, String> {
    program
        .try_map_graphs_with_state(|_, graph, data, context| {
            run_one_body(graph, &data.identities, &mut context.effect_ids)
        })
        .map(|program| program.retag())
}

/// Expand every physical SOAC in the skeleton.
pub fn run_one_body(
    mut graph: EGraph,
    regions: &ProgramIdentities,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<EGraph, String> {
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
        expand_one(&mut graph, bid, idx, effect_ids, regions)?;
    }
    if let Some((block, effect)) = graph.skeleton.blocks.iter().find_map(|(block, contents)| {
        contents
            .side_effects
            .iter()
            .find(|effect| matches!(effect.kind, SideEffectKind::Soac(_)))
            .map(|effect| (block, effect))
    }) {
        return Err(format!(
            "SOAC expansion left an unsupported physical operation in {block:?}: {:?}",
            effect.kind
        ));
    }
    Ok(graph)
}

/// Does this SOAC kind have a TLC→EGIR expansion implemented here?
fn is_handleable_soac(kind: &SideEffectKind) -> bool {
    let SideEffectKind::Soac(SoacEffect(_, soac)) = kind else {
        return false;
    };
    match soac {
        Soac::Screma(op) if op.is_serial() => {
            op.inputs.iter().all(|input| is_plain_array_source(&input.array))
        }
        Soac::Filter(op) => {
            !op.body.inputs.is_empty()
                && op.body.inputs.iter().all(|input| is_plain_array_source(&input.array))
        }
        // Hist reads all input arrays per element; loop length comes from the
        // first input, but every input must support the read path.
        Soac::Hist(op) => {
            !op.inputs.is_empty() && op.inputs.iter().all(|input| is_plain_array_source(&input.array))
        }
        Soac::Screma(op) if matches!(op.state, screma::PhysicalState::Segmented(_)) && op.is_map() => {
            op.form.post.is_identity() && op.inputs.iter().all(|input| is_plain_array_source(&input.array))
        }
        // Scan and reduction recipes must lower the complete canonical
        // contract before this pass.
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
    bid: BlockId,
    idx: usize,
    next_effect: &mut crate::IdSource<EffectToken>,
    regions: &ProgramIdentities,
) -> Result<(), String> {
    let se = graph.skeleton.blocks[bid].side_effects.remove(idx);
    match &se.kind {
        SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) if op.is_serial() => {
            let operands = screma::ScremaOperands::decode(op, &se.operand_nodes, se.result)?;
            let input_nids = operands.inputs().map(|operand| operand.node).collect::<Vec<_>>();
            let Some(first_input) = op.inputs.first() else {
                return Err("serial Screma has no array input".into());
            };
            let result_nid = operands.result();
            let Type::Constructed(TypeName::Tuple(_), result_fields) = graph.nodes[result_nid].ty.clone()
            else {
                return Err("Screma result must be represented as a tuple".into());
            };
            if result_fields.len() != op.result_count() {
                return Err(format!(
                    "Screma result tuple has {} fields, expected {}",
                    result_fields.len(),
                    op.result_count()
                ));
            }

            let read_inputs = input_nids
                .iter()
                .zip(&op.inputs)
                .map(|(&node, input)| (node, input.array.clone(), input.element()))
                .collect::<Vec<_>>();
            let len_input = (input_nids[0], first_input.array.clone());
            let reduction_components = op.form.reduction_result_count();
            let scan_components = op.form.scan_input_count();
            let post_count = op.form.post.result_types.len();
            let uninit_id = catalog().known().uninit;

            // Carried order is post-result arrays, scan accumulators, then
            // reduction accumulators.  The external tuple is independently
            // rebound in Futhark order: reductions first, post arrays second.
            let mut carried = Vec::new();
            let mut post_carried_types = Vec::with_capacity(post_count);
            let mut post_is_output_view = Vec::with_capacity(post_count);
            for post in 0..post_count {
                let field = reduction_components + post;
                let destination = op
                    .destination(field)
                    .ok_or_else(|| format!("Screma result {field} has no destination"))?;
                let initial = match destination {
                    SoacDestination::UniqueInput => {
                        return Err("unresolved UniqueInput destination reached physical expansion".into())
                    }
                    SoacDestination::Fresh => graph.intern_pure(
                        PureOp::Intrinsic {
                            id: uninit_id,
                            overload_idx: 0,
                        },
                        smallvec![],
                        result_fields[field].clone(),
                        None,
                    ),
                    SoacDestination::OutputView => operands
                        .output(field)
                        .map(|operand| operand.node)
                        .ok_or_else(|| format!("Screma post result {post} has no output-view operand"))?,
                    SoacDestination::InputBuffer => {
                        if input_nids.len() != 1 {
                            return Err(
                                "input-buffer Screma result requires exactly one array input".into()
                            );
                        }
                        input_nids[0]
                    }
                };
                post_is_output_view.push(destination.is_output_view());
                let carried_type = graph.nodes[initial].ty.clone();
                post_carried_types.push(carried_type.clone());
                carried.push((carried_type, initial));
            }

            for scan in &op.form.scans {
                for neutral in &scan.neutral {
                    carried.push((graph.nodes[*neutral].ty.clone(), *neutral));
                }
            }
            for reduction in &op.form.reductions {
                for neutral in &reduction.neutral {
                    carried.push((graph.nodes[*neutral].ty.clone(), *neutral));
                }
            }

            let mut result_indices = (0..reduction_components)
                .map(|component| post_count + scan_components + component)
                .collect::<Vec<_>>();
            result_indices.extend(0..post_count);
            let mut result_field_types = op
                .form
                .reductions
                .iter()
                .flat_map(|reduction| reduction.operator.result_types.iter().cloned())
                .collect::<Vec<_>>();
            result_field_types.extend(post_carried_types.iter().cloned());
            let result_tuple_type =
                Type::Constructed(TypeName::Tuple(op.result_count()), result_field_types);
            graph.retype_node(result_nid, result_tuple_type.clone());
            let result = ResultBinding::TupleFromCarried {
                result_node: result_nid,
                tuple_ty: result_tuple_type,
                indices: result_indices,
            };

            expand_loop(
                graph,
                bid,
                idx,
                &len_input,
                &carried,
                &result,
                next_effect,
                false,
                |graph, next_effect, body, lane, carried_values| {
                    let input_elements = read_inputs
                        .iter()
                        .map(|(array, array_type, element_type)| {
                            emit_read_element(
                                graph,
                                body,
                                *array,
                                lane,
                                array_type,
                                element_type,
                                next_effect,
                            )
                        })
                        .collect::<Vec<_>>();
                    let pre_values = emit_screma_lambda(graph, regions, &op.form.pre, input_elements);

                    let mut pre_offset = 0;
                    let mut scan_offset = post_count;
                    let mut new_scans = Vec::with_capacity(scan_components);
                    for scan in &op.form.scans {
                        let width = scan.neutral.len();
                        let mut arguments = carried_values[scan_offset..scan_offset + width].to_vec();
                        arguments.extend_from_slice(&pre_values[pre_offset..pre_offset + width]);
                        new_scans.extend(emit_screma_lambda(graph, regions, &scan.operator, arguments));
                        pre_offset += width;
                        scan_offset += width;
                    }

                    let mut reduction_offset = post_count + scan_components;
                    let mut new_reductions = Vec::with_capacity(reduction_components);
                    for reduction in &op.form.reductions {
                        let width = reduction.neutral.len();
                        let mut arguments =
                            carried_values[reduction_offset..reduction_offset + width].to_vec();
                        arguments.extend_from_slice(&pre_values[pre_offset..pre_offset + width]);
                        new_reductions.extend(emit_screma_lambda(
                            graph,
                            regions,
                            &reduction.operator,
                            arguments,
                        ));
                        pre_offset += width;
                        reduction_offset += width;
                    }

                    let mut post_arguments = new_scans.clone();
                    post_arguments.extend_from_slice(&pre_values[pre_offset..]);
                    let post_values = emit_screma_lambda(graph, regions, &op.form.post, post_arguments);
                    debug_assert_eq!(post_values.len(), post_count);

                    let mut next = Vec::with_capacity(carried_values.len());
                    for post in 0..post_count {
                        let output = carried_values[post];
                        let value = post_values[post];
                        if post_is_output_view[post] {
                            let place = graph.intern_pure(
                                PureOp::ViewIndex,
                                smallvec![output, lane],
                                op.form.post.result_types[post].clone(),
                                None,
                            );
                            let effect_in = alloc_effect(next_effect);
                            let effect_out = alloc_effect(next_effect);
                            graph.skeleton.blocks[body].side_effects.push(SideEffect {
                                kind: SideEffectKind::Effect(EffectOp::Store),
                                operand_nodes: smallvec![place, value],
                                result: None,
                                effects: Some((effect_in, effect_out)),
                                span: None,
                            });
                            next.push(output);
                        } else {
                            next.push(emit_write_element(
                                graph,
                                output,
                                lane,
                                value,
                                &post_carried_types[post],
                                &op.form.post.result_types[post],
                            ));
                        }
                    }
                    next.extend(new_scans);
                    next.extend(new_reductions);
                    LoopBody {
                        tail: body,
                        carried: next,
                    }
                },
            );
        }
        SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) => {
            let input_count = op.body.inputs.len();
            let read_inputs = se.operand_nodes[..input_count]
                .iter()
                .copied()
                .zip(&op.body.inputs)
                .map(|(node, input)| (node, input.array.clone(), input.element()))
                .collect::<Vec<_>>();
            let map_body = op.body.map.seg_body();
            let map_func = map_body.map(|body| body.region);
            let output_elem_ty = op.body.output_element_type();
            let predicate_body =
                op.body.predicate.seg_body().expect("validated Filter predicate has a region");
            let pred_func = predicate_body.region;
            let (output, plan) = match &op.state {
                filter::ScheduledState::Loop { storage, .. } => (storage.clone(), filter::Plan::Loop),
                filter::ScheduledState::Pipeline { storage, plan, .. } => {
                    let output = filter::Output::Runtime {
                        scratch: storage.scratch,
                        length: storage.length,
                    };
                    let config = filter::ParallelConfig {
                        buffers: plan.buffers,
                        scan_workgroup_width: plan.scan_workgroup_width,
                    };
                    let plan = match plan.stage {
                        filter::ParallelStage::Flags => filter::Plan::Flags(config),
                        filter::ParallelStage::Scan => filter::Plan::Scan(config),
                        filter::ParallelStage::Scatter => filter::Plan::Scatter(config),
                    };
                    (output, plan)
                }
            };

            let map_captures = map_body.map(|body| body.captures.clone()).unwrap_or_default();
            let captures = predicate_body.captures.clone();
            let result_nid = se.result.expect("Filter has a result");

            let spec = FilterLoop {
                read_inputs,
                output_elem_ty,
                output,
                map_func,
                map_captures,
                pred_func,
                captures,
                result_node: result_nid,
            };
            match plan {
                filter::Plan::Flags(config) => {
                    build_filter_flags(graph, bid, idx, spec, config.buffers.flags, next_effect)
                }
                filter::Plan::Scan(config) => build_filter_scan(
                    graph,
                    bid,
                    idx,
                    spec,
                    config.buffers,
                    config.scan_workgroup_width,
                    next_effect,
                ),
                filter::Plan::Scatter(config) => {
                    build_filter_scatter(graph, bid, idx, spec, config.buffers, next_effect)
                }
                filter::Plan::Loop => build_filter_loop(graph, bid, idx, spec, next_effect),
            }
        }
        SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) => {
            let n_inputs = op.inputs.len();
            let input_nids = &se.operand_nodes[..n_inputs];
            let read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>, Vec<u8>)> = input_nids
                .iter()
                .zip(op.inputs.iter())
                .map(|(nid, input)| {
                    (
                        *nid,
                        input.array.clone(),
                        input.element(),
                        input.dimensions.clone(),
                    )
                })
                .collect();
            let len_input = (input_nids[0], op.inputs[0].array.clone());
            let result_nid = se.result.expect("Hist has a result");

            let hist = HistLoop {
                form: op.form.clone(),
                read_inputs,
                len_input,
                result_node: result_nid,
            };
            match &op.state {
                hist::PhysicalState::Atomic { space, operations } => {
                    build_hist_atomic(graph, bid, idx, hist, space, operations, next_effect, regions)
                }
                hist::PhysicalState::Bucket {
                    space,
                    stage,
                    topology,
                } => match stage {
                    hist::ParallelStage::Init => build_bucket_init(graph, bid, idx, hist, next_effect),
                    hist::ParallelStage::Insert => build_bucket_insert(
                        graph,
                        bid,
                        idx,
                        hist,
                        space,
                        topology.as_ref(),
                        next_effect,
                        regions,
                    ),
                    hist::ParallelStage::Finish => build_bucket_finish(graph, bid, idx, hist.result_node),
                },
                hist::PhysicalState::Serial => build_hist_loop(graph, bid, idx, hist, next_effect, regions),
            }
        }
        SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op)))
            if matches!(op.state, screma::PhysicalState::Segmented(_)) && op.is_map() =>
        {
            let screma::PhysicalState::Segmented(segment) = &op.state else {
                unreachable!()
            };
            let operands = screma::ScremaOperands::decode(op, &se.operand_nodes, se.result)?;
            let input_nodes = operands.inputs().map(|operand| operand.node).collect::<Vec<_>>();
            let read_inputs = input_nodes
                .iter()
                .zip(&op.inputs)
                .map(|(&node, input)| (node, input.array.clone(), input.element()))
                .collect::<Vec<_>>();
            let Some(first_input) = read_inputs.first() else {
                return Err("segmented map has no array input".into());
            };
            let mut output_views = Vec::with_capacity(op.result_count());
            for field in 0..op.result_count() {
                let destination = op
                    .destination(field)
                    .ok_or_else(|| format!("segmented map result {field} has no destination"))?;
                let output = if destination.is_output_view() {
                    operands
                        .output(field)
                        .map(|operand| operand.node)
                        .ok_or_else(|| format!("segmented map result {field} has no output view"))?
                } else if destination.is_input_buffer() && input_nodes.len() == 1 {
                    input_nodes[0]
                } else {
                    return Err(format!(
                        "segmented map result {field} has unsupported destination {destination:?}"
                    ));
                };
                output_views.push(output);
            }
            build_parallel_screma_map(
                graph,
                bid,
                idx,
                &segment.space,
                (first_input.0, first_input.1.clone()),
                &read_inputs,
                &op.form.pre,
                &output_views,
                operands.result(),
                next_effect,
                regions,
            );
        }
        _ => return Err("SOAC expansion target changed after selection".into()),
    }
    Ok(())
}

#[cfg(test)]
#[path = "soac_expand_tests.rs"]
mod soac_expand_tests;
