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
            let read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)> = input_nids
                .iter()
                .zip(op.inputs.iter())
                .map(|(nid, input)| (*nid, input.array.clone(), input.element()))
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

fn emit_screma_lambda(
    graph: &mut EGraph,
    _regions: &ProgramIdentities,
    lambda: &screma::Lambda,
    mut arguments: Vec<NodeId>,
) -> Vec<NodeId> {
    if lambda.is_identity() {
        debug_assert_eq!(arguments.len(), lambda.result_types.len());
        return arguments;
    }
    let body = lambda.seg_body().expect("non-identity Screma lambda has a region");
    debug_assert!(_regions.contains_function(body.region));
    arguments.extend(body.captures.iter().copied());
    let result_type = match lambda.result_types.as_slice() {
        [result] => result.clone(),
        results => Type::Constructed(TypeName::Tuple(results.len()), results.to_vec()),
    };
    let result = graph.intern_pure(
        PureOp::Call(body.region),
        arguments.into_iter().collect(),
        result_type,
        None,
    );
    match lambda.result_types.as_slice() {
        [_] => vec![result],
        results => results
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                graph.intern_pure(
                    PureOp::Project { index: index as u32 },
                    smallvec![result],
                    ty.clone(),
                    None,
                )
            })
            .collect(),
    }
}
/// One expanded-loop iteration. A body can finish in a different CFG block
/// when its effectful work is conditionally executed.
struct LoopBody {
    tail: BlockId,
    carried: Vec<NodeId>,
}

/// Emit a real loop via `build_loop_skeleton`, invoking `emit_body` in the
/// body block to produce the new carried values, then wire the back-edge.
fn build_loop<F>(
    graph: &mut EGraph,
    bid: BlockId,
    idx_in_block: usize,
    len_input: &(NodeId, Type<TypeName>),
    carried: &[(Type<TypeName>, NodeId)],
    result: &ResultBinding,
    next_effect: &mut crate::IdSource<EffectToken>,
    mut emit_body: F,
) where
    F: FnMut(&mut EGraph, &mut crate::IdSource<EffectToken>, BlockId, NodeId, &[NodeId]) -> LoopBody,
{
    let handles = build_loop_skeleton(
        graph,
        bid,
        idx_in_block,
        LoopSkeletonSpec {
            carried: carried.to_vec(),
            result: result.clone(),
            len_input: len_input.clone(),
        },
    );
    let body = emit_body(
        graph,
        next_effect,
        handles.body,
        handles.idx_nid,
        &handles.carried,
    );
    debug_assert_eq!(body.carried.len(), carried.len());
    let next_i_nid = increment(graph, handles.idx_nid);
    let mut args = body.carried;
    args.push(next_i_nid);
    graph.skeleton.blocks[body.tail].term = SkeletonTerminator::Branch {
        target: handles.header,
        args,
    };
}

/// Try to unroll a small loop; if the trip count isn't statically small (or
/// `allow_unroll` is false), fall back to a real loop. Both paths share the
/// same `emit_body` closure — write iteration logic once.
fn expand_loop<F>(
    graph: &mut EGraph,
    bid: BlockId,
    idx_in_block: usize,
    len_input: &(NodeId, Type<TypeName>),
    carried: &[(Type<TypeName>, NodeId)],
    result: &ResultBinding,
    next_effect: &mut crate::IdSource<EffectToken>,
    allow_unroll: bool,
    mut emit_body: F,
) where
    F: FnMut(&mut EGraph, &mut crate::IdSource<EffectToken>, BlockId, NodeId, &[NodeId]) -> LoopBody,
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
        bid,
        idx_in_block,
        len_input,
        carried,
        result,
        next_effect,
        emit_body,
    );
}

/// Generic small-loop unroller. Returns `true` if the loop was unrolled into
/// a short CFG chain rooted at `bid`; `false` if the trip count isn't
/// statically known to be small, and the caller should fall back to emitting a
/// real loop via `build_loop_skeleton`.
///
/// `emit_body(graph, next_effect, block, idx_const_nid, carried_in)` produces
/// the `carried_out` NodeIds and the block that continues the iteration.
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
    F: FnMut(&mut EGraph, &mut crate::IdSource<EffectToken>, BlockId, NodeId, &[NodeId]) -> LoopBody,
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

    // Stash side-effects and the original continuation. A body may introduce
    // selections, so the suffix belongs to its final continuation block, not
    // necessarily the original block.
    let suffix: Vec<SideEffect> = graph.skeleton.blocks[bid].side_effects.drain(idx_in_block..).collect();
    let original_term = std::mem::replace(
        &mut graph.skeleton.blocks[bid].term,
        SkeletonTerminator::Unreachable,
    );

    let mut carried_nids: Vec<NodeId> = carried.iter().map(|(_, init)| *init).collect();
    let mut current = bid;
    for i in 0..n {
        let idx_nid = graph.intern_pure(PureOp::Int(i.to_string()), smallvec![], i32_ty.clone(), None);
        let body = emit_body(graph, next_effect, current, idx_nid, &carried_nids);
        debug_assert_eq!(body.carried.len(), carried.len());
        carried_nids = body.carried;
        current = body.tail;
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

    graph.skeleton.blocks[current].side_effects.extend(suffix);
    graph.skeleton.blocks[current].term = original_term;
    true
}

/// `Scan[OutputView]`: `new_acc = func(acc, elem, ...caps); view[i] = new_acc`
/// per iteration. One loop-carried value (scalar accumulator). Writes are
/// effectful so the SOAC's `result_node` is bound to a dummy.

/// MapInto: `y = func(elem1, ..., ...caps); view[i] = y` per iteration. No
/// loop-carried state (writes are effectful); the SOAC "result" is a dummy.

fn build_parallel_screma_map(
    graph: &mut EGraph,
    block: BlockId,
    effect_index: usize,
    space: &SegSpace,
    length_input: (NodeId, Type<TypeName>),
    read_inputs: &[(NodeId, Type<TypeName>, Type<TypeName>)],
    pre: &screma::Lambda,
    output_views: &[NodeId],
    result_node: NodeId,
    next_effect: &mut crate::IdSource<EffectToken>,
    regions: &ProgramIdentities,
) {
    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let u32_type = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_type = Type::Constructed(TypeName::Bool, vec![]);
    let after = graph.skeleton.split_block_before_effect(block, effect_index);
    graph.replace_node_preserving_type(
        result_node,
        ENode::Constant(crate::ssa::types::ConstantValue::Bool(false)),
    );
    let body = graph.skeleton.create_block();
    let known = catalog().known();
    let thread = graph.intern_pure(
        PureOp::Intrinsic {
            id: known.thread_id,
            overload_idx: 0,
        },
        smallvec![],
        u32_type,
        None,
    );
    let bitcast = catalog()
        .conversion(&TypeName::Int(32), &TypeName::UInt(32))
        .expect("catalog has structural u32-to-i32 conversion");
    let lane = graph.intern_pure(
        PureOp::Intrinsic {
            id: bitcast,
            overload_idx: 0,
        },
        smallvec![thread],
        i32_type.clone(),
        None,
    );
    let length = emit_seg_space_len(graph, space, &length_input, &i32_type);
    let condition = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Less),
        smallvec![lane, length],
        bool_type,
        None,
    );
    graph.skeleton.blocks[block].term = SkeletonTerminator::CondBranch {
        cond: condition,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args: vec![],
    };
    graph.skeleton.blocks[block].control_header = Some(ControlHeader::Selection { merge: after });

    let elements = read_inputs
        .iter()
        .map(|(array, array_type, element_type)| {
            emit_read_element(graph, body, *array, lane, array_type, element_type, next_effect)
        })
        .collect::<Vec<_>>();
    let values = emit_screma_lambda(graph, regions, pre, elements);
    debug_assert_eq!(values.len(), output_views.len());
    for ((output, value), element_type) in output_views.iter().zip(values).zip(&pre.result_types) {
        let place = graph.intern_pure(
            PureOp::ViewIndex,
            smallvec![*output, lane],
            element_type.clone(),
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

    let mut dimensions = Vec::with_capacity(space.dims().len());
    for extent in space.dims() {
        let dimension = match extent {
            SegExtent::Fixed(count) => {
                graph.intern_pure(PureOp::Int(count.to_string()), smallvec![], i32_ty.clone(), None)
            }
            SegExtent::PushConstant { node, .. } => *node,
            SegExtent::Value(node) => {
                let ty = graph.nodes[*node].ty.clone();
                if is_plain_array_source(&ty) {
                    emit_length(graph, *node, &ty, i32_ty)
                } else {
                    *node
                }
            }
            SegExtent::ResourceLength { node, .. } => {
                let ty = graph.nodes[*node].ty.clone();
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
            PureOp::BinOp(crate::op::BinaryOperator::Multiply),
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
    /// Co-iterated arrays read once per logical filter element.
    read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)>,
    /// The output element type returned by the canonical map lambda.
    output_elem_ty: Type<TypeName>,
    output: PhysicalFilterOutput,
    /// `None` denotes the validated one-input identity map.
    map_func: Option<RegionId>,
    map_captures: Vec<NodeId>,
    pred_func: RegionId,
    captures: Vec<NodeId>,
    result_node: NodeId,
}

fn filter_primary_input(spec: &FilterLoop) -> &(NodeId, Type<TypeName>, Type<TypeName>) {
    spec.read_inputs.first().expect("Filter has no input")
}

/// Read one element from every co-iterated input and invoke the canonical map
/// lambda. Identity is represented without a synthetic region.
fn filter_kept_value(
    graph: &mut EGraph,
    block: BlockId,
    index: NodeId,
    spec: &FilterLoop,
    next_effect: &mut crate::IdSource<EffectToken>,
) -> NodeId {
    let elements = spec
        .read_inputs
        .iter()
        .map(|(array, array_ty, elem_ty)| {
            emit_read_element(graph, block, *array, index, array_ty, elem_ty, next_effect)
        })
        .collect::<SmallVec<[NodeId; 4]>>();
    match &spec.map_func {
        Some(name) => {
            let mut operands = elements;
            operands.extend(spec.map_captures.iter().copied());
            graph.intern_pure(PureOp::Call(*name), operands, spec.output_elem_ty.clone(), None)
        }
        None => {
            debug_assert_eq!(elements.len(), 1);
            elements[0]
        }
    }
}
fn build_filter_loop(
    graph: &mut EGraph,
    bid: BlockId,
    idx_in_block: usize,
    spec: FilterLoop,
    next_effect: &mut crate::IdSource<EffectToken>,
) {
    if let filter::Output::Runtime { scratch, .. } = &spec.output {
        build_runtime_filter_loop(graph, bid, idx_in_block, &spec, *scratch, next_effect);
        return;
    }
    let filter::Output::Local {
        capacity,
        destination,
    } = &spec.output
    else {
        unreachable!()
    };
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let buf_ty = Type::Constructed(
        TypeName::Array,
        vec![
            spec.output_elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            capacity.clone(),
            crate::types::no_buffer(),
        ],
    );

    // Hold the suffix until the result buffer has been loaded at the head of
    // `after`; suffix effects may consume the filter result.
    let after = graph.skeleton.split_block_before_effect(bid, idx_in_block);
    let suffix = graph.skeleton.blocks[after].side_effects.drain(..).collect::<Vec<_>>();
    let buf_place = emit_alloca(graph, bid, buf_ty.clone(), next_effect, None);
    if destination.is_input_buffer() {
        emit_store(
            graph,
            bid,
            buf_place,
            filter_primary_input(&spec).0,
            next_effect,
            None,
        );
    } else if !destination.is_unplaced_fresh() {
        panic!("Filter[OutputView] not supported — see filter-consuming-input.md");
    }

    let zero = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_ty.clone(), None);
    let one = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_ty.clone(), None);
    let after_count = build_serial_filter_cfg(
        graph,
        bid,
        after,
        &spec,
        i32_ty,
        zero,
        one,
        FilterSink::Local(buf_place),
        next_effect,
    );

    let loaded = emit_load(graph, after, buf_place, buf_ty, next_effect, None);
    graph.skeleton.blocks[after].side_effects.extend(suffix);
    graph.replace_pure_node(spec.result_node, PureOp::Tuple(2), smallvec![loaded, after_count]);
}

#[derive(Clone, Copy)]
enum FilterSink {
    Local(NodeId),
    Runtime(NodeId),
}

/// Build the counted serial compaction loop shared by local and runtime
/// filters. Callers choose the index width, destination, and result format.
fn build_serial_filter_cfg(
    graph: &mut EGraph,
    bid: BlockId,
    after: BlockId,
    spec: &FilterLoop,
    index_ty: Type<TypeName>,
    zero: NodeId,
    one: NodeId,
    sink: FilterSink,
    next_effect: &mut crate::IdSource<EffectToken>,
) -> NodeId {
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let after_count = graph.add_block_param(after, index_ty.clone());
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let then_block = graph.skeleton.create_block();
    let else_block = graph.skeleton.create_block();
    let selection_merge = graph.skeleton.create_block();
    let continue_block = graph.skeleton.create_block();
    let count = graph.add_block_param(header, index_ty.clone());
    let index = graph.add_block_param(header, index_ty.clone());

    graph.skeleton.blocks[bid].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![zero, zero],
    };
    let length = emit_length(
        graph,
        filter_primary_input(spec).0,
        &filter_primary_input(spec).1,
        &index_ty,
    );
    let in_range = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Less),
        smallvec![index, length],
        bool_ty.clone(),
        None,
    );
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond: in_range,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args: vec![count],
    };
    graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
        merge: after,
        continue_block,
    });

    let kept = filter_kept_value(graph, body, index, spec, next_effect);
    let mut pred_operands: SmallVec<[NodeId; 4]> = smallvec![kept];
    pred_operands.extend(spec.captures.iter().copied());
    let predicate = graph.intern_pure(PureOp::Call(spec.pred_func), pred_operands, bool_ty, None);
    graph.skeleton.blocks[body].term = SkeletonTerminator::CondBranch {
        cond: predicate,
        then_target: then_block,
        then_args: vec![],
        else_target: else_block,
        else_args: vec![],
    };
    graph.skeleton.blocks[body].control_header = Some(ControlHeader::Selection {
        merge: selection_merge,
    });

    match sink {
        FilterSink::Local(place) => {
            emit_place_index_store(
                graph,
                then_block,
                place,
                count,
                kept,
                spec.output_elem_ty.clone(),
                next_effect,
                None,
            );
        }
        FilterSink::Runtime(view) => {
            emit_storage_store(
                graph,
                then_block,
                view,
                count,
                kept,
                spec.output_elem_ty.clone(),
                next_effect,
                None,
            );
        }
    }
    let bumped_count = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![count, one],
        index_ty.clone(),
        None,
    );
    graph.skeleton.blocks[then_block].term = SkeletonTerminator::Branch {
        target: selection_merge,
        args: vec![bumped_count],
    };
    graph.skeleton.blocks[else_block].term = SkeletonTerminator::Branch {
        target: selection_merge,
        args: vec![count],
    };

    let next_count = graph.add_block_param(selection_merge, index_ty.clone());
    graph.skeleton.blocks[selection_merge].term = SkeletonTerminator::Branch {
        target: continue_block,
        args: vec![next_count],
    };
    let continued_count = graph.add_block_param(continue_block, index_ty.clone());
    let next_index = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![index, one],
        index_ty,
        None,
    );
    graph.skeleton.blocks[continue_block].term = SkeletonTerminator::Branch {
        target: header,
        args: vec![continued_count, next_index],
    };
    after_count
}

/// Runtime-sized `filter` lowering: a single-thread serial scatter into the
/// reserved scratch storage buffer `scratch_out`. The loop carries only a
/// surviving `count` and the input index `i` (both `u32`); kept elements are
/// stored into `scratch_out[count]` and `count` is bumped. The original result
/// node is rebound to a runtime-length view `StorageView(scratch_out)[0, count]`
/// over the buffer — its type (set by `convert_soac_filter`) already carries
/// `Buffer(scratch_out)`, so the backend recovers the descriptor from the type.
/// All offsets/lengths are `u32` to match the view `{offset, len}` convention.
/// Inputs for canonical serial histogram expansion. The form owns all bucket
/// result routing and operation metadata; the loop supplies co-iterated input
/// elements and a domain length.
struct HistLoop {
    form: hist::HistForm,
    /// `(array_nid, array_type, elem_type)` per input, read per iteration.
    read_inputs: Vec<(NodeId, Type<TypeName>, Type<TypeName>)>,
    /// Loop bound source -- the first input `(nid, array_type)`.
    len_input: (NodeId, Type<TypeName>),
    result_node: NodeId,
}

/// Convert one operation's multidimensional index to the row-major scalar
/// offset used by Wyn storage views.
fn flatten_hist_index(graph: &mut EGraph, indices: &[NodeId], shape: &[NodeId]) -> NodeId {
    debug_assert_eq!(indices.len(), shape.len());
    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let Some((&first, rest)) = indices.split_first() else {
        return graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_type, None);
    };
    rest.iter().copied().zip(shape.iter().copied().skip(1)).fold(first, |linear, (index, dimension)| {
        let scaled = graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Multiply),
            smallvec![linear, dimension],
            i32_type.clone(),
            None,
        );
        graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Add),
            smallvec![scaled, index],
            i32_type.clone(),
            None,
        )
    })
}

fn hist_index_in_bounds(graph: &mut EGraph, indices: &[NodeId], shape: &[NodeId]) -> NodeId {
    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_type = Type::Constructed(TypeName::Bool, vec![]);
    let zero = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_type, None);
    indices.iter().copied().zip(shape.iter().copied()).fold(
        graph.intern_pure(PureOp::Bool(true), smallvec![], bool_type.clone(), None),
        |valid, (index, dimension)| {
            let nonnegative = graph.intern_pure(
                PureOp::BinOp(crate::op::BinaryOperator::GreaterEqual),
                smallvec![index, zero],
                bool_type.clone(),
                None,
            );
            let below = graph.intern_pure(
                PureOp::BinOp(crate::op::BinaryOperator::Less),
                smallvec![index, dimension],
                bool_type.clone(),
                None,
            );
            let in_dimension = graph.intern_pure(
                PureOp::BinOp(crate::op::BinaryOperator::LogicalAnd),
                smallvec![nonnegative, below],
                bool_type.clone(),
                None,
            );
            graph.intern_pure(
                PureOp::BinOp(crate::op::BinaryOperator::LogicalAnd),
                smallvec![valid, in_dimension],
                bool_type.clone(),
                None,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_hist_atomic_update(
    graph: &mut EGraph,
    block: BlockId,
    next: BlockId,
    place: NodeId,
    incoming: NodeId,
    value_type: Type<TypeName>,
    operation: &hist::HistOp,
    plan: hist::AtomicUpdate,
    next_effect: &mut crate::IdSource<EffectToken>,
    regions: &ProgramIdentities,
) {
    use super::graph_ops::emit_atomic;
    use crate::ssa::types::AtomicOp;

    match plan {
        hist::AtomicUpdate::Direct(atomic) => {
            emit_atomic(
                graph,
                block,
                place,
                atomic,
                &[incoming],
                value_type,
                next_effect,
                None,
            );
            graph.skeleton.blocks[block].term = SkeletonTerminator::Branch {
                target: next,
                args: vec![],
            };
        }
        hist::AtomicUpdate::CompareExchange => {
            let hist::Update::Reduce { operator, .. } = &operation.update else {
                unreachable!("atomic candidate analysis excludes ordered overwrite")
            };
            let initial = emit_atomic(
                graph,
                block,
                place,
                AtomicOp::Load,
                &[],
                value_type.clone(),
                next_effect,
                None,
            );
            let header = graph.skeleton.create_block();
            let attempt = graph.skeleton.create_block();
            let retry = graph.skeleton.create_block();
            let done = graph.skeleton.create_block();
            let expected = graph.add_block_param(header, value_type.clone());
            let bool_type = Type::Constructed(TypeName::Bool, vec![]);
            let retry_required = graph.add_block_param(header, bool_type.clone());
            let initially_retry =
                graph.intern_pure(PureOp::Bool(true), smallvec![], bool_type.clone(), None);
            graph.skeleton.blocks[block].term = SkeletonTerminator::Branch {
                target: header,
                args: vec![initial, initially_retry],
            };
            graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
                cond: retry_required,
                then_target: attempt,
                then_args: vec![],
                else_target: done,
                else_args: vec![],
            };
            graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
                merge: done,
                continue_block: retry,
            });

            let desired = emit_screma_lambda(graph, regions, operator, vec![expected, incoming])[0];
            let cas_type =
                Type::Constructed(TypeName::Tuple(2), vec![value_type.clone(), bool_type.clone()]);
            let result = emit_atomic(
                graph,
                attempt,
                place,
                AtomicOp::CompareExchange,
                &[expected, desired],
                cas_type,
                next_effect,
                None,
            );
            let observed =
                graph.intern_pure(PureOp::Project { index: 0 }, smallvec![result], value_type, None);
            let exchanged = graph.intern_pure(
                PureOp::Project { index: 1 },
                smallvec![result],
                bool_type.clone(),
                None,
            );
            let retry_after_attempt = graph.intern_pure(
                PureOp::UnaryOp(crate::op::UnaryOperator::LogicalNot),
                smallvec![exchanged],
                bool_type,
                None,
            );
            graph.skeleton.blocks[attempt].term = SkeletonTerminator::Branch {
                target: retry,
                args: vec![],
            };
            graph.skeleton.blocks[retry].term = SkeletonTerminator::Branch {
                target: header,
                args: vec![observed, retry_after_attempt],
            };
            graph.skeleton.blocks[done].term = SkeletonTerminator::Branch {
                target: next,
                args: vec![],
            };
        }
    }
}
/// One invocation processes one input element and issues an atomic update
/// for every operation. Candidate analysis has already proven that each
/// operation is a one-component integer reduction. Structurally recognised
/// operators use a native atomic; general reducers use a CAS retry loop.
fn build_hist_atomic(
    graph: &mut EGraph,
    block: BlockId,
    effect_index: usize,
    spec: HistLoop,
    space: &SegSpace,
    atomic_operations: &[hist::AtomicUpdate],
    next_effect: &mut crate::IdSource<EffectToken>,
    regions: &ProgramIdentities,
) {
    let HistLoop {
        form,
        read_inputs,
        len_input,
        result_node,
    } = spec;
    debug_assert_eq!(form.operations.len(), atomic_operations.len());
    let after = graph.skeleton.split_block_before_effect(block, effect_index);
    graph.replace_node_preserving_type(
        result_node,
        ENode::Constant(crate::ssa::types::ConstantValue::Bool(false)),
    );

    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let u32_type = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_type = Type::Constructed(TypeName::Bool, vec![]);
    let thread = graph.intern_pure(
        PureOp::Intrinsic {
            id: catalog().known().thread_id,
            overload_idx: 0,
        },
        smallvec![],
        u32_type,
        None,
    );
    let bitcast = catalog()
        .conversion(&TypeName::Int(32), &TypeName::UInt(32))
        .expect("catalog has structural u32-to-i32 conversion");
    let lane = graph.intern_pure(
        PureOp::Intrinsic {
            id: bitcast,
            overload_idx: 0,
        },
        smallvec![thread],
        i32_type,
        None,
    );
    let length = emit_seg_space_len(
        graph,
        space,
        &len_input,
        &Type::Constructed(TypeName::Int(32), vec![]),
    );
    let in_range = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Less),
        smallvec![lane, length],
        bool_type,
        None,
    );
    let body = graph.skeleton.create_block();
    graph.skeleton.blocks[block].term = SkeletonTerminator::CondBranch {
        cond: in_range,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args: vec![],
    };
    graph.skeleton.blocks[block].control_header = Some(ControlHeader::Selection { merge: after });

    let arguments = read_inputs
        .iter()
        .map(|(array, array_type, element_type)| {
            emit_read_element(graph, body, *array, lane, array_type, element_type, next_effect)
        })
        .collect::<Vec<_>>();
    let bucket_values = emit_screma_lambda(graph, regions, &form.bucket, arguments);
    let (indices, values) = bucket_values.split_at(form.index_count());
    let mut index_offset = 0;
    let mut value_offset = 0;
    let mut current = body;

    for (operation, atomic) in form.operations.iter().zip(atomic_operations) {
        let operation_indices = &indices[index_offset..index_offset + operation.index_count()];
        let operation_values = &values[value_offset..value_offset + operation.value_count()];
        index_offset += operation.index_count();
        value_offset += operation.value_count();
        let update = graph.skeleton.create_block();
        let next = graph.skeleton.create_block();
        let valid = hist_index_in_bounds(graph, operation_indices, &operation.shape);
        graph.skeleton.blocks[current].term = SkeletonTerminator::CondBranch {
            cond: valid,
            then_target: update,
            then_args: vec![],
            else_target: next,
            else_args: vec![],
        };
        graph.skeleton.blocks[current].control_header = Some(ControlHeader::Selection { merge: next });

        let bucket_index = flatten_hist_index(graph, operation_indices, &operation.shape);
        let value_type = operation.update.value_types()[0].clone();
        let place = graph.intern_pure(
            PureOp::ViewIndex,
            smallvec![operation.destinations[0], bucket_index],
            value_type.clone(),
            None,
        );
        emit_hist_atomic_update(
            graph,
            update,
            next,
            place,
            operation_values[0],
            value_type,
            operation,
            *atomic,
            next_effect,
            regions,
        );
        current = next;
    }
    graph.skeleton.blocks[current].term = SkeletonTerminator::Branch {
        target: after,
        args: vec![],
    };
}
/// Canonical serial histogram semantics. Bucket results are decoded in
/// Futhark order: all operation indices, then all operation values. A
/// multi-component reducer is invoked once with all previous components and
/// all incoming components, then its results are stored componentwise.
fn build_hist_loop(
    graph: &mut EGraph,
    bid: BlockId,
    idx_in_block: usize,
    spec: HistLoop,
    next_effect: &mut crate::IdSource<EffectToken>,
    regions: &ProgramIdentities,
) {
    use super::graph_ops::{emit_storage_store, emit_view_load};
    let HistLoop {
        form,
        read_inputs,
        len_input,
        result_node,
    } = spec;

    let result = ResultBinding::DummyBool { result_node };

    expand_loop(
        graph,
        bid,
        idx_in_block,
        &len_input,
        &[],
        &result,
        next_effect,
        true,
        move |graph, next_effect, blk, i_nid, _carried| {
            let mut arguments = Vec::with_capacity(read_inputs.len());
            for (array, array_type, element_type) in &read_inputs {
                arguments.push(emit_read_element(
                    graph,
                    blk,
                    *array,
                    i_nid,
                    array_type,
                    element_type,
                    next_effect,
                ));
            }
            let bucket_values = emit_screma_lambda(graph, regions, &form.bucket, arguments);
            debug_assert_eq!(bucket_values.len(), form.index_count() + form.value_count());
            let (indices, values) = bucket_values.split_at(form.index_count());
            let mut index_offset = 0;
            let mut value_offset = 0;

            let mut current = blk;
            for operation in &form.operations {
                let operation_indices = &indices[index_offset..index_offset + operation.index_count()];
                let operation_values = &values[value_offset..value_offset + operation.value_count()];
                index_offset += operation.index_count();
                value_offset += operation.value_count();

                // Futhark Hist ignores an update unless every index component
                // is in range. Keep both the destination load and store in
                // the selected block so serial and atomic paths agree.
                let update = graph.skeleton.create_block();
                let next = graph.skeleton.create_block();
                let valid = hist_index_in_bounds(graph, operation_indices, &operation.shape);
                graph.skeleton.blocks[current].term = SkeletonTerminator::CondBranch {
                    cond: valid,
                    then_target: update,
                    then_args: vec![],
                    else_target: next,
                    else_args: vec![],
                };
                graph.skeleton.blocks[current].control_header =
                    Some(ControlHeader::Selection { merge: next });

                let bucket_index = flatten_hist_index(graph, operation_indices, &operation.shape);
                let value_types = operation.update.value_types();
                let updated_values = match &operation.update {
                    hist::Update::OrderedOverwrite { .. } => operation_values.to_vec(),
                    hist::Update::Reduce { operator, .. } => {
                        let mut reducer_arguments = Vec::with_capacity(operation.value_count() * 2);
                        for (&destination, value_type) in operation.destinations.iter().zip(value_types) {
                            reducer_arguments.push(emit_view_load(
                                graph,
                                update,
                                destination,
                                bucket_index,
                                value_type.clone(),
                                next_effect,
                                None,
                            ));
                        }
                        reducer_arguments.extend_from_slice(operation_values);
                        emit_screma_lambda(graph, regions, operator, reducer_arguments)
                    }
                };
                debug_assert_eq!(updated_values.len(), operation.destinations.len());
                for ((&destination, value_type), updated) in
                    operation.destinations.iter().zip(value_types).zip(updated_values)
                {
                    emit_storage_store(
                        graph,
                        update,
                        destination,
                        bucket_index,
                        updated,
                        value_type.clone(),
                        next_effect,
                        None,
                    );
                }
                graph.skeleton.blocks[update].term = SkeletonTerminator::Branch {
                    target: next,
                    args: vec![],
                };
                current = next;
            }
            LoopBody {
                tail: current,
                carried: vec![],
            }
        },
    );
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
    let len = emit_length(
        graph,
        filter_primary_input(&spec).0,
        &filter_primary_input(&spec).1,
        &u32_ty,
    );
    let bounded = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Less),
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
    graph.skeleton.blocks[bid].control_header = Some(ControlHeader::Selection { merge: after });
    let kept = filter_kept_value(graph, in_range, gid, &spec, next_effect);
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
    graph.skeleton.blocks[in_range].control_header = Some(ControlHeader::Selection { merge: pred_merge });
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
    bid: BlockId,
    idx: usize,
    spec: FilterLoop,
    work: FilterWorkBuffers,
    scan_workgroup_width: u32,
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
    let input_len = emit_length(
        graph,
        filter_primary_input(&spec).0,
        &filter_primary_input(&spec).1,
        &u32_ty,
    );
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
        PureOp::Uint(scan_workgroup_width.to_string()),
        smallvec![],
        u32_ty.clone(),
        None,
    );
    let total_threads = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Multiply),
        smallvec![nwg, wg_width],
        u32_ty.clone(),
        None,
    );
    let total_minus_one = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Subtract),
        smallvec![total_threads, one],
        u32_ty.clone(),
        None,
    );
    let len_plus = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![input_len, total_minus_one],
        u32_ty.clone(),
        None,
    );
    let chunk_size = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Divide),
        smallvec![len_plus, total_threads],
        u32_ty.clone(),
        None,
    );
    let raw_chunk_start = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Multiply),
        smallvec![gid, chunk_size],
        u32_ty.clone(),
        None,
    );
    let u32_min = catalog()
        .specialize_numeric(catalog().known().min, &TypeName::UInt(32))
        .expect("catalog has u32 min specialization");
    let chunk_start = graph.intern_pure(
        PureOp::Intrinsic {
            id: u32_min,
            overload_idx: 0,
        },
        smallvec![raw_chunk_start, input_len],
        u32_ty.clone(),
        None,
    );
    let remaining = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Subtract),
        smallvec![input_len, chunk_start],
        u32_ty.clone(),
        None,
    );
    let chunk_len = graph.intern_pure(
        PureOp::Intrinsic {
            id: u32_min,
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
        PureOp::BinOp(crate::op::BinaryOperator::Less),
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
    graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
        merge: after,
        continue_block: body,
    });
    let flags = intern_storage_view(graph, work.flags, u32_ty.clone(), None);
    let offsets = intern_storage_view(graph, work.offsets, u32_ty.clone(), None);
    let global_i = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
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
        PureOp::BinOp(crate::op::BinaryOperator::Add),
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
    let next_i = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Add),
        smallvec![i, one],
        u32_ty.clone(),
        None,
    );
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
    let len = emit_length(
        graph,
        filter_primary_input(&spec).0,
        &filter_primary_input(&spec).1,
        &u32_ty,
    );
    let bounded = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Less),
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
    graph.skeleton.blocks[bid].control_header = Some(ControlHeader::Selection { merge: after });
    let flags = intern_storage_view(graph, work.flags, u32_ty.clone(), None);
    let offsets = intern_storage_view(graph, work.offsets, u32_ty.clone(), None);
    let flag_place = graph.intern_pure(PureOp::ViewIndex, smallvec![flags, gid], u32_ty.clone(), None);
    let flag = emit_load(graph, in_range, flag_place, u32_ty.clone(), next_effect, None);
    let one = intern_u32(graph, 1, None);
    let keep = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Equal),
        smallvec![flag, one],
        bool_ty,
        None,
    );
    graph.skeleton.blocks[in_range].term = SkeletonTerminator::CondBranch {
        cond: keep,
        then_target: write,
        then_args: vec![],
        else_target: skip,
        else_args: vec![],
    };
    graph.skeleton.blocks[in_range].control_header = Some(ControlHeader::Selection { merge });
    let offset_place = graph.intern_pure(PureOp::ViewIndex, smallvec![offsets, gid], u32_ty.clone(), None);
    let inclusive = emit_load(graph, write, offset_place, u32_ty.clone(), next_effect, None);
    let output_index = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Subtract),
        smallvec![inclusive, one],
        u32_ty.clone(),
        None,
    );
    let kept = filter_kept_value(graph, write, gid, &spec, next_effect);
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
    bid: BlockId,
    idx_in_block: usize,
    spec: &FilterLoop,
    scratch_out: crate::BindingRef,
    next_effect: &mut crate::IdSource<EffectToken>,
) {
    use super::graph_ops::{intern_storage_view, intern_u32};

    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let scratch_view = intern_storage_view(graph, scratch_out, spec.output_elem_ty.clone(), None);
    let after = graph.skeleton.split_block_before_effect(bid, idx_in_block);
    let zero = intern_u32(graph, 0, None);
    let one = intern_u32(graph, 1, None);
    let after_count = build_serial_filter_cfg(
        graph,
        bid,
        after,
        spec,
        u32_ty.clone(),
        zero,
        one,
        FilterSink::Runtime(scratch_view),
        next_effect,
    );

    if let filter::Output::Runtime {
        length: filter::RuntimeLength::Stored(length),
        ..
    } = &spec.output
    {
        let length_view = intern_storage_view(graph, *length, u32_ty.clone(), None);
        emit_storage_store(
            graph,
            after,
            length_view,
            zero,
            after_count,
            u32_ty,
            next_effect,
            None,
        );
    }
    graph.replace_pure_node(
        spec.result_node,
        PureOp::StorageView(crate::op::PureViewSource::Storage(scratch_out)),
        smallvec![zero, after_count],
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
        PureOp::BinOp(crate::op::BinaryOperator::Less),
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

    graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
        merge: after,
        continue_block: body,
    });

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
        PureOp::BinOp(crate::op::BinaryOperator::Add),
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
    let actual_arr_ty =
        graph.nodes.get(arr_nid).map(|node| &node.ty).filter(|ty| is_plain_array_source(ty)).cloned();
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
    let actual_arr_ty =
        graph.nodes.get(arr_nid).map(|node| &node.ty).filter(|ty| is_plain_array_source(ty)).cloned();
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
            PureOp::BinOp(crate::op::BinaryOperator::Multiply),
            smallvec![idx_nid, step_nid],
            elem_ty.clone(),
            None,
        );
        graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Add),
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
