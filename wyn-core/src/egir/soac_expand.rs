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
    alloc_effect, detached_alloca, emit_alloca, emit_load, emit_place_index_store, emit_storage_store,
    emit_store,
};
use super::program::{
    PhysicalEGraph as EGraph, PhysicalFilterOutput, PhysicalFilterWorkBuffers as FilterWorkBuffers,
    PhysicalFunc, PhysicalSegSpace as SegSpace, PhysicalSideEffect as SideEffect,
    PhysicalSideEffectKind as SideEffectKind, PhysicalSoac as Soac,
};
use super::soac::{filter, hist, screma};
use crate::ast::TypeName;
use crate::types::{is_array_variant_view, is_virtual_array, TypeExt};

use super::types::{
    as_soa_tuple, soac_element_type, ArrayLayout, EffectOp, EffectToken, PlaceDestination, PlaceId, PureOp,
    RegionId, ResultBinding, ResultDestination, SkeletonTerminator, SoacDestination, SoacEffect, ValueId,
    ValueKind, ViewId, WynLanguage,
};

type CallableMap = crate::LookupMap<RegionId, PhysicalFunc>;

mod array_io;
mod call_abi;
mod filter_lowering;
mod flow_normalize;
mod hist_lowering;
mod loop_builder;
mod screma_lowering;

use array_io::emit_read_element;
use filter_lowering::{
    build_filter_flags, build_filter_loop, build_filter_scan, build_filter_scatter, FilterLoop,
};
use flow_normalize::normalize_place_backed_flow;
use hist_lowering::{
    build_bucket_finish, build_bucket_init, build_bucket_insert, build_hist_atomic, build_hist_loop,
    HistLoop,
};
use loop_builder::{expand_loop, LoopBody, LoopResultBinding, LoopResultSource};
use screma_lowering::{build_parallel_screma_map, emit_screma_lambda};

/// Expand every graph-bearing body and rebuild the program at the
/// post-expansion checkpoint.
pub fn expand_soacs(program: super::parallelize::Planned) -> Result<SoacsExpanded, String> {
    let callables = program
        .functions
        .iter()
        .map(|function| (function.region, function.clone()))
        .collect::<CallableMap>();
    let program = program.try_map_graphs_with_state(|_, graph, _, context| {
        run_one_body(graph, &callables, &mut context.effect_ids)
    })?;
    call_abi::resolve(program).map(|program| program.retag())
}

/// Expand every physical SOAC in the skeleton.
pub fn run_one_body(
    mut graph: EGraph,
    callables: &CallableMap,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<EGraph, String> {
    // Re-scan after every expansion because splitting a block moves the
    // remaining suffix. Selecting the first operation preserves producer to
    // consumer order, so a resolved destination is visible to later updates.
    while let Some((bid, idx)) = graph.skeleton.blocks.iter().find_map(|(bid, block)| {
        block.side_effects.iter().position(|effect| is_handleable_soac(&effect.kind)).map(|idx| (bid, idx))
    }) {
        expand_one(&mut graph, bid, idx, effect_ids, callables)?;
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
    normalize_place_backed_flow(&mut graph, effect_ids)?;
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

fn bind_result_alias(graph: &mut EGraph, result: ValueId, replacement: ValueId) {
    if result == replacement {
        return;
    }
    graph.replace_value_references(result, replacement);
    graph.install_aliases([(result, replacement)]);
}

fn bind_result_value(graph: &mut EGraph, result: &ResultBinding<Type<TypeName>>, replacement: ValueId) {
    let abi = super::types::by_value_function_result::<WynLanguage>(result.ty().clone());
    let replacement = super::graph_ops::bind_by_value_result(graph, &abi, replacement);
    bind_result_binding(graph, result, &replacement);
}

fn bind_result_binding(
    graph: &mut EGraph,
    result: &ResultBinding<Type<TypeName>>,
    replacement: &ResultBinding<Type<TypeName>>,
) {
    assert_eq!(result.ty(), replacement.ty());
    let results = result.values();
    let replacements = replacement.values();
    assert_eq!(
        results.len(),
        replacements.len(),
        "result ABI changed during SOAC expansion"
    );
    for (result, replacement) in results.into_iter().zip(replacements) {
        bind_result_alias(graph, result, replacement);
    }
    super::graph_ops::fold_exposed_projections(graph);
    for replacement in replacement.values() {
        super::graph_ops::normalize_place_backed_value_consumers(graph, replacement);
    }
}

fn value_binding(graph: &mut EGraph, ty: &Type<TypeName>, value: ValueId) -> ResultBinding<Type<TypeName>> {
    let abi = super::types::by_value_function_result::<WynLanguage>(ty.clone());
    super::graph_ops::bind_by_value_result(graph, &abi, value)
}

fn emit_mapped_result_stores(
    graph: &mut EGraph,
    block: BlockId,
    lane: ValueId,
    produced: &ResultBinding<Type<TypeName>>,
    destination: &ResultBinding<Type<TypeName>>,
    next_effect: &mut crate::IdSource<EffectToken>,
) -> BlockId {
    if let Some((array_ty, destination)) = destination.single_destination() {
        let element_ty =
            crate::types::array_elem(array_ty).expect("mapped result destination is not an array");
        assert_eq!(element_ty, produced.ty());
        let place = match destination {
            ResultDestination::ReturnValue(view) => {
                graph.add_view_index_place(graph.view_id(*view), lane, element_ty.clone(), None)
            }
            ResultDestination::Place(PlaceDestination::Fixed(place)) => {
                graph.add_index_place(*place, lane, element_ty.clone(), None)
            }
            ResultDestination::Place(PlaceDestination::Bounded { storage, .. }) => {
                graph.add_index_place(*storage, lane, element_ty.clone(), None)
            }
        };
        return super::graph_ops::emit_result_to_place(graph, block, produced, place, next_effect, None)
            .expect("mapped result must be writable through its selected destination");
    }

    let produced_fields = produced.top_level_fields();
    let destination_fields = destination.top_level_fields();
    assert_eq!(produced_fields.len(), destination_fields.len());
    produced_fields.iter().zip(&destination_fields).fold(block, |tail, (produced, destination)| {
        emit_mapped_result_stores(graph, tail, lane, produced, destination, next_effect)
    })
}

fn expand_one(
    graph: &mut EGraph,
    bid: BlockId,
    idx: usize,
    next_effect: &mut crate::IdSource<EffectToken>,
    callables: &CallableMap,
) -> Result<(), String> {
    let se = graph.skeleton.blocks[bid].side_effects.remove(idx);
    match &se.kind {
        SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op)))
            if op.is_serial()
                || (matches!(op.state, screma::PhysicalState::Segmented(_))
                    && (0..op.result_count())
                        .any(|field| matches!(op.destination(field), Some(SoacDestination::Fresh)))) =>
        {
            let operands = screma::ScremaOperands::decode(op, &se.operands, se.result.as_ref())?;
            let input_nids = operands
                .inputs()
                .map(|operand| {
                    operand.operand.value().ok_or_else(|| "Screma input is not a value or view".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let Some(first_input) = op.inputs.first() else {
                return Err("serial Screma has no array input".into());
            };
            let result_fields = operands.result_fields();

            let read_inputs = input_nids
                .iter()
                .zip(&op.inputs)
                .map(|(&node, input)| (node, input.array.clone(), input.element()))
                .collect::<Vec<_>>();
            let len_input = (input_nids[0], first_input.array.clone());
            let reduction_components = op.form.reduction_result_count();
            let scan_components = op.form.scan_input_count();
            let post_count = op.form.post.result_types.len();
            // Only scalar accumulators cross the loop edge. Array results are
            // written through a selected local or external destination.
            let mut carried = Vec::new();
            let mut post_sinks = Vec::with_capacity(post_count);
            let mut loop_results = Vec::with_capacity(op.result_count());
            let mut prelude = Vec::new();
            let mut place_backed_results = Vec::new();
            for post in 0..post_count {
                let field = reduction_components + post;
                let destination = op
                    .destination(field)
                    .ok_or_else(|| format!("Screma result {field} has no destination"))?;
                let result = &result_fields[field];
                let sink = match destination {
                    SoacDestination::UniqueInput => {
                        return Err("unresolved UniqueInput destination reached physical expansion".into())
                    }
                    SoacDestination::Fresh => {
                        let mut views = Vec::with_capacity(result.destination_count());
                        let sink = result.map_destinations(|ty, _| {
                            let (place, effect) = detached_alloca(graph, ty.clone(), next_effect, None);
                            prelude.push(effect);
                            let view_ty = crate::types::view_array_of(ty, crate::types::no_buffer());
                            let view = graph.add_place_view(place, view_ty, None);
                            views.push(view.value());
                            place_backed_results.push(view.value());
                            ResultDestination::Place(PlaceDestination::Fixed(place))
                        });
                        let mut views = views.into_iter();
                        let binding = result.map_destinations(|_, _| {
                            ResultDestination::ReturnValue(
                                views.next().expect("fresh destination view count changed"),
                            )
                        });
                        bind_result_binding(graph, result, &binding);
                        sink
                    }
                    SoacDestination::OutputView => {
                        let output =
                            operands.output(field).and_then(|operand| operand.operand.value()).ok_or_else(
                                || format!("Screma post result {post} has no output-view operand"),
                            )?;
                        let binding = output_component_views(graph, result, output)?;
                        bind_result_binding(graph, result, &binding);
                        mapped_view_destination(graph, result.ty(), output)?
                    }
                    SoacDestination::InputBuffer => {
                        if input_nids.len() != 1 {
                            return Err(
                                "input-buffer Screma result requires exactly one array input".into()
                            );
                        }
                        let output = input_nids[0];
                        let output = value_binding(graph, result.ty(), output);
                        bind_result_binding(graph, result, &output);
                        output
                    }
                };
                post_sinks.push(sink);
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

            for component in 0..reduction_components {
                loop_results.push(LoopResultBinding {
                    result: result_fields[component].clone(),
                    source: LoopResultSource::Carried(scan_components + component),
                });
            }

            let prelude_count = prelude.len();
            graph.skeleton.blocks[bid].side_effects.splice(idx..idx, prelude);

            let continuation = expand_loop(
                graph,
                bid,
                idx + prelude_count,
                &len_input,
                &carried,
                &loop_results,
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
                    let pre_results =
                        emit_screma_lambda(graph, body, callables, &op.form.pre, input_elements);
                    let pre_values = super::soac::lambda::materialize_result_values(graph, &pre_results);

                    let mut pre_offset = 0;
                    let mut scan_offset = 0;
                    let mut new_scans = Vec::with_capacity(scan_components);
                    for scan in &op.form.scans {
                        let width = scan.neutral.len();
                        let mut arguments = carried_values[scan_offset..scan_offset + width].to_vec();
                        arguments.extend_from_slice(&pre_values[pre_offset..pre_offset + width]);
                        let results = emit_screma_lambda(graph, body, callables, &scan.operator, arguments);
                        new_scans.extend(super::soac::lambda::materialize_result_values(graph, &results));
                        pre_offset += width;
                        scan_offset += width;
                    }

                    let mut reduction_offset = scan_components;
                    let mut new_reductions = Vec::with_capacity(reduction_components);
                    for reduction in &op.form.reductions {
                        let width = reduction.neutral.len();
                        let mut arguments =
                            carried_values[reduction_offset..reduction_offset + width].to_vec();
                        arguments.extend_from_slice(&pre_values[pre_offset..pre_offset + width]);
                        let results =
                            emit_screma_lambda(graph, body, callables, &reduction.operator, arguments);
                        new_reductions
                            .extend(super::soac::lambda::materialize_result_values(graph, &results));
                        pre_offset += width;
                        reduction_offset += width;
                    }

                    let mut post_arguments = new_scans.clone();
                    post_arguments.extend_from_slice(&pre_values[pre_offset..]);
                    let post_results =
                        emit_screma_lambda(graph, body, callables, &op.form.post, post_arguments);
                    debug_assert_eq!(post_results.len(), post_count);

                    let mut next = Vec::with_capacity(carried_values.len());
                    let mut tail = body;
                    for (produced, sink) in post_results.iter().zip(&post_sinks) {
                        tail = emit_mapped_result_stores(graph, tail, lane, produced, sink, next_effect);
                    }
                    next.extend(new_scans);
                    next.extend(new_reductions);
                    LoopBody { tail, carried: next }
                },
            );
            for result in place_backed_results {
                super::graph_ops::materialize_place_backed_projections(
                    graph,
                    result,
                    continuation,
                    next_effect,
                );
            }
        }
        SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) => {
            let input_count = op.body.inputs.len();
            let read_inputs = se.operands[..input_count]
                .iter()
                .copied()
                .zip(&op.body.inputs)
                .map(|(operand, input)| {
                    (
                        operand.value().expect("Filter input is a value or view"),
                        input.array.clone(),
                        input.element(),
                    )
                })
                .collect::<Vec<_>>();
            let map_body = op.body.map.seg_body();
            let map_func = map_body
                .map(|body| callables.get(&body.region).expect("Filter map callable boundary").clone());
            let output_elem_ty = op.body.output_element_type();
            let predicate_body =
                op.body.predicate.seg_body().expect("validated Filter predicate has a region");
            let pred_func =
                callables.get(&predicate_body.region).expect("Filter predicate callable boundary").clone();
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
            let result_nid = se.value_result().expect("Filter has one by-value result root");

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
            let input_nids = &se.operands[..n_inputs];
            let read_inputs: Vec<(ValueId, Type<TypeName>, Type<TypeName>, Vec<u8>, ArrayLayout)> =
                input_nids
                    .iter()
                    .zip(op.inputs.iter())
                    .map(|(nid, input)| {
                        (
                            nid.value().expect("Hist input is a value or view"),
                            input.array.clone(),
                            input.element(),
                            input.dimensions.clone(),
                            input.layout.clone(),
                        )
                    })
                    .collect();
            let len_input = (
                input_nids[0].value().expect("Hist input is a value or view"),
                op.inputs[0].array.clone(),
            );
            let result_nid = se.value_result().expect("Hist has one by-value result root");

            let hist = HistLoop {
                form: op.form.clone(),
                read_inputs,
                len_input,
                result_node: result_nid,
            };
            match &op.state {
                hist::PhysicalState::Atomic { space, operations } => {
                    build_hist_atomic(graph, bid, idx, hist, space, operations, next_effect, callables)
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
                        callables,
                    ),
                    hist::ParallelStage::Finish => build_bucket_finish(graph, bid, idx, hist.result_node),
                },
                hist::PhysicalState::Serial => {
                    build_hist_loop(graph, bid, idx, hist, next_effect, callables)
                }
            }
        }
        SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op)))
            if matches!(op.state, screma::PhysicalState::Segmented(_)) && op.is_map() =>
        {
            let screma::PhysicalState::Segmented(segment) = &op.state else {
                unreachable!()
            };
            let operands = screma::ScremaOperands::decode(op, &se.operands, se.result.as_ref())?;
            let input_nodes = operands
                .inputs()
                .map(|operand| {
                    operand
                        .operand
                        .value()
                        .ok_or_else(|| "segmented map input is not a value or view".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let read_inputs = input_nodes
                .iter()
                .zip(&op.inputs)
                .map(|(&node, input)| (node, input.array.clone(), input.element()))
                .collect::<Vec<_>>();
            let Some(first_input) = read_inputs.first() else {
                return Err("segmented map has no array input".into());
            };
            let result_fields = operands.result_fields();
            let mut output_views = Vec::with_capacity(op.result_count());
            for field in 0..op.result_count() {
                let destination = op
                    .destination(field)
                    .ok_or_else(|| format!("segmented map result {field} has no destination"))?;
                let output = if destination.is_output_view() {
                    operands
                        .output(field)
                        .and_then(|operand| operand.operand.value())
                        .ok_or_else(|| format!("segmented map result {field} has no output view"))?
                } else if destination.is_input_buffer() && input_nodes.len() == 1 {
                    input_nodes[0]
                } else {
                    return Err(format!(
                        "segmented map result {field} has unsupported destination {destination:?}"
                    ));
                };
                let result = &result_fields[field];
                let output = value_binding(graph, result.ty(), output);
                bind_result_binding(graph, result, &output);
                if output.destination_count() != result.destination_count() {
                    return Err(format!(
                        "segmented map result {field} has {} physical leaves but its output has {}",
                        result.destination_count(),
                        output.destination_count()
                    ));
                }
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
                next_effect,
                callables,
            );
        }
        _ => return Err("SOAC expansion target changed after selection".into()),
    }
    Ok(())
}

fn mapped_view_destination(
    graph: &EGraph,
    logical_array: &Type<TypeName>,
    view: ValueId,
) -> Result<ResultBinding<Type<TypeName>>, String> {
    let Type::Constructed(TypeName::Array, arguments) = graph.nodes[view].ty().clone() else {
        return Err("mapped output destination is not an array view".into());
    };
    let mut arguments = arguments;
    arguments[0] = soac_element_type(logical_array);
    Ok(ResultBinding::destination(
        Type::Constructed(TypeName::Array, arguments),
        ResultDestination::ReturnValue(view),
    ))
}

fn output_component_views(
    graph: &mut EGraph,
    result: &ResultBinding<Type<TypeName>>,
    view: ValueId,
) -> Result<ResultBinding<Type<TypeName>>, String> {
    let leaves = result.destination_leaves();
    if leaves.len() == 1 {
        return Ok(result.map_destinations(|_, _| ResultDestination::ReturnValue(view)));
    }
    let parent_ty = graph.nodes[view].ty().clone();
    let parent_elem = crate::types::array_elem(&parent_ty)
        .ok_or_else(|| "mapped output destination is not an array view".to_owned())?;
    let parent_elem_bytes = crate::ssa::layout::type_byte_size(parent_elem)
        .ok_or_else(|| "mapped output destination has no physical element size".to_owned())?;
    let region = parent_ty
        .array_buffer()
        .cloned()
        .ok_or_else(|| "mapped output destination has no storage region".to_owned())?;
    let mut offset_bytes = 0u32;
    let mut component_views = Vec::with_capacity(leaves.len());
    for leaf in &leaves {
        let bytes = crate::ssa::layout::type_byte_size(leaf.ty()).ok_or_else(|| {
            "a structured mapped output component must have a fixed physical size".to_owned()
        })?;
        if offset_bytes % parent_elem_bytes != 0 {
            return Err("mapped output component is not aligned to its storage element".into());
        }
        let Type::Constructed(TypeName::Size(length), _) =
            leaf.ty().array_size().ok_or_else(|| "mapped output component is not an array".to_owned())?
        else {
            return Err("a structured mapped output component must have a fixed length".into());
        };
        let offset = super::graph_ops::intern_u32(graph, offset_bytes / parent_elem_bytes, None);
        let length = super::graph_ops::intern_u32(
            graph,
            u32::try_from(*length).map_err(|_| "mapped output length exceeds u32")?,
            None,
        );
        let view_ty = crate::types::view_array_of(leaf.ty(), region.clone());
        component_views.push(super::graph_ops::intern_inherited_view(
            graph, view, offset, length, view_ty, None,
        ));
        offset_bytes = offset_bytes
            .checked_add(bytes)
            .ok_or_else(|| "mapped output component offsets overflow u32".to_owned())?;
    }
    let mut component_views = component_views.into_iter();
    Ok(result.map_destinations(|_, _| {
        ResultDestination::ReturnValue(
            component_views.next().expect("component view count matches result leaves"),
        )
    }))
}

#[cfg(test)]
#[path = "soac_expand_tests.rs"]
mod soac_expand_tests;
