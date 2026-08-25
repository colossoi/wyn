//! Screma expansion helpers.

use super::array_io::{emit_read_element, emit_seg_space_len};
use super::loop_builder::{expand_loop, LoopBody, LoopResultBinding, LoopResultSource};
use super::{load_result_arguments, result_is_addressable, value_binding, CallableMap};
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::egir::graph_ops::{
    bind_by_value_result, emit_result_to_indexed_destination, rebind_physical_result,
};
use crate::egir::graph_ops::{detached_alloca, materialize_place_backed_projections};
use crate::egir::physical_call_abi::emit_call;
use crate::egir::program::Func;
use crate::egir::soac::lambda::logical_result_fields;
use crate::egir::soac::screma;
use crate::egir::structured_cfg::{finish_guarded_selection, replace_effect_with_guarded_selection};
use crate::egir::types::{
    by_value_function_result, EGraph, EffectToken, Physical, PlaceDestination, PureOp, ResultBinding,
    ResultDestination, SegSpace, SideEffect, SideEffectKind, SideEffectSite, Soac, SoacEffect,
    SoacOwnership, ValueId, WynLanguage,
};
use crate::flow::BlockId;
use crate::op;
use crate::types;
use crate::BindingRef;
use polytype::Type;
use smallvec::smallvec;
use wyn_base::IdSource;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MappedCallMode {
    DirectDestinationPassing,
    StructuredStore,
}

fn mapped_call_mode(
    callee: &Func<Physical>,
    lambda: &screma::Lambda,
    destinations: &[ResultBinding<Type<TypeName>>],
) -> Result<MappedCallMode, String> {
    let results = match lambda.result_types.as_slice() {
        [] => Vec::new(),
        [_] => vec![callee.result().clone()],
        _ => callee.result().top_level_fields(),
    };
    if results.len() != destinations.len() {
        return Err(format!(
            "mapped lambda has {} logical results but {} destinations",
            results.len(),
            destinations.len()
        ));
    }

    let mut has_structured_store = false;
    for (index, (result, destination)) in results.iter().zip(destinations).enumerate() {
        let result_leaves = result.destination_leaves();
        let destination_leaves = destination.destination_leaves();
        let direct = result_leaves.len() == destination_leaves.len()
            && result_leaves.iter().zip(&destination_leaves).all(|(result_leaf, destination_leaf)| {
                destination_leaf.single_destination().and_then(|(array_ty, _)| types::array_elem(array_ty))
                    == Some(result_leaf.ty())
            });
        if direct {
            continue;
        }

        let structured = result.is_product()
            && destination.single_destination().and_then(|(array_ty, _)| types::array_elem(array_ty))
                == Some(result.ty());
        if structured {
            has_structured_store = true;
            continue;
        }

        return Err(format!(
            "mapped lambda result {index} of type {:?} does not match destination type {:?}",
            result.ty(),
            destination.ty()
        ));
    }

    Ok(if has_structured_store {
        MappedCallMode::StructuredStore
    } else {
        MappedCallMode::DirectDestinationPassing
    })
}

pub(super) fn emit_screma_lambda(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    callables: &CallableMap,
    lambda: &screma::Lambda,
    mut arguments: Vec<ValueId>,
    mapped_destinations: Option<(&[ResultBinding<Type<TypeName>>], ValueId)>,
    next_effect: &mut IdSource<EffectToken>,
) -> Vec<ResultBinding<Type<TypeName>>> {
    if lambda.is_identity() {
        debug_assert_eq!(arguments.len(), lambda.result_types.len());
        return arguments
            .into_iter()
            .zip(&lambda.result_types)
            .map(|(argument, ty)| {
                let abi = by_value_function_result::<WynLanguage>(ty.clone());
                bind_by_value_result(graph, &abi, argument)
            })
            .collect();
    }
    let body = lambda.seg_body().expect("non-identity Screma lambda has a region");
    let callee = callables.get(&body.region).expect("Screma lambda callable boundary");
    let mut operands = arguments.drain(..).map(|argument| graph.operand_ref(argument)).collect::<Vec<_>>();
    operands.extend(body.captures.iter().copied());
    let result = match mapped_destinations {
        None => emit_call(graph, block, callee, operands, None, next_effect),
        Some((destinations, lane)) => match mapped_call_mode(callee, lambda, destinations)
            .expect("mapped Screma result must have a recognized destination shape")
        {
            MappedCallMode::DirectDestinationPassing => emit_call(
                graph,
                block,
                callee,
                operands,
                Some((destinations, lane)),
                next_effect,
            ),
            MappedCallMode::StructuredStore => emit_call(graph, block, callee, operands, None, next_effect),
        },
    }
    .expect("Screma lambda call must match its canonical boundary");
    logical_result_fields(&result, &lambda.result_types)
}

fn fresh_result_destination(
    graph: &mut EGraph<Physical>,
    result: &ResultBinding<Type<TypeName>>,
    next_effect: &mut IdSource<EffectToken>,
    prelude: &mut Vec<SideEffect<Physical>>,
) -> (
    ResultBinding<Type<TypeName>>,
    ResultBinding<Type<TypeName>>,
    Vec<ValueId>,
) {
    if result.is_product() {
        let mut sinks = Vec::new();
        let mut bindings = Vec::new();
        let mut views = Vec::new();
        for field in result.top_level_fields() {
            let (sink, binding, field_views) =
                fresh_result_destination(graph, &field, next_effect, prelude);
            sinks.push(sink);
            bindings.push(binding);
            views.extend(field_views);
        }
        return (
            ResultBinding::product(result.ty().clone(), sinks),
            ResultBinding::product(result.ty().clone(), bindings),
            views,
        );
    }

    let (place, effect) = detached_alloca(graph, result.ty().clone(), next_effect, None);
    prelude.push(effect);
    let view_ty = types::view_array_of(result.ty(), types::no_buffer());
    let view = graph.add_place_view(place, view_ty, None).value();
    (
        ResultBinding::destination(
            result.ty().clone(),
            ResultDestination::Place(PlaceDestination::Fixed(place)),
        ),
        ResultBinding::destination(result.ty().clone(), ResultDestination::ReturnValue(view)),
        vec![view],
    )
}

pub(super) fn expand_screma(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx: usize,
    next_effect: &mut IdSource<EffectToken>,
    callables: &CallableMap,
) -> Result<(), String> {
    let effect = graph.skeleton.blocks[bid]
        .side_effects
        .get(idx)
        .ok_or_else(|| format!("missing Screma effect {idx} in {bid:?}"))?
        .clone();
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &effect.kind else {
        return Err("Screma lowering received a different effect family".into());
    };

    let requires_serial = op.is_serial()
        || (matches!(op.state, screma::PhysicalState::Segmented(_))
            && effect.result.as_ref().is_some_and(|result| {
                result.top_level_fields().iter().any(|field| !result_is_addressable(graph, field))
            }));
    if requires_serial {
        return expand_serial_screma(graph, bid, idx, &effect, op, next_effect, callables);
    }
    if matches!(op.state, screma::PhysicalState::Segmented(_)) && op.is_map() {
        return expand_segmented_map(graph, bid, idx, &effect, op, next_effect, callables);
    }
    Err("SOAC expansion target changed after Screma selection".into())
}

fn expand_serial_screma(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx: usize,
    effect: &SideEffect<Physical>,
    op: &screma::Op<Physical>,
    next_effect: &mut IdSource<EffectToken>,
    callables: &CallableMap,
) -> Result<(), String> {
    let operands = screma::ScremaOperands::decode(op, &effect.operands, effect.result.as_ref())?;
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
    let mut carried = Vec::new();
    let mut post_sinks = Vec::with_capacity(post_count);
    let mut loop_results = Vec::with_capacity(op.result_count());
    let mut prelude = Vec::new();
    let mut place_backed_results = Vec::new();
    for post in 0..post_count {
        let field = reduction_components + post;
        let result = &result_fields[field];
        let sink = if result_is_addressable(graph, result) {
            result.clone()
        } else {
            match op.ownership(field).ok_or_else(|| format!("Screma result {field} has no ownership"))? {
                SoacOwnership::Fresh => {
                    let (sink, binding, views) =
                        fresh_result_destination(graph, result, next_effect, &mut prelude);
                    place_backed_results.extend(views);
                    rebind_physical_result(graph, result, &binding)?;
                    sink
                }
                SoacOwnership::UniqueInput => {
                    if input_nids.len() != 1 {
                        return Err("unique-input Screma result requires exactly one array input".into());
                    }
                    let output = value_binding(graph, result.ty(), input_nids[0]);
                    rebind_physical_result(graph, result, &output)?;
                    output
                }
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
                    emit_read_element(graph, body, *array, lane, array_type, element_type, next_effect)
                })
                .collect::<Vec<_>>();
            let pre_results = emit_screma_lambda(
                graph,
                body,
                callables,
                &op.form.pre,
                input_elements,
                None,
                next_effect,
            );

            let mut pre_offset = 0;
            let mut scan_offset = 0;
            let mut new_scans = Vec::with_capacity(scan_components);
            for scan in &op.form.scans {
                let width = scan.neutral.len();
                let mut arguments = carried_values[scan_offset..scan_offset + width].to_vec();
                arguments.extend(load_result_arguments(
                    graph,
                    body,
                    &pre_results[pre_offset..pre_offset + width],
                    next_effect,
                ));
                let results = emit_screma_lambda(
                    graph,
                    body,
                    callables,
                    &scan.operator,
                    arguments,
                    None,
                    next_effect,
                );
                new_scans.extend(load_result_arguments(graph, body, &results, next_effect));
                pre_offset += width;
                scan_offset += width;
            }

            let mut reduction_offset = scan_components;
            let mut new_reductions = Vec::with_capacity(reduction_components);
            for reduction in &op.form.reductions {
                let width = reduction.neutral.len();
                let mut arguments = carried_values[reduction_offset..reduction_offset + width].to_vec();
                arguments.extend(load_result_arguments(
                    graph,
                    body,
                    &pre_results[pre_offset..pre_offset + width],
                    next_effect,
                ));
                let results = emit_screma_lambda(
                    graph,
                    body,
                    callables,
                    &reduction.operator,
                    arguments,
                    None,
                    next_effect,
                );
                new_reductions.extend(load_result_arguments(graph, body, &results, next_effect));
                pre_offset += width;
                reduction_offset += width;
            }

            let mut post_inputs = new_scans
                .iter()
                .map(|value| {
                    let ty = graph.nodes[*value].ty().clone();
                    value_binding(graph, &ty, *value)
                })
                .collect::<Vec<_>>();
            post_inputs.extend_from_slice(&pre_results[pre_offset..]);
            let post_results = if op.form.post.is_identity() {
                post_inputs
            } else {
                let post_arguments = load_result_arguments(graph, body, &post_inputs, next_effect);
                emit_screma_lambda(
                    graph,
                    body,
                    callables,
                    &op.form.post,
                    post_arguments,
                    Some((&post_sinks, lane)),
                    next_effect,
                )
            };
            debug_assert_eq!(post_results.len(), post_count);

            let mut next = Vec::with_capacity(carried_values.len());
            let mut tail = body;
            for (produced, sink) in post_results.iter().zip(&post_sinks) {
                tail = emit_result_to_indexed_destination(graph, tail, produced, sink, lane, next_effect)?;
            }
            next.extend(new_scans);
            next.extend(new_reductions);
            Ok(LoopBody { tail, carried: next })
        },
    )?;
    for result in place_backed_results {
        materialize_place_backed_projections(graph, result, continuation, next_effect);
    }
    Ok(())
}

fn expand_segmented_map(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx: usize,
    effect: &SideEffect<Physical>,
    op: &screma::Op<Physical>,
    next_effect: &mut IdSource<EffectToken>,
    callables: &CallableMap,
) -> Result<(), String> {
    let screma::PhysicalState::Segmented(segment) = &op.state else {
        return Err("segmented map lowering received a non-segmented Screma".into());
    };
    let operands = screma::ScremaOperands::decode(op, &effect.operands, effect.result.as_ref())?;
    let input_nodes = operands
        .inputs()
        .map(|operand| {
            operand.operand.value().ok_or_else(|| "segmented map input is not a value or view".to_owned())
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
    let mut destinations = Vec::with_capacity(op.result_count());
    for field in 0..op.result_count() {
        let result = &result_fields[field];
        let destination = if result_is_addressable(graph, result) {
            result.clone()
        } else if op.ownership(field) == Some(SoacOwnership::UniqueInput) && input_nodes.len() == 1 {
            let destination = value_binding(graph, result.ty(), input_nodes[0]);
            rebind_physical_result(graph, result, &destination)?;
            destination
        } else {
            return Err(format!(
                "segmented map result {field} has no addressable destination"
            ));
        };
        if destination.destination_count() != result.destination_count() {
            return Err(format!(
                "segmented map result {field} has {} physical leaves but its destination has {}",
                result.destination_count(),
                destination.destination_count()
            ));
        }
        destinations.push(destination);
    }
    build_parallel_screma_map(
        graph,
        bid,
        idx,
        &segment.space,
        (first_input.0, first_input.1.clone()),
        &read_inputs,
        &op.form.pre,
        &destinations,
        next_effect,
        callables,
    )
}

/// `Scan[OutputView]`: `new_acc = func(acc, elem, ...caps); view[i] = new_acc`
/// per iteration. One loop-carried value (scalar accumulator). Writes are
/// effectful so the SOAC's `result_node` is bound to a dummy.

/// MapInto: `y = func(elem1, ..., ...caps); view[i] = y` per iteration. No
/// loop-carried state (writes are effectful); the SOAC "result" is a dummy.

pub(super) fn build_parallel_screma_map(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    effect_index: usize,
    space: &SegSpace<BindingRef>,
    length_input: (ValueId, Type<TypeName>),
    read_inputs: &[(ValueId, Type<TypeName>, Type<TypeName>)],
    pre: &screma::Lambda,
    output_views: &[ResultBinding<Type<TypeName>>],
    next_effect: &mut IdSource<EffectToken>,
    callables: &CallableMap,
) -> Result<(), String> {
    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let u32_type = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_type = Type::Constructed(TypeName::Bool, vec![]);
    let mut lane = None;
    let guarded = replace_effect_with_guarded_selection(
        graph,
        SideEffectSite {
            block,
            index: effect_index,
        },
        |graph| {
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
                .ok_or_else(|| "catalog has no structural u32-to-i32 conversion".to_owned())?;
            let thread_lane = graph.intern_pure(
                PureOp::Intrinsic {
                    id: bitcast,
                    overload_idx: 0,
                },
                smallvec![thread],
                i32_type.clone(),
                None,
            );
            lane = Some(thread_lane);
            let length = emit_seg_space_len(graph, space, &length_input, &i32_type);
            Ok(graph.intern_pure(
                PureOp::BinOp(op::BinaryOperator::Less),
                smallvec![thread_lane, length],
                bool_type,
                None,
            ))
        },
    )?;
    let body = guarded.body;
    let after = guarded.continuation;
    let lane = lane.ok_or_else(|| "guard construction did not record its lane".to_owned())?;
    let _replaced_effect = guarded.effect;

    let elements = read_inputs
        .iter()
        .map(|(array, array_type, element_type)| {
            emit_read_element(graph, body, *array, lane, array_type, element_type, next_effect)
        })
        .collect::<Vec<_>>();
    let results = emit_screma_lambda(
        graph,
        body,
        callables,
        pre,
        elements,
        Some((output_views, lane)),
        next_effect,
    );
    assert_eq!(results.len(), output_views.len());
    let mut tail = body;
    for (output, result) in output_views.iter().zip(results) {
        tail = emit_result_to_indexed_destination(graph, tail, &result, output, lane, next_effect)?;
    }
    finish_guarded_selection(graph, tail, after);
    Ok(())
}
