//! Screma expansion helpers.

use super::array_io::{emit_read_element, emit_seg_space_len};
use super::*;
use crate::egir::graph_ops::{bind_by_value_result, emit_result_to_indexed_destination};
use crate::egir::physical_call_abi::emit_call;
use crate::egir::soac::lambda::logical_result_fields;
use crate::egir::structured_cfg::{finish_guarded_selection, replace_effect_with_guarded_selection};
use crate::egir::types::{by_value_function_result, SideEffectSite};
use crate::op;
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
