//! Screma expansion helpers.

use super::array_io::{emit_read_element, emit_seg_space_len};
use super::*;
use crate::op;
use crate::IdSource;

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
                let abi = super::super::types::by_value_function_result::<WynLanguage>(ty.clone());
                super::super::graph_ops::bind_by_value_result(graph, &abi, argument)
            })
            .collect();
    }
    let body = lambda.seg_body().expect("non-identity Screma lambda has a region");
    let callee = callables.get(&body.region).expect("Screma lambda callable boundary");
    let mut operands = arguments.drain(..).map(|argument| graph.operand_ref(argument)).collect::<Vec<_>>();
    operands.extend(body.captures.iter().copied());
    let result =
        super::call_abi::emit_call(graph, block, callee, operands, mapped_destinations, next_effect)
            .expect("Screma lambda call must match its canonical boundary");
    super::super::soac::lambda::logical_result_fields(&result, &lambda.result_types)
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
) {
    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let u32_type = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_type = Type::Constructed(TypeName::Bool, vec![]);
    let after = graph.skeleton.split_block_before_effect(block, effect_index);
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
        PureOp::BinOp(op::BinaryOperator::Less),
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
        tail = super::emit_mapped_result_stores(graph, tail, lane, &result, output, next_effect);
    }
    graph.skeleton.blocks[tail].term = SkeletonTerminator::Branch {
        target: after,
        args: vec![],
    };
}
