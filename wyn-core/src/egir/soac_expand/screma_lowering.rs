//! Screma expansion helpers.

use super::array_io::{emit_read_element, emit_seg_space_len};
use super::*;

pub(super) fn emit_screma_lambda(
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

/// `Scan[OutputView]`: `new_acc = func(acc, elem, ...caps); view[i] = new_acc`
/// per iteration. One loop-carried value (scalar accumulator). Writes are
/// effectful so the SOAC's `result_node` is bound to a dummy.

/// MapInto: `y = func(elem1, ..., ...caps); view[i] = y` per iteration. No
/// loop-carried state (writes are effectful); the SOAC "result" is a dummy.

pub(super) fn build_parallel_screma_map(
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
