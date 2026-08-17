//! Histogram expansion implementations.

use super::array_io::{
    emit_flat_domain_coordinates, emit_read_ranked_coordinates, emit_read_ranked_element,
    emit_seg_space_dimensions, emit_seg_space_len,
};
use super::loop_builder::{expand_loop, LoopBody};
use super::screma_lowering::emit_screma_lambda;
use super::*;

/// Inputs for canonical serial histogram expansion. The form owns all bucket
/// result routing and operation metadata; the loop supplies co-iterated input
/// elements and a domain length.
pub(super) struct HistLoop {
    pub(super) form: hist::HistForm,
    /// `(array_nid, array_type, leaf_type, rank)` per input.
    pub(super) read_inputs: Vec<(ValueId, Type<TypeName>, Type<TypeName>, Vec<u8>, ArrayLayout)>,
    /// Loop bound source -- the first input `(nid, array_type)`.
    pub(super) len_input: (ValueId, Type<TypeName>),
    pub(super) result_node: ValueId,
}

/// Convert one operation's multidimensional index to the row-major scalar
/// offset used by Wyn storage views.
fn flatten_hist_index(graph: &mut EGraph<Physical>, indices: &[ValueId], shape: &[ValueId]) -> ValueId {
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

fn hist_index_in_bounds(graph: &mut EGraph<Physical>, indices: &[ValueId], shape: &[ValueId]) -> ValueId {
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
    graph: &mut EGraph<Physical>,
    block: BlockId,
    next: BlockId,
    place: PlaceId,
    incoming: ValueId,
    value_type: Type<TypeName>,
    operation: &hist::HistOp,
    plan: hist::AtomicUpdate,
    next_effect: &mut crate::IdSource<EffectToken>,
    regions: &CallableMap,
) {
    use super::super::graph_ops::emit_atomic;
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
                args: graph.admit_flow_values([initial, initially_retry]),
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

            let result = emit_screma_lambda(
                graph,
                attempt,
                regions,
                operator,
                vec![expected, incoming],
                None,
                next_effect,
            );
            let desired = super::super::soac::lambda::result_argument_values(graph, &result)[0];
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
                args: graph.admit_flow_values([observed, retry_after_attempt]),
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
pub(super) fn build_hist_atomic(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    effect_index: usize,
    spec: HistLoop,
    space: &SegSpace<BindingRef>,
    atomic_operations: &[hist::AtomicUpdate],
    next_effect: &mut crate::IdSource<EffectToken>,
    regions: &CallableMap,
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
        ValueKind::Constant(crate::ssa::types::ConstantValue::Bool(false)),
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
        .map(|(array, array_type, element_type, dimensions, layout)| {
            emit_read_ranked_element(
                graph,
                body,
                *array,
                lane,
                array_type,
                element_type,
                u8::try_from(dimensions.len()).expect("SOAC input rank exceeds u8"),
                layout,
                next_effect,
            )
        })
        .collect::<Vec<_>>();
    let bucket_results =
        emit_screma_lambda(graph, body, regions, &form.bucket, arguments, None, next_effect);
    let bucket_values = super::super::soac::lambda::result_argument_values(graph, &bucket_results);
    debug_assert_eq!(
        bucket_values.len(),
        form.guard_count() + form.index_count() + form.value_count()
    );
    let (guards, bucket_values) = bucket_values.split_at(form.guard_count());
    let (indices, values) = bucket_values.split_at(form.index_count());
    let mut guard_offset = 0;
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
        let in_bounds = hist_index_in_bounds(graph, operation_indices, &operation.shape);
        let valid = if matches!(operation.emission, hist::Emission::Guarded) {
            let active = guards[guard_offset];
            guard_offset += 1;
            graph.intern_pure(
                PureOp::BinOp(crate::op::BinaryOperator::LogicalAnd),
                smallvec![active, in_bounds],
                Type::Constructed(TypeName::Bool, vec![]),
                None,
            )
        } else {
            in_bounds
        };
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
        let place =
            graph.add_view_index_place(operation.destinations[0], bucket_index, value_type.clone(), None);
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
/// Canonical order: guards for guarded operations, all operation indices, then
/// all operation values. A multi-component reducer is invoked once with all
/// previous components and all incoming components, then its results are
/// stored componentwise.
pub(super) fn build_hist_loop(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx_in_block: usize,
    spec: HistLoop,
    next_effect: &mut crate::IdSource<EffectToken>,
    regions: &CallableMap,
) {
    use super::super::graph_ops::{emit_storage_store, emit_view_load};
    let HistLoop {
        form,
        read_inputs,
        len_input,
        result_node,
    } = spec;

    let results = [LoopResultBinding {
        result: graph.value_result(result_node),
        source: LoopResultSource::ConstantFalse,
    }];

    expand_loop(
        graph,
        bid,
        idx_in_block,
        &len_input,
        &[],
        &results,
        next_effect,
        true,
        move |graph, next_effect, blk, i_nid, _carried| {
            let mut arguments = Vec::with_capacity(read_inputs.len());
            for (array, array_type, element_type, dimensions, layout) in &read_inputs {
                arguments.push(emit_read_ranked_element(
                    graph,
                    blk,
                    *array,
                    i_nid,
                    array_type,
                    element_type,
                    u8::try_from(dimensions.len()).expect("SOAC input rank exceeds u8"),
                    layout,
                    next_effect,
                ));
            }
            let bucket_results =
                emit_screma_lambda(graph, blk, regions, &form.bucket, arguments, None, next_effect);
            let bucket_values = super::super::soac::lambda::result_argument_values(graph, &bucket_results);
            debug_assert_eq!(
                bucket_values.len(),
                form.guard_count() + form.index_count() + form.value_count()
            );
            let (guards, bucket_values) = bucket_values.split_at(form.guard_count());
            let (indices, values) = bucket_values.split_at(form.index_count());
            let mut guard_offset = 0;
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
                let in_bounds = hist_index_in_bounds(graph, operation_indices, &operation.shape);
                let valid = if matches!(operation.emission, hist::Emission::Guarded) {
                    let active = guards[guard_offset];
                    guard_offset += 1;
                    graph.intern_pure(
                        PureOp::BinOp(crate::op::BinaryOperator::LogicalAnd),
                        smallvec![active, in_bounds],
                        Type::Constructed(TypeName::Bool, vec![]),
                        None,
                    )
                } else {
                    in_bounds
                };
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
                    hist::Update::BucketInsert { .. } => {
                        unreachable!("bucket insertion must use its dedicated serial or pipeline lowering")
                    }
                    hist::Update::Reduce { operator, .. } => {
                        let mut reducer_arguments = Vec::with_capacity(operation.value_count() * 2);
                        for (destination, value_type) in operation.destinations.iter().zip(value_types) {
                            reducer_arguments.push(emit_view_load(
                                graph,
                                update,
                                destination.value(),
                                bucket_index,
                                value_type.clone(),
                                next_effect,
                                None,
                            ));
                        }
                        reducer_arguments.extend_from_slice(operation_values);
                        let results = emit_screma_lambda(
                            graph,
                            update,
                            regions,
                            operator,
                            reducer_arguments,
                            None,
                            next_effect,
                        );
                        super::super::soac::lambda::result_argument_values(graph, &results)
                    }
                };
                debug_assert_eq!(updated_values.len(), operation.destinations.len());
                for ((destination, value_type), updated) in
                    operation.destinations.iter().zip(value_types).zip(updated_values)
                {
                    emit_storage_store(
                        graph,
                        update,
                        destination.value(),
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

fn emit_thread_lane(graph: &mut EGraph<Physical>) -> ValueId {
    emit_thread_coordinate(graph, catalog().known().thread_id)
}

fn emit_thread_coordinate(graph: &mut EGraph<Physical>, builtin: crate::builtins::BuiltinId) -> ValueId {
    let u32_type = Type::Constructed(TypeName::UInt(32), vec![]);
    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let thread = graph.intern_pure(
        PureOp::Intrinsic {
            id: builtin,
            overload_idx: 0,
        },
        smallvec![],
        u32_type,
        None,
    );
    let convert = catalog()
        .conversion(&TypeName::Int(32), &TypeName::UInt(32))
        .expect("catalog has structural u32-to-i32 conversion");
    graph.intern_pure(
        PureOp::Intrinsic {
            id: convert,
            overload_idx: 0,
        },
        smallvec![thread],
        i32_type,
        None,
    )
}

fn emit_dispatch_axis_extent(
    graph: &mut EGraph<Physical>,
    dimensions: &[ValueId],
    i32_type: &Type<TypeName>,
) -> ValueId {
    let one = graph.intern_pure(PureOp::Int("1".into()), smallvec![], i32_type.clone(), None);
    dimensions.iter().copied().fold(one, |product, dimension| {
        graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Multiply),
            smallvec![product, dimension],
            i32_type.clone(),
            None,
        )
    })
}

fn emit_ranked_bucket_coordinates(
    graph: &mut EGraph<Physical>,
    dimensions: &[ValueId],
    topology: &hist::DispatchTopology,
    lanes: [ValueId; 3],
    i32_type: &Type<TypeName>,
    bool_type: &Type<TypeName>,
) -> (Vec<ValueId>, ValueId) {
    let mut coordinates = vec![None; dimensions.len()];
    let mut in_range = None;
    for (axis, lane) in topology.axes.iter().zip(lanes) {
        assert!(
            axis.start <= axis.end && axis.end <= dimensions.len(),
            "planned bucket dispatch range is outside the logical domain"
        );
        let axis_dimensions = &dimensions[axis.start..axis.end];
        let axis_extent = emit_dispatch_axis_extent(graph, axis_dimensions, i32_type);
        let axis_in_range = graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Less),
            smallvec![lane, axis_extent],
            bool_type.clone(),
            None,
        );
        in_range = Some(match in_range {
            None => axis_in_range,
            Some(previous) => graph.intern_pure(
                PureOp::BinOp(crate::op::BinaryOperator::LogicalAnd),
                smallvec![previous, axis_in_range],
                bool_type.clone(),
                None,
            ),
        });
        for (offset, coordinate) in
            emit_flat_domain_coordinates(graph, lane, axis_dimensions, i32_type).into_iter().enumerate()
        {
            coordinates[axis.start + offset] = Some(coordinate);
        }
    }
    let coordinates = coordinates
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .expect("planned bucket dispatch topology covers every logical dimension");
    (
        coordinates,
        in_range.expect("bucket dispatch topology has three physical axes"),
    )
}

fn emit_overflow_flag(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    overflow: ViewId,
    next_effect: &mut crate::IdSource<EffectToken>,
) {
    use crate::ssa::types::AtomicOp;
    let u32_type = Type::Constructed(TypeName::UInt(32), vec![]);
    let zero = super::super::graph_ops::intern_u32(graph, 0, None);
    let one = super::super::graph_ops::intern_u32(graph, 1, None);
    let place = graph.add_view_index_place(overflow, zero, u32_type.clone(), None);
    super::super::graph_ops::emit_atomic(
        graph,
        block,
        place,
        AtomicOp::Exchange,
        &[one],
        u32_type,
        next_effect,
        None,
    );
}

pub(super) fn build_bucket_init(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    effect_index: usize,
    spec: HistLoop,
    next_effect: &mut crate::IdSource<EffectToken>,
) {
    use crate::ssa::types::AtomicOp;
    let after = graph.skeleton.split_block_before_effect(block, effect_index);
    graph.replace_node_preserving_type(
        spec.result_node,
        ValueKind::Constant(crate::ssa::types::ConstantValue::Bool(false)),
    );
    let operation = spec.form.operations.first().expect("bucket insertion has one operation");
    let hist::Update::BucketInsert { counts, overflow, .. } = operation.update else {
        unreachable!("bucket init requires bucket insertion")
    };
    let lane = emit_thread_lane(graph);
    let bool_type = Type::Constructed(TypeName::Bool, vec![]);
    let in_range = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Less),
        smallvec![lane, operation.shape[0]],
        bool_type.clone(),
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

    let u32_type = Type::Constructed(TypeName::UInt(32), vec![]);
    let zero_u32 = super::super::graph_ops::intern_u32(graph, 0, None);
    let count_place = graph.add_view_index_place(counts, lane, u32_type.clone(), None);
    super::super::graph_ops::emit_atomic(
        graph,
        body,
        count_place,
        AtomicOp::Exchange,
        &[zero_u32],
        u32_type,
        next_effect,
        None,
    );
    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let zero_i32 = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_type, None);
    let first = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Equal),
        smallvec![lane, zero_i32],
        bool_type,
        None,
    );
    let clear_overflow = graph.skeleton.create_block();
    let done = graph.skeleton.create_block();
    graph.skeleton.blocks[body].term = SkeletonTerminator::CondBranch {
        cond: first,
        then_target: clear_overflow,
        then_args: vec![],
        else_target: done,
        else_args: vec![],
    };
    graph.skeleton.blocks[body].control_header = Some(ControlHeader::Selection { merge: done });
    let overflow_place = graph.add_view_index_place(
        overflow,
        zero_u32,
        Type::Constructed(TypeName::UInt(32), vec![]),
        None,
    );
    super::super::graph_ops::emit_atomic(
        graph,
        clear_overflow,
        overflow_place,
        AtomicOp::Exchange,
        &[zero_u32],
        Type::Constructed(TypeName::UInt(32), vec![]),
        next_effect,
        None,
    );
    graph.skeleton.blocks[clear_overflow].term = SkeletonTerminator::Branch {
        target: done,
        args: vec![],
    };
    graph.skeleton.blocks[done].term = SkeletonTerminator::Branch {
        target: after,
        args: vec![],
    };
}

pub(super) fn build_bucket_insert(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    effect_index: usize,
    spec: HistLoop,
    space: &SegSpace<BindingRef>,
    topology: Option<&hist::DispatchTopology>,
    next_effect: &mut crate::IdSource<EffectToken>,
    regions: &CallableMap,
) {
    use crate::ssa::types::AtomicOp;
    let after = graph.skeleton.split_block_before_effect(block, effect_index);
    graph.replace_node_preserving_type(
        spec.result_node,
        ValueKind::Constant(crate::ssa::types::ConstantValue::Bool(false)),
    );
    let operation = spec.form.operations.first().expect("bucket insertion has one operation");
    let hist::Update::BucketInsert {
        counts,
        overflow,
        capacity,
        ..
    } = operation.update
    else {
        unreachable!("bucket insert stage requires bucket insertion")
    };
    let i32_type = Type::Constructed(TypeName::Int(32), vec![]);
    let bool_type = Type::Constructed(TypeName::Bool, vec![]);
    let domain_dimensions = emit_seg_space_dimensions(graph, space, &spec.len_input, &i32_type);
    let physical_lanes = [
        emit_thread_coordinate(graph, catalog().known().thread_id),
        emit_thread_coordinate(graph, catalog().known().thread_id_y),
        emit_thread_coordinate(graph, catalog().known().thread_id_z),
    ];
    let mut lanes = physical_lanes;
    let (coordinate_block, grid_loop) = if let Some(stride) = topology.and_then(|plan| plan.grid_stride) {
        let header = graph.skeleton.create_block();
        let continuation = graph.skeleton.create_block();
        let lane = graph.add_block_param(header, i32_type.clone());
        lanes[stride.axis] = lane;
        graph.skeleton.blocks[block].term = SkeletonTerminator::Branch {
            target: header,
            args: graph.admit_flow_values([physical_lanes[stride.axis]]),
        };
        (header, Some((continuation, lane, stride)))
    } else {
        (block, None)
    };
    let (domain_coordinates, mut in_range) = if let Some(topology) = topology {
        emit_ranked_bucket_coordinates(graph, &domain_dimensions, topology, lanes, &i32_type, &bool_type)
    } else {
        let lane = physical_lanes[0];
        let coordinates = emit_flat_domain_coordinates(graph, lane, &domain_dimensions, &i32_type);
        let length = emit_seg_space_len(graph, space, &spec.len_input, &i32_type);
        let in_range = graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Less),
            smallvec![lane, length],
            bool_type.clone(),
            None,
        );
        (coordinates, in_range)
    };
    if let Some((_, lane, _)) = grid_loop {
        let zero = graph.intern_pure(PureOp::Int("0".into()), smallvec![], i32_type.clone(), None);
        let did_not_wrap = graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::GreaterEqual),
            smallvec![lane, zero],
            bool_type.clone(),
            None,
        );
        in_range = graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::LogicalAnd),
            smallvec![did_not_wrap, in_range],
            bool_type.clone(),
            None,
        );
    }
    let body = graph.skeleton.create_block();
    graph.skeleton.blocks[coordinate_block].term = SkeletonTerminator::CondBranch {
        cond: in_range,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args: vec![],
    };
    let work_done = if let Some((continuation, lane, stride)) = grid_loop {
        graph.skeleton.blocks[coordinate_block].control_header = Some(ControlHeader::Loop {
            merge: after,
            continue_block: continuation,
        });
        let stride_value = graph.intern_pure(
            PureOp::Int(stride.items.to_string()),
            smallvec![],
            i32_type.clone(),
            None,
        );
        let next_lane = graph.intern_pure(
            PureOp::BinOp(crate::op::BinaryOperator::Add),
            smallvec![lane, stride_value],
            i32_type.clone(),
            None,
        );
        graph.skeleton.blocks[continuation].term = SkeletonTerminator::Branch {
            target: coordinate_block,
            args: graph.admit_flow_values([next_lane]),
        };
        continuation
    } else {
        graph.skeleton.blocks[coordinate_block].control_header =
            Some(ControlHeader::Selection { merge: after });
        after
    };

    let arguments = spec
        .read_inputs
        .iter()
        .map(|(array, array_type, leaf_type, dimensions, layout)| {
            let input_coordinates = dimensions
                .iter()
                .map(|dimension| {
                    domain_coordinates
                        .get(usize::from(*dimension))
                        .copied()
                        .expect("SOAC input dimension is outside its domain")
                })
                .collect::<Vec<_>>();
            emit_read_ranked_coordinates(
                graph,
                body,
                *array,
                &input_coordinates,
                array_type,
                leaf_type,
                layout,
                next_effect,
            )
        })
        .collect::<Vec<_>>();
    let results = emit_screma_lambda(
        graph,
        body,
        regions,
        &spec.form.bucket,
        arguments,
        None,
        next_effect,
    );
    let results = super::super::soac::lambda::result_argument_values(graph, &results);
    let [active, key, value] = results.as_slice() else {
        unreachable!("guarded bucket insertion envelope returns active, key, and value")
    };
    let check_bucket = graph.skeleton.create_block();
    graph.skeleton.blocks[body].term = SkeletonTerminator::CondBranch {
        cond: *active,
        then_target: check_bucket,
        else_args: vec![],
        else_target: work_done,
        then_args: vec![],
    };
    graph.skeleton.blocks[body].control_header = Some(ControlHeader::Selection { merge: work_done });

    let valid_key = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Less),
        smallvec![*key, operation.shape[0]],
        bool_type.clone(),
        None,
    );
    let allocate = graph.skeleton.create_block();
    let invalid = graph.skeleton.create_block();
    graph.skeleton.blocks[check_bucket].term = SkeletonTerminator::CondBranch {
        cond: valid_key,
        then_target: allocate,
        then_args: vec![],
        else_target: invalid,
        else_args: vec![],
    };
    graph.skeleton.blocks[check_bucket].control_header =
        Some(ControlHeader::Selection { merge: work_done });
    emit_overflow_flag(graph, invalid, overflow, next_effect);
    graph.skeleton.blocks[invalid].term = SkeletonTerminator::Branch {
        target: work_done,
        args: vec![],
    };

    let u32_type = Type::Constructed(TypeName::UInt(32), vec![]);
    let one = super::super::graph_ops::intern_u32(graph, 1, None);
    let count_place = graph.add_view_index_place(counts, *key, u32_type.clone(), None);
    let slot_u32 = super::super::graph_ops::emit_atomic(
        graph,
        allocate,
        count_place,
        AtomicOp::Add,
        &[one],
        u32_type,
        next_effect,
        None,
    );
    let convert = catalog()
        .conversion(&TypeName::Int(32), &TypeName::UInt(32))
        .expect("catalog has structural u32-to-i32 conversion");
    let slot = graph.intern_pure(
        PureOp::Intrinsic {
            id: convert,
            overload_idx: 0,
        },
        smallvec![slot_u32],
        i32_type,
        None,
    );
    let has_capacity = graph.intern_pure(
        PureOp::BinOp(crate::op::BinaryOperator::Less),
        smallvec![slot, capacity],
        bool_type,
        None,
    );
    let write = graph.skeleton.create_block();
    let full = graph.skeleton.create_block();
    graph.skeleton.blocks[allocate].term = SkeletonTerminator::CondBranch {
        cond: has_capacity,
        then_target: write,
        then_args: vec![],
        else_target: full,
        else_args: vec![],
    };
    graph.skeleton.blocks[allocate].control_header = Some(ControlHeader::Selection { merge: work_done });
    emit_overflow_flag(graph, full, overflow, next_effect);
    graph.skeleton.blocks[full].term = SkeletonTerminator::Branch {
        target: work_done,
        args: vec![],
    };

    let destination = operation.destinations[0];
    let destination_ty = graph.nodes[destination.value()].ty.clone();
    let row_ty = destination_ty.elem_type().expect("bucket destination must have rank two").clone();
    let row = graph.add_view_index_place(destination, *key, row_ty.clone(), None);
    let leaf_ty = row_ty.elem_type().expect("bucket destination must have rank two").clone();
    let place = graph.add_index_place(row, slot, leaf_ty, None);
    emit_store(graph, write, place, *value, next_effect, None);
    graph.skeleton.blocks[write].term = SkeletonTerminator::Branch {
        target: work_done,
        args: vec![],
    };
}

pub(super) fn build_bucket_finish(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    effect_index: usize,
    result_node: ValueId,
) {
    let after = graph.skeleton.split_block_before_effect(block, effect_index);
    graph.replace_node_preserving_type(
        result_node,
        ValueKind::Constant(crate::ssa::types::ConstantValue::Bool(false)),
    );
    graph.skeleton.blocks[block].term = SkeletonTerminator::Branch {
        target: after,
        args: vec![],
    };
}
