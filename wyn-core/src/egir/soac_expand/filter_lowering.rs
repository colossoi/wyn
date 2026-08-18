//! Filter expansion implementations.

use super::array_io::{emit_length, emit_read_element};
use crate::op;
use crate::ssa;
use crate::types;
use crate::BindingRef;
use crate::IdSource;

use super::*;

/// Scan: `new_acc = func(acc, elem, ...caps); out[i] = new_acc` per iteration.
/// Two loop-carried values: the output array (built via `_w_intrinsic_array_with`)
/// and the scalar accumulator.

/// Filter: per iteration `keep = pred(elem, ...caps); buf' = array_with(buf, count, elem);
/// count' = if keep then count+1 else count`. The buffer write is unconditional —
/// non-passing iterations overwrite the same slot on the next iteration that
/// advances `count`. Two loop-carried values: the buffer and the runtime count.
pub(super) struct FilterLoop {
    /// Co-iterated arrays read once per logical filter element.
    pub(super) read_inputs: Vec<(ValueId, Type<TypeName>, Type<TypeName>)>,
    /// The output element type returned by the canonical map lambda.
    pub(super) output_elem_ty: Type<TypeName>,
    pub(super) output: filter::Output<BindingRef>,
    /// `None` denotes the validated one-input identity map.
    pub(super) map_func: Option<Func<Physical>>,
    pub(super) map_captures: Vec<super::super::types::OperandRef>,
    pub(super) pred_func: Func<Physical>,
    pub(super) captures: Vec<super::super::types::OperandRef>,
    pub(super) result_node: ValueId,
}

fn filter_primary_input(spec: &FilterLoop) -> &(ValueId, Type<TypeName>, Type<TypeName>) {
    spec.read_inputs.first().expect("Filter has no input")
}

/// Read one element from every co-iterated input and invoke the canonical map
/// lambda. Identity is represented without a synthetic region.
fn filter_kept_value(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    index: ValueId,
    spec: &FilterLoop,
    next_effect: &mut IdSource<EffectToken>,
) -> ValueId {
    let elements = spec
        .read_inputs
        .iter()
        .map(|(array, array_ty, elem_ty)| {
            emit_read_element(graph, block, *array, index, array_ty, elem_ty, next_effect)
        })
        .collect::<SmallVec<[ValueId; 4]>>();
    match &spec.map_func {
        Some(function) => {
            let mut operands =
                elements.into_iter().map(|element| graph.operand_ref(element)).collect::<Vec<_>>();
            operands.extend(spec.map_captures.iter().copied());
            let result = super::call_abi::emit_call(graph, block, function, operands, None, next_effect)
                .expect("Filter map call must match its canonical boundary");
            result.single_value().expect("Filter map has one by-value result")
        }
        None => {
            debug_assert_eq!(elements.len(), 1);
            elements[0]
        }
    }
}
pub(super) fn build_filter_loop(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx_in_block: usize,
    spec: FilterLoop,
    next_effect: &mut IdSource<EffectToken>,
) {
    if let filter::Output::Runtime(runtime) = &spec.output {
        let filter::RuntimeBacking::Bound(data) = runtime.backing else {
            panic!("scheduled runtime filter has no backing storage");
        };
        build_runtime_filter_loop(graph, bid, idx_in_block, &spec, data, next_effect);
        return;
    }
    let filter::Output::Local { capacity, ownership } = &spec.output else {
        unreachable!()
    };
    let i32_ty = Type::Constructed(TypeName::Int(32), vec![]);
    let buf_ty = Type::Constructed(
        TypeName::Array,
        vec![
            spec.output_elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantComposite, vec![]),
            capacity.clone(),
            types::no_buffer(),
        ],
    );

    // Hold the suffix until the result buffer has been loaded at the head of
    // `after`; suffix effects may consume the filter result.
    let after = graph.skeleton.split_block_before_effect(bid, idx_in_block);
    let suffix = graph.skeleton.blocks[after].side_effects.drain(..).collect::<Vec<_>>();
    let buf_place = emit_alloca(graph, bid, buf_ty.clone(), next_effect, None);
    if *ownership == SoacOwnership::UniqueInput {
        emit_store(
            graph,
            bid,
            buf_place,
            filter_primary_input(&spec).0,
            next_effect,
            None,
        );
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
    Local(super::super::types::PlaceId),
    Runtime(ValueId),
}

/// Build the counted serial compaction loop shared by local and runtime
/// filters. Callers choose the index width, destination, and result format.
fn build_serial_filter_cfg(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    after: BlockId,
    spec: &FilterLoop,
    index_ty: Type<TypeName>,
    zero: ValueId,
    one: ValueId,
    sink: FilterSink,
    next_effect: &mut IdSource<EffectToken>,
) -> ValueId {
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
        args: graph.admit_flow_values([zero, zero]),
    };
    let length = emit_length(
        graph,
        filter_primary_input(spec).0,
        &filter_primary_input(spec).1,
        &index_ty,
    );
    let in_range = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Less),
        smallvec![index, length],
        bool_ty.clone(),
        None,
    );
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond: in_range,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args: graph.admit_flow_values([count]),
    };
    graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
        merge: after,
        continue_block,
    });

    let kept = filter_kept_value(graph, body, index, spec, next_effect);
    let mut pred_operands = vec![graph.operand_ref(kept)];
    pred_operands.extend(spec.captures.iter().copied());
    let predicate =
        super::call_abi::emit_call(graph, body, &spec.pred_func, pred_operands, None, next_effect)
            .expect("Filter predicate call must match its canonical boundary");
    let predicate = predicate.single_value().expect("Filter predicate has one by-value result");
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
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![count, one],
        index_ty.clone(),
        None,
    );
    graph.skeleton.blocks[then_block].term = SkeletonTerminator::Branch {
        target: selection_merge,
        args: graph.admit_flow_values([bumped_count]),
    };
    graph.skeleton.blocks[else_block].term = SkeletonTerminator::Branch {
        target: selection_merge,
        args: graph.admit_flow_values([count]),
    };

    let next_count = graph.add_block_param(selection_merge, index_ty.clone());
    graph.skeleton.blocks[selection_merge].term = SkeletonTerminator::Branch {
        target: continue_block,
        args: graph.admit_flow_values([next_count]),
    };
    let continued_count = graph.add_block_param(continue_block, index_ty.clone());
    let next_index = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![index, one],
        index_ty,
        None,
    );
    graph.skeleton.blocks[continue_block].term = SkeletonTerminator::Branch {
        target: header,
        args: graph.admit_flow_values([continued_count, next_index]),
    };
    after_count
}

fn filter_thread_index(graph: &mut EGraph<Physical>) -> ValueId {
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

pub(super) fn build_filter_flags(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx: usize,
    spec: FilterLoop,
    flags: BindingRef,
    next_effect: &mut IdSource<EffectToken>,
) {
    use super::super::graph_ops::{emit_storage_store, intern_storage_view, intern_u32};
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
        PureOp::BinOp(op::BinaryOperator::Less),
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
    let mut operands = vec![graph.operand_ref(kept)];
    operands.extend(spec.captures.iter().copied());
    let pred = super::call_abi::emit_call(graph, in_range, &spec.pred_func, operands, None, next_effect)
        .expect("Filter predicate call must match its canonical boundary");
    let pred = pred.single_value().expect("Filter predicate has one by-value result");
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
        ValueKind::Constant(ssa::types::ConstantValue::Bool(false)),
    );
}

pub(super) fn build_filter_scan(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx: usize,
    spec: FilterLoop,
    work: filter::WorkBuffers<BindingRef>,
    scan_workgroup_width: u32,
    next_effect: &mut IdSource<EffectToken>,
) {
    use super::super::graph_ops::{emit_load, emit_storage_store, intern_storage_view, intern_u32};
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
        PureOp::BinOp(op::BinaryOperator::Multiply),
        smallvec![nwg, wg_width],
        u32_ty.clone(),
        None,
    );
    let total_minus_one = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Subtract),
        smallvec![total_threads, one],
        u32_ty.clone(),
        None,
    );
    let len_plus = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![input_len, total_minus_one],
        u32_ty.clone(),
        None,
    );
    let chunk_size = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Divide),
        smallvec![len_plus, total_threads],
        u32_ty.clone(),
        None,
    );
    let raw_chunk_start = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Multiply),
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
        PureOp::BinOp(op::BinaryOperator::Subtract),
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
        args: graph.admit_flow_values([zero, zero]),
    };
    let i = graph.add_block_param(header, u32_ty.clone());
    let acc = graph.add_block_param(header, u32_ty.clone());
    let cond = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Less),
        smallvec![i, chunk_len],
        Type::Constructed(TypeName::Bool, vec![]),
        None,
    );
    graph.skeleton.blocks[header].term = SkeletonTerminator::CondBranch {
        cond,
        then_target: body,
        then_args: vec![],
        else_target: after,
        else_args: graph.admit_flow_values([acc]),
    };
    graph.skeleton.blocks[header].control_header = Some(ControlHeader::Loop {
        merge: after,
        continue_block: body,
    });
    let flags = intern_storage_view(graph, work.flags, u32_ty.clone(), None);
    let offsets = intern_storage_view(graph, work.offsets, u32_ty.clone(), None);
    let global_i = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![chunk_start, i],
        u32_ty.clone(),
        None,
    );
    let flag_place = graph.add_view_index_place(graph.view_id(flags), global_i, u32_ty.clone(), None);
    let flag = emit_load(graph, body, flag_place, u32_ty.clone(), next_effect, None);
    let next = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
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
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![i, one],
        u32_ty.clone(),
        None,
    );
    graph.skeleton.blocks[body].term = SkeletonTerminator::Branch {
        target: header,
        args: graph.admit_flow_values([next_i, next]),
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
        ValueKind::Constant(ssa::types::ConstantValue::Bool(false)),
    );
}

pub(super) fn build_filter_scatter(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx: usize,
    spec: FilterLoop,
    work: filter::WorkBuffers<BindingRef>,
    next_effect: &mut IdSource<EffectToken>,
) {
    use super::super::graph_ops::{emit_load, emit_storage_store, intern_storage_view, intern_u32};
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
        PureOp::BinOp(op::BinaryOperator::Less),
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
    let flag_place = graph.add_view_index_place(graph.view_id(flags), gid, u32_ty.clone(), None);
    let flag = emit_load(graph, in_range, flag_place, u32_ty.clone(), next_effect, None);
    let one = intern_u32(graph, 1, None);
    let keep = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Equal),
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
    let offset_place = graph.add_view_index_place(graph.view_id(offsets), gid, u32_ty.clone(), None);
    let inclusive = emit_load(graph, write, offset_place, u32_ty.clone(), next_effect, None);
    let output_index = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Subtract),
        smallvec![inclusive, one],
        u32_ty.clone(),
        None,
    );
    let kept = filter_kept_value(graph, write, gid, &spec, next_effect);
    let (out_binding, len_binding) = match &spec.output {
        filter::Output::Runtime(filter::RuntimeOutput {
            backing: filter::RuntimeBacking::Bound(data),
            length: filter::RuntimeLength::Stored(length),
            ..
        }) => (data, length),
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
    let len_place = graph.add_view_index_place(graph.view_id(len_view), zero, u32_ty.clone(), None);
    let count = emit_load(graph, bid, len_place, u32_ty.clone(), next_effect, None);
    graph.replace_pure_node(
        spec.result_node,
        PureOp::StorageView(op::PureViewSource::Storage(out_binding)),
        smallvec![zero, count],
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
fn build_runtime_filter_loop(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx_in_block: usize,
    spec: &FilterLoop,
    scratch_out: BindingRef,
    next_effect: &mut IdSource<EffectToken>,
) {
    use super::super::graph_ops::{intern_storage_view, intern_u32};

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

    if let filter::Output::Runtime(filter::RuntimeOutput {
        length: filter::RuntimeLength::Stored(length),
        ..
    }) = &spec.output
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
        PureOp::StorageView(op::PureViewSource::Storage(scratch_out)),
        smallvec![zero, after_count],
    );
}
