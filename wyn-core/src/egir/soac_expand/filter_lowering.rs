//! Filter expansion implementations.

use super::array_io::{emit_length, emit_read_element};
use super::CallableMap;
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::egir::graph_ops::{
    alloca, emit_place_index_store, emit_storage_store, intern_storage_view, intern_u32, load, store,
};
use crate::egir::kernel_index::{emit_chunk_arithmetic, emit_invocation_index};
use crate::egir::soac::filter;
use crate::egir::soac::lambda::emit_physical_call;
use crate::egir::soac::Lambda;
use crate::egir::structured_cfg::{install_loop, install_selection, replace_effect_with_continuation};
use crate::egir::types::{
    EGraph, EffectToken, Physical, PlaceId, PureOp, SideEffectKind, SideEffectSite, SkeletonTerminator,
    Soac, SoacEffect, SoacOwnership, ValueId, ValueKind,
};
use crate::flow::BlockId;
use crate::op;
use crate::ssa;
use crate::types;
use crate::BindingRef;
use polytype::Type;
use smallvec::smallvec;
use wyn_base::IdSource;

/// Scan: `new_acc = func(acc, elem, ...caps); out[i] = new_acc` per iteration.
/// Two loop-carried values: the output array (built via `_w_intrinsic_array_with`)
/// and the scalar accumulator.

/// Filter: per iteration `keep = pred(elem, ...caps); buf' = array_with(buf, count, elem);
/// count' = if keep then count+1 else count`. The buffer write is unconditional —
/// non-passing iterations overwrite the same slot on the next iteration that
/// advances `count`. Two loop-carried values: the buffer and the runtime count.
pub(super) struct FilterLoop<'a> {
    /// Co-iterated arrays read once per logical filter element.
    pub(super) read_inputs: Vec<(ValueId, Type<TypeName>, Type<TypeName>)>,
    /// The output element type returned by the canonical map lambda.
    pub(super) output_elem_ty: Type<TypeName>,
    pub(super) map: &'a Lambda,
    pub(super) predicate: &'a Lambda,
    pub(super) callables: &'a CallableMap,
    pub(super) result_node: ValueId,
}

pub(super) fn expand_filter(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx: usize,
    next_effect: &mut IdSource<EffectToken>,
    callables: &CallableMap,
) -> Result<(), String> {
    let effect = graph.skeleton.blocks[bid]
        .side_effects
        .get(idx)
        .ok_or_else(|| format!("missing Filter effect {idx} in {bid:?}"))?
        .clone();
    let SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) = &effect.kind else {
        return Err("Filter lowering received a different effect family".into());
    };
    op.body.validate()?;
    let input_count = op.body.inputs.len();
    let read_inputs = effect.operands[..input_count]
        .iter()
        .copied()
        .zip(&op.body.inputs)
        .map(|(operand, input)| {
            Ok((
                operand.value().ok_or_else(|| "Filter input is not a value or view".to_owned())?,
                input.array.clone(),
                input.element(),
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let output_elem_ty = op.body.output_element_type();
    let result_nid =
        effect.value_result().ok_or_else(|| "Filter has no by-value result root".to_owned())?;
    let spec = FilterLoop {
        read_inputs,
        output_elem_ty,
        map: &op.body.map,
        predicate: &op.body.predicate,
        callables,
        result_node: result_nid,
    };
    match &op.state {
        filter::ScheduledState::Loop { storage, .. } => {
            build_filter_loop(graph, bid, idx, spec, storage, next_effect)?
        }
        filter::ScheduledState::Pipeline { storage, plan, .. } => match plan.stage {
            filter::ParallelStage::Flags => {
                build_filter_flags(graph, bid, idx, spec, plan.buffers.flags, next_effect)?
            }
            filter::ParallelStage::Scan => build_filter_scan(
                graph,
                bid,
                idx,
                spec,
                plan.buffers,
                plan.scan_workgroup_width,
                next_effect,
            )?,
            filter::ParallelStage::Scatter => {
                build_filter_scatter(graph, bid, idx, spec, plan.buffers, *storage, next_effect)?
            }
        },
    }
    Ok(())
}

fn filter_primary_input<'a>(spec: &'a FilterLoop<'_>) -> &'a (ValueId, Type<TypeName>, Type<TypeName>) {
    spec.read_inputs.first().expect("Filter has no input")
}

/// Read one element from every co-iterated input and invoke the canonical map
/// lambda. Identity is represented without a synthetic region.
fn filter_kept_value(
    graph: &mut EGraph<Physical>,
    block: BlockId,
    index: ValueId,
    spec: &FilterLoop<'_>,
    next_effect: &mut IdSource<EffectToken>,
) -> Result<ValueId, String> {
    let elements = spec
        .read_inputs
        .iter()
        .map(|(array, array_ty, elem_ty)| {
            emit_read_element(graph, block, *array, index, array_ty, elem_ty, next_effect)
        })
        .collect::<Vec<_>>();
    let results = emit_physical_call(
        graph,
        block,
        spec.callables,
        spec.map,
        elements,
        None,
        next_effect,
    )?;
    results[0].single_value().ok_or_else(|| "Filter map has no single by-value result".to_owned())
}
pub(super) fn build_filter_loop(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx_in_block: usize,
    spec: FilterLoop<'_>,
    output: &filter::Output<BindingRef>,
    next_effect: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    if let filter::Output::Runtime(runtime) = output {
        return build_runtime_filter_loop(graph, bid, idx_in_block, &spec, *runtime, next_effect);
    }
    let filter::Output::Local { capacity, ownership } = output else {
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
    let replacement = replace_effect_with_continuation(
        graph,
        SideEffectSite {
            block: bid,
            index: idx_in_block,
        },
    )?;
    let after = replacement.continuation;
    let _replaced_effect = replacement.effect;
    let suffix = graph.skeleton.blocks[after].side_effects.drain(..).collect::<Vec<_>>();
    let buf_place = alloca(graph, buf_ty.clone(), next_effect, None).append_to(&mut graph.skeleton, bid);
    if *ownership == SoacOwnership::UniqueInput {
        store(buf_place, filter_primary_input(&spec).0, next_effect, None)
            .append_to(&mut graph.skeleton, bid);
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
    )?;

    let loaded = load(graph, buf_place, buf_ty, next_effect, None).append_to(&mut graph.skeleton, after);
    graph.skeleton.blocks[after].side_effects.extend(suffix);
    graph.replace_pure_node(spec.result_node, PureOp::Tuple(2), smallvec![loaded, after_count]);
    Ok(())
}

#[derive(Clone, Copy)]
enum FilterSink {
    Local(PlaceId),
    Runtime(ValueId),
}

/// Build the counted serial compaction loop shared by local and runtime
/// filters. Callers choose the index width, destination, and result format.
fn build_serial_filter_cfg(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    after: BlockId,
    spec: &FilterLoop<'_>,
    index_ty: Type<TypeName>,
    zero: ValueId,
    one: ValueId,
    sink: FilterSink,
    next_effect: &mut IdSource<EffectToken>,
) -> Result<ValueId, String> {
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
    let exit_args = graph.admit_flow_values([count]);
    install_loop(graph, header, in_range, body, after, exit_args, continue_block);

    let kept = filter_kept_value(graph, body, index, spec, next_effect)?;
    let predicate = emit_physical_call(
        graph,
        body,
        spec.callables,
        spec.predicate,
        vec![kept],
        None,
        next_effect,
    )?;
    let predicate = predicate[0]
        .single_value()
        .ok_or_else(|| "Filter predicate has no single by-value result".to_owned())?;
    install_selection(graph, body, predicate, then_block, else_block, selection_merge);

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
    Ok(after_count)
}

pub(super) fn build_filter_flags(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx: usize,
    spec: FilterLoop<'_>,
    flags: BindingRef,
    next_effect: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    graph.skeleton.blocks[bid].side_effects.drain(idx..);
    let after = graph.skeleton.create_block();
    let in_range = graph.skeleton.create_block();
    let keep = graph.skeleton.create_block();
    let drop = graph.skeleton.create_block();
    let pred_merge = graph.skeleton.create_block();
    graph.skeleton.blocks[after].term = SkeletonTerminator::Return(None);
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let gid = emit_invocation_index(graph, catalog().known().thread_id, &u32_ty)?;
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
    install_selection(graph, bid, bounded, in_range, after, after);
    let kept = filter_kept_value(graph, in_range, gid, &spec, next_effect)?;
    let pred = emit_physical_call(
        graph,
        in_range,
        spec.callables,
        spec.predicate,
        vec![kept],
        None,
        next_effect,
    )?;
    let pred = pred[0]
        .single_value()
        .ok_or_else(|| "Filter predicate has no single by-value result".to_owned())?;
    install_selection(graph, in_range, pred, keep, drop, pred_merge);
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
    Ok(())
}

pub(super) fn build_filter_scan(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx: usize,
    spec: FilterLoop<'_>,
    work: filter::WorkBuffers<BindingRef>,
    scan_workgroup_width: u32,
    next_effect: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    graph.skeleton.blocks[bid].side_effects.drain(idx..);
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let after = graph.skeleton.create_block();
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let zero = intern_u32(graph, 0, None);
    let one = intern_u32(graph, 1, None);
    let input_len = emit_length(
        graph,
        filter_primary_input(&spec).0,
        &filter_primary_input(&spec).1,
        &u32_ty,
    );
    let (gid, chunk_start, chunk_len) = emit_chunk_arithmetic(graph, scan_workgroup_width, input_len)?;
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
    let exit_args = graph.admit_flow_values([acc]);
    install_loop(graph, header, cond, body, after, exit_args, body);
    let flags = intern_storage_view(graph, work.flags, u32_ty.clone(), None);
    let offsets = intern_storage_view(graph, work.offsets, u32_ty.clone(), None);
    let global_i = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Add),
        smallvec![chunk_start, i],
        u32_ty.clone(),
        None,
    );
    let flag_place = graph.add_view_index_place(graph.view_id(flags), global_i, u32_ty.clone(), None);
    let flag =
        load(graph, flag_place, u32_ty.clone(), next_effect, None).append_to(&mut graph.skeleton, body);
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
    Ok(())
}

pub(super) fn build_filter_scatter(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx: usize,
    spec: FilterLoop<'_>,
    work: filter::WorkBuffers<BindingRef>,
    storage: filter::RuntimeStorage<BindingRef>,
    next_effect: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    let replacement = replace_effect_with_continuation(
        graph,
        SideEffectSite {
            block: bid,
            index: idx,
        },
    )?;
    let after = replacement.continuation;
    let _replaced_effect = replacement.effect;
    let in_range = graph.skeleton.create_block();
    let write = graph.skeleton.create_block();
    let skip = graph.skeleton.create_block();
    let merge = graph.skeleton.create_block();
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let gid = emit_invocation_index(graph, catalog().known().thread_id, &u32_ty)?;
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
    install_selection(graph, bid, bounded, in_range, after, after);
    let flags = intern_storage_view(graph, work.flags, u32_ty.clone(), None);
    let offsets = intern_storage_view(graph, work.offsets, u32_ty.clone(), None);
    let flag_place = graph.add_view_index_place(graph.view_id(flags), gid, u32_ty.clone(), None);
    let flag =
        load(graph, flag_place, u32_ty.clone(), next_effect, None).append_to(&mut graph.skeleton, in_range);
    let one = intern_u32(graph, 1, None);
    let keep = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Equal),
        smallvec![flag, one],
        bool_ty,
        None,
    );
    install_selection(graph, in_range, keep, write, skip, merge);
    let offset_place = graph.add_view_index_place(graph.view_id(offsets), gid, u32_ty.clone(), None);
    let inclusive =
        load(graph, offset_place, u32_ty.clone(), next_effect, None).append_to(&mut graph.skeleton, write);
    let output_index = graph.intern_pure(
        PureOp::BinOp(op::BinaryOperator::Subtract),
        smallvec![inclusive, one],
        u32_ty.clone(),
        None,
    );
    let kept = filter_kept_value(graph, write, gid, &spec, next_effect)?;
    let filter::RuntimeStorage {
        data: out_binding,
        length: len_binding,
    } = storage;
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
    let count =
        load(graph, len_place, u32_ty.clone(), next_effect, None).append_to(&mut graph.skeleton, bid);
    graph.replace_pure_node(
        spec.result_node,
        PureOp::StorageView(op::PureViewSource::Storage(out_binding)),
        smallvec![zero, count],
    );
    Ok(())
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
    spec: &FilterLoop<'_>,
    output: filter::RuntimeOutput<BindingRef>,
    next_effect: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    let filter::RuntimeBacking::Bound(scratch_out) = output.backing else {
        panic!("scheduled runtime filter has no backing storage");
    };
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let scratch_view = intern_storage_view(graph, scratch_out, spec.output_elem_ty.clone(), None);
    let replacement = replace_effect_with_continuation(
        graph,
        SideEffectSite {
            block: bid,
            index: idx_in_block,
        },
    )?;
    let after = replacement.continuation;
    let _replaced_effect = replacement.effect;
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
    )?;

    if let filter::RuntimeLength::Stored(length) = output.length {
        let length_view = intern_storage_view(graph, length, u32_ty.clone(), None);
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
    Ok(())
}
