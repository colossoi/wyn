//! Stable inclusive filter prefixes over contiguous workgroup tiles.

use super::{emit_length, filter_primary_input, FilterLoop};
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::egir::graph_ops::{
    emit_storage_store, emit_view_load, emit_workgroup_barrier, emit_workgroup_view, intern_binop,
    intern_intrinsic, intern_storage_view, intern_u32,
};
use crate::egir::kernel_index::emit_invocation_index;
use crate::egir::soac::filter;
use crate::egir::structured_cfg::{install_loop, install_selection};
use crate::egir::types::{EGraph, EffectToken, Physical, SkeletonTerminator, ValueId, ValueKind};
use crate::flow::BlockId;
use crate::op::BinaryOperator;
use crate::ssa::types::ConstantValue;
use crate::BindingRef;
use polytype::Type;
use smallvec::smallvec;
use wyn_base::IdSource;

/// Shared ownership arithmetic for scan and scatter. At least one tile per
/// group keeps the scatter divisor defined even when the input is empty.
pub(super) fn tile_partition(
    graph: &mut EGraph<Physical>,
    len: ValueId,
    groups: ValueId,
    width: u32,
) -> Result<(ValueId, ValueId), String> {
    let ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let one = intern_u32(graph, 1, None);
    let width = intern_u32(graph, width, None);
    let min = catalog()
        .specialize_numeric(catalog().known().min, &TypeName::UInt(32))
        .ok_or_else(|| "u32 min is missing from the catalog".to_owned())?;
    let op = |graph: &mut EGraph<Physical>, operator, a, b| {
        intern_binop(graph, operator, a, b, ty.clone(), None)
    };
    // Quotient/remainder avoids overflowing len + width - 1.
    let whole = op(graph, BinaryOperator::Divide, len, width);
    let remainder = op(graph, BinaryOperator::Remainder, len, width);
    let tail = intern_intrinsic(graph, min, smallvec![remainder, one], ty.clone(), None);
    let tiles = op(graph, BinaryOperator::Add, whole, tail);
    let rounding = op(graph, BinaryOperator::Subtract, groups, one);
    let rounded = op(graph, BinaryOperator::Add, tiles, rounding);
    let per_group = op(graph, BinaryOperator::Divide, rounded, groups);
    let nonzero = intern_intrinsic(graph, min, smallvec![per_group, one], ty.clone(), None);
    let empty = op(graph, BinaryOperator::Subtract, one, nonzero);
    let per_group = op(graph, BinaryOperator::Add, per_group, empty);
    Ok((tiles, per_group))
}

pub(super) fn build(
    graph: &mut EGraph<Physical>,
    bid: BlockId,
    idx: usize,
    spec: FilterLoop<'_>,
    work: filter::WorkBuffers<BindingRef>,
    width: u32,
    effects: &mut IdSource<EffectToken>,
) -> Result<(), String> {
    if !width.is_power_of_two() {
        return Err("filter scan workgroup width must be a power of two".into());
    }
    graph.skeleton.blocks[bid].side_effects.drain(idx..);
    let ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let op = |graph: &mut EGraph<Physical>, operator, a, b| {
        intern_binop(graph, operator, a, b, ty.clone(), None)
    };
    let zero = intern_u32(graph, 0, None);
    let one = intern_u32(graph, 1, None);
    let w = intern_u32(graph, width, None);
    let last = intern_u32(graph, width - 1, None);
    let lane = emit_invocation_index(graph, catalog().known().local_id, &ty)?;
    let gid = emit_invocation_index(graph, catalog().known().thread_id, &ty)?;
    let group = op(graph, BinaryOperator::Divide, gid, w);
    let groups = emit_invocation_index(graph, catalog().known().num_workgroups, &ty)?;
    let len = emit_length(
        graph,
        filter_primary_input(&spec).0,
        &filter_primary_input(&spec).1,
        &ty,
    );
    let (tiles, per_group) = tile_partition(graph, len, groups, width)?;
    let start = op(graph, BinaryOperator::Multiply, group, per_group);
    let flags = intern_storage_view(graph, work.flags, ty.clone(), None);
    let offsets = intern_storage_view(graph, work.offsets, ty.clone(), None);
    let sums = intern_storage_view(graph, work.block_sums, ty.clone(), None);
    let shared = [
        emit_workgroup_view(graph, 0, width, ty.clone(), None),
        emit_workgroup_view(graph, 1, width, ty.clone(), None),
    ];
    let header = graph.skeleton.create_block();
    let body = graph.skeleton.create_block();
    let cont = graph.skeleton.create_block();
    let after = graph.skeleton.create_block();
    graph.skeleton.blocks[bid].term = SkeletonTerminator::Branch {
        target: header,
        args: graph.admit_flow_values([zero, zero]),
    };
    let iteration = graph.add_block_param(header, ty.clone());
    let carry = graph.add_block_param(header, ty.clone());
    // The trip count is dispatch-uniform, including empty and partial groups.
    let cond = intern_binop(
        graph,
        BinaryOperator::Less,
        iteration,
        per_group,
        bool_ty.clone(),
        None,
    );
    let exit_args = graph.admit_flow_values([carry]);
    install_loop(graph, header, cond, body, after, exit_args, cont);
    let tile = op(graph, BinaryOperator::Add, start, iteration);
    let base = op(graph, BinaryOperator::Multiply, tile, w);
    let index = op(graph, BinaryOperator::Add, base, lane);
    let in_input = intern_binop(graph, BinaryOperator::Less, tile, tiles, bool_ty.clone(), None);
    let in_bounds = intern_binop(graph, BinaryOperator::Less, index, len, bool_ty.clone(), None);
    let valid = intern_binop(
        graph,
        BinaryOperator::LogicalAnd,
        in_input,
        in_bounds,
        bool_ty.clone(),
        None,
    );
    let read = graph.skeleton.create_block();
    let pad = graph.skeleton.create_block();
    let loaded = graph.skeleton.create_block();
    install_selection(graph, body, valid, read, pad, loaded);
    let flag = emit_view_load(graph, read, flags, index, ty.clone(), effects, None);
    graph.skeleton.blocks[read].term = SkeletonTerminator::Branch {
        target: loaded,
        args: graph.admit_flow_values([flag]),
    };
    graph.skeleton.blocks[pad].term = SkeletonTerminator::Branch {
        target: loaded,
        args: graph.admit_flow_values([zero]),
    };
    let mut prefix = graph.add_block_param(loaded, ty.clone());
    emit_storage_store(graph, loaded, shared[0], lane, prefix, ty.clone(), effects, None);
    emit_workgroup_barrier(graph, loaded, effects);
    let mut block = loaded;
    let mut bank = 0;
    // Ping-pong banks let every lane read the previous step and write its next
    // prefix without a read/write race. Every barrier follows the selection merge.
    for step in 0..width.trailing_zeros() {
        let distance = intern_u32(graph, 1 << step, None);
        let has_peer = intern_binop(
            graph,
            BinaryOperator::GreaterEqual,
            lane,
            distance,
            bool_ty.clone(),
            None,
        );
        let read = graph.skeleton.create_block();
        let skip = graph.skeleton.create_block();
        let merge = graph.skeleton.create_block();
        install_selection(graph, block, has_peer, read, skip, merge);
        let peer_index = op(graph, BinaryOperator::Subtract, lane, distance);
        let peer = emit_view_load(graph, read, shared[bank], peer_index, ty.clone(), effects, None);
        let next = op(graph, BinaryOperator::Add, prefix, peer);
        graph.skeleton.blocks[read].term = SkeletonTerminator::Branch {
            target: merge,
            args: graph.admit_flow_values([next]),
        };
        graph.skeleton.blocks[skip].term = SkeletonTerminator::Branch {
            target: merge,
            args: graph.admit_flow_values([prefix]),
        };
        prefix = graph.add_block_param(merge, ty.clone());
        bank = 1 - bank;
        emit_storage_store(
            graph,
            merge,
            shared[bank],
            lane,
            prefix,
            ty.clone(),
            effects,
            None,
        );
        emit_workgroup_barrier(graph, merge, effects);
        block = merge;
    }
    let total = emit_view_load(graph, block, shared[bank], last, ty.clone(), effects, None);
    let next_carry = op(graph, BinaryOperator::Add, carry, total);
    let inclusive = op(graph, BinaryOperator::Add, carry, prefix);
    // All lanes must finish reading the tile total before the next tile writes.
    emit_workgroup_barrier(graph, block, effects);
    let write = graph.skeleton.create_block();
    let written = graph.skeleton.create_block();
    install_selection(graph, block, valid, write, written, written);
    emit_storage_store(graph, write, offsets, index, inclusive, ty.clone(), effects, None);
    graph.skeleton.blocks[write].term = SkeletonTerminator::Branch {
        target: written,
        args: vec![],
    };
    graph.skeleton.blocks[written].term = SkeletonTerminator::Branch {
        target: cont,
        args: vec![],
    };
    let next_tile = op(graph, BinaryOperator::Add, iteration, one);
    graph.skeleton.blocks[cont].term = SkeletonTerminator::Branch {
        target: header,
        args: graph.admit_flow_values([next_tile, next_carry]),
    };
    let final_count = graph.add_block_param(after, ty.clone());
    let write_sum = graph.skeleton.create_block();
    let end = graph.skeleton.create_block();
    let first = intern_binop(graph, BinaryOperator::Equal, lane, zero, bool_ty, None);
    install_selection(graph, after, first, write_sum, end, end);
    emit_storage_store(graph, write_sum, sums, group, final_count, ty, effects, None);
    graph.skeleton.blocks[write_sum].term = SkeletonTerminator::Branch {
        target: end,
        args: vec![],
    };
    graph.skeleton.blocks[end].term = SkeletonTerminator::Return(None);
    graph.replace_node_preserving_type(spec.result_node, ValueKind::Constant(ConstantValue::Bool(false)));
    Ok(())
}
