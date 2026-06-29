//! EGIR-side SOAC parallelization in two explicit stages:
//!
//! 1. `reify` turns recognized entry-tail Scremas into semantic `SegMap`,
//!    `SegRed`, or `SegScan` operations.
//! 2. `lower` schedules those operations: maps remain segmented for
//!    `soac_expand`, reductions become chunked phase-1 plus tree-reduce phase
//!    2 entries, and scans currently return to the serial Screma fallback.
use crate::LookupMap;

use polytype::Type;
use smallvec::smallvec;

use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ControlHeader, InstKind};
use crate::BindingRef;

use super::graph_ops;
use super::program::{EgirEntry, EgirInner};
use super::types::{
    EGraph, NodeId, PendingScremaAccumulator, PendingSoac, PureOp, SegBinOp, SegLevel, SegOpKind, SegSpace,
    SideEffectKind, SkeletonTerminator, SoacDestination,
};
use crate::tlc::parallelize::EntryRecognition;

/// Per-workgroup width of a synthesized phase-2 tree reduce.
pub const PHASE2_WIDTH: u32 = 256;
/// Per-workgroup width used to chunk a phase-1 partial reduce.
const REDUCE_PHASE1_WIDTH: u32 = 64;

/// Reify each recognized entry's tail Screma as a semantic segmented op.
/// This pass performs no scheduling and allocates no bindings.
pub fn reify(inner: &mut EgirInner, recognitions: &LookupMap<String, EntryRecognition>) {
    for entry in inner.entry_points.iter_mut() {
        if recognitions.contains_key(&entry.name) {
            reify_tail_soac(entry);
        }
    }
}

/// Lower semantic segmented operations into executable kernel entries.
/// Pointwise `SegMap`s remain for `soac_expand`; `SegRed`s become a chunked
/// phase 1 plus synthesized phase-2 tree reductions. `SegScan` scheduling is
/// not migrated yet, so it is deliberately restored to a serial Screma.
pub fn lower(inner: &mut EgirInner, binding_ids: &mut crate::IdSource<u32>) {
    let mut new_entries: Vec<EgirEntry> = Vec::new();
    for entry in inner.entry_points.iter_mut() {
        let Some(kind) = find_pending_seg(entry).map(|(bid, idx)| {
            let se = &entry.graph.skeleton.blocks[bid].side_effects[idx];
            let SideEffectKind::Pending(PendingSoac::Seg { kind, .. }) = &se.kind else {
                unreachable!()
            };
            kind.clone()
        }) else {
            continue;
        };
        match kind {
            SegOpKind::SegMap => {}
            SegOpKind::SegRed { .. } => {
                if let Some(phases) = lower_reduce_entry(entry, binding_ids) {
                    new_entries.extend(phases);
                } else {
                    restore_serial_seg(entry);
                }
            }
            SegOpKind::SegScan { .. } => {
                restore_serial_seg(entry);
            }
        }
    }
    inner.entry_points.extend(new_entries);

    // Each distinct iteration space is its own kernel (Futhark: one SegOp per
    // SegSpace). An entry holding several pointwise SegMaps over different
    // domains is split into one entry — and one pipeline stage — per domain.
    let EgirInner {
        entry_points,
        pipeline,
        ..
    } = inner;
    let mut split_clones: Vec<EgirEntry> = Vec::new();
    for entry in entry_points.iter_mut() {
        split_clones.extend(split_multidomain_seg_maps(entry, pipeline));
    }
    entry_points.extend(split_clones);
}

/// Locate every pointwise `SegMap` side-effect that writes an output view,
/// paired with the output slot it targets. Used to decide whether an entry
/// holds several independent map domains that must split into separate kernels.
fn output_seg_maps(entry: &EgirEntry) -> Vec<(BlockId, usize, usize)> {
    let mut segs = Vec::new();
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            let SideEffectKind::Pending(PendingSoac::Seg {
                kind: SegOpKind::SegMap,
                input_array_types,
                map_destinations,
                map_capture_counts,
                ..
            }) = &se.kind
            else {
                continue;
            };
            if !map_destinations.iter().all(|d| matches!(d, SoacDestination::OutputView)) {
                continue;
            }
            // SegMap operand layout: `[inputs, map_captures, output_views]`
            // (no accumulators). The first output view follows every input and
            // every capture.
            let view_base = input_array_types.len() + map_capture_counts.iter().sum::<usize>();
            let Some(&out_view) = se.operand_nodes.get(view_base) else {
                continue;
            };
            let Some(br) = graph_ops::extract_storage_view_source(&entry.graph, out_view) else {
                continue;
            };
            let Some(slot) = entry.outputs.iter().position(|o| o.storage_binding == Some(br)) else {
                continue;
            };
            segs.push((bid, i, slot));
        }
    }
    segs
}

/// Drop the side-effects named by `drop` from `entry` and restrict its outputs
/// to the single slot `keep_slot`. The dropped maps' kernels are no longer
/// emitted; their now-dead pure nodes are pruned downstream.
fn keep_only_seg(entry: &mut EgirEntry, drop: &[(BlockId, usize)], keep_slot: usize) {
    for (bid, block) in entry.graph.skeleton.blocks.iter_mut() {
        let mut drops: Vec<usize> = drop.iter().filter(|(b, _)| *b == bid).map(|(_, i)| *i).collect();
        drops.sort_unstable_by(|a, b| b.cmp(a));
        for i in drops {
            block.side_effects.remove(i);
        }
    }
    let kept = entry.outputs[keep_slot].clone();
    entry.outputs = vec![kept];
}

/// If `entry` holds more than one pointwise output `SegMap`, split it into one
/// entry per map. The original keeps the slot-0 map (under its own name); each
/// remaining map becomes a `{name}_dispatch_{slot}` clone with its own pipeline
/// stage. Returns the clones (empty when no split is needed).
fn split_multidomain_seg_maps(
    entry: &mut EgirEntry,
    pipeline: &mut crate::pipeline_descriptor::PipelineDescriptor,
) -> Vec<EgirEntry> {
    let mut segs = output_seg_maps(entry);
    if segs.len() <= 1 {
        return Vec::new();
    }
    segs.sort_by_key(|&(_, _, slot)| slot);
    let all_locs: Vec<(BlockId, usize)> = segs.iter().map(|&(b, i, _)| (b, i)).collect();

    // Add a stub stage per clone to the same pipeline that backs `entry`, so
    // `finalize_compute_io` can resolve each one's dispatch and IO.
    use crate::pipeline_descriptor::{ComputeStage, DispatchSize, Pipeline};
    let base_name = entry.name.clone();

    let mut clones = Vec::new();
    for &(_, _, slot) in &segs[1..] {
        let mut clone = entry.clone();
        clone.name = format!("{base_name}_dispatch_{slot}");
        let drop: Vec<(BlockId, usize)> =
            all_locs.iter().copied().filter(|loc| !is_seg_for_slot(&segs, *loc, slot)).collect();
        keep_only_seg(&mut clone, &drop, slot);
        if let Some(Pipeline::Compute(cp)) = pipeline.pipelines.iter_mut().find(|p| match p {
            Pipeline::Compute(c) => c.stages.iter().any(|s| s.entry_point == base_name),
            _ => false,
        }) {
            cp.stages.push(ComputeStage {
                entry_point: clone.name.clone(),
                workgroup_size: (64, 1, 1),
                dispatch_size: DispatchSize::Fixed { x: 1, y: 1, z: 1 },
                reads: vec![],
                writes: vec![],
            });
        }
        clones.push(clone);
    }

    // The original retains the slot-0 map only.
    let slot0 = segs[0].2;
    let drop0: Vec<(BlockId, usize)> =
        all_locs.iter().copied().filter(|loc| !is_seg_for_slot(&segs, *loc, slot0)).collect();
    keep_only_seg(entry, &drop0, slot0);

    clones
}

fn is_seg_for_slot(segs: &[(BlockId, usize, usize)], loc: (BlockId, usize), slot: usize) -> bool {
    segs.iter().any(|&(b, i, s)| (b, i) == loc && s == slot)
}

fn reify_tail_soac(entry: &mut EgirEntry) {
    // A multi-output entry returning a tuple of sibling maps over distinct
    // domains carries one Screma side-effect per output (TLC fusion already
    // merged equal-domain siblings into a single multi-lane Screma). Reify
    // every one — each becomes its own SegOp, matching Futhark's one-SegOp-
    // per-SegSpace structure; the schedule pass then splits distinct spaces
    // into separate kernels.
    let locs: Vec<(BlockId, usize)> = entry
        .graph
        .skeleton
        .blocks
        .iter()
        .flat_map(|(bid, block)| {
            block.side_effects.iter().enumerate().filter_map(move |(i, se)| {
                matches!(&se.kind, SideEffectKind::Pending(PendingSoac::Screma { .. })).then_some((bid, i))
            })
        })
        .collect();
    if locs.is_empty() {
        reify_parallel_scatter(entry);
        return;
    }
    for (block_id, idx) in locs {
        reify_one_screma(entry, block_id, idx);
    }
}

fn reify_one_screma(entry: &mut EgirEntry, block_id: BlockId, idx: usize) {
    let se = &entry.graph.skeleton.blocks[block_id].side_effects[idx];
    let SideEffectKind::Pending(PendingSoac::Screma {
        map_funcs,
        accumulators,
        input_array_types,
        input_elem_types,
        map_output_elem_types,
        map_input_indices,
        map_capture_counts,
        map_destinations,
        acc_destinations,
    }) = &se.kind
    else {
        unreachable!()
    };

    let n_inputs = input_array_types.len();
    let neutrals: Vec<NodeId> = (0..accumulators.len()).map(|i| se.operand_nodes[n_inputs + i]).collect();
    let Some(kind) = reify_seg_kind(accumulators, &neutrals) else {
        // Mixed reduce+scan Scremas remain serial until EGIR can schedule both
        // result classes coherently.
        return;
    };
    if matches!(kind, SegOpKind::SegMap)
        && (map_funcs.is_empty()
            || !map_destinations
                .iter()
                .all(|dest| matches!(dest, SoacDestination::OutputView | SoacDestination::InputBuffer)))
    {
        return;
    }

    let result_types = se
        .result
        .and_then(|result| entry.graph.types.get(&result))
        .map(|ty| match ty {
            Type::Constructed(TypeName::Tuple(_), fields) => fields.clone(),
            other => vec![other.clone()],
        })
        .unwrap_or_default();
    let seg = PendingSoac::Seg {
        space: SegSpace {
            level: SegLevel::Thread,
            len: None,
        },
        kind,
        map_funcs: map_funcs.clone(),
        input_array_types: input_array_types.clone(),
        input_elem_types: input_elem_types.clone(),
        map_output_elem_types: map_output_elem_types.clone(),
        map_input_indices: map_input_indices.clone(),
        map_capture_counts: map_capture_counts.clone(),
        map_destinations: map_destinations.clone(),
        acc_destinations: acc_destinations.clone(),
        result_types,
    };
    entry.graph.skeleton.blocks[block_id].side_effects[idx].kind = SideEffectKind::Pending(seg);
}

fn reify_seg_kind(accumulators: &[PendingScremaAccumulator], neutrals: &[NodeId]) -> Option<SegOpKind> {
    debug_assert_eq!(accumulators.len(), neutrals.len());
    let operators: Vec<SegBinOp> = accumulators
        .iter()
        .zip(neutrals)
        .map(|(acc, &neutral)| SegBinOp {
            kind: acc.kind,
            step_func: acc.step_func.clone(),
            reduce_op_func: acc.reduce_op_func.clone(),
            neutral,
            shape: Vec::new(),
            step_capture_count: acc.step_capture_count,
            reduce_op_capture_count: acc.reduce_op_capture_count,
            // Wyn's source reduction contract is associative but currently has
            // no commutativity annotation, so preserve left-to-right order.
            commutative: false,
        })
        .collect();
    if operators.is_empty() {
        Some(SegOpKind::SegMap)
    } else if operators.iter().all(|op| matches!(op.kind, crate::tlc::ScremaAccumulator::Reduce)) {
        Some(SegOpKind::SegRed { operators })
    } else if operators.iter().all(|op| matches!(op.kind, crate::tlc::ScremaAccumulator::Scan)) {
        Some(SegOpKind::SegScan { operators })
    } else {
        None
    }
}

fn reify_parallel_scatter(entry: &mut EgirEntry) {
    for (_, block) in entry.graph.skeleton.blocks.iter_mut() {
        for se in block.side_effects.iter_mut() {
            if let SideEffectKind::Pending(PendingSoac::Scatter { space, .. }) = &mut se.kind {
                if space.is_none() {
                    *space = Some(SegSpace {
                        level: SegLevel::Thread,
                        len: None,
                    });
                }
                return;
            }
        }
    }
}

#[derive(Clone, Copy)]
enum ChunkInputKind {
    StorageOnly,
    StorageOrRange,
}

struct ChunkedSoacInputs {
    tid: NodeId,
    chunk_start: NodeId,
    chunk_len: NodeId,
    views: Vec<NodeId>,
}

fn chunk_soac_inputs(
    graph: &mut EGraph,
    inputs: &[(NodeId, Type<TypeName>)],
    total_threads: u32,
    kind: ChunkInputKind,
    context: &str,
) -> Result<ChunkedSoacInputs, String> {
    let (first_view, _) = inputs.first().ok_or_else(|| format!("phase1 {context}: no SOAC inputs"))?;
    let input_len = input_length_for_chunking(graph, *first_view, kind, context)?;
    let (tid, chunk_start, chunk_len) = emit_chunk_arithmetic(graph, total_threads, input_len)?;

    let mut views = Vec::with_capacity(inputs.len());
    for &(view_nid, ref view_ty) in inputs {
        views.push(chunk_view_like(
            graph,
            view_nid,
            view_ty.clone(),
            chunk_start,
            chunk_len,
            kind,
            context,
        )?);
    }

    Ok(ChunkedSoacInputs {
        tid,
        chunk_start,
        chunk_len,
        views,
    })
}

fn input_length_for_chunking(
    graph: &mut EGraph,
    view_nid: NodeId,
    kind: ChunkInputKind,
    context: &str,
) -> Result<NodeId, String> {
    if let Some(br) = graph_ops::extract_storage_view_source(graph, view_nid) {
        return Ok(emit_storage_len(graph, br));
    }
    if matches!(kind, ChunkInputKind::StorageOrRange) {
        if let Some((_, len_nid, _)) = graph_ops::extract_array_range_operands(graph, view_nid) {
            return Ok(len_nid);
        }
    }
    Err(format!("phase1 {context}: input is not a chunkable view"))
}

fn chunk_view_like(
    graph: &mut EGraph,
    view_nid: NodeId,
    view_ty: Type<TypeName>,
    chunk_start: NodeId,
    chunk_len: NodeId,
    kind: ChunkInputKind,
    context: &str,
) -> Result<NodeId, String> {
    if let Some(br) = graph_ops::extract_storage_view_source(graph, view_nid) {
        return Ok(graph_ops::intern_chunked_storage_view(
            graph,
            br,
            chunk_start,
            chunk_len,
            view_ty,
            None,
        ));
    }
    if matches!(kind, ChunkInputKind::StorageOrRange) {
        if let Some((orig_start, _, step)) = graph_ops::extract_array_range_operands(graph, view_nid) {
            let has_step = step.is_some();
            let start_ty = graph.types[&orig_start].clone();
            let new_start = graph_ops::intern_binop(graph, "+", orig_start, chunk_start, start_ty, None);
            let mut ops: smallvec::SmallVec<[NodeId; 4]> = smallvec![new_start, chunk_len];
            if let Some(s) = step {
                ops.push(s);
            }
            return Ok(graph.intern_pure(PureOp::ArrayRange { has_step }, ops, view_ty));
        }
    }
    Err(format!("phase1 {context}: input is not a chunkable view"))
}

/// Workgroup width for the single-workgroup tree-reduce phase 2: `W` threads
/// grid-stride the `T` partials into shared memory, then reduce in-shared with
/// a log-`W` tree. Kept modest so `W * sizeof(elem)` stays within the
/// workgroup shared-memory budget (256 × a 36-byte tuple ≈ 9 KB). The phase2
/// `ComputeStage` in `tlc::parallelize` must dispatch this same width.

fn build_tree_reduce_phase2(
    b: &mut super::builder::EntryBuilder,
    op_func: String,
    elem_ty: Type<TypeName>,
    init_nid: NodeId,
    partials_binding: BindingRef,
    result_binding: BindingRef,
) {
    let w = PHASE2_WIDTH;
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let bool_ty = Type::Constructed(TypeName::Bool, vec![]);
    let view_arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(0),
            // region stamped from the binding by intern_storage_view.
            crate::types::no_region(),
        ],
    );

    // ---- entry block: lid, partials view + length, shared view, result view ----
    let entry_bid = b.graph_mut().skeleton.entry;
    let graph = b.graph_mut();
    let mut eff = graph_ops::next_effect_token(graph);

    let lid = graph_ops::intern_intrinsic(
        graph,
        catalog().known().local_id,
        smallvec![],
        u32_ty.clone(),
        None,
    );
    let partials_view = graph_ops::intern_storage_view(graph, partials_binding, view_arr_ty.clone(), None);
    let len = emit_storage_len(graph, partials_binding);
    let result_view = graph_ops::intern_storage_view(graph, result_binding, view_arr_ty.clone(), None);
    // Workgroup-shared `array<elem, W>` (id 0 within this entry).
    let shared_view = graph_ops::emit_workgroup_view(graph, 0, w, view_arr_ty.clone(), None);
    let w_nid = graph_ops::intern_u32(graph, w, None);
    let zero_u32 = graph_ops::intern_u32(graph, 0, None);

    // Contiguous per-thread chunk over `partials` (not strided): thread `lid`
    // reduces `partials[start .. end)`, so the tree combines `shared[0..W]` in
    // global order and the reduction stays valid for associative,
    // non-commutative operators.
    //   chunk = ceil(len / W);  start = lid * chunk;  end = min(start+chunk, len)
    let w_minus_1 = graph_ops::intern_u32(graph, w - 1, None);
    let len_plus = graph_ops::intern_binop(graph, "+", len, w_minus_1, u32_ty.clone(), None);
    let chunk = graph_ops::intern_binop(graph, "/", len_plus, w_nid, u32_ty.clone(), None);
    let start = graph_ops::intern_binop(graph, "*", lid, chunk, u32_ty.clone(), None);
    let start_plus = graph_ops::intern_binop(graph, "+", start, chunk, u32_ty.clone(), None);
    let u32_min = catalog().lookup_by_any_name("u32.min").expect("catalog has u32.min");
    let end = graph_ops::intern_intrinsic(
        graph,
        u32_min.id,
        smallvec![start_plus, len],
        u32_ty.clone(),
        None,
    );

    // ---- blocks ----
    let grid_header = graph.skeleton.create_block();
    let grid_body = graph.skeleton.create_block();
    let grid_cont = graph.skeleton.create_block();
    let grid_after = graph.skeleton.create_block();
    let tree_header = graph.skeleton.create_block();
    let tree_body = graph.skeleton.create_block();
    let tree_then = graph.skeleton.create_block();
    let tree_sel_merge = graph.skeleton.create_block();
    let tree_cont = graph.skeleton.create_block();
    let tree_after = graph.skeleton.create_block();
    let write_blk = graph.skeleton.create_block();
    let end_blk = graph.skeleton.create_block();

    // grid_header params: (acc, i)
    let acc_in = graph.add_block_param(grid_header, 0, elem_ty.clone());
    graph.skeleton.blocks[grid_header].params.push(acc_in);
    let i_in = graph.add_block_param(grid_header, 1, u32_ty.clone());
    graph.skeleton.blocks[grid_header].params.push(i_in);

    // entry → grid_header(init, start)
    graph.skeleton.blocks[entry_bid].term = SkeletonTerminator::Branch {
        target: grid_header,
        args: vec![init_nid, start],
    };

    // grid_header: i < end ? grid_body : grid_after(acc)
    let grid_cond = graph_ops::intern_binop(graph, "<", i_in, end, bool_ty.clone(), None);
    graph.skeleton.blocks[grid_header].term = SkeletonTerminator::CondBranch {
        cond: grid_cond,
        then_target: grid_body,
        then_args: vec![],
        else_target: grid_after,
        else_args: vec![acc_in],
    };
    b.control_headers_mut().insert(
        grid_header,
        ControlHeader::Loop {
            merge: grid_after,
            continue_block: grid_cont,
        },
    );

    // grid_body: acc' = op(acc, partials[i]); → grid_cont(acc')
    let graph = b.graph_mut();
    let elem_i = graph_ops::emit_view_load(
        graph,
        grid_body,
        partials_view,
        i_in,
        elem_ty.clone(),
        &mut eff,
        None,
    );
    let acc_next = graph.intern_pure(
        PureOp::Call(op_func.clone()),
        smallvec![acc_in, elem_i],
        elem_ty.clone(),
    );
    graph.skeleton.blocks[grid_body].term = SkeletonTerminator::Branch {
        target: grid_cont,
        args: vec![acc_next],
    };

    // grid_cont(acc_c): i_next = i + W; → grid_header(acc_c, i_next)
    let acc_c = graph.add_block_param(grid_cont, 0, elem_ty.clone());
    graph.skeleton.blocks[grid_cont].params.push(acc_c);
    let one_u32 = graph_ops::intern_u32(graph, 1, None);
    let i_next = graph_ops::intern_binop(graph, "+", i_in, one_u32, u32_ty.clone(), None);
    graph.skeleton.blocks[grid_cont].term = SkeletonTerminator::Branch {
        target: grid_header,
        args: vec![acc_c, i_next],
    };

    // grid_after(acc_final): shared[lid] = acc_final; barrier; → tree_header(W/2)
    let acc_final = graph.add_block_param(grid_after, 0, elem_ty.clone());
    graph.skeleton.blocks[grid_after].params.push(acc_final);
    graph_ops::emit_storage_store(
        graph,
        grid_after,
        shared_view,
        lid,
        acc_final,
        elem_ty.clone(),
        &mut eff,
        None,
    );
    graph_ops::emit_workgroup_barrier(graph, grid_after, &mut eff);
    let w_half = graph_ops::intern_u32(graph, w / 2, None);
    graph.skeleton.blocks[grid_after].term = SkeletonTerminator::Branch {
        target: tree_header,
        args: vec![w_half],
    };

    // tree_header(stride): stride > 0 ? tree_body : tree_after
    let stride_in = graph.add_block_param(tree_header, 0, u32_ty.clone());
    graph.skeleton.blocks[tree_header].params.push(stride_in);
    let stride_cond = graph_ops::intern_binop(graph, ">", stride_in, zero_u32, bool_ty.clone(), None);
    graph.skeleton.blocks[tree_header].term = SkeletonTerminator::CondBranch {
        cond: stride_cond,
        then_target: tree_body,
        then_args: vec![],
        else_target: tree_after,
        else_args: vec![],
    };
    b.control_headers_mut().insert(
        tree_header,
        ControlHeader::Loop {
            merge: tree_after,
            continue_block: tree_cont,
        },
    );

    // tree_body: lid < stride ? tree_then : tree_sel_merge  (selection)
    let graph = b.graph_mut();
    let active = graph_ops::intern_binop(graph, "<", lid, stride_in, bool_ty.clone(), None);
    graph.skeleton.blocks[tree_body].term = SkeletonTerminator::CondBranch {
        cond: active,
        then_target: tree_then,
        then_args: vec![],
        else_target: tree_sel_merge,
        else_args: vec![],
    };
    b.control_headers_mut().insert(
        tree_body,
        ControlHeader::Selection {
            merge: tree_sel_merge,
        },
    );

    // tree_then: shared[lid] = op(shared[lid], shared[lid+stride]); → tree_sel_merge
    let graph = b.graph_mut();
    let a = graph_ops::emit_view_load(
        graph,
        tree_then,
        shared_view,
        lid,
        elem_ty.clone(),
        &mut eff,
        None,
    );
    let lid_plus = graph_ops::intern_binop(graph, "+", lid, stride_in, u32_ty.clone(), None);
    let bb = graph_ops::emit_view_load(
        graph,
        tree_then,
        shared_view,
        lid_plus,
        elem_ty.clone(),
        &mut eff,
        None,
    );
    let combined = graph.intern_pure(PureOp::Call(op_func.clone()), smallvec![a, bb], elem_ty.clone());
    graph_ops::emit_storage_store(
        graph,
        tree_then,
        shared_view,
        lid,
        combined,
        elem_ty.clone(),
        &mut eff,
        None,
    );
    graph.skeleton.blocks[tree_then].term = SkeletonTerminator::Branch {
        target: tree_sel_merge,
        args: vec![],
    };

    // tree_sel_merge → tree_cont   (selection merge; barrier lives past it)
    graph.skeleton.blocks[tree_sel_merge].term = SkeletonTerminator::Branch {
        target: tree_cont,
        args: vec![],
    };

    // tree_cont: barrier; stride_next = stride/2; → tree_header(stride_next)
    graph_ops::emit_workgroup_barrier(graph, tree_cont, &mut eff);
    let two = graph_ops::intern_u32(graph, 2, None);
    let stride_next = graph_ops::intern_binop(graph, "/", stride_in, two, u32_ty.clone(), None);
    graph.skeleton.blocks[tree_cont].term = SkeletonTerminator::Branch {
        target: tree_header,
        args: vec![stride_next],
    };

    // tree_after: lid == 0 ? write_blk : end_blk   (selection)
    let is_zero = graph_ops::intern_binop(graph, "==", lid, zero_u32, bool_ty.clone(), None);
    graph.skeleton.blocks[tree_after].term = SkeletonTerminator::CondBranch {
        cond: is_zero,
        then_target: write_blk,
        then_args: vec![],
        else_target: end_blk,
        else_args: vec![],
    };
    b.control_headers_mut().insert(tree_after, ControlHeader::Selection { merge: end_blk });

    // write_blk: result[0] = shared[0]; → end_blk
    let graph = b.graph_mut();
    let s0 = graph_ops::emit_view_load(
        graph,
        write_blk,
        shared_view,
        zero_u32,
        elem_ty.clone(),
        &mut eff,
        None,
    );
    graph_ops::emit_storage_store(
        graph,
        write_blk,
        result_view,
        zero_u32,
        s0,
        elem_ty.clone(),
        &mut eff,
        None,
    );
    graph.skeleton.blocks[write_blk].term = SkeletonTerminator::Branch {
        target: end_blk,
        args: vec![],
    };

    // end_blk is the exit; `build()` finalizes it with Return(None).
    b.set_current_block(end_blk);
}

/// Emit the chunk-arithmetic preamble (`tid`, `chunk_start`,
/// `chunk_len`) as pure nodes in `graph`. Caller supplies the
/// `input_len` NodeId (typed `u32`) — for StorageView inputs that's a
/// `_w_intrinsic_storage_len(set, binding)` call; for Range inputs
/// it's the Range's own `len` operand. Returns
/// `(tid, chunk_start, chunk_len)`.
fn emit_chunk_arithmetic(
    graph: &mut super::types::EGraph,
    total_threads: u32,
    input_len: NodeId,
) -> Result<(NodeId, NodeId, NodeId), String> {
    // TODO(parallelize-egir): add an execution-level regression for `n <
    // dispatched_threads`.  `chunk_start >= input_len` must produce an empty
    // chunk; an unchecked `input_len - chunk_start` underflows for u32 and can
    // turn an idle invocation into an out-of-bounds reader.
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    // The chunk arithmetic runs in the input's *index* type: storage-view
    // inputs index in u32 (`_w_intrinsic_storage_len`), Range inputs in the
    // range's own element type (typically i32). Computing in u32 and feeding
    // a u32 `chunk_start`/`chunk_len` into an i32 Range produced an
    // `OpCompositeConstruct` whose constituents didn't match the i32
    // `{start, step, len}` struct (spirv-val rejected it). Derive the index
    // type from `input_len` and emit all arithmetic there.
    let index_ty = graph.types[&input_len].clone();
    let is_u32 = index_ty == u32_ty;

    // `tid`/`num_workgroups` are u32 intrinsics. The returned `tid` stays u32
    // (callers use it as a `partials[tid]` storage index); the index-typed
    // copies feed the chunk math.
    let tid = graph_ops::intern_intrinsic(
        graph,
        catalog().known().thread_id,
        smallvec![],
        u32_ty.clone(),
        None,
    );
    let nwg = graph_ops::intern_intrinsic(
        graph,
        catalog().known().num_workgroups,
        smallvec![],
        u32_ty.clone(),
        None,
    );
    let tid_idx = cast_u32_to_index(graph, tid, &index_ty)?;
    let nwg_idx = cast_u32_to_index(graph, nwg, &index_ty)?;

    // Runtime total thread count = num_workgroups.x * workgroup width. With a
    // `derived_from_input_length` dispatch (~ceil(n / width) workgroups) this
    // makes chunk_size ≈ 1, so each thread reduces ~one element — a saturating
    // grid rather than a fixed `total_threads`-wide one. `total_threads` is the
    // compile-time per-workgroup width.
    let wg_width = intern_index_lit(graph, total_threads, &index_ty);
    let total = graph_ops::intern_binop(graph, "*", nwg_idx, wg_width, index_ty.clone(), None);
    let one = intern_index_lit(graph, 1, &index_ty);
    let total_minus_one = graph_ops::intern_binop(graph, "-", total, one, index_ty.clone(), None);
    let len_plus = graph_ops::intern_binop(graph, "+", input_len, total_minus_one, index_ty.clone(), None);
    let chunk_size = graph_ops::intern_binop(graph, "/", len_plus, total, index_ty.clone(), None);
    let chunk_start = graph_ops::intern_binop(graph, "*", tid_idx, chunk_size, index_ty.clone(), None);
    let remaining = graph_ops::intern_binop(graph, "-", input_len, chunk_start, index_ty.clone(), None);
    let min_name = if is_u32 { "u32.min" } else { "i32.min" };
    let min_op =
        catalog().lookup_by_any_name(min_name).ok_or_else(|| format!("{} not in catalog", min_name))?;
    let chunk_len =
        graph_ops::intern_intrinsic(graph, min_op.id, smallvec![chunk_size, remaining], index_ty, None);
    Ok((tid, chunk_start, chunk_len))
}

/// Integer literal `n` typed as `index_ty` (`u32` → `PureOp::Uint`, else
/// `PureOp::Int`).
fn intern_index_lit(graph: &mut super::types::EGraph, n: u32, index_ty: &Type<TypeName>) -> NodeId {
    let op = match index_ty {
        Type::Constructed(TypeName::UInt(32), _) => super::types::PureOp::Uint(n.to_string()),
        _ => super::types::PureOp::Int(n.to_string()),
    };
    graph.intern_pure_with_span(op, smallvec![], index_ty.clone(), None)
}

/// Cast a u32 value into `index_ty`: identity for u32, else the per-type
/// bitcast intrinsic (`i32.u32`).
fn cast_u32_to_index(
    graph: &mut super::types::EGraph,
    v: NodeId,
    index_ty: &Type<TypeName>,
) -> Result<NodeId, String> {
    match index_ty {
        Type::Constructed(TypeName::UInt(32), _) => Ok(v),
        Type::Constructed(TypeName::Int(32), _) => {
            let conv = catalog()
                .lookup_by_any_name("i32.u32")
                .ok_or_else(|| "i32.u32 not in catalog".to_string())?;
            Ok(graph_ops::intern_intrinsic(
                graph,
                conv.id,
                smallvec![v],
                index_ty.clone(),
                None,
            ))
        }
        other => Err(format!("chunk arithmetic: unsupported index type {:?}", other)),
    }
}

/// Emit a `_w_intrinsic_storage_len(set, binding)` node returning u32.
fn emit_storage_len(graph: &mut super::types::EGraph, br: BindingRef) -> NodeId {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let set_nid = graph_ops::intern_u32(graph, br.set, None);
    let binding_nid = graph_ops::intern_u32(graph, br.binding, None);
    graph_ops::intern_intrinsic(
        graph,
        catalog().known().storage_len,
        smallvec![set_nid, binding_nid],
        u32_ty,
        None,
    )
}

/// Programmatic phase 2 synthesis where the neutral element is a
/// (possibly compound) pure subgraph cloned from phase 1. Used by the
/// Screma reduce path for any NE shape (scalar literal, tuple, array,
/// etc.).
pub fn synthesize_phase2_reduce_cloning_ne(
    entry_name: &str,
    op_func: String,
    elem_ty: Type<TypeName>,
    phase1_graph: &super::types::EGraph,
    phase1_ne_nid: NodeId,
    partials_binding: BindingRef,
    result_binding: BindingRef,
) -> Result<EgirEntry, String> {
    synthesize_phase2_reduce_cloning_ne_named(
        format!("{}_phase2_combine", entry_name),
        op_func,
        elem_ty,
        phase1_graph,
        phase1_ne_nid,
        partials_binding,
        result_binding,
    )
}

/// Same as `synthesize_phase2_reduce_cloning_ne` but caller picks the
/// full entry name — Screma's multi-accumulator path uses suffixes like
/// `_phase2_combine_0`, `_phase2_combine_1` so each combiner has a
/// unique entry point.
pub fn synthesize_phase2_reduce_cloning_ne_named(
    full_name: String,
    op_func: String,
    elem_ty: Type<TypeName>,
    phase1_graph: &super::types::EGraph,
    phase1_ne_nid: NodeId,
    partials_binding: BindingRef,
    result_binding: BindingRef,
) -> Result<EgirEntry, String> {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(full_name, (PHASE2_WIDTH, 1, 1));
    b.declare_intermediate_storage(partials_binding, elem_ty.clone());
    b.declare_output_storage(result_binding, elem_ty.clone());

    let init_nid = graph_ops::clone_pure_subgraph(phase1_graph, b.graph_mut(), phase1_ne_nid)?;
    build_tree_reduce_phase2(
        &mut b,
        op_func,
        elem_ty,
        init_nid,
        partials_binding,
        result_binding,
    );
    Ok(b.build())
}

fn find_pending_seg(entry: &EgirEntry) -> Option<(BlockId, usize)> {
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            if matches!(&se.kind, SideEffectKind::Pending(PendingSoac::Seg { .. })) {
                return Some((bid, i));
            }
        }
    }
    None
}

fn restore_serial_seg(entry: &mut EgirEntry) {
    let Some((block_id, idx)) = find_pending_seg(entry) else {
        return;
    };
    let kind = entry.graph.skeleton.blocks[block_id].side_effects[idx].kind.clone();
    let SideEffectKind::Pending(PendingSoac::Seg {
        kind,
        map_funcs,
        input_array_types,
        input_elem_types,
        map_output_elem_types,
        map_input_indices,
        map_capture_counts,
        map_destinations,
        acc_destinations,
        ..
    }) = kind
    else {
        unreachable!()
    };
    let operators = match kind {
        SegOpKind::SegMap => Vec::new(),
        SegOpKind::SegRed { operators } | SegOpKind::SegScan { operators } => operators,
    };
    let accumulators = operators
        .into_iter()
        .map(|op| PendingScremaAccumulator {
            kind: op.kind,
            step_func: op.step_func,
            reduce_op_func: op.reduce_op_func,
            step_capture_count: op.step_capture_count,
            reduce_op_capture_count: op.reduce_op_capture_count,
        })
        .collect();
    entry.graph.skeleton.blocks[block_id].side_effects[idx].kind =
        SideEffectKind::Pending(PendingSoac::Screma {
            map_funcs,
            accumulators,
            input_array_types,
            input_elem_types,
            map_output_elem_types,
            map_input_indices,
            map_capture_counts,
            map_destinations,
            acc_destinations,
        });
}

fn project_root_index(graph: &super::types::EGraph, value: NodeId, root: NodeId) -> Option<u32> {
    let mut cur = value;
    let mut last_index = None;
    loop {
        if cur == root {
            return last_index;
        }
        match &graph.nodes[cur] {
            super::types::ENode::Pure {
                op: super::types::PureOp::Project { index },
                operands,
            } => {
                last_index = Some(*index);
                cur = operands[0];
            }
            _ => return None,
        }
    }
}

fn lower_reduce_entry(
    entry: &mut EgirEntry,
    binding_ids: &mut crate::IdSource<u32>,
) -> Option<Vec<EgirEntry>> {
    let total_threads = REDUCE_PHASE1_WIDTH;

    // 2. Locate the semantic SegRed and pull its metadata.
    let Some((block_id, idx)) = find_pending_seg(entry) else {
        return None;
    };
    let (
        reduce_funcs,
        n_maps,
        map_output_view_ops,
        input_view_data,
        init_nids,
        elem_tys,
        screma_result_nid,
    ) = {
        let se = &entry.graph.skeleton.blocks[block_id].side_effects[idx];
        let SideEffectKind::Pending(PendingSoac::Seg {
            kind: SegOpKind::SegRed { operators },
            map_funcs,
            input_array_types,
            map_destinations,
            acc_destinations,
            map_capture_counts,
            ..
        }) = &se.kind
        else {
            return None;
        };
        let n_accs = operators.len();
        // TODO(parallelize-egir): add a positive test for a SegRed whose map,
        // step, and/or combine lambda captures a scalar.  Falling back to the
        // serial Screma is correct today, but the parallel path must eventually
        // thread these captures through both generated phases.
        if operators.iter().any(|a| a.step_capture_count != 0 || a.reduce_op_capture_count != 0) {
            return None;
        }
        if map_capture_counts.iter().any(|&c| c != 0) {
            return None;
        }
        let n_inputs = input_array_types.len();
        if n_inputs == 0 {
            return None;
        }
        let n_maps = map_funcs.len();
        // Map outputs that have been retargeted to OutputView so the
        // chunked writes inside the Screma loop body land in the right
        // buffer. Fresh destinations would require building an immutable
        // array per loop iteration — incompatible with chunked parallel
        // writes.
        if !map_destinations.iter().all(|d| matches!(d, SoacDestination::OutputView)) {
            return None;
        }
        // Reduce accumulators expect Fresh destination (scalar result
        // routed via a Project-based Store outside the Screma loop).
        if !acc_destinations.iter().all(|d| matches!(d, SoacDestination::Fresh)) {
            return None;
        }
        // Operand layout (gate enforces zero captures everywhere):
        //   [inputs(n_inputs), init_accs(n_accs), map_output_views(n_maps),
        //    acc_output_views(0 — all Fresh)]
        let input_view_data: Vec<(NodeId, Type<TypeName>)> = (0..n_inputs)
            .map(|k| {
                let v = se.operand_nodes[k];
                (v, entry.graph.types[&v].clone())
            })
            .collect();
        let init_nids: Vec<NodeId> = operators.iter().map(|op| op.neutral).collect();
        let base = n_inputs + n_accs;
        let map_view_ops: Vec<usize> = (0..n_maps).map(|m| base + m).collect();
        let result = se.result?;
        let elem_tys: Vec<Type<TypeName>> =
            init_nids.iter().map(|n| entry.graph.types[n].clone()).collect();
        let reduce_funcs: Vec<String> = operators.iter().map(|a| a.reduce_op_func.clone()).collect();
        (
            reduce_funcs,
            n_maps,
            map_view_ops,
            input_view_data,
            init_nids,
            elem_tys,
            result,
        )
    };

    let n_accs = elem_tys.len();
    let acc_bindings: Vec<(BindingRef, BindingRef)> = (0..n_accs)
        .map(|_| {
            (
                BindingRef::new(crate::egir::from_tlc::AUTO_STORAGE_SET, binding_ids.next_id()),
                BindingRef::new(crate::egir::from_tlc::AUTO_STORAGE_SET, binding_ids.next_id()),
            )
        })
        .collect();

    // TODO(parallelize-egir): make this lowering transactional and add a test
    // where a late routing check fails.  All validation must happen before the
    // first graph mutation so returning `None` can never leave a half-lowered
    // entry behind.

    // 3. Chunk all input views and every map output view; swap them back
    // into the Screma operand list.
    let chunked = chunk_soac_inputs(
        &mut entry.graph,
        &input_view_data,
        total_threads,
        ChunkInputKind::StorageOrRange,
        "SegRed",
    )
    .ok()?;
    let chunk_start = chunked.chunk_start;
    let chunk_len = chunked.chunk_len;
    {
        let se = &mut entry.graph.skeleton.blocks[block_id].side_effects[idx];
        for (i, &new_view) in chunked.views.iter().enumerate() {
            se.operand_nodes[i] = new_view;
        }
    }
    for (m_idx, op_idx) in map_output_view_ops.iter().enumerate() {
        let orig_view = entry.graph.skeleton.blocks[block_id].side_effects[idx].operand_nodes[*op_idx];
        let view_ty = entry.graph.types[&orig_view].clone();
        let chunked_view = chunk_view_like(
            &mut entry.graph,
            orig_view,
            view_ty,
            chunk_start,
            chunk_len,
            ChunkInputKind::StorageOnly,
            &format!("SegRed map output {m_idx}"),
        )
        .ok()?;
        let se = &mut entry.graph.skeleton.blocks[block_id].side_effects[idx];
        se.operand_nodes[*op_idx] = chunked_view;
    }

    // 4. For each reduce accumulator, find the Store that consumes
    // `Project { index: n_maps + acc_i } of screma_result_nid` and
    // redirect its place to `partials_i[tid]`. Track the original
    // binding for clearing the entry's output slot.
    let mut keeper_stores: Vec<Option<(BlockId, usize, Option<BindingRef>)>> = vec![None; n_accs];
    let mut duplicate_stores: Vec<(BlockId, usize)> = Vec::new();
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            if !matches!(&se.kind, SideEffectKind::Inst(InstKind::Store { .. })) {
                continue;
            }
            let Some(&value_nid) = se.operand_nodes.get(1) else {
                continue;
            };
            let Some(root_idx) = project_root_index(&entry.graph, value_nid, screma_result_nid) else {
                continue;
            };
            if (root_idx as usize) < n_maps {
                continue; // map output Stores aren't generated for OutputView destinations
            }
            let acc_i = (root_idx as usize) - n_maps;
            if acc_i >= n_accs {
                continue;
            }
            if keeper_stores[acc_i].is_none() {
                let orig_bind = se
                    .operand_nodes
                    .get(0)
                    .and_then(|&p| graph_ops::extract_storage_view_source(&entry.graph, p));
                keeper_stores[acc_i] = Some((bid, i, orig_bind));
            } else {
                duplicate_stores.push((bid, i));
            }
        }
    }
    for (acc_i, slot) in keeper_stores.iter().enumerate() {
        if slot.is_none() {
            // No store for this accumulator's tuple field — bail out
            // (caller falls back to serial via plan = None in another
            // path; but here we've already partially mutated the entry,
            // which is a Screma transform bug. Keep this as a hard fail
            // for diagnosis.)
            let _ = acc_i;
            return None;
        }
    }
    // Redirect each keeper store.
    for (acc_i, (partials_binding, _)) in acc_bindings.iter().enumerate() {
        let (blk, sx, _) = keeper_stores[acc_i].expect("checked above");
        let elem_ty = elem_tys[acc_i].clone();
        let arr_ty = Type::Constructed(
            TypeName::Array,
            vec![
                elem_ty.clone(),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
                Type::Variable(0),
                crate::types::no_region(),
            ],
        );
        let partials_view =
            graph_ops::intern_storage_view(&mut entry.graph, *partials_binding, arr_ty, None);
        let new_place = entry.graph.intern_pure(
            super::types::PureOp::ViewIndex,
            smallvec![partials_view, chunked.tid],
            elem_ty,
        );
        entry.graph.skeleton.blocks[blk].side_effects[sx].operand_nodes[0] = new_place;
    }
    // Drop duplicates highest-idx-first so earlier indices stay valid.
    duplicate_stores.sort_by(|a, b| b.1.cmp(&a.1));
    for (bid, sx) in duplicate_stores {
        entry.graph.skeleton.blocks[bid].side_effects.remove(sx);
    }

    // 5. Clear the entry.outputs slot for each redirected accumulator
    // binding; map output slots keep their storage_binding. Register
    // each partials buffer as Intermediate.
    for (acc_i, (partials_binding, _)) in acc_bindings.iter().enumerate() {
        if let Some((_, _, Some(orig))) = keeper_stores[acc_i] {
            for o in entry.outputs.iter_mut() {
                if o.storage_binding == Some(orig) {
                    o.storage_binding = None;
                }
            }
        }
        entry.storage_bindings.push(crate::interface::StorageBindingDecl {
            binding: *partials_binding,
            role: crate::interface::StorageRole::Intermediate,
            elem_ty: elem_tys[acc_i].clone(),
            length: None,
        });
    }

    // 6. Synthesize one phase 2 entry per accumulator. Cloning NEs from
    // the post-mutation graph is harmless — the NE node is pure and
    // hasn't been touched.
    let phase1_snapshot = entry.graph.clone();
    let mut phase2s = Vec::with_capacity(n_accs);
    for (acc_i, (partials_binding, result_binding)) in acc_bindings.iter().enumerate() {
        let phase2_name = if n_accs == 1 {
            format!("{}_phase2_combine", entry.name)
        } else {
            format!("{}_phase2_combine_{}", entry.name, acc_i)
        };
        let phase2 = synthesize_phase2_reduce_cloning_ne_named(
            phase2_name,
            reduce_funcs[acc_i].clone(),
            elem_tys[acc_i].clone(),
            &phase1_snapshot,
            init_nids[acc_i],
            *partials_binding,
            *result_binding,
        )
        .ok()?;
        phase2s.push(phase2);
    }
    // Scheduling consumed the semantic SegRed. Phase 1 is now an ordinary
    // per-invocation Screma over the thread's chunk; `soac_expand` lowers that
    // local loop while the synthesized phase-2 entries combine its partials.
    restore_serial_seg(entry);
    Some(phase2s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tlc::ScremaAccumulator;

    fn accumulator(kind: ScremaAccumulator, stem: &str) -> PendingScremaAccumulator {
        PendingScremaAccumulator {
            kind,
            step_func: format!("{stem}_step"),
            reduce_op_func: format!("{stem}_combine"),
            step_capture_count: 1,
            reduce_op_capture_count: 2,
        }
    }

    fn neutral(graph: &mut EGraph, index: usize) -> NodeId {
        graph.add_func_param(index, Type::Constructed(TypeName::Unit, vec![]))
    }

    #[test]
    fn reduction_accumulator_reifies_as_seg_red_operator() {
        let mut graph = EGraph::new();
        let ne = neutral(&mut graph, 0);
        let kind = reify_seg_kind(&[accumulator(ScremaAccumulator::Reduce, "sum")], &[ne])
            .expect("a reduction is a supported semantic SegOp");
        let SegOpKind::SegRed { operators } = kind else {
            panic!("reduction must reify as SegRed")
        };
        assert_eq!(operators.len(), 1);
        assert_eq!(operators[0].step_func, "sum_step");
        assert_eq!(operators[0].reduce_op_func, "sum_combine");
        assert_eq!(operators[0].neutral, ne);
        assert!(operators[0].shape.is_empty());
        assert_eq!(operators[0].step_capture_count, 1);
        assert_eq!(operators[0].reduce_op_capture_count, 2);
        assert!(
            !operators[0].commutative,
            "Wyn does not yet declare commutativity"
        );
    }

    #[test]
    fn scan_accumulator_reifies_as_seg_scan_operator() {
        let mut graph = EGraph::new();
        let ne = neutral(&mut graph, 0);
        let kind = reify_seg_kind(&[accumulator(ScremaAccumulator::Scan, "prefix")], &[ne])
            .expect("a scan is a supported semantic SegOp");
        assert!(matches!(kind, SegOpKind::SegScan { operators } if operators.len() == 1));
    }

    #[test]
    fn mixed_reduce_and_scan_stays_serial_until_joint_scheduler_exists() {
        let accumulators = [
            accumulator(ScremaAccumulator::Reduce, "sum"),
            accumulator(ScremaAccumulator::Scan, "prefix"),
        ];
        let mut graph = EGraph::new();
        let neutrals = [neutral(&mut graph, 0), neutral(&mut graph, 1)];
        assert!(reify_seg_kind(&accumulators, &neutrals).is_none());
    }
}
