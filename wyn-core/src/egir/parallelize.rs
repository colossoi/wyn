//! EGIR-side SOAC parallelization.
//!
//! Consumes the `ParallelizationPlan`s produced by `tlc::parallelize`:
//! Map entries get their tail SOAC wrapped in `PendingSoac::Parallel`
//! for the lane-indexed kernel; Reduce and Redomap entries get an
//! in-place phase1 rewrite (chunked input + store-to-partials) plus a
//! freshly-synthesized phase2-combine EgirEntry; Scan gets an in-place
//! phase1 (chunked scan + chunked reduce → block_sums) plus two
//! synthesized entries (phase 2: sequential scan of block_sums into
//! block_offsets; phase 3: chunked apply of block_offsets to output).
use std::collections::HashMap;

use polytype::Type;
use smallvec::smallvec;

use crate::BindingRef;
use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ConstantValue, InstKind};

use super::graph_ops;
use super::program::{EgirEntry, EgirFunc, EgirInner};
use super::types::{
    EGraph, NodeId, PendingSoac, PureOp, SideEffectKind, SkeletonTerminator, SoacDestination,
};
use crate::ssa::types::ControlHeader;
use crate::tlc::parallelize::{ParallelStrategy, ParallelizationPlan};

/// Walk every entry; for each entry that has a plan, do the EGIR-side
/// transformation that strategy requires.
///
/// Map     → wrap the tail SOAC in `PendingSoac::Parallel` so
///            `soac_expand` emits the lane-indexed kernel.
/// Reduce  → in-place phase1 rewrite (chunked reduce + store-to-
///            partials), then synthesize a phase2-combine EgirEntry.
/// Redomap → same shape as Reduce phase1/phase2, but phase2 uses the
///            redomap's pure `reduce_func` combiner and the NE
///            subgraph is cloned across EGraphs.
/// Scan    → in-place phase1 rewrite (chunked scan + chunked reduce →
///            block_sums) and two synthesized entries (phase 2 +
///            phase 3). See `transform_scan_entry`.
pub fn run(inner: &mut EgirInner, plans: &HashMap<String, ParallelizationPlan>) {
    // Collect entries-to-add separately so we don't mutate
    // `inner.entry_points` while iterating it.
    let mut new_entries: Vec<EgirEntry> = Vec::new();
    let mut new_functions: Vec<EgirFunc> = Vec::new();
    for entry in inner.entry_points.iter_mut() {
        let Some(plan) = plans.get(&entry.name) else {
            continue;
        };
        match plan.strategy {
            ParallelStrategy::Map => rewrite_tail_map(entry),
            ParallelStrategy::Reduce | ParallelStrategy::Redomap => {
                if let Some(phase2) = transform_accumulator_entry(entry, plan) {
                    new_entries.push(phase2);
                }
            }
            ParallelStrategy::Scan => {
                if let Some((phases, swap_wrapper)) = transform_scan_entry(entry, plan) {
                    new_entries.extend(phases);
                    new_functions.push(swap_wrapper);
                }
            }
            ParallelStrategy::Screma => {
                if let Some((phases, swap_wrapper)) = transform_screma_entry(entry, plan) {
                    new_entries.extend(phases);
                    if let Some(sw) = swap_wrapper {
                        new_functions.push(sw);
                    }
                }
            }
        }
    }
    inner.entry_points.extend(new_entries);
    inner.functions.extend(new_functions);
}

fn rewrite_tail_map(entry: &mut super::program::EgirEntry) {
    // Walk every block and locate the (unique) tail Map. analyze_entry
    // already proved the entry's tail expression is the Map; from_tlc
    // turns it into a `PendingSoac::Map` side-effect. Wrap it.
    for (_, block) in entry.graph.skeleton.blocks.iter_mut() {
        for se in block.side_effects.iter_mut() {
            let SideEffectKind::Pending(ref pending) = se.kind else {
                continue;
            };
            // Only entry-tail pointwise SOACs that target output views are
            // parallelizable here. Intermediate Fresh producers must stay
            // serial unless a strategy-specific transform rewrites them.
            match pending {
                PendingSoac::Map {
                    destination: SoacDestination::OutputView,
                    ..
                } => {}
                PendingSoac::Screma {
                    accumulators,
                    map_destinations,
                    ..
                } if accumulators.is_empty()
                    && !map_destinations.is_empty()
                    && map_destinations.iter().all(|dest| *dest == SoacDestination::OutputView) => {}
                _ => continue,
            }
            // PendingSoac derives Clone, so the clean version is just to
            // clone the inner SOAC out, then overwrite. The wrapper is
            // dominantly a thin marker — clone cost is one Vec<Type> per
            // entry per compilation.
            let SideEffectKind::Pending(inner) = se.kind.clone() else {
                unreachable!()
            };
            se.kind = SideEffectKind::Pending(PendingSoac::Parallel {
                serial: Box::new(inner),
            });
            return;
        }
    }
}

/// Shared accumulator-style migration. Reduce and Redomap both become a
/// chunked phase 1 that writes per-thread partials, followed by a synthesized
/// phase 2 that combines those partials into the final result. The
/// strategy-specific parts are represented by `AccumulatorPhase2`.
fn transform_accumulator_entry(entry: &mut EgirEntry, plan: &ParallelizationPlan) -> Option<EgirEntry> {
    let phase_bindings = accumulator_phase_bindings(plan, plan.strategy)?;
    let phase2_spec = extract_accumulator_phase2(entry, plan.strategy)?;
    let phase2 = build_accumulator_phase2(entry, phase_bindings, phase2_spec)?;

    match plan.strategy {
        ParallelStrategy::Reduce => {
            phase1_transform_reduce(entry, phase_bindings.total_threads, phase_bindings.partials).ok()?;
        }
        ParallelStrategy::Redomap => {
            phase1_transform_redomap(entry, phase_bindings.total_threads, phase_bindings.partials).ok()?;
        }
        _ => return None,
    }
    Some(phase2)
}

enum AccumulatorPhase2 {
    ConstantNe {
        func: String,
        elem_ty: Type<TypeName>,
        init: ConstantValue,
    },
    CloneNe {
        func: String,
        elem_ty: Type<TypeName>,
        ne_nid: NodeId,
    },
}

fn extract_accumulator_phase2(entry: &EgirEntry, strategy: ParallelStrategy) -> Option<AccumulatorPhase2> {
    match strategy {
        ParallelStrategy::Reduce => {
            let (block, idx) = find_pending_reduce(entry)?;
            let se = &entry.graph.skeleton.blocks[block].side_effects[idx];
            let func = match &se.kind {
                SideEffectKind::Pending(PendingSoac::Reduce { func, .. }) => func.clone(),
                _ => return None,
            };
            let init_nid = *se.operand_nodes.get(1)?;
            let init = extract_constant(&entry.graph, init_nid)?;
            let result_nid = se.result?;
            let elem_ty = entry.graph.types[&result_nid].clone();
            Some(AccumulatorPhase2::ConstantNe { func, elem_ty, init })
        }
        ParallelStrategy::Redomap => {
            let (block, idx) = find_pending_redomap(entry)?;
            let se = &entry.graph.skeleton.blocks[block].side_effects[idx];
            let (input_count, func) = match &se.kind {
                SideEffectKind::Pending(PendingSoac::Redomap {
                    reduce_func,
                    input_array_types,
                    ..
                }) => (input_array_types.len(), reduce_func.clone()),
                _ => return None,
            };
            let ne_nid = *se.operand_nodes.get(input_count)?;
            let result_nid = se.result?;
            let elem_ty = entry.graph.types[&result_nid].clone();
            Some(AccumulatorPhase2::CloneNe {
                func,
                elem_ty,
                ne_nid,
            })
        }
        _ => None,
    }
}

fn build_accumulator_phase2(
    entry: &EgirEntry,
    phase_bindings: AccumulatorPhaseBindings,
    spec: AccumulatorPhase2,
) -> Option<EgirEntry> {
    match spec {
        AccumulatorPhase2::ConstantNe { func, elem_ty, init } => Some(synthesize_phase2_reduce(
            &entry.name,
            func,
            elem_ty,
            init,
            phase_bindings.partials,
            phase_bindings.result,
        )),
        AccumulatorPhase2::CloneNe {
            func,
            elem_ty,
            ne_nid,
        } => {
            let phase1_graph_snapshot = entry.graph.clone();
            synthesize_phase2_reduce_cloning_ne(
                &entry.name,
                func,
                elem_ty,
                &phase1_graph_snapshot,
                ne_nid,
                phase_bindings.partials,
                phase_bindings.result,
            )
            .ok()
        }
    }
}

#[derive(Clone, Copy)]
struct AccumulatorPhaseBindings {
    partials: BindingRef,
    result: BindingRef,
    total_threads: u32,
}

fn accumulator_phase_bindings(
    plan: &ParallelizationPlan,
    expected: ParallelStrategy,
) -> Option<AccumulatorPhaseBindings> {
    use crate::tlc::parallelize::{DispatchModel, PlannedBindings};
    if plan.strategy != expected {
        return None;
    }
    let (partials, result) = match (expected, &plan.bindings) {
        (ParallelStrategy::Reduce, PlannedBindings::Reduce { partials, result })
        | (ParallelStrategy::Redomap, PlannedBindings::Redomap { partials, result }) => {
            (*partials, *result)
        }
        _ => return None,
    };
    let total_threads = match plan.dispatch {
        DispatchModel::Fixed { local_size, .. } => local_size[0],
        DispatchModel::DerivedFromInputLength { workgroup_size, .. } => workgroup_size,
    };
    Some(AccumulatorPhaseBindings {
        partials,
        result,
        total_threads,
    })
}

fn extract_constant(graph: &super::types::EGraph, nid: NodeId) -> Option<ConstantValue> {
    match &graph.nodes[nid] {
        super::types::ENode::Constant(c) => Some(*c),
        super::types::ENode::Pure { op, .. } => match op {
            super::types::PureOp::Uint(s) => s.parse::<u32>().ok().map(ConstantValue::U32),
            super::types::PureOp::Int(s) => s.parse::<i32>().ok().map(ConstantValue::I32),
            super::types::PureOp::Float(s) => s.parse::<f32>().ok().map(ConstantValue::from_f32),
            super::types::PureOp::Bool(b) => Some(ConstantValue::Bool(*b)),
            _ => None,
        },
        _ => None,
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

/// Rewrite an EgirEntry in place: its tail `PendingSoac::Reduce` becomes a
/// chunked reduce + store-to-partials. Caller passes the per-workgroup
/// thread count and the partials binding (set, binding) the TLC analysis
/// reserved.
///
/// Preconditions:
/// - Entry's body has exactly one `PendingSoac::Reduce` side-effect.
/// - Entry's body has exactly one `InstKind::Store` whose value operand
///   is the reduce's result (the auto-allocated output store).
/// - The reduce's input is a `PureOp::StorageView(Storage{set, binding})`
///   (i.e. the source array is a storage buffer, not a Range).
pub fn phase1_transform_reduce(
    entry: &mut EgirEntry,
    total_threads: u32,
    partials_binding: BindingRef,
) -> Result<(), String> {
    // 1. Locate the Reduce side-effect and pull the operands.
    let (reduce_block, reduce_idx) =
        find_pending_reduce(entry).ok_or_else(|| "no PendingSoac::Reduce in entry".to_string())?;
    let (result_nid, input_view_nid, input_view_ty) = {
        let se = &entry.graph.skeleton.blocks[reduce_block].side_effects[reduce_idx];
        let input = se.operand_nodes[0];
        let result = se.result.ok_or_else(|| "Reduce missing result".to_string())?;
        let input_ty = entry.graph.types[&input].clone();
        (result, input, input_ty)
    };

    // 2. Build chunk arithmetic in the same block as the reduce.
    let chunked = chunk_soac_inputs(
        &mut entry.graph,
        &[(input_view_nid, input_view_ty)],
        total_threads,
        ChunkInputKind::StorageOnly,
        "Reduce",
    )?;

    // 4. Swap the Reduce's input operand.
    entry.graph.skeleton.blocks[reduce_block].side_effects[reduce_idx].operand_nodes[0] = chunked.views[0];

    // 5. Find the auto-output Store (writes the reduce result), rewrite
    //    its place to ViewIndex(partials_view, tid).
    let (store_block, store_idx) =
        find_store_of(entry, result_nid).ok_or_else(|| "no Store writes the Reduce result".to_string())?;
    let elem_ty = entry.graph.types[&result_nid].clone();
    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(0),
            // region stamped from the binding by intern_storage_view.
            crate::types::no_region(),
        ],
    );
    let partials_view = graph_ops::intern_storage_view(&mut entry.graph, partials_binding, arr_ty, None);
    let new_place = entry.graph.intern_pure(
        super::types::PureOp::ViewIndex,
        smallvec![partials_view, chunked.tid],
        elem_ty.clone(),
    );
    {
        let se = &mut entry.graph.skeleton.blocks[store_block].side_effects[store_idx];
        se.operand_nodes[0] = new_place;
    }

    // 7. The outputs[] originally pointed at the auto-allocated result
    //    binding; phase1 no longer writes there. Clear it. (Phase2 — when
    //    synthesized — will take ownership of that binding.) Also append
    //    the partials Intermediate decl.
    let result_binding = entry.outputs.get(0).and_then(|o| o.storage_binding);
    if let Some(o) = entry.outputs.get_mut(0) {
        o.storage_binding = None;
    }
    entry.storage_bindings.push(crate::interface::StorageBindingDecl {
        binding: crate::BindingRef::new(partials_binding.set, partials_binding.binding),
        role: crate::interface::StorageRole::Intermediate,
        elem_ty,
        length: None,
    });
    let _ = result_binding; // caller needs this for phase2 synthesis (not done here)

    Ok(())
}

/// Programmatic phase2 synthesis: build a single-threaded
/// `EgirEntry` that reads `partials`, reduces it, and stores the
/// final scalar to `result_binding[0]`.
///
/// Uses `EntryBuilder` for graph construction — no TLC `Def` or
/// `from_tlc::run` round-trip.
///
/// `init` is described by `ReduceInit` rather than by a NodeId because
/// EGraph nodes don't cross between entries; the caller passes the
/// neutral element abstractly so phase2 re-creates it inside its own
/// graph. Today only scalar literal initializers are supported, which
/// covers all Wyn `reduce` ne shapes that pass through this path
/// (defunctionalization hoists complex inits to lifted-let bindings).
pub fn synthesize_phase2_reduce(
    entry_name: &str,
    op_func: String,
    elem_ty: Type<TypeName>,
    init: ConstantValue,
    partials_binding: BindingRef,
    result_binding: BindingRef,
) -> EgirEntry {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(format!("{}_phase2_combine", entry_name), (PHASE2_WIDTH, 1, 1));
    b.declare_intermediate_storage(partials_binding, elem_ty.clone());
    b.declare_output_storage(result_binding, elem_ty.clone());

    let init_nid = b.emit_constant(init, elem_ty.clone());
    build_tree_reduce_phase2(
        &mut b,
        op_func,
        elem_ty,
        init_nid,
        partials_binding,
        result_binding,
    );
    b.build()
}

/// Workgroup width for the single-workgroup tree-reduce phase 2: `W` threads
/// grid-stride the `T` partials into shared memory, then reduce in-shared with
/// a log-`W` tree. Kept modest so `W * sizeof(elem)` stays within the
/// workgroup shared-memory budget (256 × a 36-byte tuple ≈ 9 KB). The phase2
/// `ComputeStage` in `tlc::parallelize` must dispatch this same width.
pub const PHASE2_WIDTH: u32 = 256;

/// Build the workgroup-parallel tree-reduce body for phase 2 into `b`'s graph.
/// `init_nid` is the neutral element, already interned in `b`'s graph.
///
/// `W = PHASE2_WIDTH` threads each grid-stride over `partials[lid, lid+W, …]`
/// accumulating into a register, write the partial to `shared[lid]`, barrier,
/// then tree-reduce shared memory (`stride = W/2 … 1`, `shared[lid] =
/// op(shared[lid], shared[lid+stride])`) with a barrier each step; thread 0
/// writes `shared[0]` to `result[0]`. Cost `O(T/W + log W)`.
///
/// Correctness: associative, not commutative. Both the grid-stride
/// (`op(acc, partials[i])`, increasing `i`) and the tree (`op(lower, higher)`)
/// combine in deterministic left→right order. Barriers sit in uniform control
/// flow (the loop's continue block + after the grid loop), reached by every
/// invocation — required for `OpControlBarrier`.
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

/// Rewrite an EgirEntry in place: its tail `PendingSoac::Redomap`
/// becomes a chunked redomap + store-to-partials. Same shape as
/// `phase1_transform_reduce` but operates on a Redomap side-effect,
/// which has N parallel inputs that all share length.
pub fn phase1_transform_redomap(
    entry: &mut EgirEntry,
    total_threads: u32,
    partials_binding: BindingRef,
) -> Result<(), String> {
    let (block_id, idx) =
        find_pending_redomap(entry).ok_or_else(|| "no PendingSoac::Redomap in entry".to_string())?;

    let (input_view_data, result_nid) = {
        let se = &entry.graph.skeleton.blocks[block_id].side_effects[idx];
        let n = match &se.kind {
            SideEffectKind::Pending(PendingSoac::Redomap {
                input_array_types, ..
            }) => input_array_types.len(),
            _ => return Err("expected PendingSoac::Redomap".to_string()),
        };
        let result = se.result.ok_or_else(|| "Redomap missing result".to_string())?;
        let mut view_data: Vec<(NodeId, Type<TypeName>)> = Vec::with_capacity(n);
        for i in 0..n {
            let view_nid = se.operand_nodes[i];
            let view_ty = entry.graph.types[&view_nid].clone();
            view_data.push((view_nid, view_ty));
        }
        (view_data, result)
    };

    // Chunk arith uses the first input's length.
    let chunked = chunk_soac_inputs(
        &mut entry.graph,
        &input_view_data,
        total_threads,
        ChunkInputKind::StorageOrRange,
        "Redomap",
    )?;

    // Build a chunked replacement for each input — same shape as the
    // original (StorageView gets new offset/len; ArrayRange gets a new
    // start = old_start + chunk_start, new len = chunk_len, same step).
    let se = &mut entry.graph.skeleton.blocks[block_id].side_effects[idx];
    for (i, &new_view) in chunked.views.iter().enumerate() {
        se.operand_nodes[i] = new_view;
    }

    // Redirect the entry's output store(s) to `partials[tid]`, writing the
    // *whole* result per thread (AoS). A scalar result has one Store of
    // `result_nid`; an AoS tuple result is SoA-decomposed by
    // `emit_compute_output_stores` into per-component / per-element Stores of
    // `Project…(result_nid)`. Find them all (value projects from the result),
    // repoint the first to write `result_nid` into the partials slot, and drop
    // the rest — phase2 reads the AoS partials and writes the AoS result.
    let elem_ty = entry.graph.types[&result_nid].clone();
    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(0),
            // region stamped from the binding by intern_storage_view.
            crate::types::no_region(),
        ],
    );
    let partials_view = graph_ops::intern_storage_view(&mut entry.graph, partials_binding, arr_ty, None);
    let new_place = entry.graph.intern_pure(
        super::types::PureOp::ViewIndex,
        smallvec![partials_view, chunked.tid],
        elem_ty.clone(),
    );

    let mut output_stores: Vec<(BlockId, usize)> = Vec::new();
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            if matches!(&se.kind, SideEffectKind::Inst(InstKind::Store { .. }))
                && se
                    .operand_nodes
                    .get(1)
                    .is_some_and(|&v| value_projects_from(&entry.graph, v, result_nid))
            {
                output_stores.push((bid, i));
            }
        }
    }
    let (keep_block, keep_idx) =
        *output_stores.first().ok_or_else(|| "no Store writes the Redomap result".to_string())?;
    {
        let se = &mut entry.graph.skeleton.blocks[keep_block].side_effects[keep_idx];
        se.operand_nodes[0] = new_place;
        se.operand_nodes[1] = result_nid;
    }
    // Drop the remaining decomposed output stores (highest index first so the
    // earlier indices — including the keeper — stay valid).
    let mut to_remove: Vec<(BlockId, usize)> = output_stores.into_iter().skip(1).collect();
    to_remove.sort_by(|a, b| b.1.cmp(&a.1));
    for (bid, idx) in to_remove {
        entry.graph.skeleton.blocks[bid].side_effects.remove(idx);
    }

    // The original output binding(s) are now unused (phase2 owns the result);
    // clear them and register the partials buffer.
    for o in entry.outputs.iter_mut() {
        o.storage_binding = None;
    }
    entry.storage_bindings.push(crate::interface::StorageBindingDecl {
        binding: crate::BindingRef::new(partials_binding.set, partials_binding.binding),
        role: crate::interface::StorageRole::Intermediate,
        elem_ty,
        length: None,
    });

    Ok(())
}

/// True if `value` is `root` or a chain of `Project`s rooted at `root` — used
/// to recognize the (possibly SoA-decomposed) Stores of a reduce result.
fn value_projects_from(graph: &super::types::EGraph, value: NodeId, root: NodeId) -> bool {
    let mut cur = value;
    loop {
        if cur == root {
            return true;
        }
        match &graph.nodes[cur] {
            super::types::ENode::Pure {
                op: super::types::PureOp::Project { .. },
                operands,
            } => cur = operands[0],
            _ => return false,
        }
    }
}

/// Programmatic phase2 synthesis where the neutral element is a
/// (possibly compound) pure subgraph cloned from phase1. Used by
/// Redomap whose `NO_HIT`-style NE values are tuples / arrays of
/// constants, not scalar literals.
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

fn find_pending_redomap(entry: &EgirEntry) -> Option<(BlockId, usize)> {
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            if let SideEffectKind::Pending(PendingSoac::Redomap { .. }) = &se.kind {
                return Some((bid, i));
            }
        }
    }
    None
}

fn find_pending_reduce(entry: &EgirEntry) -> Option<(BlockId, usize)> {
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            if let SideEffectKind::Pending(PendingSoac::Reduce { .. }) = &se.kind {
                return Some((bid, i));
            }
        }
    }
    None
}

fn find_store_of(entry: &EgirEntry, value_nid: NodeId) -> Option<(BlockId, usize)> {
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            if matches!(&se.kind, SideEffectKind::Inst(InstKind::Store { .. }))
                && se.operand_nodes.get(1) == Some(&value_nid)
            {
                return Some((bid, i));
            }
        }
    }
    None
}

fn find_pending_scan(entry: &EgirEntry) -> Option<(BlockId, usize)> {
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            if let SideEffectKind::Pending(PendingSoac::Scan { .. }) = &se.kind {
                return Some((bid, i));
            }
        }
    }
    None
}

fn find_pending_screma(entry: &EgirEntry) -> Option<(BlockId, usize)> {
    for (bid, block) in &entry.graph.skeleton.blocks {
        for (i, se) in block.side_effects.iter().enumerate() {
            if let SideEffectKind::Pending(PendingSoac::Screma { .. }) = &se.kind {
                return Some((bid, i));
            }
        }
    }
    None
}

/// Mixed-Screma migration dispatcher. Gates and routes to the
/// per-kind transform: Reduce → 2-phase tree-reduce; Scan → 3-phase
/// block-prefix scan. Mirrors the gate in
/// `tlc::parallelize::screma_egir_parallelisable`.
fn transform_screma_entry(
    entry: &mut EgirEntry,
    plan: &ParallelizationPlan,
) -> Option<(Vec<EgirEntry>, Option<EgirFunc>)> {
    use crate::tlc::parallelize::{PlannedBindings, PlannedScremaAccumulatorKind};
    let PlannedBindings::Screma { accumulators, .. } = &plan.bindings else {
        return None;
    };
    if accumulators.len() != 1 {
        return None;
    }
    let all_reduce = accumulators.iter().all(|a| matches!(a.kind, PlannedScremaAccumulatorKind::Reduce));
    let single_scan =
        accumulators.len() == 1 && matches!(accumulators[0].kind, PlannedScremaAccumulatorKind::Scan);
    if all_reduce {
        let phases = transform_screma_reduce_entry(entry, plan)?;
        Some((phases, None))
    } else if single_scan {
        let (phase2, phase3, swap_wrapper) = transform_screma_scan_entry(entry, plan)?;
        Some((vec![phase2, phase3], Some(swap_wrapper)))
    } else {
        None
    }
}

/// 0+ map outputs (no captures) + N>=1 Reduce accumulators (no
/// captures, scalar-literal NEs). The entry is mutated in place into a
/// chunked phase 1 (chunked input + chunked map output views +
/// store-to-`partials_i[tid]` per accumulator). One phase 2 entry is
/// synthesised per accumulator (`{entry}_phase2_combine` for N=1, or
/// `{entry}_phase2_combine_{i}` for N>=2) via
/// `synthesize_phase2_reduce_cloning_ne_named`.
fn transform_screma_reduce_entry(
    entry: &mut EgirEntry,
    plan: &ParallelizationPlan,
) -> Option<Vec<EgirEntry>> {
    use crate::tlc::parallelize::PlannedBindings;

    // 1. Pull binding/dispatch info from the plan.
    let PlannedBindings::Screma {
        map_outputs: _,
        accumulators: planned_accs,
    } = &plan.bindings
    else {
        return None;
    };
    if planned_accs.is_empty() {
        return None;
    }
    let acc_bindings: Vec<(BindingRef, BindingRef)> =
        planned_accs.iter().map(|a| Some((a.partials?, a.result?))).collect::<Option<Vec<_>>>()?;
    let n_accs = planned_accs.len();
    let total_threads = match plan.dispatch {
        crate::tlc::parallelize::DispatchModel::Fixed { local_size, .. } => local_size[0],
        crate::tlc::parallelize::DispatchModel::DerivedFromInputLength { workgroup_size, .. } => {
            workgroup_size
        }
    };

    // 2. Locate the Screma side-effect and pull its metadata.
    let (block_id, idx) = find_pending_screma(entry)?;
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
        let SideEffectKind::Pending(PendingSoac::Screma {
            map_funcs,
            accumulators,
            input_array_types,
            map_destinations,
            acc_destinations,
            map_capture_counts,
            ..
        }) = &se.kind
        else {
            return None;
        };
        if accumulators.len() != n_accs {
            return None;
        }
        if !accumulators.iter().all(|a| matches!(a.kind, crate::tlc::ScremaAccumulator::Reduce)) {
            return None;
        }
        if accumulators.iter().any(|a| a.step_capture_count != 0 || a.reduce_op_capture_count != 0) {
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
        let init_nids: Vec<NodeId> = (0..n_accs).map(|k| se.operand_nodes[n_inputs + k]).collect();
        let base = n_inputs + n_accs;
        let map_view_ops: Vec<usize> = (0..n_maps).map(|m| base + m).collect();
        let result = se.result?;
        let elem_tys: Vec<Type<TypeName>> =
            init_nids.iter().map(|n| entry.graph.types[n].clone()).collect();
        let reduce_funcs: Vec<String> = accumulators.iter().map(|a| a.reduce_op_func.clone()).collect();
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

    // 3. Chunk all input views and every map output view; swap them back
    // into the Screma operand list.
    let chunked = chunk_soac_inputs(
        &mut entry.graph,
        &input_view_data,
        total_threads,
        ChunkInputKind::StorageOrRange,
        "Screma",
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
            &format!("Screma map output {m_idx}"),
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

    // 6. Synthesise one phase 2 entry per accumulator. Cloning NEs from
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
    Some(phase2s)
}

/// If `value` is a chain of `Project`s ending with `root` as the
/// innermost operand, returns the innermost `Project`'s index. Used to
/// match Screma tuple-field Stores back to their accumulator index.
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

/// 0+ map outputs (no captures) + exactly one Scan accumulator with no
/// captures and a scalar-literal NE. Mirrors `transform_scan_entry`'s
/// 3-phase shape, adapted to Screma's operand layout: phase 1 chunks
/// the input + map output views + the scan output view, appends a
/// synthetic chunked Reduce that writes per-thread final accumulators
/// to `block_sums[tid]`; phase 2 sequentially scans `block_sums` into
/// `block_offsets`; phase 3 reads `block_offsets` and applies each
/// chunk's offset back over the scan output buffer. Phase 3 uses an
/// arg-swapped wrapper around the combiner so the scan output's
/// previously-written values land in the `acc` slot.
fn transform_screma_scan_entry(
    entry: &mut EgirEntry,
    plan: &ParallelizationPlan,
) -> Option<(EgirEntry, EgirEntry, EgirFunc)> {
    use crate::tlc::parallelize::PlannedBindings;

    // 1. Pull binding/dispatch info from the plan.
    let PlannedBindings::Screma {
        map_outputs: _,
        accumulators,
    } = &plan.bindings
    else {
        return None;
    };
    if accumulators.len() != 1 {
        return None;
    }
    let acc_planned = &accumulators[0];
    let block_sums_binding = acc_planned.block_sums?;
    let block_offsets_binding = acc_planned.block_offsets?;
    let total_threads = match plan.dispatch {
        crate::tlc::parallelize::DispatchModel::Fixed { local_size, .. } => local_size[0],
        crate::tlc::parallelize::DispatchModel::DerivedFromInputLength { workgroup_size, .. } => {
            workgroup_size
        }
    };

    // 2. Locate the Screma side-effect and pull its metadata.
    let (block_id, idx) = find_pending_screma(entry)?;
    let (
        op_func,
        reduce_func,
        map_output_view_ops,
        scan_output_view_op,
        input_view_nid,
        input_view_ty,
        init_nid,
        elem_ty,
    ) = {
        let se = &entry.graph.skeleton.blocks[block_id].side_effects[idx];
        let SideEffectKind::Pending(PendingSoac::Screma {
            map_funcs,
            accumulators,
            input_array_types,
            map_destinations,
            acc_destinations,
            map_capture_counts,
            ..
        }) = &se.kind
        else {
            return None;
        };
        if accumulators.len() != 1 {
            return None;
        }
        let acc = &accumulators[0];
        if !matches!(acc.kind, crate::tlc::ScremaAccumulator::Scan) {
            return None;
        }
        if acc.step_capture_count != 0 || acc.reduce_op_capture_count != 0 {
            return None;
        }
        if map_capture_counts.iter().any(|&c| c != 0) {
            return None;
        }
        let n_inputs = input_array_types.len();
        if n_inputs != 1 {
            return None;
        }
        if !map_destinations.iter().all(|d| matches!(d, SoacDestination::OutputView)) {
            return None;
        }
        // Realize_outputs retargets the scan accumulator (when its
        // tuple field feeds the entry's output) to OutputView with the
        // scan output buffer appended.
        if !acc_destinations.iter().all(|d| matches!(d, SoacDestination::OutputView)) {
            return None;
        }
        let n_accs = accumulators.len();
        let n_maps = map_funcs.len();
        let input_nid = se.operand_nodes[0];
        let init_nid = se.operand_nodes[n_inputs];
        // Operand layout (gate enforces zero captures everywhere):
        //   [inputs(n_inputs), init_accs(n_accs), map_output_views(n_maps),
        //    acc_output_views(n_accs — all OutputView)]
        let base = n_inputs + n_accs;
        let map_view_ops: Vec<usize> = (0..n_maps).map(|m| base + m).collect();
        let scan_output_view_op = base + n_maps; // 1 acc, OutputView
        let input_ty = entry.graph.types[&input_nid].clone();
        let elem = entry.graph.types[&init_nid].clone();
        (
            acc.step_func.clone(),
            acc.reduce_op_func.clone(),
            map_view_ops,
            scan_output_view_op,
            input_nid,
            input_ty,
            init_nid,
            elem,
        )
    };

    // 3. Chunk the input view, every map output view, and the scan
    // output view; swap each back into the Screma operand list.
    let chunked = chunk_soac_inputs(
        &mut entry.graph,
        &[(input_view_nid, input_view_ty.clone())],
        total_threads,
        ChunkInputKind::StorageOnly,
        "Screma+Scan",
    )
    .ok()?;
    let chunk_start = chunked.chunk_start;
    let chunk_len = chunked.chunk_len;
    let chunked_input_nid = chunked.views[0];
    {
        let se = &mut entry.graph.skeleton.blocks[block_id].side_effects[idx];
        se.operand_nodes[0] = chunked_input_nid;
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
            &format!("Screma map output {m_idx}"),
        )
        .ok()?;
        let se = &mut entry.graph.skeleton.blocks[block_id].side_effects[idx];
        se.operand_nodes[*op_idx] = chunked_view;
    }
    let (scan_output_storage, orig_scan_output_view_ty) = {
        let se = &entry.graph.skeleton.blocks[block_id].side_effects[idx];
        let v = se.operand_nodes[scan_output_view_op];
        let ty = entry.graph.types[&v].clone();
        let storage = graph_ops::extract_storage_view_source(&entry.graph, v)?;
        (storage, ty)
    };
    let chunked_scan_output = graph_ops::intern_chunked_storage_view(
        &mut entry.graph,
        scan_output_storage,
        chunk_start,
        chunk_len,
        orig_scan_output_view_ty,
        None,
    );
    {
        let se = &mut entry.graph.skeleton.blocks[block_id].side_effects[idx];
        se.operand_nodes[scan_output_view_op] = chunked_scan_output;
    }

    // 4. Snapshot the phase 1 graph BEFORE appending the chunked Reduce
    // (so phase 2's NE-clone sees a clean snapshot — the appended
    // Reduce is pure / additive, but the snapshot point matches the
    // plain-Scan path).
    let phase1_snapshot = entry.graph.clone();

    // 5. Append a chunked Reduce side-effect that computes each
    // thread's final accumulator and stores it to `block_sums[tid]`.
    // Operands: [chunked_input, init].
    {
        let mut next_effect = graph_ops::next_effect_token(&entry.graph);
        let reduce_operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![chunked_input_nid, init_nid];
        let result_nid = graph_ops::emit_pending_soac(
            &mut entry.graph,
            block_id,
            PendingSoac::Reduce {
                func: op_func,
                input_array_type: input_view_ty,
                input_elem_type: elem_ty.clone(),
            },
            reduce_operands,
            elem_ty.clone(),
            &mut next_effect,
            None,
        );
        // Store result_nid → block_sums[tid].
        let arr_ty = Type::Constructed(
            TypeName::Array,
            vec![
                elem_ty.clone(),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
                Type::Variable(0),
                crate::types::no_region(),
            ],
        );
        let block_sums_view =
            graph_ops::intern_storage_view(&mut entry.graph, block_sums_binding, arr_ty, None);
        graph_ops::emit_storage_store(
            &mut entry.graph,
            block_id,
            block_sums_view,
            chunked.tid,
            result_nid,
            elem_ty.clone(),
            &mut next_effect,
            None,
        );
    }

    // 6. Register block_sums + block_offsets as Intermediate on phase 1
    // (only block_sums is actually written by phase 1, but declaring
    // block_offsets here mirrors the plain-Scan path so realize_outputs
    // & verifiers stay happy — and phase 3 reads from it without an
    // extra declaration on the entry interface).
    entry.storage_bindings.push(crate::interface::StorageBindingDecl {
        binding: block_sums_binding,
        role: crate::interface::StorageRole::Intermediate,
        elem_ty: elem_ty.clone(),
        length: None,
    });
    entry.storage_bindings.push(crate::interface::StorageBindingDecl {
        binding: block_offsets_binding,
        role: crate::interface::StorageRole::Intermediate,
        elem_ty: elem_ty.clone(),
        length: None,
    });

    // 7. Synthesise phase 2 (sequential scan of block_sums) and phase 3
    // (apply offsets to scan output), reusing the existing Scan helpers.
    let phase2 = synthesize_phase2_scan(
        &entry.name,
        reduce_func.clone(),
        elem_ty.clone(),
        &phase1_snapshot,
        init_nid,
        block_sums_binding,
        block_offsets_binding,
    )
    .ok()?;
    let swap_wrapper_name = format!("{}_scan_op_swap", entry.name);
    let swap_wrapper = synthesize_swap_wrapper(
        swap_wrapper_name.clone(),
        reduce_func,
        elem_ty.clone(),
        entry.span,
    );
    let phase3 = synthesize_phase3_scan(
        &entry.name,
        swap_wrapper_name,
        elem_ty,
        scan_output_storage,
        block_offsets_binding,
        total_threads,
    );

    Some((phase2, phase3, swap_wrapper))
}

/// Orchestrate the Scan migration. Returns `Some(vec![phase2, phase3])`
/// on success; the original entry is mutated in place into phase 1.
///
/// Algorithm (three-phase parallel scan):
/// - Phase 1 (parallel, per-thread chunk): chunked Scan writing local
///   prefix-scans to `output[chunk]`, plus a chunked Reduce computing
///   the final chunk accumulator stored to `block_sums[tid]`. The
///   chunked Reduce duplicates work for cleanness — phase 1 produces
///   ~2N total ops where optimal is ~N. Optimization deferred.
/// - Phase 2 (sequential, 1×1×1 dispatch): standard scan of
///   `block_sums` into `block_offsets`.
/// - Phase 3 (parallel, per-thread chunk): chunked Map applying
///   `op(output[i], off)` to each element. Note the argument order
///   matches Map's convention `op(elem, captures...)`, which is
///   correct only for commutative scan combiners. Non-commutative
///   associative ops (string concat, matmul) would need a swap-args
///   wrapper function — deferred as follow-up work.
fn transform_scan_entry(
    entry: &mut EgirEntry,
    plan: &ParallelizationPlan,
) -> Option<(Vec<EgirEntry>, EgirFunc)> {
    use crate::tlc::parallelize::PlannedBindings;
    let (block_sums_binding, block_offsets_binding) = match plan.bindings {
        PlannedBindings::Scan {
            block_sums,
            block_offsets,
            ..
        } => (block_sums, block_offsets),
        _ => return None,
    };
    let total_threads = match plan.dispatch {
        crate::tlc::parallelize::DispatchModel::Fixed { local_size, .. } => local_size[0],
        crate::tlc::parallelize::DispatchModel::DerivedFromInputLength { workgroup_size, .. } => {
            workgroup_size
        }
    };

    // Pull metadata from the existing PendingSoac::Scan before phase 1
    // mutates the entry: op_func, elem_ty, the init NodeId (for cloning
    // into phase 2), original captures count, and the destination kind.
    // `OutputView` means a separate auto-bound output buffer; `InputBuffer`
    // means writes route back to the input.
    // `op_func` is the per-element step (phase 1, reads raw inputs);
    // `reduce_func` is the pure combiner (phases 2 & 3, which merge already-
    // transformed values). They differ only for a map-fused scan.
    let (op_func, reduce_func, elem_ty, init_nid, captures_count, consuming) = {
        let (block, idx) = find_pending_scan(entry)?;
        let se = &entry.graph.skeleton.blocks[block].side_effects[idx];
        let (func, reduce, elem, dest) = match &se.kind {
            SideEffectKind::Pending(PendingSoac::Scan {
                func,
                reduce_func,
                input_elem_type,
                destination,
                ..
            }) => (
                func.clone(),
                reduce_func.clone(),
                input_elem_type.clone(),
                destination.clone(),
            ),
            _ => return None,
        };
        let n = se.operand_nodes.len();
        let (captures_count, is_consuming) = match dest {
            SoacDestination::OutputView => {
                // [input, init, ...captures, output_view]
                if n < 3 {
                    return None;
                }
                (n - 3, false)
            }
            SoacDestination::InputBuffer => {
                // [input, init, ...captures] — no output_view appended
                if n < 2 {
                    return None;
                }
                (n - 2, true)
            }
            SoacDestination::Fresh => return None,
        };
        let init = se.operand_nodes[1];
        (func, reduce, elem, init, captures_count, is_consuming)
    };

    // Output binding: for consuming scans, write back to the input
    // binding; otherwise use the entry's auto-bound output.
    let output_binding = if consuming {
        let (block, idx) = find_pending_scan(entry)?;
        let input_view_nid = entry.graph.skeleton.blocks[block].side_effects[idx].operand_nodes[0];
        graph_ops::extract_storage_view_source(&entry.graph, input_view_nid)?
    } else {
        entry.outputs.first()?.storage_binding?
    };

    // Phase 2 needs a NE NodeId in its own graph — clone from phase 1's
    // graph BEFORE phase1 mutates the entry (the source NodeIds may grow
    // dependents).
    let phase2 = {
        let phase1_graph_snapshot = entry.graph.clone();
        // Phase 2 scans the per-chunk block sums (already-transformed values),
        // so it uses the pure combiner, not the element-step.
        synthesize_phase2_scan(
            &entry.name,
            reduce_func.clone(),
            elem_ty.clone(),
            &phase1_graph_snapshot,
            init_nid,
            block_sums_binding,
            block_offsets_binding,
        )
        .ok()?
    };

    // Phase 3 adds each block's offset to its already-locally-scanned output
    // elements — again merging transformed values, so the pure combiner.
    let swap_wrapper_name = format!("{}_scan_op_swap", entry.name);
    let swap_wrapper = synthesize_swap_wrapper(
        swap_wrapper_name.clone(),
        reduce_func.clone(),
        elem_ty.clone(),
        entry.span,
    );

    let phase3 = synthesize_phase3_scan(
        &entry.name,
        swap_wrapper_name,
        elem_ty.clone(),
        output_binding,
        block_offsets_binding,
        total_threads,
    );

    phase1_transform_scan(
        entry,
        total_threads,
        op_func,
        elem_ty,
        captures_count,
        block_sums_binding,
    )
    .ok()?;

    // Phase 1 wrote a partial+local-scan to `output` and final chunk
    // accumulators to `block_sums`; the entry's auto-bound output
    // (`output_binding`) is now reused by phase 3, which reads from it
    // and writes the corrected prefix back. Declare the additional
    // intermediate binding on the entry interface.
    entry.storage_bindings.push(crate::interface::StorageBindingDecl {
        binding: crate::BindingRef::new(block_offsets_binding.set, block_offsets_binding.binding),
        role: crate::interface::StorageRole::Intermediate,
        elem_ty: entry.storage_bindings[0].elem_ty.clone(),
        length: None,
    });
    let _ = output_binding;

    Some((vec![phase2, phase3], swap_wrapper))
}

/// In-place rewrite of the original entry into "phase 1": the existing
/// `PendingSoac::Scan{OutputView}` is chunked (input and output_view
/// operands swapped for chunked StorageViews), and a chunked Reduce +
/// Store sequence is appended so each thread also writes its final
/// chunk accumulator to `block_sums[tid]`. The duplicate Reduce reads
/// the same chunked input as the Scan — ~2N total ops. Acceptable
/// tradeoff for avoiding hand-rolled EGIR loops; optimization deferred.
pub fn phase1_transform_scan(
    entry: &mut EgirEntry,
    total_threads: u32,
    op_func: String,
    elem_ty: Type<TypeName>,
    captures_count: usize,
    block_sums_binding: BindingRef,
) -> Result<(), String> {
    let (scan_block, scan_idx) =
        find_pending_scan(entry).ok_or_else(|| "no PendingSoac::Scan in entry".to_string())?;

    // Snapshot operands and destination before we mutate.
    let (input_view_nid, input_view_ty, init_nid, captures_vec, consuming) = {
        let se = &entry.graph.skeleton.blocks[scan_block].side_effects[scan_idx];
        let consuming = matches!(
            &se.kind,
            SideEffectKind::Pending(PendingSoac::Scan {
                destination: SoacDestination::InputBuffer,
                ..
            }),
        );
        let input_view = se.operand_nodes[0];
        let init = se.operand_nodes[1];
        let captures: Vec<NodeId> = se.operand_nodes[2..2 + captures_count].to_vec();
        let input_ty = entry.graph.types[&input_view].clone();
        (input_view, input_ty, init, captures, consuming)
    };

    let chunked = chunk_soac_inputs(
        &mut entry.graph,
        &[(input_view_nid, input_view_ty.clone())],
        total_threads,
        ChunkInputKind::StorageOnly,
        "Scan",
    )?;
    let chunked_input = chunked.views[0];

    // Replace the Scan's input operand with the chunked view. For
    // `OutputView` destination, also chunk the appended `output_view`
    // operand. For `InputBuffer` destination, soac_expand writes back
    // to chunked_input[i] (which IS output[chunk_start+i]) automatically
    // — no separate output_view to chunk.
    {
        let se = &mut entry.graph.skeleton.blocks[scan_block].side_effects[scan_idx];
        se.operand_nodes[0] = chunked_input;
    }
    if !consuming {
        let (output_view_nid, output_view_ty) = {
            let se = &entry.graph.skeleton.blocks[scan_block].side_effects[scan_idx];
            let n = se.operand_nodes.len();
            let v = se.operand_nodes[n - 1];
            (v, entry.graph.types[&v].clone())
        };
        let output_storage = graph_ops::extract_storage_view_source(&entry.graph, output_view_nid)
            .ok_or_else(|| "Scan output_view is not a StorageView".to_string())?;
        let chunked_output = graph_ops::intern_chunked_storage_view(
            &mut entry.graph,
            output_storage,
            chunked.chunk_start,
            chunked.chunk_len,
            output_view_ty,
            None,
        );
        let se = &mut entry.graph.skeleton.blocks[scan_block].side_effects[scan_idx];
        let last = se.operand_nodes.len() - 1;
        se.operand_nodes[last] = chunked_output;
    }

    // Add a chunked Reduce computing the per-thread final accumulator.
    // Operands: [chunked_input, init, ...captures].
    let mut reduce_operands: smallvec::SmallVec<[NodeId; 4]> = smallvec![chunked_input, init_nid];
    reduce_operands.extend(captures_vec.iter().copied());
    let final_acc = {
        let mut next_effect = graph_ops::next_effect_token(&entry.graph);
        let result = graph_ops::emit_pending_soac(
            &mut entry.graph,
            scan_block,
            PendingSoac::Reduce {
                func: op_func,
                input_array_type: input_view_ty,
                input_elem_type: elem_ty.clone(),
            },
            reduce_operands,
            elem_ty.clone(),
            &mut next_effect,
            None,
        );
        result
    };

    // Store the final accumulator to block_sums[tid].
    {
        let mut next_effect = graph_ops::next_effect_token(&entry.graph);
        let arr_ty = Type::Constructed(
            TypeName::Array,
            vec![
                elem_ty.clone(),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
                Type::Variable(0),
                // region stamped from the binding by intern_storage_view.
                crate::types::no_region(),
            ],
        );
        let block_sums_view =
            graph_ops::intern_storage_view(&mut entry.graph, block_sums_binding, arr_ty, None);
        graph_ops::emit_storage_store(
            &mut entry.graph,
            scan_block,
            block_sums_view,
            chunked.tid,
            final_acc,
            elem_ty.clone(),
            &mut next_effect,
            None,
        );
    }

    entry.storage_bindings.push(crate::interface::StorageBindingDecl {
        binding: crate::BindingRef::new(block_sums_binding.set, block_sums_binding.binding),
        role: crate::interface::StorageRole::Intermediate,
        elem_ty,
        length: None,
    });
    Ok(())
}

/// Synthesize phase 2: a `1×1×1` compute entry that runs a sequential
/// scan over `block_sums` and writes the prefixes into `block_offsets`.
/// Uses `PendingSoac::Scan{OutputView}` so `soac_expand` handles the
/// loop.
pub fn synthesize_phase2_scan(
    entry_name: &str,
    op_func: String,
    elem_ty: Type<TypeName>,
    phase1_graph: &super::types::EGraph,
    phase1_ne_nid: NodeId,
    block_sums_binding: BindingRef,
    block_offsets_binding: BindingRef,
) -> Result<EgirEntry, String> {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(format!("{}_phase2_scan_sums", entry_name), (1, 1, 1));
    b.declare_intermediate_storage(block_sums_binding, elem_ty.clone());
    b.declare_intermediate_storage(block_offsets_binding, elem_ty.clone());

    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(0),
            // region stamped from the binding by intern_storage_view.
            crate::types::no_region(),
        ],
    );
    let block_sums_view = b.emit_storage_view(block_sums_binding, arr_ty.clone());
    let block_offsets_view = b.emit_storage_view(block_offsets_binding, arr_ty.clone());
    let init_nid = graph_ops::clone_pure_subgraph(phase1_graph, b.graph_mut(), phase1_ne_nid)?;
    b.emit_pending_scan_into(
        op_func,
        block_sums_view,
        arr_ty,
        elem_ty,
        init_nid,
        vec![],
        block_offsets_view,
    );
    Ok(b.build())
}

/// Synthesize phase 3: a chunked compute entry where each thread reads
/// `off = block_offsets[tid]`, then applies `op(off, output[i])` to
/// each element of `output[chunk]`. Map's body-call convention is
/// `func(elem, ...captures)`, which would give `op(elem, off)` —
/// silently wrong for non-commutative associative ops (string concat,
/// matmul). Caller supplies the name of a swap-args wrapper synthesized
/// alongside this entry (see `synthesize_scan_swap_wrapper`); the
/// wrapper's body is `\(elem, off) -> op(off, elem)`, so Map's normal
/// convention combined with the wrapper yields the correct order.
pub fn synthesize_phase3_scan(
    entry_name: &str,
    swap_wrapper_name: String,
    elem_ty: Type<TypeName>,
    output_binding: BindingRef,
    block_offsets_binding: BindingRef,
    total_threads: u32,
) -> EgirEntry {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(
        format!("{}_phase3_add_offsets", entry_name),
        (total_threads, 1, 1),
    );
    b.declare_output_storage(output_binding, elem_ty.clone());
    b.declare_intermediate_storage(block_offsets_binding, elem_ty.clone());

    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
            Type::Variable(0),
            // region stamped from the binding by intern_storage_view.
            crate::types::no_region(),
        ],
    );
    let _output_view = b.emit_storage_view(output_binding, arr_ty.clone());
    let block_offsets_view = b.emit_storage_view(block_offsets_binding, arr_ty.clone());

    // tid, chunk_start, chunk_len from output length.
    let output_len = {
        let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let set_nid = graph_ops::intern_u32(b.graph_mut(), output_binding.set, None);
        let binding_nid = graph_ops::intern_u32(b.graph_mut(), output_binding.binding, None);
        graph_ops::intern_intrinsic(
            b.graph_mut(),
            catalog().known().storage_len,
            smallvec![set_nid, binding_nid],
            u32_ty,
            None,
        )
    };
    let (tid, chunk_start, chunk_len) = emit_chunk_arithmetic(b.graph_mut(), total_threads, output_len)
        .expect("phase3: chunk arithmetic must succeed (u32.min is in the prelude)");

    // off = block_offsets[tid]: ViewIndex + Load.
    let off_place = b.graph_mut().intern_pure(
        super::types::PureOp::ViewIndex,
        smallvec![block_offsets_view, tid],
        elem_ty.clone(),
    );
    let off = b.emit_load(off_place, elem_ty.clone());

    let chunked_output = graph_ops::intern_chunked_storage_view(
        b.graph_mut(),
        output_binding,
        chunk_start,
        chunk_len,
        arr_ty.clone(),
        None,
    );

    // Call the swap-args wrapper instead of `op` directly: Map emits
    // `wrapper(elem, off)`, which the wrapper rewrites to `op(off, elem)`.
    b.emit_pending_map_into(
        swap_wrapper_name,
        chunked_output,
        arr_ty.clone(),
        elem_ty.clone(),
        elem_ty,
        vec![off],
        chunked_output,
    );
    b.build()
}

/// Build a two-argument helper EgirFunc named `wrapper_name` whose body
/// is `inner(b, a)` — a swap-args wrapper around an existing
/// `T -> T -> T` function.
fn synthesize_swap_wrapper(
    wrapper_name: String,
    inner: String,
    elem_ty: Type<TypeName>,
    span: crate::ast::Span,
) -> EgirFunc {
    let mut graph = EGraph::new();
    let a_nid = graph.add_func_param(0, elem_ty.clone());
    let b_nid = graph.add_func_param(1, elem_ty.clone());
    let call_nid = graph.intern_pure(PureOp::Call(inner), smallvec![b_nid, a_nid], elem_ty.clone());
    let entry_block = graph.skeleton.entry;
    graph.skeleton.blocks[entry_block].term = SkeletonTerminator::Return(Some(call_nid));
    EgirFunc::new(
        wrapper_name,
        span,
        None,
        vec![
            (elem_ty.clone(), "a".to_string()),
            (elem_ty.clone(), "b".to_string()),
        ],
        elem_ty,
        graph,
        HashMap::new(),
    )
}
