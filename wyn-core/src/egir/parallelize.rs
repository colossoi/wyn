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

use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ConstantValue, InstKind};

use super::graph_ops;
use super::program::{EgirEntry, EgirFunc, EgirInner};
use super::types::{
    EGraph, NodeId, PendingSoac, PureOp, SideEffectKind, SkeletonTerminator, SoacDestination,
};
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
            ParallelStrategy::Reduce => {
                if let Some(phase2) = transform_reduce_entry(entry, plan) {
                    new_entries.push(phase2);
                }
            }
            ParallelStrategy::Redomap => {
                if let Some(phase2) = transform_redomap_entry(entry, plan) {
                    new_entries.push(phase2);
                }
            }
            ParallelStrategy::Scan => {
                if let Some((phases, swap_wrapper)) = transform_scan_entry(entry, plan) {
                    new_entries.extend(phases);
                    new_functions.push(swap_wrapper);
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
            // Only the entry's tail Map (which targets the auto-bound
            // OutputView) is parallelizable. Intermediate Maps that
            // produce fresh arrays for downstream consumers (e.g. a
            // Map → Reduce fusion that hasn't fused) must stay serial.
            match pending {
                PendingSoac::Map {
                    destination: SoacDestination::OutputView,
                    ..
                } => {}
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

/// Orchestrate the Reduce migration for one entry: extract the
/// op-func, the init constant, and the partials/result bindings the
/// TLC plan reserved; then run `phase1_transform_reduce` on the
/// existing entry and build a fresh phase2 via
/// `synthesize_phase2_reduce`.
///
/// Returns `Some(phase2_entry)` on success, `None` if the entry
/// doesn't match the expected reduce shape (caller leaves it alone
/// and reduce keeps flowing through the TLC-side synthesis path).
fn transform_reduce_entry(entry: &mut EgirEntry, plan: &ParallelizationPlan) -> Option<EgirEntry> {
    use crate::tlc::parallelize::PlannedBindings;
    let (partials_binding, result_binding) = match plan.bindings {
        PlannedBindings::Reduce { partials, result } => (partials, result),
        _ => return None,
    };
    let total_threads = match plan.dispatch {
        crate::tlc::parallelize::DispatchModel::Fixed { local_size, .. } => local_size[0],
        crate::tlc::parallelize::DispatchModel::DerivedFromInputLength { workgroup_size, .. } => {
            workgroup_size
        }
    };

    // Inspect the entry to pull out the op-func name + init constant
    // BEFORE phase1 mutates the body.
    let (op_func, init_const, elem_ty) = {
        let (block, idx) = find_pending_reduce(entry)?;
        let se = &entry.graph.skeleton.blocks[block].side_effects[idx];
        let func = match &se.kind {
            SideEffectKind::Pending(PendingSoac::Reduce { func, .. }) => func.clone(),
            _ => return None,
        };
        let init_nid = *se.operand_nodes.get(1)?;
        let init_const = extract_constant(&entry.graph, init_nid)?;
        let result_nid = se.result?;
        let elem_ty = entry.graph.types[&result_nid].clone();
        (func, init_const, elem_ty)
    };

    phase1_transform_reduce(entry, total_threads, partials_binding).ok()?;

    let phase2 = synthesize_phase2_reduce(
        &entry.name,
        op_func,
        elem_ty,
        init_const,
        partials_binding,
        result_binding,
    );
    Some(phase2)
}

/// Orchestrate the Redomap migration. Mirrors `transform_reduce_entry`
/// but uses Redomap's `reduce_func` (the pure combiner for cross-thread
/// merging in phase 2) and clones the NE subgraph rather than expecting
/// a scalar `ConstantValue`.
fn transform_redomap_entry(entry: &mut EgirEntry, plan: &ParallelizationPlan) -> Option<EgirEntry> {
    use crate::tlc::parallelize::PlannedBindings;
    let (partials_binding, result_binding) = match plan.bindings {
        PlannedBindings::Redomap { partials, result } => (partials, result),
        _ => return None,
    };
    let total_threads = match plan.dispatch {
        crate::tlc::parallelize::DispatchModel::Fixed { local_size, .. } => local_size[0],
        crate::tlc::parallelize::DispatchModel::DerivedFromInputLength { workgroup_size, .. } => {
            workgroup_size
        }
    };

    // Before phase1 mutates the body, pull out: reduce_func (the
    // cross-thread combiner — distinct from `func`, the per-element
    // op), the NE NodeId in phase1's graph, and the element type.
    let (reduce_func, ne_nid, elem_ty) = {
        let (block, idx) = find_pending_redomap(entry)?;
        let se = &entry.graph.skeleton.blocks[block].side_effects[idx];
        let (input_count, reduce_func) = match &se.kind {
            SideEffectKind::Pending(PendingSoac::Redomap {
                reduce_func,
                input_array_types,
                ..
            }) => (input_array_types.len(), reduce_func.clone()),
            _ => return None,
        };
        // Operand layout: [input_0, ..., input_{n-1}, init, ...captures].
        let init_nid = *se.operand_nodes.get(input_count)?;
        let result_nid = se.result?;
        let elem_ty = entry.graph.types[&result_nid].clone();
        (reduce_func, init_nid, elem_ty)
    };

    // Snapshot of the NE subgraph for cloning into phase2. We do this
    // before phase1 transforms the entry (the source NodeIds may grow
    // dependents).
    let phase2 = {
        let phase1_graph_snapshot = entry.graph.clone();
        synthesize_phase2_reduce_cloning_ne(
            &entry.name,
            reduce_func,
            elem_ty,
            &phase1_graph_snapshot,
            ne_nid,
            partials_binding,
            result_binding,
        )
        .ok()?
    };

    phase1_transform_redomap(entry, total_threads, partials_binding).ok()?;
    Some(phase2)
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
    partials_binding: (u32, u32),
) -> Result<(), String> {
    // 1. Locate the Reduce side-effect and pull the operands.
    let (reduce_block, reduce_idx) =
        find_pending_reduce(entry).ok_or_else(|| "no PendingSoac::Reduce in entry".to_string())?;
    let (result_nid, input_view_ty, view_storage) = {
        let se = &entry.graph.skeleton.blocks[reduce_block].side_effects[reduce_idx];
        let input = se.operand_nodes[0];
        let result = se.result.ok_or_else(|| "Reduce missing result".to_string())?;
        let input_ty = entry.graph.types[&input].clone();
        let storage = graph_ops::extract_storage_view_source(&entry.graph, input)
            .ok_or_else(|| "Reduce input is not a StorageView".to_string())?;
        (result, input_ty, storage)
    };

    // 2. Build chunk arithmetic in the same block as the reduce.
    let input_len = emit_storage_len(&mut entry.graph, view_storage.0, view_storage.1);
    let (tid, chunk_start, chunk_len) = emit_chunk_arithmetic(&mut entry.graph, total_threads, input_len)?;

    // 3. Build a new chunked StorageView with [chunk_start, chunk_len].
    let chunked_view = graph_ops::intern_chunked_storage_view(
        &mut entry.graph,
        view_storage.0,
        view_storage.1,
        chunk_start,
        chunk_len,
        input_view_ty,
        None,
    );

    // 4. Swap the Reduce's input operand.
    entry.graph.skeleton.blocks[reduce_block].side_effects[reduce_idx].operand_nodes[0] = chunked_view;

    // 5. Find the auto-output Store (writes the reduce result), rewrite
    //    its place to ViewIndex(partials_view, tid).
    let (store_block, store_idx) =
        find_store_of(entry, result_nid).ok_or_else(|| "no Store writes the Reduce result".to_string())?;
    let elem_ty = entry.graph.types[&result_nid].clone();
    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Variable(0),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );
    let partials_view = graph_ops::intern_storage_view(
        &mut entry.graph,
        partials_binding.0,
        partials_binding.1,
        arr_ty,
        None,
    );
    let new_place = entry.graph.intern_pure(
        super::types::PureOp::ViewIndex,
        smallvec![partials_view, tid],
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
        set: partials_binding.0,
        binding: partials_binding.1,
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
    partials_binding: (u32, u32),
    result_binding: (u32, u32),
) -> EgirEntry {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(format!("{}_phase2_combine", entry_name), (1, 1, 1));
    b.declare_intermediate_storage(partials_binding.0, partials_binding.1, elem_ty.clone());
    b.declare_output_storage(result_binding.0, result_binding.1, elem_ty.clone());

    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Variable(0),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );
    let view = b.emit_storage_view(partials_binding.0, partials_binding.1, arr_ty.clone());
    let init_nid = b.emit_constant(init, elem_ty.clone());
    let r = b.emit_pending_reduce(op_func, view, arr_ty, elem_ty.clone(), init_nid, vec![]);
    let zero = b.emit_u32(0);
    b.emit_storage_store(result_binding.0, result_binding.1, zero, r, elem_ty);
    b.build()
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
fn emit_storage_len(graph: &mut super::types::EGraph, set: u32, binding: u32) -> NodeId {
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let set_nid = graph_ops::intern_u32(graph, set, None);
    let binding_nid = graph_ops::intern_u32(graph, binding, None);
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
    partials_binding: (u32, u32),
) -> Result<(), String> {
    let (block_id, idx) =
        find_pending_redomap(entry).ok_or_else(|| "no PendingSoac::Redomap in entry".to_string())?;

    let (input_count, input_view_data, result_nid) = {
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
        (n, view_data, result)
    };

    // Chunk arith uses the first input's length.
    let first_view_nid = input_view_data[0].0;
    let input_len = if let Some((set, binding)) =
        graph_ops::extract_storage_view_source(&entry.graph, first_view_nid)
    {
        emit_storage_len(&mut entry.graph, set, binding)
    } else if let Some((_, len_nid, _)) =
        graph_ops::extract_array_range_operands(&entry.graph, first_view_nid)
    {
        len_nid
    } else {
        return Err("phase1 Redomap: first input is neither StorageView nor ArrayRange".into());
    };
    let (tid, chunk_start, chunk_len) = emit_chunk_arithmetic(&mut entry.graph, total_threads, input_len)?;

    // Build a chunked replacement for each input — same shape as the
    // original (StorageView gets new offset/len; ArrayRange gets a new
    // start = old_start + chunk_start, new len = chunk_len, same step).
    let mut new_views: Vec<NodeId> = Vec::with_capacity(input_count);
    for (view_nid, view_ty) in input_view_data {
        let chunked =
            if let Some((set, binding)) = graph_ops::extract_storage_view_source(&entry.graph, view_nid) {
                graph_ops::intern_chunked_storage_view(
                    &mut entry.graph,
                    set,
                    binding,
                    chunk_start,
                    chunk_len,
                    view_ty,
                    None,
                )
            } else if let Some((orig_start, _, step)) =
                graph_ops::extract_array_range_operands(&entry.graph, view_nid)
            {
                let has_step = step.is_some();
                let start_ty = entry.graph.types[&orig_start].clone();
                let new_start =
                    graph_ops::intern_binop(&mut entry.graph, "+", orig_start, chunk_start, start_ty, None);
                let mut ops: smallvec::SmallVec<[NodeId; 4]> = smallvec![new_start, chunk_len];
                if let Some(s) = step {
                    ops.push(s);
                }
                entry.graph.intern_pure(super::types::PureOp::ArrayRange { has_step }, ops, view_ty)
            } else {
                return Err("phase1 Redomap: input neither StorageView nor ArrayRange".into());
            };
        new_views.push(chunked);
    }

    let se = &mut entry.graph.skeleton.blocks[block_id].side_effects[idx];
    for (i, new_view) in new_views.into_iter().enumerate() {
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
            Type::Variable(0),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );
    let partials_view = graph_ops::intern_storage_view(
        &mut entry.graph,
        partials_binding.0,
        partials_binding.1,
        arr_ty,
        None,
    );
    let new_place = entry.graph.intern_pure(
        super::types::PureOp::ViewIndex,
        smallvec![partials_view, tid],
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
        set: partials_binding.0,
        binding: partials_binding.1,
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
    partials_binding: (u32, u32),
    result_binding: (u32, u32),
) -> Result<EgirEntry, String> {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(format!("{}_phase2_combine", entry_name), (1, 1, 1));
    b.declare_intermediate_storage(partials_binding.0, partials_binding.1, elem_ty.clone());
    b.declare_output_storage(result_binding.0, result_binding.1, elem_ty.clone());

    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Variable(0),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );
    let view = b.emit_storage_view(partials_binding.0, partials_binding.1, arr_ty.clone());
    let init_nid = graph_ops::clone_pure_subgraph(phase1_graph, b.graph_mut(), phase1_ne_nid)?;
    let r = b.emit_pending_reduce(op_func, view, arr_ty, elem_ty.clone(), init_nid, vec![]);
    let zero = b.emit_u32(0);
    b.emit_storage_store(result_binding.0, result_binding.1, zero, r, elem_ty);
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
    let (op_func, elem_ty, init_nid, captures_count, consuming) = {
        let (block, idx) = find_pending_scan(entry)?;
        let se = &entry.graph.skeleton.blocks[block].side_effects[idx];
        let (func, elem, dest) = match &se.kind {
            SideEffectKind::Pending(PendingSoac::Scan {
                func,
                input_elem_type,
                destination,
                ..
            }) => (func.clone(), input_elem_type.clone(), destination.clone()),
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
        (func, elem, init, captures_count, is_consuming)
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
        synthesize_phase2_scan(
            &entry.name,
            op_func.clone(),
            elem_ty.clone(),
            &phase1_graph_snapshot,
            init_nid,
            block_sums_binding,
            block_offsets_binding,
        )
        .ok()?
    };

    let swap_wrapper_name = format!("{}_scan_op_swap", entry.name);
    let swap_wrapper = synthesize_swap_wrapper(
        swap_wrapper_name.clone(),
        op_func.clone(),
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
        set: block_offsets_binding.0,
        binding: block_offsets_binding.1,
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
    block_sums_binding: (u32, u32),
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

    let input_storage = graph_ops::extract_storage_view_source(&entry.graph, input_view_nid)
        .ok_or_else(|| "Scan input is not a StorageView".to_string())?;

    let input_len = emit_storage_len(&mut entry.graph, input_storage.0, input_storage.1);
    let (tid, chunk_start, chunk_len) = emit_chunk_arithmetic(&mut entry.graph, total_threads, input_len)?;

    let chunked_input = graph_ops::intern_chunked_storage_view(
        &mut entry.graph,
        input_storage.0,
        input_storage.1,
        chunk_start,
        chunk_len,
        input_view_ty.clone(),
        None,
    );

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
            output_storage.0,
            output_storage.1,
            chunk_start,
            chunk_len,
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
                Type::Variable(0),
                Type::Constructed(TypeName::ArrayVariantView, vec![]),
            ],
        );
        let block_sums_view = graph_ops::intern_storage_view(
            &mut entry.graph,
            block_sums_binding.0,
            block_sums_binding.1,
            arr_ty,
            None,
        );
        graph_ops::emit_storage_store(
            &mut entry.graph,
            scan_block,
            block_sums_view,
            tid,
            final_acc,
            elem_ty.clone(),
            &mut next_effect,
            None,
        );
    }

    entry.storage_bindings.push(crate::interface::StorageBindingDecl {
        set: block_sums_binding.0,
        binding: block_sums_binding.1,
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
    block_sums_binding: (u32, u32),
    block_offsets_binding: (u32, u32),
) -> Result<EgirEntry, String> {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(format!("{}_phase2_scan_sums", entry_name), (1, 1, 1));
    b.declare_intermediate_storage(block_sums_binding.0, block_sums_binding.1, elem_ty.clone());
    b.declare_intermediate_storage(block_offsets_binding.0, block_offsets_binding.1, elem_ty.clone());

    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Variable(0),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );
    let block_sums_view = b.emit_storage_view(block_sums_binding.0, block_sums_binding.1, arr_ty.clone());
    let block_offsets_view =
        b.emit_storage_view(block_offsets_binding.0, block_offsets_binding.1, arr_ty.clone());
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
    output_binding: (u32, u32),
    block_offsets_binding: (u32, u32),
    total_threads: u32,
) -> EgirEntry {
    use super::builder::EntryBuilder;
    let mut b = EntryBuilder::new_compute(
        format!("{}_phase3_add_offsets", entry_name),
        (total_threads, 1, 1),
    );
    b.declare_output_storage(output_binding.0, output_binding.1, elem_ty.clone());
    b.declare_intermediate_storage(block_offsets_binding.0, block_offsets_binding.1, elem_ty.clone());

    let arr_ty = Type::Constructed(
        TypeName::Array,
        vec![
            elem_ty.clone(),
            Type::Variable(0),
            Type::Constructed(TypeName::ArrayVariantView, vec![]),
        ],
    );
    let _output_view = b.emit_storage_view(output_binding.0, output_binding.1, arr_ty.clone());
    let block_offsets_view =
        b.emit_storage_view(block_offsets_binding.0, block_offsets_binding.1, arr_ty.clone());

    // tid, chunk_start, chunk_len from output length.
    let output_len = {
        let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
        let set_nid = graph_ops::intern_u32(b.graph_mut(), output_binding.0, None);
        let binding_nid = graph_ops::intern_u32(b.graph_mut(), output_binding.1, None);
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
        output_binding.0,
        output_binding.1,
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
