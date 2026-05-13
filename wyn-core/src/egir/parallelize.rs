//! EGIR-side SOAC parallelization.
//!
//! Consumes the `ParallelizationPlan`s produced by `tlc::parallelize`:
//! Map entries get their tail SOAC wrapped in `PendingSoac::Parallel`
//! for the lane-indexed kernel; Reduce and Redomap entries get an
//! in-place phase1 rewrite (chunked input + store-to-partials) plus a
//! freshly-synthesized phase2-combine EgirEntry. Scan still flows
//! through the TLC-side three-phase synthesis path.
use std::collections::HashMap;

use polytype::Type;
use smallvec::smallvec;

use crate::ast::TypeName;
use crate::builtins::catalog;
use crate::ssa::framework::BlockId;
use crate::ssa::types::{ConstantValue, InstKind};

use super::graph_ops;
use super::program::{EgirEntry, EgirInner};
use super::types::{NodeId, PendingSoac, SideEffectKind, SoacDestination};
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
/// Scan    → still flows through TLC-side three-phase synthesis.
pub fn run(inner: &mut EgirInner, plans: &HashMap<String, ParallelizationPlan>) {
    // Collect entries-to-add separately so we don't mutate
    // `inner.entry_points` while iterating it.
    let mut new_entries: Vec<EgirEntry> = Vec::new();
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
                // Scan still goes through TLC-side three-phase synthesis.
            }
        }
    }
    inner.entry_points.extend(new_entries);
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
    let tid = graph_ops::intern_intrinsic(
        graph,
        catalog().known().thread_id,
        smallvec![],
        u32_ty.clone(),
        None,
    );
    let total = graph_ops::intern_u32(graph, total_threads, None);
    let one = graph_ops::intern_u32(graph, 1, None);
    let total_minus_one = graph_ops::intern_binop(graph, "-", total, one, u32_ty.clone(), None);
    let len_plus = graph_ops::intern_binop(graph, "+", input_len, total_minus_one, u32_ty.clone(), None);
    let chunk_size = graph_ops::intern_binop(graph, "/", len_plus, total, u32_ty.clone(), None);
    let chunk_start = graph_ops::intern_binop(graph, "*", tid, chunk_size, u32_ty.clone(), None);
    let remaining = graph_ops::intern_binop(graph, "-", input_len, chunk_start, u32_ty.clone(), None);
    let u32_min =
        catalog().lookup_by_any_name("u32.min").ok_or_else(|| "u32.min not in catalog".to_string())?;
    let chunk_len =
        graph_ops::intern_intrinsic(graph, u32_min.id, smallvec![chunk_size, remaining], u32_ty, None);
    Ok((tid, chunk_start, chunk_len))
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

    // Redirect the auto-output Store to partials[tid].
    let (store_block, store_idx) =
        find_store_of(entry, result_nid).ok_or_else(|| "no Store writes the Redomap result".to_string())?;
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

    // Clear outputs[0].storage_binding (phase2 takes it); register partials.
    if let Some(o) = entry.outputs.get_mut(0) {
        o.storage_binding = None;
    }
    entry.storage_bindings.push(crate::interface::StorageBindingDecl {
        set: partials_binding.0,
        binding: partials_binding.1,
        role: crate::interface::StorageRole::Intermediate,
        elem_ty,
    });

    Ok(())
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
