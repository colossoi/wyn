//! EGIR-side SOAC parallelization.
//!
//! Consumes the `ParallelizationPlan`s produced by `tlc::parallelize`:
//! Map entries get their tail SOAC wrapped in `PendingSoac::Parallel`
//! for the lane-indexed kernel; Reduce entries get an in-place phase1
//! rewrite (chunked reduce + store-to-partials) plus a freshly-
//! synthesized phase2-combine EgirEntry. Redomap and Scan still flow
//! through the TLC-side phase-synthesis path.
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
/// Map  → wrap the tail SOAC in `PendingSoac::Parallel` so `soac_expand`
///        emits the lane-indexed kernel.
/// Reduce → in-place phase1 rewrite (chunked reduce + store-to-partials),
///          then synthesize a phase2 EgirEntry and append it.
/// Other strategies (Redomap, Scan) are skipped — they still flow
/// through the TLC-side phase-entry synthesis.
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
            ParallelStrategy::Redomap | ParallelStrategy::Scan => {
                // Pending: redomap and scan still go through TLC-side
                // phase synthesis. See greedy-baking-elephant plan.
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

    // 2. Build chunk arithmetic in the same block as the reduce. All
    //    emission goes through `graph_ops`; no in-module helpers.
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let g = &mut entry.graph;
    let tid =
        graph_ops::intern_intrinsic(g, catalog().known().thread_id, smallvec![], u32_ty.clone(), None);
    let total = graph_ops::intern_u32(g, total_threads, None);
    let set_nid = graph_ops::intern_u32(g, view_storage.0, None);
    let binding_nid = graph_ops::intern_u32(g, view_storage.1, None);
    let input_len = graph_ops::intern_intrinsic(
        g,
        catalog().known().storage_len,
        smallvec![set_nid, binding_nid],
        u32_ty.clone(),
        None,
    );
    let one = graph_ops::intern_u32(g, 1, None);
    let total_minus_one = graph_ops::intern_binop(g, "-", total, one, u32_ty.clone(), None);
    let len_plus = graph_ops::intern_binop(g, "+", input_len, total_minus_one, u32_ty.clone(), None);
    let chunk_size = graph_ops::intern_binop(g, "/", len_plus, total, u32_ty.clone(), None);
    let chunk_start = graph_ops::intern_binop(g, "*", tid, chunk_size, u32_ty.clone(), None);
    let remaining = graph_ops::intern_binop(g, "-", input_len, chunk_start, u32_ty.clone(), None);
    let u32_min =
        catalog().lookup_by_any_name("u32.min").ok_or_else(|| "u32.min not in catalog".to_string())?;
    let chunk_len = graph_ops::intern_intrinsic(
        g,
        u32_min.id,
        smallvec![chunk_size, remaining],
        u32_ty.clone(),
        None,
    );

    // 3. Build a new chunked StorageView with [chunk_start, chunk_len].
    let chunked_view = graph_ops::intern_chunked_storage_view(
        g,
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
