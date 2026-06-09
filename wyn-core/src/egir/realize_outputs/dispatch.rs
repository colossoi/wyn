//! Per-source DPS dispatch — the shape classifier and emission helpers.
//!
//! Two entry points, matching the two cases in `realize_outputs`:
//!
//!   * `compute_slot_source` — classifies a slot source NodeId and
//!     emits the appropriate write into a storage `OutputView`. Handles
//!     SOAC retargeting (`Map(Fresh)` / `Scan(Fresh)` → `OutputView`),
//!     fixed-size aggregates (per-element stores), scalars/vectors
//!     (single store at index 0), and consuming Scan (no-op — already
//!     in its input buffer).
//!   * `graphics_slot_source` — emits one store to the slot's
//!     `OutputSlot { index }` place. Graphics outputs are scalar /
//!     vector / matrix in practice; if a runtime-sized array ever
//!     surfaces here it's a contract violation upstream.

use polytype::Type;
use smallvec::smallvec;
use std::collections::HashMap;

use crate::BindingRef;
use crate::ast::TypeName;
use crate::ssa::framework::BlockId;
use crate::types::TypeExt;

use super::super::from_tlc::ConvertError;
use super::super::graph_ops;
use super::super::types::{EGraph, ENode, NodeId, PendingSoac, PureOp, SideEffectKind, SoacDestination};

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod dispatch_tests;

/// Realise one compute slot source as a DPS write into the slot's
/// `OutputView { binding }`. Classification:
///
///   1. Consuming `Scan(InputBuffer)` → no-op.
///   2. Retargetable `Map(Fresh)` / `Scan(Fresh)` → retarget destination,
///      rewrite sibling `Index` consumers to view loads.
///   3. Fixed-size aggregate (`[Size(n)]T`) → element stores.
///   4. Runtime-sized array, not from a retargetable SOAC → error.
///   5. Scalar / vector / matrix → single store at index 0.
///
/// `multi_source` is `true` when the slot has more than one source
/// (e.g. both arms of an `If` write the same slot). Multi-source slots
/// reject any sibling `Pure` consumer of the source — sibling-Index
/// rewrites for multi-source slots would need Phi-tracking across CFG
/// paths.
#[allow(clippy::too_many_arguments)]
pub fn compute_slot_source(
    graph: &mut EGraph,
    aliases: &mut HashMap<NodeId, NodeId>,
    next_effect: &mut u32,
    block: BlockId,
    source: NodeId,
    slot_index: usize,
    slot_ty: &Type<TypeName>,
    binding: BindingRef,
    multi_source: bool,
) -> Result<(), ConvertError> {
    // 1. Consuming Scan: nothing to emit.
    if result_soac_is_consuming_scan(graph, source) {
        return Ok(());
    }

    // 2. Retargetable Map/Scan(Fresh).
    if result_soac_is_map_or_scan(graph, source) {
        let elem_ty = slot_ty.elem_type().cloned().expect("Map/Scan slot output is always an array");
        let view = graph_ops::intern_storage_view(graph, binding, elem_ty.clone(), None);
        if multi_source {
            reject_sibling_consumers(graph, source, slot_index)?;
        } else {
            rewrite_sibling_index_consumers(
                graph,
                aliases,
                block,
                next_effect,
                source,
                view,
                elem_ty,
                slot_index,
            )?;
        }
        retarget_map_scan(graph, source, view);
        return Ok(());
    }

    // 3. Fixed-size aggregate.
    let fixed_size = slot_ty.array_size().and_then(|s| {
        if let Type::Constructed(TypeName::Size(n), _) = s { Some(*n) } else { None }
    });
    if let (Some(n), Some(et)) = (fixed_size, slot_ty.elem_type().cloned()) {
        let view = graph_ops::intern_storage_view(graph, binding, et.clone(), None);
        for j in 0..n {
            let elem =
                graph.intern_pure(PureOp::Project { index: j as u32 }, smallvec![source], et.clone());
            let idx = graph_ops::intern_u32(graph, j as u32, None);
            graph_ops::emit_storage_store(graph, block, view, idx, elem, et.clone(), next_effect, None);
        }
        return Ok(());
    }

    // 4. Runtime-sized array not produced by a retargetable SOAC: error.
    if is_unsized_array(slot_ty) {
        return Err(ConvertError::Unsupported(format!(
            "compute output #{} is a runtime-sized array ({:?}) not produced by a \
             retargetable map/scan: only map/scan results stream into a runtime-sized \
             output. Wrap the producer in a `map`, or return a fixed-size array",
            slot_index, slot_ty
        )));
    }

    // 5. Scalar / vector / matrix.
    let view = graph_ops::intern_storage_view(graph, binding, slot_ty.clone(), None);
    let idx0 = graph_ops::intern_u32(graph, 0, None);
    graph_ops::emit_storage_store(
        graph,
        block,
        view,
        idx0,
        source,
        slot_ty.clone(),
        next_effect,
        None,
    );
    Ok(())
}

/// Realise one graphics output as a store to the slot's
/// `OutputSlot { index }` place. The whole value (which is a scalar,
/// vector, or matrix in practice) is written in one operation.
pub fn graphics_slot_source(
    graph: &mut EGraph,
    block: BlockId,
    next_effect: &mut u32,
    source: NodeId,
    slot_index: usize,
    slot_ty: &Type<TypeName>,
) {
    let place = graph.intern_pure(
        PureOp::OutputSlot { index: slot_index },
        smallvec![],
        slot_ty.clone(),
    );
    graph_ops::emit_store(graph, block, place, source, next_effect, None);
}

// ----------------------------------------------------------------------------
// Classifier predicates
// ----------------------------------------------------------------------------

/// True iff the side-effect producing `result` is a `PendingSoac::Scan`
/// with `destination: InputBuffer`. The scan writes its prefix into its
/// input buffer; the entry's auto-bound output is unused.
pub(crate) fn result_soac_is_consuming_scan(graph: &EGraph, result: NodeId) -> bool {
    for (_, block) in &graph.skeleton.blocks {
        for se in &block.side_effects {
            if se.result == Some(result) {
                return matches!(
                    &se.kind,
                    SideEffectKind::Pending(PendingSoac::Scan {
                        destination: SoacDestination::InputBuffer,
                        ..
                    })
                );
            }
        }
    }
    false
}

/// True iff the side-effect producing `result` is a Map or Scan(Fresh)
/// SOAC retargetable to an `OutputView`. Scans already targeting
/// `InputBuffer` are skipped here.
pub(crate) fn result_soac_is_map_or_scan(graph: &EGraph, result: NodeId) -> bool {
    for (_, block) in &graph.skeleton.blocks {
        for se in &block.side_effects {
            if se.result == Some(result) {
                return matches!(
                    &se.kind,
                    SideEffectKind::Pending(PendingSoac::Map { .. })
                        | SideEffectKind::Pending(PendingSoac::Scan {
                            destination: SoacDestination::Fresh,
                            ..
                        })
                );
            }
        }
    }
    false
}

/// True iff `ty` is an Array whose size is a free variable or
/// placeholder — runtime-sized rather than a known constant.
pub(crate) fn is_unsized_array(ty: &Type<TypeName>) -> bool {
    ty.array_size()
        .map(|s| {
            matches!(
                s,
                Type::Variable(_) | Type::Constructed(TypeName::SizePlaceholder, _)
            )
        })
        .unwrap_or(false)
}

// ----------------------------------------------------------------------------
// Retarget + sibling-Index rewrite
// ----------------------------------------------------------------------------

/// Retarget the Map/Scan producing `target_result` to write into
/// `output_view` instead of allocating a fresh buffer. Flips its
/// `destination` to `OutputView` and appends the view as its last
/// operand. Callers must pre-screen with `result_soac_is_map_or_scan`.
pub(crate) fn retarget_map_scan(graph: &mut EGraph, target_result: NodeId, output_view: NodeId) {
    for (_, block) in graph.skeleton.blocks.iter_mut() {
        for se in &mut block.side_effects {
            if se.result != Some(target_result) {
                continue;
            }
            let kind = se.kind.clone();
            match kind {
                SideEffectKind::Pending(PendingSoac::Map {
                    func,
                    input_array_types,
                    input_elem_types,
                    output_elem_type,
                    destination: _,
                }) => {
                    se.kind = SideEffectKind::Pending(PendingSoac::Map {
                        func,
                        input_array_types,
                        input_elem_types,
                        output_elem_type,
                        destination: SoacDestination::OutputView,
                    });
                    se.operand_nodes.push(output_view);
                }
                SideEffectKind::Pending(PendingSoac::Scan {
                    func,
                    reduce_func,
                    input_array_type,
                    input_elem_type,
                    destination: _,
                }) => {
                    se.kind = SideEffectKind::Pending(PendingSoac::Scan {
                        func,
                        reduce_func,
                        input_array_type,
                        input_elem_type,
                        destination: SoacDestination::OutputView,
                    });
                    se.operand_nodes.push(output_view);
                }
                other => panic!(
                    "retarget_map_scan: side effect for target_result={:?} \
                     is not Map/Scan: {:?} — caller must screen with \
                     result_soac_is_map_or_scan first",
                    target_result, other
                ),
            }
            return;
        }
    }
    panic!(
        "retarget_map_scan: no side effect produced target_result={:?}",
        target_result
    );
}

/// For each `Index(source, k)` consumer in the graph, synthesise a
/// `ViewIndex + Load` against the slot's output view and alias the
/// Index NodeId to the load's result. The alias is consulted at
/// extraction time so every downstream `demand` transparently
/// redirects through the view.
///
/// Side-effect operand uses of `source` (other than the SOAC's own
/// `result`) and non-Index Pure consumer shapes (`Project`, `Tuple`,
/// `ArrayLit`, `Call`, …) are rejected with a v1 `Unsupported`
/// diagnostic.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rewrite_sibling_index_consumers(
    graph: &mut EGraph,
    aliases: &mut HashMap<NodeId, NodeId>,
    block: BlockId,
    next_effect: &mut u32,
    source: NodeId,
    view: NodeId,
    elem_ty: Type<TypeName>,
    slot_index: usize,
) -> Result<(), ConvertError> {
    // Reject side-effect operand uses (other than the SOAC's own result).
    for (_, blk) in &graph.skeleton.blocks {
        for se in &blk.side_effects {
            if se.result == Some(source) {
                continue;
            }
            if se.operand_nodes.contains(&source) {
                return Err(ConvertError::Unsupported(format!(
                    "compute output #{}: SOAC result is also used as a \
                     side-effect operand; v1 supports only sibling Index reads \
                     of the same SOAC result alongside output retargeting",
                    slot_index
                )));
            }
        }
    }

    let consumers: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter_map(|(nid, node)| match node {
            ENode::Pure { operands, .. } if operands.contains(&source) => Some(nid),
            _ => None,
        })
        .collect();

    for cid in consumers {
        let (op, operands) = match &graph.nodes[cid] {
            ENode::Pure { op, operands } => (op.clone(), operands.clone()),
            _ => unreachable!("filtered to Pure above"),
        };
        match op {
            PureOp::Index if operands.len() == 2 && operands[0] == source => {
                let idx_nid = operands[1];
                let load_result = graph_ops::emit_view_load(
                    graph,
                    block,
                    view,
                    idx_nid,
                    elem_ty.clone(),
                    next_effect,
                    None,
                );
                aliases.insert(cid, load_result);
            }
            other_op => {
                return Err(ConvertError::Unsupported(format!(
                    "compute output #{}: SOAC result flows into an unsupported \
                     consumer shape (PureOp::{:?}); v1 supports only Index{{result, *}} \
                     consumers alongside output retargeting",
                    slot_index, other_op
                )));
            }
        }
    }
    Ok(())
}

/// V1 cap: a multi-source slot can't have any non-self Pure consumer
/// of its sources — rewriting `Index(merged, k)` would need
/// Phi-tracking across CFG paths.
pub(crate) fn reject_sibling_consumers(
    graph: &EGraph,
    source: NodeId,
    slot_index: usize,
) -> Result<(), ConvertError> {
    let has_consumer = graph.nodes.iter().any(|(nid, node)| match node {
        ENode::Pure { operands, .. } if nid != source => operands.contains(&source),
        _ => false,
    });
    if has_consumer {
        return Err(ConvertError::Unsupported(format!(
            "compute output #{}: a slot written from multiple control-flow \
             paths (e.g. both arms of an `If`) is consumed elsewhere as a \
             value; v1 only supports multi-source slots that are pure \
             outputs",
            slot_index
        )));
    }
    Ok(())
}

/// If `source` is produced by a runtime `filter` (a `PendingSoac::Filter` with
/// `scratch_out = Some(_)`), retarget it so its serial compaction writes the
/// entry's output buffer directly and skip the normal DPS store. Returns
/// `false` (no retarget) for a static Bounded filter or any non-filter source —
/// the caller then falls back to `compute_slot_source`.
///
/// The filter's already-reserved scratch binding is repurposed as the paired
/// **length cell** (`u32`, 4 bytes): the loop stores the surviving count there
/// so the host can read how many of the capacity-`n` output elements are valid.
/// The output buffer is sized `LikeInput` on the filter's input (capacity `n`).
#[allow(clippy::too_many_arguments)]
pub fn retarget_filter_output(
    graph: &mut EGraph,
    storage_bindings: &mut [crate::interface::StorageBindingDecl],
    output: &mut crate::ssa::types::EntryOutput,
    source: NodeId,
    pipeline: &mut crate::pipeline_descriptor::PipelineDescriptor,
    entry_name: &str,
) -> Result<bool, ConvertError> {
    use crate::pipeline_descriptor::BufferLen;
    let out_binding = output.storage_binding.expect("compute output has a storage binding");

    // Find the filter side-effect producing `source` and retarget it in place.
    let mut retargeted: Option<(BindingRef, Type<TypeName>, Type<TypeName>)> = None;
    'outer: for (_bid, block) in graph.skeleton.blocks.iter_mut() {
        for se in block.side_effects.iter_mut() {
            if se.result != Some(source) {
                continue;
            }
            if let SideEffectKind::Pending(PendingSoac::Filter {
                scratch_out,
                len_out,
                input_array_type,
                input_elem_type,
                ..
            }) = &mut se.kind
            {
                let Some(scratch) = *scratch_out else {
                    // Static Bounded filter — not a runtime scratch producer.
                    return Ok(false);
                };
                let input_arr_ty = input_array_type.clone();
                let elem_ty = input_elem_type.clone();
                // Compact straight into the output buffer; reuse the scratch
                // binding as the paired length cell.
                *scratch_out = Some(out_binding);
                *len_out = Some(scratch);
                retargeted = Some((scratch, input_arr_ty, elem_ty));
            }
            break 'outer;
        }
    }
    let Some((scratch, input_arr_ty, elem_ty)) = retargeted else {
        return Ok(false);
    };

    // Repurpose the scratch binding's declaration as the u32 length cell.
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    let len_cell_len = BufferLen::Fixed { bytes: 4 };
    for decl in storage_bindings.iter_mut() {
        if decl.binding == scratch {
            decl.elem_ty = u32_ty.clone();
            decl.length = Some(len_cell_len.clone());
            break;
        }
    }

    // Size the output buffer to the input's element count (capacity n). The
    // input region is concrete after `pin_entry_regions`; if it isn't (no
    // host buffer to mirror), leave the host to size it.
    let out_len = crate::types::array_view_region(&input_arr_ty).and_then(|in_binding| {
        let elem_bytes = crate::ssa::layout::type_byte_size(&elem_ty)?;
        Some(BufferLen::LikeInput {
            set: in_binding.set,
            binding: in_binding.binding,
            elem_bytes,
            src_elem_bytes: elem_bytes,
        })
    });
    output.length = out_len.clone();

    // `publish_implicit_bindings` already ran (in `from_tlc`), so patch the
    // already-published descriptor bindings to match: the output buffer's
    // host-facing length, and the repurposed length cell.
    if let Some(out_len) = out_len {
        set_pipeline_binding_length(pipeline, entry_name, out_binding, out_len);
    }
    set_pipeline_binding_length(pipeline, entry_name, scratch, len_cell_len);
    Ok(true)
}

/// Update the host descriptor's `length` for the storage binding `br` in the
/// pipeline owning `entry_name`. A no-op if the binding isn't found.
fn set_pipeline_binding_length(
    pipeline: &mut crate::pipeline_descriptor::PipelineDescriptor,
    entry_name: &str,
    br: BindingRef,
    len: crate::pipeline_descriptor::BufferLen,
) {
    use crate::pipeline_descriptor::{Binding, Pipeline};
    let bindings = pipeline.pipelines.iter_mut().find_map(|p| match p {
        Pipeline::Compute(cp) if cp.entry_point == entry_name => Some(&mut cp.bindings),
        Pipeline::MultiCompute(mc) if mc.stages.iter().any(|s| s.entry_point == entry_name) => {
            Some(&mut mc.bindings)
        }
        _ => None,
    });
    let Some(bindings) = bindings else { return };
    for b in bindings.iter_mut() {
        if let Binding::StorageBuffer {
            set, binding, length, ..
        } = b
        {
            if *set == br.set && *binding == br.binding {
                *length = Some(len);
                return;
            }
        }
    }
}
