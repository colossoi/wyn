//! Per-source DPS dispatch — the shape classifier and emission helpers.
//!
//! Two entry points, matching the two cases in `realize_outputs`:
//!
//!   * `compute_slot_source` — classifies a slot source NodeId and
//!     emits the appropriate write into a storage `OutputView`. Handles
//!     Screma `Project` retargeting, fixed-size aggregates (per-element
//!     stores), scalars/vectors (single store at index 0), and consuming
//!     Scan (no-op — already in its input buffer).
//!   * `graphics_slot_source` — emits one store to the slot's
//!     `OutputSlot { index }` place. Graphics outputs are scalar /
//!     vector / matrix in practice; if a runtime-sized array ever
//!     surfaces here it's a contract violation upstream.

use crate::{LookupMap, LookupSet};
use polytype::Type;
use smallvec::smallvec;

use crate::ast::TypeName;
use crate::flow::BlockId;
use crate::types::TypeExt;
use crate::ResourceId;

use super::super::from_tlc::ConvertError;
use super::super::graph_ops;
use super::super::program::OutputWriter;
use super::super::soac::filter;
use super::super::types::{
    EGraph, ENode, EffectToken, NodeId, PureOp, Raw, SideEffectIndex, SideEffectKind, SkeletonTerminator,
    Soac, SoacDestination, SoacEffect, SoacPlacement,
};

/// The set of Pure nodes reachable from an entry's live outputs — the operand
/// of every `Return(Some(_))` terminator and the value operands of every
/// non-SOAC side effect (stores), followed transitively through Pure operands.
/// Mirrors the reachability the post-realization verifier walks
/// (`realize_outputs::verify`). Nodes outside this set are dead: they have no
/// runtime effect, so consuming `source` there must not fail a slot.
fn reachable_from_outputs(graph: &EGraph<Raw>) -> LookupSet<NodeId> {
    let mut roots: Vec<NodeId> = Vec::new();
    for (_, block) in &graph.skeleton.blocks {
        if let SkeletonTerminator::Return(Some(r)) = block.term {
            roots.push(r);
        }
        for se in &block.side_effects {
            // A SOAC's array operands are inputs, not output writes — excluded,
            // matching the verifier. Store operands carry written values.
            match &se.kind {
                SideEffectKind::Soac(SoacEffect(_, _)) => continue,
                _ => roots.extend(se.operand_nodes.iter().copied()),
            }
        }
    }
    wyn_graph::reachable_set(roots, wyn_graph::WalkOrder::DepthFirst, |nid, out| {
        if let ENode::Pure { operands, .. } = &graph.nodes[nid] {
            out.extend(operands.iter().copied());
        }
    })
}

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod dispatch_tests;

/// Realise one compute slot source as a DPS write into the slot's
/// `OutputView { binding }`. Classification:
///
///   1. Consuming Scan accumulator → no-op (the Screma already wrote
///      into the input buffer).
///   2. Retargetable `Project(Screma, k)` → flip the Screma's
///      `map_destinations[k]` / `acc_destinations[k]` to `OutputView`,
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
///
/// `effect_index` is the caller's snapshot of the graph's side-effect
/// producers. Every mutation on this path appends side effects or
/// rewrites them in place, so existing sites stay valid across the
/// caller's whole slot loop.
#[allow(clippy::too_many_arguments)]
pub fn compute_slot_source(
    graph: &mut EGraph<Raw>,
    effect_index: &SideEffectIndex,
    aliases: &mut LookupMap<NodeId, NodeId>,
    effect_ids: &mut crate::IdSource<EffectToken>,
    block: BlockId,
    source: NodeId,
    slot_index: usize,
    slot_ty: &Type<TypeName>,
    resource: ResourceId,
    multi_source: bool,
) -> Result<Vec<OutputWriter>, ConvertError> {
    // 1. Consuming Scan: nothing to emit.
    if result_soac_is_consuming_scan(graph, effect_index, source) {
        let writer = projected_effect_result(graph, effect_index, source)
            .expect("consuming scan projection must name its producing effect");
        return Ok(vec![OutputWriter::Value(writer)]);
    }

    // 2. Retargetable array projection of Screma(Fresh). Field 0 is the
    // mapped output; field 1 is retargetable only for scan accumulators.
    if let (Some(elem_ty), Some((screma_result, field_idx))) = (
        slot_ty.elem_type().cloned(),
        result_soac_is_array_projection(graph, effect_index, source),
    ) {
        let view = graph_ops::intern_resource_view(graph, resource, elem_ty.clone(), None);
        if multi_source {
            reject_sibling_consumers(graph, source, slot_index)?;
        } else {
            rewrite_sibling_index_consumers(
                graph, aliases, block, effect_ids, source, view, elem_ty, slot_index,
            )?;
        }
        retarget_array_projection(graph, effect_index, screma_result, field_idx, view);
        // The Project node operationally produces the view at runtime
        // (the Screma's loop body wrote field 0 through the view).
        // Update its type to match so verify_no_abstract doesn't flag
        // the Composite array type. Also alias for NodeId substitution.
        if let Some(view_ty) = graph.types.get(&view).cloned() {
            graph.retype_node(source, view_ty);
        }
        aliases.insert(source, view);
        return Ok(vec![OutputWriter::Value(screma_result)]);
    }

    // 3. Fixed-size aggregate.
    let fixed_size = slot_ty.array_size().and_then(|s| {
        if let Type::Constructed(TypeName::Size(n), _) = s {
            Some(*n)
        } else {
            None
        }
    });
    if let (Some(n), Some(et)) = (fixed_size, slot_ty.elem_type().cloned()) {
        let view = graph_ops::intern_resource_view(graph, resource, et.clone(), None);
        let mut writers = Vec::with_capacity(n);
        for j in 0..n {
            let elem = graph.intern_pure(
                PureOp::Project { index: j as u32 },
                smallvec![source],
                et.clone(),
                None,
            );
            let idx = graph_ops::intern_u32(graph, j as u32, None);
            let effect =
                graph_ops::emit_storage_store(graph, block, view, idx, elem, et.clone(), effect_ids, None);
            writers.push(OutputWriter::Effect(effect));
        }
        return Ok(writers);
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
    let view = graph_ops::intern_resource_view(graph, resource, slot_ty.clone(), None);
    let idx0 = graph_ops::intern_u32(graph, 0, None);
    let effect = graph_ops::emit_storage_store(
        graph,
        block,
        view,
        idx0,
        source,
        slot_ty.clone(),
        effect_ids,
        None,
    );
    Ok(vec![OutputWriter::Effect(effect)])
}

/// Realise one graphics output as a store to the slot's
/// `OutputSlot { index }` place. The whole value (which is a scalar,
/// vector, or matrix in practice) is written in one operation.
pub fn graphics_slot_source(
    graph: &mut EGraph<Raw>,
    block: BlockId,
    effect_ids: &mut crate::IdSource<EffectToken>,
    source: NodeId,
    slot_index: usize,
    slot_ty: &Type<TypeName>,
) -> OutputWriter {
    let place = graph.intern_pure(
        PureOp::OutputSlot { index: slot_index },
        smallvec![],
        slot_ty.clone(),
        None,
    );
    OutputWriter::Effect(graph_ops::emit_store(
        graph, block, place, source, effect_ids, None,
    ))
}

fn projected_effect_result(
    graph: &EGraph<Raw>,
    effect_index: &SideEffectIndex,
    source: NodeId,
) -> Option<NodeId> {
    let ENode::Pure { operands, .. } = &graph.nodes[source] else {
        return effect_index.effect(graph, source).is_some().then_some(source);
    };
    let [producer] = operands.as_slice() else {
        return None;
    };
    effect_index.effect(graph, *producer).is_some().then_some(*producer)
}

// ----------------------------------------------------------------------------
// Classifier predicates
// ----------------------------------------------------------------------------

/// True iff `result` is `Project(Screma, k)` where the Screma's k-th
/// tuple field is a Scan accumulator with `destination: InputBuffer` —
/// i.e. a consuming scan. The scan writes its prefix into its input
/// buffer; the entry's auto-bound output is unused.
pub(crate) fn result_soac_is_consuming_scan(
    graph: &EGraph<Raw>,
    effect_index: &SideEffectIndex,
    result: NodeId,
) -> bool {
    if let ENode::Pure {
        op: PureOp::Project { index },
        operands,
    } = &graph.nodes[result]
    {
        let field_idx = *index as usize;
        if let [screma_result] = operands.as_slice() {
            if let Some(se) = effect_index.effect(graph, *screma_result) {
                if let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &se.kind {
                    let n_maps = op.lanes().maps.len();
                    if field_idx >= n_maps {
                        let acc_idx = field_idx - n_maps;
                        if op.is_scan(acc_idx)
                            && op.destination(field_idx).is_some_and(SoacDestination::is_input_buffer)
                        {
                            return true;
                        }
                    }
                }
                return false;
            }
        }
    }
    false
}

/// If `source` is a retargetable array projection of a fresh Screma, return
/// the underlying Screma result and field index.
pub(crate) fn result_soac_is_array_projection(
    graph: &EGraph<Raw>,
    effect_index: &SideEffectIndex,
    source: NodeId,
) -> Option<(NodeId, usize)> {
    let ENode::Pure {
        op: PureOp::Project { index },
        operands,
    } = &graph.nodes[source]
    else {
        return None;
    };
    let field_idx = *index as usize;
    let [screma_result] = operands.as_slice() else {
        return None;
    };
    let se = effect_index.effect(graph, *screma_result)?;
    let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &se.kind else {
        return None;
    };
    let operator_field = field_idx.checked_sub(op.lanes().maps.len());
    let supported =
        field_idx < op.lanes().maps.len() || operator_field.is_some_and(|index| op.is_scan(index));
    (supported && op.destination(field_idx).is_some_and(SoacDestination::is_unplaced))
        .then_some((*screma_result, field_idx))
}

/// True iff `ty` is an Array whose size is a free variable or
/// placeholder - runtime-sized rather than a known constant.
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

/// Retarget one array-producing side of the Screma producing `target_result`
/// to write into `output_view`.
pub(crate) fn retarget_array_projection(
    graph: &mut EGraph<Raw>,
    effect_index: &SideEffectIndex,
    target_result: NodeId,
    field_idx: usize,
    output_view: NodeId,
) {
    if let Some(se) = effect_index.effect_mut(graph, target_result) {
        let SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) = &mut se.kind else {
            panic!(
                "retarget_array_projection: side effect for \
                     target_result={:?} is not Screma: {:?}",
                target_result, se.kind
            );
        };

        // Operand layout is `[inputs.., output_views..]`; neutrals and
        // captures are explicit in the Screma body.
        let base_len = op.lanes().inputs.len();
        let mut cursor = base_len;
        let mut views = Vec::with_capacity(op.result_count());
        for field in 0..op.result_count() {
            if op.destination(field).is_some_and(SoacDestination::is_output_view) {
                views.push(Some(
                    *se.operand_nodes.get(cursor).expect("Screma output view operand missing"),
                ));
                cursor += 1;
            } else {
                views.push(None);
            }
        }

        let operator_index = field_idx.checked_sub(op.lanes().maps.len());
        if operator_index.is_some_and(|index| !op.is_scan(index)) {
            panic!("retarget_array_projection: unsupported Screma field {field_idx}");
        }
        assert!(op.place_destination(field_idx, SoacPlacement::OutputView));
        views[field_idx] = Some(output_view);

        se.operand_nodes.truncate(base_len);
        for view in views.into_iter().flatten() {
            se.operand_nodes.push(view);
        }
        return;
    }
    panic!(
        "retarget_array_projection: no side effect produced target_result={:?}",
        target_result
    );
}

/// For each `Index(source, k)` consumer in the graph, synthesize a
/// `ViewIndex + Load` against the slot's output view and alias the
/// Index NodeId to the load's result. The alias is consulted at
/// extraction time so every downstream `demand` transparently
/// redirects through the view.
///
/// Side-effect operand uses of `source` are accepted only when they
/// land in the side-effect's *input-array* region — those get
/// rewritten in place to point at the output view (and their
/// payload `input_array_types[k]` is updated, which is what
/// `emit_read_element` keys off via `is_view_source(arr_ty)`).
/// `source` reaching any other operand position (capture, init
/// accumulator, output view operand, scatter dest, scatter
/// capture) is rejected as v1 `Unsupported` — those positions
/// expect something other than "read this array per element" and
/// the substitution would be a semantic mismatch.
/// Non-Index Pure consumer shapes (`Project`, `Tuple`, `ArrayLit`,
/// `Call`, …) are still rejected with the original v1 `Unsupported`
/// diagnostic.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rewrite_sibling_index_consumers(
    graph: &mut EGraph<Raw>,
    aliases: &mut LookupMap<NodeId, NodeId>,
    block: BlockId,
    effect_ids: &mut crate::IdSource<EffectToken>,
    source: NodeId,
    view: NodeId,
    elem_ty: Type<TypeName>,
    slot_index: usize,
) -> Result<(), ConvertError> {
    // Walk every block's side-effects, classifying each operand
    // reference to `source`. Input-region hits are queued for
    // rewrite; everything else fails the slot. The SOAC's own
    // `result == source` production is skipped — that one's the
    // retargeting case `retarget_array_projection` already handles.
    let mut input_hits: Vec<(BlockId, usize, usize)> = Vec::new();
    for (skel_bid, blk) in &graph.skeleton.blocks {
        for (se_idx, se) in blk.side_effects.iter().enumerate() {
            if se.result == Some(source) {
                continue;
            }
            for (op_idx, &op_nid) in se.operand_nodes.iter().enumerate() {
                if op_nid != source {
                    continue;
                }
                match &se.kind {
                    SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                        // Screma operand layout: [inputs.., output_views..].
                        // Captures and neutrals live in the typed body.
                        // Only the leading `input_array_types.len()`
                        // slots are array inputs read per element.
                        if op_idx < op.lanes().inputs.len() {
                            input_hits.push((skel_bid, se_idx, op_idx));
                            continue;
                        }
                        return Err(ConvertError::Unsupported(format!(
                            "compute output #{}: SOAC result reaches a Screma \
                             side-effect operand position that is not an array \
                             input (op_index={}, inputs.len()={}); v1 supports \
                             only input-array consumers via `source → output_view` \
                             substitution",
                            slot_index,
                            op_idx,
                            op.lanes().inputs.len()
                        )));
                    }
                    SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) => {
                        // Scatter operand layout: [dest_view, inputs..].
                        // Input region is `1..1+input_array_types.len()`.
                        if op_idx >= 1 && op_idx < 1 + op.body.inputs.len() {
                            input_hits.push((skel_bid, se_idx, op_idx));
                            continue;
                        }
                        return Err(ConvertError::Unsupported(format!(
                            "compute output #{}: SOAC result reaches a Scatter \
                             side-effect operand position that is not an array \
                             input (op_index={}, inputs.len()={}); v1 supports \
                             only input-array consumers via `source → output_view` \
                             substitution",
                            slot_index,
                            op_idx,
                            op.body.inputs.len()
                        )));
                    }
                    _ => {
                        return Err(ConvertError::Unsupported(format!(
                            "compute output #{}: SOAC result reaches a non-SOAC \
                             side-effect operand (op_index={}); v1 supports only \
                             input-array consumers of Screma/Scatter side effects",
                            slot_index, op_idx
                        )));
                    }
                }
            }
        }
    }

    // Apply rewrites. The view's array type is the substitution
    // target for every input-region hit; its element type must
    // match the consumer's existing `input_elem_types[k]` (the
    // SOAC produces values of its declared element type, which is
    // the binding's element type, which is the view's element
    // type — so any mismatch is an upstream bug).
    let view_arr_ty = graph.types[&view].clone();
    let view_elem_ty = view_arr_ty.elem_type().expect("output view must be Array").clone();
    for (skel_bid, se_idx, op_idx) in input_hits {
        let blk = &mut graph.skeleton.blocks[skel_bid];
        let se = &mut blk.side_effects[se_idx];
        se.operand_nodes[op_idx] = view;
        match &mut se.kind {
            SideEffectKind::Soac(SoacEffect(_, Soac::Screma(op))) => {
                let k = op_idx;
                assert_eq!(
                    op.lanes().inputs[k].element(),
                    view_elem_ty,
                    "rewrite_sibling_index_consumers: Screma input_elem_types[{}] \
                     {:?} disagrees with output view's elem type {:?}; the SOAC's \
                     produced elements should equal the entry-output binding's \
                     element type",
                    k,
                    op.lanes().inputs[k].element(),
                    view_elem_ty
                );
                op.lanes_mut().inputs[k].array = view_arr_ty.clone();
            }
            SideEffectKind::Soac(SoacEffect(_, Soac::Hist(op))) => {
                let k = op_idx - 1;
                assert_eq!(
                    op.body.inputs[k].element(),
                    view_elem_ty,
                    "rewrite_sibling_index_consumers: Scatter input_elem_types[{}] \
                     {:?} disagrees with output view's elem type {:?}; the SOAC's \
                     produced elements should equal the entry-output binding's \
                     element type",
                    k,
                    op.body.inputs[k].element(),
                    view_elem_ty
                );
                op.body.inputs[k].array = view_arr_ty.clone();
            }
            _ => unreachable!("classifier above only queues Screma or Hist input-region hits"),
        }
    }

    // Only *live* consumers matter. A `source` retargeted to an output view
    // can still be an operand of a dead aggregate node — e.g. returning
    // `(w.points, w.items)` from a helper's `{ points = map, items = map }`
    // record leaves the record `Tuple` orphaned once its projections fold to
    // the maps. Such a node has no runtime effect, so its reference to `source`
    // must not fail the slot.
    let live = reachable_from_outputs(graph);
    let consumers: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter_map(|(nid, node)| match node {
            ENode::Pure { operands, .. } if operands.contains(&source) && live.contains(&nid) => Some(nid),
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
                    effect_ids,
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
    graph: &EGraph<Raw>,
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

/// If `source` is produced by a runtime filter, retarget its `FilterOutput` so
/// serial compaction writes the
/// entry's output buffer directly and skip the normal DPS store. Returns
/// `false` (no retarget) for a static Bounded filter or any non-filter source —
/// the caller then falls back to `compute_slot_source`.
///
/// The filter's already-reserved scratch resource is repurposed as the paired
/// **length cell** (`u32`, 4 bytes): the loop stores the surviving count there
/// so the host can read how many of the capacity-`n` output elements are valid.
/// The output buffer is sized `LikeInput` on the filter's input (capacity `n`).
#[allow(clippy::too_many_arguments)]
pub fn retarget_filter_output(
    graph: &mut EGraph<Raw>,
    declarations: &mut [crate::egir::program::SemanticResourceDecl],
    resources: &mut crate::egir::program::LogicalResourceArena,
    output_resource: ResourceId,
    output: &mut crate::interface::EntryOutput,
    source: NodeId,
) -> Result<bool, ConvertError> {
    use crate::pipeline_descriptor::BufferLen;

    // Find the filter side-effect producing `source` and retarget it in place.
    let mut retargeted: Option<(ResourceId, Type<TypeName>, Type<TypeName>, Type<TypeName>)> = None;
    'outer: for (_bid, block) in graph.skeleton.blocks.iter_mut() {
        for se in block.side_effects.iter_mut() {
            if se.result != Some(source) {
                continue;
            }
            if let SideEffectKind::Soac(SoacEffect(_, Soac::Filter(op))) = &mut se.kind {
                let filter::Output::Runtime { scratch, .. } = &mut op.state.storage else {
                    // Static Bounded filter — not a runtime scratch producer.
                    return Ok(false);
                };
                let scratch = *scratch;
                let (input, output_elem_ty) = match &op.body.input {
                    filter::Input::Plain(input) => (input, input.element()),
                    filter::Input::Mapped {
                        input,
                        output_element_type,
                        ..
                    } => (input, output_element_type.clone()),
                };
                let input_arr_ty = input.array.clone();
                let input_elem_ty = input.element();
                // Compact straight into the output resource; reuse the
                // scratch resource as the paired length cell.
                op.state.storage = filter::Output::Runtime {
                    scratch: crate::egir::program::SemanticResourceRef(output_resource),
                    length: filter::RuntimeLength::Stored(scratch),
                };
                retargeted = Some((scratch.0, input_arr_ty, input_elem_ty, output_elem_ty));
            }
            break 'outer;
        }
    }
    let Some((scratch, input_arr_ty, input_elem_ty, output_elem_ty)) = retargeted else {
        return Ok(false);
    };

    // Repurpose the scratch resource's declaration as the u32 length cell.
    let u32_ty = Type::Constructed(TypeName::UInt(32), vec![]);
    for decl in declarations.iter_mut() {
        if decl.resource.0 == scratch {
            decl.elem_ty = u32_ty.clone();
            decl.size = crate::egir::program::LogicalSize::FixedBytes(4);
            break;
        }
    }
    if let Some(resource) = resources.get_mut(scratch) {
        resource.elem_ty = u32_ty;
        resource.size = crate::egir::program::LogicalSize::FixedBytes(4);
    }

    // Size the output buffer to the input's element count (capacity n). The
    // input region is concrete after `pin_entry_buffers`; if it isn't (no
    // host buffer to mirror), leave the host to size it.
    let out_len = input_arr_ty.array_buffer().and_then(|region| {
        let Type::Constructed(TypeName::Resource(input_resource), _) = region else {
            return None;
        };
        let in_binding = resources
            .get(*input_resource)
            .filter(|resource| resource.id() == *input_resource)
            .and_then(crate::egir::program::LogicalResource::host_binding)?;
        let src_elem_bytes = crate::ssa::layout::storage_elem_stride(&input_elem_ty)?;
        let elem_bytes = crate::ssa::layout::storage_elem_stride(&output_elem_ty)?;
        Some(BufferLen::LikeInput {
            set: in_binding.set,
            binding: in_binding.binding,
            elem_bytes,
            src_elem_bytes,
        })
    });
    *output.storage_length_mut().expect("realized compute output must be storage") = out_len;
    Ok(true)
}
