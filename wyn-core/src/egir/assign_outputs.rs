//! Uniform destination-passing for entry-point outputs.
//!
//! After `from_tlc`, a non-unit entry's body terminates in
//! `Return(Some(result))`. This pass turns that returned value into writes
//! to the entry's output destinations and rewrites the terminator to
//! `Return(None)`, leaving the body unit-producing.
//!
//! One mechanism covers every output shape — single result or tuple field,
//! compute or graphics, streaming SOAC or fixed aggregate or scalar:
//!   * compute output produced by a retargetable Map/Scan(Fresh) → stream the
//!     SOAC directly into the bound output storage view (`OutputView`);
//!   * compute output that's a consuming Scan(InputBuffer)        → no store
//!     (the result already lives in the input buffer);
//!   * compute output that's a fixed-size aggregate               → element
//!     stores through `ViewIndex` + `Store`;
//!   * compute output that's a scalar/vector/matrix               → `Store` at
//!     index 0;
//!   * compute output that's a runtime-sized array not produced by a
//!     retargetable SOAC                                          → clean
//!     `ConvertError::Unsupported`;
//!   * graphics output                                            → `OutputSlot`
//!     place + `Store`.
//!
//! Runs after `from_tlc::run`, before `parallelize`: the SOAC→OutputView
//! rewrite must precede SOAC wrapping/expansion.

use polytype::Type;
use smallvec::smallvec;

use crate::ast::TypeName;
use crate::ssa::framework::BlockId;
use crate::ssa::types::{EntryOutput, ExecutionModel};
use crate::types::TypeExt;

use super::from_tlc::ConvertError;
use super::graph_ops;
use super::program::{EgirEntry, EgirInner};
use super::types::{
    EGraph, ENode, NodeId, PendingSoac, PureOp, SideEffectKind, SkeletonTerminator, SoacDestination,
};

/// Where a single output value is written.
enum Dest {
    /// Compute output bound to a storage buffer at `binding`.
    StorageView(crate::BindingRef),
    /// Graphics output written to the `index`-th location/builtin slot.
    OutputSlot {
        index: usize,
    },
}

/// One flattened output: the value-producing node and its destination.
struct Slot {
    /// Position in the entry's `outputs` (and the graphics location index).
    index: usize,
    ty: Type<TypeName>,
    source: NodeId,
    dest: Dest,
}

/// Assign every entry's outputs by destination-passing. See module docs.
pub fn run(inner: &mut EgirInner) -> Result<(), ConvertError> {
    for entry in &mut inner.entry_points {
        assign_entry_outputs(entry)?;
    }
    Ok(())
}

fn assign_entry_outputs(entry: &mut EgirEntry) -> Result<(), ConvertError> {
    if entry.outputs.is_empty() {
        // Unit entry: body already terminates in `Return(None)`.
        return Ok(());
    }

    let EgirEntry {
        graph,
        outputs,
        execution_model,
        aliases,
        ..
    } = entry;

    // Locate the unique `Return(Some(result))` terminator (functional body:
    // one tail value through the merge block).
    let mut return_loc: Option<(BlockId, NodeId)> = None;
    for (bid, block) in &graph.skeleton.blocks {
        if let SkeletonTerminator::Return(Some(r)) = block.term {
            assert!(
                return_loc.is_none(),
                "assign_outputs: entry body has more than one Return(Some(..)) terminator"
            );
            return_loc = Some((bid, r));
        }
    }
    let (return_block, result) = match return_loc {
        Some(x) => x,
        // Already unit-producing — nothing to assign.
        None => return Ok(()),
    };

    let is_compute = matches!(execution_model, ExecutionModel::Compute { .. });
    let slots = flatten_outputs(graph, result, outputs, is_compute);

    let mut next_effect = graph_ops::next_effect_token(graph);
    for slot in slots {
        lower_slot(graph, aliases, return_block, &mut next_effect, result, &slot)?;
    }

    // The body is now unit-producing.
    graph.skeleton.blocks[return_block].term = SkeletonTerminator::Return(None);
    Ok(())
}

/// Flatten the returned `result` into one `Slot` per declared output,
/// deriving each destination from the `EntryOutput` metadata.
fn flatten_outputs(
    graph: &mut EGraph,
    result: NodeId,
    outputs: &[EntryOutput],
    is_compute: bool,
) -> Vec<Slot> {
    let sources = output_sources(graph, result, outputs);
    outputs
        .iter()
        .enumerate()
        .map(|(i, output)| {
            let dest = if is_compute {
                let br = output.storage_binding.expect("BUG: compute output without storage binding");
                Dest::StorageView(br)
            } else {
                Dest::OutputSlot { index: i }
            };
            Slot {
                index: i,
                ty: output.ty.clone(),
                source: sources[i],
                dest,
            }
        })
        .collect()
}

/// Per-output source nodes: the single result, the operands of a literal
/// `Tuple(n)` result, or `Project(result, i)` for an opaque tuple.
fn output_sources(graph: &mut EGraph, result: NodeId, outputs: &[EntryOutput]) -> Vec<NodeId> {
    let n = outputs.len();
    if n == 1 {
        return vec![result];
    }
    if let ENode::Pure {
        op: PureOp::Tuple(k),
        operands,
    } = &graph.nodes[result]
    {
        if *k == n && operands.len() == n {
            return operands.to_vec();
        }
    }
    outputs
        .iter()
        .enumerate()
        .map(|(i, output)| {
            graph.intern_pure(
                PureOp::Project { index: i as u32 },
                smallvec![result],
                output.ty.clone(),
            )
        })
        .collect()
}

// `entry_result` is the entry's returned-value NodeId. It's threaded so the
// multi-consumer scan can exclude that result node from "real" value-flow
// consumers — `output_sources` already decomposes the result tuple /
// projection chain when populating slot sources.
fn lower_slot(
    graph: &mut EGraph,
    aliases: &mut std::collections::HashMap<NodeId, NodeId>,
    block: BlockId,
    next_effect: &mut u32,
    entry_result: NodeId,
    slot: &Slot,
) -> Result<(), ConvertError> {
    let binding = match slot.dest {
        Dest::StorageView(br) => br,
        Dest::OutputSlot { index } => {
            // Graphics: write the whole value to the output slot.
            let place = graph.intern_pure(PureOp::OutputSlot { index }, smallvec![], slot.ty.clone());
            graph_ops::emit_store(graph, block, place, slot.source, next_effect, None);
            return Ok(());
        }
    };

    // Consuming Scan(InputBuffer): the result lives in the input buffer; the
    // parallel-scan reroute / host pipeline reads it there. No store.
    if result_soac_is_consuming_scan(graph, slot.source) {
        return Ok(());
    }

    // Retargetable Map/Scan(Fresh): stream the SOAC into the output view.
    if result_soac_is_map_or_scan(graph, slot.source) {
        let elem_ty = slot.ty.elem_type().cloned().expect("Map/Scan slot output is always an array");
        let view = graph_ops::intern_storage_view(graph, binding, elem_ty.clone(), None);

        // Slot 0's binding becomes the shared buffer (both backends declare
        // output bindings as read-write — SPIR-V skips `NonWritable` on
        // outputs, WGSL emits `read_write` for `var<storage>`). Any other
        // consumer of the SOAC's result reads via the same view.
        rewrite_other_index_consumers_to_loads(
            graph,
            aliases,
            block,
            next_effect,
            entry_result,
            slot,
            view,
            elem_ty,
        )?;
        rewrite_map_scan_to_into(graph, slot.source, view);
        return Ok(());
    }

    // Fixed-size aggregate: element-store each of the N elements.
    let fixed_size = slot.ty.array_size().and_then(|s| {
        if let Type::Constructed(TypeName::Size(n), _) = s { Some(*n) } else { None }
    });
    if let (Some(n), Some(et)) = (fixed_size, slot.ty.elem_type().cloned()) {
        let view = graph_ops::intern_storage_view(graph, binding, et.clone(), None);
        for j in 0..n {
            let elem = graph.intern_pure(
                PureOp::Project { index: j as u32 },
                smallvec![slot.source],
                et.clone(),
            );
            let idx = graph_ops::intern_u32(graph, j as u32, None);
            graph_ops::emit_storage_store(graph, block, view, idx, elem, et.clone(), next_effect, None);
        }
        return Ok(());
    }

    // Runtime-sized array that no retargetable SOAC produced: storing the
    // whole view at index 0 would emit nonsense. Surface the contract
    // violation as a clean diagnostic.
    if is_unsized_array(&slot.ty) {
        return Err(ConvertError::Unsupported(format!(
            "compute output #{} is a runtime-sized array ({:?}) not produced by a \
             retargetable map/scan: only map/scan results stream into a runtime-sized \
             output. Wrap the producer in a `map`, or return a fixed-size array",
            slot.index, slot.ty
        )));
    }

    // Scalar / vector / matrix: store the whole value at index 0.
    let view = graph_ops::intern_storage_view(graph, binding, slot.ty.clone(), None);
    let idx0 = graph_ops::intern_u32(graph, 0, None);
    graph_ops::emit_storage_store(
        graph,
        block,
        view,
        idx0,
        slot.source,
        slot.ty.clone(),
        next_effect,
        None,
    );
    Ok(())
}

// ----------------------------------------------------------------------------
// Multi-consumer SOAC: rewrite sibling Index reads to storage loads
// ----------------------------------------------------------------------------

/// When retargeting a `Map`/`Scan(Fresh)` SOAC to write directly into
/// `slot`'s output view, any *other* reader of the SOAC's result NodeId
/// would still demand the (now-vanished) in-register Composite array.
/// For each such consumer of `slot.source`:
///
/// - `Pure { Index, [slot.source, idx] }` → synthesise a `ViewIndex + Load`
///   against the slot's output view (both backends expose output bindings
///   read-write) and alias the Index NodeId to the load's result. The
///   alias is consulted at extraction time (`elaborate::run` merges
///   `EgirEntry.aliases` into the extraction `best` map), so every
///   downstream `demand` transparently redirects.
/// - Any other Pure-node consumer shape (`Project`, `Tuple`, `ArrayLit`,
///   `Call`, …) → clean `ConvertError::Unsupported`. v1 cap; extend
///   arm-by-arm as needed.
/// - Side-effect operand reference (other than the SOAC's own `result`)
///   → clean `ConvertError::Unsupported`. Same v1 cap.
fn rewrite_other_index_consumers_to_loads(
    graph: &mut EGraph,
    aliases: &mut std::collections::HashMap<NodeId, NodeId>,
    block: BlockId,
    next_effect: &mut u32,
    entry_result: NodeId,
    slot: &Slot,
    view: NodeId,
    elem_ty: Type<TypeName>,
) -> Result<(), ConvertError> {
    let source = slot.source;

    // Reject side-effect operand uses (other than the SOAC's own result).
    for (_bid, blk) in &graph.skeleton.blocks {
        for se in &blk.side_effects {
            if se.result == Some(source) {
                continue;
            }
            if se.operand_nodes.contains(&source) {
                return Err(ConvertError::Unsupported(format!(
                    "compute output #{}: SOAC result is also used as a \
                     side-effect operand; v1 supports only sibling Index reads \
                     of the same SOAC result alongside output retargeting",
                    slot.index
                )));
            }
        }
    }

    // Collect every Pure node whose operands include `source`. Skip the
    // entry's return-value NodeId itself — `output_sources` already
    // decomposes the result tuple/projection chain, so its uses of `source`
    // don't represent real value-flow consumers.
    let consumers: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter_map(|(nid, node)| match node {
            ENode::Pure { operands, .. } if nid != entry_result && operands.contains(&source) => Some(nid),
            _ => None,
        })
        .collect();

    // Classify and rewrite each consumer.
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
                    slot.index, other_op
                )));
            }
        }
    }

    Ok(())
}

// ----------------------------------------------------------------------------
// SOAC retargeting helpers
// ----------------------------------------------------------------------------

/// True iff the side-effect producing `result` is a `PendingSoac::Scan` with
/// `destination: InputBuffer` — the scan writes its prefix-scan back into its
/// input buffer. The auto-bound entry output is unused; the caller (the
/// parallel-scan reroute or the host pipeline) reads the result from the input
/// binding.
fn result_soac_is_consuming_scan(graph: &EGraph, result: NodeId) -> bool {
    for (_bid, block) in &graph.skeleton.blocks {
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

/// True iff the side-effect producing `result` is a Map or Scan(Fresh) SOAC
/// that can be retargeted to an OutputView. Scans already at `InputBuffer` are
/// skipped — those are handled by `result_soac_is_consuming_scan`.
fn result_soac_is_map_or_scan(graph: &EGraph, result: NodeId) -> bool {
    for (_bid, block) in &graph.skeleton.blocks {
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

/// Retarget the Map/Scan producing `target_result` to write directly into
/// `output_view`: flip its `destination` to `OutputView` and append the view
/// as its last operand. Callers must screen with `result_soac_is_map_or_scan`.
fn rewrite_map_scan_to_into(graph: &mut EGraph, target_result: NodeId, output_view: NodeId) {
    for (_bid, block) in graph.skeleton.blocks.iter_mut() {
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
                    "rewrite_map_scan_to_into: side effect for target_result={:?} \
                     is not Map/Scan: {:?} — caller must screen with \
                     result_soac_is_map_or_scan first",
                    target_result, other
                ),
            }
            return;
        }
    }
    panic!(
        "rewrite_map_scan_to_into: no side effect produced target_result={:?}",
        target_result
    );
}

/// True iff `ty` is an array whose size is a free variable or a placeholder
/// (i.e. runtime-sized rather than a known constant).
fn is_unsized_array(ty: &Type<TypeName>) -> bool {
    ty.array_size()
        .map(|s| {
            matches!(
                s,
                Type::Variable(_) | Type::Constructed(TypeName::SizePlaceholder, _)
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
#[path = "assign_outputs_tests.rs"]
mod assign_outputs_tests;
