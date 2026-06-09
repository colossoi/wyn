//! Whole-graph output realization.
//!
//! After `from_tlc`, every entry has its writes laid out as one of:
//!
//!   * **Compute (slot-collected)**: each declared output has a
//!     `Vec<SlotSource>` populated by `from_tlc::convert_slot_store`
//!     during conversion. Sources are `(block, value)` pairs — one per
//!     CFG path that writes the slot. A simple body produces a single
//!     source per slot; an `If`-fork inside an `OutputSlotStore` produces
//!     two; nested ifs produce more.
//!   * **Graphics**: the entry's body terminates in `Return(Some(value))`
//!     where `value` is a scalar/vector/fixed aggregate (or a `Tuple`
//!     of them across multiple location-decorated outputs).
//!
//! This pass materialises both shapes into side-effect stores. It needs
//! the whole graph to do its job correctly: sibling `Index` consumers
//! of a slot's SOAC source can be produced anywhere in the body and
//! must all be rewritten to load from the slot's view, which requires
//! the consumer-finding walk over the completed graph. That's why this
//! is a separate phase from `from_tlc`, not a part of it.
//!
//! After this pass the invariant pinned by `verify` holds: no
//! runtime-sized Composite array is reachable from any entry output or
//! output-side-effect operand. Downstream passes (parallelize, soac
//! expansion, materialize, codegen) can rely on it.
//!
//! Runs after `from_tlc::run`, before `parallelize`: the SOAC→OutputView
//! rewrite must precede SOAC wrapping/expansion.

use crate::ssa::framework::BlockId;
use crate::ssa::types::ExecutionModel;
#[allow(unused_imports)]
use ExecutionModel as _;

use super::from_tlc::ConvertError;
use super::graph_ops;
use super::program::{EgirEntry, EgirInner};
use super::types::{NodeId, SkeletonTerminator};

pub mod dispatch;
pub mod verify;

/// Realize every entry's outputs into side-effect stores. After this
/// pass, `verify::check` confirms the invariant.
pub fn run(inner: &mut EgirInner) -> Result<(), ConvertError> {
    // `publish_implicit_bindings` already ran in `from_tlc`, so a runtime
    // `filter` output retarget here must patch the host descriptor in
    // `inner.pipeline` directly (split-borrowed from `entry_points`).
    let EgirInner {
        entry_points,
        pipeline,
        ..
    } = inner;
    for entry in entry_points.iter_mut() {
        realize_entry(entry, pipeline)?;
    }
    if cfg!(debug_assertions) {
        verify::check(inner)?;
    }
    Ok(())
}

fn realize_entry(
    entry: &mut EgirEntry,
    pipeline: &mut crate::pipeline_descriptor::PipelineDescriptor,
) -> Result<(), ConvertError> {
    if entry.outputs.is_empty() {
        return Ok(());
    }
    if !entry.slot_sources.is_empty() {
        // DPS path: slot-collected entries (post-`normalize_outputs`).
        realize_compute_slots(entry, pipeline)
    } else {
        // Return-value classifier: graphics entries, plus compute
        // entries synthesised after `normalize_outputs` (gather
        // prepass entries created by `lift_gathers`, phase
        // intermediates from `parallelize`, etc.). These return a
        // value through `Return(Some(_))` and get classified the
        // old way.
        realize_legacy_return(entry)
    }
}

/// Slot-sources path (compute entries post-`normalize_outputs`). Each
/// declared output's `SlotSource`s independently lower to a DPS write
/// into the shared `OutputView`. Multi-source slots (`If`-forks etc.)
/// share one view; runtime CFG picks which source's write fires.
fn realize_compute_slots(
    entry: &mut EgirEntry,
    pipeline: &mut crate::pipeline_descriptor::PipelineDescriptor,
) -> Result<(), ConvertError> {
    let entry_name = entry.name.clone();
    let EgirEntry {
        graph,
        outputs,
        aliases,
        slot_sources,
        storage_bindings,
        ..
    } = entry;
    let mut next_effect = graph_ops::next_effect_token(graph);

    for (slot_index, output) in outputs.iter_mut().enumerate() {
        let binding = output.storage_binding.expect("BUG: compute output without storage binding");
        let sources = slot_sources.get(slot_index).cloned().unwrap_or_default();
        if sources.is_empty() {
            return Err(ConvertError::Unsupported(format!(
                "compute output #{} has no source — `normalize_outputs` \
                 should have emitted at least one `OutputSlotStore` for \
                 every declared output",
                slot_index
            )));
        }

        // A runtime `filter` whose result is this output retargets directly:
        // its serial loop compacts into the output buffer and writes a paired
        // length cell. No DPS store is emitted — the filter *is* the writer.
        if sources.len() == 1
            && dispatch::retarget_filter_output(
                graph,
                storage_bindings,
                output,
                sources[0].value,
                pipeline,
                &entry_name,
            )?
        {
            continue;
        }

        let multi_source = sources.len() > 1;
        for src in &sources {
            dispatch::compute_slot_source(
                graph,
                aliases,
                &mut next_effect,
                src.block,
                src.value,
                slot_index,
                &output.ty,
                binding,
                multi_source,
            )?;
        }
    }
    Ok(())
}

/// Return-value classifier path. The body terminates in
/// `Return(Some(value))`; classify `value` (possibly a `Tuple` for
/// multi-location / multi-output entries) and emit one store per
/// declared output.
///
/// Used by:
///   * Graphics entries (vertex / fragment) — outputs are scalar /
///     vector / matrix written to `OutputSlot { index }` places.
///   * Compute entries synthesised after `normalize_outputs` (gather
///     prepasses from `lift_gathers`, phase intermediates from
///     `parallelize`, etc.) — outputs are storage-buffer-bound; the
///     SOAC at the tail may need retargeting via `compute_slot_source`.
fn realize_legacy_return(entry: &mut EgirEntry) -> Result<(), ConvertError> {
    let is_compute = matches!(entry.execution_model, ExecutionModel::Compute { .. });
    let EgirEntry {
        graph,
        outputs,
        aliases,
        ..
    } = entry;

    let mut return_loc: Option<(BlockId, NodeId)> = None;
    for (bid, block) in &graph.skeleton.blocks {
        if let SkeletonTerminator::Return(Some(r)) = block.term {
            assert!(
                return_loc.is_none(),
                "realize_outputs: entry body has more than one Return(Some(..)) terminator"
            );
            return_loc = Some((bid, r));
        }
    }
    let (return_block, result) = match return_loc {
        Some(x) => x,
        None => return Ok(()),
    };

    let mut next_effect = graph_ops::next_effect_token(graph);

    let sources = output_sources(graph, result, outputs);

    for (slot_index, output) in outputs.iter().enumerate() {
        let source = sources[slot_index];
        if is_compute {
            let binding = output.storage_binding.expect("BUG: compute output without storage binding");
            // Single-source slot for legacy compute. Sibling-Index
            // rewrites are still valid (they're against the single
            // result NodeId).
            dispatch::compute_slot_source(
                graph,
                aliases,
                &mut next_effect,
                return_block,
                source,
                slot_index,
                &output.ty,
                binding,
                /* multi_source */ false,
            )?;
        } else {
            dispatch::graphics_slot_source(
                graph,
                return_block,
                &mut next_effect,
                source,
                slot_index,
                &output.ty,
            );
        }
    }

    graph.skeleton.blocks[return_block].term = SkeletonTerminator::Return(None);
    Ok(())
}

/// Per-output source nodes: the single result, the operands of a literal
/// `Tuple(n)` result, or `Project(result, i)` for an opaque tuple.
fn output_sources(
    graph: &mut super::types::EGraph,
    result: NodeId,
    outputs: &[crate::ssa::types::EntryOutput],
) -> Vec<NodeId> {
    use super::types::{ENode, PureOp};
    use smallvec::smallvec;

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
