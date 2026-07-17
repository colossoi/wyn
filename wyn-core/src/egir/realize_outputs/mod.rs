//! Whole-graph output realization.
//!
//! After `from_tlc`, every entry has its writes laid out as one of:
//!
//!   * **Compute (slot-routed)**: each declared output has explicit
//!     `SlotSource`s derived from the original TLC tail during conversion.
//!     Sources are `(block, value)` pairs, one per CFG path that produces the
//!     slot. A simple body produces one source; branches can produce several.
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
//! This pass establishes concrete output writers. The later residency planner
//! materializes any runtime composite arrays exposed by non-output consumers;
//! its verifier checks the completed representation boundary.
//!
//! Runs after `from_tlc::run`, before segmentation: the SOAC→OutputView
//! rewrite must precede SOAC wrapping/expansion.

use crate::flow::{BlockId, ExecutionModel};
#[allow(unused_imports)]
use ExecutionModel as _;

use super::from_tlc::ConvertError;
use super::program::{
    host_resource_map, Entry, OutputRoute, OutputSlotId, OutputWriter, RawEntry, RawProgram,
    SemanticResourceRef, SlotSource,
};
use super::types::{EGraph, EffectToken, NodeId, Raw, SkeletonTerminator};
use crate::ResourceId;
use std::collections::HashMap;

pub mod dispatch;
pub mod reconcile;
pub mod verify;

/// Realize every entry's outputs into side-effect stores. After this
/// pass, `verify::check` confirms the invariant.
pub fn run(
    inner: &mut RawProgram,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), ConvertError> {
    let by_binding = host_resource_map(&inner.resources);
    let RawProgram { ir, resources } = inner;
    for entry in &mut ir.entry_points {
        realize_entry(entry, &by_binding, resources, effect_ids)?;
    }
    // Output retargeting can rewrite a captured `map` result from a Composite
    // array to a storage view; sync each capturing region's parameter type so
    // the region body lowers consistently.
    reconcile::run(ir)?;
    Ok(())
}

fn realize_entry(
    entry: &mut RawEntry,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
    resources: &mut [super::program::LogicalResource],
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), ConvertError> {
    if entry.outputs.is_empty() {
        return Ok(());
    }
    if matches!(entry.execution_model, ExecutionModel::Compute { .. }) {
        if entry.output_routes.is_empty() {
            synthesize_compute_routes(entry);
        }
        realize_compute_slots(entry, by_binding, resources, effect_ids)?;
        clear_compute_returns(entry);
        Ok(())
    } else {
        realize_graphics_returns(entry, effect_ids)
    }
}

/// Compute entry points publish exclusively through their output routes. Once
/// those writers exist, no value may remain on an entry terminator.
fn clear_compute_returns(entry: &mut RawEntry) {
    for (_, block) in &mut entry.graph.skeleton.blocks {
        if matches!(block.term, SkeletonTerminator::Return(Some(_))) {
            block.term = SkeletonTerminator::Return(None);
        }
    }
}

/// Compute slot-source path. Each
/// declared output's `SlotSource`s independently lower to a DPS write
/// into the shared `OutputView`. Multi-source slots (`If`-forks etc.)
/// share one view; runtime CFG picks which source's write fires.
fn realize_compute_slots(
    entry: &mut RawEntry,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
    resources: &mut [super::program::LogicalResource],
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), ConvertError> {
    let Entry {
        graph,
        outputs,
        aliases,
        output_routes,
        resource_declarations,
        ..
    } = entry;
    // One producer snapshot for the whole slot loop: everything below
    // appends side effects or rewrites them in place, so sites stay valid.
    let effect_index = graph.side_effect_index();

    for (slot_index, output) in outputs.iter_mut().enumerate() {
        let binding = output.storage_binding().expect("BUG: compute output without storage binding");
        let resource = *by_binding.get(&binding).expect("compute output must have a semantic resource");
        let route_indices: Vec<usize> = output_routes
            .iter()
            .enumerate()
            .filter_map(|(index, route)| (route.slot == OutputSlotId(slot_index)).then_some(index))
            .collect();
        let sources: Vec<SlotSource> =
            route_indices.iter().map(|&index| output_routes[index].source).collect();
        if sources.is_empty() {
            return Err(ConvertError::Unsupported(format!(
                "compute output #{} has no source — TLC-to-EGIR conversion \
                 must derive at least one route for every declared output",
                slot_index
            )));
        }

        // A runtime `filter` whose result is this output retargets directly:
        // its serial loop compacts into the output buffer and writes a paired
        // length cell. No DPS store is emitted — the filter *is* the writer.
        if sources.len() == 1
            && dispatch::retarget_filter_output(
                graph,
                resource_declarations,
                resources,
                resource,
                output,
                sources[0].value,
            )?
        {
            output_routes[route_indices[0]].writers = vec![OutputWriter::Value(sources[0].value)];
            continue;
        }

        let multi_source = sources.len() > 1;
        for (&route_index, src) in route_indices.iter().zip(&sources) {
            let mut writers = source_value_writers(graph, &effect_index, src.value);
            writers.extend(dispatch::compute_slot_source(
                graph,
                &effect_index,
                aliases,
                effect_ids,
                src.block,
                src.value,
                slot_index,
                &output.ty,
                resource,
                multi_source,
            )?);
            dedup_output_writers(&mut writers);
            output_routes[route_index].writers = writers;
        }
    }
    Ok(())
}

/// Convert a generated compute return into explicit output routes before
/// materializing its writers.
///
///
///   * Graphics entries (vertex / fragment) — outputs are scalar /
///     vector / matrix written to `OutputSlot { index }` places.
///   * Generated compute entries — outputs are storage-buffer-bound; the
///     SOAC at the tail may need retargeting via `compute_slot_source`.
fn synthesize_compute_routes(entry: &mut RawEntry) {
    let Entry {
        graph,
        outputs,
        output_routes,
        ..
    } = entry;

    let Some((return_block, result)) = unique_value_return(graph) else {
        return;
    };
    let sources = output_sources(graph, result, outputs);
    for (slot, source) in sources.into_iter().enumerate() {
        output_routes.push(OutputRoute {
            source: SlotSource {
                block: return_block,
                value: source,
            },
            slot: OutputSlotId(slot),
            writers: Vec::new(),
        });
    }
}

/// Graphics entries retain return values because their ABI is location-based
/// IO, not storage output routes.
fn realize_graphics_returns(
    entry: &mut RawEntry,
    effect_ids: &mut crate::IdSource<EffectToken>,
) -> Result<(), ConvertError> {
    let Entry {
        graph,
        outputs,
        output_routes,
        ..
    } = entry;
    let Some((return_block, result)) = unique_value_return(graph) else {
        return Ok(());
    };
    let effect_index = graph.side_effect_index();
    for (slot, (output, source)) in outputs.iter().zip(output_sources(graph, result, outputs)).enumerate() {
        let mut writers = source_value_writers(graph, &effect_index, source);
        writers.push(dispatch::graphics_slot_source(
            graph,
            return_block,
            effect_ids,
            source,
            slot,
            &output.ty,
        ));
        dedup_output_writers(&mut writers);
        output_routes.push(OutputRoute {
            source: SlotSource {
                block: return_block,
                value: source,
            },
            slot: OutputSlotId(slot),
            writers,
        });
    }

    graph.skeleton.blocks[return_block].term = SkeletonTerminator::Return(None);
    Ok(())
}

fn unique_value_return(graph: &EGraph<Raw>) -> Option<(BlockId, NodeId)> {
    let mut returns = graph.skeleton.blocks.iter().filter_map(|(block, body)| {
        let SkeletonTerminator::Return(Some(value)) = body.term else {
            return None;
        };
        Some((block, value))
    });
    let result = returns.next();
    assert!(
        returns.next().is_none(),
        "realize_outputs: entry body has more than one Return(Some(..)) terminator"
    );
    result
}

fn source_value_writers(
    graph: &EGraph<Raw>,
    effect_index: &super::types::SideEffectIndex,
    source: NodeId,
) -> Vec<OutputWriter> {
    let mut writers = Vec::new();
    wyn_graph::for_each_reachable(
        [source],
        wyn_graph::WalkOrder::DepthFirst,
        |node, dependencies| {
            if effect_index.site(node).is_none() {
                dependencies.extend(graph.nodes[node].children());
            }
        },
        |node| {
            if effect_index
                .effect(graph, node)
                .is_some_and(|effect| matches!(effect.kind, super::types::SideEffectKind::Soac(_, _)))
            {
                writers.push(OutputWriter::Value(node));
            }
        },
    );
    writers
}

fn dedup_output_writers(writers: &mut Vec<OutputWriter>) {
    let mut unique = Vec::with_capacity(writers.len());
    for writer in writers.drain(..) {
        if !unique.contains(&writer) {
            unique.push(writer);
        }
    }
    *writers = unique;
}

/// Per-output source nodes: the single result, the operands of a literal
/// `Tuple(n)` result, or `Project(result, i)` for an opaque tuple.
fn output_sources(
    graph: &mut EGraph<Raw>,
    result: NodeId,
    outputs: &[super::ir::EntryOutput<SemanticResourceRef, super::types::WynLanguage>],
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
                None,
            )
        })
        .collect()
}
