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
use super::program::{OutputRoute, OutputSlotId, OutputWriter, SemanticEntry, SemanticProgram, SlotSource};
use super::types::{NodeId, SkeletonTerminator};
use crate::ResourceId;
use std::collections::HashMap;

pub mod dispatch;
pub mod reconcile;
pub mod verify;

/// Realize every entry's outputs into side-effect stores. After this
/// pass, `verify::check` confirms the invariant.
pub fn run(inner: &mut SemanticProgram) -> Result<(), ConvertError> {
    let by_binding = inner
        .resources
        .iter()
        .filter_map(|resource| match resource.origin {
            super::program::ResourceOrigin::Host(binding) => Some((binding, resource.id)),
            super::program::ResourceOrigin::Compiler(_) => None,
        })
        .collect::<HashMap<_, _>>();
    let resources = &mut inner.resources;
    for entry in inner.entry_points.iter_mut() {
        realize_entry(entry, &by_binding, resources)?;
    }
    // Output retargeting can rewrite a captured `map` result from a Composite
    // array to a storage view; sync each capturing region's parameter type so
    // the region body lowers consistently.
    reconcile::run(inner)?;
    if cfg!(debug_assertions) {
        verify::check(inner)?;
    }
    Ok(())
}

fn realize_entry(
    entry: &mut SemanticEntry,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
    resources: &mut [super::program::LogicalResource],
) -> Result<(), ConvertError> {
    if entry.outputs.is_empty() {
        return Ok(());
    }
    if !entry.output_routes.is_empty() {
        // DPS path: slot-collected entries (post-`normalize_outputs`).
        realize_compute_slots(entry, by_binding, resources)
    } else {
        // Return-value classifier: graphics entries, plus compute
        // entries synthesised after `normalize_outputs` (gather
        // prepass entries created by `lift_gathers`, phase
        // intermediates from `parallelize`, etc.). These return a
        // value through `Return(Some(_))` and get classified the
        // old way.
        realize_legacy_return(entry, by_binding)
    }
}

/// Slot-sources path (compute entries post-`normalize_outputs`). Each
/// declared output's `SlotSource`s independently lower to a DPS write
/// into the shared `OutputView`. Multi-source slots (`If`-forks etc.)
/// share one view; runtime CFG picks which source's write fires.
fn realize_compute_slots(
    entry: &mut SemanticEntry,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
    resources: &mut [super::program::LogicalResource],
) -> Result<(), ConvertError> {
    let SemanticEntry {
        graph,
        outputs,
        aliases,
        output_routes,
        resource_declarations,
        ..
    } = entry;
    let mut next_effect = graph_ops::next_effect_token(graph);
    // One producer snapshot for the whole slot loop: everything below
    // appends side effects or rewrites them in place, so sites stay valid.
    let effect_index = graph.side_effect_index();

    for (slot_index, output) in outputs.iter_mut().enumerate() {
        let binding = output.storage_binding.expect("BUG: compute output without storage binding");
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
                &mut next_effect,
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
fn realize_legacy_return(
    entry: &mut SemanticEntry,
    by_binding: &HashMap<crate::BindingRef, ResourceId>,
) -> Result<(), ConvertError> {
    let is_compute = matches!(entry.execution_model, ExecutionModel::Compute { .. });
    let SemanticEntry {
        graph,
        outputs,
        aliases,
        output_routes,
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
    let effect_index = graph.side_effect_index();

    let sources = output_sources(graph, result, outputs);

    for (slot_index, output) in outputs.iter().enumerate() {
        let source = sources[slot_index];
        if is_compute {
            let binding = output.storage_binding.expect("BUG: compute output without storage binding");
            let resource = *by_binding.get(&binding).expect("compute output must have a semantic resource");
            // Single-source slot for legacy compute. Sibling-Index
            // rewrites are still valid (they're against the single
            // result NodeId).
            let mut writers = source_value_writers(graph, &effect_index, source);
            writers.extend(dispatch::compute_slot_source(
                graph,
                &effect_index,
                aliases,
                &mut next_effect,
                return_block,
                source,
                slot_index,
                &output.ty,
                resource,
                /* multi_source */ false,
            )?);
            dedup_output_writers(&mut writers);
            output_routes.push(OutputRoute {
                source: SlotSource {
                    block: return_block,
                    value: source,
                },
                slot: OutputSlotId(slot_index),
                writers,
            });
        } else {
            let mut writers = source_value_writers(graph, &effect_index, source);
            writers.push(dispatch::graphics_slot_source(
                graph,
                return_block,
                &mut next_effect,
                source,
                slot_index,
                &output.ty,
            ));
            dedup_output_writers(&mut writers);
            output_routes.push(OutputRoute {
                source: SlotSource {
                    block: return_block,
                    value: source,
                },
                slot: OutputSlotId(slot_index),
                writers,
            });
        }
    }

    graph.skeleton.blocks[return_block].term = SkeletonTerminator::Return(None);
    Ok(())
}

fn source_value_writers(
    graph: &super::types::EGraph,
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
                .is_some_and(|effect| matches!(effect.kind, super::types::SideEffectKind::Soac(_)))
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
